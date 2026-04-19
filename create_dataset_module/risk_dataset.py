"""
RiskDataset: torch.utils.data.Dataset serving RiskSample objects, backed
by a directory of Trajectory.npz files + an index.jsonl catalog.

Given a chosen frame ``t`` inside a trajectory, one sample contains:
  - pts_seq : list of T_ctx (N_i, 4) tensors (LiDAR-frame)
  - action_seq, ego_vel_seq
  - risk_{05s,1s,2s} at frame t
  - traj_future_xyyaw : (H, 3) future planar ego poses (world x, y, yaw) for t+1..t+H

Depth -> point-cloud conversion reuses the pure-numpy helpers from
module_pointpillar (so we do NOT need CUDA or the voxel_op extension to
iterate the dataset).
"""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PP_PKG = os.path.join(_ROOT, "PointPillars_module")
for _p in (_ROOT, _PP_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PointPillars_module.types import (  # noqa: E402
    CameraToLidarExtrinsics,
    DepthCameraIntrinsics,
    DepthPreprocessConfig,
    RiskBatch,
    RiskSample,
    Trajectory,
)
from module_pointpillar import (  # noqa: E402
    PointPillarsNeckExtractor,
)


def _preprocess_depth_to_pts(
    depth_m: np.ndarray,
    intrinsics: DepthCameraIntrinsics,
    extrinsics: CameraToLidarExtrinsics,
    cfg: DepthPreprocessConfig,
) -> np.ndarray:
    """
    Instance-free equivalent of PointPillarsNeckExtractor.preprocess_depth_frame.
    Uses only the static helpers so we never need to build a live
    PointPillars model (no CUDA, no voxel_op) just to iterate the dataset.
    """
    pts_cam = PointPillarsNeckExtractor.depth_to_points_camera(
        depth_m, intrinsics, cfg.min_range, cfg.max_range
    )
    pts_lidar = PointPillarsNeckExtractor.camera_to_lidar(pts_cam, extrinsics)
    if cfg.scale_factor != 1.0 and pts_lidar.shape[0] > 0:
        pts_lidar = (pts_lidar * np.float32(cfg.scale_factor)).astype(
            np.float32, copy=False
        )
    pts4 = PointPillarsNeckExtractor.add_intensity(pts_lidar, cfg)
    if cfg.voxel_downsample > 0.0:
        pts4 = PointPillarsNeckExtractor.voxel_downsample(pts4, cfg.voxel_downsample)
    if 0.0 < cfg.subsample_ratio < 1.0 and pts4.shape[0] > 0:
        rng = np.random.default_rng(cfg.random_seed)
        n_keep = max(1, int(pts4.shape[0] * cfg.subsample_ratio))
        idx = rng.choice(pts4.shape[0], size=n_keep, replace=False)
        pts4 = pts4[idx]
    return pts4.astype(np.float32, copy=False)


# ---------------------------------------------------------------------
# Split helper (§ 5.6 scene-stratified)
# ---------------------------------------------------------------------

def scene_stratified_split(
    scene_ids: Sequence[int],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Return three disjoint scene-id lists (train, val, test) stratified by
    scene so no scene leaks across the boundary. Matches § 5.6.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0; got {ratios}")
    uniq = sorted(set(int(s) for s in scene_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    train = uniq[:n_tr]
    val = uniq[n_tr:n_tr + n_val]
    test = uniq[n_tr + n_val:]
    return train, val, test


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def _default_preprocess_cfg() -> DepthPreprocessConfig:
    return DepthPreprocessConfig(
        intensity_mode="normalized_range",
        voxel_downsample=0.05,
        min_range=0.3,
        max_range=8.0,
        subsample_ratio=1.0,
        scale_factor=6.0,
        low_point_warn_ratio=0.0,  # silence per-sample warnings in bulk
    )


def bev_cache_relpath(traj_relpath: str, frame_idx: int) -> str:
    """
    Stable cache key for one BEV frame.

    Example:
        traj_relpath="s0001_r0002.npz", frame_idx=7
        -> "s0001_r0002/f000007.pt"
    """
    stem = Path(traj_relpath).stem
    return str(Path(stem) / f"f{int(frame_idx):06d}.pt")


class RiskDataset(Dataset):
    """
    One item per (rollout_file, frame_t) pair.

    Args:
        root: directory containing ``index.jsonl`` + ``s*_r*.npz``.
        cfg:  DataGenConfig used to produce the data (for camera spec /
              risk horizons). If None, sane defaults matching § 5.1.
        T_ctx: context length in frames; default 10 (from EnvConfig).
        preprocess_cfg: DepthPreprocessConfig applied when converting
            depth frames to (N, 4) tensors.
        scene_filter: optional list[int] of allowed scene_ids (for splits).
    """

    def __init__(
        self,
        root: str,
        cfg=None,
        T_ctx: int = 10,
        preprocess_cfg: Optional[DepthPreprocessConfig] = None,
        scene_filter: Optional[Sequence[int]] = None,
        extrinsics_convention: str = "pybullet_to_kitti",
        traj_horizon: int = 10,
        bev_cache_root: Optional[str] = None,
        include_action_seq: bool = True,
        include_ego_vel_seq: bool = True,
    ) -> None:
        self.root = Path(root)
        self.T_ctx = int(T_ctx)
        self.cfg = cfg
        self.preprocess_cfg = preprocess_cfg or _default_preprocess_cfg()
        self.extrinsics_convention = extrinsics_convention
        valid_conv = {
            "pybullet_to_kitti",
            "opencv_to_kitti",
            "identity",
            "from_trajectory",
        }
        if self.extrinsics_convention not in valid_conv:
            raise ValueError(
                f"unknown extrinsics_convention={self.extrinsics_convention!r}; "
                f"expected one of {sorted(valid_conv)}"
            )
        self.traj_horizon = int(traj_horizon)
        if self.traj_horizon < 1:
            raise ValueError("traj_horizon must be >= 1")
        self.bev_cache_root = Path(bev_cache_root) if bev_cache_root else None
        self.include_action_seq = bool(include_action_seq)
        self.include_ego_vel_seq = bool(include_ego_vel_seq)
        # Per-worker tiny LRU to avoid reloading the same rollout .npz for
        # adjacent samples that share ``meta["path"]``.
        self._traj_cache: "OrderedDict[str, Trajectory]" = OrderedDict()
        self._traj_cache_max = 8

        # Horizon lookahead (frames). Defaults mirror § 5.1.
        self.h_05s = int(getattr(cfg, "horizon_05s_frames", 10)) if cfg else 10
        self.h_1s = int(getattr(cfg, "horizon_1s_frames", 20)) if cfg else 20
        self.h_2s = int(getattr(cfg, "horizon_2s_frames", 40)) if cfg else 40

        self._scene_filter = (
            set(int(s) for s in scene_filter) if scene_filter is not None else None
        )

        self.entries: List[Dict] = self._build_index()

    # ---------- index ----------
    def _build_index(self) -> List[Dict]:
        index_file = self.root / "index.jsonl"
        if not index_file.exists():
            raise FileNotFoundError(f"index.jsonl not found at {index_file}")

        rows: List[Dict] = []
        with index_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

        entries: List[Dict] = []
        for r in rows:
            scene_id = int(r["scene_id"])
            if self._scene_filter is not None and scene_id not in self._scene_filter:
                continue
            T = int(r["T"])
            t_lo = self.T_ctx - 1
            # Include every frame that has a full trajectory target slice
            # ``ego_state[t+1 : t+1+traj_horizon]`` (exclusive upper ``T - traj_horizon``).
            # Longer risk horizons may be truncated near episode end; per-column
            # ``risk_label_valid`` in ``__getitem__`` masks those out in focal_bce.
            t_hi_excl = T - self.traj_horizon
            for t in range(t_lo, max(t_lo, t_hi_excl)):
                entries.append({
                    "path": str(self.root / r["path"]),
                    "traj_relpath": str(r["path"]),
                    "scene_id": scene_id,
                    "rollout_id": int(r["rollout_id"]),
                    "t": int(t),
                })
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    # ---------- fetch ----------
    def _load_traj_cached(self, path: str) -> Trajectory:
        hit = self._traj_cache.get(path)
        if hit is not None:
            self._traj_cache.move_to_end(path)
            return hit
        traj = Trajectory.from_npz(path)
        self._traj_cache[path] = traj
        if len(self._traj_cache) > self._traj_cache_max:
            self._traj_cache.popitem(last=False)
        return traj

    def __getitem__(self, idx: int) -> RiskSample:
        meta = self.entries[idx]
        traj = self._load_traj_cached(meta["path"])
        t = int(meta["t"])
        T_ctx = self.T_ctx

        fx, fy, cx, cy = traj.cam_intrinsics.tolist()
        H_img, W_img = traj.depth.shape[1], traj.depth.shape[2]

        intrinsics = DepthCameraIntrinsics(
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            width=int(W_img), height=int(H_img),
            near=0.1,
            far=float(max(1.0, self.preprocess_cfg.max_range)),
        )

        pts_seq: List[torch.Tensor] = []
        for tau in range(t - T_ctx + 1, t + 1):
            if self.bev_cache_root is not None:
                rel = bev_cache_relpath(meta["traj_relpath"], tau)
                bev_path = self.bev_cache_root / rel
                if not bev_path.is_file():
                    raise FileNotFoundError(
                        f"BEV cache frame missing: {bev_path} "
                        f"(traj={meta['traj_relpath']} tau={tau})"
                    )
                bev = torch.load(bev_path, map_location="cpu")
                if not torch.is_tensor(bev):
                    raise ValueError(f"invalid cached BEV object at {bev_path}")
                if bev.ndim != 3:
                    raise ValueError(
                        f"cached BEV must be (C,H,W), got {tuple(bev.shape)} at {bev_path}"
                    )
                pts_seq.append(bev.float().contiguous())
                continue

            depth_m = traj.depth[tau].astype(np.float32)
            # Keep depth-camera modality (D435i-like), then map camera axes
            # into a KITTI-style local frame for PointPillars.
            # NOTE:
            #   - default path uses convention presets (no per-frame world pose)
            #   - "from_trajectory" reproduces the previous world-frame behavior
            if self.extrinsics_convention == "from_trajectory":
                extrinsics = CameraToLidarExtrinsics(
                    R=traj.cam_extr_R[tau].astype(np.float32),
                    t=traj.cam_extr_t[tau].astype(np.float32),
                    convention="identity",
                )
            else:
                extrinsics = CameraToLidarExtrinsics(
                    R=None,
                    t=np.zeros(3, dtype=np.float32),
                    convention=self.extrinsics_convention,
                )
            # Unproject -> local KITTI-style frame -> scale_factor ->
            # intensity -> voxel. No torch-grad path here; this is
            # CPU-side dataloader work, and the points are consumed by
            # PointPillarsNeckExtractor.extract_neck* on the training
            # device inside FullPipeline.forward.
            pts = _preprocess_depth_to_pts(
                depth_m, intrinsics, extrinsics, self.preprocess_cfg,
            )
            pts_seq.append(torch.from_numpy(pts))

        if self.include_action_seq:
            action_seq = torch.from_numpy(
                traj.action[t - T_ctx + 1 : t + 1].astype(np.float32)
            )
        else:
            action_seq = torch.empty((T_ctx, 0), dtype=torch.float32)
        if self.include_ego_vel_seq:
            ego_vel_seq = torch.from_numpy(
                traj.ego_vel[t - T_ctx + 1 : t + 1].astype(np.float32)
            )
        else:
            ego_vel_seq = torch.empty((T_ctx, 0), dtype=torch.float32)
        H = self.traj_horizon
        fut = traj.ego_state[t + 1 : t + 1 + H, :].astype(np.float32)
        traj_future = fut[:, [0, 1, 5]].copy()
        T = int(traj.T)
        remain = T - t  # frames from t inclusive to end-1; lookahead [t : t+H) needs remain >= H
        valid = torch.tensor(
            [
                1.0 if remain >= self.h_05s else 0.0,
                1.0 if remain >= self.h_1s else 0.0,
                1.0 if remain >= self.h_2s else 0.0,
            ],
            dtype=torch.float32,
        )
        return RiskSample(
            pts_seq=pts_seq,
            action_seq=action_seq,
            ego_vel_seq=ego_vel_seq,
            risk_05s=torch.tensor(float(traj.risk_05s[t]), dtype=torch.float32),
            risk_1s=torch.tensor(float(traj.risk_1s[t]), dtype=torch.float32),
            risk_2s=torch.tensor(float(traj.risk_2s[t]), dtype=torch.float32),
            risk_label_valid=valid,
            traj_future_xyyaw=torch.from_numpy(traj_future),
            scene_id=int(meta["scene_id"]),
            rollout_id=int(meta["rollout_id"]),
            frame_t=t,
        )

    # ---------- class-balance helper ----------
    def risk_1s_array(self) -> np.ndarray:
        """
        Return a (len(ds),) float32 array of risk_1s per sample, without
        actually loading each .npz in full. For positive oversampling, we
        only need the label, which requires reading the single risk_1s
        array of each trajectory. Cached per-file.
        """
        cache: Dict[str, np.ndarray] = {}
        out = np.zeros(len(self.entries), dtype=np.float32)
        for i, meta in enumerate(self.entries):
            p = meta["path"]
            if p not in cache:
                with np.load(p, allow_pickle=False) as data:
                    cache[p] = np.asarray(data["risk_1s"]).astype(np.float32)
            out[i] = float(cache[p][meta["t"]])
        return out


# ---------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------

def collate_riskbatch(samples: Sequence[RiskSample]) -> RiskBatch:
    """
    Stack a list of RiskSample into a RiskBatch.

    pts_seq is kept as list-of-list: outer axis = time, inner axis = batch.
    This is the exact contract that FullPipeline.forward() consumes.
    """
    if not samples:
        raise ValueError("samples list is empty.")
    T_ctx = len(samples[0].pts_seq)
    if any(len(s.pts_seq) != T_ctx for s in samples):
        raise ValueError(
            "all RiskSample objects must share the same T_ctx "
            f"(got {[len(s.pts_seq) for s in samples]})"
        )

    pts_seq: List[List[torch.Tensor]] = [
        [s.pts_seq[t] for s in samples] for t in range(T_ctx)
    ]
    action_seq = torch.stack([s.action_seq for s in samples], dim=0)
    ego_vel_seq = torch.stack([s.ego_vel_seq for s in samples], dim=0)
    risk_05s = torch.stack([s.risk_05s for s in samples], dim=0)
    risk_1s = torch.stack([s.risk_1s for s in samples], dim=0)
    risk_2s = torch.stack([s.risk_2s for s in samples], dim=0)
    risk_label_valid = torch.stack([s.risk_label_valid for s in samples], dim=0)
    traj_future_xyyaw = torch.stack([s.traj_future_xyyaw for s in samples], dim=0)

    return RiskBatch(
        pts_seq=pts_seq,
        action_seq=action_seq,
        ego_vel_seq=ego_vel_seq,
        risk_05s=risk_05s,
        risk_1s=risk_1s,
        risk_2s=risk_2s,
        risk_label_valid=risk_label_valid,
        traj_future_xyyaw=traj_future_xyyaw,
    )


__all__ = [
    "RiskDataset",
    "collate_riskbatch",
    "scene_stratified_split",
    "bev_cache_relpath",
    "_default_preprocess_cfg",
    "_preprocess_depth_to_pts",
]
