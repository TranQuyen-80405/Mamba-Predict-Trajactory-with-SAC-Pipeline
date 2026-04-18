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

from data_contracts import RiskBatch, RiskSample, Trajectory  # noqa: E402
from module_pointpillar import (  # noqa: E402
    CameraToLidarExtrinsics,
    DepthCameraIntrinsics,
    DepthPreprocessConfig,
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
        extrinsics_convention: str = "identity",
        traj_horizon: int = 10,
    ) -> None:
        self.root = Path(root)
        self.T_ctx = int(T_ctx)
        self.cfg = cfg
        self.preprocess_cfg = preprocess_cfg or _default_preprocess_cfg()
        self.extrinsics_convention = extrinsics_convention
        self.traj_horizon = int(traj_horizon)
        if self.traj_horizon < 1:
            raise ValueError("traj_horizon must be >= 1")

        # Horizon lookahead (frames). Defaults mirror § 5.1.
        self.h_05s = int(getattr(cfg, "horizon_05s_frames", 10)) if cfg else 10
        self.h_1s = int(getattr(cfg, "horizon_1s_frames", 20)) if cfg else 20
        self.h_2s = int(getattr(cfg, "horizon_2s_frames", 40)) if cfg else 40
        self._max_horizon = max(self.h_05s, self.h_1s, self.h_2s)

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
            # Need lookahead for risk labels and ego_state[t+1:t+1+H] for trajectory targets.
            t_hi = min(T - self._max_horizon, T - self.traj_horizon)
            for t in range(t_lo, max(t_lo, t_hi)):
                entries.append({
                    "path": str(self.root / r["path"]),
                    "scene_id": scene_id,
                    "rollout_id": int(r["rollout_id"]),
                    "t": int(t),
                })
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    # ---------- fetch ----------
    def __getitem__(self, idx: int) -> RiskSample:
        meta = self.entries[idx]
        traj = Trajectory.from_npz(meta["path"])
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
            depth_m = traj.depth[tau].astype(np.float32)
            R_tau = traj.cam_extr_R[tau].astype(np.float32)
            t_tau = traj.cam_extr_t[tau].astype(np.float32)
            # For dataset-time work we want the points in a body-relative
            # frame so the pretrained KITTI weights stay in-distribution.
            # Use the local extrinsics_convention (identity => cam-frame).
            extrinsics = CameraToLidarExtrinsics(
                R=R_tau,
                t=t_tau,
                convention="identity",
            )
            # Unproject -> world via per-frame R|t -> scale_factor ->
            # intensity -> voxel. No torch-grad path here; this is
            # CPU-side dataloader work, and the points are consumed by
            # PointPillarsNeckExtractor.extract_neck* on the training
            # device inside FullPipeline.forward.
            pts = _preprocess_depth_to_pts(
                depth_m, intrinsics, extrinsics, self.preprocess_cfg,
            )
            pts_seq.append(torch.from_numpy(pts))

        action_seq = torch.from_numpy(
            traj.action[t - T_ctx + 1 : t + 1].astype(np.float32)
        )
        ego_vel_seq = torch.from_numpy(
            traj.ego_vel[t - T_ctx + 1 : t + 1].astype(np.float32)
        )
        H = self.traj_horizon
        fut = traj.ego_state[t + 1 : t + 1 + H, :].astype(np.float32)
        traj_future = fut[:, [0, 1, 5]].copy()
        return RiskSample(
            pts_seq=pts_seq,
            action_seq=action_seq,
            ego_vel_seq=ego_vel_seq,
            risk_05s=torch.tensor(float(traj.risk_05s[t]), dtype=torch.float32),
            risk_1s=torch.tensor(float(traj.risk_1s[t]), dtype=torch.float32),
            risk_2s=torch.tensor(float(traj.risk_2s[t]), dtype=torch.float32),
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
    traj_future_xyyaw = torch.stack([s.traj_future_xyyaw for s in samples], dim=0)

    return RiskBatch(
        pts_seq=pts_seq,
        action_seq=action_seq,
        ego_vel_seq=ego_vel_seq,
        risk_05s=risk_05s,
        risk_1s=risk_1s,
        risk_2s=risk_2s,
        traj_future_xyyaw=traj_future_xyyaw,
    )


__all__ = [
    "RiskDataset",
    "collate_riskbatch",
    "scene_stratified_split",
]
