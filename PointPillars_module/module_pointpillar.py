# =====================================================================
# module_pointpillar.py
# ---------------------------------------------------------------------
# PointPillars wrapper: takes a point cloud (x, y, z, intensity),
# runs pillar_layer -> pillar_encoder -> backbone -> neck, and STOPS at
# the neck. Returns the BEV feature tensor for downstream pipelines
# (e.g. PyBullet + RL).
#
# =====================================================================
# ENVIRONMENT - PINNED VERSIONS
# ---------------------------------------------------------------------
# This stripped build only needs 3 runtime deps (no numba / open3d / cv2
# / PyYAML / tqdm). The head, NMS, anchor generator and visualisation
# helpers have been deleted.
#
#   Python:   3.9 - 3.12
#   torch:    >=1.8, built against the SAME CUDA toolkit as nvcc
#             (torch cu121 <-> nvcc 12.1, torch cu118 <-> nvcc 11.8, ...)
#   numpy:    any version compatible with torch
#   ninja:    required once, to build the voxelization CUDA extension
#
# CONFLICT-AVOIDANCE CHECKLIST
# [1] torch <-> CUDA toolkit: the torch wheel CUDA major/minor MUST match
#     the nvcc used to build pointpillars/ops/voxel_op.
# [2] setuptools too new (>=70) has ABI changes; pin setuptools <70 if
#     the CUDAExtension build fails.
#
# QUICK INSTALL (Colab, CUDA 12.1 runtime)
#   pip install --index-url https://download.pytorch.org/whl/cu121 \
#       torch==2.3.1+cu121
#   pip install numpy ninja
#   # Build the voxelization CUDA extension once, from the repo root:
#   pip install -e .
#
# =====================================================================
# REQUIRED COMPANION FILES (must exist at runtime)
# ---------------------------------------------------------------------
# 1) Model code:     pointpillars/model/            (PointPillars only)
# 2) CUDA op:        pointpillars/ops/voxel_op*     (built .pyd / .so)
# 3) Weights (.pt):  pretrained/epoch_160.pth
#    (head.* / anchors_generator.* keys are ignored on load)
#
# =====================================================================
# DEFAULT CONFIG (matches the pretrained weights shipped with the repo)
# ---------------------------------------------------------------------
#   nclasses           = 3
#   voxel_size         = [0.16, 0.16, 4]
#   point_cloud_range  = [0, -39.68, -3, 69.12, 39.68, 1]
#   max_num_points     = 32
#   max_voxels         = (16000, 40000)             # (train, test)
#   backbone out chans = [64, 128, 256]
#   neck out chans     = [128, 128, 128]
#   neck feature map   = (B, 384, 248, 216) with the default range/voxel
#
# =====================================================================
# INPUT
# ---------------------------------------------------------------------
# - Single frame: np.ndarray OR torch.Tensor, shape (N, 4), dtype float32
#   columns in order: x, y, z, intensity
# - Multi-frame batch: list of tensors/ndarrays with shape (N_i, 4)
# - Points must live in a LiDAR-style frame (x forward, y left, z up)
#   so they fall inside point_cloud_range. If they come from a depth
#   camera (e.g. PyBullet), rotate/translate axes before feeding them in.
#
# OUTPUT
# ---------------------------------------------------------------------
# - NeckFeatureOutput.feature: torch.Tensor (B, 384, H, W), float32, CUDA
#   With the default config: H=248, W=216
# =====================================================================

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# NOTE: `PointPillars` is imported LAZILY inside `_build_model` because that
# import triggers the CUDA `voxel_op` extension. Keeping it out of the
# module-level import lets downstream code use the pure-Python / NumPy
# helpers (depth_to_points_camera, camera_to_lidar, add_intensity, ...) and
# the PointPillarsNeckExtractor freeze API on a CPU-only / no-CUDA box, and
# lets the unit tests import this file without requiring a built voxel_op.
if False:  # type-checking only
    from pointpillars.model import PointPillars


# =====================================================================
# SPEC / VARIABLE DEFINITIONS (grouped for clarity and easy extension)
# ---------------------------------------------------------------------

@dataclass
class PointPillarsConfig:
    """
    Hyper-parameters for PointPillars plus runtime settings.
    Matches the pretrained epoch_160.pth checkpoint (KITTI, 3 classes).
    """
    # model hyper-params
    nclasses: int = 3
    voxel_size: List[float] = field(
        default_factory=lambda: [0.16, 0.16, 4.0]
    )
    point_cloud_range: List[float] = field(
        default_factory=lambda: [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
    )
    max_num_points: int = 32
    max_voxels_train: int = 16000
    max_voxels_test: int = 40000

    # runtime
    ckpt_path: str = "pretrained/epoch_160.pth"
    device: str = "cuda"

    @property
    def max_voxels(self) -> Tuple[int, int]:
        return (self.max_voxels_train, self.max_voxels_test)


@dataclass
class PointCloudInput:
    """
    Spec for a single input point cloud frame.
    points: (N, 4) float32 with columns [x, y, z, intensity].
    """
    points: torch.Tensor
    frame_id: Optional[str] = None  # optional metadata


@dataclass
class NeckFeatureOutput:
    """
    Output produced by the neck extractor.
    feature: (B, C, H, W) float32, device is CUDA when config.device='cuda'.
    """
    feature: torch.Tensor
    batch_size: int
    channels: int
    height: int
    width: int
    device: str


# ---------------------------------------------------------------------
# Depth-camera helpers: specs for turning a depth buffer into a (N, 4)
# point cloud in the LiDAR frame that PointPillars expects.
# ---------------------------------------------------------------------

@dataclass
class DepthCameraIntrinsics:
    """
    Pinhole camera intrinsics.
    fx, fy, cx, cy are in pixels; width/height are the image size.
    near / far are only used by pybullet_depth_to_meters() to convert
    PyBullet's normalized depth buffer [0, 1] into metric depth.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    near: float = 0.1
    # Default is 8.0 m to match the indoor Stage A/B spec
    # (docs/strategy_full_pipeline.md §5.1 / §6.6). Override for outdoor data.
    far: float = 8.0


@dataclass
class CameraToLidarExtrinsics:
    """
    Rigid transform applied per point:  p_lidar = R @ p_cam + t

    If R is None, a preset rotation matrix is chosen from 'convention':
      - 'opencv_to_kitti':
            camera frame  = (x-right, y-down, z-forward)   [OpenCV style]
            lidar  frame  = (x-forward, y-left, z-up)      [KITTI style]
      - 'pybullet_to_kitti':
            camera frame  = (x-right, y-up, z-backward)    [OpenGL / PyBullet]
            lidar  frame  = (x-forward, y-left, z-up)
      - 'identity': no rotation.

    t is the position of the camera origin expressed in the LiDAR frame
    (in meters). Use it to raise the 'virtual LiDAR' above the ground,
    e.g. t=[0, 0, 1.6] to mimic KITTI's 1.6 m Velodyne height.
    """
    R: Optional[np.ndarray] = None
    t: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    convention: str = "opencv_to_kitti"

    def matrix(self) -> np.ndarray:
        if self.R is not None:
            return np.asarray(self.R, dtype=np.float32)
        if self.convention == "opencv_to_kitti":
            return np.array(
                [[0.0, 0.0, 1.0],
                 [-1.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0]],
                dtype=np.float32,
            )
        if self.convention == "pybullet_to_kitti":
            return np.array(
                [[0.0, 0.0, -1.0],
                 [-1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0]],
                dtype=np.float32,
            )
        if self.convention == "identity":
            return np.eye(3, dtype=np.float32)
        raise ValueError(
            f"Unknown extrinsics convention: {self.convention}"
        )


@dataclass
class DepthPreprocessConfig:
    """
    Domain-gap mitigation knobs applied AFTER unprojection.

    These preprocessing steps do NOT fix the train/test domain gap
    completely (depth camera vs Velodyne LiDAR); they only make the
    input a bit more palatable to the pretrained network. For best
    downstream quality, fine-tune the neck on depth-camera data.

    - intensity_mode: how to synthesize the 4th channel.
        * 'zero'              : constant 0.0
        * 'constant'          : fill with intensity_value
        * 'normalized_range'  : 1 - clip(||xyz|| / max_range, 0, 1)
                                (near points get high 'intensity')
    - voxel_downsample: edge length (m) of a uniform voxel filter.
        0 disables it. Helps equalize density with KITTI LiDAR.
    - min_range / max_range: metric clip on camera-frame z to drop
        invalid / far readings before unprojection.
    - subsample_ratio: random subsampling (1.0 = keep all).
    - scale_factor: multiplies the final LiDAR-frame xyz by a constant
        BEFORE voxelization. Used for the indoor-scale hack described in
        docs/strategy_full_pipeline.md §4.1 and docs/module_pointpillar.md §7:
        depth-camera points in an indoor scene only fill the first ~5 m of the
        KITTI point_cloud_range (~70 m), so the BEV canvas is mostly empty.
        Setting scale_factor in [5, 10] inflates the scene to fill the canvas
        without retraining. 1.0 disables it (KITTI-scale LiDAR input).
    - low_point_warn_ratio: if, after range filtering in extract_neck*, the
        retained fraction of points drops below this threshold, emit a
        UserWarning. Useful as an OOD / frame-convention / scale-factor
        smoke check. Set to 0.0 to disable.
    """
    intensity_mode: str = "zero"
    intensity_value: float = 0.0
    voxel_downsample: float = 0.0
    min_range: float = 0.3
    # Indoor default. Override for outdoor (Velodyne-range) scenes.
    max_range: float = 8.0
    subsample_ratio: float = 1.0
    random_seed: Optional[int] = None
    scale_factor: float = 1.0
    low_point_warn_ratio: float = 0.05


# =====================================================================
# MAIN MODULE
# ---------------------------------------------------------------------

class PointPillarsNeckExtractor:
    """
    Loads pretrained PointPillars and runs only up to the 'neck' stage.
    The head / NMS / anchor blocks are deleted from the model and from
    the state_dict on load, saving compute when only the BEV feature
    is needed.

    Two forward entry points:
      - extract_neck(...)           : wrapped in @torch.no_grad(). Use for
                                      Stage B inference / streaming at 20 Hz.
      - extract_neck_forward(...)   : NO @torch.no_grad(). Use for Stage A
                                      training where gradient must flow into
                                      pp.neck (A2 unfreeze) or deeper.

    Freeze API (matches docs/strategy_finetune_with_SAC.md §4 regimes):
      - freeze_all()                : S1 behavior; locks every submodule in
                                      .eval() with requires_grad=False.
      - unfreeze_neck()             : A2 / S2 behavior; only pp.neck trains.
      - set_trainable([names])      : fine-grained; names ⊆
                                      {"pillar_layer", "pillar_encoder",
                                       "backbone", "neck"}.

    Note: PillarLayer has no trainable parameters (just voxelization); its
    freeze state is a no-op w.r.t. gradients but we still flip .eval() for
    consistency.
    """

    # Canonical submodule names; matches the _load_weights prefix filter and
    # the freeze regimes S1/S2/S3 in docs/strategy_finetune_with_SAC.md §4.
    _SUBMODULE_NAMES: Tuple[str, ...] = (
        "pillar_layer",
        "pillar_encoder",
        "backbone",
        "neck",
    )

    def __init__(self, config: Optional[PointPillarsConfig] = None):
        self.config: PointPillarsConfig = config or PointPillarsConfig()
        self.device: torch.device = torch.device(self.config.device)
        self.model: "PointPillars" = self._build_model()
        self._load_weights()
        self.model.eval()

    # ---------- init helpers ----------
    def _build_model(self) -> "PointPillars":
        # Lazy import: this is the only path that needs the CUDA voxel_op
        # extension. Raising ImportError here (instead of at module load
        # time) keeps the pure-Python helpers usable on a CPU-only box.
        from pointpillars.model import PointPillars  # noqa: WPS433
        model = PointPillars(
            nclasses=self.config.nclasses,
            voxel_size=self.config.voxel_size,
            point_cloud_range=self.config.point_cloud_range,
            max_num_points=self.config.max_num_points,
            max_voxels=self.config.max_voxels,
        )
        model.to(self.device)
        return model

    def _load_weights(self) -> None:
        ckpt = self.config.ckpt_path
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Checkpoint .pth not found: {ckpt}"
            )
        state = torch.load(ckpt, map_location=self.device)
        # The shipped checkpoint was trained with the full detector; this
        # stripped model only keeps pillar_layer / pillar_encoder / backbone
        # / neck. Drop all keys that belong to removed submodules and load
        # non-strictly so the remaining weights map 1:1.
        kept_prefixes = (
            "pillar_layer.",
            "pillar_encoder.",
            "backbone.",
            "neck.",
        )
        state = {k: v for k, v in state.items() if k.startswith(kept_prefixes)}
        self.model.load_state_dict(state, strict=False)

    # ---------- pre-processing helpers ----------
    @staticmethod
    def as_tensor(pc: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Convert a point cloud into a contiguous float32 torch.Tensor.
        The shape must be (N, 4).
        """
        if isinstance(pc, np.ndarray):
            t = torch.from_numpy(pc.astype(np.float32, copy=False))
        elif isinstance(pc, torch.Tensor):
            t = pc.float()
        else:
            raise TypeError(
                f"points must be np.ndarray or torch.Tensor, "
                f"got {type(pc)}"
            )
        if t.ndim != 2 or t.size(1) != 4:
            raise ValueError(
                f"Point cloud must have shape (N, 4) [x, y, z, intensity], "
                f"got {tuple(t.shape)}"
            )
        return t.contiguous()

    def filter_range(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Keep only points that lie inside point_cloud_range using a half-open
        [x0, x1) × [y0, y1) × [z0, z1) convention (matches KITTI /
        SECOND voxelization semantics). Prevents the degenerate
        num_points=0 case that crashes the CUDA voxel op.
        """
        x0, y0, z0, x1, y1, z1 = self.config.point_cloud_range
        m = (
            (pts[:, 0] >= x0) & (pts[:, 0] < x1) &
            (pts[:, 1] >= y0) & (pts[:, 1] < y1) &
            (pts[:, 2] >= z0) & (pts[:, 2] < z1)
        )
        return pts[m]

    @staticmethod
    def _warn_low_retention(
        kept: int,
        total: int,
        ratio_threshold: float,
        frame_idx: Optional[int] = None,
    ) -> None:
        """
        Emit a UserWarning if the retained fraction after range filtering
        is below `ratio_threshold`. Early-detects OOD frames, wrong
        coordinate convention, or missing scale_factor for indoor scenes.
        """
        if ratio_threshold <= 0.0 or total == 0:
            return
        ratio = kept / total
        if ratio < ratio_threshold:
            prefix = f"frame {frame_idx}: " if frame_idx is not None else ""
            warnings.warn(
                f"{prefix}only {kept}/{total} points ({ratio:.1%}) left "
                f"after point_cloud_range filtering (threshold "
                f"{ratio_threshold:.1%}). Check coordinate convention, "
                f"extrinsics.t, or DepthPreprocessConfig.scale_factor "
                f"(indoor-scale hack, see docs/strategy_full_pipeline.md §4.1).",
                stacklevel=3,
            )

    # ---------- freeze / unfreeze API ----------
    @staticmethod
    def _set_requires_grad(module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad_(flag)

    def freeze_all(self) -> None:
        """
        Freeze every submodule: requires_grad=False and .eval() mode.
        Matches Stage A1 warm-up and Stage B regime S1. Pair with
        install_bn_lock (docs/strategy_finetune_with_SAC.md §4.2) on the
        top-level model to prevent an external .train() from flipping BN
        running stats back on.
        """
        for name in self._SUBMODULE_NAMES:
            sm = getattr(self.model, name)
            self._set_requires_grad(sm, False)
            sm.eval()

    def unfreeze_neck(self) -> None:
        """
        Unfreeze ONLY pp.neck (requires_grad=True, .train() mode). The other
        submodules stay frozen in .eval(). Matches Stage A2 and S2. Call
        freeze_all() first to establish the baseline.
        """
        self._set_requires_grad(self.model.neck, True)
        self.model.neck.train()

    def set_trainable(self, names: Iterable[str]) -> None:
        """
        Fine-grained freeze policy. Submodules in `names` become trainable
        (requires_grad=True + .train()); the rest are frozen
        (requires_grad=False + .eval()). Unknown names raise ValueError.

        Valid names: {"pillar_layer", "pillar_encoder", "backbone", "neck"}.
        """
        names_set = set(names)
        unknown = names_set - set(self._SUBMODULE_NAMES)
        if unknown:
            raise ValueError(
                f"Unknown submodule(s): {sorted(unknown)}. "
                f"Valid options: {list(self._SUBMODULE_NAMES)}."
            )
        for sub in self._SUBMODULE_NAMES:
            sm = getattr(self.model, sub)
            if sub in names_set:
                self._set_requires_grad(sm, True)
                sm.train()
            else:
                self._set_requires_grad(sm, False)
                sm.eval()

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Convenience: all parameters currently with requires_grad=True."""
        return [p for p in self.model.parameters() if p.requires_grad]

    # ---------- depth-camera helpers ----------
    @staticmethod
    def pybullet_depth_to_meters(
        depth_buffer: np.ndarray,
        near: float,
        far: float,
    ) -> np.ndarray:
        """
        Convert a PyBullet depth buffer (values in [0, 1]) into metric
        depth in meters. Formula matches pybullet.getCameraImage docs.
        Input/output shape: (H, W) float32.
        """
        z_buf = depth_buffer.astype(np.float32, copy=False)
        return (far * near) / (far - (far - near) * z_buf)

    @staticmethod
    def depth_to_points_camera(
        depth_meters: np.ndarray,
        intrinsics: DepthCameraIntrinsics,
        min_range: float = 0.3,
        max_range: float = 40.0,
    ) -> np.ndarray:
        """
        Unproject a (H, W) metric depth image into a (N, 3) point cloud
        in the camera frame (x-right, y-down, z-forward for OpenCV-style
        intrinsics). Points with z outside [min_range, max_range] or
        non-finite are dropped.
        """
        H, W = depth_meters.shape
        if (H, W) != (intrinsics.height, intrinsics.width):
            raise ValueError(
                f"Depth shape {(H, W)} does not match intrinsics "
                f"{(intrinsics.height, intrinsics.width)}."
            )
        z = depth_meters.astype(np.float32, copy=False)
        valid = np.isfinite(z) & (z > min_range) & (z < max_range)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32)

        u = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
        v = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)
        z_v = z[valid]
        x_v = (u[valid] - intrinsics.cx) * z_v / intrinsics.fx
        y_v = (v[valid] - intrinsics.cy) * z_v / intrinsics.fy
        return np.stack([x_v, y_v, z_v], axis=1).astype(np.float32)

    @staticmethod
    def camera_to_lidar(
        pts_cam: np.ndarray,
        extrinsics: CameraToLidarExtrinsics,
    ) -> np.ndarray:
        """
        Apply p_lidar = R @ p_cam + t to every point.
        pts_cam: (N, 3) float32 in the camera frame.
        Returns (N, 3) float32 in the LiDAR frame.
        """
        if pts_cam.shape[0] == 0:
            return pts_cam
        R = extrinsics.matrix()
        t = np.asarray(extrinsics.t, dtype=np.float32).reshape(3)
        return (pts_cam.astype(np.float32, copy=False) @ R.T) + t

    @staticmethod
    def add_intensity(
        pts_xyz: np.ndarray,
        cfg: DepthPreprocessConfig,
    ) -> np.ndarray:
        """
        Attach a synthetic intensity column to (N, 3) points, returning
        (N, 4). The depth camera has no real reflectance; this is just
        a stand-in so voxelization still receives a 4-D feature vector.
        """
        n = pts_xyz.shape[0]
        if cfg.intensity_mode == "zero":
            i = np.zeros((n, 1), dtype=np.float32)
        elif cfg.intensity_mode == "constant":
            i = np.full((n, 1), cfg.intensity_value, dtype=np.float32)
        elif cfg.intensity_mode == "normalized_range":
            r = np.linalg.norm(pts_xyz, axis=1)
            r_max = max(cfg.max_range, 1e-6)
            i = (1.0 - np.clip(r / r_max, 0.0, 1.0)).astype(np.float32)
            i = i[:, None]
        else:
            raise ValueError(
                f"Unknown intensity_mode: {cfg.intensity_mode}"
            )
        return np.concatenate(
            [pts_xyz.astype(np.float32, copy=False), i], axis=1
        )

    @staticmethod
    def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
        """
        Simple first-point-per-voxel downsampling. Keeps one point per
        unique integer voxel key computed from xyz. No-op if voxel<=0
        or pts is empty.
        """
        if voxel <= 0.0 or pts.shape[0] == 0:
            return pts
        keys = np.floor(pts[:, :3] / voxel).astype(np.int64)
        _, first_idx = np.unique(keys, axis=0, return_index=True)
        first_idx.sort()
        return pts[first_idx]

    def preprocess_depth_frame(
        self,
        depth_meters: np.ndarray,
        intrinsics: DepthCameraIntrinsics,
        extrinsics: CameraToLidarExtrinsics,
        cfg: Optional[DepthPreprocessConfig] = None,
    ) -> np.ndarray:
        """
        End-to-end depth pre-processing:
            depth (H, W) meters
            -> unproject to camera-frame xyz
            -> rotate / translate into LiDAR frame
            -> optional scale_factor (indoor-scale hack, §4.1 of the
               full_pipeline doc)
            -> add synthetic intensity
            -> optional voxel downsample + random subsample

        Returns (N, 4) float32 ready for extract_neck().
        """
        cfg = cfg or DepthPreprocessConfig()
        pts_cam = self.depth_to_points_camera(
            depth_meters, intrinsics, cfg.min_range, cfg.max_range
        )
        pts_lidar = self.camera_to_lidar(pts_cam, extrinsics)
        # Apply indoor-scale hack BEFORE intensity / voxel-downsample so the
        # downstream voxel grid lands on the scaled coordinates.
        if cfg.scale_factor != 1.0 and pts_lidar.shape[0] > 0:
            pts_lidar = (pts_lidar * np.float32(cfg.scale_factor)).astype(
                np.float32, copy=False
            )
        pts4 = self.add_intensity(pts_lidar, cfg)
        if cfg.voxel_downsample > 0.0:
            pts4 = self.voxel_downsample(pts4, cfg.voxel_downsample)
        if 0.0 < cfg.subsample_ratio < 1.0 and pts4.shape[0] > 0:
            rng = np.random.default_rng(cfg.random_seed)
            n_keep = max(1, int(pts4.shape[0] * cfg.subsample_ratio))
            idx = rng.choice(pts4.shape[0], size=n_keep, replace=False)
            pts4 = pts4[idx]
        return pts4.astype(np.float32, copy=False)

    # ---------- main API ----------
    def _in_range_dummy_point(self) -> torch.Tensor:
        """
        Single placeholder point inside the half-open KITTI range cube.
        Used only to keep the CUDA voxel op alive when a frame is empty;
        the corresponding BEV row is zeroed before return.
        """
        x0, y0, z0, x1, y1, z1 = self.config.point_cloud_range
        eps = 1e-3
        # Half-open [min, max) on each axis — stay comfortably inside.
        return torch.tensor(
            [
                [
                    min(x0 + eps, x1 - eps),
                    min(y0 + eps, y1 - eps),
                    min(z0 + eps, z1 - eps),
                    0.0,
                ]
            ],
            device=self.device,
            dtype=torch.float32,
        )

    def _prepare_batch(
        self,
        batched_pts: List[Union[np.ndarray, torch.Tensor]],
        do_range_filter: bool,
        warn_low_retention_ratio: float,
    ) -> Tuple[List[torch.Tensor], List[bool]]:
        """
        Shared input-normalization path used by both extract_neck variants.
        Moves each point cloud to self.device, optionally range-filters,
        emits a warning on very low retention.

        If a frame has zero points after filtering, a dummy in-range point
        is fed through the voxel pipeline and `_run_neck` zeros that batch
        row in the neck tensor so callers get a silent (B, 384, H, W) tensor.
        """
        if not batched_pts:
            raise ValueError("batched_pts is empty.")

        prepared: List[torch.Tensor] = []
        was_empty_after_filter: List[bool] = []
        for i, pc in enumerate(batched_pts):
            t = self.as_tensor(pc).to(self.device, non_blocking=True)
            if do_range_filter:
                n_before = int(t.size(0))
                t = self.filter_range(t)
                n_after = int(t.size(0))
                self._warn_low_retention(
                    n_after, n_before, warn_low_retention_ratio, frame_idx=i
                )
            empty = t.size(0) == 0
            was_empty_after_filter.append(empty)
            if empty:
                t = self._in_range_dummy_point()
            prepared.append(t)
        return prepared, was_empty_after_filter

    def _run_neck(
        self,
        batched_pts: List[Union[np.ndarray, torch.Tensor]],
        do_range_filter: bool,
        warn_low_retention_ratio: float,
    ) -> NeckFeatureOutput:
        """
        Core forward: pillar_layer -> pillar_encoder -> backbone -> neck.
        Gradient behavior is determined by the caller (extract_neck wraps
        this in torch.no_grad(); extract_neck_forward does not).
        """
        prepared, empty_rows = self._prepare_batch(
            batched_pts, do_range_filter, warn_low_retention_ratio
        )
        m = self.model
        # 1) voxelize (PillarLayer itself is @torch.no_grad()-decorated
        #    and has no trainable params — voxelization is non-diff)
        pillars, coors_batch, npoints_per_pillar = m.pillar_layer(prepared)
        # 2) pillar feature net (PFN)
        pillar_features = m.pillar_encoder(
            pillars, coors_batch, npoints_per_pillar
        )
        # 3) 2D backbone
        xs = m.backbone(pillar_features)
        # 4) neck (FPN-style, concatenates 3 branches)
        neck_feat = m.neck(xs)
        if any(empty_rows):
            for bi, is_empty in enumerate(empty_rows):
                if is_empty:
                    neck_feat[bi].zero_()

        B, C, H, W = neck_feat.shape
        return NeckFeatureOutput(
            feature=neck_feat,
            batch_size=int(B),
            channels=int(C),
            height=int(H),
            width=int(W),
            device=str(neck_feat.device),
        )

    @torch.no_grad()
    def extract_neck(
        self,
        batched_pts: List[Union[np.ndarray, torch.Tensor]],
        do_range_filter: bool = True,
        warn_low_retention_ratio: float = 0.05,
    ) -> NeckFeatureOutput:
        """
        Inference / streaming entry point (Stage B, 20 Hz control loop).
        Wrapped in @torch.no_grad(), so NO gradient graph is built.

        Args:
            batched_pts:     list of point clouds, each shape (N_i, 4).
            do_range_filter: whether to clip points to point_cloud_range.
            warn_low_retention_ratio:
                if >0 and (kept / total) drops below this after filtering,
                emit a UserWarning. Default 0.05 (5%). Set to 0.0 to silence.

        Returns:
            NeckFeatureOutput: feature shape (B, 384, H, W),
            B = len(batched_pts). Lives on self.device.
        """
        return self._run_neck(
            batched_pts, do_range_filter, warn_low_retention_ratio
        )

    def extract_neck_forward(
        self,
        batched_pts: List[Union[np.ndarray, torch.Tensor]],
        do_range_filter: bool = True,
        warn_low_retention_ratio: float = 0.05,
    ) -> NeckFeatureOutput:
        """
        Training entry point (Stage A). Identical to extract_neck BUT
        without @torch.no_grad(), so gradient can flow into any unfrozen
        submodule (e.g. pp.neck in A2; the entire encoder in S3).

        Call pp.freeze_all() / pp.unfreeze_neck() / pp.set_trainable(...)
        BEFORE invoking this to control exactly which subnetwork receives
        gradient. PillarLayer itself is non-differentiable (hard
        voxelization), so the earliest layer that can actually be trained
        is pillar_encoder.
        """
        return self._run_neck(
            batched_pts, do_range_filter, warn_low_retention_ratio
        )


# =====================================================================
# NOTEBOOK USAGE GUIDE (.ipynb)
# ---------------------------------------------------------------------
# The example below uses the indoor robot-dog spec from
# docs/strategy_full_pipeline.md §5.1 / §6.6:
#   depth_hw=(160, 120), fov_h=90°, near=0.1, far=8.0, dt=0.05 (20 Hz).
# For KITTI / outdoor use, scale width/height/fov accordingly and set
# scale_factor=1.0 (no indoor hack needed).
#
# 1) Import:
#    from module_pointpillar import (
#        PointPillarsConfig,
#        PointPillarsNeckExtractor,
#        DepthCameraIntrinsics,
#        CameraToLidarExtrinsics,
#        DepthPreprocessConfig,
#        NeckFeatureOutput,     # optional, only to read shape fields
#    )
#
# 2) Build a config (override ckpt_path / device as needed):
#    cfg = PointPillarsConfig(
#        ckpt_path="pretrained/epoch_160.pth",
#        device="cuda",
#    )
#
# 3) Instantiate the extractor (do this once per session):
#    extractor = PointPillarsNeckExtractor(cfg)
#
# 4) Prepare a (N, 4) float32 point cloud. Two supported sources:
#
#    4a) From a KITTI .bin file (already in LiDAR frame):
#         pts = np.fromfile("xxx.bin", dtype=np.float32).reshape(-1, 4)
#
#    4b) From a PyBullet / OpenCV depth camera (recommended pipeline,
#        indoor robot-dog spec):
#
#         # 160x120, horizontal FoV 90° -> fx = fy = (W/2) / tan(45°) = 80.
#         intr = DepthCameraIntrinsics(
#             fx=80.0, fy=80.0, cx=80.0, cy=60.0,
#             width=160, height=120,
#             near=0.1, far=8.0,
#         )
#         extr = CameraToLidarExtrinsics(
#             convention="pybullet_to_kitti",   # or "opencv_to_kitti"
#             # Mount height for a robot dog (~0.3 to 0.6 m); tune per URDF.
#             t=np.array([0.0, 0.0, 0.4], dtype=np.float32),
#         )
#         pcfg = DepthPreprocessConfig(
#             intensity_mode="normalized_range",
#             voxel_downsample=0.05,     # ~5 cm voxel at indoor scale
#             min_range=0.3, max_range=8.0,
#             # Indoor-scale hack: multiply xyz by ~5-10x so the scene
#             # fills the KITTI-scale point_cloud_range (~70 m) that the
#             # pretrained weights expect. See §4.1 of the full-pipeline
#             # doc for the rationale.
#             scale_factor=6.0,
#             # Early OOD detection: warn if <5% of points survive range
#             # filtering (often = wrong convention or missing scale hack).
#             low_point_warn_ratio=0.05,
#         )
#
#         # PyBullet gives depth in [0, 1]; convert to meters first.
#         depth_m = extractor.pybullet_depth_to_meters(depth_buf,
#                                                     intr.near, intr.far)
#         pts = extractor.preprocess_depth_frame(depth_m, intr, extr, pcfg)
#         # pts shape: (N, 4), already in LiDAR frame, ready to use.
#
# 5) Run neck extraction:
#
#    # Stage B (inference / streaming, 20 Hz) — no gradient graph:
#    out = extractor.extract_neck([pts])              # batch_size = 1
#    print(out.feature.shape)                         # (1, 384, 248, 216)
#
#    # Stage A (training, e.g. A2 with neck unfrozen) — gradient flows:
#    extractor.freeze_all()
#    extractor.unfreeze_neck()                        # A2 regime
#    out = extractor.extract_neck_forward([pts])
#    loss = some_downstream_loss(out.feature)
#    loss.backward()
#
#    # Fine-grained: only pp.neck + pp.backbone trainable:
#    extractor.set_trainable(["neck", "backbone"])
#
# 6) Handling the two known domain issues:
#
#    [A] Coordinate frame (fully fixed in this file)
#        Depth sensors return xyz in a camera frame; KITTI is in a LiDAR
#        frame. preprocess_depth_frame() applies R (rotation preset) + t
#        (translation) so nearly all points land inside
#        point_cloud_range. If you hit "No points left ..." or the
#        low-retention UserWarning, try:
#          - switch convention ("opencv_to_kitti" <-> "pybullet_to_kitti"),
#          - raise extrinsics.t[z] (virtual LiDAR height),
#          - increase DepthPreprocessConfig.scale_factor (indoor-scale),
#          - call extract_neck(..., do_range_filter=False) to inspect xyz.
#
#    [B] Depth-camera vs Velodyne LiDAR (partially mitigated only)
#        The pretrained weights were trained on 360° Velodyne scans.
#        A depth camera has a narrow FoV, denser near pixels, sparser
#        far ones, and no real intensity. Mitigations:
#          - DepthPreprocessConfig(voxel_downsample=0.05-0.15) to match
#            LiDAR-like sparsity,
#          - intensity_mode='normalized_range' for a non-trivial 4th channel,
#          - subsample_ratio < 1.0 to further thin dense regions,
#          - scale_factor in [5, 10] for indoor scenes.
#        These help but do NOT close the gap. Stage A §5.4 (A2) unfreezes
#        pp.neck to adapt features to the depth-camera distribution; use
#        extract_neck_forward + unfreeze_neck() for that.
# =====================================================================
