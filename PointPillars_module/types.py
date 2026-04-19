"""
Centralized dataclass definitions for the PointPillars pipeline.

This is the single source of truth for data/config contracts used across
PointPillars, dataset generation, Stage A, and Stage B.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

_TRAJECTORY_ARRAY_FIELDS: Tuple[str, ...] = (
    "depth",
    "rgb",
    "cam_intrinsics",
    "cam_extr_R",
    "cam_extr_t",
    "ego_state",
    "ego_vel",
    "action",
    "contact_flag",
    "risk_05s",
    "risk_1s",
    "risk_2s",
    "obstacle_aabb",
)


@dataclass
class PointPillarsConfig:
    nclasses: int = 3
    voxel_size: List[float] = field(default_factory=lambda: [0.16, 0.16, 4.0])
    point_cloud_range: List[float] = field(
        default_factory=lambda: [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
    )
    max_num_points: int = 32
    max_voxels_train: int = 16000
    max_voxels_test: int = 40000
    ckpt_path: str = "pretrained/epoch_160.pth"
    device: str = "cuda"

    @property
    def max_voxels(self) -> Tuple[int, int]:
        return (self.max_voxels_train, self.max_voxels_test)


@dataclass
class PointCloudInput:
    points: torch.Tensor
    frame_id: Optional[str] = None


@dataclass
class NeckFeatureOutput:
    feature: torch.Tensor
    batch_size: int
    channels: int
    height: int
    width: int
    device: str


@dataclass
class DepthCameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    near: float = 0.1
    far: float = 8.0


@dataclass
class CameraToLidarExtrinsics:
    R: Optional[np.ndarray] = None
    t: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    convention: str = "opencv_to_kitti"

    def matrix(self) -> np.ndarray:
        if self.R is not None:
            return np.asarray(self.R, dtype=np.float32)
        if self.convention == "opencv_to_kitti":
            return np.array(
                [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                dtype=np.float32,
            )
        if self.convention == "pybullet_to_kitti":
            return np.array(
                [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            )
        if self.convention == "identity":
            return np.eye(3, dtype=np.float32)
        raise ValueError(f"Unknown extrinsics convention: {self.convention}")


@dataclass
class DepthPreprocessConfig:
    intensity_mode: str = "zero"
    intensity_value: float = 0.0
    voxel_downsample: float = 0.0
    min_range: float = 0.3
    max_range: float = 8.0
    subsample_ratio: float = 1.0
    random_seed: Optional[int] = None
    scale_factor: float = 1.0
    low_point_warn_ratio: float = 0.05


@dataclass
class Trajectory:
    scene_id: int
    rollout_id: int
    T: int
    depth: np.ndarray
    rgb: np.ndarray
    cam_intrinsics: np.ndarray
    cam_extr_R: np.ndarray
    cam_extr_t: np.ndarray
    ego_state: np.ndarray
    ego_vel: np.ndarray
    action: np.ndarray
    contact_flag: np.ndarray
    risk_05s: np.ndarray
    risk_1s: np.ndarray
    risk_2s: np.ndarray
    obstacle_aabb: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 6), dtype=np.float32)
    )

    def __post_init__(self) -> None:
        T = int(self.T)
        self.depth = np.asarray(self.depth)
        if self.depth.ndim != 3 or self.depth.shape[0] != T:
            raise ValueError(f"depth must be (T, H, W); got {self.depth.shape} vs T={T}")
        if self.depth.dtype != np.float16:
            raise ValueError(f"depth must be float16; got {self.depth.dtype}")
        if not np.isfinite(self.depth).all() and not np.isfinite(self.depth).any():
            raise ValueError("depth contains no finite values.")
        finite = self.depth[np.isfinite(self.depth)]
        if finite.size > 0:
            d_min, d_max = float(finite.min()), float(finite.max())
            if d_min < 0.0 or d_max > 10.0:
                raise ValueError(
                    f"depth out of sanity range [0, 10] m: min={d_min:.3f}, max={d_max:.3f}"
                )
        self.rgb = np.asarray(self.rgb)
        if self.rgb.size == 0:
            if self.rgb.dtype != np.uint8:
                raise ValueError(f"rgb placeholder must be uint8; got {self.rgb.dtype}")
        else:
            exp = (T, self.depth.shape[1], self.depth.shape[2], 3)
            if self.rgb.shape != exp:
                raise ValueError(f"rgb must have shape {exp} or be empty; got {self.rgb.shape}")
            if self.rgb.dtype != np.uint8:
                raise ValueError(f"rgb must be uint8; got {self.rgb.dtype}")
        for fname, expected_shape in (
            ("cam_intrinsics", (4,)),
            ("cam_extr_R", (T, 3, 3)),
            ("cam_extr_t", (T, 3)),
            ("ego_state", (T, 6)),
            ("ego_vel", (T, 6)),
        ):
            arr = np.asarray(getattr(self, fname))
            if arr.shape != expected_shape:
                raise ValueError(f"{fname} must have shape {expected_shape}; got {arr.shape}")
        self.action = np.asarray(self.action)
        if self.action.ndim != 2 or self.action.shape[0] != T:
            raise ValueError(f"action must be (T, A); got {self.action.shape}")
        self.contact_flag = np.asarray(self.contact_flag).astype(np.bool_)
        if self.contact_flag.shape != (T,):
            raise ValueError(f"contact_flag must be (T,); got {self.contact_flag.shape}")
        for fname in ("risk_05s", "risk_1s", "risk_2s"):
            arr = np.asarray(getattr(self, fname), dtype=np.float32)
            if arr.shape != (T,):
                raise ValueError(f"{fname} must be (T,); got {arr.shape}")
            bad = set(np.unique(arr).tolist()) - {0.0, 1.0}
            if bad:
                raise ValueError(f"{fname} must contain only 0.0/1.0; found {sorted(bad)}")
            setattr(self, fname, arr)
        R = np.asarray(self.cam_extr_R, dtype=np.float32)
        I = np.broadcast_to(np.eye(3, dtype=np.float32), R.shape)
        err = np.abs(np.matmul(R, R.transpose(0, 2, 1)) - I).max()
        if err > 1e-3:
            raise ValueError(f"cam_extr_R is not orthogonal (max |R R^T - I| = {err:.2e})")

    def to_npz(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scene_id": np.asarray(self.scene_id, dtype=np.int64),
            "rollout_id": np.asarray(self.rollout_id, dtype=np.int64),
            "T": np.asarray(self.T, dtype=np.int64),
        }
        for fname in _TRAJECTORY_ARRAY_FIELDS:
            payload[fname] = np.asarray(getattr(self, fname))
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def from_npz(cls, path: Union[str, Path]) -> "Trajectory":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            kwargs = {
                "scene_id": int(data["scene_id"]),
                "rollout_id": int(data["rollout_id"]),
                "T": int(data["T"]),
            }
            for fname in _TRAJECTORY_ARRAY_FIELDS:
                kwargs[fname] = np.array(data[fname])
        return cls(**kwargs)


@dataclass
class RiskSample:
    pts_seq: List[torch.Tensor]
    action_seq: torch.Tensor
    ego_vel_seq: torch.Tensor
    risk_05s: torch.Tensor
    risk_1s: torch.Tensor
    risk_2s: torch.Tensor
    traj_future_xyyaw: torch.Tensor
    risk_label_valid: torch.Tensor = field(
        default_factory=lambda: torch.ones(3, dtype=torch.float32)
    )
    scene_id: int = 0
    rollout_id: int = 0
    frame_t: int = 0


@dataclass
class RiskBatch:
    pts_seq: List[List[torch.Tensor]]
    action_seq: torch.Tensor
    ego_vel_seq: torch.Tensor
    risk_05s: torch.Tensor
    risk_1s: torch.Tensor
    risk_2s: torch.Tensor
    risk_label_valid: torch.Tensor
    traj_future_xyyaw: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.action_seq.shape[0])

    @property
    def t_ctx(self) -> int:
        return int(self.action_seq.shape[1])

    def risk_targets(self) -> torch.Tensor:
        return torch.stack([self.risk_05s, self.risk_1s, self.risk_2s], dim=-1)


@dataclass
class ProprioState:
    base_lin_vel: np.ndarray
    base_ang_vel: np.ndarray
    goal_rel: np.ndarray
    heading_err: float
    last_action: np.ndarray
    joint_q: Optional[np.ndarray] = None
    joint_dq: Optional[np.ndarray] = None

    def to_tensor(
        self,
        include_joint_state: bool = False,
        include_last_action: bool = True,
    ) -> torch.Tensor:
        parts = [
            np.asarray(self.base_lin_vel, dtype=np.float32).reshape(3),
            np.asarray(self.base_ang_vel, dtype=np.float32).reshape(3),
            np.asarray(self.goal_rel, dtype=np.float32).reshape(3),
            np.asarray([self.heading_err], dtype=np.float32),
        ]
        if include_last_action:
            parts.append(np.asarray(self.last_action, dtype=np.float32).reshape(-1))
        if include_joint_state:
            if self.joint_q is None or self.joint_dq is None:
                raise ValueError("include_joint_state=True but joint_q / joint_dq are None.")
            parts.append(np.asarray(self.joint_q, dtype=np.float32).reshape(-1))
            parts.append(np.asarray(self.joint_dq, dtype=np.float32).reshape(-1))
        flat = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        return torch.from_numpy(flat)


@dataclass
class Transition:
    s: np.ndarray
    s_next: np.ndarray
    action: np.ndarray
    r_env: float
    r_risk: float
    done: bool
    episode_id: int = 0
    frame_idx: int = 0


@dataclass
class DataGenConfig:
    n_scenes: int = 300
    rollouts_per_scene: int = 50
    frames_per_rollout: int = 400
    dt: float = 0.05
    depth_hw: Tuple[int, int] = (160, 120)
    camera_fov_h_deg: float = 90.0
    camera_near: float = 0.1
    camera_far: float = 8.0
    policy_random_p: float = 0.15
    policy_scripted_p: float = 0.10
    policy_adversarial_p: float = 0.75
    policy_stationary_p: float = 0.0
    depth_noise_std: float = 0.01
    drop_pixel_prob: float = 0.02
    camera_jitter_deg: float = 1.0
    obstacle_texture_rand: bool = True
    lighting_rand: bool = True
    terminate_on_contact: bool = True
    post_contact_grace_frames: int = 0
    save_rgb: bool = False
    out_dir: str = "dataset/pybullet_risk_v1"
    seed: int = 0
    horizon_05s_frames: int = 10
    horizon_1s_frames: int = 20
    horizon_2s_frames: int = 40

    def __post_init__(self) -> None:
        total = (
            self.policy_random_p
            + self.policy_scripted_p
            + self.policy_adversarial_p
            + self.policy_stationary_p
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "policy_*_p (random+scripted+adversarial+stationary) must sum to 1.0; "
                f"got {total:.6f}"
            )
        if self.post_contact_grace_frames < 0:
            raise ValueError(
                "post_contact_grace_frames must be >= 0; "
                f"got {self.post_contact_grace_frames}"
            )


@dataclass
class EnvConfig:
    dt: float = 0.05
    max_episode_steps: int = 400
    T_ctx: int = 10
    depth_hw: Tuple[int, int] = (160, 120)
    camera_fov_h_deg: float = 90.0
    camera_near: float = 0.1
    camera_far: float = 8.0
    goal_radius: float = 0.3
    w_goal: float = 5.0
    w_progress: float = 1.0
    w_collision: float = 20.0
    w_time: float = 0.01
    w_action_norm: float = 0.001
    lambda_risk: float = 2.0
    action_dim: int = 3
    action_bounds: Tuple[float, float] = (-1.0, 1.0)
    proprio_dim: int = 10
    include_joint_state: bool = False
    include_last_action: bool = True


@dataclass
class RewardNormalizer:
    momentum: float = 0.01
    eps: float = 1e-8
    mean: float = 0.0
    var: float = 1.0

    def update(self, r_batch: torch.Tensor) -> None:
        r = float(r_batch.mean().detach().cpu())
        self.mean = (1.0 - self.momentum) * self.mean + self.momentum * r
        v = float(r_batch.var(unbiased=False).detach().cpu())
        self.var = (1.0 - self.momentum) * self.var + self.momentum * max(v, self.eps)

    def normalize(self, r: torch.Tensor) -> torch.Tensor:
        std = (self.var + self.eps) ** 0.5
        return (r - self.mean) / (std + self.eps)

__all__ = [
    "PointPillarsConfig",
    "PointCloudInput",
    "NeckFeatureOutput",
    "DepthCameraIntrinsics",
    "CameraToLidarExtrinsics",
    "DepthPreprocessConfig",
    "Trajectory",
    "RiskSample",
    "RiskBatch",
    "ProprioState",
    "Transition",
    "DataGenConfig",
    "EnvConfig",
    "RewardNormalizer",
]
