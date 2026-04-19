# =====================================================================
# data_contracts.py
# ---------------------------------------------------------------------
# Single source of truth for all dataclasses exchanged between the
# perception stream (PointPillars -> SpatialReducer -> Mamba -> RiskHead)
# and the Stage A / Stage B training loops.
#
# Every identifier here is spelled exactly as in
# docs/strategy_full_pipeline.md § 3 (data contracts) and § 5.1 / § 6.6
# (configs). Do not rename fields without updating those sections first
# (see § 14 maintenance rule).
# =====================================================================

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


# =====================================================================
# 1. Raw PyBullet rollout artifact  (§ 3.1)
# =====================================================================

_TRAJECTORY_ARRAY_FIELDS: Tuple[str, ...] = (
    "depth", "rgb",
    "cam_intrinsics", "cam_extr_R", "cam_extr_t",
    "ego_state", "ego_vel",
    "action",
    "contact_flag",
    "risk_05s", "risk_1s", "risk_2s",
    "obstacle_aabb",
)

_TRAJECTORY_SCALAR_FIELDS: Tuple[str, ...] = ("scene_id", "rollout_id", "T")


@dataclass
class Trajectory:
    """
    One contiguous rollout produced by the dataset generator. Saved as a
    single .npz file on disk (see ``to_npz`` / ``from_npz``).

    Mirrors docs/strategy_full_pipeline.md § 3.1 verbatim.
    """

    scene_id: int
    rollout_id: int
    T: int

    depth: np.ndarray            # (T, H_img, W_img) float16, meters
    rgb: np.ndarray              # (T, H_img, W_img, 3) uint8 (optional)

    cam_intrinsics: np.ndarray   # (4,) float32 [fx, fy, cx, cy]
    cam_extr_R: np.ndarray       # (T, 3, 3) float32, camera->world
    cam_extr_t: np.ndarray       # (T, 3)    float32, camera origin in world

    ego_state: np.ndarray        # (T, 6) float32 [x, y, z, roll, pitch, yaw]
    ego_vel: np.ndarray          # (T, 6) float32 [vx, vy, vz, wx, wy, wz]

    action: np.ndarray           # (T, A) float32

    contact_flag: np.ndarray     # (T,) bool
    risk_05s: np.ndarray         # (T,) float32 in {0.0, 1.0}
    risk_1s: np.ndarray          # (T,) float32 in {0.0, 1.0}
    risk_2s: np.ndarray          # (T,) float32 in {0.0, 1.0}

    obstacle_aabb: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 6), dtype=np.float32)
    )

    # ---------- validation ----------
    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        T = int(self.T)

        self.depth = np.asarray(self.depth)
        if self.depth.ndim != 3 or self.depth.shape[0] != T:
            raise ValueError(
                f"depth must be (T, H, W); got {self.depth.shape} vs T={T}"
            )
        if self.depth.dtype != np.float16:
            raise ValueError(
                f"depth must be float16; got {self.depth.dtype}"
            )
        if np.isfinite(self.depth).all() is False:  # pragma: no cover
            # Pybullet sometimes returns +inf on invalid pixels; we allow NaN
            # only if explicitly requested. Stricter sanity: reject if ALL
            # values are non-finite (a clear bug).
            if not np.isfinite(self.depth).any():
                raise ValueError("depth contains no finite values.")
        # Range check on finite portion only (tolerates a few inf sentinels).
        finite = self.depth[np.isfinite(self.depth)]
        if finite.size > 0:
            d_min = float(finite.min())
            d_max = float(finite.max())
            if d_min < 0.0 or d_max > 10.0:
                raise ValueError(
                    f"depth out of sanity range [0, 10] m: "
                    f"min={d_min:.3f}, max={d_max:.3f}"
                )

        for fname, expected_shape in [
            ("cam_intrinsics", (4,)),
            ("cam_extr_R", (T, 3, 3)),
            ("cam_extr_t", (T, 3)),
            ("ego_state", (T, 6)),
            ("ego_vel", (T, 6)),
        ]:
            arr = np.asarray(getattr(self, fname))
            if arr.shape != expected_shape:
                raise ValueError(
                    f"{fname} must have shape {expected_shape}; "
                    f"got {arr.shape}"
                )

        self.action = np.asarray(self.action)
        if self.action.ndim != 2 or self.action.shape[0] != T:
            raise ValueError(
                f"action must be (T, A); got {self.action.shape}"
            )

        # Binary fields
        self.contact_flag = np.asarray(self.contact_flag).astype(np.bool_)
        if self.contact_flag.shape != (T,):
            raise ValueError(
                f"contact_flag must be (T,); got {self.contact_flag.shape}"
            )
        for fname in ("risk_05s", "risk_1s", "risk_2s"):
            arr = np.asarray(getattr(self, fname), dtype=np.float32)
            if arr.shape != (T,):
                raise ValueError(
                    f"{fname} must be (T,); got {arr.shape}"
                )
            uniq = np.unique(arr)
            bad = set(uniq.tolist()) - {0.0, 1.0}
            if bad:
                raise ValueError(
                    f"{fname} must contain only 0.0/1.0; found {sorted(bad)}"
                )
            setattr(self, fname, arr)

        # cam_extr_R orthogonality (R R^T ~ I)
        R = np.asarray(self.cam_extr_R, dtype=np.float32)
        I = np.broadcast_to(np.eye(3, dtype=np.float32), R.shape)
        err = np.abs(np.matmul(R, R.transpose(0, 2, 1)) - I).max()
        if err > 1e-3:
            raise ValueError(
                f"cam_extr_R is not orthogonal (max |R R^T - I| = {err:.2e})"
            )

    # ---------- npz (de)serialization ----------
    def to_npz(self, path: Union[str, Path]) -> Path:
        """
        Save this Trajectory as a compressed .npz. Scalars (scene_id,
        rollout_id, T) are stored as 0-d int64 arrays; the other fields
        keep their dtypes exactly.
        """
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
                # Preserve dtype; .copy() so the file handle can close safely.
                kwargs[fname] = np.array(data[fname])
        return cls(**kwargs)


# =====================================================================
# 2. In-memory training sample  (§ 3.2)
# =====================================================================

@dataclass
class RiskSample:
    """
    A single (input, target) pair fed into the Stage A network.
    One sample = one temporal window of length T_ctx ending at frame t.

    Mirrors docs/strategy_full_pipeline.md § 3.2.
    """

    pts_seq: List[torch.Tensor]        # length T_ctx, each (N_i, 4) float32
    action_seq: torch.Tensor           # (T_ctx, A) float32
    ego_vel_seq: torch.Tensor          # (T_ctx, 6) float32

    risk_05s: torch.Tensor             # () float32 in {0.0, 1.0}
    risk_1s: torch.Tensor              # () float32
    risk_2s: torch.Tensor              # () float32
    # Per-horizon mask (1 = include in focal loss). Training excludes truncated ends;
    # defaults to all ones. Shape (3,) for [0.5s, 1s, 2s].
    risk_label_valid: torch.Tensor = field(
        default_factory=lambda: torch.ones(3, dtype=torch.float32)
    )

    # Planar future ego poses (world frame): rows are frames t+1..t+H; cols (x, y, yaw).
    traj_future_xyyaw: torch.Tensor    # (H, 3) float32

    scene_id: int = 0
    rollout_id: int = 0
    frame_t: int = 0


# =====================================================================
# 3. Batched training tensors  (§ 3.3)
# =====================================================================

@dataclass
class RiskBatch:
    """
    Output of the DataLoader collate_fn. This is what the training step
    consumes.

    pts_seq is a list-of-list: outer axis = time (length T_ctx), inner
    axis = batch. ``PointPillarsNeckExtractor.extract_neck_forward``
    accepts ``List[Tensor]`` of length B per time step.
    """

    pts_seq: List[List[torch.Tensor]]  # len T_ctx; inner: B tensors (N, 4)
    action_seq: torch.Tensor           # (B, T_ctx, A)
    ego_vel_seq: torch.Tensor          # (B, T_ctx, 6)

    risk_05s: torch.Tensor             # (B,) float32
    risk_1s: torch.Tensor              # (B,) float32
    risk_2s: torch.Tensor              # (B,) float32
    risk_label_valid: torch.Tensor     # (B, 3) float32 in {0, 1}

    traj_future_xyyaw: torch.Tensor    # (B, H, 3) float32

    @property
    def batch_size(self) -> int:
        return int(self.action_seq.shape[0])

    @property
    def t_ctx(self) -> int:
        return int(self.action_seq.shape[1])

    def risk_targets(self) -> torch.Tensor:
        """Stack the three horizons into (B, 3) for focal-BCE."""
        return torch.stack([self.risk_05s, self.risk_1s, self.risk_2s], dim=-1)


# =====================================================================
# 4. Proprioceptive state  (§ 2.5 / SAC-doc § 10.2)
# =====================================================================

@dataclass
class ProprioState:
    """
    Proprioceptive + goal vector that the SAC Actor / Critic consume
    directly (no perception feature). Matches the layout in
    docs/strategy_finetune_with_SAC.md § 10.2.
    """

    base_lin_vel: np.ndarray           # (3,)
    base_ang_vel: np.ndarray           # (3,)
    goal_rel: np.ndarray               # (3,)
    heading_err: float                 # (1,)
    last_action: np.ndarray            # (A,)
    joint_q: Optional[np.ndarray] = None        # (dof,)
    joint_dq: Optional[np.ndarray] = None       # (dof,)

    def to_tensor(
        self,
        include_joint_state: bool = False,
        include_last_action: bool = True,
    ) -> torch.Tensor:
        """
        Flatten into a 1-D float32 tensor whose layout exactly matches
        the order documented in SAC-doc § 10.2.
        """
        parts = [
            np.asarray(self.base_lin_vel, dtype=np.float32).reshape(3),
            np.asarray(self.base_ang_vel, dtype=np.float32).reshape(3),
            np.asarray(self.goal_rel, dtype=np.float32).reshape(3),
            np.asarray([self.heading_err], dtype=np.float32),
        ]
        if include_last_action:
            parts.append(
                np.asarray(self.last_action, dtype=np.float32).reshape(-1)
            )
        if include_joint_state:
            if self.joint_q is None or self.joint_dq is None:
                raise ValueError(
                    "include_joint_state=True but joint_q / joint_dq are None."
                )
            parts.append(np.asarray(self.joint_q, dtype=np.float32).reshape(-1))
            parts.append(np.asarray(self.joint_dq, dtype=np.float32).reshape(-1))
        flat = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        return torch.from_numpy(flat)


# =====================================================================
# 5. Stage B replay transition  (§ 6.4)
# =====================================================================

@dataclass
class Transition:
    """
    One replay-buffer record. Proprio-state only (no perception tensor)
    per docs/strategy_full_pipeline.md § 6.4.
    """

    s: np.ndarray                      # (d_s,) float32
    s_next: np.ndarray                 # (d_s,) float32
    action: np.ndarray                 # (A,)   float32
    r_env: float                       # np.float32
    r_risk: float                      # np.float32 (0 in BASELINE)
    done: bool
    episode_id: int = 0
    frame_idx: int = 0


# =====================================================================
# 6. Configs  (§ 5.1 / § 6.6)
# =====================================================================

@dataclass
class DataGenConfig:
    """
    Offline PyBullet rollout generator config. Defaults mirror
    docs/strategy_full_pipeline.md § 5.1 verbatim.
    """

    n_scenes: int = 300
    rollouts_per_scene: int = 50
    frames_per_rollout: int = 400
    dt: float = 0.05                   # 20 Hz

    depth_hw: Tuple[int, int] = (160, 120)
    camera_fov_h_deg: float = 90.0
    camera_near: float = 0.1
    camera_far: float = 8.0

    policy_random_p: float = 0.5
    policy_scripted_p: float = 0.3
    policy_adversarial_p: float = 0.2
    # Zero velocity / angular rate — lets dynamic obstacles in the scene
    # create contact while the robot barely moves (see strategy_create_trajectory_label.md §12).
    policy_stationary_p: float = 0.0

    depth_noise_std: float = 0.01
    drop_pixel_prob: float = 0.02
    camera_jitter_deg: float = 1.0
    obstacle_texture_rand: bool = True
    lighting_rand: bool = True

    # Early termination (v3.3). When True, a rollout stops on the first
    # contact frame (+ optional grace frames) so we don't record "robot
    # lying on its side" garbage after the collision. Arrays are sliced
    # down to the actual length before writing.
    terminate_on_contact: bool = True
    post_contact_grace_frames: int = 0

    # RGB is only useful for human debugging; skip it by default to save
    # ~80% of the dataset size. When False, Trajectory.rgb is stored as a
    # length-0 placeholder array.
    save_rgb: bool = False

    out_dir: str = "dataset/pybullet_risk_v1"
    seed: int = 0

    # Risk-label horizons (frames) at 20 Hz. Kept in config for visibility.
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
                f"policy_*_p (random+scripted+adversarial+stationary) must sum to 1.0; got {total:.6f}"
            )
        if self.post_contact_grace_frames < 0:
            raise ValueError(
                "post_contact_grace_frames must be >= 0; "
                f"got {self.post_contact_grace_frames}"
            )


@dataclass
class EnvConfig:
    """
    Runtime env config for Stage B SAC. Mirrors
    docs/strategy_full_pipeline.md § 6.6.
    """

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

    lambda_risk: float = 2.0           # 0.0 for BASELINE

    action_dim: int = 3                # (v_x, v_y, w_yaw)
    action_bounds: Tuple[float, float] = (-1.0, 1.0)

    proprio_dim: int = 10              # 3+3+3+1 without last_action / joints
    include_joint_state: bool = False
    include_last_action: bool = True


# =====================================================================
# Convenience: resolve proprio dimension from a flag set
# =====================================================================

def proprio_dim_from_cfg(
    cfg: EnvConfig,
    dof: int = 0,
) -> int:
    """
    Compute d_s from an EnvConfig + optional joint DoF count. Used by the
    Actor/Critic network builders in rl/networks.py (Phase-4 work).
    """
    d = 3 + 3 + 3 + 1
    if cfg.include_last_action:
        d += cfg.action_dim
    if cfg.include_joint_state:
        if dof <= 0:
            raise ValueError(
                "include_joint_state=True requires a positive dof count."
            )
        d += 2 * dof
    return d


# =====================================================================
# Introspection helpers
# =====================================================================

def trajectory_field_names() -> Tuple[str, ...]:
    """All field names on Trajectory, in declaration order."""
    return tuple(f.name for f in fields(Trajectory))


def dump_index_row(path: Union[str, Path], row: dict) -> None:
    """Append a single JSON line to an ``index.jsonl`` catalog."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
