"""
DatasetEnv — thin wrapper around pybullet_navigation.RL_Env for offline
Stage A dataset generation.

Responsibilities:
  * Force DIRECT mode by default (no GUI during long rollouts).
  * Silence the CSV / PNG logging that RL_Env does on construction.
  * Override get_camera_data to match the indoor spec
    (160x120, FoV_h=90°, near=0.1, far=8.0) from
    docs/strategy_full_pipeline.md § 5.1 / § 6.6.
  * Expose accessors that the DataGenerator needs to fill a Trajectory:
        get_depth_frame, get_cam_intrinsics, get_cam_extrinsics_Rt,
        get_ego_state, get_ego_vel, get_contact_flag.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# pybullet is imported lazily via pybullet_navigation. We only touch it when
# the class is actually instantiated so pure-numpy tests can import this
# module without PyBullet being installed.
try:
    import pybullet_navigation as _pbn  # noqa: F401  (used at runtime)
    _HAS_PYBULLET = True
    _IMPORT_ERROR: Optional[BaseException] = None
except Exception as _err:
    _HAS_PYBULLET = False
    _IMPORT_ERROR = _err


class DatasetEnv:
    """
    Dataset-time wrapper. Lazy ``__init__`` — it spins up PyBullet the
    first time you call a method that needs it, so test imports stay
    cheap on machines without pybullet.
    """

    DEFAULT_DEPTH_HW: Tuple[int, int] = (160, 120)   # (W, H) per § 5.1
    DEFAULT_FOV_DEG: float = 90.0
    DEFAULT_NEAR: float = 0.1
    DEFAULT_FAR: float = 8.0

    def __init__(
        self,
        gui: bool = False,
        depth_hw: Tuple[int, int] = DEFAULT_DEPTH_HW,
        fov_h_deg: float = DEFAULT_FOV_DEG,
        near: float = DEFAULT_NEAR,
        far: float = DEFAULT_FAR,
        seed: Optional[int] = None,
    ) -> None:
        if not _HAS_PYBULLET:
            raise ImportError(
                "DatasetEnv requires pybullet_navigation / pybullet. "
                f"Original import error: {_IMPORT_ERROR!r}"
            )
        self.depth_hw = depth_hw
        self.fov_h_deg = fov_h_deg
        self.near = near
        self.far = far

        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)

        # Build the env and silence the logging hooks before they touch disk.
        self._inner = self._make_inner(gui=gui)

    # ---------- factory ----------
    def _make_inner(self, gui: bool):
        from pybullet_navigation import RL_Env

        class _QuietRLEnv(RL_Env):
            # Override the logging / drawing hooks to no-ops BEFORE __init__
            # runs them.
            def _init_logging_files(self_inner):
                self_inner.logs_root_dir = None
                self_inner.run_id = ""
                self_inner.logs_dir = None
                self_inner.episodes_dir = None
                self_inner.epoch_csv_path = None
                self_inner.position_csv_path = None
                self_inner.base_map_path = None
                self_inner.obstacles_csv_path = None
                self_inner.current_episode_dir = None
                self_inner.current_episode_log_csv_path = None
                self_inner.current_episode_position_csv_path = None
                self_inner.current_episode_pc_csv_path = None

            def _init_episode_logging(self_inner):
                return

            def _append_position_log(self_inner, x, y):
                return

            def _append_experiment_log(self_inner, row):
                return

            def _append_pointcloud_csv(self_inner, step, pc):
                return

            def _try_print_pc(self_inner, step):
                return

            def draw_episode_initial_map(self_inner, save_path=None):
                return None

            def draw_episode_path(self_inner, episode=None, save_path=None):
                return None

            def _close_episode(self_inner, reason="end"):
                return

        return _QuietRLEnv(gui=gui)

    # ---------- low-level accessors used by the generator ----------
    def get_camera_data(self):
        """
        One call returns everything DataGenerator needs for a single frame:
        (rgb, depth_m, R_cam_to_world, t_cam_in_world).
        """
        import pybullet as p  # noqa: F401

        W, H = self.depth_hw
        rgb, depth_m, _seg, _pc_cam, _pc_world = self._inner.get_camera_data(
            width=W, height=H, fov=self.fov_h_deg, near=self.near, far=self.far,
        )

        cam_pos, forward, right, up = self._inner._camera_pose()
        # world_points = cam_pos + right * x_cam - up * y_cam + forward * z_cam
        # -> R (cam->world) has columns [right, -up, forward]
        R = np.stack([right, -up, forward], axis=1).astype(np.float32)
        t = cam_pos.astype(np.float32).reshape(3)
        return rgb.astype(np.uint8), depth_m.astype(np.float32), R, t

    def get_depth_frame(self) -> np.ndarray:
        """(H, W) float32 metric depth using the configured intrinsics."""
        _, depth_m, _, _ = self.get_camera_data()
        return depth_m

    def get_cam_intrinsics(self) -> np.ndarray:
        """(4,) float32 [fx, fy, cx, cy] for the current camera spec."""
        W, H = self.depth_hw
        fov_h_rad = float(np.deg2rad(self.fov_h_deg))
        fx = (W / 2.0) / float(np.tan(fov_h_rad / 2.0))
        # Keep square pixels: derive fy from fx and aspect ratio so non-square
        # depth frames (e.g., 160x120) are not vertically warped.
        fy = fx * (H / W)
        cx, cy = W / 2.0, H / 2.0
        return np.array([fx, fy, cx, cy], dtype=np.float32)

    def get_cam_extrinsics_Rt(self) -> Tuple[np.ndarray, np.ndarray]:
        """(R (3x3) float32, t (3,) float32) camera->world."""
        _, _, R, t = self.get_camera_data()
        return R, t

    def get_ego_state(self) -> np.ndarray:
        """(6,) float32 [x, y, z, roll, pitch, yaw]."""
        import pybullet as p
        pos, orn = p.getBasePositionAndOrientation(self._inner.robot)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        return np.array(
            [pos[0], pos[1], pos[2], roll, pitch, yaw], dtype=np.float32
        )

    def get_ego_vel(self) -> np.ndarray:
        """(6,) float32 [vx, vy, vz, wx, wy, wz]."""
        import pybullet as p
        lin, ang = p.getBaseVelocity(self._inner.robot)
        return np.array(
            [lin[0], lin[1], lin[2], ang[0], ang[1], ang[2]], dtype=np.float32
        )

    def get_contact_flag(self) -> bool:
        """True iff the robot body is in contact with any non-ground body."""
        return bool(self._inner.check_collision())

    # ---------- control ----------
    def drive(self, linear_x: float, angular_z: float) -> None:
        self._inner.drive(float(linear_x), float(angular_z))

    def step_physics(self, n_sub: Optional[int] = None) -> None:
        """
        Advance physics by ``n_sub`` simulation sub-steps. Default matches
        RL_Env: SIM_FREQ // CTRL_FREQ steps per control tick (4 sub-steps
        at the 200/50 defaults).
        """
        import pybullet as p
        from pybullet_navigation import SIM_FREQ, CTRL_FREQ
        n = int(n_sub) if n_sub is not None else (SIM_FREQ // CTRL_FREQ)
        for _ in range(n):
            p.stepSimulation()
        self._inner._bounce_dynamic_obs()

    def step(self, linear_x: float, angular_z: float) -> None:
        """One control tick: drive(action) -> physics sub-steps."""
        self.drive(linear_x, angular_z)
        self.step_physics()

    # ---------- scene info ----------
    def get_obstacle_aabb(self) -> np.ndarray:
        """
        (M, 6) float32 array of axis-aligned bounding boxes for every
        obstacle (static + dynamic) in the scene. Columns:
        [x_min, y_min, z_min, x_max, y_max, z_max].
        """
        import pybullet as p
        rows = []
        for bid in self._inner.static_obs + self._inner.dynamic_obs:
            mn, mx = p.getAABB(bid)
            rows.append([mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]])
        if not rows:
            return np.zeros((0, 6), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def reset_scene(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)
        self._inner._reset_world_for_next_episode()
        self._inner._reset_episode_metrics()

    def close(self) -> None:
        try:
            import pybullet as p
            if p.isConnected(self._inner.client):
                p.disconnect(self._inner.client)
        except Exception:  # pragma: no cover - shutdown races
            pass
