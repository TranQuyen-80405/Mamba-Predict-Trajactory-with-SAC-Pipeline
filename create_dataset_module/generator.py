"""
DataGenerator: roll a DatasetEnv forward with a mixture of policies and
serialize each rollout as ``Trajectory.to_npz`` + an ``index.jsonl`` row.

Risk labels are derived exactly as specified in
docs/strategy_full_pipeline.md § 5.1:

    risk_05s[t] = any(contact_flag[t : t + 10])
    risk_1s[t]  = any(contact_flag[t : t + 20])
    risk_2s[t]  = any(contact_flag[t : t + 40])

v3.3 changes:
  * Domain randomization is actually applied: depth_noise_std,
    drop_pixel_prob, camera_jitter_deg (from DataGenConfig).
  * Rollouts optionally terminate on first contact (+ grace frames) and
    the arrays are sliced to the real length before serialization.
  * RGB is skipped by default (save_rgb=False); the placeholder on disk
    is a length-0 uint8 array to keep the dataclass happy.
  * run() aggregates per-policy counts, positive ratios, and early-
    termination stats and prints / stores them on ``self.last_stats``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PP_PKG = os.path.join(_ROOT, "PointPillars_module")
for _p in (_ROOT, _PP_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_contracts import (  # noqa: E402
    DataGenConfig,
    Trajectory,
    dump_index_row,
)

from .policies import (  # noqa: E402
    AdversarialPolicy,
    RandomPolicy,
    ScriptedPolicy,
    StationaryPolicy,
    _BasePolicy,
)


# ---------------------------------------------------------------------
# Risk-label helper
# ---------------------------------------------------------------------

def lookahead_any(contact_flag: np.ndarray, horizon: int) -> np.ndarray:
    """
    Return a (T,) float32 array where ``out[t] = any(contact_flag[t:t+H])``.

    Edge case: for t near the end of the trajectory the window shrinks -
    we do NOT pad with True, so the last few frames get fewer chances to
    be flagged. This matches § 5.1 of the strategy doc.
    """
    flag = np.asarray(contact_flag, dtype=np.bool_)
    T = int(flag.shape[0])
    out = np.zeros(T, dtype=np.float32)
    if T == 0 or horizon <= 0:
        return out
    for t in range(T):
        end = min(T, t + horizon)
        out[t] = float(flag[t:end].any())
    return out


# ---------------------------------------------------------------------
# Domain randomization helpers
# ---------------------------------------------------------------------

def _apply_depth_noise(
    depth_m: np.ndarray,
    std: float,
    rng: np.random.Generator,
    far: float,
) -> np.ndarray:
    """Additive Gaussian noise on valid depth pixels. Clipped to [0, far]."""
    if std <= 0.0:
        return depth_m
    noise = rng.normal(loc=0.0, scale=float(std), size=depth_m.shape).astype(
        depth_m.dtype, copy=False
    )
    out = depth_m + noise
    np.clip(out, 0.0, float(far), out=out)
    return out


def _apply_pixel_dropout(
    depth_m: np.ndarray,
    prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Drop random pixels to 0 (depth sensors fail stochastically)."""
    if prob <= 0.0:
        return depth_m
    mask = rng.random(size=depth_m.shape) < float(prob)
    if mask.any():
        depth_m = depth_m.copy()
        depth_m[mask] = 0.0
    return depth_m


def _apply_camera_jitter(
    R: np.ndarray,
    jitter_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Perturb camera->world rotation by a small random rotation drawn from
    SO(3) around a uniformly sampled axis. ``jitter_deg`` is the 1-sigma
    magnitude (isotropic, not peak).
    """
    if jitter_deg <= 0.0:
        return R
    sigma_rad = float(np.deg2rad(jitter_deg))
    # Random axis (uniform on the sphere) + Gaussian angle magnitude.
    axis = rng.normal(size=3).astype(np.float32)
    axis /= (np.linalg.norm(axis) + 1e-8)
    angle = float(rng.normal(0.0, sigma_rad))
    c, s = float(np.cos(angle)), float(np.sin(angle))
    x, y, z = axis.tolist()
    K = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float32
    )
    dR = np.eye(3, dtype=np.float32) + s * K + (1.0 - c) * (K @ K)
    out = (dR @ R.astype(np.float32)).astype(np.float32)
    return out


# ---------------------------------------------------------------------
# DataGenerator
# ---------------------------------------------------------------------

class DataGenerator:
    """
    Configurable driver that runs ``n_scenes * rollouts_per_scene`` rollouts
    and writes them to ``out_dir``.

    After ``run()`` completes, ``self.last_stats`` holds the aggregate
    summary dict that was also printed to stdout (see ``_print_summary``).
    """

    def __init__(
        self,
        cfg: DataGenConfig,
        out_dir: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.out_dir = Path(out_dir if out_dir is not None else cfg.out_dir)
        self.index_path = self.out_dir / "index.jsonl"
        self.rng = np.random.default_rng(cfg.seed)
        self.last_stats: Dict[str, object] = {}

    # ---------- public entry points ----------
    def run(self) -> int:
        """
        Execute the full generation plan. Returns the number of .npz files
        written. Populates ``self.last_stats`` with summary metrics.
        """
        from .env_wrapper import DatasetEnv

        self.out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        total_frames = 0
        total_pos = {"05s": 0, "1s": 0, "2s": 0}
        policy_counts: Dict[str, int] = {
            "random": 0,
            "scripted": 0,
            "adversarial": 0,
            "stationary": 0,
        }
        terminated_early = 0

        for s in range(self.cfg.n_scenes):
            env = DatasetEnv(
                gui=False,
                depth_hw=self.cfg.depth_hw,
                fov_h_deg=self.cfg.camera_fov_h_deg,
                near=self.cfg.camera_near,
                far=self.cfg.camera_far,
                seed=int(self.cfg.seed + s),
            )
            try:
                for r in range(self.cfg.rollouts_per_scene):
                    traj, meta = self._rollout(env, scene_id=s, rollout_id=r)
                    name = f"s{s:04d}_r{r:02d}.npz"
                    path = self.out_dir / name
                    traj.to_npz(path)
                    n_pos_1s = int((traj.risk_1s > 0).sum())
                    n_pos_05s = int((traj.risk_05s > 0).sum())
                    n_pos_2s = int((traj.risk_2s > 0).sum())
                    dump_index_row(self.index_path, {
                        "path": name,
                        "scene_id": s,
                        "rollout_id": r,
                        "T": int(traj.T),
                        "policy": meta["policy"],
                        "terminated_on_contact": bool(meta["terminated_on_contact"]),
                        "n_positive_05s": n_pos_05s,
                        "n_positive_1s": n_pos_1s,
                        "n_positive_2s": n_pos_2s,
                    })
                    written += 1
                    total_frames += int(traj.T)
                    total_pos["05s"] += n_pos_05s
                    total_pos["1s"] += n_pos_1s
                    total_pos["2s"] += n_pos_2s
                    policy_counts[meta["policy"]] += 1
                    if meta["terminated_on_contact"]:
                        terminated_early += 1
                    if r + 1 < self.cfg.rollouts_per_scene:
                        env.reset_scene()
            finally:
                env.close()

        self.last_stats = {
            "written": int(written),
            "total_frames": int(total_frames),
            "positive_ratio_05s": (total_pos["05s"] / total_frames) if total_frames else 0.0,
            "positive_ratio_1s": (total_pos["1s"] / total_frames) if total_frames else 0.0,
            "positive_ratio_2s": (total_pos["2s"] / total_frames) if total_frames else 0.0,
            "policy_counts": dict(policy_counts),
            "terminated_early": int(terminated_early),
        }
        self._print_summary()
        return written

    # ---------- single rollout ----------
    def _pick_policy(self) -> "tuple[_BasePolicy, str]":
        u = float(self.rng.random())
        pr = float(self.cfg.policy_random_p)
        ps = float(self.cfg.policy_scripted_p)
        pa = float(self.cfg.policy_adversarial_p)
        # pst = policy_stationary_p; pr+ps+pa+pst == 1.0 (validated in DataGenConfig)
        if u < pr:
            return RandomPolicy(dt=self.cfg.dt), "random"
        if u < pr + ps:
            return ScriptedPolicy(), "scripted"
        if u < pr + ps + pa:
            return AdversarialPolicy(), "adversarial"
        return StationaryPolicy(), "stationary"

    def _rollout(
        self,
        env,
        scene_id: int,
        rollout_id: int,
    ) -> "tuple[Trajectory, Dict[str, object]]":
        cfg = self.cfg
        T_max = int(cfg.frames_per_rollout)
        W, H = cfg.depth_hw

        policy, policy_name = self._pick_policy()
        rollout_seed = int(cfg.seed + scene_id * 1000 + rollout_id)
        policy.reset(np.random.default_rng(rollout_seed))
        # Separate stream for domain randomization so it doesn't perturb
        # policy sampling reproducibility.
        dr_rng = np.random.default_rng(rollout_seed ^ 0xA5A5A5A5)

        intrinsics = env.get_cam_intrinsics()

        depth = np.zeros((T_max, H, W), dtype=np.float16)
        rgb_store_raw = cfg.save_rgb
        rgb = (
            np.zeros((T_max, H, W, 3), dtype=np.uint8)
            if rgb_store_raw
            else None
        )
        R_seq = np.zeros((T_max, 3, 3), dtype=np.float32)
        t_seq = np.zeros((T_max, 3), dtype=np.float32)
        ego_state = np.zeros((T_max, 6), dtype=np.float32)
        ego_vel = np.zeros((T_max, 6), dtype=np.float32)
        action = np.zeros((T_max, 2), dtype=np.float32)
        contact = np.zeros((T_max,), dtype=np.bool_)

        obstacle_aabb = env.get_obstacle_aabb()

        T_actual = T_max
        contact_frame = -1
        for t in range(T_max):
            rgb_t, depth_t, R_t, t_vec = env.get_camera_data()

            # --- domain randomization (v3.3) ---------------------------
            depth_t = np.clip(depth_t, 0.0, cfg.camera_far)
            depth_t = _apply_depth_noise(
                depth_t, cfg.depth_noise_std, dr_rng, cfg.camera_far
            )
            depth_t = _apply_pixel_dropout(
                depth_t, cfg.drop_pixel_prob, dr_rng
            )
            R_t = _apply_camera_jitter(
                R_t, cfg.camera_jitter_deg, dr_rng
            )
            # -----------------------------------------------------------

            depth[t] = depth_t.astype(np.float16)
            if rgb_store_raw:
                rgb[t] = rgb_t
            R_seq[t] = R_t
            t_seq[t] = t_vec
            ego_state[t] = env.get_ego_state()
            ego_vel[t] = env.get_ego_vel()

            obs = {
                "ego_xy": ego_state[t, :2],
                "yaw": float(ego_state[t, 5]),
                "obstacles_xy": 0.5 * (
                    obstacle_aabb[:, 0:2] + obstacle_aabb[:, 3:5]
                ) if obstacle_aabb.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32),
            }
            a = policy.act(obs).astype(np.float32)
            action[t] = a
            env.step(float(a[0]), float(a[1]))
            contact[t] = env.get_contact_flag()

            # --- early termination (v3.3) ------------------------------
            if cfg.terminate_on_contact and contact[t] and contact_frame < 0:
                contact_frame = t
                # Keep `post_contact_grace_frames` extra frames so the
                # risk_* lookahead windows near the contact still see
                # "real" post-impact physics if the user wants it.
                T_actual = min(T_max, t + 1 + cfg.post_contact_grace_frames)
                break
            # -----------------------------------------------------------

        # Slice arrays down to the actual length.
        if T_actual < T_max:
            depth = depth[:T_actual]
            if rgb_store_raw:
                rgb = rgb[:T_actual]
            R_seq = R_seq[:T_actual]
            t_seq = t_seq[:T_actual]
            ego_state = ego_state[:T_actual]
            ego_vel = ego_vel[:T_actual]
            action = action[:T_actual]
            contact = contact[:T_actual]

        risk_05s = lookahead_any(contact, cfg.horizon_05s_frames)
        risk_1s = lookahead_any(contact, cfg.horizon_1s_frames)
        risk_2s = lookahead_any(contact, cfg.horizon_2s_frames)

        if not rgb_store_raw:
            # Placeholder: length-0 uint8 tensor. Trajectory doesn't
            # validate rgb shape, so this is safe for to_npz/from_npz.
            rgb = np.zeros((0,), dtype=np.uint8)

        traj = Trajectory(
            scene_id=int(scene_id),
            rollout_id=int(rollout_id),
            T=int(T_actual),
            depth=depth,
            rgb=rgb,
            cam_intrinsics=intrinsics,
            cam_extr_R=R_seq,
            cam_extr_t=t_seq,
            ego_state=ego_state,
            ego_vel=ego_vel,
            action=action,
            contact_flag=contact,
            risk_05s=risk_05s,
            risk_1s=risk_1s,
            risk_2s=risk_2s,
            obstacle_aabb=obstacle_aabb,
        )
        meta = {
            "policy": policy_name,
            "terminated_on_contact": bool(contact_frame >= 0),
            "contact_frame": int(contact_frame),
        }
        return traj, meta

    # ---------- summary printer ----------
    def _print_summary(self) -> None:
        st = self.last_stats
        if not st:
            return
        tot = int(st["total_frames"])
        tx = int(st["terminated_early"])
        tx_pct = (100.0 * tx / st["written"]) if st["written"] else 0.0
        print("")
        print("=" * 60)
        print(f" DataGenerator summary  (out_dir = {self.out_dir})")
        print("=" * 60)
        print(f"   rollouts written     : {st['written']}")
        print(f"   frames total         : {tot}")
        print(f"   early-terminated     : {tx} ({tx_pct:.1f}%)")
        print(f"   positive ratio 0.5s  : {st['positive_ratio_05s']*100:.2f}%")
        print(f"   positive ratio 1.0s  : {st['positive_ratio_1s']*100:.2f}%")
        print(f"   positive ratio 2.0s  : {st['positive_ratio_2s']*100:.2f}%")
        pc = st["policy_counts"]
        total_pc = sum(pc.values()) or 1
        for name in ("random", "scripted", "adversarial", "stationary"):
            v = pc.get(name, 0)
            print(f"   policy {name:<12}: {v}  ({100.0 * v / total_pc:.1f}%)")
        # Sanity hints so the caller spots imbalance without grepping.
        p1 = st["positive_ratio_1s"]
        if p1 < 0.05:
            print("   WARN  positive(1s) < 5% -> consider raising policy_adversarial_p")
        elif p1 > 0.5:
            print("   WARN  positive(1s) > 50% -> consider lowering policy_adversarial_p")
        print("=" * 60)


__all__ = [
    "DataGenerator",
    "lookahead_any",
    # Exposed for unit testing / reuse:
    "_apply_depth_noise",
    "_apply_pixel_dropout",
    "_apply_camera_jitter",
]
