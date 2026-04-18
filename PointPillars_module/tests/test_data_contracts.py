"""
Unit tests for data_contracts.py.

All tests are pure numpy/torch and MUST run on a CPU-only box.

Run:
    cd PointPillars_module
    python -m unittest tests.test_data_contracts -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from data_contracts import (  # noqa: E402
    DataGenConfig,
    EnvConfig,
    ProprioState,
    RiskBatch,
    RiskSample,
    Trajectory,
    Transition,
    dump_index_row,
    proprio_dim_from_cfg,
    trajectory_field_names,
)


# =====================================================================
# Helpers
# =====================================================================

def _make_traj(T: int = 30, A: int = 2, H: int = 120, W: int = 160) -> Trajectory:
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.2, 7.0, size=(T, H, W)).astype(np.float16)
    rgb = (rng.integers(0, 255, size=(T, H, W, 3))).astype(np.uint8)

    cam_intr = np.array([80.0, 80.0, 80.0, 60.0], dtype=np.float32)
    # identity rotation for all frames -> trivially orthogonal
    R = np.broadcast_to(np.eye(3, dtype=np.float32), (T, 3, 3)).copy()
    t = np.zeros((T, 3), dtype=np.float32)

    ego_state = np.zeros((T, 6), dtype=np.float32)
    ego_vel = np.zeros((T, 6), dtype=np.float32)
    action = rng.uniform(-1, 1, size=(T, A)).astype(np.float32)

    contact_flag = np.zeros((T,), dtype=np.bool_)
    contact_flag[T - 5] = True  # one contact 5 frames before end

    # Manually derive horizons to keep the factory honest.
    def lookahead(flag: np.ndarray, horizon: int) -> np.ndarray:
        out = np.zeros_like(flag, dtype=np.float32)
        for i in range(len(flag)):
            out[i] = float(flag[i: i + horizon].any())
        return out

    risk_05s = lookahead(contact_flag, 10)
    risk_1s = lookahead(contact_flag, 20)
    risk_2s = lookahead(contact_flag, 40)

    return Trajectory(
        scene_id=7, rollout_id=3, T=T,
        depth=depth, rgb=rgb,
        cam_intrinsics=cam_intr,
        cam_extr_R=R, cam_extr_t=t,
        ego_state=ego_state, ego_vel=ego_vel,
        action=action,
        contact_flag=contact_flag,
        risk_05s=risk_05s, risk_1s=risk_1s, risk_2s=risk_2s,
    )


# =====================================================================
# Trajectory
# =====================================================================

class TestTrajectoryBasics(unittest.TestCase):
    def test_construct_and_fields(self):
        traj = _make_traj()
        self.assertEqual(traj.scene_id, 7)
        self.assertEqual(traj.rollout_id, 3)
        self.assertEqual(traj.T, 30)
        self.assertEqual(traj.depth.dtype, np.float16)
        self.assertEqual(traj.depth.shape, (30, 120, 160))
        self.assertEqual(traj.cam_extr_R.shape, (30, 3, 3))

    def test_field_introspection(self):
        names = trajectory_field_names()
        # scene_id / rollout_id / T always come first; the full set is stable.
        self.assertEqual(names[:3], ("scene_id", "rollout_id", "T"))
        for required in (
            "depth", "cam_intrinsics", "cam_extr_R", "cam_extr_t",
            "ego_state", "ego_vel", "action", "contact_flag",
            "risk_05s", "risk_1s", "risk_2s",
        ):
            self.assertIn(required, names)


class TestTrajectoryValidation(unittest.TestCase):
    def test_bad_depth_dtype(self):
        traj = _make_traj()
        bad = traj.depth.astype(np.float32)
        with self.assertRaisesRegex(ValueError, "depth must be float16"):
            Trajectory(
                scene_id=traj.scene_id, rollout_id=traj.rollout_id, T=traj.T,
                depth=bad, rgb=traj.rgb,
                cam_intrinsics=traj.cam_intrinsics,
                cam_extr_R=traj.cam_extr_R, cam_extr_t=traj.cam_extr_t,
                ego_state=traj.ego_state, ego_vel=traj.ego_vel,
                action=traj.action,
                contact_flag=traj.contact_flag,
                risk_05s=traj.risk_05s, risk_1s=traj.risk_1s,
                risk_2s=traj.risk_2s,
            )

    def test_depth_out_of_range(self):
        traj = _make_traj()
        bad_depth = traj.depth.copy()
        bad_depth[0, 0, 0] = np.float16(50.0)  # > 10 m
        with self.assertRaisesRegex(ValueError, "depth out of sanity range"):
            Trajectory(
                scene_id=traj.scene_id, rollout_id=traj.rollout_id, T=traj.T,
                depth=bad_depth, rgb=traj.rgb,
                cam_intrinsics=traj.cam_intrinsics,
                cam_extr_R=traj.cam_extr_R, cam_extr_t=traj.cam_extr_t,
                ego_state=traj.ego_state, ego_vel=traj.ego_vel,
                action=traj.action,
                contact_flag=traj.contact_flag,
                risk_05s=traj.risk_05s, risk_1s=traj.risk_1s,
                risk_2s=traj.risk_2s,
            )

    def test_non_binary_risk(self):
        traj = _make_traj()
        bad = traj.risk_1s.copy()
        bad[0] = 0.5
        with self.assertRaisesRegex(ValueError, "risk_1s must contain only"):
            Trajectory(
                scene_id=traj.scene_id, rollout_id=traj.rollout_id, T=traj.T,
                depth=traj.depth, rgb=traj.rgb,
                cam_intrinsics=traj.cam_intrinsics,
                cam_extr_R=traj.cam_extr_R, cam_extr_t=traj.cam_extr_t,
                ego_state=traj.ego_state, ego_vel=traj.ego_vel,
                action=traj.action,
                contact_flag=traj.contact_flag,
                risk_05s=traj.risk_05s, risk_1s=bad,
                risk_2s=traj.risk_2s,
            )

    def test_non_orthogonal_rotation(self):
        traj = _make_traj()
        bad_R = traj.cam_extr_R.copy()
        bad_R[5, 0, 0] = 2.0                   # break R R^T = I
        with self.assertRaisesRegex(ValueError, "cam_extr_R is not orthogonal"):
            Trajectory(
                scene_id=traj.scene_id, rollout_id=traj.rollout_id, T=traj.T,
                depth=traj.depth, rgb=traj.rgb,
                cam_intrinsics=traj.cam_intrinsics,
                cam_extr_R=bad_R, cam_extr_t=traj.cam_extr_t,
                ego_state=traj.ego_state, ego_vel=traj.ego_vel,
                action=traj.action,
                contact_flag=traj.contact_flag,
                risk_05s=traj.risk_05s, risk_1s=traj.risk_1s,
                risk_2s=traj.risk_2s,
            )

    def test_shape_mismatch(self):
        traj = _make_traj()
        bad_action = traj.action[:-1]
        with self.assertRaisesRegex(ValueError, "action must be"):
            Trajectory(
                scene_id=traj.scene_id, rollout_id=traj.rollout_id, T=traj.T,
                depth=traj.depth, rgb=traj.rgb,
                cam_intrinsics=traj.cam_intrinsics,
                cam_extr_R=traj.cam_extr_R, cam_extr_t=traj.cam_extr_t,
                ego_state=traj.ego_state, ego_vel=traj.ego_vel,
                action=bad_action,
                contact_flag=traj.contact_flag,
                risk_05s=traj.risk_05s, risk_1s=traj.risk_1s,
                risk_2s=traj.risk_2s,
            )


class TestTrajectoryNpzRoundtrip(unittest.TestCase):
    def test_roundtrip_preserves_dtypes_and_values(self):
        traj = _make_traj()
        with tempfile.TemporaryDirectory() as tmp:
            out = Trajectory.to_npz(traj, Path(tmp) / "r.npz")
            self.assertTrue(out.exists())

            loaded = Trajectory.from_npz(out)

            self.assertEqual(loaded.scene_id, traj.scene_id)
            self.assertEqual(loaded.rollout_id, traj.rollout_id)
            self.assertEqual(loaded.T, traj.T)

            # depth dtype + exact bitwise equality for the float16 payload
            self.assertEqual(loaded.depth.dtype, np.float16)
            np.testing.assert_array_equal(loaded.depth, traj.depth)

            for fname in (
                "cam_intrinsics", "cam_extr_R", "cam_extr_t",
                "ego_state", "ego_vel", "action",
                "risk_05s", "risk_1s", "risk_2s",
            ):
                np.testing.assert_array_equal(
                    getattr(loaded, fname), getattr(traj, fname)
                )
            np.testing.assert_array_equal(loaded.contact_flag, traj.contact_flag)


# =====================================================================
# RiskSample / RiskBatch
# =====================================================================

class TestRiskSampleAndBatch(unittest.TestCase):
    def _make_sample(self, t_ctx: int = 3, A: int = 2) -> RiskSample:
        # Variable per-frame N_i to exercise the list-of-tensors contract.
        pts_seq = [torch.randn(5 + i, 4, dtype=torch.float32) for i in range(t_ctx)]
        H = 10
        return RiskSample(
            pts_seq=pts_seq,
            action_seq=torch.zeros(t_ctx, A, dtype=torch.float32),
            ego_vel_seq=torch.zeros(t_ctx, 6, dtype=torch.float32),
            risk_05s=torch.tensor(0.0, dtype=torch.float32),
            risk_1s=torch.tensor(1.0, dtype=torch.float32),
            risk_2s=torch.tensor(1.0, dtype=torch.float32),
            traj_future_xyyaw=torch.zeros(H, 3, dtype=torch.float32),
            scene_id=0, rollout_id=0, frame_t=7,
        )

    def test_risk_sample_shapes(self):
        s = self._make_sample(t_ctx=4, A=3)
        self.assertEqual(len(s.pts_seq), 4)
        for pts in s.pts_seq:
            self.assertEqual(pts.ndim, 2)
            self.assertEqual(pts.shape[1], 4)
        self.assertEqual(s.action_seq.shape, (4, 3))

    def test_collate_into_risk_batch(self):
        T_CTX, A, B = 3, 2, 2
        samples = [self._make_sample(t_ctx=T_CTX, A=A) for _ in range(B)]

        # Mimic the collate contract that risk_dataset.py will ship.
        pts_seq = [[samples[b].pts_seq[t] for b in range(B)] for t in range(T_CTX)]
        action_seq = torch.stack([s.action_seq for s in samples], dim=0)
        ego_vel_seq = torch.stack([s.ego_vel_seq for s in samples], dim=0)
        risk_05s = torch.stack([s.risk_05s for s in samples], dim=0)
        risk_1s = torch.stack([s.risk_1s for s in samples], dim=0)
        risk_2s = torch.stack([s.risk_2s for s in samples], dim=0)
        traj_future_xyyaw = torch.stack([s.traj_future_xyyaw for s in samples], dim=0)

        batch = RiskBatch(
            pts_seq=pts_seq,
            action_seq=action_seq, ego_vel_seq=ego_vel_seq,
            risk_05s=risk_05s, risk_1s=risk_1s, risk_2s=risk_2s,
            traj_future_xyyaw=traj_future_xyyaw,
        )

        self.assertEqual(batch.batch_size, B)
        self.assertEqual(batch.t_ctx, T_CTX)
        self.assertEqual(len(batch.pts_seq), T_CTX)
        self.assertEqual(len(batch.pts_seq[0]), B)
        self.assertEqual(batch.action_seq.shape, (B, T_CTX, A))
        self.assertEqual(batch.risk_targets().shape, (B, 3))
        self.assertEqual(batch.traj_future_xyyaw.shape, (B, 10, 3))


# =====================================================================
# ProprioState
# =====================================================================

class TestProprioState(unittest.TestCase):
    def _make(self, A: int = 3, dof: int = 0) -> ProprioState:
        return ProprioState(
            base_lin_vel=np.zeros(3, dtype=np.float32),
            base_ang_vel=np.zeros(3, dtype=np.float32),
            goal_rel=np.ones(3, dtype=np.float32),
            heading_err=0.25,
            last_action=np.zeros(A, dtype=np.float32),
            joint_q=np.zeros(dof, dtype=np.float32) if dof else None,
            joint_dq=np.zeros(dof, dtype=np.float32) if dof else None,
        )

    def test_minimal_dim_no_last_action_no_joints(self):
        p = self._make(A=3)
        v = p.to_tensor(include_joint_state=False, include_last_action=False)
        self.assertEqual(v.shape, (10,))  # 3+3+3+1
        self.assertEqual(v.dtype, torch.float32)

    def test_dim_with_last_action(self):
        p = self._make(A=3)
        v = p.to_tensor(include_joint_state=False, include_last_action=True)
        self.assertEqual(v.shape, (13,))  # 10 + A

    def test_dim_with_joint_state(self):
        p = self._make(A=3, dof=12)
        v = p.to_tensor(include_joint_state=True, include_last_action=True)
        self.assertEqual(v.shape, (37,))  # 13 + 2*12

    def test_joint_state_missing_raises(self):
        p = self._make(A=3, dof=0)
        with self.assertRaises(ValueError):
            p.to_tensor(include_joint_state=True)


# =====================================================================
# Configs
# =====================================================================

class TestConfigDefaults(unittest.TestCase):
    def test_data_gen_cfg_defaults(self):
        c = DataGenConfig()
        self.assertEqual(c.depth_hw, (160, 120))
        self.assertAlmostEqual(c.camera_fov_h_deg, 90.0)
        self.assertAlmostEqual(c.camera_far, 8.0)
        self.assertAlmostEqual(c.dt, 0.05)
        self.assertEqual(c.horizon_05s_frames, 10)
        self.assertEqual(c.horizon_1s_frames, 20)
        self.assertEqual(c.horizon_2s_frames, 40)

    def test_data_gen_cfg_policy_mix_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "policy_\\*_p .*must sum to 1"):
            DataGenConfig(
                policy_random_p=0.5,
                policy_scripted_p=0.5,
                policy_adversarial_p=0.5,
                policy_stationary_p=0.0,
            )

    def test_data_gen_cfg_stationary_default_zero(self):
        c = DataGenConfig()
        self.assertAlmostEqual(c.policy_stationary_p, 0.0)

    def test_env_cfg_defaults(self):
        c = EnvConfig()
        self.assertEqual(c.depth_hw, (160, 120))
        self.assertAlmostEqual(c.camera_far, 8.0)
        self.assertEqual(c.T_ctx, 10)
        self.assertAlmostEqual(c.lambda_risk, 2.0)

    def test_proprio_dim_helper(self):
        c = EnvConfig(include_last_action=False, include_joint_state=False)
        self.assertEqual(proprio_dim_from_cfg(c), 10)
        c = EnvConfig(include_last_action=True, include_joint_state=False)
        self.assertEqual(proprio_dim_from_cfg(c), 10 + c.action_dim)
        c = EnvConfig(include_last_action=True, include_joint_state=True)
        self.assertEqual(proprio_dim_from_cfg(c, dof=12), 10 + c.action_dim + 24)
        with self.assertRaises(ValueError):
            proprio_dim_from_cfg(EnvConfig(include_joint_state=True), dof=0)


# =====================================================================
# Transition + index helper
# =====================================================================

class TestTransitionAndIndex(unittest.TestCase):
    def test_transition_fields(self):
        tr = Transition(
            s=np.zeros(13, dtype=np.float32),
            s_next=np.zeros(13, dtype=np.float32),
            action=np.zeros(3, dtype=np.float32),
            r_env=0.1, r_risk=-0.2, done=False,
            episode_id=1, frame_idx=5,
        )
        self.assertEqual(tr.s.shape, (13,))
        self.assertAlmostEqual(tr.r_risk, -0.2)
        self.assertFalse(tr.done)

    def test_dump_index_row_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "index.jsonl"
            dump_index_row(path, {"scene_id": 0, "rollout_id": 1})
            dump_index_row(path, {"scene_id": 0, "rollout_id": 2})
            self.assertTrue(path.exists())
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            import json
            self.assertEqual(json.loads(lines[1])["rollout_id"], 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
