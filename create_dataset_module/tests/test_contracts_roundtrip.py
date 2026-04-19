"""
Exercise the data_contracts Trajectory <-> npz round-trip at the scale
we actually produce (small but realistic).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_PP_PKG = os.path.join(_ROOT, "PointPillars_module")
for _p in (_ROOT, _PP_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PointPillars_module.types import Trajectory  # noqa: E402


class TestRoundtrip(unittest.TestCase):
    def test_medium_trajectory_preserves_all_fields(self):
        T, H, W, A = 50, 120, 160, 2
        rng = np.random.default_rng(42)
        depth = rng.uniform(0.1, 7.9, size=(T, H, W)).astype(np.float16)
        rgb = rng.integers(0, 255, size=(T, H, W, 3)).astype(np.uint8)
        R = np.broadcast_to(np.eye(3, dtype=np.float32), (T, 3, 3)).copy()
        t = rng.normal(size=(T, 3)).astype(np.float32)
        ego_state = rng.normal(size=(T, 6)).astype(np.float32)
        ego_vel = rng.normal(size=(T, 6)).astype(np.float32)
        action = rng.uniform(-1, 1, size=(T, A)).astype(np.float32)
        contact = np.zeros(T, dtype=np.bool_)
        contact[10] = True
        contact[30] = True

        def lk(f, h):
            out = np.zeros_like(f, dtype=np.float32)
            for i in range(len(f)):
                out[i] = float(f[i:i + h].any())
            return out

        traj = Trajectory(
            scene_id=11, rollout_id=2, T=T,
            depth=depth, rgb=rgb,
            cam_intrinsics=np.array([80, 80, 80, 60], dtype=np.float32),
            cam_extr_R=R, cam_extr_t=t,
            ego_state=ego_state, ego_vel=ego_vel,
            action=action,
            contact_flag=contact,
            risk_05s=lk(contact, 10), risk_1s=lk(contact, 20),
            risk_2s=lk(contact, 40),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = traj.to_npz(Path(tmp) / "traj.npz")
            self.assertTrue(path.exists())
            back = Trajectory.from_npz(path)

        self.assertEqual(back.T, T)
        np.testing.assert_array_equal(back.depth, traj.depth)
        np.testing.assert_array_equal(back.contact_flag, traj.contact_flag)
        np.testing.assert_allclose(back.action, traj.action)
        np.testing.assert_allclose(back.risk_1s, traj.risk_1s)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
