"""
Smoke-tests the DatasetEnv -> PyBullet bridge.

Skipped entirely when pybullet / pybullet_navigation are not importable.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_PYBULLET_OK = True
try:
    import pybullet  # noqa: F401
    import pybullet_navigation  # noqa: F401
except Exception:
    _PYBULLET_OK = False


@unittest.skipUnless(_PYBULLET_OK, "pybullet / pybullet_navigation not available")
class TestDatasetEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from create_dataset_module.env_wrapper import DatasetEnv
        cls._DatasetEnv = DatasetEnv

    def test_camera_spec_matches_indoor_defaults(self):
        env = self._DatasetEnv(gui=False, seed=0)
        try:
            intr = env.get_cam_intrinsics()
            self.assertEqual(intr.shape, (4,))
            fx, fy, cx, cy = intr.tolist()
            # With W=160, FoV_h=90°: fx = (W/2)/tan(45°) = 80; fy = fx * (H/W) for square pixels.
            self.assertAlmostEqual(fx, 80.0, places=5)
            self.assertAlmostEqual(fy, 60.0, places=5)
            self.assertAlmostEqual(cx, 80.0, places=5)
            self.assertAlmostEqual(cy, 60.0, places=5)
        finally:
            env.close()

    def test_depth_frame_shape_and_range(self):
        env = self._DatasetEnv(gui=False, seed=1)
        try:
            depth = env.get_depth_frame()
            # RL_Env returns (H, W) for depth, not (W, H).
            self.assertEqual(depth.shape, (120, 160))
            self.assertEqual(depth.dtype, np.float32)
            self.assertGreaterEqual(float(depth.min()), 0.0)
            self.assertLessEqual(float(depth.max()), 8.0 + 1e-4)
        finally:
            env.close()

    def test_extrinsics_are_orthogonal(self):
        env = self._DatasetEnv(gui=False, seed=2)
        try:
            R, t = env.get_cam_extrinsics_Rt()
            self.assertEqual(R.shape, (3, 3))
            self.assertEqual(t.shape, (3,))
            err = float(np.abs(R @ R.T - np.eye(3)).max())
            self.assertLess(err, 1e-3)
        finally:
            env.close()

    def test_step_updates_contact_flag_api(self):
        env = self._DatasetEnv(gui=False, seed=3)
        try:
            env.step(0.5, 0.0)
            flag = env.get_contact_flag()
            # No strict assertion on the value - just that it's a bool.
            self.assertIsInstance(flag, bool)
        finally:
            env.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
