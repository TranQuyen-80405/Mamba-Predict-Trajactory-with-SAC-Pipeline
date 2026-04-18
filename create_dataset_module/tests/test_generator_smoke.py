"""
DataGenerator smoke test: 1 scene x 1 rollout x 20 frames. Skipped when
PyBullet is not installed.
"""

from __future__ import annotations

import json
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

_PYBULLET_OK = True
try:
    import pybullet  # noqa: F401
    import pybullet_navigation  # noqa: F401
except Exception:
    _PYBULLET_OK = False


@unittest.skipUnless(_PYBULLET_OK, "pybullet / pybullet_navigation not available")
class TestDataGeneratorSmoke(unittest.TestCase):
    def test_one_rollout_20_frames(self):
        from create_dataset_module.generator import DataGenerator
        from data_contracts import DataGenConfig, Trajectory

        with tempfile.TemporaryDirectory() as tmp:
            cfg = DataGenConfig(
                n_scenes=1, rollouts_per_scene=1, frames_per_rollout=20,
                out_dir=tmp, seed=0,
                # Balanced mix so either policy branch is hit safely.
                policy_random_p=1.0,
                policy_scripted_p=0.0,
                policy_adversarial_p=0.0,
                # Keep the original length invariant for this smoke test.
                terminate_on_contact=False,
                # Exercise domain rand / rgb-off defaults explicitly.
                depth_noise_std=0.0,
                drop_pixel_prob=0.0,
                camera_jitter_deg=0.0,
                save_rgb=False,
            )
            gen = DataGenerator(cfg, out_dir=tmp)
            written = gen.run()

            self.assertEqual(written, 1)

            # Verify the index and the payload.
            index = Path(tmp) / "index.jsonl"
            self.assertTrue(index.exists())
            lines = index.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            row = json.loads(lines[0])
            self.assertEqual(row["scene_id"], 0)
            self.assertEqual(row["T"], 20)

            traj = Trajectory.from_npz(Path(tmp) / row["path"])
            self.assertEqual(traj.T, 20)
            self.assertEqual(traj.depth.shape, (20, 120, 160))
            self.assertEqual(traj.depth.dtype, np.float16)
            # Orthogonality was validated in __post_init__; just sanity-check
            # the risk horizons' monotonic ordering when a contact happens.
            if traj.contact_flag.any():
                self.assertGreaterEqual(
                    float(traj.risk_2s.sum()), float(traj.risk_1s.sum())
                )
                self.assertGreaterEqual(
                    float(traj.risk_1s.sum()), float(traj.risk_05s.sum())
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
