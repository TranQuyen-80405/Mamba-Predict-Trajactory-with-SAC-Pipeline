"""
v3.3 feature tests: early-termination on contact, save_rgb=False,
and last_stats summary. PyBullet is required.
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
class TestGeneratorFeatures(unittest.TestCase):
    def _cfg(self, tmp: str, **over):
        from data_contracts import DataGenConfig
        base = dict(
            n_scenes=1, rollouts_per_scene=1, frames_per_rollout=80,
            out_dir=tmp, seed=0,
            policy_random_p=0.0,
            policy_scripted_p=0.0,
            policy_adversarial_p=1.0,
            terminate_on_contact=True,
            post_contact_grace_frames=0,
            save_rgb=False,
            depth_noise_std=0.0,
            drop_pixel_prob=0.0,
            camera_jitter_deg=0.0,
        )
        base.update(over)
        return DataGenConfig(**base)

    # ---------- early termination ----------
    def test_early_terminate_shortens_arrays(self):
        """
        Drive straight at the nearest obstacle so contact is guaranteed
        well before the 80-frame cap; check that the written trajectory
        ends at the contact frame with all arrays the same length.
        """
        from create_dataset_module.generator import DataGenerator
        from data_contracts import Trajectory

        with tempfile.TemporaryDirectory() as tmp:
            # Adversarial: robot drives at ~0.3-0.5 m/s once friction and
            # wheel spin-up are taken into account. Obstacles in the
            # default pybullet_navigation map are at |r|>=3 m, so the
            # first collision needs roughly 120-300 frames. Cap high so
            # the test remains deterministic across machines.
            T_cap = 500
            gen = DataGenerator(
                self._cfg(tmp, frames_per_rollout=T_cap), out_dir=tmp
            )
            written = gen.run()
            self.assertEqual(written, 1)

            row = json.loads((Path(tmp) / "index.jsonl").read_text().splitlines()[0])
            self.assertLess(row["T"], T_cap)
            self.assertTrue(row["terminated_on_contact"])
            self.assertEqual(row["policy"], "adversarial")

            traj = Trajectory.from_npz(Path(tmp) / row["path"])
            T = traj.T
            self.assertEqual(traj.depth.shape[0], T)
            self.assertEqual(traj.cam_extr_R.shape[0], T)
            self.assertEqual(traj.cam_extr_t.shape[0], T)
            self.assertEqual(traj.ego_state.shape[0], T)
            self.assertEqual(traj.ego_vel.shape[0], T)
            self.assertEqual(traj.action.shape[0], T)
            self.assertEqual(traj.contact_flag.shape[0], T)
            self.assertEqual(traj.risk_1s.shape[0], T)
            # Last frame should be the contact frame when grace == 0.
            self.assertTrue(bool(traj.contact_flag[-1]))
            # risk_1s should be 1 at the last frame (contact is within the window).
            self.assertEqual(float(traj.risk_1s[-1]), 1.0)

    def test_terminate_disabled_runs_full_length(self):
        """When terminate_on_contact is False, T equals frames_per_rollout."""
        from create_dataset_module.generator import DataGenerator

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp, terminate_on_contact=False, frames_per_rollout=25)
            gen = DataGenerator(cfg, out_dir=tmp)
            gen.run()
            row = json.loads(
                (Path(tmp) / "index.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(row["T"], 25)
            self.assertFalse(row["terminated_on_contact"])

    # ---------- save_rgb flag ----------
    def test_save_rgb_false_stores_placeholder(self):
        from create_dataset_module.generator import DataGenerator
        from data_contracts import Trajectory

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp, save_rgb=False, frames_per_rollout=15,
                            terminate_on_contact=False,
                            policy_random_p=1.0, policy_adversarial_p=0.0)
            DataGenerator(cfg, out_dir=tmp).run()
            row = json.loads(
                (Path(tmp) / "index.jsonl").read_text().splitlines()[0]
            )
            traj = Trajectory.from_npz(Path(tmp) / row["path"])
            self.assertEqual(traj.rgb.size, 0)
            self.assertEqual(traj.rgb.dtype, np.uint8)
            # Depth is still the real thing.
            self.assertEqual(traj.depth.shape, (15, 120, 160))

    def test_save_rgb_true_stores_frames(self):
        from create_dataset_module.generator import DataGenerator
        from data_contracts import Trajectory

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp, save_rgb=True, frames_per_rollout=10,
                            terminate_on_contact=False,
                            policy_random_p=1.0, policy_adversarial_p=0.0)
            DataGenerator(cfg, out_dir=tmp).run()
            row = json.loads(
                (Path(tmp) / "index.jsonl").read_text().splitlines()[0]
            )
            traj = Trajectory.from_npz(Path(tmp) / row["path"])
            self.assertEqual(traj.rgb.shape, (10, 120, 160, 3))
            self.assertEqual(traj.rgb.dtype, np.uint8)

    # ---------- last_stats surface ----------
    def test_last_stats_populated(self):
        from create_dataset_module.generator import DataGenerator

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp, n_scenes=1, rollouts_per_scene=2,
                            frames_per_rollout=15,
                            policy_random_p=0.5, policy_scripted_p=0.0,
                            policy_adversarial_p=0.5,
                            terminate_on_contact=False)
            gen = DataGenerator(cfg, out_dir=tmp)
            gen.run()
            st = gen.last_stats
            self.assertEqual(st["written"], 2)
            self.assertEqual(st["total_frames"], 30)
            self.assertIn("positive_ratio_1s", st)
            self.assertIn("policy_counts", st)
            total_pc = sum(st["policy_counts"].values())
            self.assertEqual(total_pc, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
