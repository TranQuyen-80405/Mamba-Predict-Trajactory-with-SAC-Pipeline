"""
Unit tests for the v3.3 domain-randomization helpers in generator.py.
These are PyBullet-free; they only exercise the numpy math.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_PP_PKG = os.path.join(_ROOT, "PointPillars_module")
for _p in (_ROOT, _PP_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from create_dataset_module.generator import (
    _apply_camera_jitter,
    _apply_depth_noise,
    _apply_pixel_dropout,
)


class TestDepthNoise(unittest.TestCase):
    def test_zero_std_is_identity(self):
        depth = np.full((4, 5), 2.5, dtype=np.float32)
        rng = np.random.default_rng(0)
        out = _apply_depth_noise(depth, 0.0, rng, far=8.0)
        np.testing.assert_array_equal(out, depth)

    def test_nonzero_std_perturbs_and_stays_in_range(self):
        depth = np.full((32, 32), 3.0, dtype=np.float32)
        rng = np.random.default_rng(0)
        out = _apply_depth_noise(depth, 0.05, rng, far=8.0)
        self.assertFalse(np.allclose(out, depth))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 8.0)
        self.assertAlmostEqual(float(out.mean()), 3.0, delta=0.05)


class TestPixelDropout(unittest.TestCase):
    def test_zero_prob_is_identity(self):
        depth = np.full((4, 5), 2.0, dtype=np.float32)
        rng = np.random.default_rng(0)
        out = _apply_pixel_dropout(depth, 0.0, rng)
        np.testing.assert_array_equal(out, depth)

    def test_nonzero_prob_creates_zeros(self):
        depth = np.full((200, 200), 2.0, dtype=np.float32)
        rng = np.random.default_rng(0)
        out = _apply_pixel_dropout(depth, 0.1, rng)
        frac_zero = float((out == 0.0).mean())
        self.assertAlmostEqual(frac_zero, 0.1, delta=0.02)
        non_zero = out[out != 0.0]
        self.assertTrue(np.allclose(non_zero, 2.0))


class TestCameraJitter(unittest.TestCase):
    def test_zero_jitter_is_identity(self):
        R = np.eye(3, dtype=np.float32)
        rng = np.random.default_rng(0)
        out = _apply_camera_jitter(R, 0.0, rng)
        np.testing.assert_array_equal(out, R)

    def test_jitter_preserves_orthogonality(self):
        R = np.eye(3, dtype=np.float32)
        rng = np.random.default_rng(0)
        for _ in range(16):
            out = _apply_camera_jitter(R, 1.0, rng)
            err = float(np.abs(out @ out.T - np.eye(3)).max())
            self.assertLess(err, 1e-4)

    def test_jitter_magnitude_is_small(self):
        R = np.eye(3, dtype=np.float32)
        rng = np.random.default_rng(0)
        deltas = []
        for _ in range(64):
            out = _apply_camera_jitter(R, 1.0, rng)
            deltas.append(float(np.abs(out - R).max()))
        # Mean max-element perturbation should be well below one radian.
        self.assertLess(float(np.mean(deltas)), 0.1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
