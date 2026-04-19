"""
Unit tests for losses.py.

Pure CPU. Run:
    cd PointPillars_module
    python -m unittest tests.test_losses -v
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from losses import focal_bce, oversample_positive_indices  # noqa: E402


class TestFocalBCE(unittest.TestCase):
    def test_gamma_zero_matches_vanilla_bce(self):
        torch.manual_seed(0)
        logits = torch.randn(8, 3)
        targets = (torch.rand(8, 3) > 0.5).float()

        ours = focal_bce(logits, targets, gamma=0.0, weight=None, reduction="none")
        ref = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        self.assertTrue(torch.allclose(ours, ref, atol=1e-6))

    def test_focal_factor_matches_closed_form(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 3)
        targets = (torch.rand(4, 3) > 0.5).float()
        gamma = 2.0

        ours = focal_bce(logits, targets, gamma=gamma, weight=None, reduction="none")

        prob = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, prob, 1.0 - prob)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        ref = ((1.0 - pt) ** gamma) * bce

        self.assertTrue(torch.allclose(ours, ref, atol=1e-6))

    def test_weight_scales_columns(self):
        logits = torch.zeros(2, 3)
        targets = torch.ones(2, 3)
        w = (1.0, 0.8, 0.5)

        loss_w = focal_bce(logits, targets, gamma=0.0, weight=w, reduction="none")
        loss_u = focal_bce(logits, targets, gamma=0.0, weight=None, reduction="none")

        expected = loss_u * torch.tensor(w)
        self.assertTrue(torch.allclose(loss_w, expected, atol=1e-6))

    def test_easy_examples_downweighted_when_gamma_positive(self):
        # Confident-correct example: logit large, target=1.
        # Its focal loss must be strictly smaller than the vanilla BCE.
        logits = torch.tensor([[5.0]])
        targets = torch.tensor([[1.0]])

        bce = F.binary_cross_entropy_with_logits(logits, targets).item()
        focal = focal_bce(
            logits, targets, gamma=2.0, weight=None, reduction="mean"
        ).item()
        self.assertLess(focal, bce)

    def test_reduction_modes(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 3)
        targets = (torch.rand(4, 3) > 0.5).float()
        none_val = focal_bce(logits, targets, reduction="none")
        self.assertEqual(none_val.shape, (4, 3))
        m_val = focal_bce(logits, targets, reduction="mean")
        s_val = focal_bce(logits, targets, reduction="sum")
        self.assertTrue(torch.allclose(m_val, none_val.mean()))
        self.assertTrue(torch.allclose(s_val, none_val.sum()))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            focal_bce(torch.zeros(2, 3), torch.zeros(2, 4))

    def test_weight_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            focal_bce(
                torch.zeros(2, 3), torch.zeros(2, 3), weight=(1.0, 2.0)
            )


class TestOversamplePositiveIndices(unittest.TestCase):
    def test_positives_duplicated(self):
        risk = np.array([0, 0, 1, 0, 1], dtype=np.float32)
        idx = oversample_positive_indices(risk, factor=10)
        # 3 negatives + 2 positives * 10 = 23 indices
        self.assertEqual(len(idx), 23)
        counts = {i: idx.count(i) for i in range(5)}
        self.assertEqual(counts[0], 1)
        self.assertEqual(counts[1], 1)
        self.assertEqual(counts[2], 10)
        self.assertEqual(counts[3], 1)
        self.assertEqual(counts[4], 10)

    def test_factor_one_preserves_all_indices(self):
        risk = np.array([0, 1, 1, 0], dtype=np.float32)
        idx = oversample_positive_indices(risk, factor=1)
        self.assertEqual(sorted(idx), [0, 1, 2, 3])

    def test_accepts_torch_tensor(self):
        risk = torch.tensor([1.0, 0.0, 1.0])
        idx = oversample_positive_indices(risk, factor=3)
        # 1 negative + 2 positives * 3 = 7
        self.assertEqual(len(idx), 7)

    def test_bad_factor_raises(self):
        with self.assertRaises(ValueError):
            oversample_positive_indices(np.zeros(3), factor=0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
