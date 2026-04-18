"""
Unit tests for PointPillars_module/models/*.

Tests are split by GPU-ness:
  - SpatialReducer, RiskHead, MambaTemporal(GRU backend): pure torch, CPU-ok.
  - MambaTemporal(mamba backend): skip unless mamba_ssm importable + CUDA.
  - FullPipeline: skip unless CUDA + checkpoint present (voxel_op extension).

Run:
    cd PointPillars_module
    python -m unittest tests.test_models -v
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from models.mamba_temporal import MambaTemporal, _try_import_mamba  # noqa: E402
from models.risk_head import RiskHead  # noqa: E402
from models.spatial_reducer import SpatialReducer  # noqa: E402

_CKPT = os.path.join(_PKG_ROOT, "pretrained", "epoch_160.pth")
_HAS_CUDA = torch.cuda.is_available()
_HAS_CKPT = os.path.exists(_CKPT)
_HAS_MAMBA = _try_import_mamba() is not None
_CAN_RUN_MODEL = _HAS_CUDA and _HAS_CKPT


# =====================================================================
# SpatialReducer
# =====================================================================

class TestSpatialReducer(unittest.TestCase):
    def test_forward_shape_full_spec(self):
        reducer = SpatialReducer().eval()
        # Use small H/W here; AdaptiveAvgPool handles any input size.
        x = torch.zeros(2, 384, 24, 24)
        with torch.no_grad():
            out = reducer(x)
        self.assertEqual(out.shape, (2, 16, 256))

    def test_forward_shape_with_kitti_size(self):
        reducer = SpatialReducer().eval()
        # Match the actual BEV output size produced by the neck.
        # CPU-only: keep batch=1 and use a downsized spatial map to stay fast.
        x = torch.zeros(1, 384, 32, 28)
        with torch.no_grad():
            out = reducer(x)
        self.assertEqual(out.shape, (1, 16, 256))

    def test_parameter_count_within_budget(self):
        reducer = SpatialReducer()
        n_params = sum(p.numel() for p in reducer.parameters())
        # Three stride-2 3x3 convs 384->256, 256->256, 256->256 + BNs
        # ≈ 0.88M + 0.59M + 0.59M = ~2.06 M. Allow a small band around 2.1 M.
        self.assertLess(n_params, 3_000_000)
        self.assertGreater(n_params, 1_500_000)

    def test_rejects_bad_input_rank(self):
        reducer = SpatialReducer()
        with self.assertRaises(ValueError):
            reducer(torch.zeros(2, 384, 24))


# =====================================================================
# MambaTemporal - GRU backend (always available)
# =====================================================================

class TestMambaTemporalGRU(unittest.TestCase):
    def test_forward_shape_and_final_hidden(self):
        model = MambaTemporal(d_model=256, n_blocks=2, backend="gru").eval()
        seq = torch.randn(2, 16, 256)
        with torch.no_grad():
            out = model(seq)
        self.assertEqual(out.shape, (2, 16, 256))
        h_T = out[:, -1, :]
        self.assertEqual(h_T.shape, (2, 256))

    def test_step_runs_and_returns_correct_shape(self):
        model = MambaTemporal(d_model=256, n_blocks=2, backend="gru").eval()
        hidden = None
        tok = torch.randn(2, 256)
        with torch.no_grad():
            out_t, hidden = model.step(tok, hidden)
        self.assertEqual(out_t.shape, (2, 256))
        # For GRU backend, the hidden state is a tensor of shape (L, B, D)
        self.assertEqual(hidden.shape, (2, 2, 256))

    def test_step_matches_forward_final(self):
        # Feeding sequence one token at a time via step() must match the
        # last column of a bulk forward (within numerical tolerance).
        torch.manual_seed(0)
        model = MambaTemporal(d_model=256, n_blocks=2, backend="gru").eval()
        seq = torch.randn(1, 5, 256)

        with torch.no_grad():
            bulk = model(seq)                  # (1, 5, 256)

            hidden = None
            out_step = None
            for t in range(5):
                out_step, hidden = model.step(seq[:, t, :], hidden)

        self.assertTrue(torch.allclose(bulk[:, -1, :], out_step, atol=1e-5))

    def test_rejects_wrong_dim(self):
        model = MambaTemporal(d_model=256, backend="gru").eval()
        with self.assertRaises(ValueError):
            model(torch.randn(2, 4, 128))


# =====================================================================
# MambaTemporal - mamba-ssm backend (needs CUDA + install)
# =====================================================================

@unittest.skipUnless(_HAS_MAMBA and _HAS_CUDA, "mamba-ssm / CUDA not available")
class TestMambaTemporalSSM(unittest.TestCase):
    def test_forward_shape_and_determinism(self):
        torch.manual_seed(0)
        model = MambaTemporal(
            d_model=256, n_blocks=2, backend="mamba"
        ).cuda().eval()
        seq = torch.randn(2, 16, 256, device="cuda")
        with torch.no_grad():
            out1 = model(seq)
            out2 = model(seq)
        self.assertEqual(out1.shape, (2, 16, 256))
        self.assertTrue(torch.allclose(out1, out2))


# =====================================================================
# RiskHead
# =====================================================================

class TestRiskHead(unittest.TestCase):
    def test_forward_shape(self):
        head = RiskHead().eval()
        out = head(torch.randn(4, 256))
        self.assertEqual(out.shape, (4, 3))

    def test_outputs_are_logits_not_probabilities(self):
        # Logits should be allowed to be negative — a sigmoid-wrapped head
        # would clip to (0, 1).
        torch.manual_seed(0)
        head = RiskHead().eval()
        x = torch.randn(256, 256) * 5.0
        with torch.no_grad():
            out = head(x)
        self.assertTrue((out.min().item() < 0.0) or (out.max().item() > 1.0))

    def test_rejects_wrong_dim(self):
        head = RiskHead(in_dim=256)
        with self.assertRaises(ValueError):
            head(torch.randn(4, 128))


# =====================================================================
# FullPipeline - GPU-only (voxel_op needs CUDA + PP ckpt)
# =====================================================================

@unittest.skipUnless(_CAN_RUN_MODEL, "CUDA or PointPillars ckpt missing")
class TestFullPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Lazy imports so this file still loads on CPU-only boxes.
        from module_pointpillar import (
            PointPillarsConfig,
            PointPillarsNeckExtractor,
        )
        from models.full_pipeline import FullPipeline

        cls._FullPipeline = FullPipeline
        cfg = PointPillarsConfig(ckpt_path=_CKPT, device="cuda")
        cls._pp = PointPillarsNeckExtractor(cfg)

    def _make_fake_pts(self, n: int = 200) -> torch.Tensor:
        rng = torch.Generator().manual_seed(0)
        pts = torch.empty(n, 4)
        pts[:, 0] = torch.rand(n, generator=rng) * 60.0        # x in [0, 60]
        pts[:, 1] = (torch.rand(n, generator=rng) - 0.5) * 60  # y
        pts[:, 2] = (torch.rand(n, generator=rng) - 0.5) * 2   # z
        pts[:, 3] = 0.0
        return pts.float()

    def test_forward_returns_b3_logits_and_gradient_flows_to_reducer(self):
        FP = self._FullPipeline
        pipe = FP(self._pp).cuda()

        # Stage A setup: neck trainable, other perception frozen.
        self._pp.freeze_all()
        self._pp.unfreeze_neck()
        for m in (pipe.reducer, pipe.mamba, pipe.head):
            for p in m.parameters():
                p.requires_grad_(True)
            m.train()

        T_CTX = 2
        B = 2
        pts_seq = [
            [self._make_fake_pts(150 + t * 10 + b) for b in range(B)]
            for t in range(T_CTX)
        ]
        logits = pipe(pts_seq)
        self.assertEqual(logits.shape, (B, 3))

        targets = torch.zeros_like(logits)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets
        )
        loss.backward()

        reducer_grad = sum(
            (p.grad.abs().sum().item() if p.grad is not None else 0.0)
            for p in pipe.reducer.parameters()
        )
        head_grad = sum(
            (p.grad.abs().sum().item() if p.grad is not None else 0.0)
            for p in pipe.head.parameters()
        )
        self.assertGreater(reducer_grad, 0.0)
        self.assertGreater(head_grad, 0.0)

    def test_freeze_perception_clears_requires_grad(self):
        FP = self._FullPipeline
        pipe = FP(self._pp).cuda()
        pipe.freeze_perception()
        total_trainable = sum(
            p.numel() for p in pipe.perception_parameters() if p.requires_grad
        )
        self.assertEqual(total_trainable, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
