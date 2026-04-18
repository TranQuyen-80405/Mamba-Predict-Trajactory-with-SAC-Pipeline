"""
Unit tests for LSTM / Transformer temporal modules, factory, and FullPipeline
integration with a mocked PointPillars (no CUDA / no checkpoint).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from module_pointpillar import NeckFeatureOutput  # noqa: E402
from models.full_pipeline import FullPipeline  # noqa: E402
from models.temporal_encoders import LSTMTemporal, TransformerEncoderTemporal  # noqa: E402
from models.temporal_factory import build_temporal  # noqa: E402


class _MockPointPillars:
    """Minimal stand-in matching FullPipeline's ``pp`` + ``pp.model`` contract."""

    def __init__(self) -> None:
        self.model = nn.Linear(1, 1)

    def freeze_all(self) -> None:
        pass

    def extract_neck_forward(self, pts_list):
        b = len(pts_list)
        dev = pts_list[0].device
        feat = torch.zeros(
            b, 384, 248, 216, device=dev, dtype=torch.float32, requires_grad=False
        )
        return NeckFeatureOutput(
            feature=feat,
            batch_size=b,
            channels=384,
            height=248,
            width=216,
            device=str(dev),
        )


@pytest.mark.parametrize("kind", ["gru", "lstm", "transformer"])
def test_build_temporal_forward_shape(kind: str) -> None:
    d = 256
    m = build_temporal(kind, d_model=d, n_blocks=2)
    b, l = 3, 160
    x = torch.randn(b, l, d)
    y = m(x)
    assert y.shape == (b, l, d)


def test_lstm_step_matches_forward_last() -> None:
    d = 256
    lstm = LSTMTemporal(d_model=d, n_layers=2)
    b, l = 2, 5
    x = torch.randn(b, l, d)
    out = lstm(x)
    h = None
    last = None
    for t in range(l):
        last, h = lstm.step(x[:, t, :], h)
    torch.testing.assert_close(last, out[:, -1, :], rtol=1e-4, atol=1e-4)


def test_transformer_step_raises() -> None:
    tr = TransformerEncoderTemporal(d_model=64, nhead=4, num_layers=1)
    with pytest.raises(RuntimeError, match="streaming"):
        tr.step(torch.randn(2, 64), None)


def test_full_pipeline_mock_logits_shape() -> None:
    pp = _MockPointPillars()
    temporal = build_temporal("gru", d_model=256, n_blocks=2)
    pipe = FullPipeline(pp, mamba=temporal, token_dim=256)
    b, t_ctx = 2, 10
    pts_seq = [
        [torch.randn(32, 4) for _ in range(b)] for _ in range(t_ctx)
    ]
    logits = pipe(pts_seq)
    assert logits.shape == (b, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mamba_factory_import_if_cuda() -> None:
    pytest.importorskip("mamba_ssm", reason="mamba-ssm not installed")
    m = build_temporal("mamba", d_model=64, n_blocks=1, mamba_backend="mamba")
    x = torch.randn(2, 16, 64, device="cuda")
    y = m(x)
    assert y.shape == x.shape
