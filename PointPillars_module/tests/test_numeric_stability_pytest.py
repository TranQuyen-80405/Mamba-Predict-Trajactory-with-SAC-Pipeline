"""Focal loss clamp, SAC actor log_std bounds, gradient helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from losses import focal_bce  # noqa: E402
from train_stage_b_sac import ActorMLP  # noqa: E402
from utils.gradient_health import grad_norm_l2  # noqa: E402


def test_focal_bce_extreme_logits_finite() -> None:
    logits = torch.tensor([[100.0, -100.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0]])
    loss = focal_bce(logits, targets, gamma=2.0)
    assert torch.isfinite(loss)


def test_actor_logstd_clamped_and_sample_finite() -> None:
    a = ActorMLP(state_dim=8, act_dim=2, hidden=32)
    s = torch.randn(4, 8)
    mu, logstd = a(s)
    assert logstd.min() >= ActorMLP.LOGSTD_MIN - 1e-6
    assert logstd.max() <= ActorMLP.LOGSTD_MAX + 1e-6
    act, logp = a.sample(s)
    assert torch.isfinite(act).all()
    assert torch.isfinite(logp).all()


def test_grad_norm_helper() -> None:
    w = torch.nn.Parameter(torch.ones(3, requires_grad=True))
    loss = (w ** 2).sum()
    loss.backward()
    n = float(grad_norm_l2([w]))
    assert abs(n - (3.0 ** 0.5 * 2)) < 1e-4  # grad is 2*w = 2
