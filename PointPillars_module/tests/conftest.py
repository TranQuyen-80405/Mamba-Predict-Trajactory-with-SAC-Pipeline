"""Shared pytest paths and markers for PointPillars_module."""

from __future__ import annotations

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from pretrained_ckpt_resolve import resolve_pointpillars_ckpt  # noqa: E402

_CKPT_RESOLVED = resolve_pointpillars_ckpt(_PKG_ROOT)
CKPT_PATH = _CKPT_RESOLVED or os.path.join(_PKG_ROOT, "pretrained", "epoch_160_raw.pth")
HAS_CKPT = _CKPT_RESOLVED is not None
HAS_CUDA = torch.cuda.is_available()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "needs_model: needs built voxel op + checkpoint (CPU or CUDA)",
    )
    config.addinivalue_line(
        "markers",
        "needs_cuda: needs torch.cuda + checkpoint",
    )


@pytest.fixture(scope="session")
def ckpt_path() -> str:
    if not HAS_CKPT or _CKPT_RESOLVED is None:
        pytest.skip("missing pretrained/epoch_160_raw.pth or epoch_160.pth")
    return _CKPT_RESOLVED


@pytest.fixture(scope="session")
def has_checkpoint() -> bool:
    return HAS_CKPT


@pytest.fixture(scope="session")
def has_cuda() -> bool:
    return HAS_CUDA
