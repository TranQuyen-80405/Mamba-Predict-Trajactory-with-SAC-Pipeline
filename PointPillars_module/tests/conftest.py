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

CKPT_PATH = os.path.join(_PKG_ROOT, "pretrained", "epoch_160.pth")
HAS_CKPT = os.path.exists(CKPT_PATH)
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
    return CKPT_PATH


@pytest.fixture(scope="session")
def has_checkpoint() -> bool:
    return HAS_CKPT


@pytest.fixture(scope="session")
def has_cuda() -> bool:
    return HAS_CUDA
