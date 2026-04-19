"""
Thin shim over PointPillars_module.data_contracts.DataGenConfig.

Keeping a separate file here means notebook users can do
``from create_dataset_module.config import DataGenConfig`` without first
understanding the PointPillars sibling package. The underlying dataclass
is the single source of truth.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PP_PKG = os.path.join(_ROOT, "PointPillars_module")
if _PP_PKG not in sys.path:
    sys.path.insert(0, _PP_PKG)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PointPillars_module.types import DataGenConfig, EnvConfig  # noqa: E402

__all__ = ["DataGenConfig", "EnvConfig"]
