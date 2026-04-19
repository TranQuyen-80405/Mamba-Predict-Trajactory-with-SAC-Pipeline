# =====================================================================
# data_contracts.py
# ---------------------------------------------------------------------
# Single source of truth for all dataclasses exchanged between the
# perception stream (PointPillars -> SpatialReducer -> Mamba -> RiskHead)
# and the Stage A / Stage B training loops.
#
# Every identifier here is spelled exactly as in
# docs/strategy_full_pipeline.md § 3 (data contracts) and § 5.1 / § 6.6
# (configs). Do not rename fields without updating those sections first
# (see § 14 maintenance rule).
# =====================================================================

from __future__ import annotations

import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Tuple, Union

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PointPillars_module.types import (
    DataGenConfig,
    EnvConfig,
    ProprioState,
    RiskBatch,
    RiskSample,
    Trajectory,
    Transition,
)


# =====================================================================
# Convenience: resolve proprio dimension from a flag set
# =====================================================================

def proprio_dim_from_cfg(
    cfg: EnvConfig,
    dof: int = 0,
) -> int:
    """
    Compute d_s from an EnvConfig + optional joint DoF count. Used by the
    Actor/Critic network builders in rl/networks.py (Phase-4 work).
    """
    d = 3 + 3 + 3 + 1
    if cfg.include_last_action:
        d += cfg.action_dim
    if cfg.include_joint_state:
        if dof <= 0:
            raise ValueError(
                "include_joint_state=True requires a positive dof count."
            )
        d += 2 * dof
    return d


# =====================================================================
# Introspection helpers
# =====================================================================

def trajectory_field_names() -> Tuple[str, ...]:
    """All field names on Trajectory, in declaration order."""
    return tuple(f.name for f in fields(Trajectory))


def dump_index_row(path: Union[str, Path], row: dict) -> None:
    """Append a single JSON line to an ``index.jsonl`` catalog."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


__all__ = [
    "Trajectory",
    "RiskSample",
    "RiskBatch",
    "ProprioState",
    "Transition",
    "DataGenConfig",
    "EnvConfig",
    "proprio_dim_from_cfg",
    "trajectory_field_names",
    "dump_index_row",
]
