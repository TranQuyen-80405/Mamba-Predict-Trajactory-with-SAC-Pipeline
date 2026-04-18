"""
Stage A full trainer (A1/A2 schedules, single run).

The working training entrypoint used in notebooks / Colab comparisons today is
``train_stage_a_compare.py`` (temporal ablations + TensorBoard).

This module is reserved for a future unified A1/A2 loop without changing the
compare API. See ``docs/strategy_train_stage_A.md`` and
``docs/skill_avoid_gradient_boom.md``.
"""

from __future__ import annotations

__all__: list[str] = []
