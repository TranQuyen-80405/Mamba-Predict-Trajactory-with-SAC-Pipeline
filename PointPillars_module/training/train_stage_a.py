"""
Stage A full trainer (A1/A2 schedules, single run).

Use the Python trainers instead of notebooks:

* ``training/train_stage_a_mamba.py``, … — one backbone each.
* ``training/train_stage_a_compare.py`` — multi-backbone compare (same data split).
* Shared loop + TensorBoard + JSON exports: ``training/stage_a_single_run.py``.
* Package root ``train_stage_a_compare.py`` re-exports the compare module for notebook imports.

This module is reserved for a future unified A1/A2 loop. See
``docs/strategy_train_stage_A.md`` and ``docs/skill_avoid_gradient_boom.md``.
"""

from __future__ import annotations

__all__: list[str] = []
