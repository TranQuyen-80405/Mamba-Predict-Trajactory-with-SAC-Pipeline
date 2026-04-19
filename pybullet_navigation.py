"""
Compatibility entry point: ``import pybullet_navigation`` from the repo root.

The real implementation (RL_Env, MAP_SIZE, obstacle spawn, collision checks) lives
in ``env/pybullet_navigation.py``. A duplicate copy used to exist here and went
stale, which broke dataset generation (wrong obstacle layout / collision logic).

Always edit ``env/pybullet_navigation.py``.
"""

from __future__ import annotations

from env.pybullet_navigation import *  # noqa: F403
