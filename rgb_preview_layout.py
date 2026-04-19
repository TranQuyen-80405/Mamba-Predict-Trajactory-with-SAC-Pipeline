"""Shim — use ``tools.rgb_preview_layout`` (kept at root for old notebook imports)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.rgb_preview_layout import *  # noqa: F403
