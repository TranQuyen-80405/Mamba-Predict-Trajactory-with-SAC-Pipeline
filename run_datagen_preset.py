"""
Wrapper — implementation: ``scripts/datagen/run_datagen_preset.py``.

Run from repo root (same as before)::
  python run_datagen_preset.py experiment
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_IMPL = Path(__file__).resolve().parent / "scripts" / "datagen" / "run_datagen_preset.py"


def main() -> None:
    sys.argv[0] = str(_IMPL)
    runpy.run_path(str(_IMPL), run_name="__main__")


if __name__ == "__main__":
    main()
