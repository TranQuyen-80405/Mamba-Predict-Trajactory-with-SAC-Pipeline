"""
Wrapper — implementation: ``scripts/check_dataset_stats.py``.

Run from repo root:
  python run_dataset_stats.py --data_root data/stage_a_experiment
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_IMPL = Path(__file__).resolve().parent / "scripts" / "check_dataset_stats.py"


def main() -> None:
    sys.argv[0] = str(_IMPL)
    runpy.run_path(str(_IMPL), run_name="__main__")


if __name__ == "__main__":
    main()
