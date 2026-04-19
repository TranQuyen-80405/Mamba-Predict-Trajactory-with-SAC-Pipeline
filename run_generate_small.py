"""
Wrapper — implementation: ``scripts/datagen/run_generate_small.py``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_IMPL = Path(__file__).resolve().parent / "scripts" / "datagen" / "run_generate_small.py"


def main() -> None:
    sys.argv[0] = str(_IMPL)
    runpy.run_path(str(_IMPL), run_name="__main__")


if __name__ == "__main__":
    main()
