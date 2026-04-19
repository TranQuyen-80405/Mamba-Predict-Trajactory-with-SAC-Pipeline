"""
Stage A — **Transformer** encoder over the token sequence (``TransformerEncoderTemporal``).

Example (from repo root)::

    python PointPillars_module/training/train_stage_a_transformer.py --data_root data/stage_a_experiment \\
        --ckpt PointPillars_module/pretrained/epoch_160.pth
"""

from __future__ import annotations

import sys
from pathlib import Path

_pp = Path(__file__).resolve().parent.parent
if str(_pp) not in sys.path:
    sys.path.insert(0, str(_pp))

from training.stage_a_single_run import main_cli

if __name__ == "__main__":
    main_cli("transformer")
