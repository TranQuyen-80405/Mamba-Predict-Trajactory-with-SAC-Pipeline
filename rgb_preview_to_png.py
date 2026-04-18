"""
Ghi loạt frame RGB từ Trajectory .npz ra PNG (matplotlib Agg — dùng cho notebook
khi kernel không có matplotlib).

Mặc định tối đa 30 frame, lưới 5×6 — thể hiện quỹ đạo theo thời gian.

Usage:
    .venv\\Scripts\\python.exe rgb_preview_to_png.py data/stage_a_rgb_spotcheck/s0000_r00.npz
    .venv\\Scripts\\python.exe rgb_preview_to_png.py path/to/file.npz path/out.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from rgb_preview_layout import grid_rows_cols, sample_frame_indices

_mpl = os.environ.get("MPLBACKEND", "")
if "inline" in _mpl.lower() or _mpl.startswith("module://"):
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# Tối đa số frame trên một ảnh (montage)
MAX_FRAMES = 30


def render_rgb_montage(rgb: np.ndarray, out_path: Path, max_frames: int = MAX_FRAMES) -> None:
    if rgb.size == 0 or rgb.ndim < 3:
        raise ValueError("RGB array empty or invalid")
    t = rgb.shape[0]
    idxs = sample_frame_indices(t, max_frames)
    n = len(idxs)
    rows, cols = grid_rows_cols(n)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.15, rows * 2.05))
    axs = np.atleast_1d(axs).ravel()
    for k in range(rows * cols):
        ax = axs[k]
        if k < n:
            ax.imshow(rgb[idxs[k]])
            ax.set_title(f"t={idxs[k]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("RGB montage (time →)", fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    npz_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else npz_path.with_name("rgb_preview.png")

    z = np.load(npz_path)
    rgb = z["rgb"]
    render_rgb_montage(rgb, out_path)
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
