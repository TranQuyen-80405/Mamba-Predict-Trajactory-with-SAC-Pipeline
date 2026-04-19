"""
Sampling + grid shape for RGB trajectory montages (numpy only — no matplotlib).

Dùng chung ``tools/rgb_preview_to_png.py`` và cell notebook khi kernel có matplotlib.
"""

from __future__ import annotations


def sample_frame_indices(t: int, max_frames: int = 30) -> list[int]:
    """Chọn tối đa `max_frames` chỉ số frame, trải đều từ 0 .. t-1 (quỹ đạo theo thời gian)."""
    if t <= 0:
        return []
    n = min(max_frames, t)
    if n == 1:
        return [0]
    return [int(round(i * (t - 1) / (n - 1))) for i in range(n)]


def grid_rows_cols(n: int, max_cols: int = 6) -> tuple[int, int]:
    """Lưới subplot: mặc định tối đa 6 cột (5 hàng × 6 cột = 30 ô)."""
    if n <= 0:
        return 1, 1
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    return rows, cols
