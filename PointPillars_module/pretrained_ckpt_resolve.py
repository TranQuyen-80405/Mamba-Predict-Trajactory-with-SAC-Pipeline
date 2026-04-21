"""Resolve default PointPillars neck checkpoint path (prefer raw export)."""

from __future__ import annotations

import os


def resolve_pointpillars_ckpt(pkg_root: str) -> str | None:
    """
    Prefer ``pretrained/epoch_160_raw.pth``, then ``epoch_160.pth``.

    The raw export is the current training default; older trees may only ship
    ``epoch_160.pth``.
    """
    sub = os.path.join(pkg_root, "pretrained")
    for name in ("epoch_160_raw.pth", "epoch_160.pth"):
        p = os.path.join(sub, name)
        if os.path.isfile(p):
            return p
    return None
