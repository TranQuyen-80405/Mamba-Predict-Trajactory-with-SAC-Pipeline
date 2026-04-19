"""
Inspect MambaTemporal resolution (mamba-ssm vs GRU) and device context.

Used at Stage A startup to confirm whether CUDA-backed mamba-ssm is active
or the repo fell back to nn.GRU (see ``models/mamba_temporal.py``).
"""

from __future__ import annotations

from typing import Any

import torch


def log_mamba_temporal_runtime(mamba_module: Any, *, device: torch.device) -> None:
    """Print one-line summary: backend string, device, CUDA availability."""
    cls = type(mamba_module).__name__
    if cls != "MambaTemporal":
        print(
            f"[mamba_runtime] temporal module is {cls!r} (not MambaTemporal); "
            f"skipping backend probe.",
            flush=True,
        )
        return

    be = getattr(mamba_module, "backend", "?")
    cuda_ok = torch.cuda.is_available()
    print(
        f"[mamba_runtime] MambaTemporal backend={be!r} train_device={device} "
        f"torch.cuda.is_available()={cuda_ok}",
        flush=True,
    )
    if be == "mamba" and device.type != "cuda":
        print(
            "[mamba_runtime] Warning: MambaTemporal backend is mamba-ssm but "
            "model/device is not CUDA — performance may be poor or unsupported.",
            flush=True,
        )
    if be == "gru" and cuda_ok and device.type == "cuda":
        print(
            "[mamba_runtime] Using GRU fallback inside MambaTemporal (mamba-ssm "
            "unavailable, import failed, or backend='gru').",
            flush=True,
        )


__all__ = ["log_mamba_temporal_runtime"]
