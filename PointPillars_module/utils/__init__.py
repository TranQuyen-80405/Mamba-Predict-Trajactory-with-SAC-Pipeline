"""Training utilities (gradient / numerics helpers)."""

from .gradient_health import (
    EPS,
    grad_norm_l2,
    max_grad_value,
    soft_update_polyak,
)

__all__ = [
    "EPS",
    "grad_norm_l2",
    "max_grad_value",
    "soft_update_polyak",
]
