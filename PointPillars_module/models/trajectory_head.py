"""MLP head: temporal embedding -> short future planar trajectory (x, y, yaw)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryHead(nn.Module):
    """
    Predicts H future ego poses (world frame) aligned with ``ego_state[:, [0,1,5]]``.

    Output is reshaped to (B, H, 3); loss is typically smooth L1 or MSE vs ground truth.
    """

    def __init__(
        self,
        in_dim: int = 256,
        horizon: int = 10,
        hidden1: int = 128,
        hidden2: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.horizon = int(horizon)
        out_dim = self.horizon * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, out_dim),
        )

    def forward(self, h_T: torch.Tensor) -> torch.Tensor:
        if h_T.ndim != 2 or h_T.shape[-1] != self.in_dim:
            raise ValueError(
                f"TrajectoryHead expects (B, {self.in_dim}); got {tuple(h_T.shape)}"
            )
        flat = self.net(h_T)
        return flat.view(-1, self.horizon, 3)
