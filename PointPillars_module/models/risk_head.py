"""
RiskHead: collision probability predictor for three horizons.

Exact spec from docs/strategy_full_pipeline.md § 4.4:

    Input:  h_T  (B, 256)

    Linear(256, 128) + ReLU + Dropout(0.1)
    Linear(128,  64) + ReLU + Dropout(0.1)
    Linear( 64,   3)                          # no sigmoid - BCEWithLogitsLoss

    Output: logits (B, 3)   -> sigmoid(logits) = [p_05s, p_1s, p_2s]

Parameters ~41 k.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RiskHead(nn.Module):
    """Tiny MLP that maps a temporal hidden state to 3 horizon logits."""

    def __init__(
        self,
        in_dim: int = 256,
        hidden1: int = 128,
        hidden2: int = 64,
        num_horizons: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_horizons = num_horizons
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_horizons),
        )

    def forward(self, h_T: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_T: (B, D) final hidden state from MambaTemporal.

        Returns:
            logits (B, num_horizons). Apply ``torch.sigmoid`` only for
            inference — the training loss uses BCEWithLogits.
        """
        if h_T.ndim != 2 or h_T.shape[-1] != self.in_dim:
            raise ValueError(
                f"RiskHead expects (B, {self.in_dim}); got {tuple(h_T.shape)}"
            )
        return self.net(h_T)
