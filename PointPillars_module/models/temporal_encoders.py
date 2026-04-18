"""
Additional sequence encoders for Stage A ablations (same bulk forward contract
as MambaTemporal: (B, L, D) -> (B, L, D)).

LSTM: torch.nn.LSTM, supports streaming step() for Stage B.
Transformer: torch.nn.TransformerEncoder with causal mask; step() is not supported.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class LSTMTemporal(nn.Module):
    """Stacked LSTM, batch_first. Hidden state is (h, c) tuple."""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        drop = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=drop,
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 3 or seq.shape[-1] != self.d_model:
            raise ValueError(
                f"LSTMTemporal.forward expects (B, L, D={self.d_model}); "
                f"got {tuple(seq.shape)}"
            )
        out, _ = self.lstm(seq)
        return out

    def step(
        self,
        tok_t: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if tok_t.ndim != 2 or tok_t.shape[-1] != self.d_model:
            raise ValueError(
                f"step() expects (B, D={self.d_model}); got {tuple(tok_t.shape)}"
            )
        x = tok_t.unsqueeze(1)
        out, new_hidden = self.lstm(x, hidden)
        return out.squeeze(1), new_hidden


class TransformerEncoderTemporal(nn.Module):
    """
    Causal Transformer encoder (library: torch.nn only).
    Use LayerNorm-first blocks for stability on short sequences.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 3 or seq.shape[-1] != self.d_model:
            raise ValueError(
                f"TransformerEncoderTemporal.forward expects "
                f"(B, L, D={self.d_model}); got {tuple(seq.shape)}"
            )
        b, l, _ = seq.shape
        causal = torch.triu(
            torch.ones(l, l, device=seq.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.encoder(seq, mask=causal, is_causal=False)

    def step(
        self,
        tok_t: torch.Tensor,
        hidden: Optional[object] = None,
    ) -> Tuple[torch.Tensor, object]:
        raise RuntimeError(
            "TransformerEncoderTemporal does not implement streaming step(); "
            "use mamba, gru, or lstm for Stage B."
        )
