"""
Factory for temporal encoders used inside FullPipeline (Stage A comparisons).
"""

from __future__ import annotations

from typing import Literal, Union

import torch.nn as nn

from .mamba_temporal import MambaTemporal
from .temporal_encoders import LSTMTemporal, TransformerEncoderTemporal

TemporalKind = Literal["mamba", "gru", "lstm", "transformer"]


def build_temporal(
    kind: Union[str, TemporalKind],
    *,
    d_model: int = 256,
    n_blocks: int = 2,
    d_state: int = 16,
    expand: int = 2,
    lstm_dropout: float = 0.0,
    transformer_nhead: int = 8,
    transformer_dim_ff: int = 512,
    transformer_dropout: float = 0.1,
    mamba_backend: str = "auto",
) -> nn.Module:
    """
    Build a temporal module with forward(seq) -> (B, L, D).

    * mamba / gru : MambaTemporal (mamba-ssm or GRU per backend resolution).
    * lstm         : LSTMTemporal (PyTorch nn.LSTM).
    * transformer  : TransformerEncoderTemporal (PyTorch nn.TransformerEncoder).
    """
    k = str(kind).lower().strip()
    if k == "mamba":
        be = "auto" if mamba_backend == "auto" else "mamba"
        return MambaTemporal(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            n_blocks=n_blocks,
            backend=be,  # type: ignore[arg-type]
        )
    if k == "gru":
        return MambaTemporal(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            n_blocks=n_blocks,
            backend="gru",
        )
    if k == "lstm":
        return LSTMTemporal(d_model=d_model, n_layers=n_blocks, dropout=lstm_dropout)
    if k == "transformer":
        return TransformerEncoderTemporal(
            d_model=d_model,
            nhead=transformer_nhead,
            num_layers=n_blocks,
            dim_feedforward=transformer_dim_ff,
            dropout=transformer_dropout,
        )
    raise ValueError(
        f"unknown temporal kind {kind!r}; "
        f"expected one of: mamba, gru, lstm, transformer"
    )
