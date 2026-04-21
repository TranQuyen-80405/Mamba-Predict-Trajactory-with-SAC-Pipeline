"""
Factory for temporal encoders used inside FullPipeline (Stage A comparisons).
"""

from __future__ import annotations

from typing import Literal, Optional, Union

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
    temporal_dropout: float = 0.1,
    lstm_dropout: Optional[float] = None,
    transformer_nhead: int = 8,
    transformer_dim_ff: int = 512,
    transformer_dropout: Optional[float] = None,
    mamba_backend: str = "auto",
) -> nn.Module:
    """
    Build a temporal module with forward(seq) -> (B, L, D).

    * mamba / gru : MambaTemporal (mamba-ssm or GRU per backend resolution).
    * lstm         : LSTMTemporal (PyTorch nn.LSTM).
    * transformer  : TransformerEncoderTemporal (PyTorch nn.TransformerEncoder).

    ``temporal_dropout`` regularizes mamba, GRU, and LSTM (and lower-bounds
    transformer dropout at 0.1 — ``nn.TransformerEncoderLayer`` uses one
    ``dropout`` for both attention and feed-forward sublayers).
    """
    td = float(temporal_dropout)
    lstm_d = float(lstm_dropout) if lstm_dropout is not None else td
    trans_d = float(transformer_dropout) if transformer_dropout is not None else max(td, 0.1)
    k = str(kind).lower().strip()
    if k == "mamba":
        be = "auto" if mamba_backend == "auto" else "mamba"
        return MambaTemporal(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            n_blocks=n_blocks,
            backend=be,  # type: ignore[arg-type]
            dropout=td,
        )
    if k == "gru":
        return MambaTemporal(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            n_blocks=n_blocks,
            backend="gru",
            dropout=td,
        )
    if k == "lstm":
        return LSTMTemporal(d_model=d_model, n_layers=n_blocks, dropout=lstm_d)
    if k == "transformer":
        return TransformerEncoderTemporal(
            d_model=d_model,
            nhead=transformer_nhead,
            num_layers=n_blocks,
            dim_feedforward=transformer_dim_ff,
            dropout=trans_d,
        )
    raise ValueError(
        f"unknown temporal kind {kind!r}; "
        f"expected one of: mamba, gru, lstm, transformer"
    )
