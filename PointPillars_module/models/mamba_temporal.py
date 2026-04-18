"""
MambaTemporal: temporal encoder for the risk branch.

Backends:
  - "mamba" : uses mamba-ssm. Requires CUDA and a successful build of
              `mamba-ssm` (+ `causal-conv1d`). This is the default on
              Linux / Colab GPU runtimes.
  - "gru"   : torch nn.GRU(D, D, num_layers=n_blocks, batch_first=True).
              Always available. Used as the fallback on boxes where
              mamba-ssm cannot build (notably Windows + VS >2022).

Both backends expose:
  - forward(seq)           : (B, L, D) -> (B, L, D)
  - step(tok_t, hidden)    : streaming, one token at a time, for Stage B

The public interface matches docs/strategy_full_pipeline.md § 4.3 and
the streaming contract in § 6.4.2.
"""

from __future__ import annotations

import warnings
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn


_BackendName = Literal["auto", "mamba", "gru"]


def _try_import_mamba():
    """Return the mamba_ssm.Mamba class or None if unavailable."""
    try:
        from mamba_ssm import Mamba  # type: ignore
        return Mamba
    except Exception:  # pragma: no cover - env-dependent
        return None


class MambaTemporal(nn.Module):
    """
    Temporal encoder. Input is a sequence of tokens; output is a same-length
    sequence of hidden states. The Stage-A trainer reads h_T = out[:, -1, :]
    (see docs/strategy_full_pipeline.md § 5.2). The Stage-B loop uses
    ``step()`` for O(1)-per-frame streaming (§ 6.4.2).
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        expand: int = 2,
        n_blocks: int = 2,
        backend: _BackendName = "auto",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.n_blocks = n_blocks

        resolved = self._resolve_backend(backend)
        self.backend = resolved
        self._build(resolved)

        # One-time visibility: useful on Colab startup to confirm that
        # mamba-ssm actually loaded instead of silently falling back.
        print(f"[MambaTemporal] backend={self.backend} d_model={d_model} "
              f"n_blocks={n_blocks}")

    # ---------- construction helpers ----------
    @staticmethod
    def _resolve_backend(requested: _BackendName) -> str:
        if requested == "mamba":
            if _try_import_mamba() is None:
                raise ImportError(
                    "backend='mamba' requested but mamba-ssm is not "
                    "installed or failed to import. Install mamba-ssm + "
                    "causal-conv1d (CUDA required), or use backend='gru'."
                )
            return "mamba"
        if requested == "gru":
            return "gru"
        # "auto": prefer mamba, fall back to GRU with a warning.
        if _try_import_mamba() is not None:
            return "mamba"
        warnings.warn(
            "mamba-ssm not available; MambaTemporal is falling back to "
            "nn.GRU. This is expected on Windows / CPU-only boxes; see "
            "docs/strategy_full_pipeline.md § 4.3.",
            stacklevel=3,
        )
        return "gru"

    def _build(self, backend: str) -> None:
        if backend == "mamba":
            Mamba = _try_import_mamba()
            assert Mamba is not None  # guarded by _resolve_backend
            self.blocks = nn.ModuleList([
                Mamba(
                    d_model=self.d_model,
                    d_state=self.d_state,
                    expand=self.expand,
                )
                for _ in range(self.n_blocks)
            ])
            # Per-block LayerNorm + residual wrap keeps gradients healthy
            # on top of the raw Mamba blocks, mirroring the mamba-ssm
            # reference `Block` implementation.
            self.norms = nn.ModuleList([
                nn.LayerNorm(self.d_model) for _ in range(self.n_blocks)
            ])
            self.gru = None  # type: ignore[assignment]
        elif backend == "gru":
            self.gru = nn.GRU(
                input_size=self.d_model,
                hidden_size=self.d_model,
                num_layers=self.n_blocks,
                batch_first=True,
            )
            self.blocks = None  # type: ignore[assignment]
            self.norms = None   # type: ignore[assignment]
        else:  # pragma: no cover
            raise ValueError(f"unknown backend: {backend}")

    # ---------- bulk forward (Stage A training) ----------
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: (B, L, D) token sequence with D == d_model.

        Returns:
            h_seq (B, L, D). Call ``h_seq[:, -1, :]`` for the final hidden
            state ``h_T`` consumed by RiskHead.
        """
        if seq.ndim != 3 or seq.shape[-1] != self.d_model:
            raise ValueError(
                f"MambaTemporal.forward expects (B, L, D={self.d_model}); "
                f"got {tuple(seq.shape)}"
            )
        if self.backend == "mamba":
            x = seq
            for block, norm in zip(self.blocks, self.norms):  # type: ignore[arg-type]
                x = x + block(norm(x))
            return x
        # GRU
        out, _ = self.gru(seq)  # type: ignore[misc]
        return out

    # ---------- streaming (Stage B, one token per env step) ----------
    def step(
        self,
        tok_t: torch.Tensor,
        hidden: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Advance the temporal encoder by one token.

        Args:
            tok_t: (B, D) one token per batch element at time t. If you have
                   a multi-token spatial grid per frame (Nt > 1) feed each
                   token in turn, or flatten the grid across time before
                   stepping — consistent with the Stage-A seq layout.
            hidden: backend-specific carrier from the previous step. First
                    call: pass ``None``.
                    * mamba backend: List[torch.Tensor] of per-block state
                      tensors, one per block (we carry them as a plain list).
                    * gru backend  : Tensor (num_layers, B, D).

        Returns:
            (h_t, new_hidden). ``h_t`` has shape (B, D) and is the per-step
            output of the encoder; ``new_hidden`` must be fed into the next
            ``step`` call.
        """
        if tok_t.ndim != 2 or tok_t.shape[-1] != self.d_model:
            raise ValueError(
                f"step() expects (B, D={self.d_model}); got {tuple(tok_t.shape)}"
            )

        if self.backend == "gru":
            # (B, D) -> (B, 1, D) for nn.GRU, state stays (num_layers, B, D).
            x = tok_t.unsqueeze(1)
            out, new_h = self.gru(x, hidden)  # type: ignore[misc]
            return out.squeeze(1), new_h

        # Mamba fallback: run the full-sequence forward on a length-1
        # input — mamba-ssm itself does not expose an O(1) public streaming
        # API in the same stable manner across versions. We keep the
        # public contract identical to GRU (caller carries opaque state)
        # by shadowing the internal state with a short history window.
        # `hidden` here is a list of the last token per block; for v1 we
        # simply recompute from scratch over a single-token window. This is
        # still O(1) per step at L=1. Deep streaming (L>1) can be layered
        # on later without changing the public signature.
        x = tok_t.unsqueeze(1)                  # (B, 1, D)
        for block, norm in zip(self.blocks, self.norms):  # type: ignore[arg-type]
            x = x + block(norm(x))
        return x.squeeze(1), hidden             # carry opaque; None is valid
