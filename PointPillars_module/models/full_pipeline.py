"""
FullPipeline: glues PointPillars + SpatialReducer + MambaTemporal + RiskHead.

Two entry points, mirroring the two-stage training plan:

  * forward(pts_seq_bt) : Stage A training. Accepts the list-of-list layout
                          produced by the RiskBatch collate_fn (§3.3),
                          runs ``pp.extract_neck_forward`` (gradient path)
                          at every time step, feeds the stacked token grid
                          through Mamba, and returns risk logits.
  * step(pts_t, hidden) : Stage B streaming. ``no_grad`` path per §6.4.2.

Call ``freeze_perception()`` once at the start of Stage B to honour the
default S1 freeze regime.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

# module_pointpillar lives one folder up; import it by absolute name so the
# package is still importable when installed in editable mode.
try:
    from module_pointpillar import (
        PointPillarsNeckExtractor,
    )
except ImportError as _err:  # pragma: no cover - environment-dependent
    PointPillarsNeckExtractor = None  # type: ignore[assignment]
    _IMPORT_ERROR: Optional[BaseException] = _err
else:
    _IMPORT_ERROR = None

from .mamba_temporal import MambaTemporal
from .risk_head import RiskHead
from .spatial_reducer import SpatialReducer


class FullPipeline(nn.Module):
    """
    End-to-end perception-stream wrapper.

    Note that ``PointPillarsNeckExtractor`` is *not* a subclass of
    nn.Module — it wraps one internally (``self.pp.model``). Registering
    that inner module under ``self.pp_model`` is what lets PyTorch
    serialize / freeze / move-to-device the perception weights.
    """

    def __init__(
        self,
        pp: "PointPillarsNeckExtractor",
        reducer: Optional[SpatialReducer] = None,
        mamba: Optional[nn.Module] = None,
        head: Optional[RiskHead] = None,
        *,
        bev_channels: int = 384,
        token_dim: int = 256,
    ) -> None:
        super().__init__()

        if _IMPORT_ERROR is not None and pp is None:  # pragma: no cover
            raise _IMPORT_ERROR

        self.pp = pp                                # keep plain attribute
        # Register the underlying nn.Module so state_dict / .to() pick it
        # up alongside the other perception-stream submodules.
        self.pp_model: nn.Module = pp.model

        self.reducer = reducer or SpatialReducer(
            in_channels=bev_channels, out_dim=token_dim,
        )
        self.mamba = mamba or MambaTemporal(d_model=token_dim)
        self.head = head or RiskHead(in_dim=token_dim)

        self.token_dim = token_dim
        self.num_tokens_per_frame = self.reducer.num_tokens

    # ---------- freeze utilities ----------
    def freeze_perception(self) -> None:
        """
        Stage-B default (regime S1): lock the entire perception stream in
        .eval() with requires_grad=False. Matches
        docs/strategy_finetune_with_SAC.md § 4.2.
        """
        self.pp.freeze_all()
        for m in (self.reducer, self.mamba, self.head):
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()

    def perception_parameters(self) -> List[torch.nn.Parameter]:
        """
        Return all parameters inside the perception stream (pp + reducer +
        mamba + head), useful when building the aux-BCE optimizer group in
        Stage B-plus.
        """
        params: List[torch.nn.Parameter] = []
        params.extend(self.pp_model.parameters())
        params.extend(self.reducer.parameters())
        params.extend(self.mamba.parameters())
        params.extend(self.head.parameters())
        return params

    # ---------- Stage A encoding ----------
    def forward_to_h_T(self, pts_seq_bt: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Run perception + temporal encoder; return final hidden (B, D) before heads.
        """
        if not pts_seq_bt:
            raise ValueError("pts_seq_bt is empty.")
        T_ctx = len(pts_seq_bt)
        B = len(pts_seq_bt[0])
        if any(len(frame_list) != B for frame_list in pts_seq_bt):
            raise ValueError("pts_seq_bt has inconsistent batch size per time step.")

        tok_list: List[torch.Tensor] = []
        for t in range(T_ctx):
            frame = pts_seq_bt[t]
            if not frame:
                raise ValueError("pts_seq_bt contains an empty frame list.")
            first = frame[0]
            if first.ndim == 3:
                # Cached BEV mode: each item is already one (C,H,W) feature map.
                bev = torch.stack(frame, dim=0)
            else:
                # Raw points mode: run PointPillars neck online.
                # Stage A1: ``extract_neck_forward`` is the differentiable path
                # (vs ``extract_neck`` + no_grad). With ``pp.freeze_all()``,
                # perception weights do not update; gradients train reducer /
                # temporal / heads from the BEV feature tensor.
                neck = self.pp.extract_neck_forward(frame)
                bev = neck.feature                    # (B, 384, H, W)
            tok_t = self.reducer(bev)             # (B, Nt, D)
            tok_list.append(tok_t)

        seq = torch.stack(tok_list, dim=1)
        B_, T_, Nt_, D_ = seq.shape
        seq = seq.reshape(B_, T_ * Nt_, D_)
        h_seq = self.mamba(seq)                   # (B, L, D)
        return h_seq[:, -1, :]                    # (B, D)

    def forward(
        self,
        pts_seq_bt: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Stage A training path.

        Args:
            pts_seq_bt: outer list over time (length T_ctx), inner list
                        over batch (length B). Each entry is a (N_i, 4)
                        float32 tensor of points in the LiDAR frame. This
                        matches the ``RiskBatch.pts_seq`` contract (§3.3).

        Returns:
            logits (B, 3). Apply ``torch.sigmoid`` at inference time only.
        """
        h_T = self.forward_to_h_T(pts_seq_bt)
        return self.head(h_T)

    # ---------- Stage B streaming step ----------
    @torch.no_grad()
    def step(
        self,
        pts_t: List[torch.Tensor],
        hidden=None,
    ) -> Tuple[torch.Tensor, object]:
        """
        Per-env-step streaming path (§6.4.2). One call = one frame.

        Args:
            pts_t:  list of B point-cloud tensors for the current frame.
            hidden: opaque MambaTemporal state carried across env steps.
                    First call: ``None``.

        Returns:
            (p_risk, new_hidden) with p_risk shape (B, 3) in probability space.
        """
        # Keep all train-time stochastic layers disabled in streaming inference.
        self.eval()

        # Prefer true FP16 autocast on CUDA. On CPU-only debug boxes, keep the
        # same logic path with a no-op context.
        pp_param = next(self.pp_model.parameters(), None)
        device_type = pp_param.device.type if pp_param is not None else "cpu"
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device_type == "cuda"
            else nullcontext()
        )

        with amp_ctx:
            neck = self.pp.extract_neck(pts_t)       # frozen path
            bev = neck.feature
            tok_grid = self.reducer(bev)             # (B, Nt, D)
            # Feed each spatial token through the temporal encoder one at a
            # time so hidden state tracks the streaming contract.
            h_t = None
            _, Nt, _ = tok_grid.shape
            for i in range(Nt):
                h_t, hidden = self.mamba.step(tok_grid[:, i, :], hidden)
            assert h_t is not None
            logits = self.head(h_t)
            p_risk = torch.sigmoid(logits)
        return p_risk, hidden

    # ---------- misc ----------
    def extra_repr(self) -> str:
        return (
            f"num_tokens_per_frame={self.num_tokens_per_frame}, "
            f"token_dim={self.token_dim}"
        )
