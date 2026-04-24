"""
Stage A losses.

focal_bce(logits, targets, gamma=2.0, weight=(1.0, 0.8, 0.5), valid_mask=None, label_smoothing=0.0)
    Focal binary cross-entropy (§ 5.3 of docs/strategy_full_pipeline.md).
    * logits  : (B, K) raw outputs (no sigmoid).
    * targets : (B, K) float in {0, 1}.
    * gamma   : focusing parameter; gamma=0 recovers vanilla BCEWithLogits.
    * weight  : per-horizon scalar weights, length K ([0.5s, 1s, 2s]).
    * valid_mask : optional (B, K) in {{0,1}}; zeros excluded from the mean.

oversample_positive_indices(risk_1s, factor=10)
    Sampler helper: indices list with positives duplicated ``factor`` times.
    Used by the RiskDataset in create_dataset_module.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_NUMERIC_EPS = 1e-8


class MultiTaskLossWrapper(nn.Module):
    """
    Homoscedastic uncertainty weighting for two-task optimization.

    Kendall et al. (2018):
        L = exp(-s1) * L1 + s1 + exp(-s2) * L2 + s2
    where s1,s2 are learnable log variances.
    """

    def __init__(
        self,
        init_log_var_risk: float = 0.0,
        init_log_var_traj: float = 0.0,
        log_var_min: float = -5.0,
        log_var_max: float = 5.0,
    ) -> None:
        super().__init__()
        self.log_var_risk = nn.Parameter(torch.tensor(float(init_log_var_risk)))
        self.log_var_traj = nn.Parameter(torch.tensor(float(init_log_var_traj)))
        self.log_var_min = float(log_var_min)
        self.log_var_max = float(log_var_max)
        if self.log_var_min >= self.log_var_max:
            raise ValueError("log_var_min must be < log_var_max")

    def _clamped(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=self.log_var_min, max=self.log_var_max)

    def forward(self, loss_risk: torch.Tensor, loss_traj: torch.Tensor) -> torch.Tensor:
        s_risk = self._clamped(self.log_var_risk)
        s_traj = self._clamped(self.log_var_traj)
        return (
            torch.exp(-s_risk) * loss_risk + s_risk
            + torch.exp(-s_traj) * loss_traj + s_traj
        )


def focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    weight: Optional[Sequence[float]] = (1.0, 0.8, 0.5),
    valid_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Focal BCE with logits. Per-sample, per-horizon loss is:

        p_t = sigmoid(logits) if target==1 else 1 - sigmoid(logits)
        focal = -(1 - p_t)**gamma * log(p_t)

    With ``weight`` applied as a scalar multiplier per horizon column.
    If ``valid_mask`` is (B, K) with values in {0, 1}, masked elements are
    omitted from the mean (truncated-lookahead frames / per-horizon drops).

    ``label_smoothing`` (0–0.5) pulls hard 0/1 targets toward 0.5 (multi-label
    BCE). Implemented without ``label_smoothing=`` on ``binary_cross_entropy_with_logits``
    for PyTorch builds that omit that keyword.
    """
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} and targets {tuple(targets.shape)} "
            f"must match."
        )
    ls = float(label_smoothing)
    if ls < 0.0 or ls > 0.5:
        raise ValueError(f"label_smoothing must be in [0, 0.5]; got {ls}")

    t = targets.to(dtype=logits.dtype)
    if ls > 0.0:
        t = t * (1.0 - ls) + 0.5 * ls
    # BCE with logits gives us -log(p_t) per element (numerically stable).
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")

    if gamma == 0.0:
        focal = bce
    else:
        # p_t = exp(-bce) (since bce = -log p_t). Clamp away from {0,1} for (1-pt)**gamma.
        pt = torch.exp(-bce)
        pt = torch.clamp(pt, min=_NUMERIC_EPS, max=1.0 - _NUMERIC_EPS)
        focal = ((1.0 - pt) ** gamma) * bce

    if weight is not None:
        w = torch.as_tensor(weight, dtype=focal.dtype, device=focal.device)
        if w.ndim != 1 or w.shape[0] != focal.shape[-1]:
            raise ValueError(
                f"weight length {tuple(w.shape)} must match last dim "
                f"{focal.shape[-1]} of logits."
            )
        focal = focal * w

    if valid_mask is not None:
        vm = valid_mask.to(dtype=focal.dtype, device=focal.device)
        if vm.shape != focal.shape:
            raise ValueError(
                f"valid_mask shape {tuple(vm.shape)} must match logits "
                f"{tuple(focal.shape)}."
            )
        focal = focal * vm
        if reduction == "mean":
            denom = vm.sum().clamp_min(1.0)
            return focal.sum() / denom
        if reduction == "sum":
            return focal.sum()
        if reduction == "none":
            return focal
        raise ValueError(f"unknown reduction: {reduction}")

    if reduction == "mean":
        return focal.mean()
    if reduction == "sum":
        return focal.sum()
    if reduction == "none":
        return focal
    raise ValueError(f"unknown reduction: {reduction}")


def oversample_positive_indices(
    risk_1s: Union[np.ndarray, Sequence[float], torch.Tensor],
    factor: int = 10,
) -> List[int]:
    """
    Return a list of indices into ``risk_1s`` where every positive index is
    duplicated ``factor`` times and every negative index appears exactly
    once. Scene-stratified splitting (§ 5.6) is the caller's responsibility;
    this helper only handles class balance within a split.

    Example:
        idxs = oversample_positive_indices(ds.risk_1s, factor=10)
        sampler = SubsetRandomSampler(idxs)
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1; got {factor}")

    arr = np.asarray(
        risk_1s.detach().cpu().numpy() if torch.is_tensor(risk_1s) else risk_1s
    ).astype(np.float32).reshape(-1)

    pos_idx = np.where(arr > 0.5)[0]
    neg_idx = np.where(arr <= 0.5)[0]

    out: List[int] = neg_idx.tolist() + pos_idx.repeat(factor).tolist()
    return out


__all__ = ["focal_bce", "oversample_positive_indices", "MultiTaskLossWrapper"]
