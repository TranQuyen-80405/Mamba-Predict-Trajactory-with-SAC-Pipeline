"""
FullPipeline + trajectory head: joint risk logits and short-horizon ego pose forecast.
Used by ``train_stage_a_compare`` for the notebook comparison (risk + trajectory accuracy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn

from .full_pipeline import FullPipeline

if TYPE_CHECKING:
    from module_pointpillar import PointPillarsNeckExtractor
from .trajectory_head import TrajectoryHead


class FullPipelineRiskAndTraj(FullPipeline):
    """
    Same backbone as ``FullPipeline``; adds ``TrajectoryHead`` on the same ``h_T``.
    ``forward`` returns ``(risk_logits, traj_pred)`` with ``traj_pred`` shape ``(B, H, 3)``.

    Optional **learnable multi-task loss** (Kendall-style): ``log_vars[0:2]`` so
    ``L = exp(-v_1) * L_risk + exp(-v_2) * L_traj + v_1 + v_2``, with ``v_i`` init 0.
    """

    def __init__(
        self,
        pp: "PointPillarsNeckExtractor",
        *,
        traj_horizon: int = 10,
        reducer: Optional[nn.Module] = None,
        mamba: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        traj_head: Optional[TrajectoryHead] = None,
        bev_channels: int = 384,
        token_dim: int = 256,
        learnable_task_loss: bool = True,
    ) -> None:
        super().__init__(
            pp,
            reducer=reducer,
            mamba=mamba,
            head=head,
            bev_channels=bev_channels,
            token_dim=token_dim,
        )
        self.traj_horizon = int(traj_horizon)
        self.traj_head = traj_head or TrajectoryHead(
            in_dim=token_dim, horizon=self.traj_horizon
        )
        if learnable_task_loss:
            self.log_vars: Optional[nn.Parameter] = nn.Parameter(torch.zeros(2))
        else:
            self.log_vars = None

    def forward(
        self,
        pts_seq_bt: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_T = self.forward_to_h_T(pts_seq_bt)
        return self.head(h_T), self.traj_head(h_T)
