"""
Stage B — SAC reference modules (proprio-only Actor / twin Critic).

This file is the **implementation anchor** for `docs/strategy_finetune_with_SAC.md` §7–§8
and `docs/skill_avoid_gradient_boom.md`: `log_std` clamp, stable Gaussian + tanh log-prob,
separate gradient clipping hooks, Polyak target update.

A full env + replay training loop is **not** included here — wire `sample` / critics into
your runner and call the clip helpers **after the backward that produced those grads**.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_PKG = os.path.dirname(os.path.abspath(__file__))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from utils.gradient_health import EPS, soft_update_polyak

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


class ActorMLP(nn.Module):
    """
    Squashed Gaussian policy on proprio state.
    Final layers use small uniform init to avoid saturated actions at startup.
    """

    LOGSTD_MIN = -5.0
    LOGSTD_MAX = 2.0

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden: int = 256,
        final_init_std: float = 3e-3,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.logstd = nn.Linear(hidden, act_dim)
        nn.init.uniform_(self.mu.weight, -final_init_std, final_init_std)
        nn.init.uniform_(self.logstd.weight, -final_init_std, final_init_std)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logstd.bias)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(s)
        mu = self.mu(h)
        logstd = self.logstd(h).clamp(self.LOGSTD_MIN, self.LOGSTD_MAX)
        return mu, logstd

    def sample(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logstd = self(s)
        std = logstd.exp().clamp(min=EPS)
        noise = torch.randn_like(mu)
        u = mu + std * noise
        a = torch.tanh(u)
        # log N(u | mu, std) - sum over action dim
        log_pu = (-0.5 * ((u - mu) / std) ** 2 - logstd - _HALF_LOG_2PI).sum(-1)
        # Jacobian of tanh: sum_a [ log(1 - tanh(u)^2) ]  (stable form)
        log_det = (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
        log_pa = log_pu - log_det
        return a, log_pa


class TwinCritic(nn.Module):
    """Twin Q, concat(state, action) MLP."""

    def __init__(self, state_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()

        def branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(state_dim + act_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )

        self.q1 = branch()
        self.q2 = branch()

    def forward(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


def clip_grad_actor(actor: nn.Module, max_norm: float = 1.0) -> torch.Tensor:
    return torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm)


def clip_grad_critic(critic: TwinCritic, max_norm: float = 1.0) -> torch.Tensor:
    params = list(critic.q1.parameters()) + list(critic.q2.parameters())
    return torch.nn.utils.clip_grad_norm_(params, max_norm)


def clip_grad_alpha(log_alpha: torch.nn.Parameter, max_norm: float = 1.0) -> torch.Tensor:
    return torch.nn.utils.clip_grad_norm_([log_alpha], max_norm)


@dataclass
class RewardNormalizer:
    """
    Incremental reward scaling (EMA). Use on scalar rewards before TD targets if spikes occur.
    y = (r - mean) / (std + eps). Call ``update`` each step from raw env + shaped reward.
    """

    momentum: float = 0.01
    eps: float = 1e-8
    mean: float = 0.0
    var: float = 1.0

    def update(self, r_batch: torch.Tensor) -> None:
        r = float(r_batch.mean().detach().cpu())
        self.mean = (1.0 - self.momentum) * self.mean + self.momentum * r
        v = float(r_batch.var(unbiased=False).detach().cpu())
        self.var = (1.0 - self.momentum) * self.var + self.momentum * max(v, self.eps)

    def normalize(self, r: torch.Tensor) -> torch.Tensor:
        std = (self.var + self.eps) ** 0.5
        return (r - self.mean) / (std + self.eps)


__all__ = [
    "ActorMLP",
    "TwinCritic",
    "RewardNormalizer",
    "clip_grad_actor",
    "clip_grad_critic",
    "clip_grad_alpha",
    "soft_update_polyak",
    "EPS",
]
