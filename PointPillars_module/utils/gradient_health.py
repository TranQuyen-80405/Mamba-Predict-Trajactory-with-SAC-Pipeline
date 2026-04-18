"""
Numerical hygiene helpers: gradient norms, Polyak updates, small epsilons.

See docs/skill_avoid_gradient_boom.md.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Tuple

import torch
import torch.nn as nn

EPS = 1e-8


def grad_norm_l2(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    """Total L2 norm of gradients (0 if no grad)."""
    total_sq = torch.zeros((), device="cpu")
    found = False
    for p in parameters:
        if p.grad is None:
            continue
        found = True
        g = p.grad.detach()
        total_sq = total_sq.to(g.device) + g.float().pow(2).sum()
    if not found:
        return torch.zeros((), device="cpu")
    return total_sq.sqrt().cpu()


def max_grad_value(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    """Max absolute value among all parameter gradients (-inf if none)."""
    m = torch.tensor(float("-inf"))
    found = False
    for p in parameters:
        if p.grad is None:
            continue
        found = True
        g = p.grad.detach().abs().max().cpu()
        m = torch.maximum(m, g.float())
    if not found:
        return torch.tensor(0.0)
    return m


def soft_update_polyak(
    target: nn.Module,
    online: nn.Module,
    tau: float,
) -> None:
    """θ' ← (1-τ) θ' + τ θ. Modules must share architecture."""
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1]; got {tau}")
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), online.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def iter_actor_critic_alpha_params(
    actor: nn.Module,
    critic_q1: nn.Module,
    critic_q2: nn.Module,
    log_alpha: torch.nn.Parameter,
) -> Tuple[Iterator[torch.nn.Parameter], ...]:
    """Param iterators for separate per-group grad clipping."""
    return (
        actor.parameters(),
        list(critic_q1.parameters()) + list(critic_q2.parameters()),
        iter([log_alpha]),
    )
