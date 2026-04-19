"""
SpatialReducer: dimensional bridge between PointPillars BEV output and
the Mamba temporal encoder.

Exact spec from docs/strategy_full_pipeline.md § 4.2:

    Input:  (B, 384, 248, 216)
    Conv2d(384, 256, k=3, s=2, p=1) + BN + ReLU   -> (B, 256, 124, 108)
    Conv2d(256, 256, k=3, s=2, p=1) + BN + ReLU   -> (B, 256,  62,  54)
    Conv2d(256, 256, k=3, s=2, p=1) + BN + ReLU   -> (B, 256,  31,  27)
    AdaptiveAvgPool2d((4, 4))                      -> (B, 256,   4,   4)
    flatten last two dims                          -> (B, 16, 256)

Parameters ~2.1 M. The 4x4 grid is a deliberate sweet spot: enough
spatial resolution to distinguish front/left/right/back quadrants and
inner/outer rings, without blowing up the Mamba sequence length.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _init_conv_he(m: nn.Module) -> None:
    """Kaiming uniform for ReLU conv stacks (fan_in)."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class SpatialReducer(nn.Module):
    """BEV feature (B,C_in,H,W) -> flat token grid (B, Nt, D)."""

    def __init__(
        self,
        in_channels: int = 384,
        hidden_channels: int = 256,
        out_dim: int = 256,
        pool_size: int = 4,
    ) -> None:
        super().__init__()
        if out_dim != hidden_channels:
            raise ValueError(
                f"out_dim ({out_dim}) must equal hidden_channels "
                f"({hidden_channels}) for this spec (flatten-last-two-dims)."
            )

        self.out_dim = out_dim
        self.pool_size = pool_size
        self.num_tokens = pool_size * pool_size

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.apply(_init_conv_he)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev: (B, C_in, H, W). Typically (B, 384, 248, 216) from
                 PointPillarsNeckExtractor.

        Returns:
            Token grid (B, Nt=pool_size^2, D=out_dim).
        """
        if bev.ndim != 4:
            raise ValueError(
                f"SpatialReducer expects 4-D BEV input (B,C,H,W); got "
                f"{tuple(bev.shape)}"
            )
        x = self.stage1(bev)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)                        # (B, D, pool, pool)
        B, D, _, _ = x.shape
        tokens = x.view(B, D, self.num_tokens).transpose(1, 2).contiguous()
        return tokens                           # (B, Nt, D)
