import torch
import torch.nn as nn
from typing import Optional
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class RecurrentTrajectoryPredictor(nn.Module):
    """
    1. Trajectory Forecasting: Recurrent Baseline Architecture (LSTM/GRU)
    
    After PointPillars converts point clouds into BEV features,
    a Global Average Pooling (GAP) layer compresses the spatial dimensions (H, W) into a sequence of vectors.
    The LSTM/GRU core updates the hidden state across time steps.
    The output is the future coordinates (x, y, yaw) via a Fully Connected layer.
    """
    def __init__(self, in_channels: int, hidden_size: int, horizon: int, num_layers: int = 1, rnn_type: str = 'LSTM'):
        super().__init__()
        self.horizon = horizon
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
            
        self.head = nn.Linear(hidden_size, horizon * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: BEV Features (B, T, C, H, W)
        Returns:
            Predicted trajectory (B, horizon, 3)
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        x_gap = self.gap(x_flat).view(B, T, C) # -> (B, T, C)
        
        out, _ = self.rnn(x_gap) # out: (B, T, hidden_size)
        last_hidden = out[:, -1, :] # Take the last time step (B, hidden_size)
        
        traj = self.head(last_hidden) # (B, horizon * 3)
        return traj.view(B, self.horizon, 3)


class TransformerTrajectoryPredictor(nn.Module):
    """
    2. Trajectory Forecasting: Temporal Transformer Architecture (Transformer Encoder)
    
    Each time slice is treated as a "token". Uses Self-attention mechanism 
    to understand the entire movement history simultaneously instead of recursively.
    The Transformer's output is passed through an MLP Head to predict coordinates.
    """
    def __init__(self, in_channels: int, d_model: int, nhead: int, num_layers: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.input_proj = nn.Linear(in_channels, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, horizon * 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: BEV Features (B, T, C, H, W)
        Returns:
            Predicted trajectory (B, horizon, 3)
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        x_gap = self.gap(x_flat).view(B, T, C)
        
        tokens = self.input_proj(x_gap) # (B, T, d_model)
        
        out = self.transformer(tokens) # (B, T, d_model)
        last_token = out[:, -1, :] # (B, d_model)
        
        traj = self.head(last_token)
        return traj.view(B, self.horizon, 3)


class MambaTrajectoryPredictor(nn.Module):
    """
    3. Trajectory Forecasting: Traj-Mamba Architecture (State Space Model)
    
    Replaces Attention with Mamba (Selective Scan). Maintains O(n) speed and
    compresses feature sequences efficiently. 
    Requires `mamba_ssm` library to be installed.
    """
    def __init__(self, in_channels: int, d_model: int, d_state: int, d_conv: int, expand: int, horizon: int):
        super().__init__()
        if Mamba is None:
            raise ImportError("Please install mamba_ssm to use this module.")
            
        self.horizon = horizon
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.input_proj = nn.Linear(in_channels, d_model)
        
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        self.head = nn.Linear(d_model, horizon * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: BEV Features (B, T, C, H, W)
        Returns:
            Predicted trajectory (B, horizon, 3)
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        x_gap = self.gap(x_flat).view(B, T, C)
        
        tokens = self.input_proj(x_gap)
        
        out = self.mamba(tokens) # (B, T, d_model)
        last_hidden = out[:, -1, :] # (B, d_model)
        
        traj = self.head(last_hidden)
        return traj.view(B, self.horizon, 3)
