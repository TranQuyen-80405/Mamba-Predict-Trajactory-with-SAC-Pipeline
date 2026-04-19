"""
Perception-stream models for the RobotDog pipeline.

Exposes:
    SpatialReducer    -  BEV (B,384,H,W) -> token grid (B,Nt,D)    § 4.2
    MambaTemporal     -  temporal encoder, mamba-ssm or GRU fallback § 4.3
    RiskHead          -  (B,D) -> (B,3) logits, no sigmoid           § 4.4
    FullPipeline      -  PointPillars -> SpatialReducer -> Mamba -> RiskHead

All identifiers mirror docs/strategy_full_pipeline.md § 4 verbatim.
"""

from .spatial_reducer import SpatialReducer
from .mamba_temporal import MambaTemporal
from .temporal_encoders import LSTMTemporal, TransformerEncoderTemporal
from .temporal_factory import TemporalKind, build_temporal
from .risk_head import RiskHead
from .full_pipeline import FullPipeline

__all__ = [
    "SpatialReducer",
    "MambaTemporal",
    "LSTMTemporal",
    "TransformerEncoderTemporal",
    "TemporalKind",
    "build_temporal",
    "RiskHead",
    "FullPipeline",
]
