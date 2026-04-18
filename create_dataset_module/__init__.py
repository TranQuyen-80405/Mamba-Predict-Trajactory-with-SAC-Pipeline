"""
create_dataset_module — offline Stage A dataset generation from PyBullet.

Public surface:
    DatasetEnv          thin wrapper around pybullet_navigation.RL_Env
    RandomPolicy / ScriptedPolicy / AdversarialPolicy / StationaryPolicy
    DataGenerator       rollouts -> Trajectory.npz + index.jsonl
    RiskDataset         torch.utils.data.Dataset of RiskSample
    collate_riskbatch   collate_fn returning RiskBatch

All identifiers mirror docs/strategy_full_pipeline.md § 3 / § 5.
"""

from .env_wrapper import DatasetEnv
from .policies import (
    AdversarialPolicy,
    RandomPolicy,
    ScriptedPolicy,
    StationaryPolicy,
)
from .generator import DataGenerator
from .risk_dataset import RiskDataset, collate_riskbatch

__all__ = [
    "DatasetEnv",
    "RandomPolicy",
    "ScriptedPolicy",
    "AdversarialPolicy",
    "StationaryPolicy",
    "DataGenerator",
    "RiskDataset",
    "collate_riskbatch",
]
