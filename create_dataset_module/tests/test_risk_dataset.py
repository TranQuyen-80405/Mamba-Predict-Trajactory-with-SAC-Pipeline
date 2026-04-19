"""
RiskDataset from a synthetic Trajectory fixture.

All PyBullet-free. Produces a fake rollout directly via the Trajectory
dataclass, writes it to disk along with a tiny index.jsonl, then walks
the dataset through collate_riskbatch.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_PP_PKG = os.path.join(_ROOT, "PointPillars_module")
for _p in (_ROOT, _PP_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PointPillars_module.types import DataGenConfig, Trajectory  # noqa: E402
from create_dataset_module.risk_dataset import (  # noqa: E402
    RiskDataset,
    collate_riskbatch,
    scene_stratified_split,
)


def _make_trajectory(T: int = 60, scene_id: int = 0, rollout_id: int = 0):
    rng = np.random.default_rng(scene_id * 10 + rollout_id)
    H, W = 120, 160
    depth = rng.uniform(0.3, 7.5, size=(T, H, W)).astype(np.float16)
    rgb = rng.integers(0, 255, size=(T, H, W, 3)).astype(np.uint8)
    R = np.broadcast_to(np.eye(3, dtype=np.float32), (T, 3, 3)).copy()
    t = np.zeros((T, 3), dtype=np.float32)
    ego_state = np.zeros((T, 6), dtype=np.float32)
    ego_vel = np.zeros((T, 6), dtype=np.float32)
    action = rng.uniform(-1, 1, size=(T, 2)).astype(np.float32)

    contact = np.zeros(T, dtype=np.bool_)
    contact[50] = True

    def lk(flag, H_):
        out = np.zeros_like(flag, dtype=np.float32)
        for i in range(len(flag)):
            out[i] = float(flag[i:i + H_].any())
        return out

    return Trajectory(
        scene_id=scene_id, rollout_id=rollout_id, T=T,
        depth=depth, rgb=rgb,
        cam_intrinsics=np.array([80, 80, 80, 60], dtype=np.float32),
        cam_extr_R=R, cam_extr_t=t,
        ego_state=ego_state, ego_vel=ego_vel,
        action=action, contact_flag=contact,
        risk_05s=lk(contact, 10), risk_1s=lk(contact, 20),
        risk_2s=lk(contact, 40),
    )


def _write_fixture(root: Path, n_scenes: int = 2, T: int = 60) -> None:
    index = root / "index.jsonl"
    with index.open("w", encoding="utf-8") as f:
        for s in range(n_scenes):
            traj = _make_trajectory(T=T, scene_id=s, rollout_id=0)
            name = f"s{s:04d}_r00.npz"
            traj.to_npz(root / name)
            row = {"path": name, "scene_id": s, "rollout_id": 0, "T": T,
                   "n_positive_1s": int((traj.risk_1s > 0).sum())}
            f.write(json.dumps(row) + "\n")


class TestRiskDataset(unittest.TestCase):
    def test_len_and_entry_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_fixture(Path(tmp), n_scenes=1, T=60)
            ds = RiskDataset(root=tmp, cfg=DataGenConfig(), T_ctx=10)
            # valid frames are bounded by trajectory target horizon:
            # t in [T_ctx-1, T-traj_horizon) => [9, 50) for T=60, H=10.
            self.assertEqual(len(ds), 60 - 10 - (10 - 1))
            sample = ds[0]
            self.assertEqual(len(sample.pts_seq), 10)
            for pts in sample.pts_seq:
                self.assertEqual(pts.shape[1], 4)
                self.assertEqual(pts.dtype, torch.float32)
            self.assertEqual(sample.action_seq.shape, (10, 2))
            self.assertEqual(sample.ego_vel_seq.shape, (10, 6))
            self.assertEqual(sample.traj_future_xyyaw.shape, (10, 3))

    def test_collate_batch_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_fixture(Path(tmp), n_scenes=1, T=60)
            ds = RiskDataset(root=tmp, cfg=DataGenConfig(), T_ctx=10)
            items = [ds[i] for i in range(min(3, len(ds)))]
            batch = collate_riskbatch(items)
            B = len(items)
            self.assertEqual(batch.batch_size, B)
            self.assertEqual(batch.t_ctx, 10)
            self.assertEqual(len(batch.pts_seq), 10)
            self.assertEqual(len(batch.pts_seq[0]), B)
            self.assertEqual(batch.action_seq.shape, (B, 10, 2))
            self.assertEqual(batch.risk_targets().shape, (B, 3))
            self.assertEqual(batch.traj_future_xyyaw.shape, (B, 10, 3))

    def test_risk_1s_array_returns_all_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_fixture(Path(tmp), n_scenes=2, T=60)
            ds = RiskDataset(root=tmp, cfg=DataGenConfig(), T_ctx=10)
            arr = ds.risk_1s_array()
            self.assertEqual(arr.shape, (len(ds),))
            self.assertTrue(set(np.unique(arr).tolist()).issubset({0.0, 1.0}))

    def test_scene_filter_restricts_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_fixture(Path(tmp), n_scenes=3, T=60)
            ds_full = RiskDataset(root=tmp, cfg=DataGenConfig(), T_ctx=10)
            ds_one = RiskDataset(
                root=tmp, cfg=DataGenConfig(), T_ctx=10, scene_filter=[1],
            )
            self.assertLess(len(ds_one), len(ds_full))
            for e in ds_one.entries:
                self.assertEqual(e["scene_id"], 1)


class TestSceneStratifiedSplit(unittest.TestCase):
    def test_disjoint_scene_splits(self):
        scenes = list(range(10))
        tr, va, te = scene_stratified_split(scenes, (0.6, 0.2, 0.2), seed=0)
        total = set(tr) | set(va) | set(te)
        self.assertEqual(total, set(scenes))
        self.assertEqual(set(tr) & set(va), set())
        self.assertEqual(set(tr) & set(te), set())
        self.assertEqual(set(va) & set(te), set())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
