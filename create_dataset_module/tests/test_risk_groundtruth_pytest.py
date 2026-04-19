"""
Pytest: risk ground-truth — lookahead_any, Trajectory npz round-trip,
RiskDataset windows, collate_riskbatch, edge cases.

Run from repo root:
    python -m pytest create_dataset_module/tests/test_risk_groundtruth_pytest.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_PP_PKG = _ROOT / "PointPillars_module"
for _p in (_ROOT, _PP_PKG):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from create_dataset_module.generator import lookahead_any  # noqa: E402
from create_dataset_module.risk_dataset import (  # noqa: E402
    RiskDataset,
    collate_riskbatch,
)
from data_contracts import DataGenConfig, Trajectory  # noqa: E402


# ---------------------------------------------------------------------------
# 1) lookahead_any — core labeling
# ---------------------------------------------------------------------------


class TestLookaheadAnyCore:
    def test_index_35_contact_50_horizons(self) -> None:
        """Single contact at 50: t=35 sees it in H=20 but not H=10."""
        T = 100
        contact = np.zeros(T, dtype=np.bool_)
        contact[50] = True

        r05 = lookahead_any(contact, 10)
        r1 = lookahead_any(contact, 20)
        r2 = lookahead_any(contact, 40)

        assert r05[35] == 0.0
        assert r1[35] == 1.0
        assert r2[35] == 1.0

    def test_monotonicity_risk_horizons(self) -> None:
        """risk_2s[t] >= risk_1s[t] >= risk_05s[t] for binary lookahead."""
        rng = np.random.default_rng(0)
        T = 80
        contact = rng.random(T) > 0.92
        r05 = lookahead_any(contact, 10)
        r1 = lookahead_any(contact, 20)
        r2 = lookahead_any(contact, 40)

        assert np.all(r2 + 1e-6 >= r1)
        assert np.all(r1 + 1e-6 >= r05)
        assert set(np.unique(r05).tolist()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# 2) Trajectory npz round-trip
# ---------------------------------------------------------------------------


def _dummy_trajectory(T: int = 25) -> Trajectory:
    H, W = 120, 160
    rng = np.random.default_rng(42)
    depth = rng.uniform(0.5, 7.0, size=(T, H, W)).astype(np.float16)
    rgb = np.zeros((T, H, W, 3), dtype=np.uint8)
    R = np.broadcast_to(np.eye(3, dtype=np.float32), (T, 3, 3)).copy()
    tvec = np.zeros((T, 3), dtype=np.float32)
    ego_state = np.zeros((T, 6), dtype=np.float32)
    ego_vel = np.zeros((T, 6), dtype=np.float32)
    action = rng.uniform(-1, 1, size=(T, 3)).astype(np.float32)
    contact = np.zeros(T, dtype=np.bool_)
    contact[T // 2] = True
    r05 = lookahead_any(contact, 10)
    r1 = lookahead_any(contact, 20)
    r2 = lookahead_any(contact, 40)
    return Trajectory(
        scene_id=7,
        rollout_id=3,
        T=T,
        depth=depth,
        rgb=rgb,
        cam_intrinsics=np.array([80.0, 80.0, 80.0, 60.0], dtype=np.float32),
        cam_extr_R=R,
        cam_extr_t=tvec,
        ego_state=ego_state,
        ego_vel=ego_vel,
        action=action,
        contact_flag=contact,
        risk_05s=r05,
        risk_1s=r1,
        risk_2s=r2,
    )


class TestTrajectorySerialization:
    def test_to_npz_from_npz_preserves_shapes_and_dtypes(self) -> None:
        traj = _dummy_trajectory(T=30)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roundtrip.npz"
            traj.to_npz(path)
            back = Trajectory.from_npz(path)

        assert back.T == traj.T
        assert back.depth.shape == traj.depth.shape
        assert back.depth.dtype == np.float16
        assert back.contact_flag.shape == (traj.T,)
        assert back.contact_flag.dtype == np.bool_
        for name in ("risk_05s", "risk_1s", "risk_2s"):
            a = getattr(back, name)
            assert a.shape == (traj.T,)
            assert a.dtype == np.float32
            torch.testing.assert_close(
                torch.from_numpy(a), torch.from_numpy(getattr(traj, name))
            )


# ---------------------------------------------------------------------------
# 3) RiskDataset + collate
# ---------------------------------------------------------------------------


def _write_index_and_npz(root: Path, T: int = 70) -> None:
    traj = _dummy_trajectory(T=T)
    name = "s0000_r00.npz"
    traj.to_npz(root / name)
    row = {
        "path": name,
        "scene_id": 0,
        "rollout_id": 0,
        "T": T,
        "n_positive_1s": int((traj.risk_1s > 0).sum()),
    }
    with (root / "index.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


class TestRiskDatasetWindowAndCollate:
    def test_pts_seq_length_t_ctx_10(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_index_and_npz(root, T=70)
            ds = RiskDataset(root=str(root), cfg=DataGenConfig(), T_ctx=10)
            assert len(ds) > 0
            sample = ds[0]
            assert len(sample.pts_seq) == 10
            for i, pts in enumerate(sample.pts_seq):
                assert pts.ndim == 2
                assert pts.shape[1] == 4
                assert pts.dtype == torch.float32

    def test_collate_riskbatch_pts_layout_and_labels_b3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_index_and_npz(root, T=70)
            ds = RiskDataset(root=str(root), cfg=DataGenConfig(), T_ctx=10)
            B = min(4, len(ds))
            samples = [ds[i] for i in range(B)]
            batch = collate_riskbatch(samples)

            assert batch.batch_size == B
            assert batch.t_ctx == 10
            assert len(batch.pts_seq) == 10
            for t in range(10):
                assert len(batch.pts_seq[t]) == B
                for b in range(B):
                    assert batch.pts_seq[t][b].shape[1] == 4

            targets = batch.risk_targets()
            assert targets.shape == (B, 3)
            torch.testing.assert_close(
                targets[:, 0], batch.risk_05s, rtol=0, atol=0
            )
            torch.testing.assert_close(
                targets[:, 1], batch.risk_1s, rtol=0, atol=0
            )
            torch.testing.assert_close(
                targets[:, 2], batch.risk_2s, rtol=0, atol=0
            )
            assert batch.traj_future_xyyaw.shape == (B, 10, 3)


# ---------------------------------------------------------------------------
# 4) Edge cases — truncated lookahead, no index errors
# ---------------------------------------------------------------------------


class TestLookaheadEdgeCases:
    def test_horizon_larger_than_T_no_index_error(self) -> None:
        """Horizon may exceed T; slicing uses min(T, t+H) — must not raise."""
        T = 8
        contact = np.zeros(T, dtype=np.bool_)
        contact[3] = True
        out = lookahead_any(contact, 50)
        assert out.shape == (T,)
        assert float(out[3]) == 1.0
        # Earlier frames still see the contact in [t : min(T, t+H)).
        assert float(out[0]) == 1.0

    def test_tiny_T_large_H(self) -> None:
        contact = np.array([False, True, False], dtype=np.bool_)
        out = lookahead_any(contact, 1000)
        assert out.shape == (3,)
        assert float(out[0]) == 1.0
        assert float(out[1]) == 1.0
        assert float(out[2]) == 0.0

    def test_empty_contact(self) -> None:
        assert lookahead_any(np.zeros(0, dtype=np.bool_), 10).shape == (0,)


class TestRiskDatasetShortTrajectory:
    """Dataset index may be empty but must not raise."""

    def test_index_counts_full_traj_window(self) -> None:
        # T=50, T_ctx=10, traj=10 -> t_hi_excl=40, t_lo=9 -> range(9,40) length 31
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_index_and_npz(root, T=50)
            ds = RiskDataset(root=str(root), cfg=DataGenConfig(), T_ctx=10)
            assert len(ds) == 31
            _ = ds[0]

    def test_T_below_threshold_empty_dataset(self) -> None:
        # Need T - traj_horizon <= T_ctx - 1 with traj=10, T_ctx=10 => T <= 19
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_index_and_npz(root, T=19)
            ds = RiskDataset(root=str(root), cfg=DataGenConfig(), T_ctx=10)
            assert len(ds) == 0

    def test_per_horizon_valid_mask_zeros_long_horizon(self) -> None:
        """Near episode end, 2s horizon may be masked while traj target still valid."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_index_and_npz(root, T=30)
            ds = RiskDataset(root=str(root), cfg=DataGenConfig(), T_ctx=10, traj_horizon=10)
            assert len(ds) > 0
            sample = ds[-1]
            assert sample.risk_label_valid[0].item() == 1.0  # 0.5s fits in remain=11
            assert sample.risk_label_valid[1].item() == 0.0  # 1s needs 20 frames ahead
            assert sample.risk_label_valid[2].item() == 0.0
