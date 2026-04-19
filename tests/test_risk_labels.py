from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import fields
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from PointPillars_module.data_contracts import RiskSample, Trajectory

LOGGER = logging.getLogger("risk_labels")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "stage_a_experiment"
INDEX_PATH = DATA_ROOT / "index.jsonl"

H05 = 10
H1 = 20
H2 = 40


def _load_rows(index_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sample_rows(rows: List[Dict]) -> List[Dict]:
    seed = int(os.getenv("RISK_LABEL_TEST_SEED", "13"))
    max_files = int(os.getenv("RISK_LABEL_TEST_MAX_FILES", "8"))
    n = min(len(rows), max(5, min(max_files, 10)))
    rng = random.Random(seed)
    return rng.sample(rows, k=n)


def _lookahead_any(contact: np.ndarray, horizon: int) -> np.ndarray:
    out = np.zeros_like(contact, dtype=np.float32)
    T = int(contact.shape[0])
    for t in range(T):
        out[t] = float(np.any(contact[t : min(T, t + horizon)]))
    return out


@pytest.fixture(scope="module")
def sampled_trajectories() -> List[Trajectory]:
    if not INDEX_PATH.is_file():
        pytest.skip(f"dataset index not found: {INDEX_PATH}")

    rows = _load_rows(INDEX_PATH)
    if not rows:
        pytest.skip(f"index is empty: {INDEX_PATH}")

    picked = _sample_rows(rows)
    trajs: List[Trajectory] = []
    for r in picked:
        p = DATA_ROOT / str(r["path"])
        if not p.is_file():
            pytest.skip(f"missing trajectory file listed in index: {p}")
        trajs.append(Trajectory.from_npz(p))

    LOGGER.warning(
        "Risk-label check sampled %d trajectories from %s",
        len(trajs),
        INDEX_PATH,
    )
    return trajs


def test_risk_arrays_shape_dtype_and_binary(sampled_trajectories: List[Trajectory]) -> None:
    checked_frames = 0
    for traj in sampled_trajectories:
        T = int(traj.T)
        assert traj.contact_flag.shape == (T,)
        for name in ("risk_05s", "risk_1s", "risk_2s"):
            arr = getattr(traj, name)
            assert arr.dtype == np.float32, (
                f"{name} dtype must be float32, got {arr.dtype} "
                f"(scene={traj.scene_id} rollout={traj.rollout_id})"
            )
            assert arr.shape == (T,), (
                f"{name} shape must match T={T}, got {arr.shape} "
                f"(scene={traj.scene_id} rollout={traj.rollout_id})"
            )
            uniq = set(np.unique(arr).tolist())
            assert uniq.issubset({0.0, 1.0}), (
                f"{name} has non-binary values {sorted(uniq)} "
                f"(scene={traj.scene_id} rollout={traj.rollout_id})"
            )
        checked_frames += T

    LOGGER.warning(
        "Risk-label shape/type validation passed on %d trajectories (%d frames)",
        len(sampled_trajectories),
        checked_frames,
    )


def test_physical_monotonicity_invariant(sampled_trajectories: List[Trajectory]) -> None:
    violations: List[str] = []
    total = 0
    for traj in sampled_trajectories:
        r05 = traj.risk_05s
        r1 = traj.risk_1s
        r2 = traj.risk_2s
        for t in range(int(traj.T)):
            total += 1
            if r05[t] == 1.0 and (r1[t] != 1.0 or r2[t] != 1.0):
                violations.append(
                    f"scene={traj.scene_id} rollout={traj.rollout_id} frame={t} "
                    f"(risk_05s=1 but risk_1s={r1[t]}, risk_2s={r2[t]})"
                )
            if r1[t] == 1.0 and r2[t] != 1.0:
                violations.append(
                    f"scene={traj.scene_id} rollout={traj.rollout_id} frame={t} "
                    f"(risk_1s=1 but risk_2s={r2[t]})"
                )

    if violations:
        pytest.fail("Monotonicity violations:\n" + "\n".join(violations[:50]))

    LOGGER.warning(
        "Monotonicity invariant holds on %d sampled trajectories (%d frames).",
        len(sampled_trajectories),
        total,
    )


def test_truncation_end_of_episode_integrity(sampled_trajectories: List[Trajectory]) -> None:
    # Ensure labels match documented truncated-lookahead semantics near episode end.
    # Training can then safely combine this with risk_label_valid masking.
    has_valid_mask = any(f.name == "risk_label_valid" for f in fields(RiskSample))
    tail_frames_checked = 0
    truncated_2s_frames = 0

    for traj in sampled_trajectories:
        T = int(traj.T)
        contact = np.asarray(traj.contact_flag, dtype=np.bool_)

        exp05 = _lookahead_any(contact, H05)
        exp1 = _lookahead_any(contact, H1)
        exp2 = _lookahead_any(contact, H2)

        np.testing.assert_array_equal(
            traj.risk_05s,
            exp05,
            err_msg=f"risk_05s mismatch (scene={traj.scene_id} rollout={traj.rollout_id})",
        )
        np.testing.assert_array_equal(
            traj.risk_1s,
            exp1,
            err_msg=f"risk_1s mismatch (scene={traj.scene_id} rollout={traj.rollout_id})",
        )
        np.testing.assert_array_equal(
            traj.risk_2s,
            exp2,
            err_msg=f"risk_2s mismatch (scene={traj.scene_id} rollout={traj.rollout_id})",
        )

        start = max(0, T - H2)
        tail_frames_checked += (T - start)
        # Full 2.0s lookahead exists only when t + H2 <= T.
        for t in range(start, T):
            if (t + H2) > T:
                truncated_2s_frames += 1

    assert tail_frames_checked > 0
    assert truncated_2s_frames > 0
    assert has_valid_mask, "RiskSample should expose risk_label_valid for truncated horizons."

    LOGGER.warning(
        "Truncation integrity checked: tail_frames=%d truncated_2s_frames=%d "
        "valid_mask_present=%s",
        tail_frames_checked,
        truncated_2s_frames,
        has_valid_mask,
    )
