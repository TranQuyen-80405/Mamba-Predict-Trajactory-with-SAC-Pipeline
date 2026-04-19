"""Unit tests for Stage A train scripts (imports, metrics, CLI --help)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


def test_validate_backbone() -> None:
    from training.stage_a_single_run import validate_backbone

    assert validate_backbone("Mamba") == "mamba"
    with pytest.raises(ValueError):
        validate_backbone("not_a_backbone")


def test_traj_error_metrics_identical() -> None:
    from training.stage_a_single_run import traj_error_metrics

    pred = np.zeros((2, 5, 3), dtype=np.float32)
    gt = np.zeros((2, 5, 3), dtype=np.float32)
    m = traj_error_metrics(pred, gt)
    assert m["traj_ade_xy_m"] == 0.0
    assert m["traj_fde_xy_m"] == 0.0
    assert m["traj_rmse_all"] == 0.0


def test_compare_parse_models() -> None:
    from training.train_stage_a_compare import _parse_models

    assert _parse_models(" mamba , lstm ") == ["mamba", "lstm"]


def test_validate_stage() -> None:
    from training.stage_a_single_run import validate_stage

    assert validate_stage("A2") == "a2"
    with pytest.raises(ValueError):
        validate_stage("a3")


def test_collect_val_metrics_empty_loader_returns_nans() -> None:
    from training.stage_a_single_run import collect_val_metrics

    class _DummyPP:
        def __init__(self) -> None:
            self.model = type("_M", (), {"neck": nn.Identity()})()

    class _DummyModel:
        def __init__(self) -> None:
            self.reducer = nn.Identity()
            self.mamba = nn.Identity()
            self.head = nn.Identity()
            self.traj_head = nn.Identity()
            self.pp = _DummyPP()

    m = collect_val_metrics(  # type: ignore[arg-type]
        _DummyModel(),
        [],
        torch.device("cpu"),
        measure_inference_latency=True,
    )
    assert m["val_sample_count"] == 0.0
    assert np.isnan(m["ap_risk_1s"])
    assert np.isnan(m["traj_ade_xy_m"])
    assert np.isnan(m["val_inference_ms_per_sample"])


@pytest.mark.parametrize(
    "script",
    [
        "training/train_stage_a_mamba.py",
        "training/train_stage_a_rnn_gru.py",
        "training/train_stage_a_lstm.py",
        "training/train_stage_a_transformer.py",
        "training/stage_a_single_run.py",
    ],
)
def test_train_script_help_exits_zero(script: str) -> None:
    path = _PKG_ROOT / script
    r = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True,
        text=True,
        cwd=str(_PKG_ROOT),
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert "data_root" in (r.stdout + r.stderr)
    assert "--stage" in (r.stdout + r.stderr)
    assert "--early_stop_patience" in (r.stdout + r.stderr)


def test_stage_a_single_run_requires_backbone_when_run_as_main() -> None:
    """Direct ``python training/stage_a_single_run.py`` must require --backbone."""
    path = _PKG_ROOT / "training" / "stage_a_single_run.py"
    r = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True,
        text=True,
        cwd=str(_PKG_ROOT),
        timeout=60,
    )
    assert r.returncode == 0
    assert "--backbone" in r.stdout
