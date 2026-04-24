"""Unit tests for Stage A train scripts (imports, metrics, CLI --help)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


def test_validate_backbone() -> None:
    from training.train_stage_a_compare import validate_backbone

    assert validate_backbone("Mamba") == "mamba"
    with pytest.raises(ValueError):
        validate_backbone("not_a_backbone")


def test_compare_parse_models() -> None:
    from training.train_stage_a_compare import _parse_models

    assert _parse_models(" mamba , lstm ") == ["mamba", "lstm"]


def test_validate_stage() -> None:
    from training.train_stage_a_compare import validate_stage

    assert validate_stage("A2") == "a2"
    with pytest.raises(ValueError):
        validate_stage("a3")


@pytest.mark.parametrize(
    "script",
    [
        "training/train_stage_a_mamba.py",
        "training/train_stage_a_rnn_gru.py",
        "training/train_stage_a_lstm.py",
        "training/train_stage_a_transformer.py",
        "training/train_stage_a_compare.py",
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


def test_train_stage_a_compare_has_models_flag() -> None:
    """Compare entrypoint should expose --models for backbone selection."""
    path = _PKG_ROOT / "training" / "train_stage_a_compare.py"
    r = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True,
        text=True,
        cwd=str(_PKG_ROOT),
        timeout=60,
    )
    assert r.returncode == 0
    assert "--models" in r.stdout
