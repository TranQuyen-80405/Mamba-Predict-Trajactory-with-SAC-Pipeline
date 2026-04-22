"""
Named DataGenConfig presets for CLI / notebook subprocess.

Dùng khi Jupyter kernel không có torch: notebook gọi
  .venv\\Scripts\\python.exe scripts/datagen/run_datagen_preset.py <preset>
(thư mục gốc repo là cwd), hoặc wrapper ở root ``run_datagen_preset.py``.

Presets:
  smoke_nb       — giống cell tuỳ chọn notebook (data/stage_a_smoke_nb)
  experiment     — bộ nhỏ để so sánh method (data/stage_a_experiment); xem docs/strategy_experiment_protocol.md
  experiment_2gpu — bộ cân bằng nhãn risk cho train compare trên 2x 5060 Ti
  full           — dataset lớn (data/stage_a_full)
  rgb_spotcheck  — 1 scene × 1 rollout ngắn, save_rgb=True → data/stage_a_rgb_spotcheck
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root: .../Pipeline
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HERE = str(_REPO_ROOT)
sys.path.insert(0, _HERE)
sys.path.insert(0, str(_REPO_ROOT / "PointPillars_module"))

from create_dataset_module import DataGenerator
from create_dataset_module.config import DataGenConfig


def _cfg_smoke_nb() -> DataGenConfig:
    return DataGenConfig(
        out_dir=os.path.join(_HERE, "data", "stage_a_smoke_nb"),
        n_scenes=10,
        rollouts_per_scene=2,
        frames_per_rollout=200,
        policy_random_p=0.15,
        policy_scripted_p=0.10,
        policy_adversarial_p=0.75,
        seed=0,
    )


def _cfg_experiment() -> DataGenConfig:
    """
    Small dataset for method comparison (Stage A/B lab runs).
    See docs/strategy_experiment_protocol.md §2.1.
    """
    return DataGenConfig(
        out_dir=os.path.join(_HERE, "data", "stage_a_experiment"),
        n_scenes=24,
        rollouts_per_scene=3,
        frames_per_rollout=120,
        policy_random_p=0.15,
        policy_scripted_p=0.10,
        policy_adversarial_p=0.75,
        save_rgb=False,
        seed=0,
    )


def _cfg_experiment_2gpu() -> DataGenConfig:
    """
    Balanced Stage-A comparison preset for dual mid-range GPUs.

    Goals:
      - keep the set small enough for 4-backbone compare runs
      - target risk_1s class balance in the 5-30% QA band
      - preserve shortcut-buster coverage via StationaryPolicy
    """
    return DataGenConfig(
        out_dir=os.path.join(_HERE, "data", "stage_a_experiment_2gpu_balanced_v7"),
        n_scenes=32,
        rollouts_per_scene=4,
        frames_per_rollout=220,
        # 90% mostly-safe behavior
        policy_random_p=0.55,
        policy_scripted_p=0.35,
        # 10% risky/shortcut-buster behavior
        policy_adversarial_p=0.05,
        policy_stationary_p=0.05,
        camera_jitter_deg=0.0,
        terminate_on_contact=True,
        post_contact_grace_frames=0,
        save_rgb=False,
        seed=23,
    )


def _cfg_full() -> DataGenConfig:
    return DataGenConfig(
        out_dir=os.path.join(_HERE, "data", "stage_a_full"),
        n_scenes=200,
        rollouts_per_scene=4,
        frames_per_rollout=400,
        policy_random_p=0.15,
        policy_scripted_p=0.10,
        policy_adversarial_p=0.75,
        seed=42,
    )


def _cfg_rgb_spotcheck() -> DataGenConfig:
    """Ít frame, bật RGB để mở ảnh kiểm hành vi sim (dataset train nên để save_rgb=False)."""
    return DataGenConfig(
        out_dir=os.path.join(_HERE, "data", "stage_a_rgb_spotcheck"),
        n_scenes=1,
        rollouts_per_scene=1,
        frames_per_rollout=120,
        policy_random_p=0.15,
        policy_scripted_p=0.10,
        policy_adversarial_p=0.75,
        save_rgb=True,
        seed=0,
    )


PRESETS = {
    "experiment": _cfg_experiment,
    "experiment_2gpu": _cfg_experiment_2gpu,
    "smoke_nb": _cfg_smoke_nb,
    "full": _cfg_full,
    "rgb_spotcheck": _cfg_rgb_spotcheck,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "preset",
        choices=sorted(PRESETS.keys()),
        help="Tên preset (experiment | experiment_2gpu | smoke_nb | full | rgb_spotcheck)",
    )
    args = p.parse_args()
    cfg = PRESETS[args.preset]()
    gen = DataGenerator(cfg)
    gen.run()
    print("last_stats:", gen.last_stats)


if __name__ == "__main__":
    main()
