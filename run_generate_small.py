"""
run_generate_small.py
Config A — smoke test locally (~5 minutes on CPU).

Run from repo root with the venv active:
    python run_generate_small.py

Outputs:
    data/stage_a_smoke/
        index.jsonl
        s0000_r00.npz, s0000_r01.npz, ...
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "PointPillars_module"))

from create_dataset_module import DataGenerator
from create_dataset_module.config import DataGenConfig


def main() -> None:
    cfg = DataGenConfig(
        out_dir="data/stage_a_smoke",
        n_scenes=10,
        rollouts_per_scene=2,
        frames_per_rollout=200,
        policy_random_p=0.5,
        policy_scripted_p=0.3,
        policy_adversarial_p=0.2,
        seed=0,
    )

    print("[config A] starting small smoke generation ...")
    print(f"  target frames (pre-term) = "
          f"{cfg.n_scenes * cfg.rollouts_per_scene * cfg.frames_per_rollout}")
    print(f"  out_dir = {cfg.out_dir}")

    gen = DataGenerator(cfg)
    gen.run()

    print("\n[config A] done. last_stats:")
    for k, v in gen.last_stats.items():
        print(f"  {k!s:24s} = {v}")


if __name__ == "__main__":
    main()
