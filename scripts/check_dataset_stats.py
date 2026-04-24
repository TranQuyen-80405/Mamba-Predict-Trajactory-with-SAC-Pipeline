"""
Inspect an existing Stage-A dataset without re-running datagen.

Example:
  python run_dataset_stats.py --data_root data/stage_a_experiment --T_ctx 40 --traj_horizon 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Allow running as `python scripts/check_dataset_stats.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_ratios(text: str) -> Tuple[float, float, float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError(
            f"--split_ratios must have 3 comma-separated floats, got: {text!r}"
        )
    s = sum(vals)
    if s <= 0:
        raise ValueError(f"--split_ratios must sum to > 0, got {vals}")
    return (vals[0] / s, vals[1] / s, vals[2] / s)


def _read_index_rows(data_root: Path) -> List[Dict]:
    idx = data_root / "index.jsonl"
    if not idx.is_file():
        raise FileNotFoundError(f"index.jsonl not found: {idx}")
    rows: List[Dict] = []
    with idx.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"index.jsonl has no rows: {idx}")
    return rows


def _effective_t(npz_path: Path) -> int:
    with np.load(npz_path, allow_pickle=False) as data:
        keys = ("risk_05s", "risk_1s", "risk_2s", "ego_state")
        arr_lens = [int(np.asarray(data[k]).shape[0]) for k in keys if k in data]
        if not arr_lens:
            raise ValueError(f"{npz_path}: missing expected arrays {keys}")
        return int(min(arr_lens))


def _print_horizon_stats(tag: str, stats: Dict[str, Dict[str, int]]) -> None:
    print(f"\n[{tag}] horizon sample stats")
    for k in ("risk_05s", "risk_1s", "risk_2s"):
        st = stats.get(k, {"valid": 0, "positive": 0, "negative": 0})
        valid = int(st["valid"])
        pos = int(st["positive"])
        ratio = (float(pos) / float(valid)) if valid > 0 else 0.0
        print(
            f"  {k:<9} valid={valid:6d} positive={pos:6d} "
            f"negative={int(st['negative']):6d} pos_rate={ratio*100:6.2f}%"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Check existing dataset statistics.")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument(
        "--split_ratios",
        type=str,
        default="0.75,0.25,0.0",
        help="train,val,test ratios for scene_stratified_split, e.g. 0.75,0.25,0.0",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--T_ctx",
        type=int,
        default=40,
        help="Context length used by training (for sample-level stats).",
    )
    ap.add_argument(
        "--traj_horizon",
        type=int,
        default=10,
        help="Future trajectory horizon used by training (for sample-level stats).",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    rows = _read_index_rows(data_root)
    npz_paths = sorted({str((data_root / r["path"]).resolve()) for r in rows})

    n_rollouts = len(rows)
    scene_ids = sorted({int(r["scene_id"]) for r in rows})
    n_scenes = len(scene_ids)

    total_index_t = int(sum(int(r.get("T", 0)) for r in rows))
    total_effective_t = 0
    missing = 0
    for p in npz_paths:
        pp = Path(p)
        if not pp.is_file():
            missing += 1
            continue
        total_effective_t += _effective_t(pp)

    pos05 = int(sum(int(r.get("n_positive_05s", 0)) for r in rows))
    pos1 = int(sum(int(r.get("n_positive_1s", 0)) for r in rows))
    pos2 = int(sum(int(r.get("n_positive_2s", 0)) for r in rows))
    denom = max(1, total_effective_t)

    print("=" * 64)
    print(f"Dataset check: {data_root}")
    print("=" * 64)
    print(f"index rows (rollouts)        : {n_rollouts}")
    print(f"npz files                    : {len(npz_paths)}")
    print(f"scene count                  : {n_scenes}")
    print(f"scene ids                    : {scene_ids}")
    print(f"sum(T) from index            : {total_index_t}")
    print(f"sum(effective T on disk)     : {total_effective_t}")
    print(f"missing rollout files        : {missing}")
    print(f"positive ratio 0.5s (index)  : {100.0 * pos05 / denom:.2f}%")
    print(f"positive ratio 1.0s (index)  : {100.0 * pos1 / denom:.2f}%")
    print(f"positive ratio 2.0s (index)  : {100.0 * pos2 / denom:.2f}%")

    # Sample-level stats with the same indexing logic as training.
    from create_dataset_module.risk_dataset import RiskDataset, scene_stratified_split

    ratios = _parse_ratios(args.split_ratios)
    tr_s, va_s, te_s = scene_stratified_split(scene_ids, ratios, seed=int(args.seed))
    print("\nSplit (scene_stratified_split)")
    print(f"  ratios                     : {ratios}")
    print(f"  train scenes ({len(tr_s):2d})           : {tr_s}")
    print(f"  val scenes   ({len(va_s):2d})           : {va_s}")
    print(f"  test scenes  ({len(te_s):2d})           : {te_s}")

    def _build_split_ds(scenes: Sequence[int]) -> RiskDataset:
        return RiskDataset(
            root=str(data_root),
            T_ctx=int(args.T_ctx),
            scene_filter=scenes,
            traj_horizon=int(args.traj_horizon),
            include_action_seq=False,
            include_ego_vel_seq=False,
        )

    tr_ds = _build_split_ds(tr_s)
    va_ds = _build_split_ds(va_s)
    te_ds = _build_split_ds(te_s) if te_s else None
    print("\nSample counts (after T_ctx / traj_horizon windowing)")
    print(f"  train samples              : {len(tr_ds)}")
    print(f"  val samples                : {len(va_ds)}")
    print(f"  test samples               : {len(te_ds) if te_ds is not None else 0}")

    _print_horizon_stats("train", tr_ds.horizon_label_stats())
    _print_horizon_stats("val", va_ds.horizon_label_stats())
    if te_ds is not None:
        _print_horizon_stats("test", te_ds.horizon_label_stats())
    print("\nDone.")


if __name__ == "__main__":
    main()
