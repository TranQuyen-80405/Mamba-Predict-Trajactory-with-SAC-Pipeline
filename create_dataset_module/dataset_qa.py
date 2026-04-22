from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np

from PointPillars_module.types import Trajectory


@dataclass
class VerifyDataSanityResult:
    n_files: int
    n_frames: int
    positive_ratio_1s: float
    depth_max_observed: float
    low_motion_warnings: List[str]


def _risk_order_violations(r05: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> int:
    a = np.asarray(r05, dtype=np.float32).ravel()
    b = np.asarray(r1, dtype=np.float32).ravel()
    c = np.asarray(r2, dtype=np.float32).ravel()
    n = min(len(a), len(b), len(c))
    bad = 0
    for i in range(n):
        if a[i] > b[i] + 1e-5 or b[i] > c[i] + 1e-5:
            bad += 1
    return bad


def verify_data_sanity(
    dataset_path: Union[str, Path],
    *,
    max_range: float = 8.0,
    min_pos_ratio_1s: float = 0.05,
    max_pos_ratio_1s: float = 0.30,
    check_balance: bool = True,
    min_ego_xy_var: float = 1e-8,
) -> VerifyDataSanityResult:
    root = Path(dataset_path)
    if root.is_file() and root.suffix == ".npz":
        files = [root]
    elif root.is_dir():
        index = root / "index.jsonl"
        files = []
        if index.is_file():
            with index.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    files.append(root / row["path"])
        else:
            files = sorted(root.glob("*.npz"))
    else:
        raise FileNotFoundError(f"Invalid dataset path: {root}")

    if not files:
        raise AssertionError(f"No npz files found under {root}")

    total_frames = 0
    total_pos1 = 0
    depth_max = 0.0
    low_motion: List[str] = []

    for fp in files:
        traj = Trajectory.from_npz(fp)
        d = np.asarray(traj.depth, dtype=np.float32)
        finite = d[np.isfinite(d) & (d > 0)]
        dmax = float(finite.max()) if finite.size else 0.0
        depth_max = max(depth_max, dmax)
        if dmax > max_range + 1e-3:
            raise AssertionError(f"{fp.name}: depth_max={dmax:.4f} > {max_range}")

        bad = _risk_order_violations(traj.risk_05s, traj.risk_1s, traj.risk_2s)
        if bad:
            raise AssertionError(f"{fp.name}: {bad} frames violate risk_05s<=risk_1s<=risk_2s")

        xy = np.asarray(traj.ego_state[:, :2], dtype=np.float64)
        if xy.shape[0] >= 2:
            vxy = 0.5 * (float(np.var(xy[:, 0])) + float(np.var(xy[:, 1])))
            if vxy < min_ego_xy_var:
                low_motion.append(f"{fp.name}: ego_xy_var={vxy:.2e}")

        total_frames += int(traj.T)
        total_pos1 += int((np.asarray(traj.risk_1s) > 0.5).sum())

    pos_ratio = total_pos1 / max(1, total_frames)
    if check_balance and len(files) > 1:
        if not (min_pos_ratio_1s <= pos_ratio <= max_pos_ratio_1s):
            raise AssertionError(
                f"positive_ratio_1s={pos_ratio:.4f} not in [{min_pos_ratio_1s}, {max_pos_ratio_1s}]"
            )

    return VerifyDataSanityResult(
        n_files=len(files),
        n_frames=total_frames,
        positive_ratio_1s=pos_ratio,
        depth_max_observed=depth_max,
        low_motion_warnings=low_motion,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_path")
    ap.add_argument("--max-range", type=float, default=8.0)
    ap.add_argument("--min-pos-1s", type=float, default=0.05)
    ap.add_argument("--max-pos-1s", type=float, default=0.30)
    ap.add_argument("--no-balance-check", action="store_true")
    args = ap.parse_args()

    out = verify_data_sanity(
        args.dataset_path,
        max_range=args.max_range,
        min_pos_ratio_1s=args.min_pos_1s,
        max_pos_ratio_1s=args.max_pos_1s,
        check_balance=not args.no_balance_check,
    )
    msg = (
        f"OK: {out.n_files} file(s), {out.n_frames} frames, "
        f"positive_ratio_1s={out.positive_ratio_1s:.4f}, depth_max={out.depth_max_observed:.3f}m"
    )
    if out.low_motion_warnings:
        msg += f" | {len(out.low_motion_warnings)} low-motion notice(s)"
    print(msg)


if __name__ == "__main__":
    main()
