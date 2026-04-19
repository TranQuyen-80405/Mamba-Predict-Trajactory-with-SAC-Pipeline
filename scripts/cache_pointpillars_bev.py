"""
Precompute frozen PointPillars neck BEV features per trajectory frame.

Usage (repo root):
  python scripts/cache_pointpillars_bev.py --data_root data/stage_a_experiment \
      --ckpt PointPillars_module/pretrained/epoch_160.pth \
      --out_root data/stage_a_experiment_bev_cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_PP = _ROOT / "PointPillars_module"
for _p in (_ROOT, _PP, _ROOT / "create_dataset_module"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from create_dataset_module.risk_dataset import (  # noqa: E402
    _default_preprocess_cfg,
    _preprocess_depth_to_pts,
    bev_cache_relpath,
)
from data_contracts import Trajectory  # noqa: E402
from module_pointpillar import (  # noqa: E402
    CameraToLidarExtrinsics,
    DepthCameraIntrinsics,
    PointPillarsConfig,
    PointPillarsNeckExtractor,
)


def _load_index_rows(data_root: Path) -> List[Dict]:
    idx = data_root / "index.jsonl"
    if not idx.is_file():
        raise FileNotFoundError(f"index.jsonl not found: {idx}")
    rows: List[Dict] = []
    with idx.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute existing cached frames.",
    )
    ap.add_argument(
        "--save_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Storage dtype for cached BEV tensors.",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.save_dtype == "float16" else torch.float32

    rows = _load_index_rows(data_root)
    unique_traj = sorted({str(r["path"]) for r in rows})
    if not unique_traj:
        raise ValueError("index.jsonl has no rows.")

    pp_cfg = PointPillarsConfig(ckpt_path=args.ckpt, device=dev)
    extractor = PointPillarsNeckExtractor(pp_cfg)
    extractor.freeze_all()
    preprocess_cfg = _default_preprocess_cfg()

    n_saved = 0
    n_skipped = 0
    for i, rel in enumerate(unique_traj):
        traj_path = data_root / rel
        traj = Trajectory.from_npz(traj_path)
        fx, fy, cx, cy = traj.cam_intrinsics.tolist()
        H_img, W_img = traj.depth.shape[1], traj.depth.shape[2]
        intr = DepthCameraIntrinsics(
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            width=int(W_img),
            height=int(H_img),
            near=0.1,
            far=float(max(1.0, preprocess_cfg.max_range)),
        )
        print(f"[{i+1}/{len(unique_traj)}] cache {rel} (T={traj.T})", flush=True)

        for tau in range(int(traj.T)):
            rel_bev = bev_cache_relpath(rel, tau)
            out_path = out_root / rel_bev
            if out_path.is_file() and not args.overwrite:
                n_skipped += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)

            depth_m = traj.depth[tau].astype("float32")
            extr = CameraToLidarExtrinsics(
                R=traj.cam_extr_R[tau].astype("float32"),
                t=traj.cam_extr_t[tau].astype("float32"),
                convention="identity",
            )
            pts = _preprocess_depth_to_pts(depth_m, intr, extr, preprocess_cfg)
            neck = extractor.extract_neck([pts])
            bev = neck.feature.squeeze(0).detach().cpu().to(dtype).contiguous()
            torch.save(bev, out_path)
            n_saved += 1

    summary = {
        "data_root": str(data_root),
        "out_root": str(out_root),
        "ckpt": str(Path(args.ckpt).resolve()),
        "device": dev,
        "save_dtype": args.save_dtype,
        "n_trajectories": len(unique_traj),
        "n_saved_frames": n_saved,
        "n_skipped_existing": n_skipped,
    }
    with (out_root / "cache_meta.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"Done. saved={n_saved} skipped={n_skipped} meta={out_root / 'cache_meta.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
