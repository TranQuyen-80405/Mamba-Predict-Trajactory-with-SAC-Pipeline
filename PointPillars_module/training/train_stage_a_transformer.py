"""
Stage A — **Transformer** encoder over the token sequence (``TransformerEncoderTemporal``).

Example (from repo root)::

    python PointPillars_module/training/train_stage_a_transformer.py --data_root data/stage_a_experiment \\
        --ckpt PointPillars_module/pretrained/epoch_160.pth
"""

from __future__ import annotations

import sys
from pathlib import Path

_pp = Path(__file__).resolve().parent.parent
if str(_pp) not in sys.path:
    sys.path.insert(0, str(_pp))

from training.train_stage_a_compare import run_experiment

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stage A transformer-only runner")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=str(_pp / "pretrained" / "epoch_160_raw.pth"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--cosine_eta_min", type=float, default=1e-6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--log_root", type=str, default="runs/stage_a_compare")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--T_ctx", type=int, default=40)
    ap.add_argument("--lr_warmup_iters", type=int, default=500)
    ap.add_argument("--positive_weight", type=float, default=8.0)
    ap.add_argument("--traj_horizon", type=int, default=10)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--mamba_backend", type=str, default="auto")
    ap.add_argument("--weights_dir", type=str, default="")
    ap.add_argument("--no_save_weights", action="store_true")
    ap.add_argument("--stage", type=str, choices=["a1", "a2"], default="a1")
    ap.add_argument("--lr_neck", type=float, default=5e-6)
    ap.add_argument("--lr_rest", type=float, default=1.67e-5)
    ap.add_argument("--early_stop_patience", type=int, default=8)
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0)
    ap.add_argument("--bev_cache_root", type=str, default="")
    ap.add_argument(
        "--extrinsics_convention",
        type=str,
        default="auto",
        choices=["auto", "pybullet_to_kitti", "opencv_to_kitti", "identity", "from_trajectory"],
    )
    ap.add_argument("--depth_scale_factor", type=float, default=1.0)
    ap.add_argument("--risk_label_smoothing", type=float, default=0.05)
    ap.add_argument("--temporal_dropout", type=float, default=0.1)
    args = ap.parse_args()
    run_experiment(
        data_root=args.data_root,
        ckpt_path=args.ckpt,
        models=["transformer"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_root=args.log_root,
        device=args.device or None,
        seed=args.seed,
        num_workers=args.num_workers,
        positive_oversample_weight=args.positive_weight,
        T_ctx=args.T_ctx,
        traj_horizon=args.traj_horizon,
        grad_clip=args.grad_clip,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        cosine_eta_min=args.cosine_eta_min,
        mamba_backend=args.mamba_backend,
        weights_dir=(args.weights_dir or None),
        save_weights=(not args.no_save_weights),
        stage=args.stage,
        lr_neck=args.lr_neck,
        lr_rest=args.lr_rest,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        lr_warmup_iters=args.lr_warmup_iters,
        bev_cache_root=(args.bev_cache_root or None),
        extrinsics_convention=args.extrinsics_convention,
        depth_scale_factor=args.depth_scale_factor,
        risk_label_smoothing=args.risk_label_smoothing,
        temporal_dropout=args.temporal_dropout,
    )
