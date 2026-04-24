"""
Stage A — temporal backbone **Mamba** (mamba-ssm stack inside ``MambaTemporal``).

Pipeline: ``data_root`` (``index.jsonl`` + rollouts) → train risk + trajectory → validate.

Logs: TensorBoard + ``metrics.jsonl`` + ``val_metrics_final.json`` under ``log_root/<run_name>/``.

Example (from repo root)::

    python PointPillars_module/training/train_stage_a_mamba.py --data_root data/stage_a_experiment \\
        --ckpt PointPillars_module/pretrained/epoch_160.pth --epochs 3 --log_root runs/stage_a
"""

from __future__ import annotations

import sys
from pathlib import Path

_pp = Path(__file__).resolve().parent.parent
if str(_pp) not in sys.path:
    sys.path.insert(0, str(_pp))

from training.train_stage_a_compare import run_experiment


class _TeeStream:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stage A mamba-only runner")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument(
        "--ckpt",
        type=str,
        default=str(_pp / "pretrained" / "epoch_160_raw.pth"),
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-microbatch size. With long T_ctx+BEV, 32 can OOM ~24–32GB; use 4–8 + AMP, or --gradient_accumulation_steps to match effective batch.",
    )
    ap.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Default 8 * batch 8 = 64 effective (similar to old 32*2) with lower peak VRAM.",
    )
    ap.add_argument("--cosine_eta_min", type=float, default=1e-6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--log_root", type=str, default="runs/stage_a_compare")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers (spawn + file_system sharing). 0 = most reliable. If workers crash: "
            "Docker --shm-size=8g+; do not set TORCH_SHARING_STRATEGY=file_descriptor on small /dev/shm. "
            "Code caps OMP/BLAS threads=1 in each worker. Try 1 then 2–4 if stable."
        ),
    )
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
    ap.add_argument("--progress_log_every", type=int, default=50)
    ap.add_argument(
        "--append_train_log",
        action="store_true",
        help="Append to log_root/train.log instead of overwriting (default: overwrite each run).",
    )
    ap.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
        help="If >0, stop each epoch after N micro-batches (smoke/debug). 0=full epoch.",
    )
    ap.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable CUDA mixed precision (uses more VRAM; only if training is unstable in fp16).",
    )
    args = ap.parse_args()
    _mtb = int(args.max_train_batches) if int(args.max_train_batches) > 0 else None
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    train_log_path = log_root / "train.log"
    _log_mode = "a" if args.append_train_log else "w"
    with train_log_path.open(_log_mode, encoding="utf-8") as log_f:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(old_out, log_f)
        sys.stderr = _TeeStream(old_err, log_f)
        try:
            run_experiment(
                data_root=args.data_root,
                ckpt_path=args.ckpt,
                models=["mamba"],
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
                progress_log_every=args.progress_log_every,
                max_train_batches=_mtb,
                use_amp=(not bool(args.no_amp)),
            )
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
