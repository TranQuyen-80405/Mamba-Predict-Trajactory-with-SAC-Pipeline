"""
Train / compare Stage A temporal backbones (mamba, gru, lstm, transformer).

Delegates each run to ``training.stage_a_single_run.run_stage_a_training`` (same artifacts:
TensorBoard, ``metrics.jsonl``, ``val_metrics_final.json``, checkpoints).

For **one backbone per script**, prefer:

* ``train_stage_a_mamba.py``
* ``train_stage_a_rnn_gru.py``
* ``train_stage_a_lstm.py``
* ``train_stage_a_transformer.py``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Union

_PKG_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PKG_ROOT.parent
for _p in (_PKG_ROOT, _REPO_ROOT, _REPO_ROOT / "create_dataset_module"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

if os.environ.get("TORCH_SHARING_STRATEGY", "").lower() == "file_system":
    mp.set_sharing_strategy("file_system")

from training.stage_a_single_run import (  # noqa: E402
    run_stage_a_training,
    validate_backbone,
    validate_stage,
)


def _parse_models(s: str) -> List[str]:
    return [validate_backbone(x.strip()) for x in s.split(",") if x.strip()]


def run_experiment(
    *,
    data_root: str,
    ckpt_path: str,
    models: List[str],
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 5e-5,
    weight_decay: float = 0.05,
    log_root: str = "runs/stage_a_compare",
    device: Union[str, None] = None,
    seed: int = 0,
    num_workers: int = 4,
    T_ctx: int = 40,
    gradient_accumulation_steps: int = 2,
    lr_warmup_iters: int = 500,
    cosine_eta_min: float = 1e-6,
    positive_oversample_weight: float = 8.0,
    traj_horizon: int = 10,
    traj_loss_weight: float = 0.5,
    grad_clip: float = 1.0,
    mamba_backend: str = "auto",
    weights_dir: Union[str, None] = None,
    save_weights: bool = True,
    stage: str = "a1",
    lr_neck: float = 5e-6,
    lr_rest: float = 1.67e-5,
    early_stop_patience: int = 8,
    early_stop_min_delta: float = 0.0,
    bev_cache_root: Union[str, None] = None,
    extrinsics_convention: str = "auto",
    depth_scale_factor: float = 1.0,
    risk_label_smoothing: float = 0.05,
    temporal_dropout: float = 0.1,
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Train each requested temporal backbone with the same data split and hyperparameters.

    Returns:
        Mapping model_name -> metrics (floats) plus paths from ``run_stage_a_training``.
    """
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_env > 1 and dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    is_dist = dist.is_available() and dist.is_initialized()
    is_main = (not is_dist) or (dist.get_rank() == 0)
    results: Dict[str, Dict[str, Union[float, str]]] = {}
    log_dir_base = Path(log_root)
    if is_main:
        log_dir_base.mkdir(parents=True, exist_ok=True)
    if is_dist:
        dist.barrier()
    wdir = weights_dir or str(log_dir_base / "checkpoints")
    ph = validate_stage(stage)

    for name in models:
        row = run_stage_a_training(
            backbone=name,
            data_root=data_root,
            ckpt_path=ckpt_path,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            log_root=log_root,
            device=device,
            seed=seed,
            num_workers=num_workers,
            positive_oversample_weight=positive_oversample_weight,
            T_ctx=T_ctx,
            traj_horizon=traj_horizon,
            traj_loss_weight=traj_loss_weight,
            grad_clip=grad_clip,
            gradient_accumulation_steps=gradient_accumulation_steps,
            cosine_eta_min=cosine_eta_min,
            mamba_backend=mamba_backend,
            weights_dir=wdir,
            save_weights=save_weights,
            run_name=None,
            stage=ph,
            lr_neck=lr_neck,
            lr_rest=lr_rest,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            lr_warmup_iters=lr_warmup_iters,
            bev_cache_root=bev_cache_root,
            extrinsics_convention=extrinsics_convention,
            depth_scale_factor=depth_scale_factor,
            risk_label_smoothing=risk_label_smoothing,
            temporal_dropout=temporal_dropout,
        )
        results[name] = row

    if is_main:
        print("\n=== Risk (validation, best-val snapshot) ===", flush=True)
        print(
            f"{'model':<14} {'AP@0.5s':>9} {'AP@1s':>9} {'AP@2s':>9} {'AUC@1s':>9} "
            f"{'lat_ms':>10}",
            flush=True,
        )
        for k, m in results.items():
            lat = float(m.get("val_inference_ms_per_sample", float("nan")))
            lat_s = f"{lat:10.3f}" if not np.isnan(lat) else f"{'n/a':>10}"
            print(
                f"{k:<14} "
                f"{m.get('ap_risk_05s', float('nan')):9.4f} "
                f"{m.get('ap_risk_1s', float('nan')):9.4f} "
                f"{m.get('ap_risk_2s', float('nan')):9.4f} "
                f"{m.get('auc_risk_1s', float('nan')):9.4f} "
                f"{lat_s}",
                flush=True,
            )

        print("\n=== Trajectory forecast (validation, best-val snapshot) ===", flush=True)
        print(
            f"{'model':<14} {'RMSE_all':>10} {'ADE_xy_m':>10} {'FDE_xy_m':>10} {'RMSE_yaw':>10}",
            flush=True,
        )
        for k, m in results.items():
            print(
                f"{k:<14} "
                f"{m.get('traj_rmse_all', float('nan')):10.4f} "
                f"{m.get('traj_ade_xy_m', float('nan')):10.4f} "
                f"{m.get('traj_fde_xy_m', float('nan')):10.4f} "
                f"{m.get('traj_rmse_yaw_rad', float('nan')):10.4f}",
                flush=True,
            )
        print(f"\nTensorBoard: tensorboard --logdir {log_dir_base}", flush=True)

        summary_path = log_dir_base / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {summary_path}", flush=True)

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage A temporal backbone comparison")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument(
        "--ckpt",
        type=str,
        default=str(_PKG_ROOT / "pretrained" / "epoch_160_raw.pth"),
    )
    ap.add_argument(
        "--models",
        type=str,
        default="mamba,gru,lstm,transformer",
        help="Comma-separated: mamba,gru,lstm,transformer",
    )
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
    ap.add_argument(
        "--T_ctx",
        type=int,
        default=40,
        help="Context length in frames (20≈1.0s, 40≈2.0s at 20 Hz; default 40 matches 2s horizon).",
    )
    ap.add_argument(
        "--lr_warmup_iters",
        type=int,
        default=500,
        help="Optimizer-step LR warmup before cosine decay (§ 5.3).",
    )
    ap.add_argument("--positive_weight", type=float, default=8.0)
    ap.add_argument("--traj_horizon", type=int, default=10)
    ap.add_argument(
        "--traj_loss_weight",
        type=float,
        default=0.5,
        help="Multiplier for SmoothL1 trajectory loss vs focal risk loss.",
    )
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--mamba_backend", type=str, default="auto")
    ap.add_argument(
        "--weights_dir",
        type=str,
        default="",
        help="Directory for .pt checkpoints (default: <log_root>/checkpoints).",
    )
    ap.add_argument(
        "--no_save_weights",
        action="store_true",
        help="Disable writing stage_a_compare_*.pt files.",
    )
    ap.add_argument(
        "--stage",
        type=str,
        choices=["a1", "a2"],
        default="a1",
        help="a2: fine-tune pp.neck on sim data (two LRs).",
    )
    ap.add_argument("--lr_neck", type=float, default=5e-6)
    ap.add_argument("--lr_rest", type=float, default=1.67e-5)
    ap.add_argument("--early_stop_patience", type=int, default=8)
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0)
    ap.add_argument(
        "--bev_cache_root",
        type=str,
        default="",
        help=(
            "Precomputed BEV cache root. If omitted, defaults to "
            "<data_root>_bev_cache and is required by default workflow."
        ),
    )
    ap.add_argument(
        "--extrinsics_convention",
        type=str,
        default="auto",
        choices=["auto", "pybullet_to_kitti", "opencv_to_kitti", "identity", "from_trajectory"],
        help=(
            "Depth->LiDAR convention for Stage A dataset conversion. "
            "'auto' prefers per-frame trajectory extrinsics when present."
        ),
    )
    ap.add_argument(
        "--depth_scale_factor",
        type=float,
        default=1.0,
        help="Scale multiplier for depth-derived xyz points before PointPillars.",
    )
    ap.add_argument(
        "--risk_label_smoothing",
        type=float,
        default=0.05,
        help="Label smoothing for focal BCE on risk heads.",
    )
    ap.add_argument(
        "--temporal_dropout",
        type=float,
        default=0.1,
        help="Dropout on temporal encoder stacks.",
    )
    args = ap.parse_args()

    run_experiment(
        data_root=args.data_root,
        ckpt_path=args.ckpt,
        models=_parse_models(args.models),
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
        traj_loss_weight=args.traj_loss_weight,
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


if __name__ == "__main__":
    main()
