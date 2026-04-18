"""
Train / compare Stage A temporal backbones (mamba, gru, lstm, transformer).

Designed for:
  * CLI:  python train_stage_a_compare.py --data_root ... --ckpt ...
  * Colab: ``from train_stage_a_compare import run_experiment``

Logs TensorBoard scalars under ``log_root/<run_name>/`` and prints summary tables for:

  * **Risk:** AP / AUC per horizon (0.5s / 1s / 2s).
  * **Trajectory:** RMSE (all dims), ADE / FDE in XY (m), RMSE yaw — vs ground-truth
    future ``ego_state`` planar poses (H frames).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

_PKG_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_ROOT.parent
for _p in (_PKG_ROOT, _REPO_ROOT, _REPO_ROOT / "create_dataset_module"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from losses import focal_bce  # noqa: E402
from module_pointpillar import PointPillarsConfig, PointPillarsNeckExtractor  # noqa: E402
from models.full_pipeline_risk_traj import FullPipelineRiskAndTraj  # noqa: E402
from models.temporal_factory import build_temporal  # noqa: E402
from risk_dataset import RiskDataset, collate_riskbatch, scene_stratified_split  # noqa: E402
from utils.gradient_health import grad_norm_l2, max_grad_value  # noqa: E402


def _cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def _unique_scenes_from_index(data_root: Path) -> List[int]:
    index_file = data_root / "index.jsonl"
    scenes: set[int] = set()
    with index_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scenes.add(int(row["scene_id"]))
    return sorted(scenes)


def _traj_error_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """pred, gt: (N, H, 3) with (x, y, yaw). Yaw RMSE in radians."""
    err = pred.astype(np.float64) - gt.astype(np.float64)
    rmse_all = float(np.sqrt(np.mean(err ** 2)))
    dxy = np.linalg.norm(err[:, :, :2], axis=-1)
    ade_xy = float(np.mean(np.mean(dxy, axis=1)))
    fde_xy = float(np.mean(dxy[:, -1]))
    rmse_yaw = float(np.sqrt(np.mean(err[:, :, 2] ** 2)))
    return {
        "traj_rmse_all": rmse_all,
        "traj_ade_xy_m": ade_xy,
        "traj_fde_xy_m": fde_xy,
        "traj_rmse_yaw_rad": rmse_yaw,
    }


def _collect_val_metrics(
    model: FullPipelineRiskAndTraj,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.reducer.eval()
    model.mamba.eval()
    model.head.eval()
    model.traj_head.eval()
    logits_all: List[torch.Tensor] = []
    t_all: List[torch.Tensor] = []
    traj_pred_all: List[torch.Tensor] = []
    traj_gt_all: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            pts = [
                [p.to(device, non_blocking=True) for p in frame]
                for frame in batch.pts_seq
            ]
            logits, traj_pred = model(pts)
            logits_all.append(logits.cpu())
            targets = torch.stack(
                [batch.risk_05s, batch.risk_1s, batch.risk_2s], dim=1
            ).float()
            t_all.append(targets)
            traj_pred_all.append(traj_pred.cpu())
            traj_gt_all.append(batch.traj_future_xyyaw.float())

    logits_cat = torch.cat(logits_all, dim=0).numpy()
    t_cat = torch.cat(t_all, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_cat))

    pred_traj = torch.cat(traj_pred_all, dim=0).numpy()
    gt_traj = torch.cat(traj_gt_all, dim=0).numpy()
    out: Dict[str, float] = _traj_error_metrics(pred_traj, gt_traj)

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        return out

    names = ["risk_05s", "risk_1s", "risk_2s"]
    for i, name in enumerate(names):
        y = t_cat[:, i]
        if y.max() == y.min():
            out[f"ap_{name}"] = float("nan")
            out[f"auc_{name}"] = float("nan")
            continue
        out[f"ap_{name}"] = float(average_precision_score(y, probs[:, i]))
        try:
            out[f"auc_{name}"] = float(roc_auc_score(y, probs[:, i]))
        except ValueError:
            out[f"auc_{name}"] = float("nan")
    return out


def run_experiment(
    *,
    data_root: str,
    ckpt_path: str,
    models: Sequence[str] = ("mamba", "gru", "lstm", "transformer"),
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    log_root: str = "runs/stage_a_compare",
    device: Optional[str] = None,
    seed: int = 0,
    num_workers: int = 0,
    positive_oversample_weight: float = 8.0,
    train_val_test: Tuple[float, float, float] = (0.75, 0.25, 0.0),
    T_ctx: int = 10,
    traj_horizon: int = 10,
    traj_loss_weight: float = 0.5,
    grad_clip: float = 1.0,
    mamba_backend: str = "auto",
    weights_dir: Optional[str] = None,
    save_weights: bool = True,
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Train each requested temporal backbone with the same data split and hyperparameters.

    Joint objective: focal BCE on risk horizons + SmoothL1 on future planar trajectory.

    Returns:
        Mapping model_name -> metrics (floats) plus ``checkpoint_pt`` (path to ``.pt``)
        when ``save_weights`` is True.
    """
    data_path = Path(data_root)
    if not data_path.is_dir():
        raise FileNotFoundError(f"data_root is not a directory: {data_path}")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type != "cuda":
        print(
            "[train_stage_a_compare] Warning: PointPillars voxel path expects CUDA; "
            "CPU may fail or be extremely slow.",
            flush=True,
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    scenes = _unique_scenes_from_index(data_path)
    if len(scenes) < 2:
        raise ValueError(f"need at least 2 scenes for split; got {len(scenes)}")

    tr_s, va_s, _ = scene_stratified_split(scenes, train_val_test, seed=seed)
    if not tr_s or not va_s:
        raise ValueError(f"empty train or val split: train={tr_s} val={va_s}")

    train_ds = RiskDataset(
        str(data_path), T_ctx=T_ctx, scene_filter=tr_s, traj_horizon=traj_horizon
    )
    val_ds = RiskDataset(
        str(data_path), T_ctx=T_ctx, scene_filter=va_s, traj_horizon=traj_horizon
    )

    w_arr = train_ds.risk_1s_array()
    weights = torch.as_tensor(
        np.where(w_arr > 0.5, positive_oversample_weight, 1.0),
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(
        weights, num_samples=len(train_ds), replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_riskbatch,
        pin_memory=(dev.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_riskbatch,
        pin_memory=(dev.type == "cuda"),
    )

    results: Dict[str, Dict[str, Union[float, str]]] = {}
    log_dir_base = Path(log_root)
    log_dir_base.mkdir(parents=True, exist_ok=True)
    wdir = (Path(weights_dir) if weights_dir else log_dir_base / "weights")

    for name in models:
        name = str(name).lower().strip()
        run_name = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(str(log_dir_base / run_name))

        cfg = PointPillarsConfig(ckpt_path=ckpt_path, device=str(dev))
        pp = PointPillarsNeckExtractor(cfg)
        temporal = build_temporal(name, d_model=256, n_blocks=2, mamba_backend=mamba_backend)
        model = FullPipelineRiskAndTraj(
            pp, mamba=temporal, token_dim=256, traj_horizon=traj_horizon
        )
        model.to(dev)
        model.pp.freeze_all()

        trainable = (
            list(model.reducer.parameters())
            + list(model.mamba.parameters())
            + list(model.head.parameters())
            + list(model.traj_head.parameters())
        )
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        traj_criterion = torch.nn.SmoothL1Loss(beta=0.05)

        global_step = 0
        best_ap = -1.0
        for epoch in range(epochs):
            model.reducer.train()
            model.mamba.train()
            model.head.train()
            model.traj_head.train()
            epoch_loss = 0.0
            n_batches = 0
            epoch_max_grad_pre = 0.0
            epoch_max_grad_abs_post = 0.0
            for batch in train_loader:
                opt.zero_grad(set_to_none=True)
                pts = [
                    [p.to(dev, non_blocking=True) for p in frame]
                    for frame in batch.pts_seq
                ]
                logits, traj_hat = model(pts)
                targets = torch.stack(
                    [batch.risk_05s, batch.risk_1s, batch.risk_2s], dim=1
                ).to(dev)
                loss_r = focal_bce(logits, targets)
                traj_gt = batch.traj_future_xyyaw.to(dev)
                loss_t = traj_criterion(traj_hat, traj_gt)
                loss = loss_r + traj_loss_weight * loss_t
                loss.backward()
                gn_pre = float(grad_norm_l2(trainable))
                epoch_max_grad_pre = max(epoch_max_grad_pre, gn_pre)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                gn_post = float(grad_norm_l2(trainable))
                mg_post = float(max_grad_value(trainable))
                epoch_max_grad_abs_post = max(epoch_max_grad_abs_post, mg_post)
                opt.step()

                epoch_loss += float(loss.detach())
                n_batches += 1
                writer.add_scalar("train/loss_step", float(loss), global_step)
                writer.add_scalar("train/loss_risk", float(loss_r.detach()), global_step)
                writer.add_scalar("train/loss_traj", float(loss_t.detach()), global_step)
                writer.add_scalar("train/grad_norm_pre_clip", gn_pre, global_step)
                writer.add_scalar("train/grad_norm_post_clip", gn_post, global_step)
                writer.add_scalar("train/max_grad_abs_post_clip", mg_post, global_step)
                global_step += 1

            writer.add_scalar("train/loss_epoch", epoch_loss / max(1, n_batches), epoch)
            writer.add_scalar("train/epoch_max_grad_norm_pre_clip", epoch_max_grad_pre, epoch)
            writer.add_scalar(
                "train/epoch_max_grad_abs_post_clip", epoch_max_grad_abs_post, epoch
            )

            metrics = _collect_val_metrics(model, val_loader, dev)
            for k, v in metrics.items():
                if not np.isnan(v):
                    writer.add_scalar(f"val/{k}", v, epoch)
            ap1 = metrics.get("ap_risk_1s", float("nan"))
            if not np.isnan(ap1):
                best_ap = max(best_ap, ap1)

            ade = metrics.get("traj_ade_xy_m", float("nan"))
            print(
                f"[{run_name}] epoch {epoch + 1}/{epochs} "
                f"train_loss={epoch_loss / max(1, n_batches):.4f} "
                f"val_ap_risk_1s={ap1:.4f} val_ADE_xy={ade:.4f}",
                flush=True,
            )

        writer.add_scalar("val/best_ap_risk_1s", best_ap, 0)
        writer.close()

        row: Dict[str, Union[float, str]] = {k: float(v) for k, v in metrics.items()}
        if save_weights:
            wdir.mkdir(parents=True, exist_ok=True)
            ts = run_name.split("_", 1)[-1] if "_" in run_name else time.strftime("%Y%m%d_%H%M%S")
            out_ts = wdir / f"{name}_stage_a_compare_{ts}.pt"
            out_latest = wdir / f"{name}_stage_a_compare.pt"
            payload = {
                "format": "stage_a_compare_full_pipeline_risk_traj_v1",
                "backbone": name,
                "run_name": run_name,
                "traj_horizon": traj_horizon,
                "T_ctx": T_ctx,
                "epochs": epochs,
                "pp_ckpt_path": str(Path(ckpt_path).resolve()),
                "data_root": str(data_path.resolve()),
                "val_metrics": dict(row),
                "model_state_dict": _cpu_state_dict(model),
            }
            torch.save(payload, out_ts)
            torch.save(payload, out_latest)
            row["checkpoint_pt"] = str(out_ts.resolve())
            row["checkpoint_pt_latest"] = str(out_latest.resolve())
            print(
                f"[{run_name}] Wrote checkpoints {out_ts.name} + {out_latest.name} "
                f"(under {wdir})",
                flush=True,
            )
        results[name] = row

    print("\n=== Risk (validation, last epoch) ===", flush=True)
    hdr = f"{'model':<14} {'AP@0.5s':>9} {'AP@1s':>9} {'AP@2s':>9} {'AUC@1s':>9}"
    print(hdr, flush=True)
    for k, m in results.items():
        print(
            f"{k:<14} "
            f"{m.get('ap_risk_05s', float('nan')):9.4f} "
            f"{m.get('ap_risk_1s', float('nan')):9.4f} "
            f"{m.get('ap_risk_2s', float('nan')):9.4f} "
            f"{m.get('auc_risk_1s', float('nan')):9.4f}",
            flush=True,
        )

    print("\n=== Trajectory forecast (validation, last epoch) ===", flush=True)
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

    summary_path = log_dir_base / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {summary_path}", flush=True)

    return results


def _parse_models(s: str) -> List[str]:
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage A temporal backbone comparison")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument(
        "--ckpt",
        type=str,
        default=str(_PKG_ROOT / "pretrained" / "epoch_160.pth"),
    )
    ap.add_argument(
        "--models",
        type=str,
        default="mamba,gru,lstm,transformer",
        help="Comma-separated: mamba,gru,lstm,transformer",
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--log_root", type=str, default="runs/stage_a_compare")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--positive_weight", type=float, default=8.0)
    ap.add_argument("--traj_horizon", type=int, default=10)
    ap.add_argument(
        "--traj_loss_weight",
        type=float,
        default=0.5,
        help="Multiplier for SmoothL1 trajectory loss vs focal risk loss.",
    )
    ap.add_argument("--mamba_backend", type=str, default="auto")
    ap.add_argument(
        "--weights_dir",
        type=str,
        default="",
        help="Directory for .pt checkpoints (default: <log_root>/weights).",
    )
    ap.add_argument(
        "--no_save_weights",
        action="store_true",
        help="Disable writing stage_a_compare_*.pt files.",
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
        traj_horizon=args.traj_horizon,
        traj_loss_weight=args.traj_loss_weight,
        mamba_backend=args.mamba_backend,
        weights_dir=(args.weights_dir or None),
        save_weights=(not args.no_save_weights),
    )


if __name__ == "__main__":
    main()
