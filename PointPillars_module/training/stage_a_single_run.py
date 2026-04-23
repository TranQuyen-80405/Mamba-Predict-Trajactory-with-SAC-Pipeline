"""
Single-backbone Stage A training loop used by train_stage_a_compare.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

_PKG_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PKG_ROOT.parent
for _p in (_PKG_ROOT, _REPO_ROOT, _REPO_ROOT / "create_dataset_module"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from losses import focal_bce  # noqa: E402
from module_pointpillar import PointPillarsConfig, PointPillarsNeckExtractor  # noqa: E402
from models.full_pipeline_risk_traj import FullPipelineRiskAndTraj  # noqa: E402
from models.temporal_factory import build_temporal  # noqa: E402
from create_dataset_module.risk_dataset import (  # noqa: E402
    RiskDataset,
    collate_riskbatch,
    scene_stratified_split,
)
from PointPillars_module.types import DepthPreprocessConfig  # noqa: E402

StageAPhase = Literal["a1", "a2"]
STAGE_A_BACKBONES = ("mamba", "gru", "lstm", "transformer")
RISK_FOCAL_WEIGHTS: Tuple[float, float, float] = (1.0, 0.8, 0.5)


def validate_backbone(name: str) -> str:
    k = str(name).lower().strip()
    if k not in STAGE_A_BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; expected one of {STAGE_A_BACKBONES}")
    return k


def validate_stage(phase: str) -> StageAPhase:
    p = str(phase).lower().strip()
    if p not in ("a1", "a2"):
        raise ValueError("stage must be 'a1' or 'a2'")
    return p  # type: ignore[return-value]


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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _to_pts_seq_bt(pts_seq: Union[List[List[torch.Tensor]], torch.Tensor], device: torch.device):
    if torch.is_tensor(pts_seq):
        # [T, B, C, H, W] -> list[T] of list[B] tensors (C,H,W)
        T, B = int(pts_seq.shape[0]), int(pts_seq.shape[1])
        out: List[List[torch.Tensor]] = []
        for t in range(T):
            out.append([pts_seq[t, b].to(device, non_blocking=True) for b in range(B)])
        return out
    return [[p.to(device, non_blocking=True) for p in frame] for frame in pts_seq]


def _traj_error_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    err = pred.astype(np.float64) - gt.astype(np.float64)
    rmse_all = float(np.sqrt(np.mean(err**2)))
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


@torch.no_grad()
def _collect_val_metrics(
    model: FullPipelineRiskAndTraj,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    logits_all: List[torch.Tensor] = []
    t_all: List[torch.Tensor] = []
    valid_all: List[torch.Tensor] = []
    traj_pred_all: List[torch.Tensor] = []
    traj_gt_all: List[torch.Tensor] = []
    infer_ms_total = 0.0
    infer_n = 0
    for batch in loader:
        pts = _to_pts_seq_bt(batch.pts_seq, device)
        if device.type == "cuda":
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start_evt.record()
            logits, traj_pred = model(pts)
            end_evt.record()
            torch.cuda.synchronize(device)
            infer_ms_total += float(start_evt.elapsed_time(end_evt))
        else:
            logits, traj_pred = model(pts)
        infer_n += int(logits.shape[0])
        logits_all.append(logits.cpu())
        targets = torch.stack([batch.risk_05s, batch.risk_1s, batch.risk_2s], dim=1).float()
        t_all.append(targets)
        valid_all.append(batch.risk_label_valid.float().cpu())
        traj_pred_all.append(traj_pred.cpu())
        traj_gt_all.append(batch.traj_future_xyyaw.float())

    if not logits_all:
        return {
            "ap_risk_05s": 0.0,
            "ap_risk_1s": 0.0,
            "ap_risk_2s": 0.0,
            "auc_risk_05s": 0.0,
            "auc_risk_1s": 0.0,
            "auc_risk_2s": 0.0,
            "brier_risk_avg": 0.0,
            "traj_rmse_all": float("nan"),
            "traj_ade_xy_m": float("nan"),
            "traj_fde_xy_m": float("nan"),
            "traj_rmse_yaw_rad": float("nan"),
            "val_inference_ms_per_sample": float("nan"),
        }

    logits_cat = torch.cat(logits_all, dim=0).numpy()
    t_cat = torch.cat(t_all, dim=0).numpy()
    v_cat = torch.cat(valid_all, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_cat))

    pred_traj = torch.cat(traj_pred_all, dim=0).numpy()
    gt_traj = torch.cat(traj_gt_all, dim=0).numpy()
    out: Dict[str, float] = _traj_error_metrics(pred_traj, gt_traj)

    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    except Exception:
        out["val_inference_ms_per_sample"] = float(infer_ms_total / max(1, infer_n))
        return out

    names = ["risk_05s", "risk_1s", "risk_2s"]
    brier_scores: List[float] = []
    for i, name in enumerate(names):
        m = v_cat[:, i] > 0.5
        if not np.any(m):
            out[f"ap_{name}"] = 0.0
            out[f"auc_{name}"] = 0.0
            continue
        y = t_cat[m, i]
        p = probs[m, i]
        if np.any(y > 0.5):
            brier_scores.append(float(brier_score_loss(y, p)))
        if y.max() == y.min():
            out[f"ap_{name}"] = 0.0
            out[f"auc_{name}"] = 0.0
            continue
        out[f"ap_{name}"] = float(average_precision_score(y, p))
        try:
            out[f"auc_{name}"] = float(roc_auc_score(y, p))
        except ValueError:
            out[f"auc_{name}"] = 0.0
    out["brier_risk_avg"] = float(np.mean(brier_scores)) if brier_scores else 0.0
    out["val_inference_ms_per_sample"] = float(infer_ms_total / max(1, infer_n))
    return out


def run_stage_a_training(
    *,
    backbone: str,
    data_root: str,
    ckpt_path: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 5e-5,
    weight_decay: float = 0.05,
    log_root: str = "runs/stage_a_compare",
    device: Optional[str] = None,
    seed: int = 0,
    num_workers: int = 0,
    train_val_test: Tuple[float, float, float] = (0.75, 0.25, 0.0),
    positive_oversample_weight: float = 8.0,
    T_ctx: int = 40,
    traj_horizon: int = 10,
    traj_loss_weight: float = 0.5,
    grad_clip: float = 1.0,
    gradient_accumulation_steps: int = 2,
    cosine_eta_min: float = 1e-6,
    mamba_backend: str = "auto",
    weights_dir: Optional[str] = None,
    save_weights: bool = True,
    run_name: Optional[str] = None,
    stage: StageAPhase = "a1",
    lr_neck: float = 5e-6,
    lr_rest: float = 1.67e-5,
    early_stop_patience: int = 8,
    early_stop_min_delta: float = 0.0,
    lr_warmup_iters: int = 500,
    bev_cache_root: Optional[str] = None,
    extrinsics_convention: str = "auto",
    depth_scale_factor: float = 1.0,
    risk_label_smoothing: float = 0.05,
    temporal_dropout: float = 0.1,
) -> Dict[str, Union[float, str]]:
    backbone = validate_backbone(backbone)
    phase = validate_stage(stage)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_path = Path(data_root)
    if bev_cache_root is None or not str(bev_cache_root).strip():
        auto_cache = data_path.parent / f"{data_path.name}_bev_cache"
        bev_cache_root = str(auto_cache.resolve()) if auto_cache.is_dir() else None
    if not bev_cache_root:
        raise FileNotFoundError("BEV cache is required but not found.")

    scenes = _unique_scenes_from_index(data_path)
    tr_s, va_s, _ = scene_stratified_split(scenes, train_val_test, seed=seed)
    if not tr_s or not va_s:
        raise ValueError(f"empty train/val split: train={tr_s} val={va_s}")

    preprocess_cfg = DepthPreprocessConfig(
        intensity_mode="normalized_range",
        voxel_downsample=0.05,
        min_range=0.3,
        max_range=8.0,
        subsample_ratio=1.0,
        scale_factor=float(depth_scale_factor),
        low_point_warn_ratio=0.0,
    )
    train_ds = RiskDataset(
        str(data_path),
        T_ctx=T_ctx,
        preprocess_cfg=preprocess_cfg,
        scene_filter=tr_s,
        traj_horizon=traj_horizon,
        bev_cache_root=bev_cache_root,
        extrinsics_convention=extrinsics_convention,
        include_action_seq=False,
        include_ego_vel_seq=False,
    )
    val_ds = RiskDataset(
        str(data_path),
        T_ctx=T_ctx,
        preprocess_cfg=preprocess_cfg,
        scene_filter=va_s,
        traj_horizon=traj_horizon,
        bev_cache_root=bev_cache_root,
        extrinsics_convention=extrinsics_convention,
        include_action_seq=False,
        include_ego_vel_seq=False,
    )
    w_arr = train_ds.risk_1s_array()
    weights = torch.as_tensor(
        np.where(w_arr > 0.5, positive_oversample_weight, 1.0), dtype=torch.double
    )
    train_sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
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

    pp = PointPillarsNeckExtractor(
        PointPillarsConfig(ckpt_path=str(ckpt_path), device=str(dev))
    )
    temporal = build_temporal(
        backbone,
        d_model=256,
        n_blocks=2,
        temporal_dropout=temporal_dropout,
        mamba_backend=mamba_backend,
    )
    model = FullPipelineRiskAndTraj(
        pp,
        traj_horizon=traj_horizon,
        mamba=temporal,
        token_dim=256,
    ).to(dev)

    if phase == "a1":
        model.pp.freeze_all()
        train_params = list(model.reducer.parameters()) + list(model.mamba.parameters()) + list(model.head.parameters()) + list(model.traj_head.parameters())
        opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=weight_decay)
    else:
        model.pp.freeze_all()
        model.pp.unfreeze_neck()
        neck_params = [p for p in model.pp.model.neck.parameters() if p.requires_grad]
        rest_params = (
            list(model.reducer.parameters())
            + list(model.mamba.parameters())
            + list(model.head.parameters())
            + list(model.traj_head.parameters())
        )
        opt = torch.optim.AdamW(
            [
                {"params": neck_params, "lr": lr_neck, "weight_decay": weight_decay},
                {"params": rest_params, "lr": lr_rest, "weight_decay": weight_decay},
            ]
        )

    accum = max(1, int(gradient_accumulation_steps))
    total_opt_steps = max(1, int(np.ceil(len(train_loader) / accum)) * max(1, epochs))
    warm = max(0, min(int(lr_warmup_iters), total_opt_steps))
    if warm > 0 and warm < total_opt_steps:
        lin = LinearLR(opt, start_factor=1e-8, end_factor=1.0, total_iters=warm)
        cos = CosineAnnealingLR(opt, T_max=total_opt_steps - warm, eta_min=cosine_eta_min)
        scheduler = SequentialLR(opt, [lin, cos], milestones=[warm])
    elif warm >= total_opt_steps:
        scheduler = LinearLR(opt, start_factor=1e-8, end_factor=1.0, total_iters=total_opt_steps)
    else:
        scheduler = CosineAnnealingLR(opt, T_max=total_opt_steps, eta_min=cosine_eta_min)

    rn = run_name or (f"{backbone}_a2_{time.strftime('%Y%m%d_%H%M%S')}" if phase == "a2" else f"{backbone}_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir = Path(log_root) / rn
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))

    cfg_row = {
        "backbone": backbone,
        "device": str(dev),
        "data_root": str(data_path.resolve()),
        "ckpt_path": str(Path(ckpt_path).resolve()),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": int(accum),
        "effective_batch_size": int(batch_size * accum),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "cosine_eta_min": float(cosine_eta_min),
        "seed": int(seed),
        "T_ctx": int(T_ctx),
        "traj_horizon": int(traj_horizon),
        "traj_loss_weight": float(traj_loss_weight),
        "grad_clip": float(grad_clip),
        "mamba_backend": str(mamba_backend),
        "train_val_test": list(train_val_test),
        "positive_oversample_weight": float(positive_oversample_weight),
        "train_scenes": [int(x) for x in tr_s],
        "val_scenes": [int(x) for x in va_s],
        "run_name": rn,
        "stage": phase,
        "lr_neck": float(lr_neck) if phase == "a2" else None,
        "lr_rest": float(lr_rest) if phase == "a2" else None,
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "lr_warmup_iters": int(lr_warmup_iters),
        "risk_focal_weights": list(RISK_FOCAL_WEIGHTS),
        "bev_cache_root": str(Path(bev_cache_root).resolve()),
        "extrinsics_convention": str(extrinsics_convention),
        "depth_scale_factor": float(depth_scale_factor),
        "risk_label_smoothing": float(risk_label_smoothing),
        "temporal_dropout": float(temporal_dropout),
    }
    _write_json(run_dir / "run_config.json", cfg_row)
    metrics_jsonl = run_dir / "metrics.jsonl"
    if metrics_jsonl.exists():
        metrics_jsonl.unlink()

    best_ap = -1e9
    best_metrics: Dict[str, float] = {}
    best_state: Optional[Dict[str, Any]] = None
    best_epoch = 0
    no_improve = 0
    global_step = 0

    opt.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        model.train()
        if phase == "a1":
            # keep frozen pp in eval
            model.pp.model.eval()
        else:
            # A2: neck trains, rest remains frozen/eval
            model.pp.model.pillar_encoder.eval()
            model.pp.model.backbone.eval()
            model.pp.model.neck.train()

        epoch_loss = 0.0
        n_batches = 0
        for micro_i, batch in enumerate(train_loader):
            pts = _to_pts_seq_bt(batch.pts_seq, dev)
            logits, traj_pred = model(pts)
            targets = torch.stack([batch.risk_05s, batch.risk_1s, batch.risk_2s], dim=1).float().to(dev)
            valid = batch.risk_label_valid.float().to(dev)
            traj_gt = batch.traj_future_xyyaw.float().to(dev)

            loss_r = focal_bce(
                logits,
                targets,
                gamma=2.0,
                weight=RISK_FOCAL_WEIGHTS,
                valid_mask=valid,
                label_smoothing=risk_label_smoothing,
            )
            loss_t = F.smooth_l1_loss(traj_pred, traj_gt)
            loss = (loss_r + float(traj_loss_weight) * loss_t) / accum
            loss.backward()
            epoch_loss += float((loss_r + float(traj_loss_weight) * loss_t).detach().item())
            n_batches += 1

            if ((micro_i + 1) % accum == 0) or (micro_i + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
            if (micro_i + 1) % 100 == 0:
                print(
                    f"[{rn}] epoch {epoch + 1}/{epochs} micro_batch {micro_i + 1}/{len(train_loader)} "
                    f"loss={float((loss_r + float(traj_loss_weight) * loss_t).detach().item()):.4f}",
                    flush=True,
                )

        train_loss = epoch_loss / max(1, n_batches)
        metrics = _collect_val_metrics(model, val_loader, dev)
        ap1 = float(metrics.get("ap_risk_1s", float("nan")))
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                writer.add_scalar(f"val/{k}", float(v), epoch)

        row_epoch: Dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss_epoch": train_loss,
        }
        for k, v in metrics.items():
            row_epoch[f"val_{k}"] = float(v) if isinstance(v, (int, float)) else v
        _append_jsonl(metrics_jsonl, row_epoch)
        print(
            f"[{rn}] epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} "
            f"val_ap_risk_05s={metrics.get('ap_risk_05s', float('nan')):.4f} "
            f"val_ap_risk_1s={ap1:.4f} "
            f"val_ADE_xy={metrics.get('traj_ade_xy_m', float('nan')):.4f} "
            f"val_latency_ms/sample={metrics.get('val_inference_ms_per_sample', float('nan')):.3f}",
            flush=True,
        )

        if not np.isnan(ap1) and (ap1 > best_ap + float(early_stop_min_delta)):
            best_ap = ap1
            best_metrics = {k: float(v) for k, v in metrics.items()}
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1

        if early_stop_patience > 0 and no_improve >= int(early_stop_patience):
            print(
                f"[{rn}] Early stop: val ap_risk_1s did not improve > {early_stop_min_delta} "
                f"for {early_stop_patience} epoch(s). Best ap_risk_1s={best_ap:.4f} @ epoch {best_epoch}.",
                flush=True,
            )
            break

    writer.close()
    epochs_run = epoch + 1
    final_metrics = best_metrics if best_metrics else metrics

    val_row = {k: float(v) for k, v in final_metrics.items()}
    val_row["backbone"] = backbone
    val_row["stage"] = phase
    val_row["run_name"] = rn
    val_row["epochs_run"] = int(epochs_run)
    val_row["epochs_requested"] = int(epochs)
    val_row["best_epoch"] = int(best_epoch) if best_epoch > 0 else None
    val_row["best_ap_risk_1s"] = float(best_ap if best_ap > -1e8 else float("nan"))
    _write_json(run_dir / "val_metrics_final.json", val_row)

    ckpt_dir = Path(weights_dir or (Path(log_root) / "checkpoints")) / f"{backbone}_risk_{epochs_run}epochs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_for_save = best_state if best_state is not None else {k: v.detach().cpu() for k, v in model.state_dict().items()}
    last_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if save_weights:
        pt_best = ckpt_dir / f"{backbone}_risk_{epochs_run}epochs_{rn}_best_val_ap.pt"
        pt_last = ckpt_dir / f"{backbone}_risk_{epochs_run}epochs_{rn}_last.pt"
        torch.save({"model_state_dict": model_for_save, "meta": val_row}, pt_best)
        torch.save({"model_state_dict": last_state, "meta": val_row}, pt_last)
        torch.save({"model_state_dict": model_for_save, "meta": val_row}, ckpt_dir / f"{backbone}_risk_best_val_ap.pt")
        torch.save({"model_state_dict": last_state, "meta": val_row}, ckpt_dir / f"{backbone}_risk_latest.pt")
    row: Dict[str, Union[float, str]] = {k: float(v) for k, v in final_metrics.items()}
    row["run_name"] = rn
    row["log_dir"] = str(run_dir.resolve())
    row["checkpoint_dir"] = str(ckpt_dir.resolve())
    return row


def main_cli(default_backbone: str) -> None:
    ap = argparse.ArgumentParser(description=f"Stage A single-backbone trainer ({default_backbone})")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=str(_PKG_ROOT / "pretrained" / "epoch_160_raw.pth"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--log_root", type=str, default="runs/stage_a_compare")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--T_ctx", type=int, default=40)
    ap.add_argument("--positive_weight", type=float, default=8.0)
    ap.add_argument("--traj_horizon", type=int, default=10)
    ap.add_argument("--traj_loss_weight", type=float, default=0.5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--mamba_backend", type=str, default="auto")
    ap.add_argument("--weights_dir", type=str, default="")
    ap.add_argument("--stage", type=str, choices=["a1", "a2"], default="a1")
    ap.add_argument("--lr_neck", type=float, default=5e-6)
    ap.add_argument("--lr_rest", type=float, default=1.67e-5)
    ap.add_argument("--early_stop_patience", type=int, default=8)
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0)
    ap.add_argument("--lr_warmup_iters", type=int, default=500)
    ap.add_argument("--cosine_eta_min", type=float, default=1e-6)
    ap.add_argument("--bev_cache_root", type=str, default="")
    ap.add_argument("--extrinsics_convention", type=str, default="auto")
    ap.add_argument("--depth_scale_factor", type=float, default=1.0)
    ap.add_argument("--risk_label_smoothing", type=float, default=0.05)
    ap.add_argument("--temporal_dropout", type=float, default=0.1)
    args = ap.parse_args()

    run_stage_a_training(
        backbone=default_backbone,
        data_root=args.data_root,
        ckpt_path=args.ckpt,
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
        save_weights=True,
        stage=validate_stage(args.stage),
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

