"""
Single-backbone Stage A training: dataset -> risk + trajectory -> validate.

Shared by the CLI entry points in ``PointPillars_module/training/`` (e.g.
``train_stage_a_mamba.py``, ``train_stage_a_compare.py``).

Artifacts per run (under ``log_root/<run_name>/``):

* TensorBoard (``SummaryWriter``)
* ``run_config.json`` — CLI hyperparameters + backbone
* ``metrics.jsonl`` — one JSON object per epoch (val metrics + train loss)
* ``metrics_all_epochs.json`` — same as array
* ``val_metrics_final.json`` — last epoch validation scalars
* ``summary_row.csv`` — single-row CSV for spreadsheet / tables
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
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
from module_pointpillar import PointPillarsNeckExtractor  # noqa: E402
from models.full_pipeline_risk_traj import FullPipelineRiskAndTraj  # noqa: E402
from models.temporal_factory import build_temporal  # noqa: E402
from risk_dataset import RiskDataset, collate_riskbatch, scene_stratified_split  # noqa: E402
from PointPillars_module.types import PointPillarsConfig  # noqa: E402
from utils.gradient_health import grad_norm_l2, max_grad_value  # noqa: E402
from utils.mamba_runtime import log_mamba_temporal_runtime  # noqa: E402

STAGE_A_BACKBONES = ("mamba", "gru", "lstm", "transformer")
# § 5.3 horizon weights [0.5s, 1s, 2s]
RISK_FOCAL_WEIGHTS: Tuple[float, float, float] = (1.0, 0.8, 0.5)
StageAPhase = Literal["a1", "a2"]


def validate_backbone(name: str) -> str:
    k = str(name).lower().strip()
    if k not in STAGE_A_BACKBONES:
        raise ValueError(
            f"unknown backbone {name!r}; expected one of {STAGE_A_BACKBONES}"
        )
    return k


def validate_stage(phase: str) -> StageAPhase:
    p = str(phase).lower().strip()
    if p not in ("a1", "a2"):
        raise ValueError("stage must be 'a1' or 'a2'")
    return p  # type: ignore[return-value]


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


def _trainable_param_count(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def _make_stage_a_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_optimizer_steps: int,
    warmup_steps: int,
    eta_min: float,
) -> Any:
    """
    Linear warmup for ``warmup_steps`` optimizer steps, then cosine decay to
    ``eta_min`` over the remaining steps (§ 5.3).
    """
    T = max(1, int(total_optimizer_steps))
    w = max(0, min(int(warmup_steps), T))
    if w <= 0:
        return CosineAnnealingLR(optimizer, T_max=T, eta_min=float(eta_min))
    if w >= T:
        return LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=T)
    lin = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=w)
    cos = CosineAnnealingLR(optimizer, T_max=T - w, eta_min=float(eta_min))
    return SequentialLR(optimizer, schedulers=[lin, cos], milestones=[w])


def _batch_pos_ratio_masked(
    labels: torch.Tensor, valid_col: torch.Tensor
) -> float:
    """Positive rate among batch rows where ``valid_col`` is 1 (truncation-aware)."""
    v = valid_col > 0.5
    if int(v.sum().item()) == 0:
        return float("nan")
    return float(((labels > 0.5) & v).float().sum() / v.float().sum())


def traj_error_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """pred, gt: (N, H, 3) with (x, y, yaw). Yaw RMSE in radians."""
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


def collect_val_metrics(
    model: FullPipelineRiskAndTraj,
    loader: DataLoader,
    device: torch.device,
    *,
    measure_inference_latency: bool = True,
) -> Dict[str, float]:
    model.reducer.eval()
    model.mamba.eval()
    model.head.eval()
    model.traj_head.eval()
    # Neck may be .train() during A2; eval() for validation (BN running stats).
    model.pp.model.neck.eval()
    logits_all: List[torch.Tensor] = []
    t_all: List[torch.Tensor] = []
    valid_all: List[torch.Tensor] = []
    traj_pred_all: List[torch.Tensor] = []
    traj_gt_all: List[torch.Tensor] = []
    infer_time_s = 0.0
    infer_n = 0
    with torch.no_grad():
        for batch in loader:
            pts = [
                [p.to(device, non_blocking=True) for p in frame]
                for frame in batch.pts_seq
            ]
            if measure_inference_latency and device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            logits, traj_pred = model(pts)
            if measure_inference_latency and device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            if measure_inference_latency:
                infer_time_s += float(t1 - t0)
                infer_n += int(logits.shape[0])
            logits_all.append(logits.cpu())
            targets = torch.stack(
                [batch.risk_05s, batch.risk_1s, batch.risk_2s], dim=1
            ).float()
            t_all.append(targets)
            valid_all.append(batch.risk_label_valid.float().cpu())
            traj_pred_all.append(traj_pred.cpu())
            traj_gt_all.append(batch.traj_future_xyyaw.float())

    if not logits_all:
        # Validation split can be empty after scene split + window filtering
        # (large T_ctx / traj_horizon on tiny datasets). Return NaNs instead
        # of crashing on torch.cat([]) so the run still writes artifacts.
        out_empty: Dict[str, float] = {
            "traj_rmse_all": float("nan"),
            "traj_ade_xy_m": float("nan"),
            "traj_fde_xy_m": float("nan"),
            "traj_rmse_yaw_rad": float("nan"),
            "ap_risk_05s": float("nan"),
            "auc_risk_05s": float("nan"),
            "ap_risk_1s": float("nan"),
            "auc_risk_1s": float("nan"),
            "ap_risk_2s": float("nan"),
            "auc_risk_2s": float("nan"),
            "val_sample_count": 0.0,
        }
        if measure_inference_latency:
            out_empty["val_inference_ms_per_sample"] = float("nan")
        print(
            "[collect_val_metrics] Warning: validation loader produced 0 batches; "
            "returning NaN metrics.",
            flush=True,
        )
        return out_empty

    logits_cat = torch.cat(logits_all, dim=0).numpy()
    t_cat = torch.cat(t_all, dim=0).numpy()
    v_cat = torch.cat(valid_all, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_cat))

    pred_traj = torch.cat(traj_pred_all, dim=0).numpy()
    gt_traj = torch.cat(traj_gt_all, dim=0).numpy()
    out: Dict[str, float] = traj_error_metrics(pred_traj, gt_traj)

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        return out

    names = ["risk_05s", "risk_1s", "risk_2s"]
    for i, name in enumerate(names):
        m = v_cat[:, i] > 0.5
        if not np.any(m):
            out[f"ap_{name}"] = float("nan")
            out[f"auc_{name}"] = float("nan")
            continue
        y = t_cat[m, i]
        p = probs[m, i]
        if y.max() == y.min():
            out[f"ap_{name}"] = float("nan")
            out[f"auc_{name}"] = float("nan")
            continue
        out[f"ap_{name}"] = float(average_precision_score(y, p))
        try:
            out[f"auc_{name}"] = float(roc_auc_score(y, p))
        except ValueError:
            out[f"auc_{name}"] = float("nan")
    if measure_inference_latency and infer_n > 0:
        out["val_inference_ms_per_sample"] = (infer_time_s / float(infer_n)) * 1000.0
    return out


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {k: ("" if v is None else v) for k, v in row.items()}
    keys = list(flat.keys())
    write_header = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            w.writeheader()
        w.writerow({k: flat[k] for k in keys})


def run_stage_a_training(
    *,
    backbone: str,
    data_root: str,
    ckpt_path: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    log_root: str = "runs/stage_a",
    device: Optional[str] = None,
    seed: int = 0,
    num_workers: int = 4,
    positive_oversample_weight: float = 8.0,
    train_val_test: Tuple[float, float, float] = (0.75, 0.25, 0.0),
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
    lr_neck: float = 3e-5,
    lr_rest: float = 1e-4,
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 0.0,
    lr_warmup_iters: int = 500,
    bev_cache_root: Optional[str] = None,
) -> Dict[str, Union[float, str]]:
    """
    Train one temporal backbone; log TensorBoard + JSON/CSV under ``log_root/<run_name>/``.

    * **a1** — PointPillars fully frozen; train reducer + temporal + heads with ``lr``.
    * **a2** — Unfreeze ``pp.neck`` only (KITTI to depth-camera adaptation); two AdamW
      groups: ``lr_neck`` on neck, ``lr_rest`` on reducer + temporal + heads (doc §5.4).

    **Batching:** default pairs such as ``batch_size=32`` × ``gradient_accumulation_steps=2``
    yield effective batch 64; reduce batch or raise accumulation if VRAM is tight.

    **LR schedule:** linear warmup for ``lr_warmup_iters`` optimizer steps (default 500), then
    cosine decay to ``cosine_eta_min`` over the remaining steps in the run (§ 5.3).

    **Early stopping:** if ``early_stop_patience > 0``, stop when ``val ap_risk_1s`` does not
    improve by more than ``early_stop_min_delta`` for that many consecutive epochs (monitors
    the headline AP metric from ``strategy_experiment_protocol.md``). Saves ``*_best_val_ap.pt``
    when improvement occurs; final JSON favors **best-val** metrics.
    """
    backbone = validate_backbone(backbone)
    phase = validate_stage(stage)
    data_path = Path(data_root)
    if not data_path.is_dir():
        raise FileNotFoundError(f"data_root is not a directory: {data_path}")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type != "cuda":
        print(
            "[stage_a_single_run] Warning: PointPillars voxel path expects CUDA; "
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
        str(data_path),
        T_ctx=T_ctx,
        scene_filter=tr_s,
        traj_horizon=traj_horizon,
        bev_cache_root=bev_cache_root,
        include_action_seq=False,
        include_ego_vel_seq=False,
    )
    val_ds = RiskDataset(
        str(data_path),
        T_ctx=T_ctx,
        scene_filter=va_s,
        traj_horizon=traj_horizon,
        bev_cache_root=bev_cache_root,
        include_action_seq=False,
        include_ego_vel_seq=False,
    )

    w_arr = train_ds.risk_1s_array()
    weights = torch.as_tensor(
        np.where(w_arr > 0.5, positive_oversample_weight, 1.0),
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(
        weights, num_samples=len(train_ds), replacement=True
    )

    _dl_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        _dl_kwargs["persistent_workers"] = True
        _dl_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_riskbatch,
        pin_memory=(dev.type == "cuda"),
        **_dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_riskbatch,
        pin_memory=(dev.type == "cuda"),
        **_dl_kwargs,
    )

    rn = run_name or (
        f"{backbone}_a2_{time.strftime('%Y%m%d_%H%M%S')}"
        if phase == "a2"
        else f"{backbone}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    log_dir_base = Path(log_root)
    run_dir = log_dir_base / rn
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    writer.add_text("hyperparams/stage", phase, 0)

    accum = max(1, int(gradient_accumulation_steps))
    if dev.type == "cuda":
        _gb = torch.cuda.get_device_properties(dev).total_memory / (1024.0**3)
        if _gb <= 16.0:
            print(
                "[stage_a_single_run] CUDA memory <= 16 GiB: if OOM, try "
                "--batch_size 16 and/or higher --gradient_accumulation_steps.",
                flush=True,
            )
    effective_batch = int(batch_size) * accum

    cfg_payload = {
        "backbone": backbone,
        "device": str(dev),
        "data_root": str(data_path.resolve()),
        "ckpt_path": str(Path(ckpt_path).resolve()),
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accum,
        "effective_batch_size": effective_batch,
        "lr": lr,
        "weight_decay": weight_decay,
        "cosine_eta_min": cosine_eta_min,
        "seed": seed,
        "T_ctx": T_ctx,
        "traj_horizon": traj_horizon,
        "traj_loss_weight": traj_loss_weight,
        "grad_clip": grad_clip,
        "mamba_backend": mamba_backend,
        "train_val_test": train_val_test,
        "positive_oversample_weight": positive_oversample_weight,
        "train_scenes": tr_s,
        "val_scenes": va_s,
        "run_name": rn,
        "stage": phase,
        "lr_neck": lr_neck if phase == "a2" else None,
        "lr_rest": lr_rest if phase == "a2" else None,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "lr_warmup_iters": lr_warmup_iters,
        "risk_focal_weights": list(RISK_FOCAL_WEIGHTS),
        "bev_cache_root": (
            str(Path(bev_cache_root).resolve()) if bev_cache_root else ""
        ),
    }
    if dev.type == "cuda":
        _p = torch.cuda.get_device_properties(dev)
        cfg_payload["cuda_device_name"] = str(_p.name)
        cfg_payload["cuda_total_memory_gb"] = round(
            _p.total_memory / (1024.0**3), 4
        )

    metrics_jsonl = run_dir / "metrics.jsonl"
    if metrics_jsonl.is_file():
        metrics_jsonl.unlink()

    cfg = PointPillarsConfig(ckpt_path=ckpt_path, device=str(dev))
    pp = PointPillarsNeckExtractor(cfg)
    temporal = build_temporal(
        backbone, d_model=256, n_blocks=2, mamba_backend=mamba_backend
    )
    model = FullPipelineRiskAndTraj(
        pp, mamba=temporal, token_dim=256, traj_horizon=traj_horizon
    )
    model.to(dev)
    model.pp.freeze_all()
    if phase == "a2":
        model.pp.unfreeze_neck()

    temporal_trainable = _trainable_param_count(model.mamba)
    writer.add_scalar("model/temporal_backbone_only_params", float(temporal_trainable), 0)
    writer.add_scalar("model/temporal_trainable_params", float(temporal_trainable), 0)
    writer.add_text(
        "model/summary",
        f"Fair backbone comparison: temporal module only (Mamba/GRU/LSTM/Transformer) "
        f"trainable params = {temporal_trainable}. "
        f"SpatialReducer + heads are additional trainable parameters (see total below).",
        0,
    )
    print(
        f"[{rn}] temporal backbone ONLY trainable parameters (fair compare): "
        f"{temporal_trainable}",
        flush=True,
    )
    log_mamba_temporal_runtime(model.mamba, device=dev)

    tail_params: List[torch.nn.Parameter] = (
        list(model.reducer.parameters())
        + list(model.mamba.parameters())
        + list(model.head.parameters())
        + list(model.traj_head.parameters())
    )
    neck_params: List[torch.nn.Parameter] = list(model.pp.model.neck.parameters())

    if phase == "a1":
        trainable = tail_params
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    else:
        trainable = neck_params + tail_params
        opt = torch.optim.AdamW(
            [
                {"params": neck_params, "lr": lr_neck, "weight_decay": weight_decay},
                {"params": tail_params, "lr": lr_rest, "weight_decay": weight_decay},
            ]
        )

    n_micro_train = len(train_loader)
    opt_steps_per_epoch = max(1, (n_micro_train + accum - 1) // accum)
    total_opt_steps = max(1, opt_steps_per_epoch * int(epochs))
    w_sched = min(int(lr_warmup_iters), total_opt_steps)
    scheduler = _make_stage_a_scheduler(
        opt,
        total_optimizer_steps=total_opt_steps,
        warmup_steps=w_sched,
        eta_min=float(cosine_eta_min),
    )
    traj_criterion = torch.nn.SmoothL1Loss(beta=0.05)

    total_trainable = int(sum(p.numel() for p in trainable))
    writer.add_scalar("model/total_trainable_params", float(total_trainable), 0)
    print(f"[{rn}] total AdamW trainable parameters: {total_trainable}", flush=True)

    cfg_payload["temporal_trainable_params"] = temporal_trainable
    cfg_payload["temporal_backbone_only_params"] = temporal_trainable
    cfg_payload["total_trainable_params"] = total_trainable
    cfg_payload["total_optimizer_steps"] = total_opt_steps
    cfg_payload["optimizer_steps_per_epoch"] = opt_steps_per_epoch
    cfg_payload["warmup_steps_scheduled"] = w_sched
    _write_json(run_dir / "run_config.json", cfg_payload)

    global_step = 0
    global_micro_step = 0
    metrics_epochs: List[Dict[str, Any]] = []
    last_metrics: Dict[str, float] = {}
    best_val_metrics: Dict[str, float] = {}
    best_epoch_1based = 0
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_ap_for_stop = -1.0
    epochs_no_improve = 0
    early_stopped = False
    es_patience = int(early_stop_patience)
    es_delta = float(early_stop_min_delta)

    for epoch in range(epochs):
        _lrs = scheduler.get_last_lr()
        for _i, _lr in enumerate(_lrs):
            writer.add_scalar(f"train/learning_rate_group_{_i}", float(_lr), epoch)
        writer.add_scalar("train/learning_rate", float(_lrs[-1]), epoch)
        model.reducer.train()
        model.mamba.train()
        model.head.train()
        model.traj_head.train()
        if phase == "a2":
            model.pp.model.neck.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_max_grad_pre = 0.0
        epoch_max_grad_abs_post = 0.0
        opt.zero_grad(set_to_none=True)
        n_micro = len(train_loader)
        for micro_i, batch in enumerate(train_loader):
            pts = [
                [p.to(dev, non_blocking=True) for p in frame]
                for frame in batch.pts_seq
            ]
            logits, traj_hat = model(pts)
            targets = torch.stack(
                [batch.risk_05s, batch.risk_1s, batch.risk_2s], dim=1
            ).to(dev)
            loss_r = focal_bce(
                logits,
                targets,
                weight=RISK_FOCAL_WEIGHTS,
                valid_mask=batch.risk_label_valid.to(dev),
            )
            traj_gt = batch.traj_future_xyyaw.to(dev)
            loss_t = traj_criterion(traj_hat, traj_gt)
            total_mb = loss_r + traj_loss_weight * loss_t
            loss = total_mb / float(accum)
            loss.backward()
            epoch_loss += float(total_mb.detach())
            n_batches += 1
            writer.add_scalar("train/loss_risk", float(loss_r.detach()), global_micro_step)
            writer.add_scalar("train/loss_traj", float(loss_t.detach()), global_micro_step)
            writer.add_scalar("train/loss_microbatch", float(total_mb.detach()), global_micro_step)
            vm = batch.risk_label_valid.to(dev)
            writer.add_scalar(
                "train/batch_pos_ratio_raw_risk_1s",
                float((batch.risk_1s.to(dev) > 0.5).float().mean()),
                global_micro_step,
            )
            rcols = (batch.risk_05s.to(dev), batch.risk_1s.to(dev), batch.risk_2s.to(dev))
            for j, tag in enumerate(
                (
                    "train/batch_pos_ratio_masked_risk_05s",
                    "train/batch_pos_ratio_masked_risk_1s",
                    "train/batch_pos_ratio_masked_risk_2s",
                )
            ):
                r = _batch_pos_ratio_masked(rcols[j], vm[:, j])
                if not np.isnan(r):
                    writer.add_scalar(tag, r, global_micro_step)
            global_micro_step += 1

            do_step = (micro_i + 1) % accum == 0 or (micro_i + 1) == n_micro
            if do_step:
                gn_pre = float(grad_norm_l2(trainable))
                epoch_max_grad_pre = max(epoch_max_grad_pre, gn_pre)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                gn_post = float(grad_norm_l2(trainable))
                mg_post = float(max_grad_value(trainable))
                epoch_max_grad_abs_post = max(epoch_max_grad_abs_post, mg_post)
                opt.step()
                scheduler.step()
                _lr_now = scheduler.get_last_lr()
                writer.add_scalar("train/lr_after_optimizer_step", float(_lr_now[-1]), global_step)
                opt.zero_grad(set_to_none=True)
                writer.add_scalar("train/loss_step", float(total_mb.detach()), global_step)
                writer.add_scalar("train/grad_norm", gn_post, global_step)
                writer.add_scalar("train/grad_norm_pre_clip", gn_pre, global_step)
                writer.add_scalar("train/grad_norm_post_clip", gn_post, global_step)
                writer.add_scalar("train/global_grad_norm_pre_clip", gn_pre, global_step)
                writer.add_scalar("train/global_grad_norm_post_clip", gn_post, global_step)
                writer.add_scalar("train/max_grad_abs_post_clip", mg_post, global_step)
                global_step += 1

        train_loss_epoch = epoch_loss / max(1, n_batches)
        writer.add_scalar("train/loss_epoch", train_loss_epoch, epoch)
        writer.add_scalar("train/epoch_max_grad_norm_pre_clip", epoch_max_grad_pre, epoch)
        writer.add_scalar(
            "train/epoch_max_grad_abs_post_clip", epoch_max_grad_abs_post, epoch
        )

        metrics = collect_val_metrics(model, val_loader, dev)
        last_metrics = metrics
        for k, v in metrics.items():
            if not np.isnan(v):
                writer.add_scalar(f"val/{k}", v, epoch)
        lat_ms = metrics.get("val_inference_ms_per_sample", float("nan"))
        if not np.isnan(lat_ms):
            writer.add_scalar("val/inference_ms_per_sample", float(lat_ms), epoch)

        ap1 = metrics.get("ap_risk_1s", float("nan"))

        if not np.isnan(ap1):
            if ap1 > best_ap_for_stop + es_delta:
                best_ap_for_stop = float(ap1)
                best_epoch_1based = epoch + 1
                best_val_metrics = {k: float(v) for k, v in metrics.items()}
                best_state_dict = _cpu_state_dict(model)
                epochs_no_improve = 0
            elif es_patience > 0:
                epochs_no_improve += 1
        elif es_patience > 0:
            epochs_no_improve += 1

        writer.add_scalar("val/best_ap_risk_1s_running", best_ap_for_stop, epoch)

        row_epoch: Dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss_epoch": train_loss_epoch,
            "train_batches": n_batches,
            "epochs_no_improve": epochs_no_improve,
        }
        for k, v in metrics.items():
            row_epoch[f"val_{k}"] = float(v) if isinstance(v, (float, int)) else v
        metrics_epochs.append(row_epoch)
        _append_jsonl(metrics_jsonl, row_epoch)

        ade = metrics.get("traj_ade_xy_m", float("nan"))
        ap05 = metrics.get("ap_risk_05s", float("nan"))
        lat_s = (
            f" val_latency_ms/sample={lat_ms:.3f}"
            if not np.isnan(lat_ms)
            else ""
        )
        print(
            f"[{rn}] epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss_epoch:.4f} "
            f"val_ap_risk_05s={ap05:.4f} val_ap_risk_1s={ap1:.4f} val_ADE_xy={ade:.4f}"
            f"{lat_s}",
            flush=True,
        )

        if es_patience > 0 and epochs_no_improve >= es_patience:
            early_stopped = True
            print(
                f"[{rn}] Early stop: val ap_risk_1s did not improve > {es_delta} "
                f"for {es_patience} epoch(s). Best ap_risk_1s={best_ap_for_stop:.4f} "
                f"@ epoch {best_epoch_1based}.",
                flush=True,
            )
            writer.add_text(
                "train/early_stop",
                f"stopped at epoch {epoch + 1}; best ap_risk_1s={best_ap_for_stop:.6f} "
                f"at epoch {best_epoch_1based}",
                epoch,
            )
            break

    epochs_run = epoch + 1
    writer.add_scalar("val/best_ap_risk_1s", best_ap_for_stop, max(0, epochs_run - 1))
    writer.close()

    _write_json(run_dir / "metrics_all_epochs.json", metrics_epochs)
    report_metrics = best_val_metrics if best_val_metrics else last_metrics
    val_row = {k: float(v) for k, v in report_metrics.items()}
    val_row["backbone"] = backbone
    val_row["stage"] = phase
    val_row["run_name"] = rn
    val_row["epochs_run"] = epochs_run
    val_row["epochs_requested"] = epochs
    val_row["best_epoch"] = (
        int(best_epoch_1based) if best_val_metrics else None
    )
    val_row["early_stopped"] = early_stopped
    val_row["best_ap_risk_1s"] = best_ap_for_stop
    _write_json(run_dir / "val_metrics_final.json", val_row)

    csv_row = {**val_row, "log_dir": str(run_dir.resolve())}
    _write_summary_csv(run_dir / "summary_row.csv", csv_row)

    row: Dict[str, Union[float, str]] = {
        k: float(v) for k, v in report_metrics.items()
    }
    row["backbone"] = backbone
    row["stage"] = phase
    row["run_name"] = rn
    row["epochs_run"] = float(epochs_run)
    row["early_stopped"] = early_stopped
    row["best_ap_risk_1s"] = best_ap_for_stop
    row["log_dir"] = str(run_dir.resolve())
    row["tensorboard_logdir"] = str(run_dir.resolve())

    _write_json(
        run_dir / "early_stop.json",
        {
            "early_stopped": early_stopped,
            "epochs_run": epochs_run,
            "epochs_requested": epochs,
            "patience": es_patience,
            "min_delta": es_delta,
            "best_epoch": best_epoch_1based if best_val_metrics else None,
            "best_ap_risk_1s": best_ap_for_stop,
        },
    )

    wdir = Path(weights_dir) if weights_dir else log_dir_base / "weights"
    if save_weights:
        wdir.mkdir(parents=True, exist_ok=True)
        tail = (
            rn[len(backbone) + 1 :].lstrip("_")
            if rn.startswith(backbone + "_")
            else time.strftime("%Y%m%d_%H%M%S")
        )
        stem = f"{backbone}_stage_a_{tail}"
        latest_stem = (
            f"{backbone}_stage_a_a2_latest"
            if phase == "a2"
            else f"{backbone}_stage_a_latest"
        )
        out_ts = wdir / f"{stem}.pt"
        out_latest = wdir / f"{latest_stem}.pt"
        last_sd = _cpu_state_dict(model)
        payload_last = {
            "format": "stage_a_full_pipeline_risk_traj_v1",
            "stage": phase,
            "backbone": backbone,
            "run_name": rn,
            "traj_horizon": traj_horizon,
            "T_ctx": T_ctx,
            "epochs_run": epochs_run,
            "epochs_requested": epochs,
            "pp_ckpt_path": str(Path(ckpt_path).resolve()),
            "data_root": str(data_path.resolve()),
            "val_metrics_last_epoch": {k: float(v) for k, v in last_metrics.items()},
            "val_metrics_selected": {k: float(v) for k, v in report_metrics.items()},
            "model_state_dict": last_sd,
        }
        torch.save(payload_last, out_ts)
        torch.save(payload_last, out_latest)
        row["checkpoint_pt"] = str(out_ts.resolve())
        row["checkpoint_pt_latest"] = str(out_latest.resolve())

        if best_state_dict is not None:
            out_best = wdir / f"{stem}_best_val_ap.pt"
            payload_best = {
                "format": "stage_a_full_pipeline_risk_traj_v1",
                "stage": phase,
                "backbone": backbone,
                "run_name": rn,
                "traj_horizon": traj_horizon,
                "T_ctx": T_ctx,
                "best_epoch": best_epoch_1based,
                "epochs_run": epochs_run,
                "pp_ckpt_path": str(Path(ckpt_path).resolve()),
                "data_root": str(data_path.resolve()),
                "val_metrics": {k: float(v) for k, v in best_val_metrics.items()},
                "model_state_dict": best_state_dict,
            }
            torch.save(payload_best, out_best)
            row["checkpoint_pt_best_val_ap"] = str(out_best.resolve())
            best_latest = (
                f"{backbone}_stage_a_a2_best_val_ap.pt"
                if phase == "a2"
                else f"{backbone}_stage_a_best_val_ap.pt"
            )
            out_best_latest = wdir / best_latest
            torch.save(payload_best, out_best_latest)
            row["checkpoint_pt_best_val_ap_latest"] = str(out_best_latest.resolve())
            print(
                f"[{rn}] Best-val checkpoint: {out_best.name} + {out_best_latest.name}",
                flush=True,
            )

        print(
            f"[{rn}] Last-epoch checkpoints: {out_ts.name} + {out_latest.name} (under {wdir})",
            flush=True,
        )

    print(f"\nTensorBoard: tensorboard --logdir {run_dir}", flush=True)
    print(f"Metrics: {run_dir / 'metrics.jsonl'}", flush=True)
    lat_final = report_metrics.get("val_inference_ms_per_sample", float("nan"))
    if not np.isnan(lat_final):
        print(
            f"Validation inference latency (best-val snapshot): "
            f"{float(lat_final):.4f} ms/sample",
            flush=True,
        )
    return row


def build_arg_parser(*, fixed_backbone: Optional[str] = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Stage A: train risk + trajectory (single temporal backbone)."
    )
    if fixed_backbone is None:
        ap.add_argument(
            "--backbone",
            type=str,
            required=True,
            choices=list(STAGE_A_BACKBONES),
            help="Temporal backbone: mamba | gru | lstm | transformer",
        )
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument(
        "--ckpt",
        type=str,
        default=str(_PKG_ROOT / "pretrained" / "epoch_160.pth"),
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Micro-batches per optimizer step (e.g. batch 32 × 2 → effective batch 64).",
    )
    ap.add_argument(
        "--cosine_eta_min",
        type=float,
        default=1e-6,
        help="Minimum LR for CosineAnnealingLR (per-epoch schedule).",
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--log_root", type=str, default="runs/stage_a")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument(
        "--T_ctx",
        type=int,
        default=40,
        help="Context frames (20≈1.0s, 40≈2.0s at 20 Hz; align with longest risk horizon).",
    )
    ap.add_argument("--positive_weight", type=float, default=8.0)
    ap.add_argument("--traj_horizon", type=int, default=10)
    ap.add_argument("--traj_loss_weight", type=float, default=0.5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument(
        "--stage",
        type=str,
        choices=["a1", "a2"],
        default="a1",
        help="a1: PP frozen (default). a2: unfreeze pp.neck only + two LRs (KITTI->sim).",
    )
    ap.add_argument(
        "--lr_neck",
        type=float,
        default=3e-5,
        help="AdamW LR for pp.neck when --stage a2 (doc default 3e-5).",
    )
    ap.add_argument(
        "--lr_rest",
        type=float,
        default=1e-4,
        help="AdamW LR for reducer+temporal+heads when --stage a2 (doc default 1e-4).",
    )
    ap.add_argument("--mamba_backend", type=str, default="auto")
    ap.add_argument(
        "--weights_dir",
        type=str,
        default="",
        help="Directory for .pt checkpoints (default: <log_root>/weights).",
    )
    ap.add_argument("--no_save_weights", action="store_true")
    ap.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run subfolder name (default: <backbone>_YYYYMMDD_HHMMSS).",
    )
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="Stop if val ap_risk_1s does not improve for this many epochs (0=disabled).",
    )
    ap.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum ap_risk_1s gain to count as improvement (default 0).",
    )
    ap.add_argument(
        "--lr_warmup_iters",
        type=int,
        default=500,
        help="Linear LR warmup length in optimizer steps before cosine decay (§ 5.3).",
    )
    ap.add_argument(
        "--bev_cache_root",
        type=str,
        default="",
        help=(
            "Optional root directory of precomputed BEV cache created by "
            "scripts/cache_pointpillars_bev.py. When set, Stage A training "
            "skips per-step PointPillars neck extraction."
        ),
    )
    return ap


def main_cli(fixed_backbone: Optional[str] = None) -> None:
    ap = build_arg_parser(fixed_backbone=fixed_backbone)
    args = ap.parse_args()
    bb = fixed_backbone if fixed_backbone else validate_backbone(args.backbone)
    run_stage_a_training(
        backbone=bb,
        data_root=args.data_root,
        ckpt_path=args.ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_root=args.log_root,
        device=(args.device or None),
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
        run_name=(getattr(args, "run_name", "") or "").strip() or None,
        stage=validate_stage(args.stage),
        lr_neck=args.lr_neck,
        lr_rest=args.lr_rest,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        lr_warmup_iters=args.lr_warmup_iters,
        bev_cache_root=(args.bev_cache_root or None),
    )


if __name__ == "__main__":
    main_cli(None)
