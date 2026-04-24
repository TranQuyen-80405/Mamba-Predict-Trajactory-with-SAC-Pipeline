"""
Train / compare Stage A temporal backbones (mamba, gru, lstm, transformer).

Historical note:
``training/stage_a_single_run.py`` existed in earlier revisions as a shared
single-backbone wrapper. It has been removed; this module now owns validation
and per-backbone run dispatch directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Before first cudaMalloc: reduce VRAM fragmentation (PyTorch 2.x, CUDA 12+).
if os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip() == "":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import average_precision_score, roc_auc_score

_PKG_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PKG_ROOT.parent
for _p in (_PKG_ROOT, _REPO_ROOT, _REPO_ROOT / "create_dataset_module"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

# Large BEV batches + default fd-based sharing can fill /dev/shm (common in
# Docker) and workers exit with "RuntimeError: DataLoader worker ... exited
# unexpectedly". Prefer file-based sharing unless the user forces fd mode:
#   TORCH_SHARING_STRATEGY=file_descriptor
_sharing = os.environ.get("TORCH_SHARING_STRATEGY", "file_system").strip().lower()
if _sharing in ("file_system", "filesystem", "fs", ""):
    try:
        mp.set_sharing_strategy("file_system")
    except (RuntimeError, ValueError):
        pass
else:
    try:
        mp.set_sharing_strategy("file_descriptor")
    except (RuntimeError, ValueError):
        pass

from create_dataset_module.risk_dataset import (
    RiskDataset,
    collate_riskbatch,
    scene_stratified_split,
)
from PointPillars_module.losses import MultiTaskLossWrapper, focal_bce
from PointPillars_module.models.full_pipeline_risk_traj import FullPipelineRiskAndTraj
from PointPillars_module.models.temporal_factory import build_temporal
from PointPillars_module.types import PointPillarsConfig
from module_pointpillar import PointPillarsNeckExtractor

def validate_backbone(name: str) -> str:
    k = str(name).strip().lower()
    allowed = {"mamba", "gru", "lstm", "transformer"}
    if k not in allowed:
        raise ValueError(
            f"unknown backbone {name!r}; expected one of {sorted(allowed)}"
        )
    return k


def validate_stage(stage: str) -> str:
    s = str(stage).strip().lower()
    if s not in {"a1", "a2"}:
        raise ValueError(f"unknown stage {stage!r}; expected one of ['a1', 'a2']")
    return s


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataloader_worker_init(_worker_id: int) -> None:
    """
    DataLoader child process: use one BLAS/OMP thread to avoid 100+ threads
    per worker (common cause of crashes / watchdog kills under spawn).
    """
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[k] = "1"


def _log_dev_shm_for_workers(num_workers: int) -> None:
    if int(num_workers) <= 0:
        return
    try:
        st = os.statvfs("/dev/shm")
        avail_gib = st.f_bavail * st.f_frsize / (1024.0**3)
        print(
            f"[DataLoader] /dev/shm free ~{avail_gib:.1f} GiB (IPC for worker batches; "
            "too small in Docker -> raise --shm-size, e.g. 8g).",
            flush=True,
        )
        if avail_gib < 0.5:
            print(
                "[DataLoader] WARNING: /dev/shm is tiny; set Docker --shm-size=8g+ or "
                "use --num_workers 0. TORCH_SHARING_STRATEGY=file_system also helps (default here).",
                flush=True,
            )
    except OSError as e:
        print(f"[DataLoader] could not stat /dev/shm: {e}", flush=True)


def _to_pts_seq_bt(
    pts_seq: Union[torch.Tensor, Sequence[Sequence[torch.Tensor]]],
    device: torch.device,
) -> List[List[torch.Tensor]]:
    if torch.is_tensor(pts_seq):
        # Cached BEV fast-path from collate_riskbatch: (T, B, C, H, W).
        if pts_seq.ndim != 5:
            raise ValueError(f"expected pts_seq tensor (T,B,C,H,W), got {tuple(pts_seq.shape)}")
        T, B = int(pts_seq.shape[0]), int(pts_seq.shape[1])
        return [
            [pts_seq[t, b].to(device=device, non_blocking=True) for b in range(B)]
            for t in range(T)
        ]
    out: List[List[torch.Tensor]] = []
    for frame in pts_seq:
        out.append([x.to(device=device, non_blocking=True) for x in frame])
    return out


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or float(y_true.sum()) <= 0.0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    uniq = np.unique(y_true)
    if uniq.size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _traj_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    dxy = pred[..., :2] - gt[..., :2]
    dyaw = pred[..., 2] - gt[..., 2]
    return {
        "traj_rmse_all": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "traj_ade_xy_m": float(np.mean(np.linalg.norm(dxy, axis=-1))),
        "traj_fde_xy_m": float(np.mean(np.linalg.norm(dxy[:, -1, :], axis=-1))),
        "traj_rmse_yaw_rad": float(np.sqrt(np.mean(dyaw**2))),
    }


def _json_sanitize(obj):
    """
    Convert NaN/inf floats to JSON-safe values (None) for metrics logging.
    """
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        if obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    return obj


def _fmt_metric(x: float, *, ndigits: int = 4) -> str:
    if isinstance(x, (float, np.floating)) and (not np.isfinite(float(x))):
        return "n/a"
    return f"{float(x):.{ndigits}f}"


def _count_trainable_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def _count_unique_params_from_optimizer(optim: torch.optim.Optimizer) -> int:
    seen = set()
    total = 0
    for group in optim.param_groups:
        for p in group["params"]:
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            total += int(p.numel())
    return total


def _collect_val_metrics(
    model: FullPipelineRiskAndTraj,
    loader: DataLoader,
    device: torch.device,
    risk_label_smoothing: float,
    multitask_loss: MultiTaskLossWrapper,
    max_val_batches: Optional[int] = None,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Validation metrics + ``val_inference_ms_per_sample``.

    That latency is **not** "Mamba-only" and **not** a single-camera-frame rate:
    it is wall time for one **full training forward** over the **entire T_ctx**
    stack: for each time step, BEV (cached) or PointPillars neck (raw points),
    then reducer, then Mamba over the full token sequence, then risk + traj
    heads. It **includes** moving ``pts_seq`` to the device; it does **not**
    include DataLoader iteration / batch collation. For deployment / control
    rates, benchmark ``FullPipeline.step()`` (one frame, streaming hidden
    state) instead of this number.
    """
    model.eval()
    y_true = [[], [], []]
    y_score = [[], [], []]
    traj_pred_chunks: List[np.ndarray] = []
    traj_gt_chunks: List[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0
    latency_ms_total = 0.0
    latency_count = 0
    _use_val_amp = bool(use_amp) and (device.type == "cuda")
    nb = 0
    with torch.no_grad():
        for batch in loader:
            nb += 1
            # Make GPU timing honest: wait for prior work to finish before the timer.
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pts_seq_bt = _to_pts_seq_bt(batch.pts_seq, device)
            with torch.amp.autocast(device_type=device.type, enabled=_use_val_amp):
                logits, traj_pred = model(pts_seq_bt)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            B = int(logits.shape[0])
            latency_ms_total += (t1 - t0) * 1000.0
            latency_count += B
            risk_targets = batch.risk_targets().to(device=device, dtype=torch.float32)
            valid = batch.risk_label_valid.to(device=device, dtype=torch.float32)
            risk_loss = focal_bce(
                logits,
                risk_targets,
                valid_mask=valid,
                label_smoothing=risk_label_smoothing,
            )
            traj_gt = batch.traj_future_xyyaw.to(device=device, dtype=torch.float32)
            traj_loss = F.smooth_l1_loss(traj_pred, traj_gt)
            total_loss += float(multitask_loss(risk_loss, traj_loss).item()) * B
            total_samples += B

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy()
            target_np = risk_targets.detach().cpu().numpy()
            for h in range(3):
                m = valid_np[:, h] > 0.5
                if np.any(m):
                    y_true[h].append(target_np[m, h])
                    y_score[h].append(probs[m, h])
            traj_pred_chunks.append(traj_pred.detach().cpu().numpy())
            traj_gt_chunks.append(traj_gt.detach().cpu().numpy())
            if max_val_batches is not None and nb >= int(max_val_batches):
                print(
                    f"[val] max_val_batches={int(max_val_batches)} (partial val for smoke / debug).",
                    flush=True,
                )
                break

    if total_samples == 0:
        return {
            "val_loss": float("nan"),
            "val_sample_count": 0.0,
            "ap_risk_05s": float("nan"),
            "ap_risk_1s": float("nan"),
            "ap_risk_2s": float("nan"),
            "auc_risk_1s": float("nan"),
            "traj_rmse_all": float("nan"),
            "traj_ade_xy_m": float("nan"),
            "traj_fde_xy_m": float("nan"),
            "traj_rmse_yaw_rad": float("nan"),
            "val_inference_ms_per_sample": float("nan"),
        }

    yt = [np.concatenate(v) if v else np.zeros((0,), dtype=np.float32) for v in y_true]
    ys = [np.concatenate(v) if v else np.zeros((0,), dtype=np.float32) for v in y_score]
    tp = np.concatenate(traj_pred_chunks, axis=0)
    tg = np.concatenate(traj_gt_chunks, axis=0)
    tmet = _traj_metrics(tp, tg)
    return {
        "val_loss": total_loss / max(1, total_samples),
        "val_sample_count": float(total_samples),
        "ap_risk_05s": _safe_ap(yt[0], ys[0]),
        "ap_risk_1s": _safe_ap(yt[1], ys[1]),
        "ap_risk_2s": _safe_ap(yt[2], ys[2]),
        "auc_risk_1s": _safe_auc(yt[1], ys[1]),
        **tmet,
        "val_inference_ms_per_sample": latency_ms_total / max(1, latency_count),
    }


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
    device: Union[str, None] = None,
    seed: int = 0,
    num_workers: int = 4,
    positive_oversample_weight: float = 8.0,
    T_ctx: int = 40,
    traj_horizon: int = 10,
    grad_clip: float = 1.0,
    gradient_accumulation_steps: int = 2,
    cosine_eta_min: float = 1e-6,
    mamba_backend: str = "auto",
    weights_dir: Union[str, None] = None,
    save_weights: bool = True,
    run_name: Union[str, None] = None,
    stage: str = "a1",
    lr_neck: float = 5e-6,
    lr_rest: float = 1.67e-5,
    early_stop_patience: int = 8,
    early_stop_min_delta: float = 0.0,
    lr_warmup_iters: int = 500,
    bev_cache_root: Union[str, None] = None,
    extrinsics_convention: str = "auto",
    depth_scale_factor: float = 1.0,
    risk_label_smoothing: float = 0.05,
    temporal_dropout: float = 0.1,
    progress_log_every: int = 50,
    max_train_batches: Optional[int] = None,
    use_amp: bool = True,
) -> Dict[str, Union[float, str]]:
    b = validate_backbone(backbone)
    stage = validate_stage(stage)
    _seed_everything(seed)
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda_amp = bool(use_amp) and (dev.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    try:
        _ss = mp.get_sharing_strategy()
    except Exception:
        _ss = "n/a"
    print(
        f"[DataLoader] torch_sharing_strategy={_ss!r} num_workers={int(num_workers)}. "
        f"Workers>0 use start method 'spawn' (avoids fork+CUDA crashes on Linux). "
        f"Still stuck? set --num_workers 0 (most stable) or lower --batch_size.",
        flush=True,
    )
    _log_dev_shm_for_workers(int(num_workers))
    if int(num_workers) > 0:
        print(
            "[DataLoader] worker_init: OMP/BLAS thread caps=1 per worker (reduces child-process thrashing).",
            flush=True,
        )
    if use_cuda_amp:
        print(
            "[train] CUDA AMP (autocast + GradScaler) ON — lowers VRAM; OOM: lower --batch_size or pass --no_amp.",
            flush=True,
        )

    root = Path(log_root)
    root.mkdir(parents=True, exist_ok=True)
    name = run_name or f"{b}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(weights_dir) if weights_dir else (root / "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "backbone": b,
        "stage": stage,
        "data_root": str(data_root),
        "bev_cache_root": str(bev_cache_root) if bev_cache_root else "",
        "ckpt_path": str(ckpt_path),
        "device": str(dev),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "num_workers": int(num_workers),
        "T_ctx": int(T_ctx),
        "traj_horizon": int(traj_horizon),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "lr_warmup_iters": int(lr_warmup_iters),
        "cosine_eta_min": float(cosine_eta_min),
        "grad_clip": float(grad_clip),
        "positive_oversample_weight": float(positive_oversample_weight),
        "risk_label_smoothing": float(risk_label_smoothing),
        "temporal_dropout": float(temporal_dropout),
        "mamba_backend": str(mamba_backend),
        "lr_neck": float(lr_neck),
        "lr_rest": float(lr_rest),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "extrinsics_convention": str(extrinsics_convention),
        "depth_scale_factor": float(depth_scale_factor),
        "log_root": str(log_root),
        "run_name": str(name),
        "weights_dir": str(ckpt_dir),
        "save_weights": bool(save_weights),
        "progress_log_every": int(progress_log_every),
        "multitask_loss": "homoscedastic_uncertainty",
        "multitask_log_var_init": {"risk": 0.0, "traj": 0.0},
        "val_inference_ms_per_sample_note": (
            "Wall time for one training val forward over the full T_ctx window: "
            "per-frame perception (BEV or PointPillars) + reducer + Mamba on the full "
            "sequence + heads; includes H2D for pts_seq. Excludes DataLoader. "
            "Not comparable to robot Hz; for that benchmark FullPipeline.step() "
            "(one frame, streaming hidden state)."
        ),
        "max_train_batches": (int(max_train_batches) if max_train_batches is not None else None),
        "use_amp": bool(use_amp),
        "cuda_amp_active": bool(use_cuda_amp),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    data_root_path = Path(data_root)
    idx_path = data_root_path / "index.jsonl"
    if not idx_path.is_file():
        raise FileNotFoundError(f"index.jsonl not found: {idx_path}")
    rows = []
    with idx_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    scene_ids = sorted({int(r["scene_id"]) for r in rows})
    train_s, val_s, _ = scene_stratified_split(scene_ids, ratios=(0.75, 0.25, 0.0), seed=seed)

    ds_train = RiskDataset(
        root=str(data_root_path),
        T_ctx=T_ctx,
        scene_filter=train_s,
        traj_horizon=traj_horizon,
        bev_cache_root=bev_cache_root,
        extrinsics_convention=extrinsics_convention,
    )
    ds_val = RiskDataset(
        root=str(data_root_path),
        T_ctx=T_ctx,
        scene_filter=val_s,
        traj_horizon=traj_horizon,
        bev_cache_root=bev_cache_root,
        extrinsics_convention=extrinsics_convention,
    )

    risk_arr = ds_train.risk_1s_array()
    sample_w = np.where(risk_arr > 0.5, float(positive_oversample_weight), 1.0).astype(np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_w),
        num_samples=len(sample_w),
        replacement=True,
    )
    # After CUDA is touched in the parent, forked DataLoader workers on Linux
    # can exit unexpectedly. Using ``spawn`` avoids inheriting a broken CUDA
    # state. (``fork`` is the default on Linux; Windows already uses spawn.)
    val_num_workers = max(0, num_workers // 2)
    if max_train_batches is not None:
        # Smoke run: val with spawn workers is slow; main process loading is fine.
        val_num_workers = 0
    dataloader_mp_ctx = None
    if num_workers > 0 or val_num_workers > 0:
        dataloader_mp_ctx = mp.get_context("spawn")
    train_loader_kwargs = {
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": (dev.type == "cuda"),
        "collate_fn": collate_riskbatch,
    }
    if num_workers > 0:
        # Keep workers alive across epochs and lower prefetch depth to reduce
        # queuing pressure from large [T,B,C,H,W] BEV batches.
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 1
        train_loader_kwargs["multiprocessing_context"] = dataloader_mp_ctx
        train_loader_kwargs["worker_init_fn"] = _dataloader_worker_init
    val_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": val_num_workers,
        "pin_memory": (dev.type == "cuda"),
        "collate_fn": collate_riskbatch,
    }
    if val_num_workers > 0:
        val_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["prefetch_factor"] = 1
        val_loader_kwargs["multiprocessing_context"] = dataloader_mp_ctx
        val_loader_kwargs["worker_init_fn"] = _dataloader_worker_init
    train_loader = DataLoader(
        ds_train,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        ds_val,
        **val_loader_kwargs,
    )

    pp_cfg = PointPillarsConfig(ckpt_path=ckpt_path, device=str(dev))
    pp = PointPillarsNeckExtractor(pp_cfg)
    if stage == "a1":
        pp.freeze_all()
    else:
        pp.freeze_all()
        pp.unfreeze_neck()
    temporal = build_temporal(
        b,
        temporal_dropout=temporal_dropout,
        mamba_backend=mamba_backend,
    )
    model = FullPipelineRiskAndTraj(pp=pp, traj_horizon=traj_horizon, mamba=temporal)
    model.to(dev)
    multitask_loss = MultiTaskLossWrapper().to(dev)
    if dev.type == "cuda":
        total_gib = float(torch.cuda.get_device_properties(dev).total_memory) / (1024.0**3)
        if total_gib <= 16.0:
            print(
                "[stage_a_single_run] CUDA memory <= 16 GiB: if OOM, try --batch_size 16 "
                "and/or higher --gradient_accumulation_steps.",
                flush=True,
            )

    params_main = list(model.reducer.parameters()) + list(model.mamba.parameters()) + list(model.head.parameters()) + list(model.traj_head.parameters())
    if stage == "a2":
        neck_params = list(model.pp.model.neck.parameters())
        optim = torch.optim.AdamW(
            [
                {"params": neck_params, "lr": lr_neck, "weight_decay": weight_decay},
                {"params": params_main, "lr": lr_rest, "weight_decay": weight_decay},
                {"params": multitask_loss.parameters(), "lr": lr_rest, "weight_decay": 0.0},
            ]
        )
        base_lr = lr_rest
    else:
        optim = torch.optim.AdamW(
            [
                {"params": params_main, "lr": lr, "weight_decay": weight_decay},
                {"params": multitask_loss.parameters(), "lr": lr, "weight_decay": 0.0},
            ]
        )
        base_lr = lr
    print(
        f"[{name}] temporal backbone ONLY trainable parameters (fair compare): "
        f"{_count_trainable_params(model.mamba)}",
        flush=True,
    )
    print(
        f"[mamba_runtime] MambaTemporal backend='{mamba_backend}' train_device={dev} "
        f"torch.cuda.is_available()={torch.cuda.is_available()}",
        flush=True,
    )
    print(
        f"[{name}] total AdamW trainable parameters: "
        f"{_count_unique_params_from_optimizer(optim)}",
        flush=True,
    )

    total_optim_steps = max(1, (epochs * max(1, len(train_loader))) // max(1, gradient_accumulation_steps))
    warmup_steps = int(max(0, lr_warmup_iters))
    global_step = 0

    best_key = -float("inf")
    best_metrics: Dict[str, float] = {}
    best_epoch = -1
    patience = 0
    history_path = run_dir / "metrics.jsonl"
    for ep in range(epochs):
        model.train()
        optim.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        train_samples = 0
        ep_t0 = time.perf_counter()
        prev_batch_end = ep_t0
        for bi, batch in enumerate(train_loader):
            batch_t0 = time.perf_counter()
            data_wait_s = batch_t0 - prev_batch_end
            pts_seq_bt = _to_pts_seq_bt(batch.pts_seq, dev)
            with torch.amp.autocast(device_type=dev.type, enabled=use_cuda_amp):
                logits, traj_pred = model(pts_seq_bt)
                risk_targets = batch.risk_targets().to(device=dev, dtype=torch.float32)
                valid = batch.risk_label_valid.to(device=dev, dtype=torch.float32)
                risk_loss = focal_bce(
                    logits,
                    risk_targets,
                    valid_mask=valid,
                    label_smoothing=risk_label_smoothing,
                )
                traj_gt = batch.traj_future_xyyaw.to(device=dev, dtype=torch.float32)
                traj_loss = F.smooth_l1_loss(traj_pred, traj_gt)
                loss = multitask_loss(risk_loss, traj_loss)

            bsz = int(logits.shape[0])
            train_loss_sum += float(loss.item()) * bsz
            train_samples += bsz
            lo = loss / max(1, gradient_accumulation_steps)
            if use_cuda_amp:
                scaler.scale(lo).backward()
            else:
                lo.backward()
            if (bi + 1) % max(1, gradient_accumulation_steps) == 0 or (bi + 1) == len(train_loader):
                if use_cuda_amp:
                    scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                # Warmup + cosine update on optimizer-step granularity.
                if global_step < warmup_steps:
                    lr_scale = float(global_step + 1) / float(max(1, warmup_steps))
                else:
                    q = float(global_step - warmup_steps) / float(max(1, total_optim_steps - warmup_steps))
                    q = min(max(q, 0.0), 1.0)
                    lr_scale = cosine_eta_min / max(base_lr, 1e-12) + 0.5 * (1.0 + np.cos(np.pi * q)) * (
                        1.0 - cosine_eta_min / max(base_lr, 1e-12)
                    )
                for pg in optim.param_groups:
                    base = pg.get("initial_lr", pg["lr"])
                    if "initial_lr" not in pg:
                        pg["initial_lr"] = base
                    pg["lr"] = max(cosine_eta_min, float(pg["initial_lr"]) * lr_scale)
                if use_cuda_amp:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
            batch_t1 = time.perf_counter()
            prev_batch_end = batch_t1
            if (bi + 1) % max(1, int(progress_log_every)) == 0:
                print(
                    f"[{name}] epoch {ep + 1}/{epochs} micro_batch "
                    f"{bi + 1}/{len(train_loader)} loss={float(loss.item()):.4f}",
                    flush=True,
                )
            if max_train_batches is not None and (bi + 1) >= int(max_train_batches):
                print(
                    f"[{name}] max_train_batches={int(max_train_batches)} reached; "
                    f"ending epoch early (smoke / debug). Prefer --gradient_accumulation_steps 1 here.",
                    flush=True,
                )
                break

        _val_cap: Optional[int] = None
        if max_train_batches is not None:
            _val_cap = max(8, int(max_train_batches) * 2)
        val_metrics = _collect_val_metrics(
            model=model,
            loader=val_loader,
            device=dev,
            risk_label_smoothing=risk_label_smoothing,
            multitask_loss=multitask_loss,
            max_val_batches=_val_cap,
            use_amp=use_amp,
        )
        train_loss = train_loss_sum / max(1, train_samples)
        row = {
            "epoch": ep + 1,
            "train_loss": train_loss,
            "epoch_wall_time_s": float(time.perf_counter() - ep_t0),
            **val_metrics,
            "log_var_risk": float(multitask_loss.log_var_risk.detach().item()),
            "log_var_traj": float(multitask_loss.log_var_traj.detach().item()),
        }
        print(
            f"[{name}] epoch {ep + 1}/{epochs} train_loss={train_loss:.4f} "
            f"val_ap_risk_05s={_fmt_metric(float(val_metrics.get('ap_risk_05s', float('nan'))))} "
            f"val_ap_risk_1s={_fmt_metric(float(val_metrics.get('ap_risk_1s', float('nan'))))} "
            f"val_ADE_xy={_fmt_metric(float(val_metrics.get('traj_ade_xy_m', float('nan'))))} "
            f"val_latency_ms/sample={_fmt_metric(float(val_metrics.get('val_inference_ms_per_sample', float('nan'))), ndigits=3)}",
            flush=True,
        )
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_sanitize(row)) + "\n")
        current = float(val_metrics.get("ap_risk_1s", float("nan")))
        if np.isfinite(current) and current > (best_key + early_stop_min_delta):
            best_key = current
            best_metrics = val_metrics
            best_epoch = ep + 1
            patience = 0
            if save_weights:
                torch.save(model.state_dict(), ckpt_dir / f"{b}_best_val_ap.pt")
        else:
            patience += 1
        if save_weights:
            torch.save(model.state_dict(), ckpt_dir / f"{b}_latest.pt")
        if patience >= max(1, early_stop_patience):
            break

    out = {
        "backbone": b,
        "run_name": name,
        "log_dir": str(run_dir),
        "weights_dir": str(ckpt_dir),
        "best_epoch": float(best_epoch),
        "ap_risk_05s": float(best_metrics.get("ap_risk_05s", float("nan"))),
        "ap_risk_1s": float(best_metrics.get("ap_risk_1s", float("nan"))),
        "ap_risk_2s": float(best_metrics.get("ap_risk_2s", float("nan"))),
        "auc_risk_1s": float(best_metrics.get("auc_risk_1s", float("nan"))),
        "traj_rmse_all": float(best_metrics.get("traj_rmse_all", float("nan"))),
        "traj_ade_xy_m": float(best_metrics.get("traj_ade_xy_m", float("nan"))),
        "traj_fde_xy_m": float(best_metrics.get("traj_fde_xy_m", float("nan"))),
        "traj_rmse_yaw_rad": float(best_metrics.get("traj_rmse_yaw_rad", float("nan"))),
        "val_inference_ms_per_sample": float(
            best_metrics.get("val_inference_ms_per_sample", float("nan"))
        ),
        "log_var_risk": float(multitask_loss.log_var_risk.detach().item()),
        "log_var_traj": float(multitask_loss.log_var_traj.detach().item()),
    }
    with (run_dir / "val_metrics_final.json").open("w", encoding="utf-8") as f:
        json.dump(_json_sanitize(out), f, indent=2)
    return out


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
    progress_log_every: int = 50,
    max_train_batches: Optional[int] = None,
    use_amp: bool = True,
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
            progress_log_every=progress_log_every,
            max_train_batches=max_train_batches,
            use_amp=use_amp,
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
                f"{_fmt_metric(float(m.get('ap_risk_05s', float('nan')))):>9} "
                f"{_fmt_metric(float(m.get('ap_risk_1s', float('nan')))):>9} "
                f"{_fmt_metric(float(m.get('ap_risk_2s', float('nan')))):>9} "
                f"{_fmt_metric(float(m.get('auc_risk_1s', float('nan')))):>9} "
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
                f"{_fmt_metric(float(m.get('traj_rmse_all', float('nan')))):>10} "
                f"{_fmt_metric(float(m.get('traj_ade_xy_m', float('nan')))):>10} "
                f"{_fmt_metric(float(m.get('traj_fde_xy_m', float('nan')))):>10} "
                f"{_fmt_metric(float(m.get('traj_rmse_yaw_rad', float('nan')))):>10}",
                flush=True,
            )
        print(f"\nTensorBoard: tensorboard --logdir {log_dir_base}", flush=True)

        summary_path = log_dir_base / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(_json_sanitize(results), f, indent=2)
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
    ap.add_argument(
        "--progress_log_every",
        type=int,
        default=50,
        help="Print train data/step timing every N batches.",
    )
    ap.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
        help="If >0, stop each training epoch after this many micro-batches (smoke/debug). 0=full epoch.",
    )
    ap.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision (higher VRAM, sometimes more stable).",
    )
    args = ap.parse_args()
    _mtb: Optional[int] = int(args.max_train_batches) if int(args.max_train_batches) > 0 else None

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


if __name__ == "__main__":
    main()
