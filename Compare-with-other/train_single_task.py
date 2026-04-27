import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

# Setup paths to import from main repo
_PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_PKG_ROOT / "PointPillars_module"))
sys.path.insert(0, str(_PKG_ROOT / "create_dataset_module"))

from PointPillars_module.losses import focal_bce
from PointPillars_module.types import DepthPreprocessConfig
from create_dataset_module.risk_dataset import RiskDataset, collate_riskbatch, scene_stratified_split
from PointPillars_module.training.stage_a_single_run import _unique_scenes_from_index, _traj_error_metrics

# Import our custom models
from trajectory_models import RecurrentTrajectoryPredictor, TransformerTrajectoryPredictor, MambaTrajectoryPredictor
from risk_models import MLPRiskPredictor, OccupancyHeatmapPredictor, SafetyValueEstimator

def build_model(task: str, model_name: str, in_channels: int, horizon: int, device: torch.device):
    if task == 'trajectory':
        if model_name == 'lstm':
            return RecurrentTrajectoryPredictor(in_channels=in_channels, hidden_size=128, horizon=horizon, rnn_type='LSTM').to(device)
        elif model_name == 'transformer':
            return TransformerTrajectoryPredictor(in_channels=in_channels, d_model=128, nhead=4, num_layers=2, horizon=horizon).to(device)
        elif model_name == 'mamba':
            return MambaTrajectoryPredictor(in_channels=in_channels, d_model=128, d_state=16, d_conv=4, expand=2, horizon=horizon).to(device)
    elif task == 'risk':
        if model_name == 'mlp':
            return MLPRiskPredictor(in_channels=in_channels, hidden_size=128, num_targets=3).to(device)
        elif model_name == 'heatmap':
            return OccupancyHeatmapPredictor(in_channels=in_channels, num_targets=3).to(device)
        elif model_name == 'safety':
            return SafetyValueEstimator(in_channels=in_channels, hidden_size=128, num_targets=3).to(device)
    raise ValueError(f"Unknown task/model combination: {task}/{model_name}")

def evaluate_risk(logits_cat, t_cat, v_cat):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    out = {}
    probs = 1.0 / (1.0 + np.exp(-logits_cat))
    names = ["risk_05s", "risk_1s", "risk_2s"]
    brier_scores = []
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
    return out

def get_pts_tensor(pts_seq, dev):
    if isinstance(pts_seq, list):
        # pts_seq is list of length T, each containing list of length B tensors
        t_list = []
        for time_step in pts_seq:
            t_list.append(torch.stack([p for p in time_step]))
        x = torch.stack(t_list).to(dev) # (T, B, C, H, W)
    else:
        # pts_seq is already a tensor (T, B, C, H, W) from collate_riskbatch
        x = pts_seq.to(dev)
    
    # Always ensure output is (Batch, Time, Channels, Height, Width)
    if x.ndim == 5:
        x = x.transpose(0, 1).contiguous()
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True, choices=['trajectory', 'risk'])
    ap.add_argument("--model", type=str, required=True, choices=['lstm', 'transformer', 'mamba', 'mlp', 'heatmap', 'safety'])
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--bev_cache_root", type=str, required=True)
    ap.add_argument("--log_root", type=str, default="runs/compare_baselines")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--T_ctx", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    preprocess_cfg = DepthPreprocessConfig()
    scenes = _unique_scenes_from_index(Path(args.data_root))
    tr_s, va_s, _ = scene_stratified_split(scenes, (0.75, 0.25, 0.0), seed=0)
    
    train_ds = RiskDataset(args.data_root, T_ctx=args.T_ctx, preprocess_cfg=preprocess_cfg, scene_filter=tr_s, traj_horizon=10, bev_cache_root=args.bev_cache_root)
    val_ds = RiskDataset(args.data_root, T_ctx=args.T_ctx, preprocess_cfg=preprocess_cfg, scene_filter=va_s, traj_horizon=10, bev_cache_root=args.bev_cache_root)

    w_arr = train_ds.risk_1s_array()
    weights = torch.as_tensor(np.where(w_arr > 0.5, 8.0, 1.0), dtype=torch.double)
    train_sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_riskbatch, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_riskbatch, num_workers=args.num_workers)

    model = build_model(args.task, args.model, in_channels=384, horizon=10, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    rn = f"{args.task}_{args.model}"
    run_dir = Path(args.log_root) / rn
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    
    log_file = run_dir / "train.log"
    def log_print(msg):
        print(msg, flush=True)
        with log_file.open("a") as f:
            f.write(msg + "\n")

    log_print(f"[{rn}] Starting training {args.task} with {args.model}")

    best_val_metric = None
    epochs_no_improve = 0
    start_epoch = 0
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    
    last_ckpt_path = ckpt_dir / f"last_checkpoint_{args.task}_{args.model}.pt"
    if last_ckpt_path.exists():
        checkpoint = torch.load(last_ckpt_path, map_location=dev)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_metric = checkpoint['best_val_metric']
        epochs_no_improve = checkpoint['epochs_no_improve']
        log_print(f"[{rn}] Resumed training from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        for micro_i, batch in enumerate(train_loader):
            x = get_pts_tensor(batch.pts_seq, dev)

            opt.zero_grad()
            
            if args.task == 'trajectory':
                out = model(x)
                traj_gt = batch.traj_future_xyyaw.float().to(dev)
                loss = F.smooth_l1_loss(out, traj_gt)
            else:
                if args.model == 'heatmap':
                    traj_gt = batch.traj_future_xyyaw[:, :, :2].float().to(dev)
                    normalized_traj = torch.clamp(traj_gt / 10.0, -1.0, 1.0)
                    out, _ = model(x, normalized_traj)
                else:
                    out = model(x)
                    
                targets = torch.stack([batch.risk_05s, batch.risk_1s, batch.risk_2s], dim=1).float().to(dev)
                valid = batch.risk_label_valid.float().to(dev)
                loss = focal_bce(out, targets, gamma=2.0, weight=(1.0, 0.8, 0.5), valid_mask=valid)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item()

            if (micro_i + 1) % 100 == 0 or (micro_i + 1) == len(train_loader):
                log_print(f"[{rn}] epoch {epoch + 1}/{args.epochs} batch {micro_i + 1}/{len(train_loader)} loss={loss.item():.4f}")

        # Validation
        model.eval()
        preds_all, gt_all, valid_all = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x = get_pts_tensor(batch.pts_seq, dev)

                if args.task == 'trajectory':
                    out = model(x)
                    traj_gt = batch.traj_future_xyyaw.float().to(dev)
                    preds_all.append(out.cpu().numpy())
                    gt_all.append(traj_gt.cpu().numpy())
                else:
                    if args.model == 'heatmap':
                        traj_gt = batch.traj_future_xyyaw[:, :, :2].float().to(dev)
                        normalized_traj = torch.clamp(traj_gt / 10.0, -1.0, 1.0)
                        out, _ = model(x, normalized_traj)
                    else:
                        out = model(x)
                        
                    targets = torch.stack([batch.risk_05s, batch.risk_1s, batch.risk_2s], dim=1).float().cpu()
                    valid = batch.risk_label_valid.float().cpu()
                    preds_all.append(out.cpu().numpy())
                    gt_all.append(targets.numpy())
                    valid_all.append(valid.numpy())

        metrics = {}
        if args.task == 'trajectory':
            pred_cat = np.concatenate(preds_all, axis=0)
            gt_cat = np.concatenate(gt_all, axis=0)
            metrics = _traj_error_metrics(pred_cat, gt_cat)
            msg = f"val_ADE_xy={metrics['traj_ade_xy_m']:.4f} val_FDE_xy={metrics['traj_fde_xy_m']:.4f}"
        else:
            pred_cat = np.concatenate(preds_all, axis=0)
            gt_cat = np.concatenate(gt_all, axis=0)
            v_cat = np.concatenate(valid_all, axis=0)
            metrics = evaluate_risk(pred_cat, gt_cat, v_cat)
            msg = f"val_ap_risk_05s={metrics.get('ap_risk_05s', 0):.4f} val_ap_risk_1s={metrics.get('ap_risk_1s', 0):.4f}"

        log_print(f"[{rn}] epoch {epoch + 1}/{args.epochs} train_loss={epoch_loss/len(train_loader):.4f} " + msg)

        # Tensorboard
        writer.add_scalar("train/loss_epoch", epoch_loss/len(train_loader), epoch)
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", float(v), epoch)

        # Early Stopping Logic
        if args.task == 'trajectory':
            current_metric = metrics['traj_ade_xy_m']
            improved = best_val_metric is None or current_metric < best_val_metric
        else:
            current_metric = metrics.get('ap_risk_1s', 0)
            improved = best_val_metric is None or current_metric > best_val_metric

        if improved:
            best_val_metric = current_metric
            epochs_no_improve = 0
            
            # Remove old best model if exists
            for old_best in ckpt_dir.glob(f"best_ep*_{args.task}_{args.model}.pt"):
                old_best.unlink()
                
            best_model_name = f"best_ep{epoch + 1}_{args.task}_{args.model}.pt"
            torch.save(model.state_dict(), ckpt_dir / best_model_name)
            log_print(f"[{rn}] New best model saved as {best_model_name}! Metric: {current_metric:.4f}")
            
            best_summary_path = run_dir / f"val_metrics_best_{args.task}_{args.model}.json"
            with best_summary_path.open("w") as f:
                json.dump(metrics, f, indent=2)
        else:
            epochs_no_improve += 1
            log_print(f"[{rn}] No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= args.patience:
                log_print(f"[{rn}] Early stopping triggered after {epoch + 1} epochs!")
                break
                
        # Save last checkpoint for resuming
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'best_val_metric': best_val_metric,
            'epochs_no_improve': epochs_no_improve
        }, last_ckpt_path)

    # Save final model
    torch.save(model.state_dict(), ckpt_dir / f"final_ep{args.epochs}_{args.task}_{args.model}.pt")
    
    summary_path = run_dir / f"val_metrics_final_{args.task}_{args.model}.json"
    with summary_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    
    log_print(f"[{rn}] Done! Logs and checkpoints saved in {run_dir}")

if __name__ == '__main__':
    main()
