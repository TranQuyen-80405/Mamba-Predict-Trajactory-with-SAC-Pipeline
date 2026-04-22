# Stage A training: failed runs, errors, causes, and fixes

This document summarizes **failed runs**, **errors**, **root causes**, and **mitigations** when training the Stage A multi-backbone comparison (Mamba, GRU, LSTM, Transformer) with PointPillars + BEV cache, 2× GPU, and `torchrun`.

---

## 1. Environment & build (CPU vs GPU, native extensions)

| Symptom / error | Typical cause | What to do |
|-----------------|---------------|------------|
| `causal-conv1d` / `mamba-ssm` build fails in setup script | Mamba stack needs CUDA + NVCC; CPU-only install breaks | In the setup script, install the Mamba stack only when `TORCH_MODE` (or equivalent) is GPU; skip on CPU with a clear warning |
| `ModuleNotFoundError: pointpillars.ops.voxel_op` | `voxel_op` not built, or `CUDA_HOME` points to wrong toolkit | Editable install with PyTorch present: `pip install -e . --no-build-isolation`; set `CUDA_HOME` to the CUDA version matching `torch.version.cuda` (e.g. 12.8) |
| `nvcc` vs PyTorch CUDA mismatch (e.g. 12.9 vs 12.8) | System CUDA toolkit is newer than PyTorch’s build | Point `CUDA_HOME` at the same major/minor as PyTorch’s bundled CUDA |
| `ModuleNotFoundError: torch` during `pip install -e` | Build isolation without torch in the build env | Use `--no-build-isolation` for torch-dependent native packages |
| `AssertionError` in neck test (CPU vs CUDA) | Normal numerical gap between CPU and CUDA paths (tolerance too tight) | Loosen `atol`/`rtol` in the comparison test (e.g. `atol=1e-2` for BEV features) if the rest of the pipeline is healthy |

---

## 2. GPU out-of-memory (OOM)

| Symptom / error | Typical cause | What to do |
|-----------------|---------------|------------|
| `torch.OutOfMemoryError` / killed process | Large `batch_size` / `T_ctx`, leaked processes on GPU, or 2 DDP ranks on the same node | Kill stale jobs; lower `--batch_size`; raise `--gradient_accumulation_steps` to preserve effective batch; try `nproc_per_node=1` to isolate; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128` |

---

## 3. DataLoader, shared memory (`/dev/shm`), `num_workers`

| Symptom / error | Typical cause | What to do |
|-----------------|---------------|------------|
| `RuntimeError: unable to allocate shared memory (shm)` | DDP (2 processes) × multiple workers × large batches exceed `/dev/shm` (Docker default 64MB is a common cap) | **Most stable on constrained hosts:** `--num_workers 0`. Or **raise shm**: e.g. Docker `--shm-size=8g` (or higher) / `--ipc=host` |
| `TORCH_SHARING_STRATEGY=file_system` still fails with `num_workers=2/4` | `file_system` reduces shm pressure but workers still pass large tensors through queues; very limited hosts can still break | Treat `num_workers` as limited by **host RAM + `/dev/shm` size**; increase them in reality, or keep `0` |
| `DataLoader worker ... killed` (host OOM) | Not enough system RAM for workers + training ranks | Lower `num_workers`, `batch_size`, or `prefetch_factor` |

**Practical note:** With 2× GPU DDP, `num_workers=0` is often acceptable if the bottleneck is BEV I/O; prefer a solid BEV cache and (if hardware allows) a larger `/dev/shm` rather than relying on env flags alone.

---

## 4. Data, labels, and NaN validation metrics

| Symptom / error | Typical cause | What to do |
|-----------------|---------------|------------|
| `val_ap_risk_05s = nan` (or 0) | No positive examples for that horizon on the val split (after windowing) | Regenerate or tune the datagen preset so each horizon has positives on val; run a **dataset stats** script before long training |
| AP/AUC undefined when there are no positives | Standard AP/AUC are ill-defined with zero positives | In code: log `0.0` (and document), plus a preflight if val has rows but no positives for a horizon |
| Warning: “0 points after point_cloud_range” | Wrong `extrinsics_convention` or `depth_scale` | Use `extrinsics_convention=auto` (when supported) and align `depth_scale_factor` across BEV cache, datagen, and training |

---

## 5. BEV cache & checkpoints

| Symptom / error | Typical cause | What to do |
|-----------------|---------------|------------|
| `FileNotFoundError` for BEV / cache | `data_root` changed but cache path is stale or missing | Default layout: `<data_root>_bev_cache`; build cache with the caching script, same extrinsics and spatial downsample as training |
| Wrong neck weights | Default pointed at `epoch_160.pth` while the repo standardised on `epoch_160_raw.pth` | Pass `--ckpt` to `epoch_160_raw.pth` (or the file you actually ship); tests resolve **`epoch_160_raw.pth` first**, then `epoch_160.pth` |
| Heavy I/O from many small `.npz` / cache files | Many small files + multiple readers | BEV cache + (optional) LRU in the dataset; for large-scale use, consider packed formats (LMDB/HDF5) later |

---

## 6. DDP, switching backbone, `ChildFailedError`

| Symptom / error | Typical cause | What to do |
|-----------------|---------------|------------|
| Failure on the **second** model (e.g. after Mamba, GRU fails at `DDP` init) | **Incorrect collectives:** e.g. `barrier()` only on non-rank-0 while rank-0 is still doing I/O — NCCL can desync | **Fix:** all ranks `barrier()` after training, rank-0 writes checkpoints, then **another `barrier()`** before returning so every rank starts the next backbone together |
| `SignalException: signal 2` | User sent SIGINT (Ctrl+C) or `kill` | Not a training bug; external stop signal |
| Warning: `destroy_process_group() was not called` | Process exits without `dist.destroy_process_group()` | Can call it at the end of `main` in DDP runs; often a resource warning only, not a failed run |

---

## 7. Losses & PyTorch compatibility

| Symptom / error | Typical cause | What to do |
|-----------------|---------------|------------|
| `TypeError: binary_cross_entropy_with_logits() got an unexpected keyword argument 'label_smoothing'` | Some PyTorch builds do not support `label_smoothing` on this op | **Manual label smoothing for binary / multi-label BCE:** `targets' = targets * (1-ε) + 0.5*ε`, then call `binary_cross_entropy_with_logits` without `label_smoothing=` |

---

## 8. Example stable training configuration (as used in this repo)

- **2-GPU DDP, avoid shm issues:** `num_workers=0`, `export TORCH_SHARING_STRATEGY=file_system` (supplemental, not a substitute for a larger `/dev/shm`), `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`.
- **Better GPU utilisation with similar effective batch:** e.g. `--batch_size 4` and `--gradient_accumulation_steps 8` instead of `1×32` when VRAM allows — **watch for OOM**.

---

## 9. Useful scripts & artifacts in the repo

- Dataset stats without regenerating: `run_dataset_stats.py` / `scripts/check_dataset_stats.py`
- BEV cache: `scripts/cache_pointpillars_bev.py` (match `extrinsics` + `spatial_downsample` with training)
- Logs: often `tee` to `runs/stage_a_compare/train.log` + TensorBoard on `log_root`

---

*Compiled from debug iterations: Ubuntu + venv, PyTorch CUDA, 2× GPU, four-backbone compare, BEV cache, DDP.*  
*Append new rows as new failure modes appear.*
