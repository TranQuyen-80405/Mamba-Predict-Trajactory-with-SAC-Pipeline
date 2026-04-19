# Optimized training strategy — Stage A (Divide & Conquer)

> **Role:** Strategic guide for **compute-efficient** Stage A training and **paper-ready ablation** design. Complements `docs/strategy_train_stage_A.md` (contracts), `docs/strategy_experiment_protocol.md` (small-scale recipe), and `docs/strategy_full_pipeline.md` (architecture owner).
>
> **Code map:** `PointPillars_module/module_pointpillar.py` (`PointPillarsNeckExtractor`), `PointPillars_module/models/{full_pipeline.py, spatial_reducer.py, mamba_temporal.py, risk_head.py}`, `create_dataset_module/{generator.py, risk_dataset.py}`, `PointPillars_module/losses.py` (`focal_bce`).
>
> **Implementation status (2026-04-19):** feature-cache path is now available via
> `scripts/cache_pointpillars_bev.py` + `RiskDataset(bev_cache_root=...)` +
> `train_stage_a_compare.py --bev_cache_root ...`. Legacy sections still marked
> "Proposed" describe optional extensions (format/layout variants), not baseline usage.

---

## 1. Divide & Conquer for the perception pipeline

### 1.1 Feature caching (one-time PointPillars / BEV cost)

**Bottleneck.** In `FullPipeline.forward`, each training sample runs `PointPillarsNeckExtractor.extract_neck_forward` **once per context frame** (\(T_\mathrm{ctx}=10\)), producing a BEV tensor **`(B, 384, 248, 216)`** per step before `SpatialReducer` (see `PointPillars_module/models/full_pipeline.py`). Voxelization + CNN neck dominate **GPU time** and **memory** versus `MambaTemporal` + `RiskHead`.

**Idea (cache BEV).** Pre-compute and store **frozen** BEV features for every **(rollout, frame index \(\tau\))** pair used by the dataset:

1. Load `Trajectory` from `.npz` (`Trajectory.from_npz`).
2. For each frame \(\tau\), convert depth → \((N,4)\) points (same path as `RiskDataset.__getitem__` today, or reuse saved points if you add them).
3. Run **`extract_neck`** (inference, `@torch.no_grad()`) or **`extract_neck_forward`** with **`PointPillarsNeckExtractor.freeze_all()`** so no gradient is needed.
4. Save **`bev[τ] ∈ \mathbb{R}^{384 \times 248 \times 216}`** as **float16** or **float32** to disk (`.pt` per frame, or stacked `.npz` per rollout, or HDF5).

**Training loop (ablation mode).** `SpatialReducer` → `MambaTemporal` → `RiskHead` consume **cached BEV sequences** only — **no** `pillar_layer` / `pillar_encoder` / `backbone` forward during training.

**Rule of thumb.** One offline cache pass replaces **\(O(\text{epochs} \times T_\mathrm{ctx})\)** heavy forwards per sample with **\(O(1)\)** disk reads — often **~70–80%** reduction in **training GPU-hours** when PointPillars stays frozen (A1 regime). Exact savings depend on batch size, I/O, and whether A2 unfreezes `pp.neck` (then partial re-cache or hybrid).

### 1.2 Decoupled training (freeze PointPillars, train head)

**Default A1** (`docs/strategy_full_pipeline.md` §5.4): **`PointPillarsNeckExtractor.freeze_all()`** — train only **`SpatialReducer`**, **`MambaTemporal`**, **`RiskHead`**.

**Why it remains robust**

- KITTI-pretrained BEV is a **strong geometric prior**; the task is to map **BEV dynamics** to **contact lookahead** labels (`lookahead_any` in `create_dataset_module/generator.py`), not to relearn low-level pillars from scratch on limited PyBullet data.
- **Domain gap** is mitigated by preprocessing (`DepthPreprocessConfig`, `module_pointpillar.py`) and optional **A2** (`unfreeze_neck()` / `set_trainable(["neck"])`).
- **Risk labels** are **simulator ground truth** from `contact_flag`, not from the encoder — freezing PP does not bias labels; it constrains **representation**.

**Ablation narrative.** “Modular / decoupled” = frozen (or cached) encoder + trainable temporal head — standard in efficient perception pipelines and aligns with **Stage B**, where the full stream is frozen and only SAC MLPs learn.

---

## 2. Implementation guidance

> **Implemented baseline:** BEV precompute + training from cache is available. The items
> below remain as extension ideas for alternative storage/collate designs.

### 2.1 Script sketch: `cache_features.py` (repo root or `PointPillars_module/scripts/`)

**Purpose:** Walk the same **`index.jsonl`** + `.npz` tree as `RiskDataset`, emit cached BEV per frame \(\tau\).

**Dependencies:** `Trajectory`, `PointPillarsNeckExtractor`, `PointPillarsConfig`, depth→points path (reuse `_preprocess_depth_to_pts` from `create_dataset_module/risk_dataset.py` or factor a shared helper).

**Pseudocode:**

```python
# cache_features.py — outline
for each row in index.jsonl:
    traj = Trajectory.from_npz(root / row["path"])
    for tau in range(traj.T):
        pts4 = depth_frame_to_pts(traj, tau)   # (N,4) float32, same as RiskDataset
        bev = extractor.extract_neck([pts4]).feature  # (1, 384, 248, 216)
        save_tensor(out_dir / f"{stem}_f{tau:05d}.pt", bev.squeeze(0).half())
```

**Index file:** e.g. `cache_index.jsonl` with `{ "path": "...", "scene_id", "rollout_id", "tau", "bev_path" }` for reproducibility.

**Storage (order of magnitude).** Per frame: \(384 \times 248 \times 216 \times 2\) bytes ≈ **40 MiB** in float16; scale by total frames. Use **compression** or **chunked HDF5** if disk-bound.

### 2.2 Extending `RiskDataset` for ablations

**Goal:** `__getitem__` returns the same **`RiskSample`** contract where **`pts_seq`** is either:

- **Raw:** list of \((N_i,4)\) tensors (current behavior), or  
- **Cached BEV:** list of **`(384, 248, 216)`** tensors — then **`FullPipeline`** must expose an alternate forward **`forward_from_bev(bev_seq_bt)`** that skips `extract_neck_forward` and starts at **`SpatialReducer`**.

**Suggested API (minimal):**

```python
class RiskDataset(Dataset):
    def __init__(..., bev_cache_root: Optional[str] = None, ...):
        self.bev_cache_root = Path(bev_cache_root) if bev_cache_root else None
```

- If **`bev_cache_root` is `None`**: current depth → points → (optional) on-the-fly neck in training.  
- If **set**: load **`bev[tau]`** for \(\tau \in [t-T_\mathrm{ctx}+1, t]\); **`RiskSample.pts_seq`** could be repurposed to hold BEV tensors **or** add an optional field **`bev_seq`** and extend **`collate_riskbatch`** / **`FullPipeline`** accordingly.

**Ablation fairness:** Fix **seed**, **split** (`scene_stratified_split`), and **cached checkpoint** of `PointPillarsNeckExtractor` so all temporal baselines see **identical BEV inputs**.

**Collate:** Reuse **`collate_riskbatch`** if `RiskSample` keeps a single sequence field; otherwise add **`collate_bev_riskbatch`** that stacks BEV to **`(B, T_\mathrm{ctx}, 384, 248, 216)`** and feeds **`forward_from_bev`**.

---

## 3. Research tables (templates)

Fill from **`utils/metrics.py`** (when available) or sklearn/torchmetrics; report **mean ± std** over \(\geq 3\) seeds.

### Table 1 — Ablation: temporal module

| Method | AUC-PR (\(risk_{1s}\)) \(\uparrow\) | AUC-ROC \(\uparrow\) | Macro-F1 \(\uparrow\) | Params (M) \(\downarrow\) | Latency (ms) \(\downarrow\) |
|--------|-------------------------------------|----------------------|------------------------|---------------------------|-----------------------------|
| Mamba (`MambaTemporal`, `backend="mamba"`) | | | | | |
| GRU (`MambaTemporal`, `backend="gru"`) | | | | | |
| LSTM* (if added as `nn.LSTM` baseline) | | | | | |
| Transformer* (e.g. `nn.TransformerEncoder`, \(L=160\)) | | | | | |

\*Optional baselines not in the default repo; keep **same** `SpatialReducer` + `RiskHead` and **cached BEV** for fair comparison.

### Table 2 — Performance by horizon

| Model | AUC-PR \(@\,0.5\mathrm{s}\) | AUC-PR \(@\,1.0\mathrm{s}\) | AUC-PR \(@\,2.0\mathrm{s}\) | Brier (avg) \(\downarrow\) |
|-------|----------------------------|-----------------------------|-----------------------------|----------------------------|
| Best temporal (T1) | | | | |
| Ablated variant (T1 − component) | | | | |

Labels: `risk_05s`, `risk_1s`, `risk_2s` from `lookahead_any` (`create_dataset_module/generator.py`); loss: **`focal_bce`** (`PointPillars_module/losses.py`).

### Table 3 — Resource efficiency

| Setting | Train wall-clock (h) \(\downarrow\) | Peak GPU mem (GB) \(\downarrow\) | AUC-PR \(@\,1\mathrm{s}\) \(\uparrow\) |
|---------|-------------------------------------|----------------------------------|----------------------------------------|
| End-to-end Stage A (`FullPipeline.forward`, PP unfrozen or forward each step) | | | |
| Modular (cached BEV + frozen PP, train `SpatialReducer` + `MambaTemporal` + `RiskHead`) | | | |
| Modular + A2 neck only (partial PP grad) | | | |

---

## 4. Robustness & contributions (paper framing)

### 4.1 Three main contributions (aligned with repo narrative)

1. **Efficient spatiotemporal modeling for risk** — **`MambaTemporal`** (SSM, linear-in-\(L\) sequence processing for \(L = T_\mathrm{ctx} \times N_t = 160\)) vs quadratic attention; **`SpatialReducer`** compresses BEV **`(384,248,216)`** → **`(16,256)`** tokens per frame.
2. **Perception-aware policy learning (unified pipeline)** — Shared **`Trajectory`** / **`RiskDataset`** / **`lookahead_any`** labels bridge **Stage A** (supervised risk) and **Stage B** (frozen **`FullPipeline.step`** → \(r_\mathrm{risk}\)); see `docs/strategy_full_pipeline.md`.
3. **Decoupled training for stability** — **Freeze** / **cache** **`PointPillarsNeckExtractor`** while adapting **`MambaTemporal`** + **`RiskHead`** reduces optimization difficulty and **GPU cost**, consistent with default **A1** and **Stage B S1**.

### 4.2 Robustness scenarios (evaluation protocol)

| Scenario | Implementation hook | Metric |
|----------|---------------------|--------|
| **Sensor noise** | `DataGenConfig.depth_noise_std`, `drop_pixel_prob`, `camera_jitter_deg` (`create_dataset_module/generator.py`) | AUC-PR / Brier on **val**; optional **Stage B** collision rate |
| **Missing frames** | Subsample or zero-drop frames in `pts_seq` / cached BEV sequence | AUC-PR degradation curve vs drop rate |
| **OOD** | New `n_scenes` / obstacle distribution not in train | AUC-PR, **ECE**, **collision rate**; detect **miscalibration** (histogram of **`sigmoid(logits)`**) |

---

## 5. Economic & environmental summary

Let **\(H_\mathrm{PP}\)** = GPU-hours spent in PointPillars per epoch and **\(H_\mathrm{tail}\)** = hours in `SpatialReducer`+`MambaTemporal`+`RiskHead`. **Caching + freeze** targets:

\[
H_\text{train} \approx H_\text{IO} + H_\mathrm{tail} \ll \sum_{\text{epochs}} \big( H_\mathrm{PP} + H_\mathrm{tail} \big).
\]

**GPU-hours** and **energy (kWh)** scale roughly linearly with **\(H\)**, so **modular training** lowers **cost** and **estimated carbon footprint** (e.g. ML CO₂ calculators per cloud region) **without changing** the **label definition** (`lookahead_any` on `contact_flag`). Final **quality** should be reported **with and without** A2 neck adaptation to show no undue regression.

---

## 6. Changelog

| Date | Version | Summary |
|------|---------|---------|
| 2026-04-18 | v1.0 | Initial optimized Stage A strategy: BEV caching, decoupled training, implementation sketch, paper tables, robustness hooks, economics. |
