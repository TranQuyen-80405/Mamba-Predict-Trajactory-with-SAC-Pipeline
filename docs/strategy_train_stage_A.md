# Stage A — Perception & Risk Prediction (Training “Bible”)

> **Scope:** end-to-end **supervised pretrain** of the perception stream that predicts multi-horizon collision risk from sequences of point clouds. This document consolidates `docs/strategy_full_pipeline.md` §3–§5, `docs/strategy_create_trajectory_label.md`, and the concrete implementations under `PointPillars_module/` and `create_dataset_module/`.
>
> **Architectural tiebreaker:** `docs/strategy_full_pipeline.md` remains authoritative if any detail conflicts; update this file in the same PR when Stage A code or contracts change.

---

## 1. Overview

### 1.1 Objective

**Stage A** learns a mapping from **recent sensor geometry** (depth-derived point clouds in a LiDAR-style frame) to **three binary collision-risk targets** over future windows:

| Output (after `sigmoid`) | Meaning |
|--------------------------|---------|
| \(p_{0.5\mathrm{s}}\) | Probability of **≥ 1 contact** in the next **0.5 s** |
| \(p_{1\mathrm{s}}\) | Probability of **≥ 1 contact** in the next **1.0 s** |
| \(p_{2\mathrm{s}}\) | Probability of **≥ 1 contact** in the next **2.0 s** |

All horizons assume **20 Hz** control (\(\Delta t = 0.05\,\mathrm{s}\)); see §4.

### 1.2 Role in the two-stage pipeline

| Deliverable | Consumer | Stage B usage |
|-------------|----------|---------------|
| Trained weights for `PointPillarsNeckExtractor` (partially: neck in A2), `SpatialReducer`, `MambaTemporal`, `RiskHead` | Frozen **risk branch** | At rollout time: `FullPipeline.step` → `p_\mathrm{risk} \in [0,1]^3` |
| **No** dedicated “state encoder” for RL | SAC | **Not used** as policy/value input in the default plan |

> **Critical invariant (default Stage B):** the SAC **Actor** and **twin Critics** consume **`ProprioState` only** (MLPs on proprio vectors). The temporal hidden state \(h\) inside `MambaTemporal` exists **only** along the perception / risk branch. **Do not** concatenate BEV, tokens, or \(h_T\) into the SAC state without an explicit architecture change and doc update (`strategy_full_pipeline.md` §1.2).

---

## 2. Data pipeline

### 2.1 From simulation to disk

| Step | Component | Path / symbol |
|------|-----------|----------------|
| Rollout | `DataGenerator.run()` | `create_dataset_module/generator.py` |
| Config | `DataGenConfig` | `PointPillars_module/data_contracts.py` |
| Artifact | `Trajectory` → one `.npz` per rollout | `Trajectory.to_npz` / `from_npz` in `data_contracts.py` |

**On-disk trajectory** (`Trajectory`, §3.1 in `strategy_full_pipeline.md`) includes, among other fields:

- `depth`: \((T, H, W)\) float16 depth in **meters**
- `contact_flag`: \((T,)\) bool
- `risk_05s`, `risk_1s`, `risk_2s`: \((T,)\) float32 in \(\{0,1\}\)

### 2.2 From disk to training batch

| Step | Component | Path |
|------|-----------|------|
| Dataset | `RiskDataset` | `create_dataset_module/risk_dataset.py` |
| Collate | `collate_riskbatch` → `RiskBatch` | same file |
| Contracts | `RiskSample`, `RiskBatch` | `PointPillars_module/data_contracts.py` |

**One `RiskSample`** = one index \(t\) with:

- `pts_seq`: `List[torch.Tensor]` of length **`T_ctx`**, each **\((N_i, 4)\)** float32, columns **`[x, y, z, intensity]`** in **LiDAR / KITTI-style** frame (after depth preprocessing — see `module_pointpillar.py` / `docs/module_pointpillar.md`).
- `action_seq`: \((T_\mathrm{ctx}, A)\), `ego_vel_seq`: \((T_\mathrm{ctx}, 6)\)
- `risk_05s`, `risk_1s`, `risk_2s`: scalars in \(\{0,1\}\) for **frame \(t\)**

**`RiskBatch`** stacks \(B\) samples:

- `pts_seq`: **list-of-list** — outer length **`T_ctx`**, inner length **`B`** (this matches `FullPipeline.forward`).
- `risk_targets()`: **\((B, 3)\)** via `torch.stack([risk_05s, risk_1s, risk_2s], dim=-1)`.

### 2.3 Context length and token sequence

| Symbol | Value | Meaning |
|--------|-------|---------|
| \(T_\mathrm{ctx}\) | **10** (default in `EnvConfig.T_ctx` / dataset) | **0.5 s** history at 20 Hz |
| \(N_t\) | **16** | Spatial tokens per frame (`SpatialReducer.num_tokens` = \(4 \times 4\)) |
| \(D\) | **256** | Token dimension (`token_dim` / `d_model`) |
| Mamba sequence length \(L\) | \(T_\mathrm{ctx} \cdot N_t = \mathbf{160}\) | `FullPipeline.forward` flattens time × space |

### 2.4 Per-frame geometry → BEV

For each time step \(t \in \{0, \ldots, T_\mathrm{ctx}-1\}\) and batch element:

1. **`PointPillarsNeckExtractor.extract_neck_forward`** (Stage A gradient path): list of **\(B\)** clouds **\((N_i, 4)\)** → **`NeckFeatureOutput.feature`** of shape **\((B, 384, 248, 216)\)**.
2. **`SpatialReducer`**: **\((B, 384, 248, 216) \rightarrow (B, 16, 256)\)**.

Stack over time → reshape to **\((B, 160, 256)\)** → **`MambaTemporal.forward`**.

---

## 3. Model architecture

**Top-level module:** `FullPipeline` — `PointPillars_module/models/full_pipeline.py`.

```
pts_seq (T_ctx lists of B clouds)
    → pp.extract_neck_forward  →  bev (B, 384, 248, 216) per frame
    → SpatialReducer           →  (B, 16, 256) per frame
    → stack + flatten          →  seq (B, 160, 256)
    → MambaTemporal            →  h_seq (B, 160, 256)
    → slice last time index    →  h_T (B, 256)
    → RiskHead                 →  logits (B, 3)
```

### 3.1 PointPillars neck

- Wrapper: `PointPillarsNeckExtractor` — `PointPillars_module/module_pointpillar.py`.
- Outputs **BEV feature map** **`(B, 384, 248, 216)`** (default `point_cloud_range` / `voxel_size`; see `module_pointpillar.md`).

### 3.2 `SpatialReducer`

- File: `PointPillars_module/models/spatial_reducer.py`.
- **\(4 \times 4\)** adaptive average pool over conv-downsampled BEV → **`N_t = 16`** tokens, each **`D = 256`**.

### 3.3 `MambaTemporal`

- File: `PointPillars_module/models/mamba_temporal.py`.
- **Role:** encode **temporal dynamics** over the **160-token** sequence (10 frames × 16 spatial tokens).
- Backends: **`mamba-ssm`** (preferred on CUDA) or **`nn.GRU`** fallback (`backend="auto"` / `"gru"`).
- **Stage A:** `forward(seq)` with `seq` shape **\((B, L, D)\)** returns **\((B, L, D)\)**; training uses **\(h_T = h_\mathrm{seq}[:, -1, :]\)** → **\((B, 256)\)**.
- **Stage B streaming:** `step(tok, hidden)` — see §6.

### 3.4 `RiskHead`

- File: `PointPillars_module/models/risk_head.py`.
- MLP on **`h_T` (B, 256)**:

  ```
  Linear(256 → 128) + ReLU + Dropout(0.1)
  Linear(128 → 64)  + ReLU + Dropout(0.1)
  Linear(64 → 3)    # raw logits — no sigmoid inside
  ```

- Output **`logits (B, 3)`**; apply **`torch.sigmoid`** only for **inference / logging**, not inside the loss (use **`BCEWithLogits`** via `focal_bce`).

---

## 4. Labeling strategy

**Authoritative narrative:** `docs/strategy_create_trajectory_label.md`.

### 4.1 Source signal

- **`contact_flag[t]`**: from simulation (`DatasetEnv.get_contact_flag()` → PyBullet contact logic), **not** from the neural net.

### 4.2 `lookahead_any`

Implementation: `create_dataset_module/generator.py`.

\[
\mathrm{risk\_*}[t] = \mathbb{1}\Big[ \exists\, k \in [t,\, t+H) : \mathrm{contact\_flag}[k] \Big]
\]

with half-open window semantics and truncation near episode end (no padding with `True`).

### 4.3 Default horizons (20 Hz)

| Label field | \(H\) (frames) | Time | `DataGenConfig` field |
|-------------|----------------|------|------------------------|
| `risk_05s` | **10** | 0.5 s | `horizon_05s_frames` |
| `risk_1s` | **20** | 1.0 s | `horizon_1s_frames` |
| `risk_2s` | **40** | 2.0 s | `horizon_2s_frames` |

### 4.4 Monotonicity (on binary labels)

For the same `contact_flag`, wider horizons are **at least as often** 1:

\[
\mathrm{risk\_2s}[t] \ge \mathrm{risk\_1s}[t] \ge \mathrm{risk\_05s}[t]
\]

(element-wise in \(\{0,1\}\) ordering).

---

## 5. Training protocol

### 5.1 Loss — focal binary cross-entropy

**Implementation:** `PointPillars_module/losses.py` — `focal_bce(logits, targets, gamma=2.0, weight=(1.0, 0.8, 0.5), reduction="mean")`.

Per element (with stable `binary_cross_entropy_with_logits`):

\[
\mathrm{BCE} = -\log p_t, \quad p_t = e^{-\mathrm{BCE}}
\]

\[
\ell_\mathrm{focal} = (1 - p_t)^{\gamma} \cdot \mathrm{BCE}
\]

Then multiply by **per-horizon weights** \(w \in \mathbb{R}^3\): default **\((1.0,\, 0.8,\, 0.5)\)** — i.e. **strongest weight on the 0.5 s horizon**, then 1 s, then 2 s.

Finally **`mean`** over all batch and horizon dimensions.

### 5.2 Class imbalance

- **Problem:** mostly “safe” frames; positives are rare.
- **Dataset generation:** include **`policy_adversarial_p`** in `DataGenConfig` (with `policy_random_p`, `policy_scripted_p`, `policy_stationary_p`; probabilities must sum to **1.0**) — `AdversarialPolicy` in `create_dataset_module/policies.py` biases toward collisions.
- **Sampling helper:** `oversample_positive_indices` in `losses.py` — duplicates indices where **`risk_1s > 0.5`** (default factor **10**); use with `SubsetRandomSampler` as in `create_dataset_module` README.

### 5.3 Sub-stages A1 / A2

| Phase | PointPillars trainable? | Trainable modules | Notes |
|-------|-------------------------|-------------------|--------|
| **A1** | **Frozen** (KITTI weights) | `SpatialReducer`, `MambaTemporal`, `RiskHead` | `pp.freeze_all()`; use `extract_neck_forward` so gradients reach reducer/Mamba/head |
| **A2** | **`neck` only** | `pp.neck` + same as A1 | `set_trainable(["neck"])` or `unfreeze_neck()` after `freeze_all()`; two LR groups (neck **\(3\times 10^{-5}\)**, others **\(10^{-4}\)** per `strategy_full_pipeline.md` §5.4) |

Optimizer recipe (A1 reference): **AdamW**, **`lr = 3\times 10^{-4}`**, **`weight_decay = 10^{-4}`**, cosine with warmup **500** iterations.

### 5.4 Training step (contract)

Conceptually (matches `FullPipeline.forward` + `RiskBatch.risk_targets()`):

```python
logits = full_pipeline(batch.pts_seq)          # (B, 3)
loss = focal_bce(logits, batch.risk_targets(), gamma=2.0, weight=(1.0, 0.8, 0.5))
```

> **Note:** `train_stage_a.py` is the intended dedicated entry point listed in `strategy_full_pipeline.md` §5.2; until it lands, use the same contract in custom loops or notebooks.

### 5.5 Metrics and split

- Log **AUC-ROC**, **AUC-PR**, **Brier** per horizon; reliability bins.
- **Split 80/10/10 by `scene_id`** — `scene_stratified_split` in `risk_dataset.py` (no frame-level leakage).

---

## 6. Integration with Stage B (SAC)

### 6.1 Risk penalty

At environment collection time, the frozen branch produces **\(p_\mathrm{risk} = \sigma(\mathrm{logits}) \in [0,1]^3\)**. The **default** reward shaping term uses the **1 s** horizon:

\[
r_\mathrm{risk} = -\lambda_\mathrm{risk} \cdot p_{1\mathrm{s}}
\]

with **`lambda_risk`** from `EnvConfig` (e.g. **2.0** for PROPOSED, **0.0** for BASELINE A/B). See `strategy_full_pipeline.md` §6.3.

### 6.2 Streaming API

- **`FullPipeline.step(pts_t, hidden)`** — `full_pipeline.py`: one frame, **`@torch.no_grad()`**, returns **`(p_risk, new_hidden)`** with **`p_risk` shape \((B, 3)\)** in **probability** space.
- Maintains **Mamba** state across steps (§6.4.2 in `strategy_full_pipeline.md`).

### 6.3 What is **not** fed to SAC (default)

> **Actor / Critic:** input is **`ProprioState`** flattened to **`s \in \mathbb{R}^{d_s}\`** — **not** \(h_T\), **not** BEV, **not** tokens.
>
> **Replay buffer (S1):** stores **proprio**, actions, **`r_\mathrm{env}`**, **`r_\mathrm{risk}`**, `done`, ids — **not** point clouds or hidden states for the default SAC update.

Stage B-plus (optional, `strategy_full_pipeline.md` §6.8) may add auxiliary losses and partial unfreezing; it **still** does not redefine the default SAC state path unless explicitly changed in docs and code.

---

## 7. File index (quick reference)

| Topic | Path |
|-------|------|
| Data contracts | `PointPillars_module/data_contracts.py` |
| Rollout + labels | `create_dataset_module/generator.py` (`lookahead_any`) |
| Policies | `create_dataset_module/policies.py` |
| Dataset + collate | `create_dataset_module/risk_dataset.py` |
| Full model | `PointPillars_module/models/full_pipeline.py` |
| Losses | `PointPillars_module/losses.py` |
| PointPillars API | `PointPillars_module/module_pointpillar.py` |
| Labeling details | `docs/strategy_create_trajectory_label.md` |
| Global architecture | `docs/strategy_full_pipeline.md` |
| SAC internals | `docs/strategy_finetune_with_SAC.md` |
| Lab / small-scale compare | `docs/strategy_experiment_protocol.md` |
| Optimized train / ablations / caching | `docs/optimized_training_strategy_stage_A.md` |

---

## 8. Changelog

| Date | Version | Summary |
|------|---------|---------|
| 2026-04-18 | v1.0 | Initial **Stage A training bible** — consolidates §3–§5 contracts, `FullPipeline` tensor shapes, focal loss, A1/A2, lookahead labeling, and Stage B **\(r_\mathrm{risk}\)** interface; explicit **no SAC input from \(h_T\)** invariant. |
| 2026-04-18 | v1.0.1 | §7 file index: cross-link `strategy_experiment_protocol.md` for downscaled method-comparison runs. |
| 2026-04-18 | v1.0.2 | §7: cross-link `optimized_training_strategy_stage_A.md` (BEV caching, decoupled training, paper tables). |
