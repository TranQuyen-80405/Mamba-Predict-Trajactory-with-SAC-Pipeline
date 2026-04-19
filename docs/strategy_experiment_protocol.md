# Experiment protocol — downscaled data & training (Stage A + B)

> **Purpose:** Run **method comparisons** (architecture, loss, backend, λ_risk, …) on a **small, cheap** footprint. Numbers here **do not** replace production targets in `strategy_full_pipeline.md` §5–§6; they define a **consistent lab recipe** and **shared metrics** so Stage A and Stage B results are comparable.
>
> **Authoritative architecture** remains `strategy_full_pipeline.md`. Conflict resolution: full-pipeline doc wins; update this file when contracts change.

---

## 1. Design principles

1. **Same contracts** — `Trajectory`, `RiskSample`, `RiskBatch`, horizons (10 / 20 / 40 frames @ 20 Hz), camera defaults `(160, 120)`, FoV 90°, near/far 0.1 / 8.0. Only **scale** (scenes, rollouts, frames, epochs) shrinks.
2. **Scene-stratified splits** — Always split by `scene_id` (`scene_stratified_split`); never by frame. Methods must use the **same** train/val/test scene lists for a fair compare.
3. **Report budget** — Log **disk estimate**, **approximate train steps**, and **wall-clock** per run so Colab / local experiments stay comparable.

---

## 2. Downscaled data generation

### 2.1 Recommended “experiment” preset (CLI)

| Field | Production-ish (doc §5.1) | **Experiment** |
|------|-----------------------------|----------------|
| `n_scenes` | 300 | **24** |
| `rollouts_per_scene` | 50 | **3** |
| `frames_per_rollout` | 400 | **120** (≈ 6 s @ 20 Hz; still ≥ 2 s lookahead window) |
| `save_rgb` | `False` | **`False`** |
| `out_dir` | e.g. `dataset/pybullet_risk_v1` | **`data/stage_a_experiment`** |

**Order-of-magnitude disk (depth float16 only):**

\[
N_\text{frames} = 24 \times 3 \times 120 = 8640
\]
\[
N_\text{frames} \times 160 \times 120 \times 2\ \text{B} \approx 0.33\ \text{GiB raw depth}
\]

Compressed `.npz` + metadata: typically **~0.2–0.5 GiB** total tree — suitable for laptops and Colab Drive.

**CLI / notebook:**

```bash
python run_datagen_preset.py experiment
```

(Preset implementation: `scripts/datagen/run_datagen_preset.py`; wrapper tại root giữ nguyên lệnh trên.)

### 2.2 When you need even smaller

Use existing **`smoke_nb`** (`10 × 2 × 200` frames) for **pipeline debug only** — not enough diversity for serious method ranking.

---

## 3. Downscaled Stage A training

| Knob | Production (doc §5.2–§5.4) | **Experiment** |
|------|----------------------------|----------------|
| A1 epochs | 5 | **2–3** |
| A2 epochs | 5 | **0–1** (optional first sweep: **skip A2**) |
| Batch size | 32 (T4) / 64 (A100) | **8–16** (tune to GPU) |
| Stop criterion | e.g. AUC-ROC `risk_1s` ≥ 0.85 | **Relative** — compare methods on **same val**; early-stop at **plateau** or **max epoch** |
| Data | subset / full | **Only** `stage_a_experiment` (or same small dir for all variants) |

**Fair compare rule:** Fix **seed**, **data path**, **split**, **batch**; vary **only** the method (e.g. `MambaTemporal(backend="gru")` vs `"mamba"`, head width, loss weights).

---

## 4. Unified evaluation criteria (Stage A **and** Stage B)

Use **one scoring table**; Stage A fills **offline** rows; Stage B adds **online** rows. Always report **val or eval scene IDs** explicitly.

### 4.1 Stage A — offline (supervised risk)

| Metric | Definition | Primary use |
|--------|------------|-------------|
| **AUC-ROC (`risk_1s`)** | Discrimination on **1 s** horizon | **Main headline** for risk quality |
| **AUC-PR (`risk_1s`)** | Important under **class imbalance** | Compare methods when positives are rare |
| **AUC-ROC / AUC-PR (`risk_05s`, `risk_2s`)** | Same for other horizons | Sanity / calibration across horizons |
| **Brier score** | Mean squared error vs 0/1 labels | **Probabilistic** quality (lower = better) |
| **ECE** (optional) | Expected calibration error (binned) | If comparing **sigmoid** outputs as probabilities |
| **Train / val loss** | `focal_bce` curve | Debugging; not sufficient alone |

**Minimum report for a method A vs B:** **AUC-PR & AUC-ROC on `risk_1s` (val)** + **Brier** on val, same split.

### 4.2 Stage B — online (SAC + frozen risk branch)

| Metric | Definition | Primary use |
|--------|------------|-------------|
| **Episode return** | \(\sum_t r_{\text{total},t}\) | Overall policy quality |
| **\(r_\text{env}\) vs \(r_\text{risk}\)** | Log means or cumulative | Is shaping helping or dominating? (`strategy_full_pipeline.md` §6.3 band) |
| **Collision rate** | Fraction of episodes with contact | Safety |
| **Success / goal** | Task-specific (if defined in env) | Task completion |
| **Risk branch telemetry** | Mean / std of \(p_{1\mathrm{s}}\) on eval | Detect **stuck** 0/1 or drift (link to Stage A Brier) |
| **Sample efficiency** | Return vs **env steps** to threshold | Compare **BASELINE** vs **PROPOSED** \(\lambda_\text{risk}\) |

**A/B framing (unchanged):** match **seeds**, **env config**, **proprio layout**; only vary **`lambda_risk`** (e.g. 0 vs 2) or policy architecture **outside** the frozen risk branch.

### 4.3 Cross-stage link

- **Good Stage A** (val AUC-PR / Brier) should correlate with **useful** \(p_{1\mathrm{s}}\) shaping in Stage B — but **not guaranteed** (sim gap). Stage B metrics are the **final arbiter** for control; Stage A metrics arbitrate **perception / risk** methods.

---

## 5. Checklist before claiming “method X wins”

- [ ] Same **`scene_stratified_split`** seeds and scene lists.
- [ ] Same **`DataGenConfig`** (except irrelevant knobs) for data used by all variants.
- [ ] Same **eval scenes** for Stage B across compared runs.
- [ ] Log **hardware**, **approximate train time**, and **CU / cost** if on Colab.

---

## 6. Changelog

| Date | Version | Summary |
|------|---------|---------|
| 2026-04-18 | v1.0 | Initial experiment protocol: downscaled preset `experiment`, shared Stage A + B metrics, fair-compare rules. |
