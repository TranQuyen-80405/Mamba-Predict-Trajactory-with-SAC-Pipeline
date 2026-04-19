# Skill — Gradient & Numerical Stability (Stage A Perception + Stage B SAC)

> **Role:** engineering playbook to keep optimization stable when Stage A (supervised risk / Mamba perception stack) and Stage B (SAC on proprio) are trained or run sequentially.  
> **Scope:** gradient management, initialization, normalization, RL-specific stabilizers, and pre-flight checklists.  
> **Authority:** architecture and invariants remain `docs/strategy_full_pipeline.md`; SAC details `docs/strategy_finetune_with_SAC.md`. This note **does not** override the rule that **SAC must not consume BEV / tokens / hidden states** — only `ProprioState` enters the Actor/Critic.

---

## 1. Stage A vs Stage B — stability surface (comparison)

| Aspect | Stage A (perception + risk) | Stage B (SAC control) |
|--------|-----------------------------|------------------------|
| **Objective** | Supervised multi-horizon risk (focal BCE on logits) | Max entropy RL: critic TD + actor + entropy temperature |
| **Main failure modes** | Class imbalance → tiny gradients on negatives; focal exploding when \(p_t \to 0\); BN drift if PP put in `.train()` by mistake | Critic over-estimation; huge \(Q\) targets if reward unscaled; actor saturation; \(\sigma \to 0\) or \(\infty\) if `log_std` unclamped |
| **Typical grad clip** | One group (reducer + temporal + head): `clip_grad_norm_(params, 1.0)` | **Separate** clips after **each** backward: actor `1.0`, critic `1.0` (see §2.1) |
| **AMP (FP16)** | Optional; PP voxel path often FP32-heavy | Optional; MLP-only updates are cheap; use GradScaler if enabled |
| **Normalization** | LayerNorm on Mamba blocks; **BatchNorm2d** in `SpatialReducer` (keep PP frozen + BN-lock in Stage B) | **No BN** in Actor/Critic by default; optional **LayerNorm** on trunk; **running stats** on proprio (Welford / RMS) — **not** BEV features (SAC does not see them) |
| **Coupling risk** | Poor Stage A calibration → wrong `r_risk` magnitudes → Stage B value function stress | N/A — fix calibration / \(\lambda_{risk}\) / reward scale before blaming SAC |

---

## 2. Gradient management (Stage A & B)

### 2.1 Global \(L_2\) norm clipping

For parameters \(\theta\), after `loss.backward()`:

\[
g \leftarrow \begin{cases}
g & \text{if } \|g\|_2 \leq c \\
g \cdot \dfrac{c}{\|g\|_2} & \text{otherwise}
\end{cases}
\]

with threshold \(c = 1.0\) (project default).

**Why separate Actor vs Critic clips in Stage B?**  
Actor and critic losses are **different magnitudes** and **different sources** of bad gradients (policy gradient noise vs bootstrapped TD). A single backward through a combined graph is **not** used in clean SAC implementations: you compute critic loss → `backward` → `clip_grad_norm_(critic_params)` → `step`; then actor loss → `backward` → `clip_grad_norm_(actor_params)` → `step` (order may vary, but **gradients must not be mixed** before clipping). If one network spikes, it does not **overwrite or rescale** the other’s gradient buffer.

**Stage A** often uses **one** optimizer over `SpatialReducer + temporal + RiskHead` with a **single** `clip_grad_norm_` on the union — acceptable because there is one scalar loss and no alternating targets.

### 2.2 Mixed precision — `torch.cuda.amp.GradScaler`

Pattern:

1. `scaler.scale(loss).backward()`
2. `scaler.unscale_(optimizer)` then `clip_grad_norm_` (clip operates in FP32 unscaled grads)
3. `scaler.step(optimizer)`; `scaler.update()`

**Rules:**

- Do **not** clip before `unscale_` on scaled gradients — norms would be wrong.
- If `scaler.step` skips due to inf/NaN, `update()` still shrinks scale — monitor skip rate.
- Stage A: autocast around **only** the parts that are FP16-safe; PointPillars / voxel ops may need **full FP32** — use selective autocast blocks.

Reference implementation hooks: `PointPillars_module/utils/gradient_health.py`, `train_stage_b_sac.py` (commented pattern).

---

## 3. Weight initialization & architecture

### 3.1 Stage A — `SpatialReducer`

- **Conv stacks:** **Kaiming (He)** init for `nn.Conv2d` with `a=sqrt(5)` / `mode=fan_in` / `nonlinearity=relu` (PyTorch default for `nn.init.kaiming_uniform_` on conv is appropriate for ReLU stages).
- **BatchNorm:** default `weight=1`, `bias=0`; ensure running stats are **not** updated when the module is frozen and in `.eval()` (Stage B risk branch).

### 3.2 Stage A — Mamba SSM (\(A\), \(\Delta\), \(B\), \(C\), …)

The **mamba-ssm** block implements structured state-space parameters. Stability practices (conceptual):

- **Discretization** uses \(\Delta\) and step-size parameterizations; poorly scaled \(\Delta\) can cause exploding hidden states.
- Prefer **library defaults** from a maintained `mamba-ssm` release; if you fork or re-init:
  - Keep \(\Delta\) in a **bounded** parameterization (e.g. softplus) as in reference implementations.
  - Initialize slow modes so that **eigenvalues of the discretized \(A\)** lie inside the stability region (consult the Mamba / S4 literature for your version).
- **Do not** stack many Mamba blocks without **normalization + residual** — our `MambaTemporal` uses LayerNorm + residual per block.

*Note:* Project code does not reinitialize internal Mamba tensors; treat this section as **design guidance** when upgrading `mamba-ssm` or debugging NaNs in the temporal trunk.

### 3.3 Stage B — SAC Actor final layer (small output scale)

To avoid **tanh saturation** and huge actions at step 0:

- Initialize the **last `mu` linear** with small weights (e.g. **uniform** or **normal** with std \(\approx 3 \times 10^{-3}\)).
- Optionally use the same small scale for **`logstd` head** bias toward a reasonable initial entropy (e.g. bias init so initial std \(\approx\) small constant).

See `train_stage_b_sac.py` — `ActorMLP` applies `final_init_std=3e-3` to `mu` and `logstd` output layers.

---

## 4. Normalization strategies

| Location | Mechanism | Notes |
|----------|-----------|--------|
| Stage A temporal | **LayerNorm** (Mamba blocks) | Stabilizes residual stream across sequence length \(L=160\). |
| Stage A BEV trunk | **BatchNorm2d** (`SpatialReducer`) | Must stay `.eval()` when frozen; BN-lock hook in Stage B (see `strategy_finetune_with_SAC.md` §4.2). |
| Stage B Actor/Critic | **No BN** (default) | Optional **LayerNorm** inside trunk if activations drift. |
| Stage B observations | **Running normalization** of **proprio vector** | Maintain EMA of mean/var per dim; normalize to \(\approx \mathcal{N}(0,1)\) then **clip** to \([-10,10]\) or squash heavy tails — **do not** claim \([-1,1]\) unless you explicitly tanh-scale each dim. |

**Important invariant:** The doc phrase “observation normalization” for SAC refers to **`ProprioState`** (velocities, goal-relative pose, last action, …). **Not** BEV or risk hidden states — those must never enter the Actor/Critic in the default pipeline.

---

## 5. Reinforcement learning — Stage B stability

### 5.1 Soft target update (Polyak)

Target networks \(\theta'\) track online \(\theta\):

\[
\theta' \leftarrow (1 - \tau)\,\theta' + \tau\,\theta
\]

with **\(\tau = 0.005\)** (small = smooth targets = lower gradient variance, slower tracking).

### 5.2 Reward scaling

**Symptom:** critic loss and TD targets explode; `Q` grows without bound.  
**Causes:** large collision penalties, dense progress terms stacked, or `r_risk` too negative if \(\lambda_{risk}\) is high.

**Mitigations:**

- **Tune coefficients** (`w_collision`, `w_goal`, …) per `strategy_full_pipeline.md` §6.6.
- **Running StandardScaler** on **total scalar reward** (or separate scales for `r_env` vs `r_risk` if you log them separately): store EMA mean/std of observed returns / rewards and divide the bootstrapped target by a constant scale so TD error \(\delta\) is \(\mathcal{O}(1)\). **Revert scale** when interpreting policy performance.
- Monitor **`E[r_risk]/E[r_env]`** ratio (doc target \(\in [-0.4, 0]\)).

### 5.3 Entropy temperature \(\alpha\) (automatic)

SAC maximizes:

\[
J(\pi) = \mathbb{E}\Big[ Q(s,a) - \alpha \log \pi(a|s) \Big]
\]

with learnable **`log_alpha`** (or fixed \(\alpha\)). **Automatic entropy tuning** adjusts \(\alpha\) so entropy tracks a target \(\mathcal{H}_\text{targ}\).

**Stability:**

- If \(\alpha \to 0\) too fast → policy collapses (no exploration); gradients through entropy term vanish.
- If \(\alpha\) explodes → policy stays too stochastic; actor loss dominated by entropy.
- **Clamp `log_alpha`** to a reasonable range (e.g. `[-20, 2]`) in log-space as a **last resort**; prefer learning-rate and target-entropy tuning first.

---

## 6. Key formulas (LaTeX)

### 6.1 Focal BCE (Stage A)

With logits \(z\), target \(y \in \{0,1\}\), \(p = \sigma(z)\):

\[
\text{BCE}(z,y) = - y \log p - (1-y)\log(1-p)
\]

Focal modulation (\(\gamma \geq 0\)):

\[
\text{FL} = (1 - p_t)^\gamma \, \text{BCE}, \quad p_t = \begin{cases} p & y=1 \\ 1-p & y=0 \end{cases}
\]

**Numerical note:** clamp \(p_t\) away from \(\{0,1\}\) by a small \(\varepsilon\) when raising to \(\gamma\) (implementation: `losses.focal_bce`).

### 6.2 SAC critic TD target (sketch)

With twin critics, clipped double-\(Q\):

\[
y = r + \gamma (1-d)\, \Big( \min_{i=1,2} Q_{\theta'_i}(s', \tilde{a}') - \alpha \log \pi(\tilde{a}'|s') \Big)
\]

\(\tilde{a}'\) sampled from current policy. Critic loss:

\[
L_Q = \mathbb{E}\Big[ (Q_{\theta_j}(s,a) - y)^2 \Big], \quad j \in \{1,2\}
\]

### 6.3 Actor log-probability (Gaussian + tanh) — stability

Use **`log_std` clamped** to \([\texttt{LOGSTD\_MIN}, \texttt{LOGSTD\_MAX}] = [-5, 2]\).  
When forming \(\sigma = \exp(\log\_std)\), use **`std = sigma.clamp_min(eps)`** with \(\varepsilon = 10^{-8}\) before division in the Gaussian density.

---

## 7. Code map (this repo)

| Component | File(s) |
|-----------|---------|
| Focal loss + \(\varepsilon\)-safe \(p_t\) | `PointPillars_module/losses.py` |
| He init for reducer convs | `PointPillars_module/models/spatial_reducer.py` |
| Grad norms / logging helpers | `PointPillars_module/utils/gradient_health.py` |
| Stage A compare train + epoch grad stats | `PointPillars_module/training/train_stage_a_compare.py` |
| SAC Actor/Critic reference + clamps + separate clips | `PointPillars_module/train_stage_b_sac.py` |

*Note:* Full `train_stage_a.py` may be added later; Stage A comparison training lives in `training/train_stage_a_compare.py` (also importable as `train_stage_a_compare` when `PointPillars_module` is on `sys.path`).

---

## 8. Checklists — before pressing Train

### Stage A (risk / perception)

- [ ] `PointPillars` frozen (`freeze_all()`); **not** toggled back to `.train()` by a blanket `model.train()` on the full pipeline.
- [ ] `clip_grad_norm_` on **trainable** heads only (reducer / temporal / risk head), with max norm `1.0`.
- [ ] If AMP: `GradScaler` + `unscale_` before clipping; PP forward in FP32 if unstable.
- [ ] Verify **class balance** (oversampling / focal \(\gamma\)) — avoid epochs with zero positive samples in a batch.
- [ ] Log **max gradient norm** per epoch (TensorBoard / stdout); investigate spikes > \(10^3\).

### Stage B (SAC)

- [ ] **Separate** `clip_grad_norm_` for **critic** and **actor** (and optionally `log_alpha` group).
- [ ] **`log_std.clamp(-5, 2)`** on Actor forward; **`std.clamp_min(1e-8)`** before division in log-prob.
- [ ] **Polyak** \(\tau=0.005\) on **both** target critics only (no target actor in standard SAC).
- [ ] **Reward scale** sane: monitor raw \(r\), scaled \(r\), and \(Q\) means; enable running reward normalization if needed.
- [ ] **`log_alpha`** optimizer separate; watch \(\alpha\) and policy entropy.
- [ ] **Proprio normalization** EMA initialized; no NaNs after env reset.
- [ ] Confirm **no tensor** from PointPillars / BEV / Mamba enters Actor or Critic `forward`.

---

## 9. Changelog

| Date | Change |
|------|--------|
| 2026-04-18 | Initial version: gradient clip strategy, AMP, init, normalization, SAC \(\tau\), reward / entropy stability, checklists; aligned with `strategy_full_pipeline.md` & `strategy_finetune_with_SAC.md`. |

---

*End. For pipeline invariants, always reconcile with `strategy_full_pipeline.md` §1 and §6.*
