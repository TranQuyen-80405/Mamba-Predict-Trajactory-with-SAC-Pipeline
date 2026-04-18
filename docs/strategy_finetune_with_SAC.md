# Strategy — SAC Detail Reference (under the Mamba-aware pipeline, v3)

> **Audience:** the next agent / engineer who will implement Stage B of the pipeline described in `strategy_full_pipeline.md`.
> **Scope of this document:** SAC-specific implementation details — Actor / Critic architecture on the proprioceptive state, replay buffer layout, Polyak averaging, BatchNorm lock hook for the frozen perception stream, hyperparameters, optimizer grouping.
>
> **This doc is NOT authoritative for the overall pipeline architecture.** The pipeline is two decoupled streams:
>
> ```
> Perception stream (frozen in default Stage B):
>     depth → PointPillars → SpatialReducer → Mamba → RiskHead → p_risk
>                                                                   │
>                                                                   └─► r_risk (reward shaping)
>
> Control stream (SAC, trainable):
>     ProprioState (base lin/ang vel + goal + heading + ...) → Actor / Critic → a_t
> ```
>
> The two streams only meet at the reward: `r_total = r_env + r_risk`. **SAC does NOT consume any perception feature.** This is a deliberate design choice so the experiment is a clean A/B test on whether the pretrained risk predictor improves RL in dynamic-obstacle scenes.
>
> See `strategy_full_pipeline.md` for full architecture, data contracts, dataset generation, and Stage A training. This doc **only** elaborates on the SAC-specific details of Stage B.
>
> **Before editing this doc, read § 21 "Maintenance rule — keep these docs in sync".** Any change to SAC internals (Actor/Critic layers, optimizer groups, BN-lock hook, hyperparameters, freeze regimes S1/S2/S3) MUST be mirrored — as a summary or cross-reference — in `strategy_full_pipeline.md` and, where relevant, in `module_pointpillar.md` in the same PR.

---

## 0. Reading map (what this doc owns vs. defers)

| Topic                                                    | Owner                        |
|----------------------------------------------------------|------------------------------|
| Overall two-stream pipeline architecture                 | `strategy_full_pipeline.md` §1 |
| Stage A (supervised pretrain on PyBullet)                | `strategy_full_pipeline.md` §5 |
| Stage B weight transfer map, frozen/unfrozen policy      | `strategy_full_pipeline.md` §6.1 |
| Replay buffer layout (what fields per transition)        | `strategy_full_pipeline.md` §6.4 |
| Environment spec (camera, dt, reward coefficients)       | `strategy_full_pipeline.md` §5.1 & §6.6 |
| Reward shaping formula (`r_total = r_env + r_risk`)      | `strategy_full_pipeline.md` §6.3 |
| Mamba streaming across env steps                         | `strategy_full_pipeline.md` §6.4.2 |
| A/B test configurations (BASELINE vs PROPOSED)           | `strategy_full_pipeline.md` §6.3.1 |
| **Actor / Critic architecture (MLP on proprio)**         | **This doc, §5 / §7**        |
| **BatchNorm lock hook for frozen perception**            | **This doc, §4.2**           |
| **SAC choice rationale + algorithm alternatives**        | **This doc, §5**             |
| **SAC-internal hyperparameters (γ, τ, buffer warmup …)** | **This doc, §12**            |
| **Optional Stage B-plus regimes (S2 / S3)**              | **This doc, §4**             |
| **Maintenance rule (trigger matrix, per-PR checklist)**  | Canonical long-form: `strategy_full_pipeline.md` §14. SAC-specific shortcut: **this doc, §21**. |

Whenever the two docs disagree, the **full pipeline doc wins** for architecture and data contracts; **this doc wins** for SAC-internal algorithmic details.

---

## 1. Intent (under the new pipeline)

1. At Stage B runtime, each env step produces a **depth buffer** from PyBullet plus a **proprioceptive state** from the robot.
2. The depth buffer is processed by the **frozen perception stream** to produce a scalar `p_risk_1s ∈ [0, 1]` which becomes `r_risk = -λ_risk · p_risk_1s`.
3. The proprioceptive state `s` is handed directly to the **Actor / Critic** — both are plain MLPs. SAC trains them with `r_total = r_env + r_risk`.
4. No labels are used. No encoder is trained. The perception stream is a closed-form dense reward function; SAC's only job is learning policy / value on a low-dim state vector.

### 1.1 Why this matters

- Keeping SAC state low-dim and identical between BASELINE and PROPOSED configurations removes confounders. The only knob is whether `r_risk` is added to the reward.
- Freezing the perception stream eliminates catastrophic forgetting by construction. Nothing is trained there, so nothing is forgotten.
- The replay buffer shrinks to ~500 B per transition (proprio + actions + two rewards). 1 M transitions fit in 500 MB RAM.

---

## 2. High-level data flow (Stage B runtime, default plan)

```
At each env step t (20 Hz):

  ── Perception stream (frozen, no_grad, runs every step) ─────────────
  depth_buf_t  (160, 120) in [0, 1]
      │  pybullet_depth_to_meters(near=0.1, far=8.0)
      ▼
  depth_m_t  (160, 120) float32, meters
      │  preprocess_depth_frame(intr, extr, cfg)        [non-differentiable]
      ▼
  pts_t      (N, 4)
      │  extract_neck([pts_t])                           [frozen, no_grad]
      ▼
  bev_t      (1, 384, 248, 216)
      │  spatial_reducer(bev_t)                          [frozen, no_grad]
      ▼
  tok_t      (1, 16, 256)
      │  mamba.step(tok_t, hidden=h_{t-1})              [frozen, streaming]
      ▼
  h_t        (1, 256)
      │  risk_head(h_t)
      ▼
  p_risk_t   (1, 3)     → r_risk_t = -λ_risk · p_risk_t[1s]

  ── Control stream (SAC, MLP on proprio) ─────────────────────────────
  s_t = env.read_proprio()     # (d_s,) ≈ 10–50
      │
      ▼
  actor.sample(s_t)            → a_t, log_pi_t
      │
      ▼
  env.step(a_t)                → s_{t+1}, r_env_t, done_t

  ── Store and train ──────────────────────────────────────────────────
  buffer.push(s_t, s_{t+1}, a_t, r_env_t, r_risk_t, done_t)

  if len(buffer) ≥ warmup:
      batch = buffer.sample(B)           # plain numpy / cpu tensors
      run SAC update (actor + 2 critics + α) on (s, s_next, a, r_env + r_risk, done)
```

Key facts:

- `preprocess_depth_frame` is **entirely non-differentiable** and fixed.
- `PillarLayer` is non-differentiable too, but that does not matter here — **nothing in the perception stream receives gradient in the default plan**.
- The only parameters receiving gradient are `actor`, `critic_q1`, `critic_q2`, `log_alpha`. All are plain MLPs on `s` / `(s, a)`.
- The Mamba hidden state is streaming — maintained across env steps on the GPU and reset on episode boundaries. Use `MambaStreamer` in `strategy_full_pipeline.md` §6.4.2.
- Gradient steps involve ONLY `s`, `a`, and scalars — no PointPillars forward, no BEV compute. Each gradient step is ~5 ms on a 3060.

### 2.1 Differences from the old SAC-only plan

Earlier drafts of this doc treated PointPillars as a trainable state encoder and included a `BEVFeatureExtractor` / `BEVStateExtractor`, a critic-only encoder update rule, and a buffer storing raw `pts`. Under the new pipeline:

- PointPillars is **not** on the SAC gradient path. No encoder is.
- There is no `BEVStateExtractor` / `BEVFeatureExtractor` anywhere. SAC consumes proprioceptive state directly from the env.
- The "critic-only encoder update" rule (DrQ-v2 convention) is therefore **not needed** in the default plan — it is only relevant if you opt into a future experiment where SAC actually consumes a BEV feature. Such an experiment is explicitly NOT this pipeline.
- The replay buffer stores proprio state vectors, not `pts`.

---

## 3. Feasibility — where can the RL gradient flow?

| Block                                                          | Parameters? | Differentiable? | Can RL gradient reach it (default plan)? | Can RL gradient reach it (Stage B-plus)?   |
|----------------------------------------------------------------|-------------|-----------------|------------------------------------------|--------------------------------------------|
| `preprocess_depth_frame` (NumPy)                               | No          | No              | —                                        | —                                          |
| `PillarLayer` (hard voxelize + scatter)                        | No          | No              | —                                        | —                                          |
| `PillarEncoder` / `Backbone` / `Neck` of PointPillars          | Yes         | Yes             | ❌ frozen                                | ✅ only `neck` (S2) or all (S3), with aux BCE |
| `SpatialReducer`, `Mamba`, `RiskHead`                          | Yes         | Yes             | ❌ frozen                                | ✅ in S2 / S3, with aux BCE                |
| `actor`, `critic_q1`, `critic_q2`, `log_alpha`                 | Yes         | Yes             | ✅ always                                | ✅ always                                   |

Conclusion — default plan: SAC gradient only flows into `actor + critics + log_alpha`, each a tiny MLP / scalar. That is intentional. Stage B-plus reopens gradient paths into the perception stream via an **auxiliary focal-BCE loss on labeled transitions** (not via SAC gradient). See §9.

---

## 4. Freeze regimes (only relevant if you opt into Stage B-plus)

> The **default Stage B plan** (see `strategy_full_pipeline.md` §6.1) freezes the entire perception stream. The only trainable modules are actor, critics, and `log_alpha`. In that plan, this section is a no-op — read it only if you promote to Stage B-plus (§6.8 of the full-pipeline doc).

The three regimes below control **which parts of the perception stream** unfreeze in Stage B-plus. The SAC control stream is unchanged across all regimes: always plain MLPs on proprio.

| Strategy | `pp.pillar_encoder` | `pp.backbone` | `pp.neck` | `spatial_reducer` / `mamba` / `risk_head` | `actor`, `critic` |
|---|---|---|---|---|---|
| **S1 — default Stage B (baseline A/B test)** | ❄ frozen | ❄ frozen | ❄ frozen | ❄ frozen | 🔥 trainable |
| **S2 — unfreeze risk branch + neck** (entry-level Stage B-plus) | ❄ frozen | ❄ frozen | 🔥 trainable (1e-5) | 🔥 trainable (5e-5) | 🔥 trainable |
| **S3 — full unfrozen** (expert Stage B-plus) | 🔥 trainable (3e-6) | 🔥 trainable (3e-6) | 🔥 trainable (1e-5) | 🔥 trainable (5e-5) | 🔥 trainable |

### 4.1 When to promote

- **S1 → S2:** only after the A/B test has been measured AND you've observed that the frozen risk predictor is miscalibrated on the SAC rollout distribution (`p_risk` histogram stuck ≈ 0 or ≈ 1, or Brier score drifts > 0.2 versus Stage A). Add aux BCE loss (see §9.2) before promoting.
- **S2 → S3:** same criterion; only if S2 still plateaus AND you have compute budget.
- **Demotion rule:** if promoting causes return to drop > 15 % over 50 k steps, revert and keep the aux BCE loss ON at the lower regime.

### 4.2 BatchNorm trap (critical in ALL regimes, including S1)

`PillarEncoder` and `Backbone` contain BatchNorm. Calling `.train()` on any parent module flips BN back to training mode, which immediately starts updating running stats on RL samples — a near-guaranteed way to destroy a frozen feature extractor.

Canonical hook (works for both default and Stage B-plus):

```python
def freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False
    m.eval()

def apply_freeze_policy(pp, risk_branch, strategy: str) -> None:
    """
    pp           : the stripped PointPillars (from module_pointpillar.py)
    risk_branch  : nn.Module wrapping spatial_reducer + mamba + risk_head
    strategy     : one of "S1", "S2", "S3"
    """
    if strategy == "S1":
        freeze_module(pp.pillar_layer)
        freeze_module(pp.pillar_encoder)
        freeze_module(pp.backbone)
        freeze_module(pp.neck)
        freeze_module(risk_branch)
    elif strategy == "S2":
        freeze_module(pp.pillar_layer)
        freeze_module(pp.pillar_encoder)
        freeze_module(pp.backbone)
        # pp.neck trainable; risk_branch trainable
    elif strategy == "S3":
        # everything trainable; still call .eval() on BN-heavy modules if you
        # want frozen running stats (recommended for S3 too).
        pass
    else:
        raise ValueError(strategy)

# This hook MUST run after every .train() call — both at init and whenever an
# external caller (e.g. a trainer framework) flips the top-level model back to
# train mode.
def install_bn_lock(top_module, frozen_submodules):
    orig_train = top_module.train
    def train(mode: bool = True):
        orig_train(mode)
        for m in frozen_submodules:
            m.eval()
        return top_module
    top_module.train = train
```

For **trainable BN** in Stage B-plus (e.g. `neck` in S2/S3, or any BN inside `spatial_reducer` / `mamba`), set `momentum=0.01` so running stats move slowly under the non-stationary replay distribution.

---

## 5. Why SAC (Soft Actor-Critic) for this problem

Pros for this setting:

- **Off-policy** → replay buffer → same env sample can be re-used many times. Sample efficiency matters because PyBullet is CPU-bound.
- **Continuous actions** — fits robot velocity / joint-target control.
- **Entropy-regularized** (auto-tuned α) — better exploration, less sensitive to reward shaping than DDPG / TD3. Specifically useful for the A/B test, where `r_risk` adds a shaping term that could otherwise over-regularize a deterministic policy.
- **Twin Q (Q1, Q2) with clipped double-Q target** — more stable than a single critic, cheap when critics are small MLPs.

Cons / things to watch:

- Because SAC uses `gradient_steps ≥ 1` per env step, per-gradient-step cost matters. In this plan a gradient step is trivial (~5 ms; no CNN, no encoder). So this worry is mostly moot here.
- With only proprioceptive state, the agent sees NO raw perception. Any benefit of perception must come through `r_risk`. That's exactly what the A/B test is measuring.

Alternatives considered:

| Algorithm | Why not chosen |
|---|---|
| PPO | On-policy, throws away rollouts each update. Unnecessary for this task. |
| DDPG / TD3 | No entropy term, brittle on sparse reward. |
| Dreamer-v3 | World-model RL; overkill here because we're testing a reward-shaping hypothesis, not a representation hypothesis. |

**SAC is the right tool for a clean A/B test of reward shaping.**

---

## 6. SAC wiring — now tiny

### 6.1 Who sees what

```
state s (proprio + goal)  ─► actor(s)       ─► action a
state s + action a        ─► critic_q1(s,a), critic_q2(s,a) ─► Q
target copies:  target_q1, target_q2     ─► Polyak of online critics (τ = 0.005)
```

There is **no encoder** in this loop. No detach dance. No re-encoding on sample. No critic-only update rule.

### 6.2 Update step (default plan)

```python
batch = buffer.sample(B)                   # tiny tensors, all on CPU by default
s, s_next = batch.s, batch.s_next
r_total = batch.r_env + batch.r_risk       # r_risk is 0 in BASELINE

# --- Critic update ---
q1, q2 = critic_q1(s, batch.action), critic_q2(s, batch.action)
with torch.no_grad():
    a_next, logp_next = actor.sample(s_next)
    q_target = torch.min(target_q1(s_next, a_next),
                         target_q2(s_next, a_next))
    y = r_total + gamma * (1 - batch.done) * (q_target - alpha * logp_next)
critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
critic_loss.backward()
optim_critic.step(); optim_critic.zero_grad()

# --- Actor update ---
a_new, log_pi = actor.sample(s)
q_new = torch.min(critic_q1(s, a_new), critic_q2(s, a_new))
actor_loss = (alpha * log_pi - q_new).mean()
actor_loss.backward()
optim_actor.step(); optim_actor.zero_grad()

# --- Alpha update ---
alpha_loss = -(log_alpha * (log_pi.detach() + target_entropy)).mean()
alpha_loss.backward()
optim_alpha.step(); optim_alpha.zero_grad()

# --- Polyak on target critics ---
polyak_update(target_q1, critic_q1, tau=0.005)
polyak_update(target_q2, critic_q2, tau=0.005)
```

### 6.3 Replay buffer content — see full_pipeline §6.4

The authoritative buffer layout is in `strategy_full_pipeline.md` §6.4. Summary:

- **Default plan (S1):** store `(s, s_next, action, r_env, r_risk, done, episode_id, frame_idx)`. Proprio state only; no `pts`, no BEV, no `z`. Per-transition size ≈ 500 B for `d_s = 30, A = 3`. Buffer of 1 M transitions ≈ 500 MB on CPU RAM.
- **Stage B-plus (S2 / S3):** augment with `pts_window: List[Tensor]` (T_ctx frames) + `risk_gt: (3,)` for a 20 % subset of transitions, to feed the aux BCE loss that anchors the unfrozen perception stream. Adds ~1.5 GB.

The old "cached-`z`" and "raw `pts`" designs from earlier revisions of this doc have been removed — they were motivated by a pipeline where SAC consumes BEV features. That pipeline is not what we are building.

### 6.4 Perception forward — kept out of the gradient loop

`extract_neck` + `spatial_reducer` + `mamba.step` + `risk_head` run **only at rollout time**, once per env step, under `torch.no_grad()`. Gradient steps do NOT re-run perception. This is what makes gradient steps cheap (~5 ms).

### 6.5 Memory budget estimate (12 GB GPU, default plan)

| Item | Size |
|---|---|
| Frozen PointPillars weights              | ≈ 20 MB (float32) |
| Frozen risk branch (Reducer + Mamba + Head) | ≈ 15 MB (float32) |
| Trainable networks (actor + 2 critics + 2 targets) | ≈ 2 MB |
| Activations, rollout forward, B = 1      | ≈ 200 MB |
| Activations + grads for MLP update (B = 256) | ≈ 20 MB |
| Streaming Mamba hidden state per env (D = 256) | negligible |
| Replay buffer in RAM                     | ≈ 500 MB (default) / 2.0 GB (Stage B-plus) |

**Default plan fits in < 2 GB VRAM.** Use the spare budget to parallelize environments (SB3 `SubprocVecEnv` with 4–8 workers) — that's the real lever for wall-clock time here, since PyBullet is the bottleneck.

---

## 7. Network architectures

### 7.1 Perception — already in `module_pointpillar.py` + `rl/risk_branch.py`

```
pts (N, 4)
    → PillarLayer                → (P, 32, 4) voxels
    → PillarEncoder              → (B, 64, Ny, Nx)
    → Backbone (3 stages)        → list of 3 tensors
    → Neck (3 upsample+concat)   → (B, 384, 248, 216)        [frozen in S1]

bev (1, 384, 248, 216)
    → SpatialReducer             → (1, 16, 256)              [frozen in S1]
    → mamba.step (streaming)     → h_t (1, 256)              [frozen in S1]
    → RiskHead                   → p_risk (1, 3)             [frozen in S1]
```

### 7.2 Actor (squashed Gaussian, plain MLP on proprio)

```python
class Actor(nn.Module):
    def __init__(self, state_dim, act_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),    nn.SiLU(),
        )
        self.mu     = nn.Linear(hidden, act_dim)
        self.logstd = nn.Linear(hidden, act_dim)
        self.LOGSTD_MIN, self.LOGSTD_MAX = -5.0, 2.0

    def forward(self, s):
        h = self.trunk(s)
        mu = self.mu(h)
        logstd = self.logstd(h).clamp(self.LOGSTD_MIN, self.LOGSTD_MAX)
        return mu, logstd

    def sample(self, s):
        mu, logstd = self(s)
        std = logstd.exp()
        noise = torch.randn_like(mu)
        u = mu + std * noise
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - logstd - 0.9189385).sum(-1)
        logp -= (2 * (0.6931472 - u - F.softplus(-2 * u))).sum(-1)
        return a, logp
```

### 7.3 Critic (twin Q, plain MLP)

```python
class Critic(nn.Module):
    def __init__(self, state_dim, act_dim, hidden=256):
        super().__init__()
        def branch():
            return nn.Sequential(
                nn.Linear(state_dim + act_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden),              nn.SiLU(),
                nn.Linear(hidden, 1),
            )
        self.q1 = branch()
        self.q2 = branch()
    def forward(self, s, a):
        sa = torch.cat([s, a], -1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)
```

No BatchNorm in Actor / Critic. Use `LayerNorm` inside the trunk if you see instability.

---

## 8. Optimizer & learning-rate plan

### 8.1 Default plan (S1) — 3 param groups + alpha

| Group | Members | LR | Weight decay |
|---|---|---|---|
| **Actor** | `actor` | `3e-4` | `0.0` |
| **Critic** | `critic_q1`, `critic_q2` | `3e-4` | `0.0` |
| **α (entropy temp)** | `log_alpha` scalar | `3e-4` | `0.0` |

```python
optim_actor  = torch.optim.AdamW(actor.parameters(),  lr=3e-4, weight_decay=0.0)
optim_critic = torch.optim.AdamW(
    list(critic_q1.parameters()) + list(critic_q2.parameters()),
    lr=3e-4, weight_decay=0.0,
)
optim_alpha  = torch.optim.Adam([log_alpha], lr=3e-4)
```

### 8.2 Stage B-plus (S2 / S3) — extra param groups for perception

On top of §8.1, add groups for the unfrozen perception-stream parameters (see `strategy_full_pipeline.md` §6.8):

| Group | Members | LR (S2)  | LR (S3)  | Weight decay |
|---|---|---|---|---|
| **PP low** (`pp.pillar_encoder`, `pp.backbone`) | only in S3 | —        | `3e-6`   | `1e-4` |
| **PP neck** (`pp.neck`) | S2 / S3 | `1e-5`   | `1e-5`   | `1e-4` |
| **Risk encoder** (`spatial_reducer`, `mamba`) | S2 / S3 | `5e-5`   | `5e-5`   | `1e-4` |
| **Risk head** (`risk_head`) | S2 / S3 | `1e-4`   | `1e-4`   | `1e-4` |

These groups receive gradient **only from the aux BCE loss** (§9.2), not from the SAC losses. The SAC optimizer still has just 2 groups (actor, critic).

### 8.3 Gradient clipping & targets

- See also `docs/skill_avoid_gradient_boom.md` for the full playbook (AMP, reward scale, observation norms, checklists).
- `torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)` after actor backward.
- `torch.nn.utils.clip_grad_norm_(critic_params, 1.0)` after critic backward.
- **Polyak averaging for target critics:** `tau = 0.005`, applied every update.
- **No target encoder** needed — there is no encoder on the SAC path.

---

## 9. Auxiliary losses

### 9.1 Default plan (S1) — NO aux loss

The default plan has **no aux loss** because the perception stream is entirely frozen. The Stage A-pretrained predictor is the implicit "anchor" — there is nothing to forget. Do not add auxiliary losses in S1; they buy nothing and cost compute.

### 9.2 Stage B-plus (S2 / S3) — focal BCE on RiskHead

When you unfreeze any part of the perception stream, you **must** add the aux BCE loss with a `λ_aux` schedule to anchor the risk predictor:

```python
# risk_gt comes from PyBullet contact lookahead (same rule as Stage A §5.1).
# Only a ~20 % subset of transitions carries risk_gt + pts_window, to keep the
# buffer cheap. Compute L_aux on that subset only.
L_aux = focal_bce(risk_head_logits, risk_gt, gamma=2.0, weight=(1.0, 0.8, 0.5))

# Optimize perception params via a DEDICATED backward pass — do not mix with SAC
# gradients. This isolates the aux-loss gradient and avoids double counting.
L_aux_total = lambda_aux * L_aux
L_aux_total.backward()
optim_perception.step(); optim_perception.zero_grad()
```

`λ_aux` schedule (see `strategy_full_pipeline.md` §6.8): `1.0 → 0.3 → 0.1 → 0.05` at steps `0 → 100 k → 500 k → 1 M`. Never zero.

### 9.3 Optional self-supervised signals

Not in scope for this plan. The SAC path sees only proprio, so there is no representation to self-supervise there. If you activate Stage B-plus and still see risk-predictor drift, increase `λ_aux` before adding any SSL head.

---

## 10. Environment (PyBullet)

### 10.1 Skeleton — aligned with full_pipeline §5.1 / §6.6

> The env used here **must match** the Stage A dataset generator camera / timing spec, otherwise the frozen risk branch sees out-of-distribution BEV features. Authoritative values live in `strategy_full_pipeline.md` §5.1 (`DataGenConfig`) and §6.6 (`EnvConfig`); this table mirrors them.

| Component      | Spec                                                                                             |
|----------------|--------------------------------------------------------------------------------------------------|
| Robot          | URDF of robot dog (`pybullet_data`'s a1 / laikago or your custom URDF)                           |
| Sensor         | Depth camera on head, **160 × 120**, `fov_h = 90°`, `near = 0.1`, `far = 8.0` m                  |
| Control / sensing frequency | **20 Hz** (`dt = 0.05`). Matches T_ctx = 10 ⇒ 0.5 s Mamba window.                  |
| Action space   | `Box(low=-1, high=1, shape=(A,))` → scaled to `(v_x, v_y, ω_yaw)` before applying                |
| Observation returned by `step()` | `{depth: (120, 160) float32 in [0, 1], proprio: (d_s,) float32}` — depth goes to perception stream, proprio goes to actor/critic. These are NOT concatenated. |
| Goal           | Random 2D point in a 5 m radius, spawned each episode (indoor scale)                             |
| Obstacles      | 5–15 random boxes / cylinders static + 0–3 moving in 30 % of scenes                              |
| Episode length | **400 env steps (20 s at 20 Hz)**. Long enough for Mamba to reason about 2 s horizons at any t.  |
| Reset          | Randomize robot pose, goal, obstacle layout; call `mamba_streamer.reset(env_idx)`.               |

### 10.2 Proprioceptive state spec

Exact layout of the `proprio` vector returned by `env.read_proprio()`:

```
s = concat(
    base_lin_vel   (3),         # body-frame linear velocity, m/s
    base_ang_vel   (3),         # body-frame angular velocity, rad/s
    goal_rel       (3),         # goal position in body frame, m  (last entry: z)
    heading_err    (1),         # atan2 to goal direction in body frame, rad
    last_action    (A,),        # previous action, recommended for smoothness
  [ joint_q        (dof,) ],    # optional, enable via cfg.include_joint_state
  [ joint_dq       (dof,) ],    # optional, enable via cfg.include_joint_state
)
```

Default `d_s` (no joint state): `3 + 3 + 3 + 1 + A = 10 + A`. With `A = 3`, `d_s = 13`.
With joint state on a 12-DoF quadruped: `d_s = 13 + 24 = 37`.

### 10.3 Domain randomization (applied at reset)

- Camera intrinsics: fx/fy ±10 %, image size fixed at 160 × 120.
- Camera extrinsics: mount height 0.3 – 0.6 m (robot-dog scale), pitch ±10°.
- Lighting: ambient 0.3 – 0.8.
- Obstacle count, size, material.
- Floor texture.
- Depth noise: additive Gaussian σ = 0.01 m; drop-pixel probability 2 %.

Keep consistent with `DataGenConfig` in the full-pipeline doc.

---

## 11. Reward design

### 11.1 Decomposition (matches full_pipeline §6.6)

```
r_env_t   =  w_goal * 1[reached_goal]                # terminal positive
          + w_progress * (dist_{t-1} - dist_t)       # dense positive
          - w_collision * 1[collision]               # terminal negative
          - w_time                                   # tiny step penalty
          - w_action_norm * ||a_t||^2                # action regularization
          - w_spin * |omega_yaw|                     # optional: discourage spinning

r_risk_t  = -lambda_risk * p_risk_1s_t               # from frozen perception stream
                                                     # (set lambda_risk = 0 for BASELINE)

r_total_t = r_env_t + r_risk_t                       # what SAC optimizes
```

### 11.2 Suggested coefficients

| Coefficient    | Default |
|----------------|--------:|
| `w_goal`       |     5.0 |
| `w_progress`   |     1.0 |
| `w_collision`  |    20.0 |
| `w_time`       |    0.01 |
| `w_action_norm`|   0.001 |
| `w_spin`       |    0.05 |
| **`lambda_risk`** (PROPOSED) | **2.0** |
| `lambda_risk` (BASELINE) | `0.0` |

### 11.3 Tuning rules

1. Monitor `E[r_risk] / E[r_env]` every 10 k env steps. Target band: `[-0.4, 0.0]`.
2. If the policy becomes overly cautious (stands still to avoid risk): lower `lambda_risk` first, then `w_collision`.
3. If collision rate stays high in PROPOSED: raise `lambda_risk` gradually (step 0.5 at a time). Raising `w_collision` alone is slower — `p_risk` provides a **dense** signal where `w_collision` is sparse.
4. `r_env` and `r_risk` are **stored separately** in the buffer (see full_pipeline §6.4). You can **re-weight `lambda_risk`** offline during gradient updates without re-rolling out episodes: `r_total = r_env + (new_λ / old_λ) · r_risk`.

---

## 12. Hyperparameters

### 12.1 SAC-internal (all regimes)

| Name                        | Default                                                 |
|-----------------------------|---------------------------------------------------------|
| Discount γ                  | `0.99`                                                  |
| Polyak τ                    | `0.005`                                                 |
| Batch size                  | `256` (S1 default; trivial memory) / `128` (S2 / S3)    |
| Replay capacity             | `1 M` transitions (default) / `500 k` (Stage B-plus, to leave RAM for `pts_window`) |
| Warmup (random actions)     | `5 k` env steps                                         |
| Gradient steps per env step | `1`                                                     |
| Target entropy              | `-dim(action)`                                          |
| Initial `log_alpha`         | `0.0` (α = 1)                                           |
| Actor LR                    | `3e-4`                                                  |
| Critic LR                   | `3e-4`                                                  |
| Alpha LR                    | `3e-4`                                                  |
| Gradient clip               | `1.0`                                                   |
| Total env steps (budget)    | `2 M` (nav-easy) / `5 M` (nav-hard)                     |

### 12.2 Reward-shaping hyperparameters

| Name          | Default                    |
|---------------|---------------------------:|
| `lambda_risk` | `2.0` (PROPOSED) / `0.0` (BASELINE) |
| `T_ctx`       | `10` (history window for Mamba) |

### 12.3 Stage B-plus extras (only when aux loss is active)

| Name                 | Default                                                         |
|----------------------|-----------------------------------------------------------------|
| `lambda_aux` schedule| `1.0 → 0.3 → 0.1 → 0.05` at steps `0 → 100 k → 500 k → 1 M`    |
| PP neck LR           | `1e-5`                                                          |
| PP backbone LR (S3)  | `3e-6`                                                          |
| SpatialReducer LR    | `5e-5`                                                          |
| Mamba LR             | `5e-5`                                                          |
| RiskHead LR          | `1e-4`                                                          |
| Aux subset ratio     | 20 % of transitions carry `pts_window` + `risk_gt`              |

---

## 13. Hardware & wall-clock estimates

Default plan (S1) is very cheap because gradient steps do not touch the perception stream. Wall-clock is dominated by PyBullet (CPU) and the per-env-step perception forward.

| GPU                    | Regime | Batch | ~FPS env step | Time to 2 M steps |
|------------------------|--------|------:|--------------:|------------------:|
| RTX 3060 12 GB         | S1     |   256 |           180 |             ~3 h  |
| RTX 3090 24 GB         | S1     |   256 |           280 |             ~2 h  |
| Colab T4 16 GB         | S1     |   128 |            80 |             ~7 h (watch session caps) |
| Colab A100 40 GB       | S1     |   256 |           300 |             ~2 h  |
| RTX 3060 12 GB         | S2     |   128 |            60 |             ~9 h  |
| RTX 3090 24 GB         | S3     |   128 |            80 |             ~7 h  |

Use `SubprocVecEnv` with 4–8 workers to improve PyBullet throughput. `torch.cuda.amp` helps little here because the trainable net is a tiny MLP; apply AMP only to the perception forward if memory becomes an issue.

---

## 14. Risk register

| Risk                           | Signal                                                                                   | Mitigation                                                                                                           |
|--------------------------------|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| **Reward hacking**             | Agent exploits a shaping bug (e.g. backs up against wall because `p_risk` is lower there). | Inspect rollouts every 100 k steps; log `E[r_risk]/E[r_env]`; cap `lambda_risk` ≤ 3.0.                               |
| **BN leakage** (all regimes)   | You forgot `.eval()` on a frozen module after an external `policy.train()`.             | Install the BN-lock hook in §4.2. Unit test: assert `pp.pillar_encoder.training == False` mid-training.              |
| **Out-of-range points**        | Policy drifts the camera pose and few points survive range filter.                      | Log average `N_pts` per frame; clip camera extrinsics; include a small reward for staying in range.                  |
| **Replay / on-policy drift**   | Early random-exploration data dominates buffer; slow convergence.                       | Prioritized Experience Replay (PER) if SAC stalls at 30 %+ buffer fill.                                              |
| **Perception timing miss**     | Perception forward exceeds 50 ms on target GPU → control-loop jitter.                   | Batch at most one env per GPU for streaming Mamba; use AMP; reduce `point_cloud_range` if needed.                    |
| **Mamba streaming hidden-state leak** | Hidden state carries across episode boundaries and poisons future frames.         | Register env-done callback → `MambaStreamer.reset(env_idx)`. Unit-test: hidden norm drops to 0 after reset.          |
| **Risk branch out-of-dist**    | `p_risk` histogram is stuck ≈ 0 or ≈ 1.                                                  | Opt into Stage B-plus (full_pipeline §6.8); unfreeze with aux BCE.                                                   |
| **Catastrophic forgetting** (Stage B-plus only) | `risk_1s` AUC drops below 0.75 during Stage B-plus.                   | Raise `lambda_aux`; lower unfrozen-branch LR; demote S2 → S1 if severe.                                              |

---

## 15. Debugging / sanity checks

Run each of these before training long.

1. **Perception determinism.** With the same `pts`, in `.eval()`, `extract_neck` and the full risk-branch forward must return identical outputs on two calls.
2. **Freeze correctness (S1).** After `apply_freeze_policy(pp, risk_branch, "S1")`, `sum(p.requires_grad for p in list(pp.parameters()) + list(risk_branch.parameters()))` must equal 0.
3. **Gradient flow correctness (S1).** Do one update step; then check:
   - `actor.parameters()` → non-zero grads.
   - `critic_q1/q2.parameters()` → non-zero grads.
   - `pp.*` and `risk_branch.*` → all grads `None`.
4. **Gradient flow correctness (S2, Stage B-plus).** Same as (3) but also `pp.neck.parameters()` and `risk_branch.*` grads are non-zero **only after the aux-BCE backward**; they should be `None` right after the SAC backward.
5. **Random-policy smoke test.** Actor random, 1 k env steps; no NaNs in critic loss; replay buffer fills.
6. **Reward decomposition.** Log `r_goal`, `r_progress`, `r_collision`, `r_time`, `r_risk` separately. All finite and bounded.
7. **`E[r_risk] / E[r_env]` sanity.** Should be in `[-0.4, 0.0]`. Outside → tune `lambda_risk`.
8. **Action statistics.** At step 10 k, action mean must not already be saturated at ±1.
9. **`p_risk` histogram.** Every 50 k steps, plot histogram over most-recent 10 k transitions. Healthy: right-skewed with mass in `[0.0, 0.3]`, heavy tail near 1.0 for near-collision frames. Bad: sharp peak at 0.5 (uninformative) or always 0/always 1 (OOD).
10. **Mamba streamer invariant.** After `MambaStreamer.reset(env_idx)`, `streamer.h[env_idx] is None` and the first `mamba.step` after reset returns a finite `h_t` with norm < 10.
11. **A/B comparability.** BASELINE and PROPOSED runs use IDENTICAL seeds, IDENTICAL env configs, IDENTICAL proprio specs. Unit-test: assert hashes of `EnvConfig` and proprio layout match.

---

## 16. Roadmap / milestones

> This roadmap assumes Stage A is already complete (see `strategy_full_pipeline.md` roadmap for the Stage A milestones).

| Phase                              | Duration (estimate) | Exit criterion                                                                                  |
|------------------------------------|---------------------|-------------------------------------------------------------------------------------------------|
| **P0. Env skeleton**               | 3–5 days            | `reset()` + `step()` work; depth values metric; scripted policy reaches goal; camera spec matches DataGenConfig; `read_proprio()` returns correct vector. |
| **P1. Perception plumbing**        | 1–2 days            | `pybullet_depth_to_meters → preprocess_depth_frame → extract_neck → risk_branch.step` runs at ≥ 20 Hz on target GPU; `MambaStreamer` resets correctly on `done`. |
| **P2. Risk-branch loader**         | 1 day               | `rl/risk_branch.py` loads a Stage A checkpoint, freezes everything, streaming `step()` returns `p_risk` in < 5 ms. |
| **P3. SAC scaffolding**            | 2–3 days            | Default plan (S1) trains BASELINE; baseline return curve logged.                                |
| **P4. A/B comparison**             | 1 week compute      | Run BASELINE and PROPOSED with 3 seeds each; report success rate, collision rate, mean steps to goal. If PROPOSED wins clearly, ~90 % chance you're done. |
| **P5. (optional) Stage B-plus S2** | 1 week              | Only if the frozen risk predictor is clearly miscalibrated (see §15.9). S2 beats S1 with aux BCE on. |
| **P6. (optional) S3 / Dreamer**    | only if P5 doesn't help | Replace SAC with a world-model agent reusing the same perception stack.                    |

---

## 17. Implementation checklist

**Environment**
- [ ] PyBullet env class conforming to Gymnasium API (`reset`, `step`, `observation_space`, `action_space`).
- [ ] Depth camera integration (`getCameraImage`) at 160 × 120, 90° FoV, near 0.1, far 8.0.
- [ ] Control/sensing at 20 Hz (`dt = 0.05`).
- [ ] Episode length 400 steps.
- [ ] `read_proprio()` returns the proprio vector specified in §10.2.
- [ ] Domain randomization knobs (matches `DataGenConfig` in full_pipeline §5.1).
- [ ] Deterministic eval mode (seeded).

**Perception**
- [ ] Reuse `PointPillarsNeckExtractor` and `preprocess_depth_frame` from `module_pointpillar.py`.
- [ ] `rl/risk_branch.py` loads Stage A checkpoint and exposes `step(pts, h_prev) -> (p_risk, h_new)`.
- [ ] `apply_freeze_policy(pp, risk_branch, strategy)` + BN-lock hook (§4.2) installed on the top module.
- [ ] `MambaStreamer` (full_pipeline §6.4.2) maintained per env; reset on `done`.

**Networks**
- [ ] `Actor` (squashed Gaussian, MLP on proprio; §7.2).
- [ ] `Critic` (twin Q, MLP on (proprio, action); §7.3).
- [ ] Target critic copies with Polyak update τ = 0.005. **No target encoder** exists in this plan.

**Optimizer**
- [ ] Separate `optim_actor`, `optim_critic`, `optim_alpha` (§8.1).
- [ ] Gradient clipping `max_norm = 1.0` on each backward.
- [ ] (Stage B-plus only) `optim_perception` for unfrozen perception params, driven by aux BCE loss only.

**SAC loop**
- [ ] Replay buffer stores `(s, s_next, action, r_env, r_risk, done, episode_id, frame_idx)` (full_pipeline §6.4). No `pts`, no BEV, no `z`.
- [ ] `r_total = r_env + r_risk` in TD target. `r_risk = 0` in BASELINE config.
- [ ] Warmup with random actions (5 k env steps).
- [ ] α auto-tuning via target entropy `-dim(action)`.
- [ ] TensorBoard / wandb logging of: episode return (total, `r_env`-only, `r_risk`-only), success rate, collision rate, critic loss, actor loss, α, `p_risk` histogram, grad norms.
- [ ] Checkpointing (actor, critic, target critics, optimizers, `log_alpha`) every N env steps.
- [ ] Evaluation loop on fixed held-out scenes every 10 k env steps.

**A/B test harness**
- [ ] Single config knob `lambda_risk`: `0.0` for BASELINE, `2.0` for PROPOSED.
- [ ] Run both with matching seeds (at least 3 seeds each).
- [ ] Report identical metrics side by side; no other hyperparameter differs.

**Stage B-plus (only if activated)**
- [ ] Buffer augmented with `pts_window` + `risk_gt` for ~20 % of transitions.
- [ ] `focal_bce(risk_head_logits, risk_gt)` added as a separate backward pass with `lambda_aux` schedule (§9.2).
- [ ] `risk_1s` AUC must stay ≥ 0.75 throughout; auto-revert to S1 if it drops below 0.70.

**Safeguards**
- [ ] Unit tests from §15.
- [ ] Auto-revert logic if promotion (S1 → S2) drops return by > 15 % over 50 k steps.

---

## 18. Open questions / future work

- **Richer proprio.** If BASELINE plateaus well below a reasonable target, add joint state + last action + short proprio history (frame stacking).
- **Feeding `p_risk` directly to the policy.** An alternative to reward shaping: concatenate `p_risk_1s` to the proprio vector. Changes the A/B test framing; revisit only after the reward-shaping A/B is conclusive.
- **Prioritized Experience Replay.** Add if sample efficiency is the bottleneck.
- **Sim-to-real gap.** Domain randomization here is a down-payment; a separate real-world calibration pass will be needed before deployment.
- **Mixed precision.** `torch.cuda.amp.autocast` on the perception forward halves activation memory; verify PointPillars ops support it.
- **Dreamer-v3 upgrade path.** If SAC plateaus even in PROPOSED, replace the policy by a world model that reuses the same frozen perception stack.

---

## 19. Quick reference — who owns what

| File                                     | Owns                                                                                              |
|------------------------------------------|---------------------------------------------------------------------------------------------------|
| `module_pointpillar.py`                  | Perception: depth → BEV neck feature. No RL logic, no risk logic.                                 |
| `module_pointpillar.md`                  | Spec of the perception module.                                                                    |
| `strategy_full_pipeline.md`              | Pipeline architecture, data contracts, Stage A + Stage B orchestration, replay buffer layout, env spec, reward decomposition, A/B test framing, Mamba streaming. |
| `strategy_finetune_with_SAC.md` (this)   | SAC-specific algorithmic details: Actor/Critic MLPs on proprio, BN-lock hook, freeze regimes S1/S2/S3, SAC hyperparameters, debugging checklist. |
| `rl/env_pybullet.py` *(to be created)*   | PyBullet Gym env. Camera spec per full_pipeline §5.1. `read_proprio()` per §10.2.                 |
| `rl/networks.py` *(to be created)*       | `Actor`, `Critic` (MLPs on proprio).                                                              |
| `rl/risk_branch.py` *(to be created)*    | Loader that assembles frozen `SpatialReducer + Mamba + RiskHead` from Stage A checkpoint. Implements `MambaStreamer`. |
| `rl/sac_agent.py` *(to be created)*      | SAC update logic per §6.2 of this doc.                                                            |
| `rl/train_sac.py` *(to be created)*      | Main loop, reward shaping hook (`r_total = r_env + r_risk`), logging, checkpointing, A/B harness. |
| `rl/aux_losses.py` *(to be created, Stage B-plus only)* | Focal-BCE aux loss wiring and `lambda_aux` scheduler.                              |

---

## 20. One-line TL;DR for the next agent

> Load a Stage A checkpoint, freeze the entire perception stream. Train SAC with **plain MLP Actor/Critic on a proprioceptive state vector** (no BEV, no encoder). Reward is `r_total = r_env + r_risk` where `r_risk = -λ_risk · p_risk_1s` comes from the frozen Mamba + RiskHead. Compare `λ_risk = 0` (BASELINE) vs `λ_risk = 2` (PROPOSED) with matched seeds to measure whether the pretrained risk predictor improves RL in dynamic-obstacle scenes. Only consider unfreezing the perception stream (Stage B-plus) if the A/B test shows the risk predictor is clearly miscalibrated.

---

## 21. Maintenance rule — keep these docs in sync

This doc is one leg of a three-doc spec set: `module_pointpillar.md`, `strategy_full_pipeline.md`, `strategy_finetune_with_SAC.md`. Any change to code OR ideas MUST keep all three coherent, otherwise the next agent will act on stale context. For the canonical, long-form version of this rule see `strategy_full_pipeline.md` § 14 — the section below is the SAC-specific shortcut.

### 21.1 Trigger matrix — when to update what

| You are changing…                                                             | Must update                                                                                          |
|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Actor / Critic architecture (layer shapes, activations, output bounds)        | This doc § 7 (owner) + cross-ref in `strategy_full_pipeline.md` § 4.5 / § 4.6.                       |
| Optimizer groups, LRs, gradient-clip, Polyak τ                                | This doc § 8 + § 12 (owner) + mirror in `strategy_full_pipeline.md` § 6.2.                           |
| Freeze regimes S1 / S2 / S3 (which modules unfreeze, at what LR)              | This doc § 4 (owner) + cross-ref in `strategy_full_pipeline.md` § 6.1 + § 6.8.                       |
| BN-lock hook (`install_bn_lock`, `apply_freeze_policy`)                       | This doc § 4.2 (owner) + cross-ref in `strategy_full_pipeline.md` § 6.5.                             |
| SAC update pseudocode (order of backward passes, detach sites)                | This doc § 6.2 (owner).                                                                              |
| Aux BCE loss wiring for Stage B-plus                                          | This doc § 9.2 (owner) + cross-ref in `strategy_full_pipeline.md` § 6.8.                             |
| Debugging checklist for gradient-flow assertions                              | This doc § 15 (owner).                                                                               |
| Pipeline architecture (new module, reward path, branches)                     | `strategy_full_pipeline.md` (owner) FIRST; then update § 1 / § 2 of this doc.                        |
| Reward formula / `λ_risk` default / A/B framing                               | `strategy_full_pipeline.md` § 6.3 (owner) + mirror in § 11 of this doc.                              |
| Env / camera spec                                                             | `strategy_full_pipeline.md` § 5.1 / § 6.6 (owner) + mirror in § 10 of this doc.                      |
| Replay buffer contents                                                        | `strategy_full_pipeline.md` § 6.4 (owner) + cross-ref in § 6.3 of this doc.                          |
| Perception module API                                                         | `module_pointpillar.md` (owner) + cross-ref in § 7.1 of this doc.                                    |

### 21.2 Per-PR checklist

Any PR that touches this doc or RL code must satisfy:

1. **Ownership respected.** Changes are made in the owning doc first (per § 0 + § 19 tables and § 21.1 above).
2. **Changelog row added** in every doc actually modified.
3. **Version bump** if the change breaks a downstream consumer (e.g. Actor/Critic signature change, replay-buffer field removed, BN-lock hook signature change).
4. **Grep pass for stale terms.** See the sweep commands in `strategy_full_pipeline.md` § 14.5.
5. **Cross-reference audit.** All `§ X.Y` pointers still resolve.
6. **Code ↔ doc parity.** Doc code blocks (e.g. `Actor`, `Critic`, `apply_freeze_policy`) must match `rl/networks.py` / `rl/sac_agent.py` verbatim. If doc is ahead of code, add `> **Proposed — not yet implemented.**` to the affected section.
7. **A/B test invariance.** Any change that affects BOTH configurations identically (proprio layout, optimizer LR, SAC internals) is fine. Any change that applies ONLY to PROPOSED (e.g. new `λ_risk` schedule) must be documented in § 11.3 and flagged in § 15.11.

### 21.3 Idea-only changes (no code yet)

Use the same convention as the other two docs: `> **Proposed — not yet implemented.**` banner + `## Pending sync` row in the changelog, removed when code lands.

### 21.4 Conflict resolution

1. Row owner in § 19 wins.
2. `strategy_full_pipeline.md` is the architectural tiebreaker if ownership is ambiguous.
3. Never resolve silently — add a changelog row explaining the fix.

### 21.5 Minimum grep checks before merging

Same commands as `strategy_full_pipeline.md` § 14.5. Run them across `PointPillars_module/*.md` and fix any non-intentional hit.

---

## 22. Changelog

| Date       | Author | Change                                              |
|------------|--------|-----------------------------------------------------|
| 2026-04-18 | v1     | Original SAC-on-PointPillars plan (PP as encoder).  |
| 2026-04-18 | v2     | Split perception into state branch + risk branch; added `BEVStateExtractor`; critic-only encoder update rule. |
| 2026-04-18 | v3     | **Pipeline simplification** — PointPillars is NOT on the SAC gradient path. Actor/Critic are plain MLPs on a proprioceptive state vector. Removed `BEVStateExtractor`, removed the critic-only encoder update rule (no encoder exists on the SAC path). Buffer stores proprio state only (~500 B / transition). Framed explicitly as an A/B test on `λ_risk`. Aux BCE loss now runs as a **separate backward pass** in Stage B-plus to keep SAC gradient clean. |
| 2026-04-18 | v3.1   | Added § 21 "Maintenance rule — keep these docs in sync": SAC-specific trigger matrix, per-PR checklist, A/B-test invariance rule. Renumbered Changelog from § 21 to § 22. Canonical long-form rule lives in `strategy_full_pipeline.md` § 14. |
