# Strategy — Full Pipeline: PointPillars → Mamba → RiskHead → SAC

> **Audience:** the next agent / engineer who will implement this.
> **Goal of this document:** capture the full technical plan for the two-stage training pipeline — (A) supervised pretrain of the perception + temporal + risk stack on a PyBullet dataset, then (B) RL fine-tune with Soft Actor-Critic. Every component, data contract, loss, optimizer and risk is specified so you can start coding from it without re-deriving the reasoning.
> **Language:** English. All identifier names in this doc match what should appear in code.
>
> **Before editing this doc, read § 14 "Maintenance rule — keep these docs in sync".** This doc is the architectural source of truth. Any change to the pipeline diagram, data contracts, freeze policy, reward formula or camera/env spec MUST be mirrored in `module_pointpillar.md` and `strategy_finetune_with_SAC.md` in the same PR.

---

## 0. Document scope

This spec covers:

1. The role of every block in the pipeline and why it exists.
2. Tensor shapes, dtypes and dataclasses at every interface.
3. Dataset generation in PyBullet for Stage A.
4. Stage A (supervised pretrain) training loop, loss and schedule.
5. Stage B (SAC fine-tune) training loop, loss, multi-param-group optimizer and auxiliary loss.
6. Risk register, debugging guide and implementation checklist.

It does **not** cover sim-to-real deployment on the physical robot dog — that is a separate document.

---

## 1. High-level pipeline

### 1.1 Architectural principle (read this first)

Perception and control are **two fully decoupled streams**. They only meet at the reward.

| Stream              | What it consumes                        | What it produces                              | Frozen in Stage B? |
|---------------------|-----------------------------------------|-----------------------------------------------|--------------------|
| **Perception stream** (PointPillars → SpatialReducer → Mamba → RiskHead) | depth / point cloud from the sensor | `p_risk` ∈ [0, 1]³ (collision probabilities at 0.5 s, 1 s, 2 s) → reward shaping term `r_risk = -λ_risk · p_risk_1s` | **Yes, fully frozen** |
| **Control stream** (Actor / Critic, SAC) | a low-dim **proprioceptive + goal** vector (base linear / angular velocity, goal relative position, optional heading) | continuous action `a_t` | No (this is what SAC trains) |

**Key point:** the Actor and Critic **never see BEV features**. PointPillars is **not** the state encoder. It is a dedicated dense-reward function that runs in parallel to SAC. SAC consumes only proprioception + goal.

This design is deliberate: the experimental goal is an **A/B test** — compare SAC performance in dynamic-obstacle environments when the only difference is the presence of `r_risk` in the reward:

- **Baseline (A):** `reward = r_env`
- **Proposed (B):** `reward = r_env + r_risk`

Everything else (Actor, Critic, state, environment, hyperparameters) is identical. That's how we measure whether the pretrained risk predictor adds value.

### 1.2 Block diagram

```
   Stage A (supervised pretrain of the perception stream):
   -------------------------------------------------------------------------
   depth_buf ──▶ PP_preproc ── pts ──▶ PointPillars ──▶ BEV feat (B,384,H,W)
                                                              │
                                                              ▼
                                                      SpatialReducer
                                                              │
                                                   tokens (B, Nt=16, D=256)
                                                              │
                                                    stack over T_ctx frames
                                                              │
                                                              ▼
                                                   Mamba  (B, T*Nt, D) → h_T (B, D)
                                                              │
                                                              ▼
                                                         RiskHead
                                                              │
                                                              ▼
                                                       logits (B, 3) ─── Focal-BCE loss
                                                                           ↑
                                                               collision GT from PyBullet


   Stage B (two decoupled streams during RL):
   -------------------------------------------------------------------------

   ─── Perception stream (frozen, runs every env step at 20 Hz) ───────────
   depth_t ─▶ PP_preproc ─▶ pts_t ─▶ PointPillars ─▶ bev_t
                                                        │
                                                        ▼
                                                 SpatialReducer
                                                        │
                                               tok_t (1, 16, 256)
                                                        │
                                        streaming Mamba (carries h_{t-1})
                                                        │
                                                   h_t (1, 256)
                                                        │
                                                        ▼
                                                   RiskHead
                                                        │
                                                  p_risk_t (1, 3)
                                                        │
                                         r_risk_t = -λ_risk · p_risk_t[1s]


   ─── Control stream (SAC, trainable, also at 20 Hz) ─────────────────────
   env.read_proprio ─▶ s_t (proprio + goal, ~30–50 dim)
                                  │
                                  ▼
                              Actor π(a|s_t)           Critic Q(s_t, a)
                                  │
                                  ▼
                                a_t
                                  │
                         env.step(a_t)  ─▶  s_{t+1},  r_env_t,  done_t


   ─── Join at reward ────────────────────────────────────────────────────
   r_total_t = r_env_t + r_risk_t     (with config BASELINE, just r_env)

   ─── Replay buffer ────────────────────────────────────────────────────
   push (s_t, a_t, r_env_t, r_risk_t, s_{t+1}, done_t)
   SAC update consumes ONLY these fields. PointPillars is NEVER re-run
   during a gradient step — r_risk was already computed online.
```

### 1.3 Stage A vs Stage B at a glance

| Aspect              | Stage A (supervised pretrain)                    | Stage B (SAC with frozen perception stream)         |
|---------------------|--------------------------------------------------|-----------------------------------------------------|
| Loss                | Focal-BCE on collision horizons                  | Standard SAC (actor + 2 critics + entropy). NO aux loss, NO perception gradient. |
| Labels used         | Yes (collision GT from PyBullet contact events)  | None                                                |
| Data source         | Offline PyBullet rollouts (fixed dataset)        | Online PyBullet env + replay buffer                 |
| Trainable modules   | PointPillars (A2 only), SpatialReducer, Mamba, RiskHead | **Only Actor, twin Critics, and `log_alpha`**. The entire perception stream is frozen in `.eval()`. |
| Reward shaping      | —                                                | `r_total = r_env − λ_risk · p_risk_1s` (when enabled)  |
| Primary goal        | Learn a trajectory-aware risk predictor          | Test whether the pretrained risk predictor improves SAC sample efficiency and collision rate versus a baseline that only uses `r_env`. |
| Expected wall-clock | 15–25 h (A100) / 30–45 h (T4)                    | **Very fast.** The policy network is tiny MLP; the only per-step cost is the frozen perception forward (~10 ms). 2 M steps in 2–5 h (A100) / 6–12 h (T4). |

> **Why fully freeze the perception stream?** It's a prerequisite for the A/B test: if the risk predictor changes during Stage B, the "proposed" configuration is not a clean reward-shaping experiment — it's a joint representation-learning experiment with many confounders. Freezing isolates the effect of `r_risk`.

---

## 2. Component context (WHY each block exists)

### 2.1 `PointPillars` — spatial perception (shared)

Turns a point cloud into a BEV feature map.

- **Why kept:** BEV is a dense, spatially-structured representation; much easier for downstream modules than raw point sets.
- **Why frozen in Stage B:** a single frozen PointPillars instance is shared by both branches (state + risk). Freezing PointPillars guarantees the risk branch stays self-consistent (its frozen Mamba + RiskHead were trained on a specific BEV distribution).
- **Why the head is stripped:** we do not need detection outputs; only BEV features. See `module_pointpillar.md` for the API (`extract_neck`).

### 2.2 `SpatialReducer` — dimensional bridge (risk branch)

Compresses `(B, 384, 248, 216)` from PointPillars down to a small token grid per frame, used as Mamba input.

- **Why needed:** BEV feature has ≈20 M floats per frame. Feeding directly into Mamba is impractical. Mamba wants a sequence of tokens.
- **Why CNN + 4×4 grid pool:** global avg pool destroys spatial layout; a `(Nt=16, D=256)` token grid per frame keeps coarse spatial structure which Mamba can reason about across time.
- **Role split:** the SpatialReducer is part of the **perception stream only** (risk prediction). The SAC control stream never touches it.

### 2.3 `Mamba` — temporal obstacle-trajectory encoder

A selective state-space model (SSM) that encodes a sequence of token grids over `T_ctx` frames into a temporal hidden state `h_T`.

- **Why Mamba (not Transformer):**
  - Linear complexity `O(L)` — important for real-time robot loop (20 Hz).
  - Streaming inference: the hidden state `h_{t}` carries across time; at deployment we update one frame at a time without reprocessing history.
  - Competitive with Transformer on long sequences.
- **Why not LSTM/GRU:** weaker long-range modeling; gradient issues on long sequences. A GRU fallback exists in case `mamba-ssm` fails to build (see § 4.3).
- **"Trajectory" interpretation:** `h_t` is an **implicit latent summary of obstacle trajectories** — it encodes where obstacles have been and, by extension, where they are likely to go. We do NOT force the model to output explicit `(x, y)` waypoints.

### 2.4 `RiskHead` — collision probability predictor

A tiny MLP that reads `h_T` and predicts collision probability for three horizons (0.5 s, 1 s, 2 s).

- **Why multi-horizon (not binary):** near-term vs far-term collision are distinct signals; multi-horizon gives a richer gradient and better calibrated risk.
- **Role in Stage B:** its output `p_risk` is consumed by the reward shaping term — **not** fed into the policy network. See § 2.6 and § 6.3.

### 2.5 `Actor` / `Critic` — SAC policy and value (proprio-based)

Standard SAC on top of the **proprioceptive + goal vector** `s` produced by the environment (no BEV, no PointPillars).

```python
@dataclass
class ProprioState:
    base_lin_vel:   np.ndarray   # (3,)  base linear velocity in body frame (m/s)
    base_ang_vel:   np.ndarray   # (3,)  base angular velocity (rad/s)
    goal_rel:       np.ndarray   # (3,)  goal position relative to base in body frame (m)
    heading_err:    float        # (1,)  angle to goal direction (rad)
    # optional, add if available:
    joint_q:        Optional[np.ndarray]  # (dof,) joint angles
    joint_dq:       Optional[np.ndarray]  # (dof,) joint velocities
    last_action:    np.ndarray   # (A,)  previous action (useful for smoothing)
    # total dimension: ~10 without joints, ~30–50 with joint state
```

- **Why proprio only (not BEV):** the user-selected A/B-test framing requires that the only difference between baseline and proposed configurations is the reward term. Keeping state identical removes confounders. Also: a small proprio state trains much faster than an encoder-based state.
- **Actor / Critic = plain MLP.** No convolutions, no attention, no auxiliary encoders. See § 4.5 / § 4.6.
- **Why SAC (not PPO):** off-policy, sample-efficient, stable for continuous control, works well with replay buffers.

### 2.6 Why fully freeze the perception stream in Stage B

1. **Cleanliness of the A/B test.** The proposed config differs from baseline ONLY by `+ r_risk` in the reward. Any perception-stream update would introduce a second changing variable and muddy the comparison.
2. **Catastrophic forgetting is eliminated.** No gradient, nothing to forget.
3. **Buffer becomes trivial.** The perception output `r_risk` is computed online and stored as a float; no `pts`, no features, no `z` needed. Total buffer ≈ 400 MB (see § 6.4).
4. **Compute is cheap.** Only a small MLP Actor / Critic is backpropagated; the per-gradient-step cost is ~5 ms.

> **What if you want to also fine-tune perception eventually?** That's a separate experiment run AFTER the A/B test has been measured. Use the "optional Stage B-plus" recipe in § 6.8; it reintroduces aux BCE and small LRs on the perception stream but is explicitly out of scope for the first-round A/B comparison.

---

## 3. Data contracts (dataclasses)

All arrays are contiguous NumPy unless noted. All tensors are contiguous PyTorch. Stage A uses NumPy on disk; Stage B uses PyTorch in-memory.

### 3.1 Raw PyBullet rollout artifact (`Trajectory`)

```python
@dataclass
class Trajectory:
    """
    One contiguous rollout produced by the dataset generator.
    Saved as a single .npz file on disk.
    """
    scene_id:       int                    # which procedural scene
    rollout_id:     int                    # rollout index within scene
    T:              int                    # number of frames

    # perception
    depth:          np.ndarray  # (T, H_img, W_img) float16, meters
                                #  after `pybullet_depth_to_meters`
    rgb:            np.ndarray  # (T, H_img, W_img, 3) uint8, optional / unused

    # ego / camera
    cam_intrinsics: np.ndarray  # (4,) float32   [fx, fy, cx, cy]
    cam_extr_R:     np.ndarray  # (T, 3, 3) float32  camera→world
    cam_extr_t:     np.ndarray  # (T, 3) float32     camera origin in world
    ego_state:      np.ndarray  # (T, 6) float32  [x, y, z, roll, pitch, yaw]
    ego_vel:        np.ndarray  # (T, 6) float32  [vx, vy, vz, wx, wy, wz]

    # control
    action:         np.ndarray  # (T, A) float32, action emitted by the
                                #  dataset-gen policy at step t

    # ground truth risk (derived offline from contact events)
    contact_flag:   np.ndarray  # (T,)   bool, is the robot in contact at t
    risk_05s:       np.ndarray  # (T,)   float32 in {0.0, 1.0}
    risk_1s:        np.ndarray  # (T,)   float32 in {0.0, 1.0}
    risk_2s:        np.ndarray  # (T,)   float32 in {0.0, 1.0}

    # scene metadata (optional)
    obstacle_aabb:  np.ndarray  # (N_obs, 6) float32, static AABBs
```

Expected physical ranges (sanity checks at load time):

- `depth`: `[0.0, 10.0]` m
- `ego_state[:, :3]`: within the scene bounds (e.g. `[-5, 5]` m)
- `risk_*`: only `0.0` or `1.0`
- `contact_flag`: monotonically **not** required (contact can end)

### 3.2 In-memory training sample (`RiskSample`)

Built by the `torch.utils.data.Dataset` wrapper. One sample = one temporal window of length `T_ctx` ending at frame `t`.

```python
@dataclass
class RiskSample:
    """
    A single (input, target) pair fed into the Stage A network.
    """
    # inputs (already preprocessed, LiDAR frame)
    pts_seq:      List[torch.Tensor]  # length T_ctx; each (N_i, 4) float32
                                      # points already in LiDAR frame
    action_seq:   torch.Tensor        # (T_ctx, A) float32
    ego_vel_seq:  torch.Tensor        # (T_ctx, 6) float32

    # targets
    risk_05s:     torch.Tensor        # () float32, binary
    risk_1s:      torch.Tensor        # () float32, binary
    risk_2s:      torch.Tensor        # () float32, binary
    traj_future_xyyaw: torch.Tensor   # (H, 3) float32 — planar future ego poses
                                      # (x, y, yaw world) for frames t+1..t+H

    # meta (not used by model)
    scene_id:     int
    rollout_id:   int
    frame_t:      int
```

Note: `pts_seq` is a list (not a stacked tensor) because each frame has a different `N_i` after range filtering. The collate function batches them per-time-step and per-batch into the format PointPillars accepts.

### 3.3 Batched training tensors (`RiskBatch`)

```python
@dataclass
class RiskBatch:
    """
    Output of the DataLoader collate_fn. This is what the training step consumes.
    """
    # One list-of-list: outer = time, inner = batch.
    # pts_seq[t]  is a list of B tensors (N_b, 4). PointPillars accepts list[Tensor].
    pts_seq:      List[List[torch.Tensor]]   # len = T_ctx
    action_seq:   torch.Tensor               # (B, T_ctx, A) float32
    ego_vel_seq:  torch.Tensor               # (B, T_ctx, 6) float32

    risk_05s:     torch.Tensor               # (B,) float32
    risk_1s:      torch.Tensor               # (B,) float32
    risk_2s:      torch.Tensor               # (B,) float32
    traj_future_xyyaw: torch.Tensor          # (B, H, 3) float32 — same semantics as §3.2
```

`H` defaults to `10` frames (0.5 s @ 20 Hz). Used by `train_stage_a_compare` for joint risk + short-horizon trajectory supervision (`FullPipelineRiskAndTraj`); Stage B streaming still consumes risk only via `FullPipeline.step`.

### 3.4 Network intermediate tensors

All perception-stream tensors are computed under `torch.no_grad()` in Stage B (streaming, 1 env per forward).

| Symbol     | Produced by      | Shape                   | dtype   | Device | Notes |
|------------|------------------|-------------------------|---------|--------|-------|
| `bev_feat` | PointPillars     | `(1, 384, 248, 216)`    | float32 | cuda   | one env per forward |
| `tok_grid` | SpatialReducer   | `(1, Nt=16, D=256)`     | float32 | cuda   | |
| `h_t`      | Mamba (streaming) | `(1, D=256)`           | float32 | cuda   | carried across env steps |
| `p_risk`   | RiskHead         | `(1, 3)`                | float32 | cuda   | `[p_05s, p_1s, p_2s]` |

In Stage A (supervised pretrain) the same symbols appear with batch `B > 1` and a time dimension — see § 3.2 / § 3.3.

### 3.5 Stage B control-stream tensors

SAC operates on a proprioceptive vector `s` directly from the environment. No perception feature is ever fed to Actor / Critic.

| Symbol    | Shape        | Note                                                                 |
|-----------|--------------|----------------------------------------------------------------------|
| `s`       | `(B, d_s)`   | proprio + goal vector, `d_s ≈ 10` (minimal) to `~50` (with joint state) |
| `s_next`  | `(B, d_s)`   | same, from next step                                                  |
| `a_t`     | `(B, A)`     | continuous action from actor (e.g. `A = 3` for `(v_x, v_y, ω_yaw)`)  |
| `logp_a`  | `(B,)`       | log-prob of sampled action                                            |
| `q1, q2`  | `(B,)`       | twin-critic values                                                    |
| `r_env`   | `(B,)`       | behavioral env reward (goal / progress / collision / time)            |
| `r_risk`  | `(B,)`       | reward shaping term, `= -λ_risk * p_risk_1s`, computed at collection. Zero in the BASELINE config. |
| `r_total` | `(B,)`       | `r_env + r_risk`; used in the TD target                               |
| `mask`    | `(B,)`       | episode-not-done mask                                                 |

---

## 4. Module specs

### 4.1 `PointPillars` (from `module_pointpillar.py`)

- Config: default `PointPillarsConfig` (KITTI).
- Call site: `model.pillar_layer → pillar_encoder → backbone → neck`.
- Input: `list[Tensor(N_i, 4) float32]` of length `B`.
- Output: `bev_feat (B, 384, 248, 216) float32`.
- Parameters: ≈4.8 M (all frozen in A1, neck unfrozen in A2).

**Domain-gap note for PyBullet indoor:** point clouds generated from depth camera should be **scaled** so they fill more of `point_cloud_range`. Easiest: multiply all `(x, y, z)` by a fixed factor `s ≈ 5–10` before `extract_neck`. See `module_pointpillar.py::CameraToLidarExtrinsics.t` and adjust externally, or apply a `scale_points(pts, s)` helper in the dataset generator. This avoids retraining PointPillars at a new spatial scale.

### 4.2 `SpatialReducer`

```
Input:  (B, 384, 248, 216)

Conv2d(384, 256, k=3, s=2, p=1) + BN + ReLU         → (B, 256, 124, 108)
Conv2d(256, 256, k=3, s=2, p=1) + BN + ReLU         → (B, 256,  62,  54)
Conv2d(256, 256, k=3, s=2, p=1) + BN + ReLU         → (B, 256,  31,  27)
AdaptiveAvgPool2d((4, 4))                            → (B, 256,   4,   4)
flatten last two dims                                → (B, 16, 256)

Output: tok_grid (B, Nt=16, D=256)
```

- Parameters: ≈2.1 M.
- Why these specific sizes: 4×4 grid is a sweet spot — enough spatial resolution to distinguish front/left/right/back quadrants + inner/outer rings.

### 4.3 `Mamba`

- Library: `mamba-ssm` (`pip install mamba-ssm`) — requires CUDA.
- Config:
  - `d_model = 256` (matches SpatialReducer output)
  - `d_state = 16`
  - `expand = 2`
  - `n_blocks = 2`
- Input: `seq_tok (B, L, D)` where `L = T_ctx * Nt`.
  - Recommended `T_ctx = 10` (at 20 Hz → 0.5 s of history).
  - → `L = 10 * 16 = 160` tokens per batch element.
- Output: `h_seq (B, L, D)`; we take `h_T = h_seq[:, -1, :]` of shape `(B, D)`.
- Parameters: ≈1.5 M.
- Alternative fallback (if `mamba-ssm` fails to build): `nn.GRU(input_size=D, hidden_size=D, num_layers=2, batch_first=True)`. Slight quality drop but same API.
- **Ablations (Stage A only):** `models/temporal_encoders.py` adds `LSTMTemporal` (`nn.LSTM`) and `TransformerEncoderTemporal` (`nn.TransformerEncoder` + causal mask). `models/temporal_factory.build_temporal` selects `mamba` / `gru` / `lstm` / `transformer`. `FullPipeline` accepts any submodule with the same bulk `forward` contract `(B, L, D) → (B, L, D)` via its `mamba=` argument. `train_stage_a_compare.py` trains each backbone on the same scene split and logs TensorBoard + `summary.json`. **Streaming `step()`** for Stage B is implemented for Mamba / GRU / LSTM; the Transformer path is training / metrics only (no `step()`).

### 4.4 `RiskHead`

```
Input:  h_T  (B, 256)

Linear(256, 128) + ReLU + Dropout(0.1)
Linear(128,  64) + ReLU + Dropout(0.1)
Linear( 64,   3)                          # no sigmoid here — BCEWithLogitsLoss

Output: logits (B, 3)   → sigmoid(logits) = [p_05s, p_1s, p_2s]
```

- Parameters: ≈41 k.

### 4.5 `Actor` (Stage B) — plain MLP on proprio state

No convolutions. Consumes `s = ProprioState` concatenated into a flat vector.

```
Input:  s (B, d_s)                 # d_s ≈ 10–50

Linear(d_s, 256) + ReLU
Linear(256, 256) + ReLU
→ mu      Linear(256, A)
→ log_std Linear(256, A)           # clamped to [-20, 2]

action = tanh(mu + std * eps),  eps ~ N(0, I)
log_prob correction for tanh squashing (standard SAC)
```

- Parameters: ≈140 k (for `d_s = 30`, `A = 3`).
- The actor is called directly on `s` — there is NO encoder sitting between state and actor, hence no `detach()` dance.

### 4.6 `Critic` (Stage B, twin Q) — plain MLP

```
Input:  s (B, d_s) + a_t (B, A)   → concat → (B, d_s + A)

Linear(d_s + A, 256) + ReLU
Linear(256, 256)     + ReLU
Linear(256, 1)                     × 2  (Q1 and Q2 independent)
```

- Parameters: ≈80 k × 2 = 160 k.
- Both Q networks share the same structure; target copies via Polyak τ = 0.005.

> **No critic-only encoder update rule is needed in the default plan**, because there is no encoder on the SAC path. The DrQ-v2 convention only becomes relevant if you later opt into Stage B-plus and choose to feed a learned BEV feature into the actor — out of scope here.

---

## 5. Stage A — supervised pretrain

### 5.1 Dataset generation (PyBullet)

**Scripts:** `dataset_generator.py` (single entry).

**Configuration:**

```python
@dataclass
class DataGenConfig:
    n_scenes:             int   = 300     # procedural scenes
    rollouts_per_scene:   int   = 50
    frames_per_rollout:   int   = 400     # 20 s at 20 Hz (long enough for 2 s risk horizon at any t)
    dt:                   float = 0.05    # 20 Hz control & sensing frequency (Mamba T_ctx=10 ⇒ 0.5 s window)
    # Camera spec is optimized for obstacle-trajectory tracking (the task Mamba actually solves):
    #   - 4:3 aspect (160 × 120) gives enough lateral pixels to resolve obstacles at 5 m.
    #   - 90° horizontal FoV captures obstacles approaching from the side.
    #   - far = 8 m matches indoor dynamics; beyond 8 m a small fast obstacle would cross
    #     the whole FoV in a single T_ctx window anyway.
    depth_hw:             Tuple[int, int] = (160, 120)
    camera_fov_h_deg:     float = 90.0
    camera_near:          float = 0.1
    camera_far:           float = 8.0

    # policy mix (must sum to 1.0)
    policy_random_p:      float = 0.5
    policy_scripted_p:    float = 0.3
    policy_adversarial_p: float = 0.2    # intentionally drive toward obstacle
    policy_stationary_p:  float = 0.0    # v=w=0; dynamic obstacles can still hit (see strategy_create_trajectory_label.md §9)

    # domain randomization
    depth_noise_std:      float = 0.01   # additive m
    drop_pixel_prob:      float = 0.02
    camera_jitter_deg:    float = 1.0
    obstacle_texture_rand:bool  = True
    lighting_rand:        bool  = True

    out_dir:              str   = "dataset/pybullet_risk_v1"
    seed:                 int   = 0
```

**Scene content:**
- Floor + 4 walls.
- 5–15 static obstacles (boxes, cylinders) placed with rejection sampling (no overlap).
- 1–3 dynamic obstacles in 30 % of scenes (simple linear motion). Optional in v1.

**Policy types:**
- `random`: continuous action sampled uniformly from `[-1, 1]^A`, smoothed with OU noise.
- `scripted`: go-to-random-waypoint with PD controller.
- `adversarial`: pick an obstacle, go-to-obstacle; forced to give **positive collision samples** → balances labels.

**Risk label derivation (offline pass over each rollout):**

```
for t in range(T):
    risk_05s[t] = any(contact_flag[t : t+10])   # 0.5 s = 10 frames at 20 Hz
    risk_1s[t]  = any(contact_flag[t : t+20])
    risk_2s[t]  = any(contact_flag[t : t+40])
```

> **Expanded narrative** (contact detection, `lookahead_any` edge cases, file pointers): see `docs/strategy_create_trajectory_label.md`.

**Expected dataset size:**
- 300 × 50 × 400 = 6 M frames total.
- After filtering the last 2 s (40 frames) of each rollout (no valid 2 s horizon): ≈ 5.4 M usable frames.
- Disk size: depth float16 (160 × 120) ≈ 38 kB/frame → ≈ 210 GB full. Recommended: train on a **1.5 M-frame subset** (75 GB) until convergence; use the full set only for final eval. Keep on local SSD or Colab Pro+ `/content/` (mount a persistent disk).
- Trade-off knob: if disk is tight, drop `frames_per_rollout` to `200` (10 s episodes) → ≈ 105 GB full, ≈ 2.4 M usable.

### 5.2 Training loop

**Library stack:**
- `torch >= 2.1`
- `mamba-ssm >= 1.2` (requires CUDA)
- `numpy`, `tqdm`, `tensorboard`
- Our own `pointpillars` (stripped) + `module_pointpillar.py`

**File layout (expected):**

```
PointPillars_module/
├── dataset_generator.py         # NEW, offline PyBullet rollout
├── risk_dataset.py              # NEW, torch Dataset + collate_fn
├── models/
│   ├── spatial_reducer.py       # NEW
│   ├── mamba_temporal.py        # NEW
│   ├── temporal_encoders.py     # NEW, LSTM + causal Transformer (ablations)
│   ├── temporal_factory.py      # NEW, build_temporal(kind)
│   ├── risk_head.py             # NEW
│   └── full_pipeline.py         # NEW, nn.Module that wires everything
├── train_stage_a_compare.py     # NEW, matched-split compare + TensorBoard
├── train_stage_a.py             # NEW
├── train_stage_b_sac.py         # NEW
├── utils/
│   └── metrics.py               # NEW, AUC / Brier / calibration
└── (existing module_pointpillar.py, pointpillars/, setup.py, ...)
```

**Pseudo-training step (Stage A):**

```python
for batch in loader:                         # batch: RiskBatch
    bev_list = []
    for t in range(T_ctx):
        feat_t = pp.extract_neck_forward(batch.pts_seq[t]).feature   # (B, 384, H, W)
        bev_list.append(feat_t)
    tok_list = [reducer(f) for f in bev_list]                        # each (B, Nt, D)
    seq = torch.stack(tok_list, dim=1).flatten(1, 2)                 # (B, T_ctx*Nt, D)
    h_seq = mamba(seq)
    h_T   = h_seq[:, -1, :]                                          # (B, D)
    logits = risk_head(h_T)                                          # (B, 3)

    loss = focal_bce(logits, torch.stack([batch.risk_05s,
                                          batch.risk_1s,
                                          batch.risk_2s], dim=-1))
    loss.backward()
    opt.step(); opt.zero_grad()
```

### 5.3 Loss

```python
def focal_bce(logits, targets, gamma=2.0, weight=(1.0, 0.8, 0.5)):
    # logits, targets: (B, 3)
    p  = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt  = targets * p + (1 - targets) * (1 - p)
    loss = ((1 - pt) ** gamma) * bce                 # (B, 3)
    w = torch.tensor(weight, device=logits.device)   # (3,)
    return (loss * w).mean()
```

### 5.4 Sub-stages

**A1 — warm-up (5 epochs):**
- PointPillars: **frozen** (load KITTI weights via `module_pointpillar.PointPillarsNeckExtractor`).
- Trainable: `SpatialReducer`, `Mamba`, `RiskHead`.
- Optimizer: AdamW, `lr = 3e-4`, `weight_decay = 1e-4`.
- Scheduler: cosine, warmup 500 iter.
- Batch: 32 (T4) / 64 (A100).

**A2 — neck unfreeze (5 epochs):**
- PointPillars: unfreeze `neck` only. Still freeze `pillar_layer`, `pillar_encoder`, `backbone` (they contain voxelization and the most transferable spatial filters).
- Two param groups:
  - `g1`: neck params → `lr = 3e-5`.
  - `g2`: SpatialReducer + Mamba + RiskHead → `lr = 1e-4`.
- Don't forget `neck.eval()` **only if** you want BatchNorm frozen — in A2 we want BN to adapt, so keep `neck.train()`.

### 5.5 Metrics (log every 500 iter)

- Per-horizon: `AUC-ROC`, `AUC-PR`, `Brier score`.
- Calibration: reliability diagram (10 bins).
- Confusion matrix at threshold 0.5.
- Stop criterion: AUC-ROC on `risk_1s` reaches ≥ 0.85 on val split.

### 5.6 Data split

- 80 / 10 / 10 by **scene_id** (not by frame). Splitting by frame leaks information.
- Stratify by positive-sample-frequency to keep rare scenes in all splits.

---

## 6. Stage B — SAC fine-tune

### 6.1 Weight transfer map (default: fully frozen perception stream)

| From Stage A checkpoint         | Into Stage B                                  | Frozen in Stage B?             |
|---------------------------------|-----------------------------------------------|--------------------------------|
| `pointpillars.*`                | `pp.*`                                        | **Yes** (`.eval()`, `no_grad`) |
| `spatial_reducer.*`             | `spatial_reducer.*`                           | **Yes**                        |
| `mamba.*`                       | `mamba.*`                                     | **Yes**                        |
| `risk_head.*`                   | `risk_head.*`                                 | **Yes**                        |
| —                               | `actor.*` (new, random init)                  | No (trainable)                 |
| —                               | `critic_q1.*, critic_q2.*` (new, random init) | No (trainable)                 |
| —                               | `log_alpha` (scalar)                          | No (trainable)                 |

Rationale: see § 2.6. The perception stream is a closed-form dense reward; it is not touched by SAC gradients.

### 6.2 Optimizer

Minimal — 3 param groups + a separate alpha optimizer:

```python
optim = torch.optim.AdamW([
    {"params": actor.parameters(),     "lr": 3e-4, "weight_decay": 0.0},
    {"params": critic_q1.parameters(), "lr": 3e-4, "weight_decay": 0.0},
    {"params": critic_q2.parameters(), "lr": 3e-4, "weight_decay": 0.0},
])
alpha_opt = torch.optim.Adam([log_alpha], lr=3e-4)
```

All perception-stream parameters are `requires_grad_(False)` and locked in `.eval()` mode via the hook in `strategy_finetune_with_SAC.md` § 4.2.

### 6.3 Loss (default plan)

```
L_total = L_actor + L_critic + L_alpha
```

- `L_critic`: twin-Q TD error against target `y = r_total + γ * (1 − done) * (min(Q1', Q2') − α * log π')`, with `r_total = r_env + r_risk` read directly from the buffer.
- `L_actor`: `E[α · log π(a|s) − Q_min(s, a)]` with `a` sampled via reparameterization.
- `L_alpha`: temperature update targeting entropy `-|A|`.

Suggested `λ_risk = 2.0`. Re-tune by monitoring `E[r_risk] / E[r_env]` online; target band `[-0.4, 0.0]`. Because `r_risk` is pre-computed and stored separately in the buffer (§ 6.4), you can re-weight `λ_risk` offline without re-rolling out episodes — just recompute `r_total = r_env + (new_λ / old_λ) · r_risk` at sample time.

### 6.3.1 A/B-test configurations

| Config   | Reward used in TD target    | Description                                        |
|----------|-----------------------------|----------------------------------------------------|
| BASELINE | `r_env`                     | No perception-stream involvement. Everything else identical. |
| PROPOSED | `r_env + r_risk`            | Risk predictor feeds reward shaping at every step. |

Everything else (Actor, Critic, hyperparameters, environment, seeds) is matched. Report the gap on collision rate, episode return, and sample efficiency in dynamic-obstacle scenes.

### 6.4 Replay buffer (minimal, proprio-based)

Because Actor/Critic consume a **proprioceptive state vector** (not perception features), and because the perception stream is frozen, the buffer is tiny: no `pts`, no `bev_feat`, no `z`. We only store proprio states, actions, reward decomposition, and metadata.

```python
@dataclass
class Transition:
    # Proprioceptive state vector (see ProprioState in § 2.5)
    s:          np.ndarray      # (d_s,) float32, d_s ≈ 10–50
    s_next:     np.ndarray      # (d_s,) float32

    # Control
    action:     np.ndarray      # (A,) float32

    # Reward decomposition — kept separate to allow offline λ_risk re-weighting
    # and per-term diagnostics without re-rolling out.
    r_env:      np.float32
    r_risk:     np.float32      # = -λ_risk * p_risk_1s at collection time; 0 in BASELINE

    # Episode control
    done:       bool

    # Meta
    episode_id: int
    frame_idx:  int
```

Per-transition size: `2 × (d_s · 4 B) + A · 4 B + ~20 B ≈ 300–500 B` (for `d_s = 30`, `A = 3`). **Buffer 1 M × 500 B ≈ 500 MB** on CPU RAM. Increase to 2 M transitions if memory allows.

**What is NOT in the buffer and why:**

| Item               | Why it's NOT stored                                                 |
|--------------------|---------------------------------------------------------------------|
| `pts`, `depth`     | Actor/Critic don't consume them. `r_risk` is already computed at collection time. |
| `bev_feat`, `tok`, `h_t`, `p_risk` | Risk branch is frozen and streaming — intermediate tensors serve only to produce `r_risk`, which is cached. |
| `z` / encoded state | No learned encoder on the SAC path; `s` itself is the state.        |
| `risk_gt` labels   | Not needed in the default plan (no aux loss). Only Stage B-plus uses them. |

#### 6.4.1 Flow summary (collection → storage → gradient)

```
Env step t (rollout, 20 Hz):
    # ---- Perception stream (frozen, per-env streaming Mamba hidden) ----
    pts_t = preprocess(depth_t)                      # voxel-downsampled
    with torch.no_grad():
        bev_t    = pp(pts_t)                          # frozen
        tok_t    = spatial_reducer(bev_t)             # frozen
        h_t      = mamba.step(tok_t, hidden=h_{t-1})  # frozen, streaming
        p_risk_t = risk_head(h_t)                     # (3,)
    r_risk_t = -lambda_risk * p_risk_t[1]            # horizon = 1 s; or 0 in BASELINE

    # ---- Control stream (SAC on proprio) ----
    s_t = read_proprio(env)                           # (d_s,)
    with torch.no_grad():
        a_t = actor.sample(s_t)
    (s_next, r_env_t, done_t, info) = env.step(a_t)

    buffer.push(Transition(s=s_t, s_next=s_next,
                           action=a_t, r_env=r_env_t, r_risk=r_risk_t,
                           done=done_t, ...))

Gradient step (SAC update):
    batch = buffer.sample(B)                          # all numpy/tensor, tiny
    s, s_next = batch.s, batch.s_next
    r_total = batch.r_env + batch.r_risk

    # Critic
    q1, q2 = critic_q1(s, batch.action), critic_q2(s, batch.action)
    with torch.no_grad():
        a_next, logp_next = actor.sample(s_next)
        q_target = torch.min(target_q1(s_next, a_next),
                             target_q2(s_next, a_next))
        y = r_total + gamma * (1 - batch.done) * (q_target - alpha * logp_next)
    L_critic = F.mse_loss(q1, y) + F.mse_loss(q2, y)
    L_critic.backward()

    # Actor
    a_new, logp = actor.sample(s)
    q_min = torch.min(critic_q1(s, a_new), critic_q2(s, a_new))
    L_actor = (alpha * logp - q_min).mean()
    L_actor.backward()
```

Gradient-step cost with `B = 256`: ~2 ms forward + ~2 ms backward on a 3060 (pure MLPs). Perception-stream forward happens ONLY at rollout time (not in the gradient loop), so gradient steps are dominated by trivial MLP ops.

#### 6.4.2 Mamba streaming across env steps

Streaming is a per-environment concern: each parallel env maintains its own Mamba hidden state across timesteps, reset on `done`.

```python
class MambaStreamer:
    def __init__(self, n_envs):
        self.h = [None] * n_envs          # hidden state per env
    def step(self, env_idx, tok_t):
        h = self.h[env_idx]
        h_new = mamba.step(tok_t, hidden=h)   # O(1) per token, see Mamba paper
        self.h[env_idx] = h_new
        return h_new
    def reset(self, env_idx):
        self.h[env_idx] = None
```

Hidden state shape: `(D=256,)` per env; negligible memory. Typical per-env-step inference cost (PP + SR + Mamba.step + RiskHead): **~10 ms** on a 3060, **~4 ms** on an A100 — comfortably inside a 50 ms (20 Hz) budget.

### 6.5 BatchNorm in RL (trivial here)

- **Frozen modules (`pp.*`, `spatial_reducer.*`, `mamba.*`, `risk_head.*`):** call `.eval()` once at the start of Stage B. Register a hook that forces those submodules back to `.eval()` at every `self.train()` call (see `strategy_finetune_with_SAC.md` § 4.2 for the exact hook).
- **Actor / Critic:** no BN (pure MLP). Use `LayerNorm` inside if stability is an issue — BN is not beneficial on a 30-dim proprio vector anyway.

There is no trainable encoder in the SAC path, so there are no non-stationarity issues with running statistics.

### 6.6 Environment / reward

```python
@dataclass
class EnvConfig:
    # Timing — must match dataset generation for the frozen Mamba to see familiar dynamics
    dt:                 float = 0.05   # 20 Hz control and perception loop
    max_episode_steps:  int   = 400    # 20 s
    T_ctx:              int   = 10     # Mamba history window (0.5 s)

    # Camera — identical to DataGenConfig (§5.1) so the frozen risk branch is in-distribution
    depth_hw:           Tuple[int, int] = (160, 120)
    camera_fov_h_deg:   float = 90.0
    camera_near:        float = 0.1
    camera_far:         float = 8.0

    # Behavioral reward coefficients (r_env)
    goal_radius:        float = 0.3
    w_goal:             float = 5.0
    w_progress:         float = 1.0
    w_collision:        float = 20.0
    w_time:             float = 0.01
    w_action_norm:      float = 0.001

    # Reward shaping (r_risk)
    lambda_risk:        float = 2.0    # applied to p_risk_1s at collection time; set to 0.0 for BASELINE

    # Action space (example for wheeled / quadruped base)
    action_dim:         int   = 3      # (v_x, v_y, ω_yaw) in body frame
    action_bounds:      Tuple[float, float] = (-1.0, 1.0)  # after tanh squash

    # Proprioceptive state spec (see ProprioState in §2.5). Populated by env.read_proprio()
    proprio_dim:        int   = 10     # base_lin_vel(3) + base_ang_vel(3) + goal_rel(3) + heading_err(1)
    include_joint_state: bool = False  # set True to extend proprio_dim by 2*dof
    include_last_action: bool = True   # appends (A,) to the proprio vector; recommended for smoothness
```

Reward decomposition:

```
r_env_t  =  w_goal * 1[reached_goal]
         + w_progress * (dist_{t-1} - dist_t)
         - w_collision * 1[collision]
         - w_time
         - w_action_norm * ||a_t||^2

r_risk_t = -lambda_risk * p_risk_1s_t        # from frozen risk branch

r_total_t = r_env_t + r_risk_t               # what SAC learns on
```

### 6.7 Stopping criterion

- Episode return (`r_total` sum) sustained over 100 episodes ≥ 50.
- Collision rate (fraction of episodes with any contact) < 5 %.
- `E[r_risk] / E[r_env]` stays in the `[-0.4, +0.0]` band — if `r_risk` becomes negligible (→ 0), the risk branch is out-of-distribution; fall back to § 6.8.

### 6.8 (Optional) Stage B-plus: unfreeze perception stream with aux BCE

Use this only **after** the default A/B test has been run and you've decided the frozen risk predictor is miscalibrated on the SAC distribution (e.g. Brier score drifts > 0.2, or `p_risk` histogram is stuck at one value).

| Change vs default | Stage B-plus                                                                 |
|-------------------|------------------------------------------------------------------------------|
| Perception stream | Unfrozen with low LR (`spatial_reducer`, `mamba`: 5e-5; `risk_head`: 1e-4; `pp.neck`: 1e-5). PP backbone stays frozen. |
| Extra loss        | `L_total += lambda_aux * focal_bce(logits, risk_gt)` where labels come from PyBullet contact lookahead (same rule as Stage A § 5.1). |
| Buffer changes    | Add `pts_window: List[Tensor]` (T_ctx frames) + `risk_gt: (3,) float32` for a subset (~20 %) of transitions. Grows buffer to ~1.5 GB. |
| λ_aux schedule    | `1.0 → 0.3 → 0.1 → 0.05` at steps `0 → 100 k → 500 k → 1 M`. Never zero.    |
| Actor / Critic    | **Unchanged** — still plain MLP on proprio state. This recipe never feeds BEV into SAC; it only refreshes the risk branch. |
| New risks         | BN drift in Mamba; resample rate-limited by CPU I/O for `pts_window`.        |

This recipe is strictly a follow-up experiment. It changes the perception stream, not the SAC path.

---

## 7. Environment / libraries / versions

Recommended stack (Colab Pro+, CUDA 12.1):

| Package          | Version          | Notes                                   |
|------------------|------------------|-----------------------------------------|
| python           | 3.10 or 3.11     | mamba-ssm wheels exist                  |
| torch            | 2.3.1+cu121      | matches nvcc 12.1                       |
| numpy            | 1.26.x           |                                         |
| ninja            | ≥ 1.11           | builds voxel_op                         |
| mamba-ssm        | 1.2.x            | requires CUDA; pip install mamba-ssm    |
| causal-conv1d    | 1.2.x            | dependency of mamba-ssm                 |
| pybullet         | 3.2.6            | physics sim                             |
| tensorboard      | ≥ 2.14           | logging                                 |
| tqdm             | latest           | progress bars                           |
| scikit-learn     | latest           | AUC / calibration metrics               |

**If mamba-ssm does not build:** fallback to `nn.GRU(D, D, 2, batch_first=True)` in `mamba_temporal.py`. Same interface.

---

## 8. Risk register

| ID  | Risk                                                      | Likelihood | Impact | Mitigation                                                                                           |
|-----|-----------------------------------------------------------|-----------:|-------:|------------------------------------------------------------------------------------------------------|
| R1  | Distribution shift Stage A → Stage B                      | High       | High   | Diverse policies in dataset gen (random + scripted + adversarial). Monitor Brier score of frozen risk branch online; if it degrades, switch to Stage B-plus (§6.8). |
| R2  | Catastrophic forgetting of risk knowledge during SAC      | *Eliminated in default plan* | — | Entire perception stream is frozen (§6.1). Only relevant if you opt into Stage B-plus (§6.8). |
| R3  | Class imbalance in risk labels (≈ 95 % negatives)         | High (Stage A only) | High | Oversample near-collision frames × 10; focal loss gamma=2; adversarial policy for more positives. Not relevant in default Stage B (no label-based loss). |
| R4  | PointPillars domain gap (KITTI outdoor vs PyBullet indoor) | Medium     | Medium | Scale-up hack (multiply coords by ~5) to fill `point_cloud_range`. See `module_pointpillar.md` §7 and §4.1 of this doc. Unfreeze neck in A2 if Stage A AUC plateaus. |
| R5  | mamba-ssm build fails on Colab                            | Medium     | Low    | GRU fallback already in design (§4.3).                                                               |
| R6  | Reward misaligned: λ_risk too high → policy freezes       | Medium     | Medium | Start `λ_risk = 2.0`; log `E[r_risk] / E[r_env]`; target ratio ≲ 0.3; reduce λ_risk if policy never moves. |
| R7  | Buffer size                                               | *Trivially small* | — | Proprio-only buffer at ~500 B / transition. 1 M transitions ≈ 500 MB. Not a concern in the default plan. |
| R8  | Stage A overfits on limited scenes                        | Low        | Medium | 300+ scenes, strong domain randomization; monitor val AUC per scene bucket.                          |
| R9  | NaN / divergence during A2 unfreeze                       | Medium     | Medium | Low LR on neck (3e-5); gradient clipping `max_norm=1.0`; check grads with `torch.nan_to_num` during first 500 iter. |
| R10 | Frozen risk branch is systematically wrong (e.g. unseen obstacle shapes) | Low | Medium | Dashboard the histogram of `p_risk_1s` across episodes; if it's bimodal-stuck (always 0 or always 1), switch to Stage B-plus. |
| R11 | Sim-to-real gap (future)                                  | —          | —      | Out of scope for this doc; depth noise + camera jitter in dataset gen reduce it.                     |

---

## 9. Debugging guide

| Symptom                                                              | Likely cause                                 | First check                                                  |
|----------------------------------------------------------------------|----------------------------------------------|--------------------------------------------------------------|
| Risk loss stuck at ~0.33 (= BCE at 0.5 for uniform)                  | class imbalance collapse                     | Per-class positive ratio in batches; enable oversampling.    |
| BEV feature map almost all zeros                                     | points outside `point_cloud_range`           | Print `pts.shape` before and after `filter_range`.           |
| Mamba hidden norm explodes (> 1e3)                                   | no warmup / bad init                         | Add warmup 500 iter; grad clip 1.0; lower LR to 1e-4.        |
| Risk AUC drops during Stage B (only meaningful in Stage B-plus)      | aux lambda too low / encoder drifts          | Raise `lambda_aux`; lower encoder LR. In default plan this is impossible — risk branch is frozen. |
| SAC reward plateaus near 0 in BASELINE                               | proprio state too impoverished               | Add joint state and `last_action` to `ProprioState`; re-run A/B. |
| SAC reward plateaus in PROPOSED but not BASELINE                     | `λ_risk` too high — policy freezes to avoid risk | Lower `λ_risk`; check `E[r_risk] / E[r_env]` stays in [-0.4, 0]. |
| Policy "freezes" (stands still to avoid risk)                        | `λ_risk` too high relative to `w_progress`   | Lower `λ_risk`; inspect `E[r_risk] / E[r_env]`.              |
| `p_risk` is always ≈ 0.5 (no signal)                                 | frozen risk branch out-of-distribution       | Enable Stage B-plus (§6.8) or re-run Stage A with more diverse scenes. |
| CUDA OOM during Stage A                                              | batch too large for T_ctx                    | Reduce batch to 16; use `torch.cuda.amp`.                    |
| `mamba-ssm` ImportError                                              | CUDA/nvcc mismatch                           | Rebuild with matching CUDA; fall back to GRU.                |
| Replay sampling very slow                                            | misconfigured worker threads                 | Buffer is plain numpy arrays — should be O(1). Check buffer wasn't accidentally made to store `pts`. |
| Mamba streaming hidden state drifts to NaN                           | `done` not propagated to MambaStreamer.reset | Register episode-end callback that calls `streamer.reset(env_idx)`. |

---

## 10. Roadmap / milestones

| # | Milestone                                         | Deliverable                                   | Est. time |
|---|---------------------------------------------------|-----------------------------------------------|----------:|
| 1 | PyBullet env + scene generator                    | `env.py`, `scene_builder.py`                  | 2–3 days  |
| 2 | `dataset_generator.py` producing 100 rollouts     | `dataset/pybullet_risk_v1/*.npz` sample       | 2 days    |
| 3 | `risk_dataset.py` + `full_pipeline.py` skeletons  | forward pass unit test                         | 1 day     |
| 4 | Stage A1 training (frozen PP)                     | checkpoint + TensorBoard AUC curves           | 1–2 days  |
| 5 | Stage A2 training (neck unfrozen)                 | improved checkpoint                           | 1–2 days  |
| 6 | Full dataset (2–3 M frames)                       | final Stage A dataset                         | 1 day     |
| 7 | Stage A final train on full dataset               | `stage_a_final.pt`                            | 1–2 days  |
| 8 | SAC env + `train_stage_b_sac.py`                  | initial SAC loop                              | 2–3 days  |
| 9 | Stage B fine-tune                                 | `stage_b_final.pt`                            | 2–3 days  |
|10 | Eval + report                                     | `report.md`, video rollouts                   | 1 day     |

**Total:** ≈ 3–4 weeks wall-clock including debugging.

---

## 11. Implementation checklist

**Dataset**
- [ ] Scene generator with ≥ 300 unique layouts.
- [ ] Policy mix (random / scripted / adversarial) with correct probabilities.
- [ ] Depth noise, camera jitter, texture and lighting randomization.
- [ ] Risk labels derived from `contact_flag` with 3 horizons.
- [ ] Disk layout: one `.npz` per rollout, under `dataset/pybullet_risk_v1/`.
- [ ] Meta file `index.jsonl` with `scene_id`, `rollout_id`, `policy_type`, positive ratios.

**Model**
- [ ] `SpatialReducer` with the exact Conv→Pool spec in §4.2.
- [ ] `Mamba` (or GRU fallback) with `d_model=256`, `n_blocks=2`.
- [ ] `RiskHead` with 3 output logits (no sigmoid inside).
- [ ] `FullPipeline.forward(pts_seq) -> logits` for Stage A.
- [ ] Stage B-runtime method `FullPipeline.step(pts_t, hidden_prev) -> (p_risk_t, hidden_t)` for streaming Mamba — used by the env loop. (No `z` anywhere; SAC consumes proprio state directly from the env.)
- [ ] `Actor(d_s → A)` and twin `Critic(d_s + A → 1)` as plain MLPs (§4.5 / §4.6).

**Stage A**
- [ ] A1: freeze PointPillars; train reducer+mamba+head; 5 epochs.
- [ ] A2: unfreeze neck; two-group optimizer; 5 epochs.
- [ ] Focal-BCE with gamma=2, weights (1.0, 0.8, 0.5).
- [ ] Oversample positives × 10.
- [ ] Log AUC-ROC/PR/Brier every 500 iter.
- [ ] Checkpoint every epoch; keep best-val.

**Stage B (default plan — fully frozen perception stream, A/B test)**
- [ ] Load Stage A weights into `pp.*`, `spatial_reducer.*`, `mamba.*`, `risk_head.*`.
- [ ] `requires_grad_(False)` on every perception-stream parameter; `.eval()` mode locked via a hook.
- [ ] Initialize `actor`, `critic_q1`, `critic_q2`, `log_alpha` from scratch.
- [ ] Optimizer has exactly 3 param groups (§6.2). Separate `alpha_opt` for `log_alpha`.
- [ ] No aux loss. No `lambda_aux`.
- [ ] Replay buffer stores `s`, `s_next`, `action`, `r_env`, `r_risk`, `done` only (§6.4). No `pts`, no `z`.
- [ ] Streaming Mamba hidden state kept per env and reset on `done` (`MambaStreamer` in §6.4.2).
- [ ] Run both configurations with identical seeds: `BASELINE` (`λ_risk = 0`) and `PROPOSED` (`λ_risk = 2.0`).
- [ ] Log `E[r_env]`, `E[r_risk]`, `E[r_risk] / E[r_env]`, collision rate, episode return every 10 k env steps.
- [ ] BN: frozen modules forced to `.eval()` via hook. Actor / Critic are pure MLPs with no BN.
- [ ] Evaluate every 10 k env steps on a fixed eval scene set disjoint from Stage A train/val.

**Stage B-plus (optional, §6.8) — only if default plan fails**
- [ ] Unfreeze `mamba`, `risk_head`, `spatial_reducer`, `pp.neck` with per-group LRs in §6.8.
- [ ] Augment buffer with `pts_window` + `risk_gt` for 20 % of transitions.
- [ ] Add `lambda_aux * focal_bce(logits, risk_gt)` to total loss with schedule.
- [ ] Monitor `risk_1s` AUC; must stay ≥ 0.75 throughout.

**Evaluation**
- [ ] Test set of held-out PyBullet scenes not seen in Stage A or B.
- [ ] Report: episode return, collision rate, mean steps to goal, risk AUC under current encoder.
- [ ] Optional: 20 s video rollouts for 5 scenes.

---

## 12. Glossary

| Term              | Meaning                                                                                  |
|-------------------|------------------------------------------------------------------------------------------|
| BEV               | Bird's-eye view — 2D top-down feature map produced by PointPillars neck.                 |
| Pillar            | A vertical voxel of infinite/large Z extent used by PointPillars.                        |
| `T_ctx`           | Number of past frames fed into Mamba per forward pass.                                   |
| Risk horizon      | The future time window used to label `risk_*`.                                           |
| Aux loss          | Auxiliary supervised loss kept in Stage B to anchor pretrained knowledge.                |
| Multi-param-group | PyTorch optimizer pattern assigning different learning rates to different sub-networks.  |
| `λ_risk`          | Reward-shaping coefficient turning `p_risk_1s` into a penalty in `r_total`. Default 2.0. |
| `λ_aux`           | Auxiliary loss coefficient; used only in Stage B-plus (§6.8).                            |
| SSM               | State-space model (the architectural family that includes Mamba).                        |
| PER               | Prioritized experience replay (optional).                                                |
| Perception stream | `PointPillars → SpatialReducer → Mamba → RiskHead → p_risk`. Produces the reward-shaping scalar `r_risk`. Frozen in the default Stage B. |
| Control stream    | `ProprioState → Actor / Critic (MLP)`. Produces `a_t`. Fully decoupled from perception — they only meet at the reward. |
| A/B test          | Compare BASELINE (`r = r_env`) vs PROPOSED (`r = r_env + r_risk`) with identical everything else. |

---

## 13. Relationship to other docs

| Doc                              | Owns (authoritative)                                                                                  | Cross-ref from this doc                             |
|----------------------------------|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| `module_pointpillar.md`          | The `PointPillarsNeckExtractor` API + depth-camera preprocessing helpers.                             | §4.1 indoor-scale hack, §2.1 frozen-PP assumption.  |
| `strategy_finetune_with_SAC.md`  | SAC-specific implementation details (Actor/Critic MLP on proprio, BN-lock hook for frozen perception, Polyak averaging, SAC hyperparameters, freeze regimes S1/S2/S3). | §4.5/4.6 actor & critic specs, §6.5 BN hook.        |
| **This doc** (`strategy_full_pipeline.md`) | End-to-end two-stage pipeline architecture + data contracts + Stage A + Stage B orchestration. | —                                                   |
| `create_dataset_module/` (code)  | Offline Stage-A dataset generation (`DatasetEnv`, policies, `DataGenerator`, `RiskDataset`, `collate_riskbatch`). Camera/env defaults mirror §5.1; risk-horizon lookahead mirrors the formula in §5.1. | §3.1 (`Trajectory`), §3.3 (`RiskBatch`), §5.1 (dataset recipe), §5.6 (scene-stratified split). |
| `PointPillars_module/models/` (code) | Stage-A perception-stream submodules: `SpatialReducer` (§4.2), `MambaTemporal` (§4.3, with `mamba-ssm` primary + `nn.GRU` fallback), `RiskHead` (§4.4), `FullPipeline` wrapper, and `losses.focal_bce` (§5.3). | §4.2–§4.4 module specs, §5.3 loss, §6.4.2 streaming step. |
| `strategy_train_stage_A.md` | **Stage A consolidated spec** — single-reader view of data pipeline (`Trajectory` → `RiskBatch`), `FullPipeline` shapes, focal-BCE + A1/A2, lookahead labels, and Stage B `r_risk` handoff. Does **not** override this doc on conflicts. | §5 (cross-check), §13 this table. |
| `strategy_experiment_protocol.md` | **Downscaled lab recipe** — small `DataGen` preset (`run_datagen_preset.py experiment`), short Stage A schedules, and **unified metrics** for comparing methods across Stage A (offline) and Stage B (online). | §5 scale-down; §6.3 A/B metrics; §13 this table. |
| `optimized_training_strategy_stage_A.md` | **Compute-efficient Stage A** — divide & conquer (BEV **feature caching**, decoupled / frozen `PointPillarsNeckExtractor`), ablation table templates, robustness scenarios; **proposed** `cache_features.py` / `RiskDataset` BEV mode until coded. | §5 training; §13; cross-check `strategy_train_stage_A.md`. |
| `skill_avoid_gradient_boom.md` | **Gradient / numerics playbook** — global clipping (incl. separate Actor vs Critic in SAC), AMP + `GradScaler`, init (He / small Actor head), normalization, Polyak \(\tau\), reward scaling, entropy \(\alpha\); pre-train checklists for Stage A & B. | §5–§6 training hygiene; mirror `strategy_finetune_with_SAC.md` §8.3; code: `utils/gradient_health.py`, `train_stage_b_sac.py`. |

Conflict rule: if any two docs disagree on the same topic, the row owner wins and the other doc must be updated to match.

---

## 14. Maintenance rule — keep these docs in sync

This doc is the architectural source of truth for the three-doc set (`module_pointpillar.md`, `strategy_full_pipeline.md`, `strategy_finetune_with_SAC.md`). Any change to code OR ideas MUST keep all three coherent, otherwise the next agent will act on stale context.

### 14.1 Trigger matrix — when to update what

| You are changing…                                                             | Must update                                                                                          |
|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Pipeline architecture (new module, branch added/removed, reward path)         | This doc (owner) first, then propagate summaries + cross-refs to the other two.                      |
| Stage A recipe (dataset, labels, loss, freeze schedule, hyperparameters)      | This doc § 5; mirror narrative in `strategy_train_stage_A.md` (same PR when §5 changes materially). **Experiment-scale overrides:** `strategy_experiment_protocol.md`. **Optimization / ablations:** `optimized_training_strategy_stage_A.md`. |
| Stage B recipe (weight transfer, optimizer groups, reward decomposition)      | This doc § 6 (owner) + mirror hyperparameters in `strategy_finetune_with_SAC.md` § 12.               |
| Replay-buffer contents                                                        | This doc § 6.4 (owner) + cross-ref in `strategy_finetune_with_SAC.md` § 6.3.                         |
| Camera / env spec (resolution, FoV, `dt`, episode length, action space)       | This doc § 5.1 (`DataGenConfig`) + § 6.6 (`EnvConfig`) (owner) + mirror in `strategy_finetune_with_SAC.md` § 10. |
| Reward formula (`r_env`, `r_risk`, `λ_risk`) or A/B framing                   | This doc § 6.3 / § 6.3.1 / § 6.6 (owner) + mirror in `strategy_finetune_with_SAC.md` § 11.           |
| Mamba streaming behavior or `T_ctx`                                           | This doc § 6.4.2 (owner).                                                                            |
| Data contracts (tensor shapes, dataclasses, dtypes)                           | This doc § 3 (owner) + perception tensors spec in `module_pointpillar.md`.                           |
| Perception module API (functions, dataclasses in `module_pointpillar.py`)     | `module_pointpillar.md` (owner) + update cross-ref in § 4.1 of this doc.                             |
| SAC internals (Actor/Critic layers, BN-lock hook, freeze regimes S1/S2/S3)    | `strategy_finetune_with_SAC.md` (owner) + update cross-ref in § 6.1 / § 6.5 of this doc.             |
| Dependency list                                                               | `module_pointpillar.md` § 2.1 + this doc § 7.                                                        |

### 14.2 Per-PR checklist

Any PR that touches any of the three docs or the perception / RL code must satisfy:

1. **Ownership respected.** Changes are made in the owning doc first (per § 13 table); other docs only receive cross-references or mirrored summaries.
2. **Changelog row added** in every doc actually modified. One row per doc, with date, author, and a 1–3 line summary.
3. **Version bump** if the change breaks a downstream consumer (removed field in a dataclass, changed tensor shape, changed default hyperparameter that affects reported results). Use `vX.Y` where X = breaking, Y = additive.
4. **Grep pass for stale terms.** If you rename / remove a concept, `rg` the repo for the old name and update every hit — diagrams, tables, checklists, and pseudocode blocks all count.
5. **Cross-reference audit.** Every `§ X.Y` pointer touched in the change must still resolve. If you renumber a section in this doc, check the other two docs for stale `§ N.M` references.
6. **Diagram parity.** The block diagrams in § 1.2 and in `strategy_finetune_with_SAC.md` § 2 must be consistent. If you update one, update the other in the same PR.
7. **Code ↔ doc parity.** If you changed code, the relevant doc section must match the new code verbatim for signatures and defaults. If you changed a doc first (idea stage), add a `> **Proposed — not yet implemented.**` banner to the affected section until code catches up.

### 14.3 Idea-only changes (no code yet)

When updating docs as a design exercise (before touching code):

- Mark the affected sections with `> **Proposed — not yet implemented.**` at the top.
- Add an entry to a `## Pending sync` subsection of the changelog table listing the code locations that still need updating.
- Remove the banner and the Pending-sync row **in the same PR that lands the code**.

### 14.4 Conflict resolution

If two docs disagree after a change:

1. The row owner in § 13 wins.
2. If the change crosses ownership boundaries or cannot be attributed cleanly, **this doc (`strategy_full_pipeline.md`) is the tiebreaker** because it owns the architectural source of truth.
3. Never resolve a conflict silently — add a changelog row explaining which doc was wrong and why it was changed.

### 14.5 Minimum grep checks before merging

Run these sweeps on the `PointPillars_module/` directory before approving a PR; all hits must be deliberate historical notes (e.g. changelog lines that explicitly describe what was removed):

```
rg -n "TODO\(sync\)|PENDING SYNC" PointPillars_module/*.md
rg -n "BEVStateExtractor|BEVFeatureExtractor" PointPillars_module/*.md    # only allowed in v2/v3 changelog notes
rg -n "10 Hz|640.*480|60°" PointPillars_module/*.md                      # old camera/control specs
rg -n "cached-?`?z`?|re-encode.*gradient" PointPillars_module/*.md        # old buffer design
rg -n "critic-only encoder update" PointPillars_module/*.md               # only allowed when explicitly saying "not needed in default plan"
```

If any hit is NOT a deliberate historical note, fix before merging.

---

## 15. Changelog

| Date       | Author | Change                                              |
|------------|--------|-----------------------------------------------------|
| 2026-04-18 | init   | First version. 2-stage pipeline spec.               |
| 2026-04-18 | v2     | First split into state-branch + risk-branch architecture (superseded by v3). |
| 2026-04-18 | v3     | **Pipeline simplification** after stakeholder clarified that PointPillars is NOT used as SAC state — it is purely a dense reward function. Changes: (1) removed `BEVStateExtractor` and the "state branch" entirely; (2) Actor/Critic are now plain MLPs on a `ProprioState` vector (base lin/ang vel + goal relative + heading); (3) entire perception stream is frozen in Stage B — optimizer has 3 param groups, no encoder updates; (4) replay buffer stores only proprio states + actions + decomposed rewards (~500 MB for 1 M transitions); (5) framed explicitly as an **A/B test** (`r = r_env` vs `r = r_env + r_risk`) to measure the value of the pretrained risk predictor; (6) added explicit Mamba streaming spec (§6.4.2, `MambaStreamer`) at 20 Hz; (7) Stage B-plus (§6.8) updated — it still leaves the SAC path as proprio-MLP; only the perception stream is unfrozen. |
| 2026-04-18 | v3.1   | Added § 14 "Maintenance rule — keep these docs in sync": trigger matrix, per-PR checklist, conflict resolution, grep sweeps. Renumbered Changelog from § 14 to § 15. Same section added to `module_pointpillar.md` (§ 12) and `strategy_finetune_with_SAC.md` (§ 21). |
| 2026-04-18 | v3.2   | Stage-A scaffold landed in code. Added `PointPillars_module/data_contracts.py` (authoritative implementation of all §3 dataclasses + `to_npz`/`from_npz` + validators), `PointPillars_module/models/{spatial_reducer,mamba_temporal,risk_head,full_pipeline}.py` (§4.2–§4.4 specs) and `PointPillars_module/losses.py` (§5.3 focal-BCE + positive oversampling helper). Added `create_dataset_module/` (PyBullet rollouts → `Trajectory.npz` → `RiskDataset` → `collate_riskbatch`) wired to `pybullet_navigation.RL_Env`. Added two new rows to §13 relationships table for `create_dataset_module/` and `PointPillars_module/models/`. `MambaTemporal` keeps `mamba-ssm` as the primary backend with an `nn.GRU` fallback for non-CUDA environments; identical `forward` / `step` public interface on both backends. No architectural / contract changes — purely additive code alignment with §3–§5. |
| 2026-04-18 | v3.5   | `DataGenConfig.policy_stationary_p` + `StationaryPolicy` (zero cmd) for dynamic-obstacle coverage; § 5.1 config snippet updated. `strategy_create_trajectory_label.md` §9 expands critical observations (imbalance, stationary env dynamics, boundary ambiguity). |
| 2026-04-18 | v3.4   | Added `docs/strategy_create_trajectory_label.md` — full description of contact-based lookahead risk labels (`lookahead_any`, horizons, edge cases, code references). Cross-link from § 5.1. |
| 2026-04-18 | v3.3   | Dataset-generation fidelity pass. `DataGenConfig` gained four fields (`terminate_on_contact: bool = True`, `post_contact_grace_frames: int = 0`, `save_rgb: bool = False`) and its existing `depth_noise_std`/`drop_pixel_prob`/`camera_jitter_deg` fields are now **actually applied** in `DataGenerator._rollout` (previously they were dead code). Rollouts now bail out on first contact + grace frames so post-collision "robot-on-its-side" frames never enter the dataset; arrays are sliced to the actual length before serialization. RGB is written as a length-0 placeholder when `save_rgb=False`, cutting on-disk size ~80%. `DataGenerator.run()` now emits a human-readable summary (per-policy counts, per-horizon positive ratio, early-termination rate) and stores it on `self.last_stats`; `index.jsonl` rows gained `policy`, `terminated_on_contact`, and the 0.5s / 2s positive counts. Unit tests cover all four behaviors (domain-rand magnitudes, early-termination slicing, rgb placeholder, stats surface). §3.1 `Trajectory` contract unchanged (rgb shape was never validated); §5.1 defaults in the doc are the logical reference — code defaults differ in favor of the new safer behavior. |
| 2026-04-18 | —      | Added `docs/strategy_train_stage_A.md` — consolidated Stage A training lifecycle (data → `FullPipeline` → focal loss → A1/A2 → Stage B `r_risk`); §13 relationships row + §14.1 trigger note. |
| 2026-04-18 | —      | Added `docs/strategy_experiment_protocol.md` + `run_datagen_preset.py` preset **`experiment`** — downscaled data/train for method comparison; unified Stage A + B metrics; §13 / §14.1 updated. |
| 2026-04-18 | —      | Added `docs/optimized_training_strategy_stage_A.md` — divide & conquer (BEV caching, decoupled training), paper-style ablation tables, robustness hooks; §13 + §14.1 trigger. |
| 2026-04-18 | v3.6   | Stage A temporal ablations in code: `temporal_encoders.py` (LSTM, causal Transformer), `temporal_factory.build_temporal`, `train_stage_a_compare.py` (focal loss, scene-stratified split, AP/AUC logging, TensorBoard). `FullPipeline.mamba=` may be any compatible `(B,L,D)→(B,L,D)` module. Transformer has no Stage B `step()`. §4.3 + file layout updated. |
| 2026-04-18 | —      | Added `docs/skill_avoid_gradient_boom.md` + `PointPillars_module/utils/gradient_health.py`, `train_stage_b_sac.py` (SAC Actor/Critic reference, clamps, reward EMA helper), focal \(p_t\) clamp in `losses.py`, He init on `SpatialReducer` convs, grad-norm TensorBoard hooks in `train_stage_a_compare.py`. §13 row for skill doc. |
| 2026-04-18 | v3.7   | **Stage A compare — joint trajectory + risk.** `RiskSample` / `RiskBatch` gain `traj_future_xyyaw` (`H×(x,y,yaw)` world poses after frame `t`). `RiskDataset` exposes `traj_horizon` (default `H=10`). `FullPipeline.forward_to_h_T` refactors encoding; new `TrajectoryHead` + `FullPipelineRiskAndTraj` predict the same `H` poses from `h_T`. `train_stage_a_compare` trains focal-BCE (risk) + SmoothL1 (trajectory) and reports validation **risk** AP/AUC (three horizons) plus **trajectory** RMSE (all dims), ADE/FDE in XY (m), RMSE yaw (rad). Stage B default path unchanged (`FullPipeline.step` = risk only). §3.2 / §3.3 updated. |

---

*End of document. If anything here contradicts the code, update this file first, then the code.*
