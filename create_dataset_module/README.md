# create_dataset_module

Offline Stage-A dataset generation for the RobotDog pipeline.

This module turns **PyBullet rollouts** into `Trajectory.npz` files and wraps
them in a `torch.utils.data.Dataset` that emits `RiskSample` / `RiskBatch`
objects matching the contracts in `docs/strategy_full_pipeline.md` §3 and §5.

## Layout

```
create_dataset_module/
  __init__.py           public surface
  config.py             re-exports DataGenConfig / EnvConfig from data_contracts
  env_wrapper.py        thin wrapper over pybullet_navigation.RL_Env
  policies.py           RandomPolicy / ScriptedPolicy / AdversarialPolicy
  generator.py          DataGenerator.run() + lookahead_any()
  risk_dataset.py       RiskDataset + collate_riskbatch + scene_stratified_split
  verify.py             end-to-end smoke: generate -> load -> collate -> pipeline
  tests/                unit tests (PyBullet-free + PyBullet smoke)
```

## Two-minute local smoke

1. Install the runtime deps (one-time):

   ```bash
   pip install torch numpy pybullet matplotlib
   # optional but recommended on Linux:
   # pip install mamba-ssm
   ```

2. Run the unit tests:

   ```bash
   python -m unittest discover -s create_dataset_module/tests -t . -v
   ```

   Risk-label / `Trajectory` / `RiskDataset` pytest suite (no PyBullet):

   ```bash
   pip install -r create_dataset_module/requirements-dev.txt
   python -m pytest create_dataset_module/tests/test_risk_groundtruth_pytest.py -v
   ```

   Full unittest discovery expects `OK` (with `pybullet` for env tests). The pytest file above is PyBullet-free.

3. Run the end-to-end smoke:

   ```bash
   python -m create_dataset_module.verify
   ```

   This will:
   1. generate **1 scene x 1 rollout x 60 frames** into a temp directory,
   2. open it as a `RiskDataset` and collate a mini-batch,
   3. optionally run `FullPipeline.forward` if a GPU + PointPillars
      checkpoint (`pretrained/epoch_160.pth`) are available; otherwise it
      prints `[SKIP]` and still exits 0.

## Experiment-scale dataset (method comparison)

For a **small** disk footprint (~0.2–0.5 GiB depth tree) suitable for comparing models, use the **`experiment`** preset:

```bash
python run_datagen_preset.py experiment
```

Defaults and evaluation criteria: **`docs/strategy_experiment_protocol.md`**.

## Generating a bigger dataset

```python
from create_dataset_module import DataGenerator
from create_dataset_module.config import DataGenConfig

cfg = DataGenConfig(
    out_dir="data/stage_a",
    n_scenes=100,
    rollouts_per_scene=4,
    frames_per_rollout=400,
    policy_random_p=0.5,
    policy_scripted_p=0.3,
    policy_adversarial_p=0.2,

    # Domain randomization (v3.3). Set any to 0/False to disable.
    depth_noise_std=0.01,       # Gaussian noise on depth (m)
    drop_pixel_prob=0.02,       # set 2% of pixels to 0
    camera_jitter_deg=1.0,      # small extrinsic rotation each frame

    # Early termination (v3.3). Default True so adversarial rollouts
    # don't record post-collision garbage.
    terminate_on_contact=True,
    post_contact_grace_frames=0,

    # RGB is only useful for human debug; skip to save ~80% disk.
    save_rgb=False,

    seed=42,
)

n = DataGenerator(cfg).run()
# After run() finishes, gen.last_stats holds the aggregate summary.
```

At the end of each run you get a summary like:

```
============================================================
 DataGenerator summary  (out_dir = data/stage_a)
============================================================
   rollouts written     : 400
   frames total         : 89_241
   early-terminated     : 61 (15.2%)
   positive ratio 0.5s  : 4.13%
   positive ratio 1.0s  : 11.87%
   positive ratio 2.0s  : 22.05%
   policy random      : 199  (49.8%)
   policy scripted    : 124  (31.0%)
   policy adversarial : 77   (19.2%)
============================================================
```

On-disk format per rollout: a single `s{scene:04d}_r{rollout:02d}.npz`
(see `Trajectory.to_npz` in `PointPillars_module/data_contracts.py`) plus
one JSON line appended to `index.jsonl`. Each index row carries
`{path, scene_id, rollout_id, T, policy, terminated_on_contact,
n_positive_05s, n_positive_1s, n_positive_2s}` for quick filtering.

## Legit for real training (not just smoke tests)

Smoke configs (`verify`, `run_generate_small`, preset `smoke_nb`) only prove the **pipeline runs**. Before a serious Stage-A train, also check:

1. **Scale** — use a large enough plan (e.g. on the order of ~200 scenes × 4 rollouts × 400 frames, or equivalent total frames). Tiny smoke sets are for debugging only.

2. **Positive ratios & policy mix** — read the `DataGenerator` summary at the end of `run()`. Aim for **positive(1s) roughly in the ~10–15% band** (adjust `policy_random_p` / `policy_scripted_p` / `policy_adversarial_p` until the WARN goes away or ratios sit in your target band). Too few positives → model never learns risk; too many → biased “always danger”. Optional **`policy_stationary_p`** (must still sum to 1.0 with the other three) runs `StationaryPolicy` (v=w=0) so **dynamic obstacles** can still create contacts — see `docs/strategy_create_trajectory_label.md` §9.

3. **Optional RGB spot-checks** — run the **Spot-check RGB** cells at the bottom of `Create_dataset.ipynb` (preset `rgb_spotcheck` via `run_datagen_preset.py`, then display frames), or set `save_rgb=True` manually for a few short rollouts. Turn `save_rgb` off again for the full training dataset (RGB ~80% of bytes).

4. **Splits** — use `scene_stratified_split` so train / val / test do not share the same obstacle layout.

## Scene-stratified splits

`scene_stratified_split` guarantees no scene leakage between train / val / test:

```python
from create_dataset_module import RiskDataset
from create_dataset_module.risk_dataset import scene_stratified_split

all_scenes = list(range(cfg.n_scenes))
tr, va, te = scene_stratified_split(all_scenes, ratios=(0.8, 0.1, 0.1), seed=0)

ds_tr = RiskDataset("data/stage_a", scene_filter=tr)
ds_va = RiskDataset("data/stage_a", scene_filter=va)
ds_te = RiskDataset("data/stage_a", scene_filter=te)
```

## Positive oversampling

For class balance on `risk_1s`:

```python
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

risk = torch.from_numpy(ds_tr.risk_1s_array())
weights = torch.where(risk > 0, 10.0, 1.0)
sampler = WeightedRandomSampler(weights, num_samples=len(ds_tr), replacement=True)

loader = DataLoader(
    ds_tr, batch_size=16, sampler=sampler,
    collate_fn=__import__("create_dataset_module").collate_riskbatch,
    num_workers=0,   # PyBullet is not fork-safe; prefer 0 on Windows
)
```

## Shipping to Colab

1. Zip the *project root* (keeps the `PointPillars_module` sibling so
   imports resolve):

   ```powershell
   # from E:\RobotDog_Project\Pipeline (Windows)
   Compress-Archive -Path `
     PointPillars_module, create_dataset_module, pybullet_navigation.py, urdf `
     -DestinationPath ../Pipeline_stage_a.zip -Force
   ```

   or

   ```bash
   # from /path/to/Pipeline (Linux / macOS)
   zip -r ../Pipeline_stage_a.zip \
       PointPillars_module create_dataset_module pybullet_navigation.py urdf
   ```

2. Upload `Pipeline_stage_a.zip` to Colab.

3. Open `create_dataset_module/colab_quickstart.ipynb` in Colab and run the
   first cell; it will unzip, install deps, run the tests, and hand you a
   ready-to-train `RiskDataset`.

## Camera & frame conventions

`DatasetEnv` hard-codes:

- depth resolution `160 x 120`,
- horizontal FoV `90 deg`,
- near / far `0.1 / 8.0` m,
- `float16` depth storage (clipped to `[0, far]`).

These numbers match `docs/strategy_full_pipeline.md` §4.1 and the
`DepthPreprocessConfig` defaults used by `RiskDataset`. If you change them
in `DataGenConfig`, also pass the same `max_range` to your
`DepthPreprocessConfig` at training time.
