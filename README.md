# Mamba-Predict-Trajactory-with-SAC-Pipeline

PointPillars → temporal encoder → risk + trajectory (Stage A) → SAC on proprio (Stage B). Specs: `docs/strategy_full_pipeline.md`.

## Top-level layout

| Path | Role |
|------|------|
| `PointPillars_module/` | Perception + models + **`training/`** (Stage A train scripts) |
| `create_dataset_module/` | PyBullet rollouts → `data/…` + `risk_dataset` |
| `notebooks/` | Exploratory / Colab-oriented Jupyter notebooks |
| `scripts/` | Utilities (`execute_compare_…`, `patch_compare_nb_paths`, …) |
| `scripts/datagen/` | Dataset presets (`run_datagen_preset.py`, `run_generate_small.py`) |
| `tools/` | Small helpers (e.g. RGB montage → PNG) |
| `docs/` | Architecture and experiment protocols |
| `data/` | Stage A datasets (tracked subsets for clone-and-run) |
| `urdf/` | Robot / diff-drive URDF for sim |

Root **wrappers** (thin; delegate to `scripts/` or `tools/`) keep old commands working:

- `run_datagen_preset.py` → `scripts/datagen/run_datagen_preset.py`
- `run_generate_small.py` → `scripts/datagen/run_generate_small.py`
- `run_dataset_stats.py` → `scripts/check_dataset_stats.py`
- `rgb_preview_to_png.py` → `tools/rgb_preview_to_png.py`

`pybullet_navigation.py` stays at repo root (imported by `create_dataset_module`).

## Stage A training (Python)

From repo root, with venv active:

```bash
python PointPillars_module/training/train_stage_a_mamba.py --data_root data/stage_a_experiment --ckpt PointPillars_module/pretrained/epoch_160.pth
```

Or use the shim: `from train_stage_a_compare import run_experiment` with `PointPillars_module` on `PYTHONPATH` / `sys.path` (see notebooks).

Default workflow now requires BEV cache at `<data_root>_bev_cache`.

```bash
python scripts/cache_pointpillars_bev.py \
  --data_root data/stage_a_experiment_2gpu_balanced_v8 \
  --ckpt PointPillars_module/pretrained/epoch_160_raw.pth \
  --out_root data/stage_a_experiment_2gpu_balanced_v8_bev_cache
```

## Check existing dataset stats

```bash
python run_dataset_stats.py --data_root data/stage_a_experiment_2gpu_balanced_v8 --T_ctx 40 --traj_horizon 10
```

This inspects `index.jsonl` + on-disk rollouts and prints:
- rollout/scene counts and effective frame totals,
- positive ratios (0.5s / 1.0s / 2.0s),
- train/val/test scene split and sample-level horizon stats (valid/positive/negative).

Recommended balanced dataset preset for Stage-A comparison on 2x 5060 Ti:

```bash
python run_datagen_preset.py experiment_2gpu
python run_dataset_stats.py --data_root data/stage_a_experiment_2gpu_balanced_v8 --T_ctx 40 --traj_horizon 10
```

## Tests

```bash
.\.venv\Scripts\python.exe -m pytest PointPillars_module\tests create_dataset_module\tests
```
