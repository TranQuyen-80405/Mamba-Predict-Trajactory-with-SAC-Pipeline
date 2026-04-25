#!/usr/bin/env bash

# cd to repo root
cd /workspace/Mamba-Predict-Trajactory-with-SAC-Pipeline

# Train Trajectory model (ví dụ: mamba)
/workspace/Mamba-Predict-Trajactory-with-SAC-Pipeline/.venv-backups/bin/python Compare-with-other/train_single_task.py \
  --task trajectory \
  --model mamba \
  --data_root /workspace/Mamba-Predict-Trajactory-with-SAC-Pipeline/data/stage_a_experiment_2gpu_balanced_v5 \
  --bev_cache_root /workspace/Mamba-Predict-Trajactory-with-SAC-Pipeline/data/stage_a_experiment_2gpu_balanced_v5_bev_cache_ds4 \
  --log_root runs/compare_baselines

# Train Risk model (ví dụ: mlp)
/workspace/Mamba-Predict-Trajactory-with-SAC-Pipeline/.venv-backups/bin/python Compare-with-other/train_single_task.py \
  --task risk \
  --model mlp \
  --data_root /workspace/Mamba-Predict-Trajactory-with-SAC-Pipeline/data/stage_a_experiment_2gpu_balanced_v5 \
  --bev_cache_root /workspace/Mamba-Predict-Trajactory-with-SAC-Pipeline/data/stage_a_experiment_2gpu_balanced_v5_bev_cache_ds4 \
  --log_root runs/compare_baselines
