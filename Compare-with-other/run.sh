#!/usr/bin/env bash

# cd to repo root
cd /workspace/Mamba-Predict-Trajactory-with-SAC-Pipeline

DATA_ROOT="/workspace/Mamba-Predict-Trajactory-with-SAC-Pipeline/data/stage_a_experiment_2gpu_balanced_v5"
CACHE_ROOT="/workspace/Mamba-Predict-Trajactory-with-SAC-Pipeline/data/stage_a_experiment_2gpu_balanced_v5_bev_cache_ds4"
LOG_ROOT="runs/compare_baselines"

echo "=== Starting sequential training for Trajectory models ==="
for model in lstm transformer mamba; do
  echo ">>> Training Trajectory Model: $model"
  .venv-backups/bin/python Compare-with-other/train_single_task.py \
    --task trajectory \
    --model $model \
    --data_root $DATA_ROOT \
    --bev_cache_root $CACHE_ROOT \
    --log_root $LOG_ROOT
done

echo "=== Starting sequential training for Risk models ==="
for model in mlp heatmap safety; do
  echo ">>> Training Risk Model: $model"
  .venv-backups/bin/python Compare-with-other/train_single_task.py \
    --task risk \
    --model $model \
    --data_root $DATA_ROOT \
    --bev_cache_root $CACHE_ROOT \
    --log_root $LOG_ROOT
done

echo "=== Finished training all models! ==="
