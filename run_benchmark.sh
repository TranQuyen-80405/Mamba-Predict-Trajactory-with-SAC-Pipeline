#!/usr/bin/env bash
# Sequential Stage A benchmark: mamba, gru, lstm, transformer.
# Each run writes to LOG_ROOT/<backbone>_<timestamp>/ (override LOG_ROOT).
#
# Usage:
#   export DATA_ROOT=/path/to/index_rollouts
#   ./run_benchmark.sh
#
# One-shot compare (same scene split + defaults as train_stage_a_compare.py):
#   BENCHMARK_COMPARE=1 ./run_benchmark.sh
#
# Optional env: CKPT, LOG_ROOT, DEVICE (passed as --device), SEED, BEV_CACHE_ROOT
# Default training flags match Stage A1 compare (T_ctx=40 ≈ 2s context, accum for 16GB GPUs).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PP="$ROOT/PointPillars_module"
TRAIN="$PP/training"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/stage_a_experiment}"
CKPT="${CKPT:-$PP/pretrained/epoch_160.pth}"
LOG_ROOT="${LOG_ROOT:-$ROOT/runs/final_comparison}"
DEVICE="${DEVICE:-}"
SEED="${SEED:-0}"
BEV_CACHE_ROOT="${BEV_CACHE_ROOT:-}"
# Repo root on PYTHONPATH so create_dataset_module resolves when training scripts run from Pipeline/
export PYTHONPATH="$ROOT:$PP${PYTHONPATH:+:$PYTHONPATH}"

COMMON_ARGS=(
  --T_ctx 40
  --num_workers 4
  --batch_size 32
  --gradient_accumulation_steps 2
  --lr_warmup_iters 500
  --positive_weight 8.0
  --grad_clip 1.0
)
if [[ -n "$BEV_CACHE_ROOT" ]]; then
  COMMON_ARGS+=(--bev_cache_root "$BEV_CACHE_ROOT")
fi

mkdir -p "$LOG_ROOT"

if [[ "${BENCHMARK_COMPARE:-0}" == "1" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  sub="${LOG_ROOT}/compare_${ts}"
  mkdir -p "$sub"
  echo "=== train_stage_a_compare (mamba,gru,lstm,transformer) -> $sub ==="
  CMD=(
    python "$TRAIN/train_stage_a_compare.py"
    --data_root "$DATA_ROOT"
    --ckpt "$CKPT"
    --log_root "$sub"
    --models "mamba,gru,lstm,transformer"
    --seed "$SEED"
    "${COMMON_ARGS[@]}"
  )
  if [[ -n "$DEVICE" ]]; then
    CMD+=(--device "$DEVICE")
  fi
  "${CMD[@]}"
  echo "Done. TensorBoard: tensorboard --logdir $sub"
  exit 0
fi

run_one() {
  local backbone="$1"
  local script
  case "$backbone" in
    gru) script="train_stage_a_rnn_gru.py" ;;
    *) script="train_stage_a_${backbone}.py" ;;
  esac
  local ts name
  ts="$(date +%Y%m%d_%H%M%S)"
  name="${backbone}_${ts}"
  echo "=== ${backbone} -> ${LOG_ROOT}/${name} ==="
  if [[ -n "$DEVICE" ]]; then
    python "$TRAIN/$script" \
      --data_root "$DATA_ROOT" \
      --ckpt "$CKPT" \
      --log_root "$LOG_ROOT" \
      --run_name "$name" \
      --seed "$SEED" \
      "${COMMON_ARGS[@]}" \
      --device "$DEVICE"
  else
    python "$TRAIN/$script" \
      --data_root "$DATA_ROOT" \
      --ckpt "$CKPT" \
      --log_root "$LOG_ROOT" \
      --run_name "$name" \
      --seed "$SEED" \
      "${COMMON_ARGS[@]}"
  fi
}

for b in mamba gru lstm transformer; do
  run_one "$b"
done

echo "Done. TensorBoard: tensorboard --logdir $LOG_ROOT"
