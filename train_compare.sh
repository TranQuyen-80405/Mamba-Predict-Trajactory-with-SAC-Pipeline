#!/usr/bin/env bash
# Stage A: so sánh backbone (mamba, gru, lstm, transformer).
# - Learnable task loss: mặc định (không dùng --fixed_task_loss).
# - BEV cache: v5 + bản nén ds4 (float16 + spatial_downsample=4).
# Chạy từ thư mục gốc repo:
#   chmod +x train_compare.sh && ./train_compare.sh
#
# Override ví dụ:
#   MODELS=transformer BATCH_SIZE=8 ./train_compare.sh
#   GRADIENT_ACCUMULATION_STEPS=4 LR=3e-5 ./train_compare.sh

set -euo pipefail
cd "$(dirname "$0")"

REPO_ROOT="$(pwd)"
# Venv chuẩn theo setup_train_env_ubuntu2404.sh: .venv-linux-2404
# (không dùng .venv — thường là bản dự phòng/Windows; override: VENV=/path …)
VENV="${VENV:-$REPO_ROOT/.venv-linux-2404}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PATH="/usr/local/cuda/bin:${PATH:-}"

# Bộ dữ liệu + cache (tạo bằng scripts/cache_pointpillars_bev.py --spatial_downsample 4)
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/stage_a_experiment_2gpu_balanced_v5}"
BEV_CACHE_ROOT="${BEV_CACHE_ROOT:-$REPO_ROOT/data/stage_a_experiment_2gpu_balanced_v5_bev_cache_ds4}"
CKPT="${CKPT:-$REPO_ROOT/PointPillars_module/pretrained/epoch_160_raw.pth}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/runs/stage_a_compare_turn-2_V5}"
TRAIN_LOG="${TRAIN_LOG:-$LOG_ROOT/train.log}"

# Hyper thống nhất với runs/.../run_config.json (stage_a_compare_turn-2_V5, stage a1)
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
COSINE_ETA_MIN="${COSINE_ETA_MIN:-1e-6}"
SEED="${SEED:-0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
POSITIVE_WEIGHT="${POSITIVE_WEIGHT:-8.0}"
STAGE="${STAGE:-a1}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-500}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"
EXTRINSICS_CONVENTION="${EXTRINSICS_CONVENTION:-auto}"
DEPTH_SCALE_FACTOR="${DEPTH_SCALE_FACTOR:-1.0}"
RISK_LABEL_SMOOTHING="${RISK_LABEL_SMOOTHING:-0.05}"
TEMPORAL_DROPOUT="${TEMPORAL_DROPOUT:-0.1}"
# Learnable task loss: mặc định bật (không truyền --fixed_task_loss) → traj_loss_weight null trong run_config

NUM_WORKERS="${NUM_WORKERS:-0}"
T_CTX="${T_CTX:-40}"
TRAJ_HORIZON="${TRAJ_HORIZON:-10}"
EPOCHS="${EPOCHS:-20}"
# Chỉ transformer như một run trong V5: MODELS=transformer ./train_compare.sh
MODELS="${MODELS:-mamba,gru,lstm}"

# Mitigate CUDA memory fragmentation (gợi ý từ PyTorch OOM).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Tránh shadow module chuẩn `types` bởi PointPillars_module/types.py
unset PYTHONPATH

# Log dòng (mỗi 100 micro-batch) + ghi file; tắt tqdm mặc định
export STAGE_A_NO_TQDM="${STAGE_A_NO_TQDM:-1}"

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$TRAIN_LOG") 2>&1
echo "[train_compare] logging to $TRAIN_LOG"

python PointPillars_module/training/train_stage_a_compare.py \
  --data_root "$DATA_ROOT" \
  --ckpt "$CKPT" \
  --bev_cache_root "$BEV_CACHE_ROOT" \
  --models "$MODELS" \
  --log_root "$LOG_ROOT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --cosine_eta_min "$COSINE_ETA_MIN" \
  --seed "$SEED" \
  --grad_clip "$GRAD_CLIP" \
  --positive_weight "$POSITIVE_WEIGHT" \
  --lr_warmup_iters "$LR_WARMUP_ITERS" \
  --stage "$STAGE" \
  --early_stop_patience "$EARLY_STOP_PATIENCE" \
  --early_stop_min_delta "$EARLY_STOP_MIN_DELTA" \
  --extrinsics_convention "$EXTRINSICS_CONVENTION" \
  --depth_scale_factor "$DEPTH_SCALE_FACTOR" \
  --risk_label_smoothing "$RISK_LABEL_SMOOTHING" \
  --temporal_dropout "$TEMPORAL_DROPOUT" \
  --num_workers "$NUM_WORKERS" \
  --T_ctx "$T_CTX" \
  --traj_horizon "$TRAJ_HORIZON" \
  --mamba_backend auto
