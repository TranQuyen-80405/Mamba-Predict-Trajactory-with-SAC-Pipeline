#!/usr/bin/env bash
set -euo pipefail

# One-shot setup for Ubuntu 24.04 training hosts.
# - Creates isolated venv on current repo
# - Installs stable Python deps for this project
# - Optionally installs GPU-ready torch/mamba stack
# - Runs sanity checks at the end

REPO_DIR="${REPO_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-.venv-linux-2404}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_MODE="${TORCH_MODE:-cpu}" # cpu | cu124 | cu121
INSTALL_MAMBA="${INSTALL_MAMBA:-1}" # 1 = install mamba-ssm stack, 0 = skip

timestamp="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${REPO_DIR}/logs"
LOG_FILE="${REPO_DIR}/logs/setup_ubuntu2404_${timestamp}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[setup] repo=${REPO_DIR}"
echo "[setup] venv=${VENV_DIR}"
echo "[setup] torch_mode=${TORCH_MODE}"
echo "[setup] install_mamba=${INSTALL_MAMBA}"
echo "[setup] log=${LOG_FILE}"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "[error] repo dir not found: ${REPO_DIR}"
  exit 1
fi

cd "${REPO_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[error] ${PYTHON_BIN} not found"
  exit 1
fi

echo "[step] install system prerequisites"
sudo apt-get update
sudo apt-get install -y python3.12-venv build-essential

echo "[step] create/refresh venv"
if [[ -d "${VENV_DIR}" ]]; then
  echo "[info] venv already exists: ${VENV_DIR}"
else
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[step] bootstrap pip toolchain"
python -m pip install -U pip wheel
python -m pip install "setuptools==81.0.0"

echo "[step] install torch"
if [[ "${TORCH_MODE}" == "cpu" ]]; then
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
elif [[ "${TORCH_MODE}" == "cu124" ]]; then
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
elif [[ "${TORCH_MODE}" == "cu121" ]]; then
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
else
  echo "[error] invalid TORCH_MODE=${TORCH_MODE}. Use cpu|cu124|cu121"
  exit 1
fi

echo "[step] install project python deps"
python -m pip install numpy ninja pytest tensorboard scikit-learn

if [[ "${INSTALL_MAMBA}" == "1" ]]; then
  echo "[step] install mamba stack"
  python -m pip install "causal-conv1d>=1.4.0" --extra-index-url https://pypi.org/simple
  python -m pip install "mamba-ssm[causal-conv1d]" --extra-index-url https://pypi.org/simple || \
    python -m pip install "mamba-ssm[causal-conv1d]" --extra-index-url https://pypi.org/simple --no-build-isolation
fi

echo "[step] sanity checks"
python -c "import torch, numpy, pytest; print('deps-ok', torch.__version__)"
python -c "import PointPillars_module.types, PointPillars_module.data_contracts, PointPillars_module.module_pointpillar, PointPillars_module.train_stage_b_sac; print('import-ok')"
python -m pytest PointPillars_module/tests/test_data_contracts.py -q
python -m pytest PointPillars_module/tests/test_temporal_encoders_pytest.py -q

echo "[done] setup completed successfully"
