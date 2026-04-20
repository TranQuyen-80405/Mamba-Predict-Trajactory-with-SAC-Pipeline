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
TORCH_MODE_REQUESTED="${TORCH_MODE:-cpu}" # cpu | auto | cu130 | cu129 | cu128 | cu126 | cu124 | cu121
TORCH_MODE="${TORCH_MODE_REQUESTED}"
INSTALL_MAMBA="${INSTALL_MAMBA:-1}" # 1 = install mamba-ssm stack, 0 = skip

timestamp="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${REPO_DIR}/logs"
LOG_FILE="${REPO_DIR}/logs/setup_ubuntu2404_${timestamp}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

# Many driver installs expose nvcc only under /usr/local/cuda/bin (not on default PATH for non-login shells).
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
fi

if [[ "${TORCH_MODE_REQUESTED}" == "auto" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    _cuda_ver="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -1)"
    echo "[setup] nvcc CUDA=${_cuda_ver:-unknown}"
    # Pick PyTorch wheel tag so torch.version.cuda matches nvcc (required for causal-conv1d / mamba-ssm builds).
    if [[ "${_cuda_ver}" =~ ^13\. ]]; then
      TORCH_MODE="cu130"
    elif [[ "${_cuda_ver}" =~ ^12\.([0-9]+) ]]; then
      _minor="${BASH_REMATCH[1]}"
      if ((_minor >= 9)); then
        TORCH_MODE="cu129"
      elif ((_minor >= 8)); then
        TORCH_MODE="cu128"
      elif ((_minor >= 6)); then
        TORCH_MODE="cu126"
      elif ((_minor >= 4)); then
        TORCH_MODE="cu124"
      elif ((_minor >= 1)); then
        TORCH_MODE="cu121"
      else
        TORCH_MODE="cu121"
      fi
    else
      echo "[warn] CUDA ${_cuda_ver:-?} not 12.x/13.x — using cu124 (set TORCH_MODE=cu130|cu129|… to override)"
      TORCH_MODE="cu124"
    fi
  else
    echo "[setup] auto: nvcc not on PATH — using cpu (install CUDA toolkit or set TORCH_MODE explicitly)"
    TORCH_MODE="cpu"
  fi
fi

echo "[setup] repo=${REPO_DIR}"
echo "[setup] venv=${VENV_DIR}"
echo "[setup] torch_mode=${TORCH_MODE_REQUESTED}"
echo "[setup] torch_mode_effective=${TORCH_MODE}"
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
# Remove any existing torch stack first: otherwise `pip install --upgrade` from cu129 may leave +cu130 installed
# (newer version number) and causal-conv1d still sees CUDA 13.0 vs nvcc 12.9.
echo "[info] uninstalling any existing torch/torchvision/torchaudio in venv (ignore errors if absent)"
python -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

_torch_pip() {
  # --force-reinstall --no-cache-dir: ensure the requested CUDA wheel replaces a stuck +cu130/+cu121 build.
  python -m pip install --no-cache-dir --upgrade --force-reinstall torch \
    --index-url "${1}" --extra-index-url https://pypi.org/simple
}

if [[ "${TORCH_MODE}" == "cpu" ]]; then
  python -m pip install --no-cache-dir --upgrade --force-reinstall torch \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
elif [[ "${TORCH_MODE}" == "cu130" ]]; then
  _torch_pip "https://download.pytorch.org/whl/cu130"
elif [[ "${TORCH_MODE}" == "cu129" ]]; then
  _torch_pip "https://download.pytorch.org/whl/cu129"
elif [[ "${TORCH_MODE}" == "cu128" ]]; then
  _torch_pip "https://download.pytorch.org/whl/cu128"
elif [[ "${TORCH_MODE}" == "cu126" ]]; then
  _torch_pip "https://download.pytorch.org/whl/cu126"
elif [[ "${TORCH_MODE}" == "cu124" ]]; then
  _torch_pip "https://download.pytorch.org/whl/cu124"
elif [[ "${TORCH_MODE}" == "cu121" ]]; then
  _torch_pip "https://download.pytorch.org/whl/cu121"
else
  echo "[error] invalid TORCH_MODE=${TORCH_MODE}. Use cpu|auto|cu130|cu129|cu128|cu126|cu124|cu121"
  exit 1
fi

python -c "import torch; print('[info] torch:', torch.__version__, '| torch.version.cuda:', torch.version.cuda)"

echo "[step] install numpy (early — avoids torch warning before CUDA check)"
python -m pip install --upgrade numpy

if [[ "${INSTALL_MAMBA}" == "1" ]] && [[ "${TORCH_MODE}" != "cpu" ]]; then
  echo "[step] check nvcc vs torch.version.cuda (causal-conv1d build)"
  python <<'PY'
import os, re, shutil, subprocess, sys

import torch

tc = torch.version.cuda
if not tc:
    print("[error] torch.version.cuda is None but TORCH_MODE is GPU — reinstall torch or use cpu")
    sys.exit(1)

_nvcc = shutil.which("nvcc")
if not _nvcc and os.path.isfile("/usr/local/cuda/bin/nvcc"):
    _nvcc = "/usr/local/cuda/bin/nvcc"
if not _nvcc:
    print("[warn] nvcc not found — skipping CUDA match check (build may still fail)")
    sys.exit(0)

r = subprocess.run([_nvcc, "-V"], capture_output=True, text=True)
nvcc_out = (r.stdout or "") + (r.stderr or "")
m = re.search(r"release (\d+\.\d+)", nvcc_out)
if not m:
    print("[warn] nvcc output unparsed — skipping CUDA match check (build may still fail)")
    sys.exit(0)

def mm(v: str) -> tuple[int, int]:
    a, b = v.split(".", 1)
    return int(a), int(b)

nv = m.group(1)
if mm(nv) != mm(tc):
    print(
        f"[error] CUDA mismatch: nvcc reports {nv}, torch.version.cuda={tc}. "
        "They must match when compiling causal-conv1d.\n"
        "  → Re-run with TORCH_MODE=auto (maps nvcc 12.9→cu129, 13.x→cu130), or install CUDA toolkit matching your torch wheel.\n"
        "  → If the venv had an old torch: rm -rf .venv-linux-2404 or rely on this script's pip install --upgrade torch."
    )
    sys.exit(1)
print(f"[check] nvcc {nv} matches torch CUDA {tc}")
PY
fi

echo "[step] install project python deps"
python -m pip install ninja pytest tensorboard scikit-learn

if [[ "${INSTALL_MAMBA}" == "1" ]]; then
  echo "[step] install mamba stack"
  # PyPI: unpinned "torch" can pull +cu130; also default pip *build isolation* installs its own torch in /tmp/pip-build-env-*
  # (often +cu130) → nvcc 12.9 vs torch 13.0 at compile. Use --no-deps and --no-build-isolation so the venv torch is used.
  python -m pip install --no-cache-dir packaging einops transformers \
    --extra-index-url https://pypi.org/simple
  if ! python -c "import triton" 2>/dev/null; then
    echo "[info] installing triton (not provided by current torch metadata in this env)"
    python -m pip install --no-cache-dir triton --extra-index-url https://pypi.org/simple
  fi
  python -m pip install --no-cache-dir "causal-conv1d>=1.4.0" \
    --extra-index-url https://pypi.org/simple --no-deps --no-build-isolation
  python -m pip install --no-cache-dir "mamba-ssm" \
    --extra-index-url https://pypi.org/simple --no-deps --no-build-isolation
  python -c "import torch; print('[post-mamba] torch:', torch.__version__, '| cuda:', torch.version.cuda)"
fi

echo "[step] sanity checks"
python -c "import torch, numpy, pytest; print('deps-ok', torch.__version__)"
python -c "import PointPillars_module.types, PointPillars_module.data_contracts, PointPillars_module.module_pointpillar, PointPillars_module.train_stage_b_sac; print('import-ok')"
python -m pytest PointPillars_module/tests/test_data_contracts.py -q
python -m pytest PointPillars_module/tests/test_temporal_encoders_pytest.py -q

echo "[done] setup completed successfully"
