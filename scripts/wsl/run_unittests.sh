#!/usr/bin/env bash
# Chay trong WSL (Ubuntu). Path repo tren Windows E: -> /mnt/e/...
set -eu

REPO_ROOT="${REPO_ROOT:-/mnt/e/RobotDog_Project/Pipeline}"
PP="${REPO_ROOT}/PointPillars_module"

echo "=== OS ==="
cat /etc/os-release 2>/dev/null | head -5
echo "=== Kernel (WSL2 = Microsoft kernel, khong phai 5.15.0-119-generic) ==="
uname -a

if ! command -v python3 >/dev/null 2>&1; then
  echo "Cai python3: sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv"
  exit 1
fi

cd "$PP"
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="$PP:${PYTHONPATH:-}"

# Uu tien: venv trong repo neu co
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.venv/bin/activate"
elif [[ -f "$PP/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$PP/.venv/bin/activate"
fi

if ! python3 -c "import pytest" 2>/dev/null; then
  echo "Chay bash scripts/wsl/install_pip_and_deps.sh de cai pip + pytest + torch + numpy + tensorboard"
  exit 1
fi

echo "=== pytest PointPillars_module/tests ==="
python3 -m pytest tests/ -q "$@"
