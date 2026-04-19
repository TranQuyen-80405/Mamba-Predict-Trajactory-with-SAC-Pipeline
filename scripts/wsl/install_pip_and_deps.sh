#!/usr/bin/env bash
set -eu
cd /tmp
if ! python3 -c "import pip" 2>/dev/null; then
  curl -sS -O https://bootstrap.pypa.io/get-pip.py
  python3 get-pip.py --user
fi
export PATH="${HOME}/.local/bin:${PATH}"
python3 -m pip install --user -U pip setuptools wheel
python3 -m pip install --user pytest
# CPU torch for conftest (import torch)
python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --user numpy ninja scikit-learn tensorboard
# Optional: full dev stack for all tests (scikit-learn, etc.)
if [[ "${INSTALL_DEV:-0}" == "1" ]]; then
  python3 -m pip install --user -r /mnt/e/RobotDog_Project/Pipeline/PointPillars_module/requirements-dev.txt 2>/dev/null || true
fi
echo "pip ok:"
python3 -m pip --version
