#!/usr/bin/env bash
set -eu
export PATH="${HOME}/.local/bin:${PATH}"
python3 -m pip install --user numpy ninja scikit-learn
