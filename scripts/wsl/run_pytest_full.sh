#!/usr/bin/env bash
set -eu
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="/mnt/e/RobotDog_Project/Pipeline/PointPillars_module"
cd /mnt/e/RobotDog_Project/Pipeline/PointPillars_module
exec python3 -m pytest tests/ -q --tb=short "$@"
