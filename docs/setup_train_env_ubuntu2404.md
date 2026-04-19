# Ubuntu 24.04 One-Shot Setup (Training Host)

This guide is the reproducible setup path for Linux training hosts to avoid dependency conflicts before running Stage A/Stage B training.

## Target machine

- OS: Ubuntu 24.04.x LTS
- Arch: x86_64
- Kernel: Linux 5.15+ (WSL or native is fine)

## Quick start

From repo root (example repo folder name: `Mamba-Predict-Trajactory-with-SAC-Pipeline`):

```bash
cd ~/Mamba-Predict-Trajactory-with-SAC-Pipeline
chmod +x scripts/setup_train_env_ubuntu2404.sh
TORCH_MODE=cpu INSTALL_MAMBA=0 ./scripts/setup_train_env_ubuntu2404.sh
```

For GPU training:

```bash
TORCH_MODE=cu124 INSTALL_MAMBA=1 ./scripts/setup_train_env_ubuntu2404.sh
```

Script behavior:

- Creates/uses venv: `.venv-linux-2404`
- Logs full output to `logs/setup_ubuntu2404_<timestamp>.log`
- Installs system deps: `python3.12-venv`, `build-essential`
- Installs Python deps in stable order:
  1) pip/wheel/setuptools, 2) torch, 3) numpy/ninja/pytest/tensorboard/scikit-learn, 4) mamba stack (optional)
- Runs sanity checks:
  - import check for key modules
  - `test_data_contracts`
  - `test_temporal_encoders_pytest`

## Known setup incidents (recorded)

These are real failures observed during environment bring-up and now handled in the script/process.

1) `ensurepip is not available` when creating venv
- Cause: missing Ubuntu package `python3.12-venv`.
- Fix: install via apt before `python -m venv`.

2) pip conflict after interrupted install (`setuptools` / broken dist-info)
- Symptom: OSError while uninstalling or replacing setuptools inside venv.
- Cause: previous install interrupted mid-transaction.
- Fix: do not reuse that broken venv for production setup; create fresh venv (`.venv-linux-2404`).

3) `No matching distribution found for ninja` while using PyTorch CPU index
- Cause: `--index-url` pointed to PyTorch index for all packages.
- Fix: install torch with PyTorch index + `--extra-index-url https://pypi.org/simple`, then install remaining deps from default PyPI.

4) Windows-native `mamba-ssm` / `causal-conv1d` install failures
- Symptom: build errors / `CalledProcessError`.
- Cause: these packages are Linux/CUDA-friendly; native Windows is fragile for source builds.
- Fix: install in Ubuntu/WSL/Colab only, with retry using `--no-build-isolation`.

5) Fresh WSL distro missing `torch/numpy/pytest`
- Cause: new distro has only base Python.
- Fix: mandatory dependency bootstrap before running any project sanity tests.

## Verified commands (Linux venv)

These commands were used as final sanity checks:

```bash
python -c "import PointPillars_module.types, PointPillars_module.data_contracts, PointPillars_module.module_pointpillar, PointPillars_module.train_stage_b_sac; print('ok')"
python -m pytest PointPillars_module/tests/test_data_contracts.py -q
python -m pytest PointPillars_module/tests/test_temporal_encoders_pytest.py -q
```

Expected:

- Import command prints `ok`
- `test_data_contracts`: pass
- `test_temporal_encoders_pytest`: pass/skip as defined by test markers

## Operational recommendation

- Keep one dedicated venv per machine (`.venv-linux-2404`), do not share with ad-hoc experiments.
- Use script + log file for every setup run; archive the log alongside training artifacts for traceability.
