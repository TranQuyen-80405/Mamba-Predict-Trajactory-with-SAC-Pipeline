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

For GPU training (CUDA toolkit **12.6** on PATH / `nvcc`: use **`cu126`** or **`auto`** so PyTorch matches `nvcc` when building `causal-conv1d`; **`cu124`** with `nvcc` 12.6 causes a CUDA version mismatch):

```bash
TORCH_MODE=auto INSTALL_MAMBA=1 ./scripts/setup_train_env_ubuntu2404.sh
```

Hoặc chỉ định thủ công:

```bash
TORCH_MODE=cu126 INSTALL_MAMBA=1 ./scripts/setup_train_env_ubuntu2404.sh
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

6) `causal-conv1d` / `mamba-ssm` build: CUDA in `nvcc` does not match PyTorch wheel (e.g. system **12.6** vs `torch+cu124` **12.4**)
- Symptom: `RuntimeError` from `torch.utils.cpp_extension._check_cuda_version`.
- Fix: install PyTorch with the matching wheel tag (`TORCH_MODE=cu126` or `TORCH_MODE=auto` when `nvcc` is CUDA 12.6+), or install a CUDA **12.4** toolkit and point `CUDA_HOME` at it.

7) `nvcc` **12.4** but PyTorch **`+cu121`** (CUDA **12.1**) — often leftover torch in `.venv-linux-2404` or `TORCH_MODE=cu121` while the toolkit on `PATH` is 12.4
- Symptom: mismatch `'12.4', '12.1'` in `_check_cuda_version`.
- Fix: `TORCH_MODE=cu124` or `TORCH_MODE=auto`, and ensure pip upgrades torch (this script uses `--upgrade`); if needed, delete the venv and rerun.

8) `nvcc` **12.9** but PyTorch **`+cu130`** (CUDA **13.0**) — pip pulled the latest CUDA13 wheel while the system toolkit is still 12.x
- Symptom: mismatch `'12.9', '13.0'` in `_check_cuda_version`.
- Fix: **`TORCH_MODE=cu129`** or **`TORCH_MODE=auto`** (maps 12.9→`cu129`). To use **cu130**, install a **CUDA 13.x** toolkit so `nvcc` reports 13.x.

9) **Even with `TORCH_MODE=cu129`**, `pip install causal-conv1d` / `mamba-ssm` can **upgrade `torch` again** — both declare a bare **`torch`** dependency on PyPI (no CUDA pin), so pip may install the **latest** torch (**+cu130**) from PyPI.org.
- Symptom: CUDA check passed, then **`causal-conv1d`** build fails again with `12.9` vs `13.0`.
- Fix: this script installs **`causal-conv1d` and `mamba-ssm` with `--no-deps`** after installing `einops` / `transformers` / etc., so **`torch` is not upgraded** by those resolves.

10) **Pip build isolation** (`/tmp/pip-build-env-*`) installs its **own** `torch` (often **+cu130**) while compiling sdist → same `12.9` vs `13.0` mismatch.
- Fix: install **`causal-conv1d` and `mamba-ssm` with `--no-build-isolation`** so compilation uses the venv’s `torch` wheel.

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
