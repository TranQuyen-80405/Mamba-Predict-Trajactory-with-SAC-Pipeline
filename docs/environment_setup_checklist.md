# Environment Setup Checklist (No Missing Critical Dependencies)

Use this checklist every time you set up a new training machine or rebuild the venv.

## 1) Preflight

- [ ] Repo cloned and on correct branch.
- [ ] You are in repo root.
- [ ] Ubuntu 24.04 (or compatible Linux) is available.
- [ ] NVIDIA driver is visible (`nvidia-smi` works) for GPU training.

## 2) Create/Refresh Environment

- [ ] Make setup script executable:
  - `chmod +x scripts/setup_train_env_ubuntu2404.sh`
- [ ] Run setup script:
  - GPU training: `TORCH_MODE=auto INSTALL_MAMBA=1 ./scripts/setup_train_env_ubuntu2404.sh`
  - CPU fallback only: `TORCH_MODE=cpu INSTALL_MAMBA=0 ./scripts/setup_train_env_ubuntu2404.sh`
- [ ] Confirm setup log created under `logs/setup_ubuntu2404_<timestamp>.log`.

## 3) CUDA + PyTorch GPU Alignment (Critical)

- [ ] Activate venv:
  - `source .venv-linux-2404/bin/activate`
- [ ] If needed, set CUDA path:
  - `export CUDA_HOME=/usr/local/cuda-12.8`
  - `export PATH=$CUDA_HOME/bin:$PATH`
- [ ] Verify torch is CUDA build (not `+cpu`):
  - `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
- [ ] Verify `nvcc` exists and reports expected toolkit:
  - `nvcc --version`
- [ ] Verify GPU visible in torch:
  - `python -c "import torch; print(torch.cuda.device_count())"`

## 4) Build PointPillars Voxel Op

- [ ] Install/build extension in editable mode:
  - `python -m pip install -e ./PointPillars_module --no-build-isolation`
- [ ] If forcing CUDA build is needed:
  - `FORCE_CUDA=1 python -m pip install -e ./PointPillars_module --no-build-isolation --force-reinstall`
- [ ] Quick smoke test (GPU hard_voxelize) passes.

## 5) Mamba Stack for 4-Model Comparison (Critical for `mamba`)

- [ ] `causal-conv1d` installed.
- [ ] `mamba-ssm` installed.
- [ ] Verify imports:
  - `python -c "import causal_conv1d, mamba_ssm; print('ok')"`

## 6) Test Gate (Must Pass Before Training)

- [ ] Full tests pass:
  - `python -m pytest`
- [ ] No fail/error in:
  - `PointPillars_module/tests/test_models.py`
  - `PointPillars_module/tests/test_temporal_encoders_pytest.py`
  - `PointPillars_module/tests/test_pointpillars_neck_pytest.py`
- [ ] If there are skips, confirm they are optional paths only (for example `pybullet` dataset env tests when not needed).

## 7) Ready-to-Train Criteria

- [ ] `torch.cuda.is_available()` is `True`.
- [ ] `pointpillars.ops.voxel_op` works on CUDA tensors.
- [ ] `mamba-ssm` import works.
- [ ] Test gate passed.
- [ ] Training command dry run starts without import/build errors.

## 8) Train Commands (Reference)

- Stage A (mamba):
  - `python PointPillars_module/training/train_stage_a_mamba.py --data_root data/stage_a_experiment --ckpt PointPillars_module/pretrained/epoch_160.pth`

---

If a setup breaks, keep the log in `logs/` and compare against this checklist from top to bottom before changing code.
