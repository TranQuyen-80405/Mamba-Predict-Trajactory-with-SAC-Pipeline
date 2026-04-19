#!/usr/bin/env python3
"""
Chạy toàn bộ `notebooks/compare_trajactory_predict_module.ipynb` bằng nbclient (giống Run All).

Điều kiện để chạy đến kết quả cuối:
  - GPU + PyTorch CUDA + mamba-ssm (Linux hoặc Colab; Windows native thường không đủ).
  - `data/stage_a_experiment/*.npz` (không chỉ index.jsonl).
  - `PointPillars_module/pretrained/epoch_160.pt` hoặc `.pth`.

Biến môi trường:
  PIPELINE_REPO_ROOT      — gốc repo (mặc định: thư mục cha của `scripts/`).
  PIPELINE_SKIP_COMPARE_SETUP — `1` để bỏ qua cell cài pip đầu tiên (torch+mamba đã sẵn).

Ví dụ:
  python scripts/execute_compare_trajactory_notebook.py
  set PIPELINE_SKIP_COMPARE_SETUP=1 && python scripts/execute_compare_trajactory_notebook.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    os.environ.setdefault("PIPELINE_REPO_ROOT", str(repo))
    nb_path = repo / "notebooks" / "compare_trajactory_predict_module.ipynb"
    out_path = repo / "notebooks" / "compare_trajactory_predict_module.executed.ipynb"

    if not nb_path.is_file():
        print("Không thấy:", nb_path, file=sys.stderr)
        return 2

    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        print("Cài: python -m pip install nbformat nbclient", file=sys.stderr)
        return 3

    nb = nbformat.read(nb_path, as_version=4)

    if os.environ.get("PIPELINE_SKIP_COMPARE_SETUP", "").lower() in ("1", "true", "yes"):
        stub = '''# PIPELINE_SKIP_COMPARE_SETUP=1: skip pip; require working torch CUDA + mamba-ssm
import torch

assert torch.cuda.is_available(), "Need GPU + PyTorch CUDA"
import mamba_ssm  # noqa: F401

print("[setup] SKIP pip | torch", torch.__version__, "|", torch.cuda.get_device_name(0))
'''
        if len(nb.cells) < 2 or nb.cells[1].cell_type != "code":
            print("Cấu trúc notebook không đúng (cell code #1).", file=sys.stderr)
            return 4
        nb.cells[1].source = stub

    client = NotebookClient(
        nb,
        timeout=7200,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(repo)}},
    )
    try:
        client.execute()
    except Exception as exc:
        print(exc, file=sys.stderr)
        nbformat.write(nb, out_path)
        print("Notebook lỗi — đã ghi một phần kết quả:", out_path, file=sys.stderr)
        return 1

    nbformat.write(nb, out_path)
    print("OK — đã chạy xong:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
