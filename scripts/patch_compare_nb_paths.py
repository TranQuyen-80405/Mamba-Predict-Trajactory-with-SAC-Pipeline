"""Inject REPO_ROOT auto-detect + npz check into notebooks/compare_trajactory_predict_module.ipynb (cell index 3)."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    p = Path(__file__).resolve().parent.parent / "notebooks" / "compare_trajactory_predict_module.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    src = "".join(nb["cells"][3]["source"])
    if "_detect_repo_root" in src:
        print("already patched:", p)
        return
    old = (
        'REPO_ROOT = "/content/Pipeline"  # đổi nếu clone Drive: /content/drive/MyDrive/.../Pipeline\n'
        'DATA_ROOT = os.path.join(REPO_ROOT, "data", "stage_a_experiment")'
    )
    new = '''def _detect_repo_root() -> str:
    env = os.environ.get("PIPELINE_REPO_ROOT", "").strip()
    if env and os.path.isdir(env):
        return env
    if os.path.isdir("/content/Pipeline"):
        return "/content/Pipeline"
    cwd = os.path.abspath(os.getcwd())
    for base in (cwd, os.path.dirname(cwd)):
        if base and os.path.isdir(os.path.join(base, "PointPillars_module")):
            return base
    return cwd


REPO_ROOT = _detect_repo_root()
DATA_ROOT = os.path.join(REPO_ROOT, "data", "stage_a_experiment")'''
    if old not in src:
        raise SystemExit("pattern not found in cell 3")
    src = src.replace(old, new)
    needle = "assert os.path.isdir(DATA_ROOT)"
    ins = """# Ít nhất một file .npz phải tồn tại (index thường có trong git nhưng .npz có thể chưa upload)
_index = os.path.join(DATA_ROOT, "index.jsonl")
if os.path.isfile(_index):
    import json as _json

    with open(_index, "r", encoding="utf-8") as _f:
        _line = _f.readline().strip()
    if _line:
        _first = _json.loads(_line).get("path", "")
        if _first:
            _np = os.path.join(DATA_ROOT, _first)
            assert os.path.isfile(_np), (
                f"Thiếu file rollout {_np} — cần các .npz cùng thư mục với index. "
                f"Chạy: python run_datagen_preset.py experiment"
            )

"""
    if ins.strip() not in src:
        src = src.replace(needle, ins + needle, 1)
    nb["cells"][3]["source"] = [seg for seg in src.splitlines(keepends=True)]
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print("patched", p)


if __name__ == "__main__":
    main()
