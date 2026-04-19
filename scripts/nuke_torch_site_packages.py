"""Remove broken/orphan torch from current interpreter's site-packages (Windows-friendly chmod rmtree)."""
from __future__ import annotations

import os
import shutil
import site
import stat
import sys
from pathlib import Path


def _onerr(fn, p, _exc) -> None:
    try:
        os.chmod(p, stat.S_IWRITE)
        fn(p)
    except OSError:
        pass


def main() -> int:
    left: list[Path] = []
    for sp in site.getsitepackages():
        root = Path(sp)
        torch_dir = root / "torch"
        if torch_dir.is_dir():
            print("rmtree", torch_dir)
            shutil.rmtree(torch_dir, onerror=_onerr)
        for di in root.glob("torch-*.dist-info"):
            print("rmtree", di)
            shutil.rmtree(di, onerror=_onerr)
        if torch_dir.is_dir():
            dead = root / f"_torch_broken_{os.getpid()}"
            try:
                torch_dir.rename(dead)
                shutil.rmtree(dead, onerror=_onerr)
                print("rename->rmtree", dead)
            except OSError as e:
                print("FAILED still:", torch_dir, e)
                left.append(torch_dir)
    if left:
        print("ERROR: could not remove:", left)
        return 1
    print("OK: no site-packages/torch left")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
