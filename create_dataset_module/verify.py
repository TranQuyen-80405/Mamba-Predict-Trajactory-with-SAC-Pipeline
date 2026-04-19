"""
End-to-end smoke verification for the Stage A data path.

Pipeline exercised:
    DataGenerator.run()        (PyBullet -> Trajectory.npz + index.jsonl)
        -> RiskDataset          (Trajectory.npz -> RiskSample)
        -> collate_riskbatch    (list[RiskSample] -> RiskBatch)
        -> FullPipeline.forward (RiskBatch.pts_seq -> risk logits)   [optional]

The FullPipeline step is skipped automatically when either
  * ``torch.cuda.is_available()`` is False, or
  * the PointPillars checkpoint cannot be located / loaded
    (e.g. a laptop with no compiled voxel_op extension).

The dataset-side path is always executed so that a CPU-only box can still
validate its half of the contract before shipping the folder to Colab.

Run from the repo root as::

    python -m create_dataset_module.verify --tmp_dir ./_verify_tmp

Exit code:
    0   all enabled stages passed
    1   a stage failed (see stderr)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PP_PKG = os.path.join(_ROOT, "PointPillars_module")
for _p in (_ROOT, _PP_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PointPillars_module.types import DataGenConfig  # noqa: E402

from .generator import DataGenerator  # noqa: E402
from .risk_dataset import RiskDataset, collate_riskbatch  # noqa: E402


def _banner(msg: str) -> None:
    bar = "-" * max(10, len(msg) + 2)
    print(f"\n{bar}\n {msg}\n{bar}")


def _small_cfg(out_dir: Path) -> DataGenConfig:
    """A tiny generation plan that finishes in a few seconds on a laptop."""
    return DataGenConfig(
        out_dir=str(out_dir),
        n_scenes=1,
        rollouts_per_scene=1,
        # Must exceed T_ctx (10) + max horizon (40 frames = 2 s) = 50
        # for at least one indexable frame; we pad a little for safety.
        frames_per_rollout=60,
        seed=0,
        depth_hw=(160, 120),
        camera_fov_h_deg=90.0,
        camera_near=0.1,
        camera_far=8.0,
        policy_random_p=1.0,
        policy_scripted_p=0.0,
        policy_adversarial_p=0.0,
    )


def _stage_generate(tmp_dir: Path) -> int:
    _banner("Stage 1/4 : DataGenerator.run()")
    try:
        import pybullet  # noqa: F401
    except ImportError:
        print("  [SKIP] pybullet is not installed; cannot run the generator.")
        return 0

    cfg = _small_cfg(tmp_dir)
    n = DataGenerator(cfg, out_dir=str(tmp_dir)).run()
    if n != cfg.n_scenes * cfg.rollouts_per_scene:
        raise RuntimeError(
            f"expected {cfg.n_scenes * cfg.rollouts_per_scene} rollouts, got {n}"
        )
    index = tmp_dir / "index.jsonl"
    assert index.exists(), f"missing index.jsonl at {index}"
    print(f"  OK  {n} rollout(s) written to {tmp_dir}")
    return n


def _stage_dataset(tmp_dir: Path) -> int:
    _banner("Stage 2/4 : RiskDataset + collate_riskbatch")
    ds = RiskDataset(str(tmp_dir), T_ctx=10)
    if len(ds) == 0:
        raise RuntimeError("RiskDataset is empty; DataGenerator did not write samples.")
    sample = ds[0]
    if len(sample.pts_seq) != 10:
        raise RuntimeError(
            f"RiskSample.pts_seq length mismatch: {len(sample.pts_seq)} != 10"
        )
    batch_size = min(2, len(ds))
    batch = collate_riskbatch([ds[i] for i in range(batch_size)])
    if batch.risk_1s.shape != (batch_size,):
        raise RuntimeError(f"risk_1s shape mismatch: {batch.risk_1s.shape}")
    if len(batch.pts_seq) != 10:
        raise RuntimeError(f"batch.pts_seq length mismatch: {len(batch.pts_seq)}")
    if len(batch.pts_seq[0]) != batch_size:
        raise RuntimeError(
            f"batch.pts_seq inner length mismatch: {len(batch.pts_seq[0])}"
        )
    print(
        f"  OK  len(ds)={len(ds)} | T_ctx=10 | B={batch_size} | "
        f"pts[0][0].shape={tuple(batch.pts_seq[0][0].shape)}"
    )
    return batch_size


def _stage_full_pipeline(tmp_dir: Path, batch_size: int) -> bool:
    _banner("Stage 3/4 : FullPipeline.forward (requires CUDA + checkpoint)")
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA unavailable; cannot instantiate PointPillarsNeckExtractor.")
        return False

    try:
        from PointPillars_module.types import PointPillarsConfig
        from module_pointpillar import PointPillarsNeckExtractor
        from models.full_pipeline import FullPipeline
    except Exception as exc:  # pragma: no cover
        print(f"  [SKIP] PointPillars imports failed: {exc}")
        return False

    try:
        pp_cfg = PointPillarsConfig(device="cuda")
        pp = PointPillarsNeckExtractor(pp_cfg)
    except Exception as exc:
        print(
            "  [SKIP] PointPillarsNeckExtractor failed to initialize "
            f"(missing checkpoint or voxel_op ext?): {exc}"
        )
        return False

    pipe = FullPipeline(pp=pp).to("cuda").eval()
    ds = RiskDataset(str(tmp_dir), T_ctx=10)
    batch = collate_riskbatch([ds[i] for i in range(batch_size)])
    pts_seq_bt = [[p.to("cuda") for p in frame] for frame in batch.pts_seq]
    with torch.no_grad():
        logits = pipe(pts_seq_bt)
    if logits.shape != (batch_size, 3):
        raise RuntimeError(f"FullPipeline returned {logits.shape}, expected ({batch_size}, 3)")
    probs = torch.sigmoid(logits).cpu().numpy()
    print(f"  OK  logits.shape={tuple(logits.shape)} | risk probs sample: {probs[0].tolist()}")
    return True


def _stage_summary(generated: int, batch_size: int, ran_pipeline: bool) -> None:
    _banner("Stage 4/4 : Summary")
    if generated == 0:
        print("  Generator was skipped (pybullet missing). Dataset stage also skipped.")
    else:
        print(f"  Generated rollouts : {generated}")
        print(f"  Collated batch     : B={batch_size}, T_ctx=10")
    if ran_pipeline:
        print("  FullPipeline.forward : PASSED on CUDA.")
    else:
        print("  FullPipeline.forward : SKIPPED (CUDA + checkpoint needed).")
    print("\nverify.py OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tmp_dir",
        default="",
        help="Directory for the smoke dataset. Defaults to a temp folder.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Do not delete --tmp_dir on exit (useful for manual inspection).",
    )
    args = parser.parse_args()

    tmp_ctx = None
    if args.tmp_dir:
        tmp_dir = Path(args.tmp_dir).resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="verify_dataset_")
        tmp_dir = Path(tmp_ctx.name)

    exit_code = 0
    try:
        generated = _stage_generate(tmp_dir)
        batch_size = 0
        ran_pipeline = False
        if generated > 0:
            batch_size = _stage_dataset(tmp_dir)
            ran_pipeline = _stage_full_pipeline(tmp_dir, batch_size)
        _stage_summary(generated, batch_size, ran_pipeline)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        if tmp_ctx is not None and not args.keep:
            tmp_ctx.cleanup()
        elif args.keep:
            print(f"\n(kept {tmp_dir})")
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
