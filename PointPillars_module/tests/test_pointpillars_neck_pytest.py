"""
Pytest suite: point-cloud preprocessing, neck feature shape, freeze flags,
empty-input robustness, CPU vs CUDA consistency.

Run from repo root or PointPillars_module:
    pip install -r requirements-dev.txt
    cd PointPillars_module && pytest tests/test_pointpillars_neck_pytest.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from module_pointpillar import (  # noqa: E402
    CameraToLidarExtrinsics,
    DepthCameraIntrinsics,
    DepthPreprocessConfig,
    PointPillarsConfig,
    PointPillarsNeckExtractor,
)
from pretrained_ckpt_resolve import resolve_pointpillars_ckpt  # noqa: E402

CKPT_PATH = resolve_pointpillars_ckpt(_PKG_ROOT)
HAS_CKPT = CKPT_PATH is not None
HAS_CUDA = torch.cuda.is_available()

_EXPECTED_NECK_SHAPE = (384, 248, 216)


def _fake_lidar_tensor(n: int, seed: int) -> torch.Tensor:
    """LiDAR-frame points inside default point_cloud_range (torch)."""
    g = torch.Generator().manual_seed(seed)
    xs = torch.rand((n,), generator=g, dtype=torch.float32) * 59.0 + 1.0
    ys = torch.rand((n,), generator=g, dtype=torch.float32) * 60.0 - 30.0
    zs = torch.rand((n,), generator=g, dtype=torch.float32) * 3.0 - 2.5
    its = torch.rand((n,), generator=g, dtype=torch.float32)
    return torch.stack([xs, ys, zs, its], dim=1)


# ---------------------------------------------------------------------------
# 1) Point cloud preprocessing (pure + integration)
# ---------------------------------------------------------------------------


class TestPointCloudPreprocessing:
    def test_camera_to_lidar_extrinsics_opencv_to_kitti(self) -> None:
        extr = CameraToLidarExtrinsics(convention="opencv_to_kitti")
        pts_cam = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pts_lidar = PointPillarsNeckExtractor.camera_to_lidar(pts_cam, extr)
        expected = torch.tensor([[3.0, -1.0, -2.0]], dtype=torch.float32)
        torch.testing.assert_close(
            torch.from_numpy(pts_lidar), expected, rtol=0, atol=1e-6
        )

    def test_camera_to_lidar_custom_R_and_t(self) -> None:
        R = np.eye(3, dtype=np.float32)
        t = np.array([0.5, -1.0, 2.0], dtype=np.float32)
        extr = CameraToLidarExtrinsics(R=R, t=t)
        pts_cam = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        out = PointPillarsNeckExtractor.camera_to_lidar(pts_cam, extr)
        torch.testing.assert_close(
            torch.from_numpy(out),
            torch.tensor([[1.5, -1.0, 2.0]], dtype=torch.float32),
            rtol=0,
            atol=1e-6,
        )

    def test_preprocess_scale_factor_xyz_not_intensity(self) -> None:
        intr = DepthCameraIntrinsics(
            fx=80.0, fy=80.0, cx=80.0, cy=60.0,
            width=160, height=120, near=0.1, far=8.0,
        )
        extr = CameraToLidarExtrinsics(convention="opencv_to_kitti")
        depth = np.full((120, 160), 2.0, dtype=np.float32)
        ext = PointPillarsNeckExtractor.__new__(PointPillarsNeckExtractor)
        cfg_a = DepthPreprocessConfig(
            scale_factor=1.0,
            intensity_mode="constant",
            intensity_value=0.25,
            min_range=0.3,
            max_range=8.0,
            voxel_downsample=0.0,
            subsample_ratio=1.0,
        )
        cfg_b = DepthPreprocessConfig(
            scale_factor=6.0,
            intensity_mode="constant",
            intensity_value=0.25,
            min_range=0.3,
            max_range=8.0,
            voxel_downsample=0.0,
            subsample_ratio=1.0,
        )
        pts_a = ext.preprocess_depth_frame(depth, intr, extr, cfg_a)
        pts_b = ext.preprocess_depth_frame(depth, intr, extr, cfg_b)
        torch.testing.assert_close(
            torch.from_numpy(pts_b[:, :3]),
            torch.from_numpy(pts_a[:, :3]) * 6.0,
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            torch.from_numpy(pts_b[:, 3]),
            torch.from_numpy(pts_a[:, 3]),
            rtol=0,
            atol=1e-6,
        )

    @pytest.mark.needs_model
    def test_list_of_tensors_variable_counts_end_to_end(self) -> None:
        if not HAS_CKPT:
            pytest.skip(f"checkpoint missing: {CKPT_PATH}")
        device = "cuda" if HAS_CUDA else "cpu"
        cfg = PointPillarsConfig(ckpt_path=CKPT_PATH, device=device)
        extractor = PointPillarsNeckExtractor(cfg)
        batch = [
            _fake_lidar_tensor(256, seed=1),
            _fake_lidar_tensor(4096, seed=2),
            _fake_lidar_tensor(128, seed=3),
        ]
        out = extractor.extract_neck(batch)
        assert out.batch_size == 3
        assert out.feature.shape == (3, *_EXPECTED_NECK_SHAPE)
        assert out.feature.dtype == torch.float32


# ---------------------------------------------------------------------------
# 2) Neck extraction + freeze
# ---------------------------------------------------------------------------


@pytest.mark.needs_model
class TestNeckExtractionAndFreeze:
    @pytest.fixture(scope="class")
    def extractor(self) -> PointPillarsNeckExtractor:
        if not HAS_CKPT:
            pytest.skip(f"checkpoint missing: {CKPT_PATH}")
        device = "cuda" if HAS_CUDA else "cpu"
        return PointPillarsNeckExtractor(
            PointPillarsConfig(ckpt_path=CKPT_PATH, device=device)
        )

    def test_extract_neck_output_shape(
        self, extractor: PointPillarsNeckExtractor
    ) -> None:
        pts = _fake_lidar_tensor(2048, seed=42)
        out = extractor.extract_neck([pts])
        assert out.feature.shape == (1, *_EXPECTED_NECK_SHAPE)

    def test_freeze_pillar_layer_and_backbone_no_grad(
        self, extractor: PointPillarsNeckExtractor
    ) -> None:
        extractor.freeze_all()
        extractor.set_trainable(["neck"])
        m = extractor.model
        backbone_grad = [p.requires_grad for p in m.backbone.parameters()]
        assert backbone_grad, "backbone must have parameters"
        assert not any(backbone_grad)
        enc_grad = [p.requires_grad for p in m.pillar_encoder.parameters()]
        assert enc_grad
        assert not any(enc_grad)
        assert any(p.requires_grad for p in m.neck.parameters())
        pillar_params = list(m.pillar_layer.parameters())
        assert not pillar_params, "PillarLayer has no trainable parameters"


# ---------------------------------------------------------------------------
# 3) Robustness
# ---------------------------------------------------------------------------


@pytest.mark.needs_model
class TestRobustness:
    @pytest.fixture(scope="class")
    def extractor(self) -> PointPillarsNeckExtractor:
        if not HAS_CKPT:
            pytest.skip(f"checkpoint missing: {CKPT_PATH}")
        device = "cuda" if HAS_CUDA else "cpu"
        return PointPillarsNeckExtractor(
            PointPillarsConfig(ckpt_path=CKPT_PATH, device=device)
        )

    def test_empty_point_cloud_returns_zero_bev(
        self, extractor: PointPillarsNeckExtractor
    ) -> None:
        empty = torch.zeros((0, 4), dtype=torch.float32)
        out = extractor.extract_neck([empty])
        assert out.feature.shape == (1, *_EXPECTED_NECK_SHAPE)
        torch.testing.assert_close(
            out.feature,
            torch.zeros_like(out.feature),
            rtol=0,
            atol=0,
        )

    @pytest.mark.needs_cuda
    def test_cpu_cuda_feature_consistency(self) -> None:
        if not HAS_CKPT or not HAS_CUDA:
            pytest.skip("needs CUDA and checkpoint")
        pts_np = _fake_lidar_tensor(1024, seed=99).cpu().numpy()
        ext_cpu = PointPillarsNeckExtractor(
            PointPillarsConfig(ckpt_path=CKPT_PATH, device="cpu")
        )
        ext_cuda = PointPillarsNeckExtractor(
            PointPillarsConfig(ckpt_path=CKPT_PATH, device="cuda")
        )
        out_cpu = ext_cpu.extract_neck([pts_np])
        out_cuda = ext_cuda.extract_neck([pts_np])
        torch.testing.assert_close(
            out_cuda.feature.cpu(),
            out_cpu.feature,
            # CUDA voxelization/backbone kernels can introduce small
            # accumulation differences versus CPU.
            rtol=1e-3,
            atol=1e-2,
        )
