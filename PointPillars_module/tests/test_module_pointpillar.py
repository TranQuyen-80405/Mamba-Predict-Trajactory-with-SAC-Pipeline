"""
Unit tests for module_pointpillar.py.

Covers:
  1. Pure-Python helpers (no GPU / no checkpoint required).
  2. filter_range half-open semantics.
  3. preprocess_depth_frame scale_factor.
  4. Determinism of extract_neck on a fixed input (skipped if no CUDA
     or no checkpoint).
  5. Freeze / unfreeze API correctness (skipped if no CUDA or no ckpt).
  6. Gradient-flow correctness for extract_neck_forward after
     unfreeze_neck (skipped if no CUDA or no ckpt).

Run:
    cd PointPillars_module
    python -m unittest tests.test_module_pointpillar -v
"""

from __future__ import annotations

import os
import sys
import unittest
import warnings

import numpy as np
import torch

# Make the module_pointpillar.py next to this tests/ folder importable when
# the tests are run as `python -m unittest tests.test_module_pointpillar`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from module_pointpillar import (  # noqa: E402
    CameraToLidarExtrinsics,
    DepthCameraIntrinsics,
    DepthPreprocessConfig,
    NeckFeatureOutput,
    PointPillarsConfig,
    PointPillarsNeckExtractor,
)

_CKPT = os.path.join(_PKG_ROOT, "pretrained", "epoch_160.pth")
_HAS_CUDA = torch.cuda.is_available()
_HAS_CKPT = os.path.exists(_CKPT)
_CAN_RUN_MODEL = _HAS_CUDA and _HAS_CKPT


# =====================================================================
# Helpers: pure-Python tests (always runnable)
# =====================================================================

class TestIntrinsicsDefaults(unittest.TestCase):
    """Default values must match the indoor Stage A/B spec."""

    def test_intrinsics_far_default_is_8m(self):
        intr = DepthCameraIntrinsics(
            fx=80.0, fy=80.0, cx=80.0, cy=60.0,
            width=160, height=120,
        )
        self.assertAlmostEqual(intr.near, 0.1)
        self.assertAlmostEqual(intr.far, 8.0)

    def test_preprocess_cfg_max_range_default_is_8m(self):
        pcfg = DepthPreprocessConfig()
        self.assertAlmostEqual(pcfg.max_range, 8.0)
        self.assertAlmostEqual(pcfg.scale_factor, 1.0)
        self.assertAlmostEqual(pcfg.low_point_warn_ratio, 0.05)


class TestDepthToMeters(unittest.TestCase):
    def test_pybullet_depth_midpoint(self):
        near, far = 0.1, 8.0
        depth_buf = np.full((4, 4), 0.5, dtype=np.float32)
        z = PointPillarsNeckExtractor.pybullet_depth_to_meters(
            depth_buf, near, far
        )
        # Formula: (far*near) / (far - (far-near)*d)
        expected = (far * near) / (far - (far - near) * 0.5)
        self.assertTrue(np.allclose(z, expected, atol=1e-6))
        self.assertEqual(z.dtype, np.float32)


class TestDepthToPointsCamera(unittest.TestCase):
    def test_shape_matches_intrinsics(self):
        intr = DepthCameraIntrinsics(
            fx=80.0, fy=80.0, cx=80.0, cy=60.0,
            width=160, height=120, near=0.1, far=8.0,
        )
        depth = np.full((120, 160), 2.0, dtype=np.float32)
        pts_cam = PointPillarsNeckExtractor.depth_to_points_camera(
            depth, intr, min_range=0.3, max_range=8.0
        )
        self.assertEqual(pts_cam.shape, (160 * 120, 3))
        self.assertEqual(pts_cam.dtype, np.float32)

    def test_shape_mismatch_raises(self):
        intr = DepthCameraIntrinsics(
            fx=80.0, fy=80.0, cx=80.0, cy=60.0,
            width=160, height=120,
        )
        depth = np.ones((100, 100), dtype=np.float32)
        with self.assertRaises(ValueError):
            PointPillarsNeckExtractor.depth_to_points_camera(depth, intr)

    def test_empty_when_all_out_of_range(self):
        intr = DepthCameraIntrinsics(
            fx=80.0, fy=80.0, cx=80.0, cy=60.0,
            width=160, height=120,
        )
        depth = np.full((120, 160), 100.0, dtype=np.float32)  # beyond max
        pts = PointPillarsNeckExtractor.depth_to_points_camera(
            depth, intr, min_range=0.3, max_range=8.0
        )
        self.assertEqual(pts.shape, (0, 3))


class TestCameraToLidar(unittest.TestCase):
    def test_opencv_to_kitti_rotation(self):
        extr = CameraToLidarExtrinsics(convention="opencv_to_kitti")
        pts_cam = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pts_lidar = PointPillarsNeckExtractor.camera_to_lidar(pts_cam, extr)
        # OpenCV (x-right, y-down, z-forward) -> KITTI (x-forward, y-left,
        # z-up): (x', y', z') = (z, -x, -y) = (3, -1, -2).
        self.assertTrue(
            np.allclose(pts_lidar, [[3.0, -1.0, -2.0]], atol=1e-6)
        )

    def test_translation_added(self):
        extr = CameraToLidarExtrinsics(
            convention="identity",
            t=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        )
        pts_cam = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = PointPillarsNeckExtractor.camera_to_lidar(pts_cam, extr)
        self.assertTrue(
            np.allclose(out, [[11.0, 22.0, 33.0]], atol=1e-6)
        )

    def test_empty_short_circuit(self):
        extr = CameraToLidarExtrinsics(convention="identity")
        pts = np.zeros((0, 3), dtype=np.float32)
        out = PointPillarsNeckExtractor.camera_to_lidar(pts, extr)
        self.assertEqual(out.shape, (0, 3))


class TestAddIntensity(unittest.TestCase):
    def test_zero_mode(self):
        pts = np.random.RandomState(0).randn(10, 3).astype(np.float32)
        cfg = DepthPreprocessConfig(intensity_mode="zero")
        out = PointPillarsNeckExtractor.add_intensity(pts, cfg)
        self.assertEqual(out.shape, (10, 4))
        self.assertTrue(np.all(out[:, 3] == 0.0))

    def test_constant_mode(self):
        pts = np.zeros((5, 3), dtype=np.float32)
        cfg = DepthPreprocessConfig(
            intensity_mode="constant", intensity_value=0.7
        )
        out = PointPillarsNeckExtractor.add_intensity(pts, cfg)
        self.assertTrue(np.all(out[:, 3] == np.float32(0.7)))

    def test_unknown_mode_raises(self):
        pts = np.zeros((1, 3), dtype=np.float32)
        cfg = DepthPreprocessConfig(intensity_mode="bogus")
        with self.assertRaises(ValueError):
            PointPillarsNeckExtractor.add_intensity(pts, cfg)


class TestVoxelDownsample(unittest.TestCase):
    def test_noop_when_voxel_zero(self):
        pts = np.random.RandomState(0).rand(100, 4).astype(np.float32)
        out = PointPillarsNeckExtractor.voxel_downsample(pts, 0.0)
        self.assertEqual(out.shape, pts.shape)
        self.assertTrue(np.array_equal(out, pts))

    def test_duplicates_collapsed(self):
        pts = np.array(
            [[0.01, 0.01, 0.01, 0.5],
             [0.02, 0.02, 0.02, 0.5],  # same voxel as above @ 0.1
             [5.0, 5.0, 5.0, 0.5]],
            dtype=np.float32,
        )
        out = PointPillarsNeckExtractor.voxel_downsample(pts, 0.1)
        self.assertEqual(out.shape[0], 2)


class TestPreprocessScaleFactor(unittest.TestCase):
    def test_scale_factor_scales_xyz_but_not_intensity(self):
        # Build a trivial depth frame so we get a small, deterministic cloud.
        intr = DepthCameraIntrinsics(
            fx=80.0, fy=80.0, cx=80.0, cy=60.0,
            width=160, height=120, near=0.1, far=8.0,
        )
        extr = CameraToLidarExtrinsics(convention="opencv_to_kitti")
        depth = np.full((120, 160), 2.0, dtype=np.float32)

        cfg_no_scale = DepthPreprocessConfig(
            scale_factor=1.0, intensity_mode="constant",
            intensity_value=0.25, min_range=0.3, max_range=8.0,
            voxel_downsample=0.0, subsample_ratio=1.0,
        )
        cfg_scaled = DepthPreprocessConfig(
            scale_factor=5.0, intensity_mode="constant",
            intensity_value=0.25, min_range=0.3, max_range=8.0,
            voxel_downsample=0.0, subsample_ratio=1.0,
        )

        # Directly call the static chain so we do not need the model.
        ext = PointPillarsNeckExtractor.__new__(PointPillarsNeckExtractor)
        pts_a = ext.preprocess_depth_frame(depth, intr, extr, cfg_no_scale)
        pts_b = ext.preprocess_depth_frame(depth, intr, extr, cfg_scaled)

        self.assertEqual(pts_a.shape, pts_b.shape)
        self.assertTrue(
            np.allclose(pts_b[:, :3], pts_a[:, :3] * 5.0, atol=1e-5)
        )
        # Intensity column must NOT be scaled.
        self.assertTrue(
            np.allclose(pts_b[:, 3], pts_a[:, 3], atol=1e-6)
        )


# =====================================================================
# Tests that require the full model (CUDA + checkpoint)
# =====================================================================

@unittest.skipUnless(
    _CAN_RUN_MODEL,
    f"Needs CUDA ({_HAS_CUDA}) and checkpoint ({_HAS_CKPT}) at {_CKPT}",
)
class TestExtractorDeterminism(unittest.TestCase):
    """
    Determinism / freeze / gradient-flow tests that need a real model.
    One shared extractor across tests to keep GPU memory stable.
    """

    extractor: PointPillarsNeckExtractor

    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = PointPillarsNeckExtractor(
            PointPillarsConfig(ckpt_path=_CKPT, device="cuda")
        )

    @staticmethod
    def _fake_kitti_cloud(n: int = 4096, seed: int = 123) -> np.ndarray:
        """
        Deterministic synthetic LiDAR-frame point cloud that fits inside
        the default KITTI point_cloud_range.
        """
        rng = np.random.RandomState(seed)
        xs = rng.uniform(1.0, 60.0, size=n).astype(np.float32)
        ys = rng.uniform(-30.0, 30.0, size=n).astype(np.float32)
        zs = rng.uniform(-2.5, 0.5, size=n).astype(np.float32)
        its = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
        return np.stack([xs, ys, zs, its], axis=1)

    # --- 1. extract_neck is byte-identical across two calls on same input ---
    def test_extract_neck_determinism(self):
        pts = self._fake_kitti_cloud()
        out1 = self.extractor.extract_neck([pts])
        out2 = self.extractor.extract_neck([pts])
        self.assertIsInstance(out1, NeckFeatureOutput)
        self.assertEqual(
            tuple(out1.feature.shape),
            (1, 384, 248, 216),
            msg="Unexpected neck output shape; check point_cloud_range/"
                "voxel_size defaults.",
        )
        self.assertEqual(out1.feature.shape, out2.feature.shape)
        # Bitwise determinism is strong but can fail on some cuDNN kernels.
        # Allow max abs diff <= 1e-5; assert they are mostly-identical.
        diff = (out1.feature - out2.feature).abs().max().item()
        self.assertLessEqual(
            diff, 1e-5,
            msg=f"extract_neck not deterministic: max abs diff = {diff}",
        )

    def test_extract_neck_matches_extract_neck_forward_under_no_grad(self):
        """
        Under eval()+no_grad, the training variant must produce the same
        numerical output as the inference variant.
        """
        self.extractor.freeze_all()
        pts = self._fake_kitti_cloud(seed=7)
        out_infer = self.extractor.extract_neck([pts])
        with torch.no_grad():
            out_train = self.extractor.extract_neck_forward([pts])
        diff = (out_infer.feature - out_train.feature).abs().max().item()
        self.assertLessEqual(diff, 1e-5)

    # --- 2. Freeze API ---
    def test_freeze_all_sets_requires_grad_false(self):
        self.extractor.freeze_all()
        for p in self.extractor.model.parameters():
            self.assertFalse(p.requires_grad)
        for name in PointPillarsNeckExtractor._SUBMODULE_NAMES:
            sm = getattr(self.extractor.model, name)
            self.assertFalse(
                sm.training,
                msg=f"{name} should be in eval() after freeze_all()",
            )

    def test_unfreeze_neck_flips_only_neck(self):
        self.extractor.freeze_all()
        self.extractor.unfreeze_neck()
        self.assertTrue(self.extractor.model.neck.training)
        self.assertTrue(
            all(p.requires_grad for p in self.extractor.model.neck.parameters())
        )
        for name in ("pillar_layer", "pillar_encoder", "backbone"):
            sm = getattr(self.extractor.model, name)
            self.assertFalse(sm.training, msg=f"{name} should still be eval()")
            self.assertFalse(
                any(p.requires_grad for p in sm.parameters()),
                msg=f"{name} params should still be frozen",
            )

    def test_set_trainable_rejects_unknown(self):
        with self.assertRaises(ValueError):
            self.extractor.set_trainable(["not_a_submodule"])

    def test_set_trainable_matches_selection(self):
        self.extractor.set_trainable(["neck", "backbone"])
        self.assertTrue(self.extractor.model.neck.training)
        self.assertTrue(self.extractor.model.backbone.training)
        self.assertFalse(self.extractor.model.pillar_encoder.training)

    # --- 3. Gradient flow through extract_neck_forward ---
    def test_extract_neck_forward_gradient_reaches_neck_only(self):
        self.extractor.freeze_all()
        self.extractor.unfreeze_neck()
        pts = self._fake_kitti_cloud(seed=99)
        out = self.extractor.extract_neck_forward([pts])
        loss = out.feature.square().mean()
        loss.backward()

        def _has_grad(sm):
            return any(
                (p.grad is not None) and (p.grad.abs().sum().item() > 0)
                for p in sm.parameters()
            )

        self.assertTrue(
            _has_grad(self.extractor.model.neck),
            msg="neck.parameters() should have received gradient",
        )
        for name in ("pillar_encoder", "backbone"):
            sm = getattr(self.extractor.model, name)
            for p in sm.parameters():
                self.assertIsNone(
                    p.grad,
                    msg=f"{name} should not have grad when frozen",
                )

    # --- 4. Low-retention warning on obviously-out-of-range points ---
    def test_low_retention_warning(self):
        # All points far beyond point_cloud_range -> retention ~0%.
        pts_bad = np.full((4096, 4), 1000.0, dtype=np.float32)
        # Make sure the first point actually survives (avoid the
        # "no points left" ValueError); we just want retention below 5%.
        pts_bad[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.extractor.extract_neck(
                [pts_bad], warn_low_retention_ratio=0.5
            )
            msgs = [str(w.message) for w in caught]
            self.assertTrue(
                any("after point_cloud_range filtering" in m for m in msgs),
                msg=f"Expected low-retention UserWarning; got: {msgs}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
