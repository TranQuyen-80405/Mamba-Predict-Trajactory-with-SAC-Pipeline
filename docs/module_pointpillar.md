# `module_pointpillar.py` — Technical Specification

> This document specifies every data type, input/output, function logic and structure of `module_pointpillar.py`.
> **Every code change to `module_pointpillar.py` MUST be reflected here (see the "Changelog" section at the bottom).**
>
> **Before editing this doc, read § 12 "Maintenance rule — keep these docs in sync".** Any change here that affects pipeline architecture, data contracts, or integration points MUST be mirrored in `strategy_full_pipeline.md` and `strategy_finetune_with_SAC.md` in the same PR.

---

## 1. Intent

Wrap the pretrained **PointPillars** model (KITTI, 3 classes) behind a small API that:

1. Accepts **one or many** point clouds of shape `(N, 4)` = `[x, y, z, intensity]`.
2. Runs the forward pass **only up to the `neck` layer** (skipping head / NMS) — saving compute when only BEV features are needed.
3. Returns a **BEV feature map** `(B, 384, H, W)` float32 on CUDA, to be consumed by downstream pipelines (PyBullet, RL, classifier, etc.) as an encoder.

It also ships a set of helpers that turn a **depth buffer** (PyBullet / OpenCV camera) into a `(N, 4)` point cloud in a LiDAR-style frame, ready to feed step 1.

---

## 2. Environment & companion files

### 2.1. Dependencies (lean stack)

Only 3 runtime dependencies. The heavy stack (`numba`, `open3d`, `opencv-python`, `PyYAML`, `tqdm`) that older versions of this module required was removed together with `pointpillars/utils/` — see Changelog v0.4.

| Package | Version                  | Note                                      |
|---------|--------------------------|-------------------------------------------|
| python  | 3.9 – 3.12               |                                           |
| torch   | any build matching nvcc  | e.g. `2.3.1+cu121` with CUDA toolkit 12.1 |
| numpy   | any, compatible with torch |                                         |
| ninja   | ≥ 1.11                   | required once to build `voxel_op`         |

Conflict-avoidance rules:
- The torch wheel CUDA major/minor MUST match the nvcc used to build `pointpillars/ops/voxel_op`.
- If `CUDAExtension` build fails, pin `setuptools < 70`.

### 2.2. Files that must exist at runtime

| Role | Path | Note |
|---|---|---|
| Model code | `pointpillars/model/` | Only `PointPillars` (head / anchors removed) |
| Built CUDA op | `pointpillars/ops/voxel_op*` (`.so` / `.pyd`) | Only voxelization; `iou3d_op` removed |
| Pretrained weights | `pretrained/epoch_160.pth` | KITTI, trained with full detector; head/anchor keys are filtered on load |

---

## 3. File structure

```
module_pointpillar.py
├── Header comment: env + stack + install + companion files
├── Imports: os, dataclass, typing, numpy, torch, PointPillars
├── Dataclasses (SPEC)
│     ├── PointPillarsConfig         # model hyper-params + runtime
│     ├── PointCloudInput            # single-frame wrapper
│     ├── NeckFeatureOutput          # output wrapper
│     ├── DepthCameraIntrinsics      # pinhole intrinsics
│     ├── CameraToLidarExtrinsics    # R, t, convention
│     └── DepthPreprocessConfig      # domain-gap mitigations
└── class PointPillarsNeckExtractor
      ├── __init__                   # build + load weights + eval()
      ├── _build_model               # construct PointPillars on device
      ├── _load_weights              # load the .pth checkpoint
      ├── as_tensor                  [staticmethod]
      ├── filter_range               # half-open [x0, x1) convention
      ├── _warn_low_retention        [staticmethod]  # OOD smoke check
      ├── freeze_all                                 # Stage A1 / Stage B S1
      ├── unfreeze_neck                              # Stage A2 / S2
      ├── set_trainable(names)                       # fine-grained freeze
      ├── trainable_parameters                       # returns list of params
      ├── pybullet_depth_to_meters   [staticmethod]
      ├── depth_to_points_camera     [staticmethod]
      ├── camera_to_lidar            [staticmethod]
      ├── add_intensity              [staticmethod]
      ├── voxel_downsample           [staticmethod]
      ├── preprocess_depth_frame     # applies scale_factor (indoor hack)
      ├── _in_range_dummy_point      # placeholder for empty frames (voxel op)
      ├── _prepare_batch             # normalize + range-filter + warn + empty flags
      ├── _run_neck                  # core forward (no_grad-free body)
      ├── extract_neck               # @torch.no_grad() — Stage B streaming
      └── extract_neck_forward       # NO no_grad — Stage A training
```

---

## 4. Dataclass specification (detailed types)

### 4.1. `PointPillarsConfig`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `nclasses` | `int` | `3` | Number of head classes (unused when only extracting neck). |
| `voxel_size` | `List[float]` (len=3) | `[0.16, 0.16, 4.0]` | Voxel edge (m) along `[x, y, z]`. |
| `point_cloud_range` | `List[float]` (len=6) | `[0, -39.68, -3, 69.12, 39.68, 1]` | `[x0, y0, z0, x1, y1, z1]` in meters. |
| `max_num_points` | `int` | `32` | Max points per pillar. |
| `max_voxels_train` | `int` | `16000` | |
| `max_voxels_test`  | `int` | `40000` | |
| `ckpt_path` | `str` | `"pretrained/epoch_160.pth"` | |
| `device` | `str` | `"cuda"` | `"cuda"` or `"cpu"` (CUDA ops require `"cuda"`). |
| `max_voxels` (property) | `Tuple[int, int]` | — | `(train, test)`, forwarded directly into `PointPillars`. |

### 4.2. `PointCloudInput`

| Field | Type | Shape / constraint |
|---|---|---|
| `points` | `torch.Tensor` | `(N, 4)` float32, columns `[x, y, z, intensity]` |
| `frame_id` | `Optional[str]` | Optional metadata |

This dataclass only standardizes the input spec if you want a typed container; `extract_neck` itself still accepts raw `np.ndarray` / `torch.Tensor`.

### 4.3. `NeckFeatureOutput`

| Field | Type | Value / Shape |
|---|---|---|
| `feature` | `torch.Tensor` | `(B, C=384, H, W)` float32, on CUDA when `config.device="cuda"` |
| `batch_size` | `int` | `= B` |
| `channels` | `int` | `= 384` with default config (128+128+128 from neck) |
| `height` | `int` | `= 248` with default config |
| `width`  | `int` | `= 216` with default config |
| `device` | `str` | e.g. `"cuda:0"` |

> `H, W` are derived from `point_cloud_range` / `voxel_size`, then backbone stride 2 and neck upsample — defaults yield `(248, 216)`.

### 4.4. `DepthCameraIntrinsics`

| Field | Type | Unit | Note |
|---|---|---|---|
| `fx`, `fy` | `float` | pixel | Focal length. |
| `cx`, `cy` | `float` | pixel | Principal point. |
| `width`, `height` | `int` | pixel | Must match `depth.shape`. |
| `near` | `float` | m | Default `0.1`. |
| `far` | `float` | m | Default `8.0` (indoor; matches `DataGenConfig` / `EnvConfig` in `strategy_full_pipeline.md`). Override for outdoor / KITTI. |

### 4.5. `CameraToLidarExtrinsics`

`p_lidar = R @ p_cam + t`

| Field | Type | Shape | Default | Note |
|---|---|---|---|---|
| `R` | `Optional[np.ndarray]` | `(3, 3)` float32 | `None` | If `None`, derived from `convention`. |
| `t` | `np.ndarray` | `(3,)` float32 | `[0, 0, 0]` | Camera origin position in the LiDAR frame (m). |
| `convention` | `str` | — | `"opencv_to_kitti"` | `"opencv_to_kitti"` \| `"pybullet_to_kitti"` \| `"identity"` |

Preset rotation matrices:

| Convention | Source frame (camera) | `R` matrix |
|---|---|---|
| `opencv_to_kitti` | x-right, y-down, z-forward | `[[0,0,1],[-1,0,0],[0,-1,0]]` |
| `pybullet_to_kitti` | x-right, y-up, z-backward | `[[0,0,-1],[-1,0,0],[0,1,0]]` |
| `identity` | already in KITTI frame | `I₃` |

> Target frame is always KITTI-style: x-forward, y-left, z-up.

### 4.6. `DepthPreprocessConfig`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `intensity_mode` | `str` | `"zero"` | `"zero"` \| `"constant"` \| `"normalized_range"` |
| `intensity_value` | `float` | `0.0` | Used when `intensity_mode="constant"`. |
| `voxel_downsample` | `float` | `0.0` | Voxel edge in meters; `0` disables. |
| `min_range` | `float` | `0.3` | Drop points with `z ≤ min_range` (camera frame). |
| `max_range` | `float` | `8.0` | Drop points with `z ≥ max_range`. Indoor default; override for outdoor. |
| `subsample_ratio` | `float` | `1.0` | ∈ `(0, 1]`; `1.0` keeps all points. |
| `random_seed` | `Optional[int]` | `None` | For reproducible subsampling. |
| `scale_factor` | `float` | `1.0` | **Indoor-scale hack** (`strategy_full_pipeline.md` § 4.1): multiplies LiDAR-frame `(x, y, z)` by a constant before voxelization. Use `5.0 – 10.0` so depth-camera points fill KITTI `point_cloud_range`. `1.0` disables. |
| `low_point_warn_ratio` | `float` | `0.05` | If the retained fraction after `filter_range` drops below this in `extract_neck*`, emit a `UserWarning`. Set `0.0` to silence. |

---

## 5. Function specification (input / output / logic)

### 5.1. `PointPillarsNeckExtractor.__init__(config=None)`

- **Input:** `config: Optional[PointPillarsConfig]`.
- **Side effects:**
  1. Sets `self.config` (uses default if `None`).
  2. Sets `self.device: torch.device`.
  3. Calls `_build_model()` → `self.model` already on `device`.
  4. Calls `_load_weights()` → loads `state_dict` from `ckpt_path`.
  5. `self.model.eval()`.

### 5.2. `_build_model() -> PointPillars`

- Builds `PointPillars(nclasses, voxel_size, point_cloud_range, max_num_points, max_voxels)` and moves it to `device`.

### 5.3. `_load_weights() -> None`

- **Raises:** `FileNotFoundError` if `ckpt_path` does not exist.
- **Behavior:** load the `.pth`, **filter the state_dict**, then `load_state_dict(state, strict=False)`.
- **Why filtering is required:** the shipped checkpoint (`pretrained/epoch_160.pth`) was trained on the full KITTI detector that had:
  - `head.conv_cls.*`, `head.conv_reg.*`, `head.conv_dir_cls.*` — the detection head
  - `anchors_generator.*` — anchor buffers
  - plus other submodule state kept only for training

  These submodules were **deleted** from `PointPillars` during the v0.4 strip (see `pointpillars/model/pointpillars.py`). Loading the raw state_dict with `strict=True` would raise `Unexpected key(s)` for every removed key.
- **Key-filter rule:** keep only entries whose key starts with one of
  ```
  "pillar_layer."   "pillar_encoder."   "backbone."   "neck."
  ```
  All other keys (head, anchors_generator, and anything else not in the stripped model) are dropped.
- **Pseudocode:**
  ```python
  state = torch.load(ckpt, map_location=self.device)
  kept = ("pillar_layer.", "pillar_encoder.", "backbone.", "neck.")
  state = {k: v for k, v in state.items() if k.startswith(kept)}
  self.model.load_state_dict(state, strict=False)
  ```

### 5.4. `as_tensor(pc) -> torch.Tensor` *(staticmethod)*

- **Input:** `np.ndarray` or `torch.Tensor`, shape must be `(N, 4)`.
- **Output:** `torch.Tensor (N, 4)` float32 **contiguous**, device unchanged.
- **Raises:** `TypeError` on wrong type; `ValueError` on wrong shape.
- **Logic:**
  1. `ndarray` → `torch.from_numpy(astype(float32))`.
  2. `Tensor` → `.float()`.
  3. Assert `ndim == 2 and size(1) == 4`.
  4. `.contiguous()`.

### 5.5. `filter_range(pts) -> torch.Tensor`

- **Input:** `torch.Tensor (N, 4)` on the same device as the model.
- **Output:** `torch.Tensor (M, 4)`, `M ≤ N`.
- **Logic:** Boolean mask with 6 conditions `x0 <= x < x1`, same for `y` and `z` (half-open `[x0, x1) × [y0, y1) × [z0, z1)`, matching KITTI / SECOND voxelization semantics); returns `pts[mask]`. Purpose: strip points outside `point_cloud_range` **before** voxelization to avoid the degenerate `num_points=0` case that crashes CUDA kernels.

### 5.5b. `_warn_low_retention(kept, total, ratio_threshold, frame_idx=None)` *(staticmethod)*

- Emits a `UserWarning` when `kept / total < ratio_threshold` after range filtering. `ratio_threshold <= 0` or `total == 0` → no-op. Used internally by `extract_neck*` via `_prepare_batch`, threshold comes from the `warn_low_retention_ratio` arg (default `0.05`). The message hints at likely causes: wrong convention, wrong `extrinsics.t`, or missing `scale_factor`.

### 5.5c. Freeze API

Three methods map 1:1 to the freeze regimes defined in `strategy_finetune_with_SAC.md` § 4. `PillarLayer` has no trainable parameters (pure voxelization), but we still flip its `.eval()` flag for consistency and to keep downstream BN-lock hooks happy.

| Method | Sets `requires_grad` on | `.eval()` / `.train()` | Use case |
|---|---|---|---|
| `freeze_all()` | every submodule → `False` | all → `.eval()` | Stage A1 warm-up; Stage B default regime S1. |
| `unfreeze_neck()` | `pp.neck` → `True` (others unchanged) | `pp.neck` → `.train()` | Stage A2 after `freeze_all()`. Also the Stage B-plus S2 baseline. |
| `set_trainable(names)` | each submodule in `names` → `True`, others → `False` | trainable → `.train()`, frozen → `.eval()` | Fine-grained; e.g. `{"neck", "backbone"}` for S3-lite. Raises `ValueError` on unknown names. |
| `trainable_parameters()` | — | — | Returns `List[nn.Parameter]` with `requires_grad=True`; convenient for building an optimizer param group. |

Valid submodule names (from `_SUBMODULE_NAMES`): `pillar_layer`, `pillar_encoder`, `backbone`, `neck`.

### 5.6. `pybullet_depth_to_meters(depth_buffer, near, far)` *(staticmethod)*

- **Input:**
  - `depth_buffer: np.ndarray (H, W)` float, values in `[0, 1]` (PyBullet's normalized depth).
  - `near, far: float` (m).
- **Output:** `np.ndarray (H, W)` float32, metric depth (m).
- **Logic:** `z = (far * near) / (far - (far - near) * depth)` — matches `pybullet.getCameraImage` documentation.

### 5.7. `depth_to_points_camera(depth_meters, intrinsics, min_range=0.3, max_range=40.0)` *(staticmethod)*

- **Input:**
  - `depth_meters: np.ndarray (H, W)` float32 (m).
  - `intrinsics: DepthCameraIntrinsics`.
  - `min_range, max_range: float` (m).
- **Output:** `np.ndarray (N, 3)` float32 in the **camera frame** (x-right, y-down, z-forward for OpenCV-style intrinsics).
- **Raises:** `ValueError` if `depth.shape` ≠ `(intrinsics.height, intrinsics.width)`.
- **Logic:**
  1. Mask `valid = finite & (z > min_range) & (z < max_range)`; if empty → return zeros `(0, 3)`.
  2. Build pixel grids `(u, v)`.
  3. `x = (u - cx) * z / fx`, `y = (v - cy) * z / fy`, keep `z`.
  4. `stack → (N, 3) float32`.

### 5.8. `camera_to_lidar(pts_cam, extrinsics)` *(staticmethod)*

- **Input:** `pts_cam: np.ndarray (N, 3)` float32; `extrinsics: CameraToLidarExtrinsics`.
- **Output:** `np.ndarray (N, 3)` float32 in the LiDAR frame.
- **Logic:** `p_lidar = pts_cam @ R.T + t`. Short-circuits when `N == 0`.

### 5.9. `add_intensity(pts_xyz, cfg)` *(staticmethod)*

- **Input:** `(N, 3)` float32; `cfg: DepthPreprocessConfig`.
- **Output:** `(N, 4)` float32, column 4 is synthesized intensity.
- **Raises:** `ValueError` on unknown `intensity_mode`.
- **Logic by `intensity_mode`:**
  - `"zero"`     → column of `0.0`.
  - `"constant"` → column of `cfg.intensity_value`.
  - `"normalized_range"` → `1 - clip(||xyz|| / max_range, 0, 1)` (near = bright).

### 5.10. `voxel_downsample(pts, voxel)` *(staticmethod)*

- **Input:** `(N, ≥3)` float32; `voxel: float` (m).
- **Output:** `(M, D)` with `M ≤ N` (first-point-per-voxel).
- **Logic:**
  1. `keys = floor(pts[:, :3] / voxel)`.
  2. `np.unique(keys, axis=0, return_index=True)` keeps first-seen index.
  3. Return `pts[sorted_idx]`.
- **No-op** when `voxel ≤ 0` or `pts` is empty.

### 5.11. `preprocess_depth_frame(depth_meters, intrinsics, extrinsics, cfg=None) -> np.ndarray`

- **Input:**
  - `depth_meters: (H, W)` float32 (m).
  - `intrinsics, extrinsics`.
  - `cfg: Optional[DepthPreprocessConfig]`.
- **Output:** `(N, 4)` float32 in the LiDAR frame, ready for `extract_neck`.
- **Pipeline:**
  ```
  depth_meters
    ├─ depth_to_points_camera        → (N, 3) camera frame
    ├─ camera_to_lidar               → (N, 3) lidar frame
    ├─ scale xyz by cfg.scale_factor → (N, 3) lidar frame, inflated for
    │                                   indoor-scale hack (§ 4.1 of the
    │                                   full-pipeline doc). No-op if == 1.0.
    ├─ add_intensity                 → (N, 4)
    ├─ voxel_downsample (if cfg)     → (M, 4)
    └─ random subsample (if cfg)     → (K, 4)
  ```
- The `scale_factor` is applied **before** `add_intensity` / `voxel_downsample` so the voxel grid operates on the inflated coordinates. Intensity column is never scaled.

### 5.12. `extract_neck(batched_pts, do_range_filter=True, warn_low_retention_ratio=0.05) -> NeckFeatureOutput`  *(inference API, `@torch.no_grad()`)*

- **Use case:** Stage B streaming / inference at 20 Hz.
- **Input:**
  - `batched_pts: List[np.ndarray | torch.Tensor]`, each element shaped `(N_i, 4)` float32.
  - `do_range_filter: bool` — whether to clip points to `point_cloud_range` before voxelization (default `True`).
  - `warn_low_retention_ratio: float` — threshold for the low-retention `UserWarning` emitted inside `_prepare_batch`. Default `0.05` (5 %); `0.0` disables.
- **Output:** `NeckFeatureOutput` whose `feature` has shape `(B, 384, H, W)` float32 on `device`.
- **Gradient behavior:** none. The body is wrapped in `@torch.no_grad()`.
- **Raises:**
  - `ValueError("batched_pts is empty.")` when the list is empty.
- **Empty frames (after optional `filter_range`):** instead of raising, the batch row for that frame is filled with **zeros** in the neck tensor `(B, 384, H, W)`. Internally a single in-range dummy point is run through the voxel pipeline (so the CUDA op always sees `N ≥ 1`), then that batch index is zeroed before returning.
- **Step-by-step logic:**
  1. `_prepare_batch(...) → (prepared, empty_row_flags)`:
     - For each frame: `as_tensor → .to(device) → filter_range (optional) → low-retention warn`.
     - If a frame has zero points after filtering: append `True` to `empty_row_flags` and substitute `_in_range_dummy_point()` for voxelization only.
  2. Calls `_run_neck(...)`:
     - `model.pillar_layer(prepared)` → voxelize + pad.
     - `model.pillar_encoder(...)` → `(B, 64, Ny, Nx)` scattered to BEV.
     - `model.backbone(...)` → list of 3 tensors at 3 strides.
     - `model.neck(xs)` → concatenates 3 upsampled branches → `(B, 384, H, W)`.
     - For each index where `empty_row_flags[i]` is `True`, `neck_feat[i].zero_()`.
  3. Packs `B, C, H, W` into `NeckFeatureOutput`.

### 5.12b. `extract_neck_forward(batched_pts, do_range_filter=True, warn_low_retention_ratio=0.05) -> NeckFeatureOutput`  *(training API, NOT wrapped in no_grad)*

- **Use case:** Stage A training — A1 (PP fully frozen, only downstream modules train; gradient still needs to *flow through* PP to reach `SpatialReducer`) and A2 (unfreeze `pp.neck` so it actually receives gradient).
- **Signature / args / output:** identical to `extract_neck`.
- **Gradient behavior:** whatever `requires_grad` state the caller set via `freeze_all` / `unfreeze_neck` / `set_trainable` is respected. `PillarLayer` is internally `@torch.no_grad()`-decorated (hard voxelization is non-differentiable), so the earliest module that can actually be trained is `pillar_encoder`.
- **Note:** under `torch.no_grad()` + `freeze_all()`, `extract_neck_forward` must return numerically identical output to `extract_neck` on the same input (enforced by `tests/test_module_pointpillar.py`).

---

## 6. Canonical end-to-end call flows

### 6.1. Input = KITTI `.bin`

```
np.fromfile(.bin).reshape(-1, 4)            # (N, 4) float32, already in LiDAR frame
        │
        v
extractor.extract_neck([pts])               # (1, 384, 248, 216) float32 CUDA
```

### 6.2. Input = PyBullet depth buffer

```
depth_buf in [0, 1]
        │  pybullet_depth_to_meters(near, far)
        v
depth_m (H, W) float32 m
        │  preprocess_depth_frame(intr, extr, cfg)
        v
pts (N, 4) float32 (LiDAR frame, synthesized intensity)
        │  extract_neck([pts])
        v
NeckFeatureOutput (1, 384, H, W)
```

---

## 7. Handling the two domain issues

| Issue | Handled in this file | Mechanism |
|---|---|---|
| **Camera frame ≠ LiDAR frame** | **Fully** | `CameraToLidarExtrinsics` + `camera_to_lidar` (3 preset conventions + translation). |
| **Narrow FoV, LiDAR origin mislocated** | **Fully** | Set `t = [0, 0, ~1.6]` to raise the "virtual LiDAR" to KITTI Velodyne height. |
| **No real intensity channel** | Mitigated | `add_intensity` with 3 modes. |
| **Depth density ≠ LiDAR density** | Mitigated | `voxel_downsample` (0.08–0.15 m) + `subsample_ratio`. |
| **Indoor scale mismatch with KITTI `point_cloud_range`** | **Fully** (via `DepthPreprocessConfig.scale_factor`) | The default `point_cloud_range` spans ~70 m × ~80 m (outdoor). Indoor depth-camera points only fill the first ~5 m. Set `DepthPreprocessConfig.scale_factor ≈ 5–10` — applied inside `preprocess_depth_frame` before voxel grid / intensity — so the scene fills the BEV canvas. Early-warning via `low_point_warn_ratio` + `_warn_low_retention` flags frames where scaling (or convention) is still wrong. See `strategy_full_pipeline.md` § 4.1 "Domain-gap note". |
| **Feature distribution mismatch with training domain** | **Not solvable here** | Fine-tune the neck on depth-camera data (see `strategy_full_pipeline.md` Stage A). |

---

## 8. Common errors & fixes

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: Checkpoint .pth not found` | Wrong `ckpt_path` | Update `PointPillarsConfig.ckpt_path`. |
| `ValueError: Point cloud must have shape (N, 4)` | Missing intensity column | Concat a zero column: `np.concatenate([xyz, zeros((N,1))], axis=1)`. |
| All points outside range (non-empty input, but nothing survives `filter_range`) | Frame becomes an **all-zero** neck row `(1, 384, H, W)` (no `ValueError`). | Still fix `convention` / `extrinsics.t[2]` / `scale_factor` if you expected real geometry — check the low-retention `UserWarning`. |
| `CUDA error: invalid configuration argument` (voxelize) | `N=0` fed directly into voxel op | Should not occur via `extract_neck*` — empty frames use a dummy point internally. If you call `pillar_layer` directly with an empty tensor, avoid it. |
| `ValueError: Depth shape ... does not match intrinsics ...` | `depth.shape ≠ (height, width)` | Update `DepthCameraIntrinsics.width/height`. |
| `ValueError: Unknown extrinsics convention` | Wrong preset name | Use one of: `opencv_to_kitti`, `pybullet_to_kitti`, `identity`. |

---

## 9. Size cheat sheet

| Tensor | Shape | Dtype | Device |
|---|---|---|---|
| Input point cloud | `(N, 4)` | float32 | any (moved to `device`) |
| After `filter_range` | `(M, 4)` | float32 | = model device |
| `pillar_features` | `(B, 64, Ny, Nx)` | float32 | CUDA |
| `backbone xs[0..2]` | `(B, 64\|128\|256, h_i, w_i)` | float32 | CUDA |
| `neck_feat` = output | `(B, 384, 248, 216)` | float32 | CUDA |

With defaults: `Ny = (39.68 - -39.68) / 0.16 ≈ 496`, `Nx = 69.12 / 0.16 = 432`; backbone stride 2 → `(248, 216)` after neck concat.

---

## 10. Changelog (update on every code change)

| Date | Logic version | Change | Files touched |
|---|---|---|---|
| 2026-04-18 | v0.1 | Initial `module_pointpillar.py`: `PointPillarsConfig`, `PointCloudInput`, `NeckFeatureOutput`, and `PointPillarsNeckExtractor` with `extract_neck` (voxel → pillar_encoder → backbone → neck). | `module_pointpillar.py` |
| 2026-04-18 | v0.2 | Translated all comments, docstrings and error messages to English. | `module_pointpillar.py` |
| 2026-04-18 | v0.3 | Added 3 dataclasses (`DepthCameraIntrinsics`, `CameraToLidarExtrinsics`, `DepthPreprocessConfig`) and 6 depth helpers (`pybullet_depth_to_meters`, `depth_to_points_camera`, `camera_to_lidar`, `add_intensity`, `voxel_downsample`, `preprocess_depth_frame`) to bridge depth camera → LiDAR frame. | `module_pointpillar.py` |
| 2026-04-18 | v0.3-doc | Created `module_pointpillar.md` specifying the entire module. | `module_pointpillar.md` |
| 2026-04-18 | v0.3-doc-en | Translated `module_pointpillar.md` fully to English. | `module_pointpillar.md` |
| 2026-04-18 | v0.4        | **Strip-down to neck-only.** Removed `Head`, `anchors_generator`, `assigners`, `get_predicted_bboxes*`, `pointpillars/model/anchors.py`, `pointpillars/ops/iou3d_module.py`, `pointpillars/ops/iou3d/`, and the entire `pointpillars/utils/` folder (with it: `numba`, `open3d`, `opencv-python`, `PyYAML`, `tqdm` dependencies). `setup.py` now builds only `voxel_op`. `_load_weights` switched to `strict=False` with a prefix key filter (`pillar_layer / pillar_encoder / backbone / neck`). | `pointpillars/model/pointpillars.py`, `pointpillars/model/__init__.py`, `pointpillars/ops/__init__.py`, `setup.py`, `requirements.txt`, `module_pointpillar.py` |
| 2026-04-18 | v0.4-doc    | Synced this md to the stripped state: §2.1 single lean dep stack, §2.2 companion file list, §5.3 rewritten, §7 adds "Indoor scale mismatch" row. | `module_pointpillar.md` |
| 2026-04-18 | v0.4.1-doc | Added § 12 "Maintenance rule — keep these docs in sync": trigger matrix, per-PR checklist, idea-only workflow, conflict resolution, minimum grep sweeps. Mirrors the long-form rule in `strategy_full_pipeline.md` § 14. | `module_pointpillar.md` |
| 2026-04-18 | v0.5 | **Stage A enablement + indoor-scale automation.** Additive + minor breaking (defaults changed). **Code (`module_pointpillar.py`):** (1) added `extract_neck_forward` — same API as `extract_neck` but without `@torch.no_grad()`, so gradient can flow into unfrozen submodules during Stage A; (2) refactored shared core into `_prepare_batch` + `_run_neck` for DRY; (3) added freeze API — `freeze_all`, `unfreeze_neck`, `set_trainable(names)`, `trainable_parameters`, plus the class constant `_SUBMODULE_NAMES`; (4) added `DepthPreprocessConfig.scale_factor` (indoor-scale hack applied inside `preprocess_depth_frame`) and `DepthPreprocessConfig.low_point_warn_ratio`; (5) added `_warn_low_retention` static method + `warn_low_retention_ratio` kwarg on both `extract_neck*` entry points; (6) `filter_range` switched to half-open `[x0, x1)` convention; (7) changed defaults — `DepthCameraIntrinsics.far: 20.0 → 8.0`, `DepthPreprocessConfig.max_range: 40.0 → 8.0` (match indoor Stage A/B `DataGenConfig` / `EnvConfig`); (8) updated in-file usage example to `160×120, 90° FoV, far=8.0, scale_factor=6.0, t=[0, 0, 0.4]`; (9) made `from pointpillars.model import PointPillars` a lazy import inside `_build_model` so the pure-Python depth helpers and the test suite can import `module_pointpillar` on a CPU-only box without a built `voxel_op` CUDA extension (import error deferred until `PointPillarsNeckExtractor(cfg)` is actually instantiated). **Tests:** new `tests/test_module_pointpillar.py` covering pure-Python helpers, `filter_range` half-open semantics, `scale_factor`, determinism of `extract_neck`, freeze API assertions, and gradient-flow correctness for `extract_neck_forward` after `unfreeze_neck`. **Docs:** this file updated — § 3 structure, § 4.4 intrinsics table, § 4.6 preprocess table, § 5.5 / § 5.5b / § 5.5c (freeze API), § 5.11 pipeline diagram, § 5.12 / § 5.12b; § 7 indoor-scale row now "Fully" handled. | `module_pointpillar.py`, `PointPillars_module/tests/test_module_pointpillar.py`, `PointPillars_module/tests/__init__.py`, `docs/module_pointpillar.md` |
| 2026-04-18 | v0.5.1 | **Empty-frame robustness + pytest neck suite.** **Code:** `_prepare_batch` now returns `(prepared, empty_row_flags)`; added `_in_range_dummy_point`; `_run_neck` zeros the neck row for any frame that had no points after optional `filter_range` (removes `ValueError` on empty frame). **Tests:** `tests/test_pointpillars_neck_pytest.py` (preprocessing / `extract_neck` shape `(B,384,248,216)` / freeze checks / CPU–CUDA consistency) + `requirements-dev.txt` (`pytest`). **Docs:** §3, §5.12, §8 table. | `module_pointpillar.py`, `PointPillars_module/tests/test_pointpillars_neck_pytest.py`, `PointPillars_module/tests/conftest.py`, `PointPillars_module/requirements-dev.txt`, `docs/module_pointpillar.md` |

> **Rule:** every PR / commit that touches `module_pointpillar.py` must append one row to the table above, stating:
> 1. Which function / dataclass changed.
> 2. Signature changes (input / output).
> 3. Any shape / dtype / device changes.

---

## 11. Relationship to other docs

| File | Owns (authoritative) | Defers to |
|---|---|---|
| `module_pointpillar.md` (this) | The `PointPillarsNeckExtractor` API + depth-camera preprocessing helpers. | Indoor-scale hack: `strategy_full_pipeline.md` § 4.1. |
| `strategy_full_pipeline.md` | End-to-end two-stream pipeline: (A) supervised pretrain `PointPillars → SpatialReducer → Mamba → RiskHead` on PyBullet collision labels; (B) SAC on proprio state with the frozen risk predictor feeding `r_risk` into the reward. | SAC-specific details (Actor/Critic MLPs, freeze regimes, BN-lock hook): `strategy_finetune_with_SAC.md`. |
| `strategy_finetune_with_SAC.md` | SAC-specific implementation details (plain-MLP Actor/Critic on a proprioceptive state vector, BN-lock hook for the frozen perception stream, freeze regimes S1/S2/S3, Polyak averaging, hyperparameters). | Overall architecture + data contracts: `strategy_full_pipeline.md`. Perception API: `module_pointpillar.md`. |

When two docs disagree on the same topic, the "Owns" column wins. Update the other doc to match.

---

## 12. Maintenance rule — keep these docs in sync

This module is one leg of a three-doc spec set: `module_pointpillar.md`, `strategy_full_pipeline.md`, `strategy_finetune_with_SAC.md`. Any change to code OR ideas MUST keep all three coherent, otherwise the next agent will act on stale context.

### 12.1 Trigger matrix — when to update what

| You are changing…                                                             | Must update                                                                                          |
|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Code inside `module_pointpillar.py` (signatures, dataclasses, default values) | `module_pointpillar.md` **and** its "Changelog" table.                                               |
| Preprocessing contract (depth → pts frame, intensity, range filter)           | `module_pointpillar.md` + `strategy_full_pipeline.md` § 4.1 / § 5.1 if the scale hack or camera spec changes. |
| Which PointPillars submodules are exposed / frozen                            | All three docs: perception API here, freeze regime in `strategy_finetune_with_SAC.md` § 4, weight-transfer map in `strategy_full_pipeline.md` § 6.1. |
| Tensor shapes / dtypes / device (e.g. `(B, 384, H, W)`)                       | `module_pointpillar.md` (owner) + data-contract tables in `strategy_full_pipeline.md` § 3.4.          |
| Dependency list (`requirements.txt`)                                          | `module_pointpillar.md` § 2.1 + `strategy_full_pipeline.md` § 7.                                     |
| Pipeline architecture (new module inserted, reward path changed, branch added/removed) | `strategy_full_pipeline.md` (owner) first, then propagate to the other two.                       |
| Reward decomposition / `λ_risk` default / A/B framing                         | `strategy_full_pipeline.md` § 6.3 / § 6.6 (owner) + `strategy_finetune_with_SAC.md` § 11.             |
| SAC internals (Actor/Critic shape, optimizer groups, buffer contents, BN hook) | `strategy_finetune_with_SAC.md` (owner) + cross-ref in `strategy_full_pipeline.md` § 6.4.             |
| Env / camera spec (resolution, FoV, `dt`, episode length)                     | `strategy_full_pipeline.md` § 5.1 + § 6.6 (owner) + mirror in `strategy_finetune_with_SAC.md` § 10.  |

### 12.2 Per-PR checklist

Any PR that touches this doc, `module_pointpillar.py`, or either strategy doc must satisfy:

1. **Ownership respected.** Changes are made in the owning doc first (per § 11 table); other docs only receive cross-references or mirrored summaries.
2. **Changelog row added** in every doc actually modified. One row per doc, with date, author, and a 1–3 line summary.
3. **Version bump** if the change breaks a downstream consumer (API signature, dataclass field removed, tensor shape changed). Use `vX.Y` in the changelog where X = breaking, Y = additive.
4. **Grep pass for stale terms.** If you rename / remove a concept (e.g. dropping `BEVStateExtractor`, renaming a dataclass), `rg` the repo for the old name and update every hit — including diagrams, tables, and checklists.
5. **Cross-reference audit.** Every `§ X.Y` or "see other_doc.md" pointer touched in the change must still resolve. Update section numbers in all docs if you renumber a section.
6. **Code ↔ doc parity.** If you changed code, the relevant doc section must match the new code verbatim for signatures and defaults. If you changed a doc first (idea stage), open a follow-up to apply the code OR add a `TODO(sync):` banner at the top until code catches up.

### 12.3 Idea-only changes (no code yet)

When updating docs as a design exercise (before touching code):

- Mark the affected sections with `> **Proposed — not yet implemented.**` at the top.
- Add an entry to a `## Pending sync` subsection of the changelog table listing the code locations that still need updating.
- Remove the banner and the Pending-sync row **in the same PR that lands the code**.

### 12.4 Conflict resolution

If two docs disagree after a change:

1. The row owner in § 11 wins.
2. If the change crosses ownership boundaries, promote `strategy_full_pipeline.md` as the tiebreaker (it is the architectural source of truth).
3. Never resolve a conflict silently — add a changelog row explaining which doc was wrong and why it was changed.

### 12.5 Minimum grep checks before merging

Run these quick grep sweeps on the three docs before approving a PR; all must come back empty or intentional:

```
rg -n "TODO\(sync\)|PENDING SYNC" PointPillars_module/*.md
rg -n "BEVStateExtractor|BEVFeatureExtractor" PointPillars_module/*.md   # must only appear in v2/v3 changelog notes
rg -n "10 Hz|640.*480|60°" PointPillars_module/*.md                     # old camera/control specs
rg -n "cached-?`?z`?|re-encode.*gradient" PointPillars_module/*.md       # old buffer design
```

If any hit is NOT a deliberate historical note, fix before merging.
