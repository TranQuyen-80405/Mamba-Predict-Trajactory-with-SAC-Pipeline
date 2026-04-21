# Checklist: chuẩn bị trước khi chạy `notebooks/compare_trajactory_predict_module.ipynb` trên Google Colab (GPU T4)

Dùng checklist này trên máy **local** (Windows/Linux), sau đó upload hoặc clone repo lên Colab. Mục tiêu: đủ **code + data Stage A + checkpoint PointPillars** đúng cấu trúc mà notebook đang `assert`.

---

## 1. Môi trường local (tùy chọn nhưng nên làm)

- [ ] Python ≥ 3.10, có thể chạy `DataGenerator` (PyBullet + deps theo repo).
- [ ] Từ thư mục gốc repo (`Pipeline/`), kiểm tra: `python -c "import sys; print(sys.version)"`.
- [ ] Cài deps cho datagen (PyBullet + NumPy + Matplotlib + Torch — xem `create_dataset_module/requirements.txt`):

  ```bash
  python -m pip install -r create_dataset_module/requirements.txt
  ```

- [ ] Nếu chỉ cần tạo data: dùng venv của project (ví dụ `.venv`) và `python run_datagen_preset.py experiment`.

---

## 2. Tạo đúng bộ dataset mà notebook kỳ vọng

Notebook mặc định: `DATA_ROOT = <REPO>/data/stage_a_experiment`.

- [ ] Trong `Pipeline/`, chạy:

  ```bash
  python run_datagen_preset.py experiment
  ```

- [ ] Sau khi xong, thư mục **`data/stage_a_experiment/`** phải tồn tại và có **`index.jsonl`** cùng các file rollout (`.npz` theo generator — contract Stage A: trajectory + nhãn risk). **Git:** repo đã track `data/` (kèm smoke / rgb_spotcheck) — sau `git clone` có thể dùng luôn; chỉ cần chạy bước trên nếu bạn muốn tạo lại hoặc đổi preset.
- [ ] **Không** dùng nhầm:
  - `data/stage_a_smoke_nb` (preset `smoke_nb`)
  - `data/stage_a_full` (preset `full`) — trừ khi bạn **đổi** `DATA_ROOT` trong notebook cho khớp.
  - `data/stage_a_rgb_spotcheck` — chỉ để spotcheck RGB, không phải bộ so sánh method chuẩn.

---

## 3. Checkpoint PointPillars (neck KITTI)

Notebook tìm trong **`PointPillars_module/pretrained/`** theo thứ tự:

1. `epoch_160.pt`
2. `epoch_160.pth`

- [ ] Đặt **một trong hai** file vào:  
  `Pipeline/PointPillars_module/pretrained/epoch_160.pt`  
  hoặc  
  `Pipeline/PointPillars_module/pretrained/epoch_160.pth`
- [ ] Repo thường **không** commit file nặng này — bạn phải **copy từ máy** hoặc Drive riêng.

---

## 4. Cấu trúc thư mục trước khi đưa lên Colab

Colab thường dùng `REPO_ROOT = "/content/Pipeline"`. Cấu trúc tối thiểu cần khớp:

```text
Pipeline/
├── notebooks/
│   └── compare_trajactory_predict_module.ipynb
├── create_dataset_module/
│   └── risk_dataset.py
├── data/
│   └── stage_a_experiment/     ← từ bước 2
│       ├── index.jsonl
│       └── ... (các .npz rollout)
├── PointPillars_module/
│   ├── training/               ← Stage A train scripts (mamba, lstm, compare, …)
│   ├── module_pointpillar.py
│   ├── models/
│   ├── pretrained/             ← từ bước 3
│   │   └── epoch_160.pt hoặc epoch_160.pth
│   └── ...
├── scripts/datagen/            ← implementation (preset experiment)
├── run_datagen_preset.py       ← wrapper ở root (cùng lệnh như trước)
└── pybullet_navigation.py
```

- [ ] Đã có đủ **`PointPillars_module`**, **`create_dataset_module`**, **`data/stage_a_experiment`**, **`pretrained/`** như trên.
- [ ] Nếu upload bằng ZIP: giải nén sao cho **`Pipeline/`** là thư mục gốc chứa `notebooks/compare_trajactory_predict_module.ipynb` (không lồng sai một cấp).

---

## 5. Đóng gói / đưa lên Colab (chọn một cách)

**Cách A — Git clone trong Colab**

- [ ] Repo đã push lên GitHub/GitLab (lưu ý: **không** commit file checkpoint nặng nếu policy repo cấm; khi đó dùng Drive cho `pretrained/`).
- [ ] Trong Colab: `!git clone <URL> Pipeline` → sửa `REPO_ROOT` nếu tên thư mục khác.

**Cách B — Upload ZIP**

- [ ] Nén folder `Pipeline` (đã có data + pretrained).
- [ ] Upload ZIP lên Colab, giải nén vào `/content/Pipeline`.

**Cách C — Google Drive**

- [ ] Copy cả `Pipeline` lên Drive (ví dụ `MyDrive/RobotDog/Pipeline`).
- [ ] Trong Colab: mount Drive, đặt `REPO_ROOT = "/content/drive/MyDrive/RobotDog/Pipeline"` (đúng path thực tế).

---

## 6. Trên Colab / Linux — thao tác runtime

- [ ] **Runtime → Change runtime type → GPU** (T4 hoặc tương đương) nếu dùng Colab.
- [ ] Cell đầu notebook cài PyTorch CUDA (thử `cu124` / `cu121` / `cu118`) rồi `mamba-ssm[causal-conv1d]` (Linux). Trên **Windows native**, cell sẽ báo lỗi có hướng dẫn: dùng **Colab** hoặc **WSL2 Ubuntu** — `mamba-ssm` upstream là **Unix + CUDA**, không có wheel ổn định cho Windows.

---

## 7. Kiểm tra nhanh trước `run_experiment` (cell Python tùy chọn)

Chạy trong Colab (đổi `REPO_ROOT` nếu cần):

```python
import os
REPO_ROOT = "/content/Pipeline"
assert os.path.isdir(os.path.join(REPO_ROOT, "data", "stage_a_experiment"))
assert os.path.isfile(os.path.join(REPO_ROOT, "data", "stage_a_experiment", "index.jsonl"))
pre = os.path.join(REPO_ROOT, "PointPillars_module", "pretrained")
assert any(os.path.isfile(os.path.join(pre, n)) for n in ("epoch_160.pt", "epoch_160.pth"))
print("OK: data + checkpoint paths")
```

- [ ] Ba dòng `assert` đều pass.

---

## 8. Chạy notebook so sánh

- [ ] Mở `notebooks/compare_trajactory_predict_module.ipynb`.
- [ ] Đảm bảo `REPO_ROOT` trong cell chính trùng path trên Colab.
- [ ] Chạy lần lượt các cell: pip → import + `run_experiment`.
- [ ] Output mong đợi: log TensorBoard dưới `runs/stage_a_compare/`, file `summary_<YYYYMMDD_HHMMSS>.json` (metrics + đường dẫn checkpoint), và thư mục **`runs/stage_a_compare/checkpoints/`**. Mỗi backbone sẽ có thư mục con dạng `{backbone}_risk_{epochs_run}epochs/` chứa checkpoint `_last.pt`, `_latest.pt`, `_best_val_ap.pt` và bản copy checkpoint nguồn (`*_source.pth`/`*.pt`).

---

## 9. Lỗi thường gặp

| Triệu chứng | Hướng xử lý |
|-------------|-------------|
| `Thiếu data: .../stage_a_experiment` | `git pull` / clone lại repo (data đã có trên remote), hoặc chạy `run_datagen_preset.py experiment` rồi commit — hoặc upload thư mục `data/stage_a_experiment` lên Drive/Colab. |
| `Thiếu checkpoint trong .../pretrained` | Copy `epoch_160.pt` hoặc `.pth` vào đúng `PointPillars_module/pretrained/`. |
| Lỗi import `train_stage_a_compare` | `os.chdir(REPO_ROOT)` và `sys.path` phải chứa `PointPillars_module` — đúng như cell notebook; module nằm trong `PointPillars_module/training/`. |
| CUDA / wheel PyTorch | Đổi index `cu124` ↔ `cu121`; hoặc dùng torch có sẵn của Colab rồi chỉ `pip install` các gói còn thiếu. |
| `causal-conv1d` / `mamba-ssm` fail trên Windows | Chạy notebook trên **Colab GPU** hoặc **WSL2 Linux**; không expect pip build thành công trên Windows native. |

---

## 10. Tham chiếu tài liệu trong repo

- `run_datagen_preset.py` — preset `experiment` → `data/stage_a_experiment`.
- `docs/strategy_experiment_protocol.md` — quy mô lab và metric thống nhất (nếu cần đối chiếu).
- `docs/strategy_full_pipeline.md` — contract Stage A / tensor shapes.

---

## 11. Changelog

| Date       | Change |
|------------|--------|
| 2026-04-18 | §1: thêm lệnh `pip install -r create_dataset_module/requirements.txt` để tránh thiếu `pybullet` / `matplotlib` / `torch` khi chạy `run_datagen_preset.py` trên Colab hoặc máy local. |
| 2026-04-18 | Repo track `data/` trên Git (Stage A experiment + smoke + rgb_spotcheck); §2 / §9 cập nhật gợi ý clone thay vì chỉ upload tay. |
| 2026-04-19 | Notebook chuyển vào `notebooks/`; Stage A train vào `PointPillars_module/training/`; cây thư mục §4 cập nhật. |

---

*Cập nhật: checklist cho luồng Colab T4 + notebook so sánh temporal Stage A.*
