# Hướng dẫn thiết lập môi trường và chạy thử nghiệm (Dành cho New Agent / Máy mới)

Tài liệu này là "Kim chỉ nam" để một AI Agent mới hoặc một kỹ sư khác có thể nhân bản (clone) chính xác 100% môi trường, dữ liệu và các cấu hình thí nghiệm từ máy cũ sang máy mới (ví dụ từ RTX-5060TI sang RTX-4070S).

## 1. Clone Source Code
Đầu tiên, kéo toàn bộ code mới nhất (bao gồm cả thư mục `Compare-with-other`) từ nhánh tương ứng trên Github:
```bash
git clone -b devices/RTX-4070S https://github.com/TranQuyen-80405/Mamba-Predict-Trajactory-with-SAC-Pipeline.git
cd Mamba-Predict-Trajactory-with-SAC-Pipeline
```

## 2. Cài đặt Môi trường Python (.venv-backups)
Dự án sử dụng một script chuẩn bị sẵn để cài đặt toàn bộ CUDA, PyTorch, Mamba-ssm và các thư viện cần thiết.
**Tuyệt đối không dùng `pip install` chay** để tránh xung đột version. Hãy chạy script với biến môi trường `VENV_DIR=.venv-backups` để tạo đúng tên thư mục môi trường y hệt máy hiện tại:

```bash
chmod +x scripts/setup_train_env_ubuntu2404.sh
VENV_DIR=.venv-backups ./scripts/setup_train_env_ubuntu2404.sh
```
Sau khi cài đặt thành công, hãy đảm bảo bạn đang kích hoạt đúng môi trường dự phòng `.venv-backups`:
```bash
source .venv-backups/bin/activate
```

## 3. Chuyển Dữ liệu (Raw Data)
Do Github có giới hạn dung lượng nên thư mục `data` không được push lên mạng. Bạn cần chuyển tay (copy qua SSH/USB) bộ dữ liệu gốc `stage_a_experiment_2gpu_balanced_v5` vào thư mục `data/` trong repo của máy mới.

Cấu trúc yêu cầu:
```text
Mamba-Predict-Trajactory-with-SAC-Pipeline/
└── data/
    └── stage_a_experiment_2gpu_balanced_v5/
        ├── scene_0000/
        ├── scene_0001/
        └── index.jsonl
```

## 4. Tái tạo BEV Cache (V5 DS4)
Sau khi có Raw data, bạn phải tạo lại bộ Cache Feature 25GB để có thể train các models mà không bị OOM (tràn RAM). 
Cần đảm bảo file trọng số `epoch_160_raw.pth` đã có sẵn tại `PointPillars_module/pretrained/`.
Chạy lệnh sau:
```bash
python scripts/cache_pointpillars_bev.py \
  --data_root data/stage_a_experiment_2gpu_balanced_v5 \
  --ckpt PointPillars_module/pretrained/epoch_160_raw.pth \
  --out_root data/stage_a_experiment_2gpu_balanced_v5_bev_cache_ds4 \
  --spatial_downsample 4 \
  --save_dtype float16
```
*(Xem thêm file `docs/how_to_recreate_v5_cache.md` nếu cần giải thích tham số)*

## 5. Chạy Huấn luyện và Đánh giá

### 5.1. Train Mô hình Đa nhiệm (Multi-task) của dự án chính
Để train mô hình chính (kết hợp Mamba dự đoán cùng lúc Trajectory và Risk), chạy script gốc:
```bash
./train_compare.sh
```
Kết quả sẽ lưu tại thư mục `runs/stage_a_compare_turn-2_V5/`.

### 5.2. Train các Mô hình Baseline để so sánh (Single-task)
Các mô hình chuẩn mực (LSTM, Transformer, Heatmap, MLP...) đã được code riêng thành Đơn nhiệm (Single Task) và nằm ở thư mục `Compare-with-other`.
Để train chúng và nhận được output (Log, Metrics, Json) khớp 100% với form code gốc, hãy chạy:
```bash
./Compare-with-other/run.sh
```
Kết quả sẽ được lưu tại `runs/compare_baselines/`. Bạn có thể chỉnh sửa file `run.sh` để đổi model muốn train (`--model lstm`, `--model heatmap`, v.v...).

---
## Tóm tắt Checklist Bắt buộc cho một AI Agent mới:
1. ✅ **Kích hoạt Virtual Env:** Phải gọi `source .venv-backups/bin/activate` hoặc gọi trực tiếp `./.venv-backups/bin/python` trước khi làm bất cứ việc gì.
2. ✅ **Kiểm tra File:** Đảm bảo tồn tại Raw Data và Cache BEV V5 trước khi chạy train.
3. ✅ **Nguyên tắc Sửa code:** Khi muốn thêm hay chỉnh sửa code Baseline, **CHỈ THAO TÁC** bên trong folder `Compare-with-other`. TUYỆT ĐỐI KHÔNG sửa hay can thiệp các file hệ thống lõi trong `PointPillars_module` để đảm bảo tính khách quan khi so sánh.
4. ✅ **Đánh giá Metric:** Luôn luôn so sánh hiệu năng dựa trên file `val_metrics_final.json` nằm trong các thư mục con của `runs/`.
