# Hướng dẫn tái tạo (Recreate) Dataset Cache V5 (BEV Cache DS4)

Tài liệu này hướng dẫn cách tạo lại chính xác 100% thư mục cache `data/stage_a_experiment_2gpu_balanced_v5_bev_cache_ds4` (dung lượng ~25GB) đã dùng để huấn luyện các mô hình Baseline và Model chính, trong trường hợp bạn thay đổi máy (ví dụ sang `RTX-4070S`) mà không thể chuyển copy file trực tiếp qua mạng.

## Mục đích của Cache V5
Trong **Stage A**, để tránh việc GPU phải nội suy và chạy lại mô hình PointPillars (cực kỳ tốn kém thời gian) cho hàng trăm ngàn frame liên tục ở mỗi epoch, chúng ta sử dụng một script để chạy qua toàn bộ dataset 1 lần duy nhất. Script này đẩy dữ liệu PointCloud qua Neck của mô hình PointPillars (pretrained `epoch_160_raw.pth`) và lưu lại các mảng BEV (Bird's Eye View) Feature dưới dạng file nén.

Thông số `ds4` nghĩa là **Downsample x4** về mặt không gian (Spatial Downsample), giúp giảm cực mạnh kích thước lưu trữ (chỉ còn khoảng 25GB) mà vẫn đảm bảo đặc trưng lõi.

---

## Các bước thực hiện

### Bước 1: Chuẩn bị dữ liệu thô (Raw Data)
Đảm bảo bạn đã copy hoặc tạo ra bộ dữ liệu gốc nằm tại đường dẫn:
```text
/workspace/Mamba-Predict-Trajactory-with-SAC-Pipeline/data/stage_a_experiment_2gpu_balanced_v5
```

### Bước 2: Kích hoạt môi trường
Hãy chắc chắn rằng bạn chạy lệnh trong môi trường chuẩn của project (có chứa Pytorch và các thư viện cần thiết):
```bash
cd /workspace/Mamba-Predict-Trajactory-with-SAC-Pipeline
source .venv-linux-2404/bin/activate
```

### Bước 3: Chạy lệnh tạo Cache
Sử dụng chính xác câu lệnh sau tại thư mục gốc của repository (copy y hệt):

```bash
python scripts/cache_pointpillars_bev.py \
  --data_root data/stage_a_experiment_2gpu_balanced_v5 \
  --ckpt PointPillars_module/pretrained/epoch_160_raw.pth \
  --out_root data/stage_a_experiment_2gpu_balanced_v5_bev_cache_ds4 \
  --spatial_downsample 4 \
  --save_dtype float16
```

**Giải thích các cờ lệnh (Flags):**
- `--data_root`: Chỉ định thư mục dữ liệu thô đầu vào.
- `--ckpt`: Trọng số mô hình PointPillars đóng băng (đã được pretrain).
- `--out_root`: Thư mục đầu ra nơi xuất các file `.npy` / `.pt` chứa tensor đặc trưng (Đúng tên `..._bev_cache_ds4`).
- `--spatial_downsample 4`: Đây là thông số quan trọng nhất (DS4). Nó sẽ thu nhỏ kích thước (H, W) của BEV Feature map 4 lần, giúp máy tính tải vào RAM rất nhanh và tương thích với kích thước đầu vào `in_channels=384` của các model.
- `--save_dtype float16`: Giữ cho dữ liệu ở chuẩn Float16 để tiết kiệm ổ cứng (25GB thay vì 50GB).

### Bước 4: Kiểm tra lại (Verify)
Tiến trình này sẽ chạy qua toàn bộ dataset (chạy khoảng 30 phút - 1 tiếng tùy tốc độ của RTX 4070S và tốc độ đọc ghi ổ cứng).
Sau khi chạy xong, hãy dùng lệnh kiểm tra dung lượng:
```bash
du -sh data/stage_a_experiment_2gpu_balanced_v5_bev_cache_ds4
```
Nếu thư mục rơi vào khoảng **~25GB**, chúc mừng bạn! Bạn đã tái tạo lại thành công 100% môi trường dữ liệu y hệt như cũ. Lúc này bạn có thể chạy `train_compare.sh` hoặc chạy `Compare-with-other/run.sh` hoàn toàn bình thường.
