# Chiến lược tạo nhãn rủi ro trên quỹ đạo (`Trajectory`) — PyBullet Stage A

Tài liệu này mô tả **đầy đủ phương pháp** dùng trong repo để gán nhãn “nguy hiểm / va chạm trong tương lai” lên từng frame của một rollout mô phỏng, khớp với `docs/strategy_full_pipeline.md` § 5.1 và mã trong `create_dataset_module/`.

**Phạm vi:** sinh dữ liệu offline (`DataGenerator` → `Trajectory.npz`), **không** mô tả training loop hay SAC.

---

## 1. Mục tiêu nhãn

Mỗi frame thời gian `t` cần có các nhãn nhị phân (thực tế lưu `float32` ∈ `{0.0, 1.0}`) trả lời câu hỏi:

> Trong **khoảng thời gian nhìn về phía trước** (0.5 s, 1 s, hoặc 2 s), robot có **ít nhất một lần va chạm** với vật cản (không tính mặt phẳng sàn) hay không?

Model Stage A (Mamba + RiskHead) học dự đoán xác suất tương ứng với ba horizon đó từ quan sát (depth → điểm → BEV → …).

---

## 2. Chuẩn thời gian và tần số

- **`dt = 0.05` s** trong `DataGenConfig` → **20 Hz** cho một “tick” điều khiển / một frame lưu trong `Trajectory`.
- Mọi “quy đổi giây ↔ số frame” trong nhãn dựa trên **20 frame = 1 giây**.

Code tham chiếu: `PointPillars_module/data_contracts.py` (`DataGenConfig.dt`), `create_dataset_module/generator.py`.

---

## 3. Tín hiệu gốc: `contact_flag[t]`

### 3.1 Định nghĩa

Với rollout dài `T` frame, có mảng `contact_flag` hình `(T,)`:

- `contact_flag[t] = 1` (True) nếu **sau bước mô phỏng tại frame đó**, robot **đang tiếp xúc** với ít nhất một body **không phải nền** (tường, obstacle tĩnh/động, …).
- `= 0` nếu không có contact như vậy.

### 3.2 Nguồn trong mã

- `DatasetEnv.get_contact_flag()` gọi `RL_Env.check_collision()` (trong `pybullet_navigation.py`), dùng PyBullet để xác định va chạm thực tế trong vật lý mô phỏng.
- Trong `DataGenerator._rollout`, mỗi frame append cờ này vào mảng `contact` rồi đưa vào `Trajectory`.

**Điểm quan trọng:** nhãn rủi ro **không** được suy trực tiếp từ depth, khoảng cách ước lượng, hay mạng học — **chỉ** từ **chuỗi sự kiện contact** đã xảy ra trong sim.

---

## 4. Phương pháp: lookahead trên `contact_flag`

### 4.1 Ý tưởng

Tại thời điểm `t`, ta nhìn **cửa sổ thời gian về phía trước** (bao gồm cả frame hiện tại):

\[
\text{risk\_*s}[t] = \mathbb{1}\Big[ \exists\, k \in [t,\, t+H) : \text{contact\_flag}[k] = 1 \Big]
\]

tức là **bất kỳ** frame nào trong đoạn `[t, t+H)` có contact thì nhãn tại `t` là **nguy hiểm** theo horizon đó.

### 4.2 Cài đặt: `lookahead_any`

File: `create_dataset_module/generator.py`.

```text
lookahead_any(contact_flag, H)[t] = any(contact_flag[t : t+H])
```

- `H` là số **frame** (không phải giây).
- Vòng lặp `t = 0 … T-1`, với mỗi `t` lấy slice `contact_flag[t : min(T, t+H)]` — nếu gần cuối rollout mà `t+H > T` thì cửa sổ **bị cắt ngắn**, **không** pad thêm `True`. Do đó vài frame cuối có thể có nhãn 0 dù “thực tế” sau khi kết thúc rollout vẫn có thể có va — vì dữ liệu không có tương lai.

Hàm trả về `np.float32` shape `(T,)`, giá trị `0.0` hoặc `1.0`.

### 4.3 Ba horizon mặc định (khớp § 5.1)

| Nhãn | Số frame `H` | Thời gian (@ 20 Hz) | Ghi chú trong code |
|------|----------------|----------------------|--------------------|
| `risk_05s` | 10 | 0.5 s | `horizon_05s_frames` trong `DataGenConfig` |
| `risk_1s`  | 20 | 1.0 s | `horizon_1s_frames` |
| `risk_2s`  | 40 | 2.0 s | `horizon_2s_frames` |

Công thức tương đương (mặc định):

```text
risk_05s = lookahead_any(contact, 10)
risk_1s  = lookahead_any(contact, 20)
risk_2s  = lookahead_any(contact, 40)
```

Nếu đổi `horizon_*_frames` trong config, generator dùng đúng các giá trị đó.

### 4.4 Thứ tự “độ rộng” nhãn

Với cùng một `contact_flag`, horizon càng dài thì điều kiện “có ít nhất một contact trong cửa sổ” càng **dễ** thỏa mãn hơn (miễn là sau `t` có va trong phạm vi dài hơn). Trên lý thuyết:

\[
\text{risk\_2s}[t] \ge \text{risk\_1s}[t] \ge \text{risk\_05s}[t]
\]

(về mặt nhị phân; bằng nhau hoặc 0/1 tùy chuỗi contact).

---

## 5. Ghép với rollout: cắt sớm và độ dài thực tế

- **`terminate_on_contact` (mặc định True):** có thể dừng rollout khi phát hiện contact; toàn bộ mảng (gồm `contact`, `risk_*`, depth, …) được **cắt** đến độ dài thực tế trước khi lưu (`DataGenConfig.post_contact_grace_frames` cho phép thêm vài frame sau contact nếu cần).
- **Ý nghĩa:** không lưu dài hàng trăm frame “robot nằm vật lý” sau va, tránh nhiễu nhãn và dữ liệu vô ích.

Chi tiết hành vi: `create_dataset_module/generator.py` (`_rollout`).

---

## 6. Lưu trữ: `Trajectory` và file `.npz`

- `Trajectory` (trong `PointPillars_module/data_contracts.py`) chứa `contact_flag` (hoặc tên trường tương ứng trong contract), `risk_05s`, `risk_1s`, `risk_2s` cùng chiều dài `T`.
- `to_npz` / `from_npz` giữ dtype và hình dạng; nhãn risk là **nhị phân**.

---

## 7. Từ `Trajectory` sang mẫu train: `RiskDataset`

- `RiskDataset` đọc các file `.npz`, với mỗi chỉ số frame `t` hợp lệ (đủ context + horizon trong phạm vi có dữ liệu) tạo một `RiskSample` gồm chuỗi quan sát và **ba nhãn** `risk_05s`, `risk_1s`, `risk_2s` **tại `t`**.
- `collate_riskbatch` gom batch cho loss (ví dụ focal BCE trên ba horizon).

Ràng buộc index (ví dụ không lấy mẫu quá gần cuối nếu không đủ horizon) được xử lý trong `risk_dataset.py`.

---

## 8. Đánh giá chất lượng nhãn (tổng quan)

- **Tỷ lệ positive** (đặc biệt `risk_1s`): summary cuối `DataGenerator.run()` in `positive ratio 1.0s`; nếu quá thấp, tăng tỷ lệ policy **adversarial** hoặc điều chỉnh scene; quá cao thì giảm.
- **Phân stratified theo scene:** `scene_stratified_split` để train/val/test không trùng layout — tránh đánh giá lạc quan.

---

## 9. Quan sát quan trọng — tránh overfit / shortcut learning

Tài liệu này mô tả nhãn **đúng với định nghĩa contact + lookahead**. Dưới đây là các **rủi ro huấn luyện** thường gặp và cách đối phó **trong repo** (hoặc gợi ý cấu hình).

### 9.A. Mất cân bằng lớp (class imbalance)

**Hiện tượng:** Trong sim, nếu robot hiếm va chạm, nhãn **0 (an toàn)** có thể chiếm **> 90–95%** frame. Model dễ học “luôn dự đoán an toàn” để giảm loss trung bình.

**Đã có trong thiết kế:**

- **Mix policy:** `policy_adversarial_p` (mặc định 0.2) cố tình hướng robot về phía obstacle → tăng mẫu positive thật trong `contact_flag`.
- **Giám sát khi sinh data:** `DataGenerator.run()` in `positive ratio 1.0s` và cảnh báo nếu **< 5%** hoặc **> 50%** — cần chỉnh `policy_*_p` cho tới khi vùng **~10–20%** cho `risk_1s` (tham khảo thực tế train, không cứng nhắc).
- **Stage A loss:** `PointPillars_module/losses.py` — **`focal_bce`** (gamma > 0) giảm trọng số mẫu dễ (p_t → 1), tập trung mẫu khó / ít.
- **Oversampling:** `oversample_positive_indices()` + `WeightedRandomSampler` (ví dụ trong `create_dataset_module/README.md`) để batch không toàn negative.

**Gợi ý:** Coi tỷ lệ positive như **metric của pipeline sinh data**, không chỉ “chạy xong là đủ”.

### 9.B. Robot đứng yên nhưng môi trường động

**Hiện tượng:** `contact_flag` chỉ biết **có contact hay không**, không phân biệt “do robot lao vào” hay “do vật cản động đâm vào robot”. Nếu mọi rollout đều cho robot **luôn chuyển động mạnh**, model có thể học shortcut gắn rủi ro với **tốc độ lớn** mà thiếu tình huống **robot gần như đứng yên** nhưng obstacle di chuyển.

**Trong sim:** `pybullet_navigation.RL_Env` có **obstacle động** (`N_DYNAMIC_OBS`, `_bounce_dynamic_obs`, …) — contact vẫn có thể xảy ra khi base gần như không chủ động di chuyển.

**Đã bổ sung trong code:**

- **`StationaryPolicy`** (`create_dataset_module/policies.py`): luôn output **v = 0, ω = 0**.
- **`policy_stationary_p`** trong `DataGenConfig` (mặc định **0.0** để giữ hành vi cũ): khi đặt > 0, phải **giảm** các `policy_*_p` khác sao cho **tổng = 1.0**. Ví dụ: `0.45 / 0.25 / 0.2 / 0.1` (random / scripted / adversarial / stationary).

**Gợi ý:** Tăng dần `policy_stationary_p` (vài % → 10%) và theo dõi `positive ratio` + chất lượng val; không cần stationary cho mọi rollout nếu đã đủ đa dạng.

### 9.C. Mơ hồ ở biên (boundary ambiguity)

**Hiện tượng:** Nhãn nhị phân **{0, 1}** tại ranh giới “sắp chạm / vừa chạm” (PyBullet contact) có thể nhạy với nhiễu vật lý; mô hình xuất xác suất ~0.5 có thể **nhảy lớp** khi threshold cố định 0.5.

**Đã có:**

- **Focal BCE** (mục 9.A): tăng gradient trên mẫu khó, giảm “lười” dự đoán cực đoan 0/1 không hiệu năng.
- **Đánh giá:** nên dùng **AUC-PR / AUC-ROC**, calibration (Brier), không chỉ accuracy — xem `docs/strategy_full_pipeline.md` § 5 (metrics).

**Gợi ý (không đổi nhãn trong repo mặc định):** Khi triển khai inference, có thể điều chỉnh **ngưỡng** theo từng horizon trên tập val; hoặc (nâng cao) nhãn mềm / khoảng cách — **ngoài** phạm vi pipeline hiện tại.

---

## 10. Tóm tắt một dòng

**Nhãn = “trong H frame tiếp theo (0.5s / 1s / 2s) có xảy ra contact trong PyBullet hay không?”, với H lấy từ `lookahead_any` trên `contact_flag`; không dùng depth làm ground truth.**

---

## 11. Tham chiếu mã nguồn (định vị nhanh)

| Nội dung | File |
|----------|------|
| Công thức lookahead & domain rand | `create_dataset_module/generator.py` |
| Contact flag từ env | `create_dataset_module/env_wrapper.py`, `pybullet_navigation.py` |
| Default horizon (frame) | `PointPillars_module/data_contracts.py` → `DataGenConfig` |
| Dataset + split | `create_dataset_module/risk_dataset.py` |
| Kiểm thử nhãn | `create_dataset_module/tests/test_risk_label.py` |
| Policy đứng yên | `create_dataset_module/policies.py` (`StationaryPolicy`), `DataGenConfig.policy_stationary_p` |
| Focal loss / oversample | `PointPillars_module/losses.py` |

---

## 12. Liên kết tài liệu kiến trúc

- Định nghĩa contract `Trajectory` và risk: `docs/strategy_full_pipeline.md` § 3.1, § 5.1.
- Bản doc này là **mở rộng giải thích** cho khối “Risk label derivation” trong § 5.1; nếu mâu thuẫn, **ưu tiên chỉnh `strategy_full_pipeline.md` trước**, rồi cập nhật code và bản doc này cho khớp.

---

## Changelog

| Version | Date | Summary |
|---------|------|---------|
| v1.1 | 2026-04-18 | §9 Critical observations (A/B/C); `StationaryPolicy` + `policy_stationary_p`; extended code table. |
| v1.0 | 2026-04-18 | First version: full description of contact-based lookahead labels, horizons, code pointers. |
