# EXP-000 – Huấn luyện gốc trên Kaggle (lịch sử)

Nguồn: hai notebook trong `notebooks/` (đã đọc output đã lưu) đối chiếu với metadata nhúng trong `helmet_best.pt` / `plate_best.pt`
(`train_args`, `train_metrics`, `train_results`). Số liệu đầy đủ: `metrics.json`; bảng so sánh tham số: `notebook_vs_checkpoint.json`.

## Helmet – notebook KHỚP checkpoint ✅
- Toàn bộ 108 tham số train của checkpoint trùng với dòng `engine/trainer:` trong log của notebook, trừ 3 khác biệt vô hại:
  `device` (0 vs "0", chỉ khác kiểu), `save_dir` (không lưu trong checkpoint), `warmup_bias_lr` (0.1 trong log, 0.0 trong checkpoint — giá trị đã bị lịch trình warm-up sửa).
- Kết quả cuối trên val trùng khớp checkpoint (P 0.983, R 0.989, mAP50 0.994, mAP50-95 0.809); thời gian 1.453 h trùng 5229 s.
- Môi trường train: Kaggle, **Tesla T4 (14.9 GB)**, Python 3.12.12, torch 2.9.0+cu126, Ultralytics 8.4.26, YOLO11s (9,428,953 tham số, 21.6 GFLOPs), AdamW tự chọn (lr 0.001429), 150 epoch, AMP.
- **Chia dữ liệu** (từ log quét nhãn): train **1014** ảnh (2 ảnh nền), val **278** ảnh (1 nền, 1384 instance), test **168** ảnh (2 nền, 818 instance). Tổng 1460 ảnh.
- **Val theo class (best.pt):**

| class | ảnh | instance | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| all | 278 | 1384 | 0.983 | 0.989 | 0.994 | 0.809 |
| helmet | 262 | 606 | 0.987 | 0.994 | 0.993 | 0.793 |
| motorcycle | 276 | 608 | 0.985 | 0.995 | 0.995 | 0.882 |
| no-helmet | 154 | 170 | 0.977 | 0.976 | 0.993 | 0.751 |

- **Test theo class (best.pt, `split='test'`):**

| class | ảnh | instance | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| all | 168 | 818 | 0.987 | 0.996 | 0.994 | 0.812 |
| helmet | 164 | 359 | 0.980 | 1.000 | 0.993 | 0.791 |
| motorcycle | 166 | 348 | 0.991 | 0.989 | 0.995 | 0.878 |
| no-helmet | 100 | 111 | 0.991 | 0.998 | 0.995 | 0.768 |

- Tốc độ báo trong notebook: 2.9 ms inference/ảnh (val) trên T4 — **không** so sánh được với FPS pipeline của EXP-001 (khác phần cứng, chỉ model, không tracker/vẽ).
- Chỗ chưa nhất quán trong notebook: markdown ở mục 2 vẫn ghi `yolo11n` và `epochs=50` trong khi code và checkpoint là `yolo11s` / 150 epoch (chú thích cũ chưa sửa).

### Đính chính AUDIT
AUDIT (mục 2.B/2.D) từng ghi "không có bằng chứng về test set". **Sai một phần**: dataset helmet **có** split test (168 ảnh) và notebook đánh giá trên đó; chỉ là kết quả đó không được lưu trong repo cho đến nay.

### Cách đọc các con số (vẫn là cảnh báo)
- Test (0.994 / 0.812) gần như **trùng** val (0.994 / 0.809): mAP50 ở mức bão hòa (0.993–0.995 cho cả 3 class) → thang đo này không còn phân biệt được chất lượng.
- Tên ảnh test mẫu trong notebook (`frame_10_579_jpg`) cho thấy ảnh là **frame trích từ video**. Nếu các split chia theo frame thay vì theo video, frame liền kề có thể nằm ở cả train và test → **tính độc lập của test chưa được xác minh** (phần kiểm tra thuộc P1). Đây là suy đoán có căn cứ, chưa phải kết luận.
- Mất cân bằng: `no-helmet` chỉ 170 instance trên val (28% của helmet/motorcycle); đây cũng là class có mAP50-95 thấp nhất (0.751 val / 0.768 test) và là class quan trọng nhất cho luật vi phạm.

## Plate – notebook KHÔNG khớp checkpoint ❌
Xác nhận nghi ngờ của bạn. 10 trên 19 tham số `train()` được ghi tường minh trong `license-plate-detector.ipynb` khác `plate_best.pt`:

| tham số | notebook | `plate_best.pt` |
|---|---|---|
| model | yolo11**s** | yolo11**n** |
| epochs | 150 | 50 |
| patience | 30 | 100 |
| name | plate_model_pro | license_plate_model |
| degrees | 15.0 | 0.0 |
| perspective | 0.001 | 0.0 |
| shear | 2.0 | 0.0 |
| scale | 0.6 | 0.5 |
| fliplr | 0.0 | 0.5 |
| cos_lr | True | False |

Giống nhau: dữ liệu (`vietnam-license-plate-v1/data.yaml`), imgsz 640, batch 32, project, mosaic 1.0, hsv_h/s/v, optimizer auto.
- Notebook **không có output nào được lưu** → không có metric, không có số ảnh/class cho plate.
- Diễn giải: `plate_best.pt` (2026-03-23, Ultralytics 8.4.25) đến từ một lần chạy sớm hơn với cấu hình mặc định mà notebook không còn;
  notebook hiện lưu là cấu hình "pro" về sau (yolo11s, augmentation riêng cho biển số, `fliplr=0` hợp lý cho biển số) nhưng **weights của cấu hình đó không phải file đang triển khai**.
- Hệ quả: (1) số plate duy nhất đáng tin vẫn là metadata của chính `plate_best.pt` (val mAP50 0.995 / mAP50-95 0.752, gộp, không test);
  (2) lần train lại plate ở P5 nên dùng đúng cấu hình notebook và coi như model mới (không phải tái lập);
  (3) `fliplr=0.5` trong checkpoint đang huấn luyện model plate trên ảnh biển số bị lật gương — có thể ảnh hưởng chất lượng OCR về sau.
