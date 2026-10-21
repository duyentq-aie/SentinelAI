# NOTES – EXP-016 (4cls)

## Lần chạy đầu tiên bị treo
Lần chạy 4cls đầu tiên trên Kaggle **bị treo hơn 3.9 giờ với output 0 B** và đã bị dừng tay. Không có log hay kết quả nào của lần đó được lưu; nguyên nhân chưa biết (nghi ngờ nghẽn nạp dữ liệu với `workers: 8` trên máy Kaggle, **chưa xác minh**). Lần chạy lại (kết quả ở đây) đặt `workers: 2` và `time: 2.5` để vừa tránh nghẽn vừa chặn thời gian.

## Cấu hình của lần chạy này
`zip` không chứa log huấn luyện, chỉ có `args.yaml` và `results.csv`, nên cấu hình dưới đây đọc từ `helmet_4cls_splitA/args.yaml`:
- model `yolo11s.pt`, dữ liệu split A 4 lớp, `epochs: 150`, `patience: 25`, `batch: 32`, `imgsz: 640`, `seed: 0`, `deterministic: true`, optimizer `auto`, `cos_lr: true`, `close_mosaic: 10`, `mosaic: 1.0`, `mixup: 0.1`, `degrees: 10.0`;
- **khác EXP-015:** `fliplr: 0.0` (EXP-015: 0.5), `workers: 2` (8), `time: 2.5` (không đặt), `name: helmet_4cls_splitA`; 4 lớp thay vì 3.
- Hệ quả của `time`: Ultralytics tính lại số epoch từ thời gian mỗi epoch, nên lịch cosine dài ≈ 178 epoch (suy từ `lr/pg0`) thay vì 150; lần chạy dừng sớm ở epoch 80 bởi patience, trước khi lịch kết thúc.
- Notebook đã dùng cho lần chạy này **không có trong repo** (không được gửi), nên chưa đối chiếu được từng ô; xem `kaggle/` để biết phần mới.
