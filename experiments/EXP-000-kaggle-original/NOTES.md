# EXP-000 – Huấn luyện gốc trên Kaggle (lịch sử, không tái lập được)

**Trạng thái: CHƯA ĐẦY ĐỦ.** Notebook helmet/plate **chưa có trong repo** (không có `notebooks/`),
nên số liệu *val + test theo class* của notebook helmet **chưa được ghi**. Tôi không tự bịa chúng.
`metrics.json` cùng thư mục chỉ chứa thứ khôi phục được từ metadata trong checkpoint (`train_metrics`,
`train_args`, `train_results`): metric gộp mọi class, trên tập **val**, cùng tập dùng để chọn `best.pt`.

## Việc còn thiếu (cần bạn cung cấp)
- Bảng val + test **theo class** của notebook helmet (P, R, mAP50, mAP50-95 cho helmet / motorcycle / no-helmet), và tên split "test" nếu có.
- File notebook (đã xóa output ảnh lớn) vào `notebooks/`. Khi có, bổ sung vào `metrics.json` mục `helmet.per_class` và `helmet.test`.

## Ghi chú về notebook plate (theo lời bạn, chưa xác minh)
Bạn cho biết **notebook plate không khớp checkpoint** `plate_best.pt`. Tôi chưa xem notebook nên không kiểm chứng được.
Hệ quả khi dùng làm bằng chứng:
- **Số trong notebook plate không được dùng làm baseline.** Nguồn số duy nhất cho plate là metadata của chính `plate_best.pt`
  (YOLO11n, 50 epoch, best epoch 46, val mAP50 0.995 / mAP50-95 0.752, ultralytics 8.4.25, 2026-03-23).
- Cần ghi rõ khác biệt khi biết cụ thể (epoch? model? dataset? ngày?) để không tuyên bố sai trong báo cáo.

## Thứ khôi phục được từ checkpoint
Xem `metrics.json`: hyperparameters đầy đủ, đường dẫn data trên Kaggle, tên class, best epoch, tổng thời gian train.
Không có: số ảnh/nhãn, per-class, test, confusion matrix, git commit (`git: None`).
