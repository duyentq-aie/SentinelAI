# EXP-001 – Baseline pipeline gốc (app.py @ v0-baseline)

Lệnh: `venv/Scripts/python.exe tools/benchmark.py --repeats 3` (từ thư mục gốc repo).
Giao thức: 1 lượt qua `test_video.mp4` (757 frame đo, 30 frame warm-up bỏ), mỗi lần lặp trong process mới;
thời gian frame = `next(generate_frames())` (đọc + 3 model + vẽ + lưu bằng chứng + JPEG encode); FP32, batch 1, 1280x720.

Môi trường: xem `env.json`. Nguồn: cắm sạc, scheme Windows "Balanced" (chưa ép High Performance). GPU: nhiệt đỉnh 61 °C,
SM clock thấp nhất 1500 MHz, công suất tối đa ≈ 20–23 W (3 lượt) → không thấy throttle nhiệt.

Độ tái lập: 3 lượt FPS = 7.00 / 6.83 / 7.29 (lệch 6.6%, hơi trên mục tiêu 5%). Hai lần chạy trước (FPS 5.5–7.1, lệch ~26%)
bị **loại**: hai benchmark đã chạy chồng nhau trên GPU. Bản này chạy một mình.

Hạn chế: `events.json` là snapshot đặc tả hành vi (43 mục/lượt, ổn định giữa 3 lượt), không phải ground truth.
Frame time gồm cả ghi ảnh vi phạm ra đĩa; chưa tách riêng (nằm trong phần còn lại ngoài các stage đã đo).
Plate model được gọi 1.73 lần/frame trung bình (chạy lại trên mọi xe vi phạm ở mọi frame, kể cả khi đã chụp).
