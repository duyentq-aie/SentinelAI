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

---

# baseline-v2 (power plan High performance, chi tiết từng phần)

Số v1 ở trên được giữ nguyên (Balanced, `metrics.json` ở thư mục này). v2 nằm ở `baseline-v2/{metrics.json,env.json,events.json}`.
Lệnh: `venv/Scripts/python.exe tools/benchmark.py --repeats 3 --detail --out experiments/EXP-001-baseline/baseline-v2`
(`benchmark.py` lấy `app.py` gốc từ tag `v0-baseline`, không dùng `app.py` wrapper hiện tại).

**Power plan:** đã chuyển sang *High performance* (GUID `3b852889-...`), cắm sạc; plan cũ là Balanced (`381b4222-...`), muốn đổi lại: `powercfg -setactive 381b4222-f694-41f0-9685-ff5bb260df2e`.
**Độ tái lập:** FPS 6.74 / 7.38 / 7.38 (trung vị **7.38**), lệch (max−min)/median = **8.7%** → *vẫn > 5%*; giữ nguyên số, không nới tiêu chí.
Lượt 1 chậm hơn hai lượt sau ~9% (suy đoán: GPU/CPU chưa đạt trạng thái ổn định, cache đĩa; chưa kiểm chứng). Không thấy throttle nhiệt
(nhiệt đỉnh 62–63 °C, SM clock thấp nhất 1500 MHz, công suất tối đa 22–28 W). Một phần lệch cũng đến từ chạy trên laptop có tiến trình nền.
High performance chỉ nâng FPS trung vị từ 7.0 (v1) lên 7.38, nằm trong biên dao động hai bên → **không kết luận được** plan có tác dụng.
Chế độ `--detail` có thêm `synchronize` và bọc hàm nên FPS có thể lệch nhẹ so với v1; dùng v2 để phân rã, dùng v1/v2 FPS chỉ để đối chiếu.

**Sửa nhầm lẫn của bảng v1:** dòng "Plate 17.1 ms" của v1 là thời gian **mỗi lần gọi** (1.73 lần/frame trung bình), không phải mỗi frame.
Mỗi frame, plate tốn ≈ 28.8 ms (v2). Vì vậy "~32 ms chưa phân bổ" của v1 phần lớn là 12.5 ms plate (do đếm theo lần gọi) + ~15 ms vẽ.

## Phân rã mỗi frame (trung vị của 3 lượt, ms/frame, tổng 135.6; "unallocated" = 2.3)

| Phần | mean | p50 | p95 | Ghi chú |
|---|---|---|---|---|
| Đọc frame | 2.13 | 2.03 | 3.26 | CPU |
| **COCO `.track` (tổng)** | 37.20 | 36.17 | 50.22 | |
| ├ preprocess | 1.52 | 1.43 | 2.35 | CPU (letterbox) |
| ├ inference | 12.05 | 11.17 | 18.44 | GPU |
| ├ postprocess (NMS) | 1.66 | 1.46 | 2.82 | |
| ├ **tracker callback** | 21.56 | 21.31 | 28.39 | CPU: copy + BoT-SORT + chọn lại kết quả |
| │ └ `tracker.update` riêng | 20.95 | 20.74 | 27.51 | |
| └ overhead khác | 0.44 | 0.39 | 0.62 | |
| **Helmet `.track` (tổng)** | 34.76 | 33.86 | 46.64 | |
| ├ preprocess / inference / postprocess | 1.42 / 11.96 / 1.62 | | | |
| ├ **tracker callback** | 19.54 | 19.26 | 25.80 | `tracker.update` 18.96 |
| └ overhead khác | 0.33 | | | |
| **Plate (tổng, 1.73 lần gọi/frame; 93% frame có ≥1 lần gọi)** | 28.76 | 27.16 | 62.93 | |
| ├ preprocess / inference / postprocess | 4.04 / 22.13 / 2.06 | | | tổng theo frame |
| └ overhead khác | 0.53 | | | |
| Vẽ (`draw_sleek_box`) | 15.05 | 14.64 | 25.66 | CPU; `img.copy()` + `addWeighted` toàn khung hình cho mỗi box |
| `cv2.resize` 1280×720 | 1.07 | 0.99 | 1.55 | |
| JPEG encode | 14.60 | 14.17 | 17.52 | CPU |
| polylines + putText + pointPolygonTest + HSV đèn + imwrite | ≈ 0.2 | | | không đáng kể |
| **Chưa phân bổ** | **2.34** | 2.29 | 3.45 | glue Python, `.cpu()`, vòng lặp |
| **Tổng frame** | **135.57** | 133.33 | 183.38 | |

Ngoài ra: GPU util trung bình (lấy mẫu mỗi 2 s) 16.5–18.3%; `process_cpu_time / wall` ≈ 1.0 (tiến trình dùng ~1 lõi, nhưng
số này **không phân biệt** CPU bận tính toán với CPU quay vòng chờ CUDA sync, nên chỉ là bằng chứng phụ).

## Kết luận GPU-bound / CPU-bound (đã đo, trừ ghi chú)
1. **GPU chỉ chiếm ≈ 34%** thời gian frame: inference COCO 12.1 + helmet 12.0 + plate 22.1 ≈ 46 ms/135.6 ms, và GPU rảnh ~83% thời gian → pipeline hiện tại **chủ yếu bị CPU/Python chặn**, không phải GPU-bound.
2. **CPU-bound: ≈ 54%** gồm tracker BoT-SORT (21.0 + 19.0 ≈ 40 ms, *lớn hơn* tổng thời gian inference COCO + helmet), vẽ 15 ms, JPEG encode 14.6 ms, đọc/resize 3 ms. (Suy đoán chưa kiểm chứng: phần lớn chi phí tracker là bù chuyển động camera GMC/optical flow của BoT-SORT; cần thử `gmc_method: none` hoặc ByteTrack ở bước đo sau.)
3. **Hệ quả cho Edge:** tăng tốc model riêng (TensorRT/INT8) chỉ giảm tối đa ≈ 46 ms → còn ≈ 90 ms/frame ≈ **11 FPS** (< mục tiêu 15). Muốn đạt ≥ 15 FPS (≤ 67 ms) phải đồng thời giảm tracker, vẽ/encode (chỉ vẽ khi cần, đẩy encode/vẽ ra luồng khác) và plate (chỉ chạy theo sự kiện).
