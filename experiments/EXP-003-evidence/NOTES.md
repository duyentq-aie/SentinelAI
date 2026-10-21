# EXP-003 – Bằng chứng đầy đủ + plate chạy theo sự kiện (P2 3.1)

## Summary (English)
- Purpose: store full evidence per event and run the plate detector only for violating tracks (event-driven) instead of on every frame.
- Result: the 23 violation labels and all 44 events are unchanged (0 missing / 0 added / 0 changed); plate calls per frame 1.729 -> 0.128 (-93%); headless FPS 7.29 -> 8.88 and stream FPS 6.56 -> 7.74 (old-policy spread 17%, new 5%).
- Conclusion: the event-driven policy removes most plate work without changing events, at the price of 1 plate lost on GPU (20/21 tracks, the missing one needs about 80 attempts versus max_attempts 30; that number is inferred, not recorded).
- Limits: no ground truth; FPS is not comparable with other sessions; both configurations stay below 15 FPS; the stored plate crop is about 54 x 43 px (the plate box itself about 26 x 22 px), small for OCR.

Thay đổi: `sentinel/evidence.py` (xem `docs/EVIDENCE.md`). Mỗi sự kiện (một loại vi phạm trên một track) lưu frame gốc, crop xe, crop biển và một dòng
JSONL theo schema nhãn ground truth của PLAN 3.1; sự kiện được ghi **kể cả khi không thấy biển**. Plate detector chỉ chạy cho track đang vi phạm,
tối đa `max_attempts` lần/track, dừng khi đã có biển. Thư mục bằng chứng bị giới hạn dung lượng (`max_dir_mb`, xóa thư mục sự kiện cũ nhất trước).
Luật **không đổi**. A/B so với chính sách cũ (`configs/pipeline/ab_plate_v0.yaml`: detector chạy mọi frame của xe vi phạm, không dừng).

## Tiêu chí: 23 sự kiện vi phạm giữ nguyên so với baseline v1 ✅
`tools/diff_events.py tests/fixtures/baseline_events_v1_clean.json <run>` (CPU, dung sai 0 frame), cả hai chính sách:
**MISSING 0 / ADDED 0 / CHANGED 0** trên toàn bộ 44 sự kiện (23 nhãn vi phạm + 21 chụp biển), tức cả frame chụp biển cũng không đổi.
(Tracks fixture/perception không đổi trong bước này: các luật và model không được chạm; test chậm perception chưa chạy lại ở 3.1 vì không có đường mã nào ảnh hưởng perception thay đổi — bằng chứng gián tiếp là hai lần chạy CPU ở trên ra đúng sự kiện nhãn như baseline.)

## Số liệu trước / sau (video test 787 frame)
| | chính sách cũ (v0) | chính sách mới (theo sự kiện) |
|---|---|---|
| **CPU** (ids = baseline): sự kiện bằng chứng / track vi phạm | 24 / 21 | 24 / 21 |
| CPU: track có biển | **21** | **21** (cần tối đa 19 lần thử, trung bình 3.4) |
| GPU: track vi phạm có biển | 21 / 21 | **20 / 21** (xem "mất 1 biển" dưới) |
| crop biển trung bình (W×H, px) CPU | 54.5 × 43.2 | 54.5 × 43.2 |
| crop biển trung bình GPU | 50.6 × 39.4 | 52.4 × 39.8 |
| crop xe trung bình (W×H) | 217 × 210 | 217 × 210 |
| **lần gọi plate / frame** (GPU) | **1.729** | **0.128** (−93%) |
| thời gian plate detector / frame (mean, ms) | 33.0 | 3.8 |
| `Pipeline.process` mean (ms) | 134.4 | 109.7 |
| **FPS headless** (đọc + process, không encode), trung vị 3 lượt | **7.29** (8.20 / 7.29 / 6.93; lệch 17.4%) | **8.88** (8.88 / 9.14 / 8.69; lệch 5.1%) |
| **FPS có stream** (+ JPEG encode, chưa tính truyền HTTP) | **6.56** (7.34 / 6.56 / 6.23; lệch 16.9%) | **7.74** (7.74 / 7.98 / 7.62; lệch 4.7%) |
| dung lượng bằng chứng (24 sự kiện, 73 file) | 6.9 MB | 6.9 MB (~0.29 MB/sự kiện; giới hạn 200 MB ≈ 700 sự kiện) |

Phần cứng: RTX 3050 Laptop 4 GB, cắm sạc, power plan High performance; PyTorch FP32, 720p, batch 1. `tools/benchmark_pipeline.py` (3 lượt, mỗi lượt một process, bỏ 30 frame warm-up).
Chi tiết từng stage: `bench_plate_*/metrics.json`. Chênh FPS +22% (headless) / +18% (stream) khớp với giảm 24.7 ms của `process`, tương ứng 29 ms bớt ở plate detector.

### Lưu ý khi đọc FPS
- Độ lệch giữa các lượt của chính sách cũ là **17%** (lượt 1 nhanh nhất rồi giảm dần khi GPU nóng 62 → 66 °C): con số 7.29 không chắc bằng 8.88 (lệch 5%). Hướng và độ lớn của cải thiện đáng tin vì khớp với mức giảm thời gian plate đo trực tiếp, nhưng con số chính xác thì không.
- **Không so trực tiếp với baseline-v2 (7.38 FPS)**: harness khác (`benchmark_pipeline.py` đọc video bằng timer riêng, bọc stage, đã tách `raw`/`display` thêm một lần `copy` 720p). Cùng mã, chính sách cũ đo được 6.56 stream — thấp hơn 7.38 của app gốc, nằm trong biên nhiễu nhưng cũng có thể do `copy` + ghi bằng chứng.
- Cả hai cấu hình đều **vẫn dưới 15 FPS**; thời gian còn lại nằm ở tracker + vẽ + encode (xem EXP-001/baseline-v2), chưa phải đối tượng của 3.1.

### Mất 1 biển ở chính sách mới (chỉ ở lần chạy GPU)
Trên lần chạy GPU, chính sách cũ tìm được biển cho 21/21 track, chính sách mới 20/21. Suy ra từ trung bình số lần thử (6.83 so với 3.18): một track chỉ có biển
nhìn thấy được sau khoảng **80 lần thử** (ngưỡng hiện tại `max_attempts: 30` bỏ cuộc trước đó). Trên lần chạy CPU không có trường hợp này (tối đa 19 lần thử).
Đây là đánh đổi có thật: tăng `max_attempts` (60–90) đổi lấy thêm vài chục lần gọi detector cho những track khó, ít tốn kém hơn nhiều so với 1.73 lần/frame. **Chưa đổi mặc định**; cần bạn quyết định.
(Con số ≈ 80 là suy ra từ trung bình; không có bản ghi từng track của lần chạy GPU vì thư mục bằng chứng tạm đã xóa.)

## Phát hiện phụ
- **[đính chính 2026-10-18]** "54 × 43" là kích thước **ảnh crop lưu `plate.jpg`** (hộp biển + viền `plate_margin` = 10 px mỗi phía, bị cắt ở mép khung), KHÔNG phải kích thước biển. Hộp biển do detector trả về đo được ≈ **26 × 22 px** (EXP-009/010, `mean_plate_box_wh`); crop lưu ≈ 46 × 40 px ở cấu hình hiện tại (số 54 × 43 thuộc chính sách/tracker cũ, chọn frame khác). Crop biển lưu trung bình ≈ 54 × 43 px (xem đính chính) ở 720p: nhỏ cho OCR (P3). Quay 1080p hoặc chọn frame tốt nhất trong nhiều lần thử (thay vì dừng ở biển đầu tiên) có thể cần thiết.
- 24 sự kiện bằng chứng: 13 `no_helmet`, 2 `overload`, 9 `wrong_way`, **0 `red_light`** — video test này không có vượt đèn đỏ nào được luật cũ bắt (đèn chỉ hiện ra 14 lần, UNKNOWN 783/787 frame).
- Khung `PLATE` trên video stream chỉ vẽ ở frame detector chạy (không còn mọi frame).

## Chưa kiểm định
Chỉ so với snapshot của chính pipeline; chưa có ground truth để nói sự kiện nào đúng. `tools/eval_events.py` chưa tồn tại (sẽ dùng schema này).
