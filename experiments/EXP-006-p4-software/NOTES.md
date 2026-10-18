# EXP-006 – P4 tối ưu phần mềm (đêm 2026-10-18, mục 4)

Mỗi thay đổi là một cấu hình riêng (`configs/pipeline/exp/p4*.yaml`), đo bằng `tools/benchmark_pipeline.py` với **5 cấu hình chạy xen kẽ** (vòng 1: cả 5, vòng 2: ..., 3 vòng) để hiện tượng nóng dần/biến thiên của máy ảnh hưởng đều.
GPU (RTX 3050 Laptop 4 GB), PyTorch FP32, 720p, batch 1, High performance. Đã bỏ 30 frame warm-up. `headless` = đọc + `Pipeline.process` (có vẽ, không encode); `stream` = + JPEG encode (chưa tính truyền HTTP).
Bảng: trung vị 3 lượt, (khoảng min–max).

| # | thay đổi | headless FPS | stream FPS | `process` mean | so với bước trước |
|---|---|---|---|---|---|
| 0 | **before** (blend toàn khung, JPEG 95) | 12.05 (11.73–13.19) | 10.34 (10.08–11.24) | 80.6 ms | — |
| a | vẽ: chỉ blend vùng nhãn (không copy cả khung) | 14.64 (14.51–15.15) | 12.22 (12.15–12.60) | 66.4 ms | **+21.5% / +18.2%**; `draw` 11.9 → 0.40 ms |
| b | + JPEG chất lượng 80 (thay 95) | 15.22 (14.57–15.25) | 12.85 (12.38–12.89) | 63.9 ms | +4% / +5% (khoảng chồng một phần); `encode` 13.4 → 12.1 ms (−10%) |
| c1 | BoT-SORT với `gmc_method: none` | **28.73** (28.55–28.82) | **21.46** (21.30–21.53) | 33.0 ms | **+89% / +67%**; COCO 31.5 → 16.3 ms, helmet 29.1 → 13.5 ms |
| c2 | ByteTrack (cả hai model) | 28.21 (23.60–29.20) | 21.07 (17.82–21.71) | 33.5 ms | tương đương c1; khoảng rộng (lượt 1 chậm 23.6) |

(c1 và c2 so với b; cả hai cùng cấu hình a+b.)

- **(a) Vẽ.** Bản cũ `img.copy()` + `addWeighted` toàn khung hình cho *mỗi* nhãn (≈ 12 ms/frame). Bản mới chỉ blend hình chữ nhật của nhãn. Cùng ảnh trong vùng nhãn và viền; phép blend toàn khung của bản cũ còn làm lệch ±1 mức sáng vài pixel ngoài nhãn, bản mới không (test: `tests/test_viz.py`, ≤ 12 pixel khác biệt, ±1, trên 200 hộp ngẫu nhiên).
- **(b) Encode chỉ khi có client.** Đã đúng từ thiết kế worker (bước 2): worker không bao giờ encode; chỉ client đang hỏi frame mới kích hoạt encode, một lần/frame (`tests/test_worker.py`). Phần đo ở đây chỉ là chất lượng JPEG (`web.jpeg_quality`, mặc định 80): giảm ~1.3 ms/frame, ~10%. **FPS headless và stream được báo riêng ở mọi dòng.**
- **(c) Tracker.** Giả thuyết trong PLAN ("chi phí tracker chủ yếu là bù chuyển động camera GMC") **được xác nhận bằng số đo**: tắt GMC giảm mỗi model ~15 ms/frame (cả hai ~31 ms; `process` 63.9 → 33.0 ms). Camera cố định nên GMC không cần, nhưng đây là thay đổi **ảnh hưởng ID**, xem dưới.

## Chất lượng của thay đổi tracker (CPU, so với baseline v3 / fixture tracks, `tools/diff_events.py`, `tools/diff_tracks.py`)
| | BoT-SORT mặc định | gmc_none | ByteTrack |
|---|---|---|---|
| sự kiện (so với v3, bỏ qua ID, ±3 frame) | 29 | **29, khớp hết** (0 vấn đề) | 29, **1 sự kiện lệch** (frame 474 → 467, sớm 7 frame) |
| COCO detection ghép được (IoU ≥ 0.5) | — | 20966 / 21005 (99.8%), 39 mất, 40 mới | 20743 / 21005 (98.8%), 262 mất, 225 mới |
| box COCO dịch chuyển (mean / max) | — | 0.29 / 14 px | 3.32 / 169 px |
| track COCO bị tách thành > 1 id mới (≥ 5 frame mỗi id) | — | 12 / 382 | 34 / 382 |
| track COCO mới gộp > 1 id cũ | — | 15 / 378 | 28 / 391 |
| track helmet bị tách / gộp | — | 2 / 174, 0 | 2 / 174, 0 |
| thay đổi detection gần đèn giao thông (≤ 60 px) | — | 7 trên 11482 | 8 trên 15284 |

(Đánh số id khác hoàn toàn giữa hai tracker nên "id-changed" thô ≈ 54% detection là vô nghĩa; các dòng "tách/gộp" là phép đo đúng.)

**Kết luận: tạm thời, chưa chấp nhận. Không đổi mặc định** (đúng quy tắc đêm nay). Lý do: chưa có video dev có nhãn để kiểm P/R sự kiện, và ID switch ảnh hưởng trực tiếp tới luật (vote theo track, chống trùng theo track).
Bằng chứng ban đầu trên 26 s video: `gmc_none` giữ nguyên toàn bộ 29 sự kiện (loại và thời điểm) và đổi rất ít ID; `ByteTrack` đổi nhiều hơn (một sự kiện lệch 7 frame). Một clip ngắn không đủ để chứng minh.
Việc cần làm khi có dev video: chạy `tools/eval_events.py` (chưa có) với cấu hình `exp/p4c_gmc_none.yaml`; nếu F1 sự kiện không giảm thì đổi một dòng trong `configs/pipeline/default.yaml` (`tracker: configs/trackers/botsort_gmc_none.yaml`).

## Lưu ý đọc số
- FPS tuyệt đối của phiên này (12 → 15 → 28) **cao hơn** EXP-003/004 (khoảng 9–10 cho cùng mã pipeline trước tối ưu) vì máy ở trạng thái khác (nhiệt đỉnh GPU 59–61 °C, thấp hơn các phiên trước). **Không so số giữa các phiên**; chỉ so các dòng trong cùng bảng (xen kẽ cùng phiên).
- Cải tiến (a) và (b) giữ nguyên toàn bộ 14 sự kiện bằng chứng (13 `no_helmet`, 1 `overload`) ở cả 5 cấu hình trên GPU (đếm theo loại), nhưng ID giữa GPU và CPU khác nhau nên việc so từng sự kiện đã làm bằng CPU như bảng trên.
- ~~Mục tiêu PLAN "≥ 15 FPS headless 720p trên proxy GPU": đạt ở cấu hình (b)…~~ **ĐÍNH CHÍNH 2026-10-18 (EXP-009):** không được kết luận "đạt 15 FPS" từ phiên này. Phiên này không có v0 làm mốc và FPS tuyệt đối đổi ~40% theo tải nền CPU; chỉ các tỉ lệ so với v0 cùng phiên (EXP-009/EXP-010) mới đáng tin. **Chưa kiểm định độ chính xác** của (c).
- ~~"FPS tuyệt đối cao hơn EXP-003/004 cho cùng mã pipeline"~~: không cùng mã/khối lượng (EXP-003/004 có wrong-way bật và 0.13–1.7 lần gọi plate/frame, EXP-006 có 0.018) — xem EXP-009.
- Phần còn lại ở cấu hình b (63.9 ms): COCO 31.5 + helmet 29.1 (đều gồm ~15 ms tracker), encode 12.1 (chỉ khi có client), đọc 1.8.
