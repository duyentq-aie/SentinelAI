# MORNING REPORT – đêm 2026-10-18

> **ĐÍNH CHÍNH (2026-10-18, sau khi bạn duyệt):** (1) mục "đạt 15 FPS" và câu "7 → 12 FPS cho cùng mã" dưới đây **không đúng**: 7.4 là v0, 12 là pipeline mới với khối lượng việc khác; biến thiên thật giữa phiên cùng mã là ~17–19% và FPS giảm ~40% khi 50% CPU bị chiếm (EXP-009). Từ nay chỉ báo tỉ lệ so với v0 đo cùng phiên. (2) GMC đã được đặt tắt mặc định (tạm thời). (3) Ảnh bằng chứng lưu chất lượng ≥ 95. Phần còn lại của báo cáo giữ nguyên như lúc viết.

**Tóm tắt:** 10/10 mục xong và đã commit (không push). `pytest` nhanh: **211 passed**. Test chậm (perception không đổi, CPU): **1 passed** (170 s, `pytest -m slow`).
Không có claim độ chính xác nào: vẫn **chưa có ground truth cho sự kiện**. Số đo là tương đối, trên một clip 26 s, máy RTX 3050 Laptop.

## Mục nào xong / dở / lỗi
| # | mục | kết quả | commit |
|---|---|---|---|
| 1 | max_attempts 90, tắt wrong_way, baseline v2 | xong. 90 tìm đủ 21/21 biển (30 mất 1) | 0aa1d87, 3d831e5 |
| 2 | luật vote cửa sổ (no-helmet, chở quá người) | xong; 16 test track giả; **chưa kiểm định**; diff so v2: −1 sự kiện (M_1), +2 (M_176 1 frame, M_702), 1 nhiễu overload bị loại | f42a066 |
| 3 | bảo mật demo | xong; 55 test; bind 127.0.0.1, không reload, whitelist nguồn, giới hạn upload | 0aa1d87 |
| 4 | tối ưu phần mềm | xong; **tracker đo nhưng KHÔNG đổi mặc định** | 0552187, d809f5b |
| 5 | ONNX | xong **phần CPU**; GPU/TensorRT không đo được (onnxruntime-gpu cần CUDA 12) | 715a721 |
| 6 | OCR | code + test xong và nối vào `events.jsonl`; **trên clip 720p đọc được 0/14 biển** (hộp biển ~26×22 px, crop lưu có viền ~46×40 px) | b86c881 |
| 7 | UI demo | xong; đã chạy server thật: stream, bảng sự kiện, thumbnail, HUD hoạt động | fa281b6 |
| 8 | công cụ tay + 2 hướng dẫn | xong; **GUI chưa được mở thử** (chỉ test phần logic) | e32f12c |
| 9 | README + REPORT v1 | xong | commit cuối |

**Lỗi/rủi ro còn nằm lại:** (1) hai GUI `camera_config.py` và `label_events.py` chưa chạy thử với cửa sổ thật; (2) OCR không đọc được trên footage hiện có; (3) `static/violations/*` đã bị tôi xóa nhầm tay (xem PROGRESS.md, chỉ là file demo cũ bị gitignore).

## Số liệu chính (chỉ so trong cùng một bảng/phiên đo; FPS tuyệt đối khác nhau giữa các phiên: 7 → 12 FPS cho cùng mã)
| bước | headless FPS (đọc + xử lý + vẽ) | stream FPS (+ JPEG) | nguồn |
|---|---|---|---|
| pipeline gốc v0 (tham chiếu) | 7.38 | – | EXP-001 baseline-v2 |
| plate chạy theo sự kiện: cũ → mới (gọi plate/frame 1.73 → 0.13) | 7.29 → 8.88 | 6.56 → 7.74 | EXP-003 (cũ dao động 17%) |
| max_attempts 30 → 90 | 10.58 → 10.26 (nhiễu) | 9.14 → 8.87 | EXP-004 |
| P4 trước (blend toàn khung, JPEG 95) | 12.05 | 10.34 | EXP-006 |
| + vẽ chỉ vùng nhãn | 14.64 (+21%) | 12.22 (+18%) | EXP-006 |
| + JPEG 80 (**mặc định hiện tại**) | 15.22 (+4%) | 12.85 (+5%) | EXP-006 |
| (+ BoT-SORT `gmc none`, **chưa nhận**) | 28.73 | 21.46 | EXP-006 |
| (+ ByteTrack, **chưa nhận**) | 28.21 | 21.07 | EXP-006 |
| CPU: PyTorch → ONNX Runtime | 4.65 → 4.97 | 4.34 → 4.52 | EXP-007 |

- Giả thuyết "tracker + GMC là phần tốn" được **xác nhận**: tắt GMC giảm ~15 ms/model (process 63.9 → 33.0 ms). Trên 26 s video, `gmc none` giữ nguyên 29/29 sự kiện (loại + thời điểm) và đổi ít ID (COCO 12/382 track bị tách), ByteTrack làm 1 sự kiện lệch 7 frame. Một clip không đủ để kết luận → đợi video dev có nhãn rồi mới đổi 1 dòng config.
- ONNX FP32 CPU cho **cùng detection** với PyTorch khi dùng đúng hình dạng đầu vào 384×640 (ONNX vuông 640×640 cho kết quả khác và chậm hơn), sự kiện giống hệt tới từng ID.
- ~~Mục tiêu "≥ 15 FPS ở 720p trên proxy GPU": đạt sát biên~~ — **bị rút lại** (xem đính chính ở đầu file).
- Server thật (1 client, mặc định, 1 lần đo không chính thức): 14–15 FPS xử lý, độ trễ 63–88 ms.

## Quyết định tôi tự chọn (đều đảo ngược được bằng 1 dòng config; lý do ở `docs/PROGRESS.md`)
`max_attempts` 90 · `wrong_way` tắt · engine luật `windowed` (W15/K5/r0.5 chưa tune) · `one_event_per_track` · JPEG 80 · blend `roi` ·
**không** đổi tracker · ONNX dùng onnxruntime CPU · OCR tắt mặc định, crop sát biển + phóng to 128 px · thêm `frame_size` vào sự kiện · camera config nhận `stop_line/direction/light_roi` ·
sửa lỗi chia cho 0 của overlay FPS (`time.time()` trên Windows).

## Việc bạn cần làm tay (theo thứ tự ưu tiên)
1. **Quay video** theo `docs/RECORDING_GUIDE.md`: ≥ 2 vị trí (≥ 1 ngã tư thấy rõ đèn + vạch dừng), **1080p**, ≥ 30 phút dev + ≥ 30 phút test (khác vị trí/buổi), ≥ 25% ban đêm, ~20 sự kiện mỗi loại ở test. *Đây vẫn là đường găng của cả dự án.* (~4–6 h)
2. Chạy `python tools/camera_config.py --video <clip> --camera-id <id>` cho mỗi vị trí (vạch dừng, hướng đi, ROI đèn, vùng) và **thử mở GUI** xem có lỗi gì không. (~0.5 h / vị trí)
3. Gán nhãn bằng `python tools/label_events.py` theo `docs/LABELING_GUIDE.md` (~8–12 h). Đây cũng là lần thử GUI thật đầu tiên.
4. Quyết định: nhận `gmc none` hay không (sau khi có video dev); giữ `max_attempts` 90; giữ JPEG 80; có cần OCR trong demo (cần video 1080p).
5. Cung cấp (nếu còn): dữ liệu `data/raw/` hai dataset Kaggle (để kiểm leakage và mAP thật), thiết bị Edge mục tiêu, hạn chót M1.

## Câu hỏi cho bạn
1. Thiết bị Edge thật bạn sẽ dùng là gì (Jetson / RPi / x86 NUC)? ONNX-CPU đã có; GPU/TensorRT cần biết thiết bị để chọn CUDA/TensorRT phù hợp.
2. Có chấp nhận cài CUDA 12 (hoặc dùng bản onnxruntime-gpu khác) để đo ONNX GPU không? Tôi không được phép đêm nay.
3. Mốc M1 và số giờ/tuần của bạn? (chưa từng được điền.)
4. OCR: bạn có thể quay biển ở độ phân giải cao hơn (cận cảnh) không? Nếu không, nên bỏ OCR khỏi phạm vi báo cáo.
5. Luật vượt đèn đỏ theo vạch dừng (3.3) cần video có đèn đỏ thật: clip hiện tại chỉ có 14 lần phát hiện đèn và không có sự kiện đèn đỏ nào.

## Chưa làm / ngoài phạm vi đêm nay
Luật vượt đèn đỏ theo vạch dừng (3.3), `tools/eval_events.py` (cần nhãn), kiểm leakage dataset (cần `data/raw/`), Dockerfile, INT8/FP16, TensorRT, huấn luyện lại.
