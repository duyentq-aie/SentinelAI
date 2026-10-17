# EXP-002 – Perception sạch: mọi model chạy trên frame gốc (P2 3.0b)

## Vấn đề (lỗi perception tồn tại từ code gốc)
Code gốc vẽ mọi thứ lên **cùng một frame** đang đưa vào model:
1. Khung đèn giao thông (`LIGHT: …`) được vẽ ngay sau COCO và **trước khi helmet model chạy**.
2. Overlay của các xe/đầu/vùng đèn đỏ được vẽ trước khi plate model xử lý crop xe, và **ảnh biển số lưu làm bằng chứng cắt từ frame đã vẽ**
   (thấy rõ viền "PLATE" xanh trong ảnh bằng chứng).
3. Bộ bù chuyển động camera của BoT-SORT nhận ảnh đã vẽ cho helmet tracker.

## Sửa
`Pipeline.process` tách hai frame: `raw` (đã resize, **không bao giờ bị vẽ**) cho COCO, helmet, plate, màu đèn và bằng chứng; `display` (bản copy) cho overlay và stream.
Không chỉnh luật nào để "bù" thay đổi này. Fixture cũ giữ làm `tests/fixtures/tracks_test_video_v0_overlay.jsonl`; fixture mới `tracks_test_video.jsonl`
(ghi trên CPU, 787 frame, sha256 video/model/tracker trong metadata).

## Perception thay đổi bao nhiêu (`tools/diff_tracks.py`, đầy đủ: `tracks_diff.json`)
| | COCO | helmet |
|---|---|---|
| detection cũ / mới | 21005 / 21005 | 5164 / 5164 |
| ghép được (cùng class, IoU ≥ 0.5) | 21005 | 5164 |
| mất / mới | 0 / 0 | 0 / 0 |
| đổi track id | 0 | 0 |
| frame có bất kỳ thay đổi | 0 / 787 | 0 / 787 |
| box dịch chuyển (mean / max) | 0.0 / 0 px | 0.0 / 1 px |
| trạng thái đèn khác nhau | 0 frame (UNKNOWN 783, GREEN 4 ở cả hai) | |

**Gần như không đổi**, và lý do đo được: đèn giao thông chỉ được phát hiện trong **14** detection (vùng nhỏ ở góc phải trên,
khoảng x 1121–1142, y 146–190), nên overlay "LIGHT" hầu như không có cơ hội làm nhiễu helmet model. Chỉ có 14/42010 detection COCO nằm trong 60 px quanh đèn
và 0/10328 detection helmet; không có detection nào đổi ở vùng đó. Trên video này lỗi *đúng về nguyên tắc nhưng hầu như vô hại về số liệu perception*.
Ở video có nhiều đèn hoặc đèn gần xe (ngã tư thật), tác động có thể lớn hơn — chưa đo được vì chưa có video đó.

## Sự kiện (`tools/diff_events.py`, baseline v0 → v1 clean)
- **23 sự kiện nhãn vi phạm: giữ nguyên** (không mất/thêm/đổi).
- **Chụp biển (plate_capture): 20 → 21.** Đây là tác động thật của việc plate model nhìn crop sạch:

| track | v0 (overlay) | v1 (sạch) | ý nghĩa |
|---|---|---|---|
| M_1 | không có | frame 23 | biển nay được tìm thấy (trước đó crop có overlay → bỏ sót) |
| M_77 | 60 | **35** | chụp sớm 25 frame, đúng lúc nhãn NO-HELMET xuất hiện (frame 35) |
| C_592 | 501 | 505 | +4 frame |
| C_735 | 584 | 577 | −7 frame |

Bằng chứng nay là crop sạch, không còn viền PLATE vẽ đè.
**Baseline sự kiện mới:** `tests/fixtures/baseline_events_v1_clean.json` (44 sự kiện: 23 nhãn + 21 chụp biển). Baseline v0 giữ lại để so sánh.
Test: luật legacy phải tái tạo đúng 23 nhãn của **cả hai** baseline (test offline), và hai baseline có cùng 23 nhãn.

## Chưa kiểm định
Các thay đổi này chỉ so với snapshot của chính pipeline, **chưa có ground truth**; không nói được "sự kiện nào đúng".
