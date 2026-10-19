# EXP-023 – vì sao track xe máy của helmet model chết trước vạch dừng (test_video) – chẩn đoán, chưa sửa gì

Câu hỏi (từ EXP-022): luật `stop_line_crossing` thấy 0 lần cắt vạch trên 63 track xe máy; track gần như không bao giờ tới vạch (khoảng cách gần nhất trung vị -168 px). Mã: `diagnose.py` (số liệu thô và kết quả: `raw_detections.json`, `results.json`), `coco_source_check.py` (what-if, `coco_source_crossings.json`), ảnh dải M_257 `m257_strip.jpg` (không commit: có người). Pipeline mặc định, 787 frame, 30 FPS; pha ĐỎ = frame 0–279 (0–9.3 s), XANH = từ frame 280. "Phía trên vạch" = y đáy hộp < 290 (vạch ở y 277–302). Detection thô = `predict` không tracker.

## (a) Có ROI / mask / lọc theo vị trí nào cắt phần trên khung không? **Không.**
Kiểm tra mã và config:
- `frame_size: [1280, 720]`: mọi frame chỉ bị resize, không crop, không mask (`Pipeline.process`).
- `Detectors._track_kw`: chỉ truyền `conf`, tracker, device; không có `classes`, `imgsz` tùy biến, ROI.
- `sentinel/perception.py::tracks_from_results`: chuyển nguyên hộp, không lọc vị trí.
- Chỗ duy nhất lọc theo vị trí là các luật: `violation_zone` (luật vùng cũ), `wrong_lane_zone` (null), vạch dừng/làn của luật mới, và việc gán đầu vào hộp xe máy (đầu nằm trong hộp). Không chỗ nào loại hộp theo vị trí.
- Chứng cứ số liệu: COCO (cùng frame, cùng pipeline) thấy 0.7–0.9 xe máy mỗi frame ở dải y 200–280 với conf 0.35 (pha xanh; pha đỏ: `results.json`), nên phần trên khung **không bị cắt**.
**Giả thuyết (a) bị bác bỏ.**

## (b) Detector có thấy xe máy phía trên vạch không? Với helmet model: **gần như không**; với COCO: có
Số detection thô lớp `motorcycle` (đếm theo hộp, y đáy < 290):

| model, conf | pha ĐỎ (280 frame) | pha XANH (507 frame) |
|---|---|---|
| helmet_best.pt, 0.55 (pipeline) | **0** / 811 detection | 13 / 1673 |
| helmet_best.pt, 0.1 | 10 / 2942 | 57 / 4475 |
| COCO yolo11s.pt, 0.35 (pipeline) | 831 / 3053 (258 frame có ≥1) | 1025 / 6315 (428 frame có ≥1) |
| COCO yolo11s.pt, 0.1 | 3033 / 10105 | 5522 / 17746 |

Phân bố theo dải 40 px (số detection mỗi frame) có trong `results.json`; ví dụ pha XANH, conf pipeline, dải 240–280: helmet 0.002, COCO 0.68; dải 200–240: COCO 0.93 (helmet không có); pha ĐỎ, conf 0.1, dải 200–240: COCO 6.3. Track của pipeline (helmet, 0.55): **0 detection phía trên vạch trong pha đỏ, 7 trong pha xanh**.
Ngay cả hạ conf xuống 0.1, helmet model chỉ có 10 + 57 detection phía trên vạch trong 787 frame (COCO: 8555 ở conf 0.1). **Giả thuyết (b) đúng cho helmet model**: nó không phát hiện xe máy ở phía xa/gần vạch trở lên. (Chưa kiểm tra nguyên nhân: xe nhỏ/xa so với dữ liệu train, độ phân giải 640 mặc định của Ultralytics; không đoán.)

## (c) Tracker có bỏ rơi đối tượng mà detector còn thấy không? **Hiếm (≈5%)**
61 track kết thúc (không tính 2 track còn sống ở 3 frame cuối video). y đáy nơi track kết thúc (`results.json`): 280–320: 6, 320–360: 13, 360–400: 5, 400–440: 5, 440–480: 10, 480–520: 7, 520–560: 1, 560–600: 5, 640–760: 9: **rải dần, không có ngưỡng sắc nét** (không giống một mask/cắt khung ở một y cố định). Với mỗi track kết thúc, xét 3 frame sau (IoU ≥ 0.3 với hộp cuối):
- 58/61: **không có detection ở conf pipeline (0.55)** gần hộp cuối. Trong đó **56/61 còn một detection conf thấp** (0.1 ≤ conf < 0.55) gần đó: track chết vì detector tụt dưới ngưỡng 0.55, không phải vì tracker ghép sai.
- 2/61: detection còn đó nhưng đã thuộc một track khác (đổi ID/hợp nhất).
- 1/61: có detection chưa ai nhận (lỗi ghép của tracker).
**Giả thuyết (c) giải thích ~3–5%** số track chết; phần lớn (56/61 ≈ 92%) là conf < 0.55 của helmet model, cộng với việc helmet model không thấy xe máy ở vùng gần vạch (b).

## M_257 kiểm tay (`m257_strip.jpg`, frame 210–260 mỗi 5)
- Helmet model: hộp M_257 chỉ sống ở frame 215–221 vì conf dao động 0.28 (f210) → 0.57 → 0.69 → 0.13 (f225): rơi dưới 0.55 nên track chết. Hộp 111×130 px **lệch và lớn hơn nhiều** người lái (~35×70): điểm giữa-đáy của hộp helmet là một xấp xỉ kém cho vị trí xe.
- COCO: thấy người lái đó ở **mọi frame 220–280**, conf 0.50–0.60, hộp gọn. y đáy: 336 (f220) → 333 → 330 → 328 → 325 → 321 → 318 → 316 → 312 (f260) → 309 → 305 → 302 (f275) → 298 (f280): **~0.6 px/frame ≈ 19 px/s**, xe nhích chậm lên vạch (vạch ở y ≈ 287 tại x ≈ 783) trong lúc đèn đỏ, **chưa qua vạch khi đèn chuyển xanh (f280)**. Đây là xe chờ đèn, **không phải ca vượt đèn**; mô tả "đang chạy" ở bản đầu README EXP-022 là sai và đã đính chính. Luật vùng cũ báo nhầm; luật mới đúng khi không báo.

## What-if: luật mới nhận track xe máy của COCO thay vì helmet (`coco_source_check.py`, chưa đổi pipeline)
Cùng tracker yaml, conf 0.35, hình học đã duyệt, đèn ROI (EXP-022): 242 track (ID vỡ nhiều hơn), **99 track có lúc ở sau vạch** (helmet: 2), **13 lần cắt vạch**: 9 lúc XANH (không báo vì đèn xanh hoặc làn exempt), 4 lúc ĐỎ **đều ở làn right-turn (exempt)**: **0 lần báo**. Theo làn: through 5 (cả 5 lúc xanh), right-turn 7, left-turn 1 (lúc xanh). Không có nhãn thật nên không nói các lần cắt này đúng đến đâu, chỉ cho thấy nguồn track COCO **tới được vạch** còn nguồn helmet thì không.

## Kết luận (theo bằng chứng)
| giả thuyết | giải thích | bằng chứng |
|---|---|---|
| (a) mask/ROI/lọc vị trí | 0% | không có trong mã/config; COCO thấy xe máy phía trên cùng frame |
| (b) detector (helmet model) không thấy / conf thấp ở gần vạch trở lên | phần lớn: 56/61 track (≈92%) chết do conf < 0.55 gần hộp cuối; phía trên vạch chỉ 0–13 detection ở conf pipeline (COCO: 831–1025) | bảng (b), (c) |
| (c) tracker | ≈3–5% (3/61) | mục (c) |
Phụ: hộp helmet model rộng/lệch (M_257: 111×130 so với người lái ~35×70), nên điểm giữa-đáy kém tin cậy kể cả khi track còn sống.

## Đề xuất sửa (CHƯA làm, chờ duyệt)
1. **Cho luật `stop_line_crossing` dùng track xe máy của COCO** (pipeline đã chạy COCO tracker mỗi frame; hiện luật chỉ lọc `M_` của helmet model). Hộp COCO gọn, thấy xe tới tận vạch (what-if: tới được vạch 99 track). Luật helmet/overload giữ nguyên với helmet model. Cần: id riêng (ví dụ `K_`) để không trùng `M_`, ghép lý do với helmet khi hộp chồng nhau (IoU), và một kiểm tra rằng ID vỡ (242 track) không tạo sự kiện kép: `one_event_per_track` + `_fired` đã chặn theo track nhưng một xe có thể đổi ID sau khi cắt vạch.
2. Không khuyên hạ `models.helmet.conf`: hạ xuống 0.1 vẫn chỉ cho 10 + 57 detection phía trên vạch, và đổi conf ảnh hưởng luật helmet/overload (EXP-005/006).
3. Để đo thật: cần video có nhãn vượt đèn (hiện clip này chỉ kiểm được "không báo oan"), và nhãn xe máy ở vùng xa nếu muốn cải thiện detector (không train trong đợt này).
