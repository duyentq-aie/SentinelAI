# EXP-022 – trạng thái đèn từ ROI trên test_video (đã xác nhận 2026-10-19)

`configs/cameras/test_video.yaml` có `light_roi: [590, 0, 632, 22]`: bộ đếm lùi của cột đèn ở trên giữa khung 1280×720. **Chủ dự án đã xem ảnh dải và xác nhận bộ đếm này khớp pha đèn của làn trong khung** (nhãn "provisional" đã bỏ; việc báo kết quả `red_light` được phép). Đề xuất ban đầu: `PROPOSAL.md`. Mã: `Pipeline` đọc đèn từ ROI khi có (log `[lights] traffic-light source: roi`).

## Kết quả `red_light` trên `test_video.mp4` (pipeline mặc định, `helmet_best.pt`, 30 FPS, 787 frame)
- **1 sự kiện `red_light`** (15 sự kiện tổng: 13 no_helmet, 1 overload, 1 red_light). Các sự kiện no_helmet và overload **không đổi** so với lượt chạy trước khi có `light_roi` (cùng 14 sự kiện, cùng frame bắt đầu và kết thúc).
- Sự kiện: `M_257`, frame 215–221 (7.17–7.37 s), đèn RED, xe máy, hộp `[712, 220, 825, 350]`. Xem 6 frame (195…245): đây là **một xe máy đang chạy** qua vạch dành cho người đi bộ, đi xa dần khỏi camera, trong khi đèn còn đỏ (đỏ đến 9.3 s) và hàng xe phía trước đứng chờ. **Chưa biết đây là vượt đèn thật hay xe thuộc làn có tín hiệu riêng** (ảnh có các đèn mũi tên rẽ trái/phải riêng): cần bạn quyết, không tự gán đúng/sai.
- Luật vùng hiện tại (`violation_zone`, giữ nguyên từ v0) báo mọi xe có điểm giữa-đáy nằm trong vùng khi đèn đỏ, **không xét chuyển động hay hướng đi**; vì vậy một xe đứng yên trong vùng cũng sẽ bị báo (ở clip này hàng xe chờ đứng phía dưới vùng nên không bị báo).

## Giới hạn của clip này (ghi theo yêu cầu của chủ dự án)
Đèn đỏ chỉ có trong **0–9.3 s** và phần lớn xe đứng chờ, nên clip này chỉ kiểm được "**không báo oan khi mọi người dừng đúng**" (hàng xe chờ không bị báo); **chưa kiểm được khả năng bắt người vượt đèn thật**: không có nhãn `red_light`, chỉ một xe chạy qua trong lúc đỏ (chưa xác nhận là vi phạm), và không thể tính P/R/F1. Muốn đo cần video có nhiều ca vượt đèn đỏ đã gán nhãn (docs/LABELING_GUIDE.md).

## Timeline (`timeline.py` → `light_timeline.csv`, mỗi frame 1 dòng)
- **ROI (OpenCV):** RED ở frame 0–279 (0–9.3 s), GREEN từ frame 280 (9.33 s) đến hết: **một lần đổi trạng thái duy nhất, không nhấp nháy**; đầy đủ 787 frame có kết luận (không UNKNOWN). Đếm điểm ảnh đỏ/xanh theo frame có trong CSV (`roi_red_px`, `roi_green_px`).
- **COCO (detector, bản ghi mặc định):** `UNKNOWN` ở 783/787 frame, `GREEN` ở 4 frame (306–312, một đèn cột khác ở x ≈ 1122–1141): nguồn này gần như không cho thông tin trên clip.
- Ảnh dải (`light_strip_1..7.jpg`, mỗi 0.5 s; hàng trên: crop ROI phóng 6×, trạng thái dự đoán bên dưới; hàng dưới: vùng `violation_zone` cộng 40 px phía trên cùng thời điểm; **không commit** vì có người): đọc bằng mắt, chữ số đỏ đếm lùi 9 → 0 trong 0–9 s, rồi xanh "29" từ 9.5 s; trong 0–9 s hàng xe máy phía trước đứng dồn ở mép dưới vùng, từ 9.5 s đã bắt đầu dịch chuyển (xem `light_strip_3.jpg`). Điều này **ủng hộ** nhưng không chứng minh bộ đếm này chi phối đúng hướng của hàng xe đó.

## Lỗi thiết kế của luật vùng cũ và luật mới `stop_line_crossing` (thiết kế lại sau khi có ca M_257)

**Luật vùng cũ** (nay là `rules.red_light.mode: legacy_zone`, giữ để test đặc trưng chạy được): xe bị báo khi điểm giữa-đáy nằm trong `violation_zone` **và** đèn đỏ ở frame đó. Hai lỗi thiết kế:
1. **Báo cả xe đứng yên**: luật chỉ nhìn vị trí, không nhìn chuyển động. Một xe dừng đúng nhưng đầu xe chồm vào vùng (hoặc hàng chờ dồn lên vùng) vẫn bị báo "vượt đèn" suốt thời gian đỏ. Xe đã qua vạch từ lúc đèn còn xanh rồi đang đi tiếp khi đèn chuyển đỏ cũng bị báo.
2. **Không biết làn**: một tín hiệu chung cho cả vùng. Làn rẽ phải/trái có đèn mũi tên riêng (ảnh test_video có biển "Làn rẽ trái", "Làn rẽ phải" với đèn mũi tên riêng) bị áp đèn của làn đi thẳng.

**Ví dụ M_257** (xem mục kết quả ở trên): xe máy đang chạy qua vạch người đi bộ khi đèn đỏ ở frame 215–221. Luật cũ báo vì điểm giữa-đáy rơi vào vùng, không phân biệt được "vượt vạch dừng khi đỏ" với "đang ở trong vùng", cũng không biết xe đó thuộc làn nào (có thể là làn rẽ phải có tín hiệu riêng). Không tự gán đúng/sai cho ca này; luật mới làm cho câu hỏi "làn nào, vượt vạch hay chưa" trả lời được.

**Luật mới** `stop_line_crossing` (`sentinel/rules/stopline.py`, mặc định): vi phạm khi điểm giữa-đáy của track **cắt vạch dừng** theo hướng vào (`camera.stop_line`, `camera.direction`) **trong lúc đèn của làn đó là RED**. Khoảng cách có dấu tới vạch, vùng chết ±`jitter_px` (4 px) chống rung; "cắt" = đổi phía từ trước vạch sang sau vạch giữa hai quan sát. Không phải vi phạm: xe đứng yên, xe đã qua vạch trước khi đèn đỏ, đi ngược hướng, cắt vạch lúc xanh, làn `exempt`. Tối đa một vi phạm cho mỗi track; kết luận được giữ `hold_s` (1 s) để recorder mở đúng một sự kiện. Làn (`camera.lanes`): polygon + `signal: roi | exempt`; xe thuộc làn chứa điểm cắt vạch; không khai báo làn thì tín hiệu chung áp cho cả vạch (có cảnh báo trong log); điểm cắt nằm ngoài mọi làn thì **không báo** và được đếm (`unassigned_crossings`). Lý do ghép bằng " & " (ví dụ `NO-HELMET & RED-LIGHT RUNNER`) để không che luật helmet/overload. Test: `tests/test_stopline_rule.py` (track tổng hợp, không cần video).

**Trạng thái trên test_video:** `configs/cameras/test_video.yaml` **chưa có** `stop_line`/`direction`/`lanes`, nên luật mới **không hoạt động** (log cảnh báo `INACTIVE`) và pipeline mặc định hiện không báo `red_light` trên clip này cho đến khi bạn duyệt đề xuất. Đề xuất (chưa vào config) vẽ ở `lanes_proposal.jpg` (gitignored), tái tạo bằng `draw_lanes_proposal.py`:
- `stop_line`: `[[400, 302], [1040, 277]]` (sát phía dưới mép gần của vạch người đi bộ; hàng xe chờ ở dưới vạch này);
- `direction`: `[[650, 430], [665, 200]]` (đi lên khung hình, hướng vào ngã tư);
- làn: `left-turn` (signal `roi`, x 400–560), `through` (`roi`, x 560 đến vạch vàng chéo), `right-turn` (`exempt`, phần bên phải vạch vàng chéo, có đèn mũi tên xanh riêng).
Chưa chắc: ranh giới left-turn/through (biển ở phía xa, không thấy vạch kẻ làn bên dưới hàng xe), và đèn mũi tên rẽ trái có cùng pha với bộ đếm hay không.

## Còn mở
1. Sự kiện `M_257`: vượt đèn thật hay làn có tín hiệu riêng? (cần bạn xác nhận; luật mới `stop_line_crossing` cần hình học vạch/hướng/làn của test_video được duyệt).
2. ROI nhỏ (42×22 px) sát mép trên khung hình: nếu camera dịch thì lệch; chưa có kiểm tra tự động.
