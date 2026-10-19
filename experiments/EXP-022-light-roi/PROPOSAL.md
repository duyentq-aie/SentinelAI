# EXP-022 – trạng thái đèn từ ROI (đề xuất ban đầu; ĐÃ ĐƯỢC DUYỆT 2026-10-19, xem README.md)

## Đã làm trong mã (không đổi hành vi khi camera chưa có `light_roi`)
`Pipeline` đọc trạng thái đèn từ `camera.light_roi` nếu có (cắt vùng cố định trên frame gốc, `sentinel/lights.py` = OpenCV HSV, không cần detector); nếu không thì dùng detector COCO như cũ. Nguồn đang dùng được ghi ra console khi khởi tạo (`[lights] traffic-light source: roi|coco`) và là `Pipeline.light_source`. Logic luật `red_light` không đổi (test). Khi dùng ROI, các box `traffic light` của COCO không còn quyết định trạng thái.

## `configs/cameras/test_video.yaml` chưa có `light_roi` → phần 4 dừng ở đây
Hiện tại `test_video.mp4` đọc đèn bằng COCO: detector chỉ thấy đèn ở **4 trên 787 frame** (một đèn cột ở khoảng x 1122–1141, y 149–190, đều "GREEN"), nên `p.light` gần như luôn `UNKNOWN` (783/787 frame) và luật `red_light` gần như không bao giờ kích hoạt trên clip này.

**Đề xuất:** `light_roi: [590, 0, 632, 22]` (toạ độ trong khung 1280×720): bộ đếm lùi của cột đèn phía trên giữa khung hình, nằm sát mép trên của ảnh. Kiểm bằng `detect_traffic_light_color` trên 6 frame (0, 150, 300, 450, 600, 780 ≈ 0, 5, 10, 15, 20, 26 s): `RED, RED, GREEN, GREEN, GREEN, GREEN`; chữ số đỏ ở 0–5 s (đang đếm 9, 4), xanh từ 10 s (29, 24, 19, 13): khớp với sự đổi pha đèn nhìn bằng mắt. Ảnh minh họa (vùng ROI trên frame, crop ROI theo thời gian kèm trạng thái): `outputs/light_roi_proposal.jpg` (gitignored: có người).

**Cần bạn xác nhận:**
1. Bộ đếm này có phải đèn chi phối hướng đi của vùng `violation_zone` hiện tại không? Trên ảnh còn có đèn mũi tên rẽ trái ở khoảng x 460–520 và x 700–735 và đèn mũi tên xanh rẽ phải ở khoảng x 840–890, tất cả sát mép trên (đầu đèn bị cắt một phần).
2. ROI nhỏ (42×22 px) và sát mép trên: một chút dịch camera sẽ làm lệch; chấp nhận hay cần ROI rộng hơn.
3. Có muốn tôi ghi `light_roi` vào `test_video.yaml`, rồi xuất timeline trạng thái đèn theo hai cách (ROI và COCO) ra CSV cùng ảnh dải (crop đèn mỗi 0.5 s, trạng thái dự đoán bên dưới) để bạn xác nhận màu thật.
