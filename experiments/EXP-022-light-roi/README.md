# EXP-022 – trạng thái đèn từ ROI trên test_video (PROVISIONAL – chờ xác nhận)

`configs/cameras/test_video.yaml` có `light_roi: [590, 0, 632, 22]` (đánh dấu "provisional – chờ xác nhận"): bộ đếm lùi của cột đèn ở trên giữa khung 1280×720. Đề xuất ban đầu: `PROPOSAL.md`. Mã: `Pipeline` đọc đèn từ ROI khi có (log `[lights] traffic-light source: roi`).

**Chưa công bố kết quả `red_light` nào.** Với `light_roi` này, mọi lần chạy pipeline trên `test_video.mp4` sẽ có trạng thái đèn RED trong 0–9.3 s và luật vùng (`violation_zone`) có thể kích hoạt; các số đó **chưa được đưa vào báo cáo** cho tới khi bạn xác nhận ROI. Các thí nghiệm trước (EXP-015…021, review A/B/C) chạy khi `light_roi` chưa có (đèn từ COCO, gần như luôn UNKNOWN) nên không có sự kiện `red_light`.

## Timeline (`timeline.py` → `light_timeline.csv`, mỗi frame 1 dòng)
- **ROI (OpenCV):** RED ở frame 0–279 (0–9.3 s), GREEN từ frame 280 (9.33 s) đến hết: **một lần đổi trạng thái duy nhất, không nhấp nháy**; đầy đủ 787 frame có kết luận (không UNKNOWN). Đếm điểm ảnh đỏ/xanh theo frame có trong CSV (`roi_red_px`, `roi_green_px`).
- **COCO (detector, bản ghi mặc định):** `UNKNOWN` ở 783/787 frame, `GREEN` ở 4 frame (306–312, một đèn cột khác ở x ≈ 1122–1141): nguồn này gần như không cho thông tin trên clip.
- Ảnh dải (`light_strip_1..7.jpg`, mỗi 0.5 s; hàng trên: crop ROI phóng 6×, trạng thái dự đoán bên dưới; hàng dưới: vùng `violation_zone` cộng 40 px phía trên cùng thời điểm; **không commit** vì có người): đọc bằng mắt, chữ số đỏ đếm lùi 9 → 0 trong 0–9 s, rồi xanh "29" từ 9.5 s; trong 0–9 s hàng xe máy phía trước đứng dồn ở mép dưới vùng, từ 9.5 s đã bắt đầu dịch chuyển (xem `light_strip_3.jpg`). Điều này **ủng hộ** nhưng không chứng minh bộ đếm này chi phối đúng hướng của hàng xe đó.

## Cần bạn xác nhận
1. Bộ đếm này có phải đèn chi phối hướng đi của vùng `violation_zone` không (xem hàng dưới của ảnh dải để biết xe đang dừng hay đi).
2. ROI nhỏ (42×22 px) sát mép trên khung hình: chấp nhận hay muốn ROI rộng hơn.
3. Sau khi xác nhận: bỏ nhãn provisional và cho phép báo kết quả `red_light`.
