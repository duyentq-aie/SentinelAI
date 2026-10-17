# Hướng dẫn quay video đánh giá

Mục tiêu: có video **chưa từng dùng để train hay chỉnh luật** (test) và video để chỉnh luật (dev), đủ đa dạng để số liệu đánh giá có ý nghĩa.
Mọi con số báo cáo cuối cùng phải đo trên **video test**; video dev chỉ để tinh chỉnh tham số.

## 1. Chọn vị trí
- **Ít nhất 2 vị trí khác nhau**; tách theo **vị trí/buổi quay**, không cắt một video dài thành dev và test (frame liền kề sẽ rò rỉ giữa hai tập).
- Ít nhất **1 vị trí là ngã tư có đèn tín hiệu** nhìn thấy rõ **cả đèn lẫn vạch dừng** trong cùng khung hình (cần cho luật vượt đèn đỏ).
- Cảnh có nhiều xe máy, có người không đội mũ, có xe chở 3 người nếu có thể (vi phạm hiếm: quay giờ cao điểm, 17:00–19:00 hoặc 6:30–8:30).
- Tránh nắng gắt rọi thẳng vào ống kính; tránh bị cây/cột che đèn và vạch dừng.

## 2. Góc và thiết bị
- Camera **cố định trên chân máy hoặc điểm tựa vững**, không cầm tay, không zoom/xoay trong lúc quay.
- Độ cao 3–6 m (cầu vượt, vỉa hè cao, tầng 2), nhìn **chếch xuống khoảng 20–40°**, thấy mặt trước/sau xe để đọc được biển số.
- **1080p, 30 fps** (không dưới 720p). Biển số chỉ ~50 px ở 720p (đo được trong EXP-003) nên 1080p là cần thiết cho OCR.
- Tắt chống rung điện tử và chế độ làm đẹp; khóa phơi sáng/lấy nét nếu máy cho phép.
- Giữ nguyên tên file gốc hoặc đặt theo mẫu `<vitri>_<ngay|dem>_<stt>.mp4`, ví dụ `ngatu_A_dem_01.mp4`.

## 3. Thời lượng và điều kiện
| nhóm | tối thiểu | ghi chú |
|---|---|---|
| dev | ~30 phút tổng, nhiều clip 5–10 phút | để chỉnh W, K, r, vạch dừng, ROI đèn |
| test | ~30 phút tổng, **khác vị trí hoặc khác buổi so với dev** | **khóa**, không xem kỹ để chỉnh luật |
| ngày / đêm | **≥ 25% tổng thời lượng là ban đêm** | đèn đường, đèn xe: cần cho đánh giá theo điều kiện |
| mưa | nếu có điều kiện, 5–10 phút | ghi `weather: rain` trong nhãn |

Mục tiêu số sự kiện (để số liệu có ý nghĩa): **≥ 20 sự kiện mỗi loại** (no_helmet, overload, red_light) ở tập test, ít nhất 30 phút video tổng để đếm báo nhầm. Vượt đèn đỏ hiếm: nếu quay xong vẫn dưới 20, vẫn báo đúng số lượng với khoảng tin cậy rộng, không suy rộng.

## 4. Quy trình
1. Quay clip → chép vào thư mục ngoài git (ví dụ `data/videos/`; thư mục `data/` đã nằm trong `.gitignore`).
2. Chạy `python tools/camera_config.py --video <clip> --camera-id <vitri>` để vẽ **vạch dừng, hướng đi, ROI đèn, vùng** → `configs/cameras/<vitri>.yaml` (một file cho mỗi vị trí; camera không được xê dịch giữa các clip).
3. Gán nhãn bằng `python tools/label_events.py --video <clip> --split dev|test` (xem `docs/LABELING_GUIDE.md`).
4. Ghi `split` đúng ngay từ đầu. **Không đổi một clip từ test sang dev** sau khi đã nhìn kết quả.

## 5. Quyền riêng tư và pháp lý
- Kiểm tra quy định quay nơi công cộng ở địa phương; không quay vào nhà dân, không quay cận mặt người.
- Video gốc (có biển số và mặt người) **giữ nội bộ**, không đưa lên GitHub. `*.mp4` và `data/` đã được `.gitignore`.
- Media công bố (demo, ảnh trong báo cáo) phải **làm mờ biển số và mặt** người không liên quan.
- Cảnh dàn dựng với người quen (xe và người của bạn, đã đồng ý) là nguồn biển số có thể công bố.
- Biển số trong nhãn (`plate_text`) là dữ liệu cá nhân: giữ trong repo riêng tư, không công bố bộ nhãn kèm video.
