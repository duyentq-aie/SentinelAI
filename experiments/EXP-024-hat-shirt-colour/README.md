# EXP-024 – "mũ cùng màu áo": đo, không sửa

Giả thuyết (từ EXP-020): người đội mũ bị báo no-helmet vì mũ ít tương phản với áo/thân. Cách đo (`measure.py`, kết quả `results.json`, `heads.csv`; ảnh ghép `group_A.jpg`, `group_B.jpg` xếp theo ΔE tăng dần, không commit vì có người):
- **Nhóm A (báo oan):** đầu của các sự kiện no_helmet báo oan ở EXP-020. Với mỗi hàng no_helmet lấy một phiên bản có báo (thứ tự: hiện tại + model cũ, v0, EXP-015, EXP-016, @10 FPS), chạy model helmet của phiên bản đó trên frame của sự kiện (conf 0.55 như pipeline, cho phép lệch ±2 frame) và lấy detection `no-helmet` nằm trong hộp xe (conf cao nhất). **21 đầu** (2 hàng overload không có đầu no-helmet nên không vào nhóm A; hàng nào không tìm lại được đầu thì ghi trong `results.json`: 0 hàng).
- **Nhóm B (đối chứng):** 60 đầu ngẫu nhiên (seed 0, tối đa 2 mỗi frame, mỗi 5 frame, không trùng nhóm A) mà model cũ `helmet_best.pt` nhận đúng là `helmet` với conf >= 0.6 (240 ứng viên).
- **Mỗi đầu:** màu trung bình trong không gian Lab của nửa trên hộp đầu ("vùng mũ") và của vùng ngay dưới đầu ("vùng thân": rộng bằng đầu +20% mỗi bên, sâu bằng một chiều cao đầu); ΔE76 = khoảng cách Euclid giữa hai màu trung bình.

## Kết quả
| | nhóm A (báo oan) | nhóm B (đối chứng) |
|---|---|---|
| n | 21 | 60 |
| ΔE trung vị | **15.99** | **16.34** |
| ΔE trung bình | 17.63 | 16.55 |
| tứ phân vị (25% / 75%) | 8.93 / 19.48 | 7.84 / 20.97 |

Mann–Whitney: kiểm định một phía (A thấp hơn B) U = 655, **p = 0.6083**; hai phía p = 0.7918; xác suất một đầu nhóm A có ΔE thấp hơn một đầu nhóm B = 0.48.

**Kết luận: giả thuyết KHÔNG được ủng hộ bởi phép đo thô này.** ΔE của nhóm A không thấp hơn nhóm B (trung vị 15.99 so với 16.34; p một phía = 0.6083, gần như ngẫu nhiên). Đây KHÔNG chứng minh màu mũ/áo vô can: n nhóm A nhỏ (21), các mẫu không độc lập (ba đầu cuối ở ảnh ghép nhóm A, ΔE 37–43, trông như cùng một người áo đỏ mũ đen ở ba thời điểm khác nhau), và vùng mũ/vùng áo chỉ ước lượng từ hộp, chưa phân đoạn (tóc, mặt, kính mũ, tay, hành khách, túi sau đầu đều lọt vào vùng). Kiểm tra bằng mắt ảnh ghép: nhóm A có cả mũ tối trên áo tối và mũ sáng trên áo sáng, nhưng cũng có nhiều ca tương phản cao (mũ đen, áo đỏ), nên màu áo không giải thích được phần lớn báo oan trên clip này.

Đo thô, chỉ để định hướng: không sửa gì, không đổi ngưỡng nào.
