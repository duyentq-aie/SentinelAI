# EXP-017 – kiểm kê video tự quay (`data/raw/field/`), 2026-10-19

## Summary (English)
- Purpose: inventory of the self-recorded videos in data/raw/field/ and a check of the clip against the helmet image set (dHash).
- Result: the folder holds one video, a byte-identical copy of test_video.mp4 (1280x720, 30 FPS, 787 frames, 26.2 s), so it is not new field data; its dHash distance to the nearest helmet image is min 12, median 19, max 21 (27 frames), while helmet images of other videos in the same set are at median 8.
- Conclusion: the clip is a different camera and scene (oblique surveillance view of an intersection), no frame is duplicated, so it is out-of-domain for the helmet data; red_light is geometrically feasible, plate OCR is not (plates too small at 720p).
- Limits: the assessment rests on 3 frames viewed by eye and one 26 s clip already used in earlier experiments; dHash only detects structural duplicates; no labels, no pipeline run in this step.

**Kết quả chính: thư mục chỉ có 1 video, và nó là file `test_video.mp4` đã dùng từ đầu dự án** (cùng sha256 `51fdf565…a16a1d` với `test_video.mp4` ở gốc repo; **bản sao trùng byte**, là một file riêng chứ không phải hard link – kiểm bằng `fsutil hardlink list`: file gốc ở repo là hard link với `outputs/bench_work/v0/test_video.mp4`, còn file trong `field/` đứng riêng). **Đây không phải field data.**. Chưa có video mới nào. Nếu bạn định thêm video quay thêm, chúng chưa nằm trong thư mục này.

Công cụ: **ffprobe không được cài** (không cài thêm); thông số lấy bằng OpenCV (độ phân giải, fps, số frame, FOURCC) và Python (dung lượng, sha256). Bảng đầy đủ: `data/field_manifest.csv` (nằm trong `data/` nên bị `.gitignore` chặn). Contact sheet (1 frame / 10 s: t = 0, 10, 20 s): `contact_test_video.jpg`, **không commit** (có biển số và mặt người).

| tên | độ phân giải | fps | số frame | thời lượng | codec | dung lượng |
|---|---|---|---|---|---|---|
| test_video.mp4 | 1280×720 | 30.0 | 787 | 26.2 s | h264 | 8 860 419 B (8.45 MiB) |

## Đánh giá từ ảnh mẫu (chỉ 3 frame, nhìn bằng mắt; chưa chạy pipeline, chưa gán nhãn)
| câu hỏi | quan sát |
|---|---|
| đèn giao thông | có cột đèn ở nửa trên khung hình, rất nhỏ ở 720p; trạng thái đèn không đọc chắc từ 3 frame |
| vạch dừng | có vạch kẻ đường dành cho người đi bộ và vạch ngang ở nút giao; mũi tên chỉ làn; không chắc đâu là vạch dừng chính thức |
| ngày / đêm | ban ngày, nắng |
| góc quay | trên cao, nhìn chéo xuống nút giao (kiểu camera giám sát), có dòng thời gian/tên địa điểm chồng trên ảnh |
| cố định / cầm tay | cố định (nền không dịch chuyển giữa t = 0, 10, 20 s) |
| biển số đọc được bằng mắt | hầu như không: xe ở xa, biển chỉ vài chục pixel; một ô tô trắng ở gần (t = 20 s) có biển nhưng vẫn rất nhỏ |
| mật độ xe máy | rất cao: nhiều chục xe máy cùng lúc, chen sát nhau ở phía trước (t = 0, 10 s) |
| xe 3 người / người không mũ | thấy nhiều xe chở 2 người; xe 3 người và người không mũ **không xác nhận được** từ 3 frame ở độ phân giải này (cần xem video đầy đủ hoặc gán nhãn) |

## Dùng được cho luật nào (từ ảnh mẫu, chưa kiểm bằng dữ liệu)
| luật | đánh giá |
|---|---|
| helmet / no_helmet | có thể: đông xe máy, nhưng đầu nhỏ và chen nhau; chất lượng phát hiện đầu chưa biết; không có nhãn |
| overload (> 2 người) | có thể, cùng hạn chế (đếm đầu khi xe chồng nhau); chưa thấy ca rõ ràng trong 3 frame |
| red_light | khả thi về hình học (có đèn, có vạch, camera cố định) nhưng cần: trạng thái đèn nhìn rõ trong frame và vạch dừng khai báo bằng `tools/camera_config.py`; chưa xác nhận có xe vượt đèn đỏ; clip chỉ 26 s |
| OCR biển số | **không**: biển nhỏ hơn nhiều so với yêu cầu (≥ ~40–50 px cao, xem PLAN, EXP-008) ở 720p |

Giới hạn: kết luận dựa trên 3 frame của một clip 26 s đã dùng cho mọi thí nghiệm trước (EXP-001…010), nên không thêm bằng chứng độc lập nào.

## So với bộ ảnh helmet (dHash, 2026-10-19) – `compare_with_helmet.py`, `compare_with_helmet.json`
27 frame (1 frame/giây) của `test_video.mp4` so với 1460 ảnh helmet, cùng dHash 64 bit như EXP-013 (Hamming nhỏ nhất tới ảnh helmet gần nhất):
- **Khoảng cách: min 12, trung vị 19, max 21** (q25 17.5, q75 20). Không frame nào ≤ 8; chỉ 1/27 frame ≤ 12.
- Đối chiếu: ảnh helmet với ảnh gần nhất **của video khác trong cùng bộ**: trung vị 8 (q75 10, max 17); với ảnh gần nhất bất kỳ (kể cả cùng video): trung vị 1. Bản trùng/gần trùng thường ≤ 4 (EXP-011/014).
- 5 cặp gần nhất (`closest_5_pairs.jpg`, không commit vì có người): d = 12, 16, 16, 16, 17. Xem bằng mắt: ảnh helmet là góc nhìn **thẳng từ trên cao xuống một đoạn đường thẳng có dải phân cách và cây** (frame_12, frame_5), còn test_video là **camera giám sát nhìn chéo ở một nút giao có đèn tín hiệu, vạch kẻ và biển hiệu cửa hàng**. Chỉ giống ở mức "mặt đường xám + xe máy + ô tô trắng".
- **Kết luận:** *không phải cùng camera / cùng cảnh* và *không có frame trùng*. Dữ liệu huấn luyện helmet và clip test không rò rỉ cho nhau theo phép đo này (dHash chỉ bắt được trùng cấu trúc ảnh; không loại trừ mọi khả năng). Hệ quả: clip test là dữ liệu **ngoài miền** so với ảnh helmet (khác góc, khác nút giao, mật độ xe cao hơn nhiều).
