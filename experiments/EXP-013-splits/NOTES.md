# EXP-013 – đề xuất chia lại helmet theo video (2026-10-18)

**Chưa chia lại gì**: `tools/make_splits.py` chỉ đọc dataset và xuất manifest (không di chuyển/copy/đổi tên ảnh). Chọn phương án rồi mới tạo thư mục split.
Lệnh: `python tools/make_splits.py --data data/raw/helmet --out experiments/EXP-013-splits --classes helmet motorcycle no-helmet plate --balance no-helmet --seed 0`

## Nhóm
17 nhóm: 16 video (`frame_1` … `frame_16`, 21–343 ảnh) + 1 nhóm `other:testing_images` (6 ảnh 1920×1080 tên `testing_images_<n>`, nguồn không rõ; hiện cả 6 nằm ở train). Mỗi nhóm đi nguyên vào một split. Khác 16 nhóm bạn nói: 6 ảnh 1080p không thuộc video nào đánh số.
Tìm kiếm có seed (tối thiểu hóa độ lệch so với tỉ lệ mục tiêu của số ảnh và của instance `no-helmet`; phạt nếu split nào thiếu một class); cùng seed cho cùng kết quả (test).

## Ba phương án (seed 0; số instance gồm cả `plate` vì dataset có 4 lớp, xem DATA_CARD)
| | train | valid | test | no-helmet (train/val/test) | ghi chú |
|---|---|---|---|---|---|
| **A** 70/15/15, ảnh và no-helmet cân bằng như nhau | 1026 ảnh, 7 nhóm | 214, 4 nhóm | 220, 6 nhóm (có 6 ảnh 1080p) | 662 / 140 / 143 | mặc định tôi đề xuất |
| **B** 70/15/15, ưu tiên no-helmet | 1014, 8 nhóm | 226, 3 nhóm | 220, 6 nhóm | 660 / 142 / 143 | gần như A (khác 2 video); valid chỉ 3 video → phương sai cao |
| **C** 60/20/20, val/test lớn hơn | 875, 7 nhóm | 294, 6 nhóm | 291, 4 nhóm | 565 / 187 / 193 | ít dữ liệu train hơn, số đo ổn định hơn; test có `frame_2`, `frame_3`, `frame_7` + 1080p |
Chi tiết từng class/video: `A|B|C/summary.md`, `summary.json`, `manifest.csv` (sha256 từng ảnh + sha256 của manifest). So với split cũ (no-helmet 664/170/111) cả ba cân bằng hơn.

## Lưu ý
- Mức tương đồng còn lại giữa video khác nhau (dHash tới ảnh train gần nhất, `cross_video_dhash.json`): test A 26.8% ≤ 4, B 24.5%, **C 9.6%**; split ngẫu nhiên cũ là 86%. Camera tĩnh cùng chỗ nên vẫn còn giống nhau giữa các video; C ít nhất, nhưng cũng là phương án có ít train nhất.
- Chỉ 16 video: kết quả test phụ thuộc mạnh vào video nào rơi vào test; nên báo CI bootstrap theo video, và video tự quay vẫn là test độc lập đáng tin hơn.
- 6 ảnh 1080p (nguồn khác) rơi vào test ở cả ba phương án vì được coi là một nhóm: ảnh hưởng nhỏ (≈ 2–3% số ảnh test) nhưng nên biết.
- Phương án nào cũng chỉ giải quyết rò rỉ kề-frame; không đảm bảo các video không cùng một cảnh.
