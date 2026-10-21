# EXP-014 – nhóm ảnh gần trùng của dataset biển số bằng perceptual hash (2026-10-18)

## Summary (English)
- Purpose: find near-duplicate image groups in the plate dataset with a 64-bit dHash (Hamming distance <= T), since file names carry no frame numbers.
- Result: at T = 4 there are 6131 groups (largest 240), 462 span several splits, and 31.4% / 29.0% of valid / test images have a train image in the same group (13.8% / 13.3% at T = 2); at T >= 6 components blow up (T = 8: one group of 3410).
- Conclusion: real leakage exists (at least about 13% of valid/test images have a near-identical train image; the figure is a lower bound by hash), so the plate_best.pt test result (mAP50 0.995, EXP-000) is not evidence of generalisation.
- Limits: the whole set is one fixed gate camera, so even a grouped split would only measure 'same camera, other vehicle'; the fix (regrouping at T = 4 or a self-recorded 1080p test) was not done.

Lệnh: `python tools/near_dup_groups.py --data data/raw/plate --out experiments/EXP-014-plate-groups --thresholds 0 2 4 6 8 --csv-for 4`
(dHash 64 bit; nối hai ảnh khi khoảng cách Hamming ≤ T; nhóm = thành phần liên thông). `groups_T4.csv` = ảnh → nhóm (T = 4). Tên file không có số frame nên đây là cách duy nhất để tạo nhóm.

| T | số nhóm (đơn lẻ) | nhóm lớn nhất | nhóm nằm ở nhiều split | % ảnh valid / test có ảnh train cùng nhóm |
|---|---|---|---|---|
| 0 | 8031 (7711) | 3 | 106 | 4.0 / 3.4 |
| 2 | 7416 (6648) | 16 | 316 | 13.8 / 13.3 |
| **4** | 6131 (5143) | 240 | 462 | **31.4 / 29.0** |
| 6 | 4398 (3553) | 964 | 388 | 51.4 / 49.0 |
| 8 | 2964 (2363) | 3410 | 274 | 66.6 / 66.9 |

## Kiểm bằng mắt (ghép ảnh test ↔ ảnh train gần nhất, 4 cặp mỗi hàng; ảnh nằm ngoài git vì có biển số thật)
- **d ≤ 2:** cùng xe, cùng khung hình (cùng biển số đọc được) nằm ở test **và** train → rò rỉ thật.
- **d 3–4:** lẫn: nhiều cặp là **cùng một xe với độ sáng khác** (bản tăng cường độ sáng/tối của Roboflow, một ảnh gốc xuất hiện ở nhiều split), một số cặp là hai xe máy khác nhau.
- **d 5–6:** phần lớn là hai xe khác nhau trong cùng một cảnh; vẫn có cặp cùng xe. Từ T = 6 trở đi thành phần liên thông phình ra (T = 8: một nhóm 3410 ảnh) do nối chuỗi và do **cảnh cố định giống nhau** → vô nghĩa làm nhóm.
- Hiểu đúng: ảnh đến từ một camera cổng gửi xe cố định (640×640, cùng nền); hash gần nhau vì cùng cảnh, không chỉ vì cùng ảnh.

## Kết luận (có giới hạn)
- Có rò rỉ thật: **ít nhất ≈ 13%** ảnh valid/test có ảnh gần như y hệt ở train (T ≤ 2), tới ≈ 30% ở T = 4 (trộn bản tăng cường độ sáng của cùng xe). Số này là cận dưới theo hash, không phải đếm chính xác.
- 102 file trùng byte trong cùng split (EXP-011) và các bản tăng cường độ sáng của cùng ảnh ở nhiều split ⇒ test của plate_best.pt (mAP50 0.995, EXP-000) không dùng làm bằng chứng khái quát hóa.
- Hướng sửa (chưa làm): chia lại theo nhóm T = 4 (nhóm lớn nhất 240 ảnh nên chia được), hoặc dùng video tự quay ở 1080p làm test. Vì cả bộ là một camera cố định, dù chia theo nhóm, test vẫn chỉ đo "cùng camera, xe khác".
