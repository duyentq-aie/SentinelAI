# DATA_CARD – tập dữ liệu detector (P1, 2026-10-18)

Đọc-only: `tools/leakage_check.py` không chia lại, không đổi tên, không xóa. Số liệu thô: `experiments/EXP-011-data-leakage/{helmet,plate}_leakage.json`. Dữ liệu nằm ở `data/raw/` (gitignored). **Chưa chia lại split, chưa train.**

## Helmet (`data/raw/helmet`, Roboflow export 2026-02-24)
| | |
|---|---|
| nguồn | Roboflow `trans-workspace / helmet-plate-detection-4-r3h7b`, version 1 |
| license | **khai báo trong `data.yaml`: CC BY 4.0** (do người đăng khai, *chưa xác minh độc lập*; cần ghi công tác giả khi phân phối lại) |
| ảnh | train 1014 / valid 278 / test 168 (tổng 1460); 1454 ảnh 3840×2160, 6 ảnh 1920×1080 |
| lớp trong dataset | **`nc: 4`** = helmet, motorcycle, no-helmet, **plate** |
| số nhãn (train / valid / test) | helmet 1870 / 606 / 359; motorcycle 1935 / 608 / 348; no-helmet 664 / 170 / 111; plate 2090 / 631 / 383 |
| lệch với checkpoint | `helmet_best.pt` có **3 lớp** (helmet, motorcycle, no-helmet); log huấn luyện EXP-000 đếm 1384 instance ở val = 606+608+170, tức lớp `plate` bị bỏ khi huấn luyện, nhưng `data.yaml` đang có vẫn là 4 lớp. Muốn huấn luyện/đánh giá lại phải chốt một ánh xạ lớp rõ ràng. |

### Rò rỉ train/val/test – **có**
Tên file dạng `frame_<video>_<n>`: ảnh là các frame trích từ **16 video**, chia **ngẫu nhiên theo frame**.
- Trùng byte: 0; cùng tên nguồn ở nhiều split: 0 (nên không có nhân đôi nguyên văn).
- Gần trùng (dHash, khoảng cách Hamming tới ảnh train gần nhất): valid: 84.5% ≤ 4, 97.8% ≤ 8; test: 86.3% ≤ 4, 99.4% ≤ 8 (trung vị 2/64).
- Kề frame: 75.6% ảnh test có frame train của **cùng video cách ≤ 1 frame**, 99.4% trong ≤ 5 frame (valid: 75.5% / 98.6%).
- Nhóm frame liên tiếp (khe ≤ 15 frame): 33 nhóm / 16 video, **29 nhóm nằm ở nhiều split**; 100% ảnh valid và test (278 + 168) thuộc nhóm bị rò.
- Hệ quả: các chỉ số của EXP-000 (mAP50 ~0.99, bão hòa) **không đo khả năng khái quát hóa**; không dùng làm tuyên bố độ chính xác. Cách sửa (chưa làm): chia theo nhóm/video (cả 16 video, giữ nguyên nhóm trong cùng một split), hoặc quay thêm video mới làm test độc lập.

## Plate (`data/raw/plate`, Roboflow)
| | |
|---|---|
| nguồn | Roboflow `school-fuhih / vietnamese-license-plate-tptd0`, version 1 (2023-07-11, "Provided by a Roboflow user") |
| license | **khai báo CC BY 4.0** (chưa xác minh độc lập) |
| ảnh | train 5845 / valid 1680 / test 832; tất cả 640×640 |
| lớp | `nc: 1`, tên `'0'`; nhãn 5989 / 1710 / 849 |
| ghi chú | notebook huấn luyện `plate_best.pt` không khớp dataset này (xem EXP-000) – **chưa chứng minh** `plate_best.pt` được huấn luyện trên chính tập này |

Rò rỉ: tên file không có số frame nên không kiểm tra kề-frame được. Không trùng byte giữa split (102 file có bản sao byte-identical **trong cùng split**). dHash: valid/test có 28–30% ảnh với ảnh train gần nhất ≤ 4 và 64% ≤ 8 (trung vị 7) – mức gần trùng thấp hơn helmet nhưng đáng kể; không kết luận được nguồn gốc (nhiều ảnh cùng cảnh/xe?).

## Việc kế tiếp (cần quyết định của chủ dự án)
1. Chốt ánh xạ lớp helmet (3 hay 4 lớp) và có train lại hay không.
2. Chia lại theo video/nhóm cho helmet **hoặc** dùng video tự quay làm test (đề xuất: video tự quay, vì chỉ có 16 video nguồn).
3. Với plate: kiểm tra nhóm cảnh (cần metadata hoặc phân cụm) trước khi tin vào số liệu test.
4. Xác minh license thật trên Roboflow trước khi phân phối dữ liệu hay weights.
