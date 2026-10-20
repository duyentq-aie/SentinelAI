# DATA_CARD – tập dữ liệu detector (P1, 2026-10-18)

Đọc-only: `tools/leakage_check.py` không chia lại, không đổi tên, không xóa. Số liệu thô: `experiments/EXP-011-data-leakage/{helmet,plate}_leakage.json`. Dữ liệu nằm ở `data/raw/` (gitignored). **Chưa chia lại split, chưa train.**

## Nguồn, giấy phép và ghi công (CC BY 4.0) – đã kiểm trên trang Roboflow Universe, 2026-10-21
| | helmet | plate |
|---|---|---|
| tên bộ dữ liệu | **helmet plate detection 4** | **vietnamese license plate** |
| tác giả | Rajab (Roboflow Universe, 2024) | school (workspace `school-fuhih`, Roboflow Universe, 2023) |
| nguồn | https://universe.roboflow.com/rajab/helmet-plate-detection-4 | https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0 |
| giấy phép | **CC BY 4.0** – https://creativecommons.org/licenses/by/4.0/ | **CC BY 4.0** – https://creativecommons.org/licenses/by/4.0/ |
| quy mô trên trang | 1,460 ảnh, 4 lớp (helmet, motorcycle, plate, no-helmet) | khoảng 8.4k ảnh, 1 lớp |
| quan hệ với dự án | **nguồn gốc bộ helmet ĐÃ XÁC MINH**: số ảnh (1014 + 278 + 168 = 1460) và 4 lớp khớp dữ liệu của dự án; `data.yaml` của bản export ghi project `trans-workspace/helmet-plate-detection-4-r3h7b` (bản sao trong workspace khác), trang gốc là của Rajab | **CHƯA xác minh** `plate_best.pt` được train từ bộ này (notebook huấn luyện không khớp weights, EXP-000), nên `plate_best.pt` **không được công bố** |

**Thay đổi đã làm so với dữ liệu gốc (yêu cầu của CC BY 4.0):**
- Chia lại train/val/test **theo video** (split A của EXP-013; bản gốc chia ngẫu nhiên theo frame và rò rỉ, xem mục "Rò rỉ" ở dưới).
- Model 3 lớp (EXP-015) **bỏ lớp `plate`** (chỉ giữ helmet, motorcycle, no-helmet); model 4 lớp (EXP-016) giữ cả 4 lớp.
- Ảnh gốc **không được phân phối lại** trong repo này (`data/` không được theo dõi bởi git). Một bản sao riêng tư đã được tải lên Kaggle chỉ để huấn luyện.
- Hộp nhãn **không** được gán lại (mAP50-95 bị giới hạn vì hộp lỏng, xem EXP-015).
- Weights EXP-015 và EXP-016 là **sản phẩm phái sinh** từ bộ helmet ở trên (ghi nguồn trong mô tả release `models-v1`).

**BibTeX** (BibTeX in the format of the Roboflow Universe page; the page's month and access-date fields were not recorded here and are left out.):
```bibtex
@misc{ helmet-plate-detection-4_dataset,
    title = { helmet plate detection 4 Dataset },
    type = { Open Source Dataset },
    author = { Rajab },
    howpublished = { \url{ https://universe.roboflow.com/rajab/helmet-plate-detection-4 } },
    url = { https://universe.roboflow.com/rajab/helmet-plate-detection-4 },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2024 },
}

@misc{ vietnamese-license-plate-tptd0_dataset,
    title = { vietnamese license plate Dataset },
    type = { Open Source Dataset },
    author = { school },
    howpublished = { \url{ https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0 } },
    url = { https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0 },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2023 },
}
```

## Helmet (`data/raw/helmet`, Roboflow export 2026-02-24)
| | |
|---|---|
| nguồn | Roboflow Universe `rajab / helmet-plate-detection-4` (Rajab, 2024); export ghi project `trans-workspace / helmet-plate-detection-4-r3h7b`, version 1 |
| license | **CC BY 4.0, đã xác minh trên trang Roboflow Universe** (ghi công tác giả, xem mục "Nguồn, giấy phép và ghi công") |
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
| nguồn | Roboflow Universe `school-fuhih / vietnamese-license-plate-tptd0`, version 1 (school, 2023-07-11, "Provided by a Roboflow user") |
| license | **CC BY 4.0, đã kiểm trên trang Roboflow Universe**; nhưng **chưa xác minh** `plate_best.pt` được train từ bộ này (xem dòng "ghi chú") |
| ảnh | train 5845 / valid 1680 / test 832; tất cả 640×640 |
| lớp | `nc: 1`, tên `'0'`; nhãn 5989 / 1710 / 849 |
| ghi chú | notebook huấn luyện `plate_best.pt` không khớp dataset này (xem EXP-000) – **chưa chứng minh** `plate_best.pt` được huấn luyện trên chính tập này |

Rò rỉ: tên file không có số frame nên không kiểm tra kề-frame được. Không trùng byte giữa split (102 file có bản sao byte-identical **trong cùng split**). dHash: valid/test có 28–30% ảnh với ảnh train gần nhất ≤ 4 và 64% ≤ 8 (trung vị 7) – mức gần trùng thấp hơn helmet nhưng đáng kể; không kết luận được nguồn gốc (nhiều ảnh cùng cảnh/xe?).

## Việc kế tiếp (cần quyết định của chủ dự án)
1. Chốt ánh xạ lớp helmet (3 hay 4 lớp) và có train lại hay không.
2. Chia lại theo video/nhóm cho helmet **hoặc** dùng video tự quay làm test (đề xuất: video tự quay, vì chỉ có 16 video nguồn).
3. Với plate: kiểm tra nhóm cảnh (cần metadata hoặc phân cụm) trước khi tin vào số liệu test.
4. ~~Xác minh license thật trên Roboflow~~ Đã làm 2026-10-21 (CC BY 4.0 cho cả hai bộ); còn lại: chưa chứng minh `plate_best.pt` train từ bộ plate, nên không công bố weights đó.

## Phụ lục (2026-10-18): dataset này có phải bản đã huấn luyện `helmet_best.pt`?
| | `data/raw/helmet/data.yaml` (Roboflow export 2026-02-24) | checkpoint `helmet_best.pt` (`model.names`, `train_args`) |
|---|---|---|
| tên lớp | 0 helmet, 1 motorcycle, 2 no-helmet, **3 plate** (nc 4) | 0 helmet, 1 motorcycle, 2 no-helmet (nc 3) |
| instance val (log huấn luyện, EXP-000) | helmet 606, motorcycle 608, no-helmet 170 (=1384) | khớp đúng: 606 / 608 / 170 |
| instance test (log) | helmet 359, motorcycle 348, no-helmet 111 (=818) | khớp đúng: 359 / 348 / 111 |
| ảnh nền (không nhãn) train / val / test | 0 / 0 / 0 (có `plate`) → **2 / 1 / 2 ảnh chỉ có nhãn plate** | log: 2 / 1 / 2 ảnh nền |
| ảnh train / val / test | 1014 / 278 / 168 | log: 1014 / 278 / 168 |
| `train_args.data` | | `.../helmet plate detection 4.v1i.yolov8/data.yaml` (cùng tên dự án Roboflow) |

**Kết luận (suy luận từ số liệu, chưa kiểm trực tiếp):** cùng ảnh và cùng nhãn cho 3 lớp đầu, nhưng bản dùng để huấn luyện **không có lớp `plate`** (các ảnh chỉ có plate trở thành ảnh nền: 2/1/2 khớp đúng). Nên *không phải cùng `data.yaml`*: tập Kaggle là bản dẫn xuất 3 lớp (bỏ lớp 3), tập local là bản export 4 lớp. Mọi phép đếm khớp tới từng instance nên khả năng cao cùng nguồn Roboflow `helmet-plate-detection-4`, version 1. Không kiểm chứng được `data.yaml` thật trên Kaggle (không còn trong checkpoint). Để huấn luyện lại 3 lớp khớp checkpoint: bỏ nhãn lớp 3 khi dựng dữ liệu (xem `tools/kaggle_train.py`).
