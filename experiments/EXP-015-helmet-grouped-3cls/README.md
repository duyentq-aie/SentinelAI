# EXP-015 – huấn luyện lại detector helmet (3 lớp) trên split theo video (split A)

Người chạy: chủ dự án, trên Kaggle (`results_3cls.zip`, giải nén ở đây; `last.pt` bỏ qua theo yêu cầu). Phân tích dưới đây do trợ lý chạy trên máy cục bộ (chỉ suy luận, không huấn luyện, không sửa `data/raw`).

| | |
|---|---|
| mô hình / công thức | YOLO11s từ `yolo11s.pt`, đúng `train_args` của `helmet_best.pt` (150 epoch, patience 25, batch 32, imgsz 640, seed 0, AdamW qua `optimizer: auto`); `helmet_3cls_splitA/args.yaml` |
| dữ liệu | split A (`experiments/EXP-013-splits/A`, manifest sha256 `fe103d4e…2279`): train 1026 / val 214 / test 214 ảnh (+ `extra_1080p` 6 ảnh), 3 lớp (`plate` bị lọc) |
| môi trường | Kaggle Tesla T4, torch 2.11.0+cu128, Python 3.13.15, ultralytics 8.4.26 (số cũ: torch 2.9.0, Python 3.12.12, cùng ultralytics) |
| kết thúc | dừng sớm ở epoch 83 (patience 25), best ở epoch 58; 2936 s |
| `weights/best.pt` | sha256 **`20ce0694eb1d8d81d6f579ede9b4d81c1aa1fbdf1d46e9391a63d04960fd7817`**, 19 159 770 byte. Bị `.gitignore` chặn (`*.pt`, đã kiểm bằng `git check-ignore`). `last.pt` không giải nén. |
| `results_3cls.zip` | sha256 `0efd2ae46460f5e91036645b4930aade987c20c9e9bd8e204734d6d9da735ebf` |
| trạng thái trong app | **không dùng**: `helmet_best.pt` của app không bị đụng; đổi model cho pipeline là bước riêng chờ bạn duyệt. |

## Kết quả theo lớp (`metrics_3cls.json`, conf mặc định của `val`)
Cột "cũ" = `helmet_best.pt`, mAP50-95 trên **test cũ** (EXP-000): helmet 0.791 / motorcycle 0.878 / no-helmet 0.768 (tổng 0.812).

**Test (214 ảnh, 5 video: frame_4, 6, 9, 13, 16)** – P / R / mAP50 / mAP50-95

| lớp | instance | P | R | mAP50 | mAP50-95 | mAP50-95 cũ | chênh |
|---|---|---|---|---|---|---|---|
| helmet | 258 | 0.866 | 0.985 | 0.962 | 0.623 | 0.791 | −0.168 |
| motorcycle | 304 | 0.999 | 0.990 | 0.995 | 0.828 | 0.878 | −0.050 |
| no-helmet | 133 | 0.833 | 0.744 | 0.844 | 0.453 | 0.768 | −0.315 |
| **tất cả** | 695 | 0.900 | 0.906 | 0.934 | **0.635** | 0.812 | −0.177 |

**Val (214 ảnh, 4 video: frame_10, 11, 14, 15)** – cũ (val cũ, 278 ảnh): helmet 0.793 / motorcycle 0.882 / no-helmet 0.751 / tổng 0.809

| lớp | instance | P | R | mAP50 | mAP50-95 | mAP50-95 cũ | chênh |
|---|---|---|---|---|---|---|---|
| helmet | 468 | 0.772 | 0.954 | 0.950 | 0.579 | 0.793 | −0.214 |
| motorcycle | 471 | 0.991 | 0.981 | 0.995 | 0.745 | 0.882 | −0.137 |
| no-helmet | 140 | 0.913 | 0.771 | 0.913 | 0.529 | 0.751 | −0.222 |
| **tất cả** | 1079 | 0.892 | 0.902 | 0.952 | 0.618 | 0.809 | −0.191 |

**extra_1080p (6 ảnh 1920×1080, nguồn không rõ) – n = 6, KHÔNG kết luận** (helmet có 4 instance, no-helmet 10)

| lớp | instance | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| helmet | 4 | 0.358 | 0.250 | 0.292 | 0.064 |
| motorcycle | 14 | 0.993 | 1.000 | 0.995 | 0.483 |
| no-helmet | 10 | 0.985 | 0.600 | 0.819 | 0.319 |
| tất cả | 28 | 0.778 | 0.617 | 0.702 | 0.289 |

### Đọc con số đúng cách
- **Số cũ không so được trực tiếp với số mới.** Số cũ đo trên split *ngẫu nhiên theo frame có rò rỉ* (EXP-011: ~86% ảnh test có ảnh train cách ≤ 4 bit dHash), trên *một tập ảnh test khác* (168 ảnh, frame lẫn với train). Số mới đo trên video mà mô hình chưa thấy frame nào, trên *tập ảnh khác*, với *môi trường huấn luyện khác* (torch 2.11 vs 2.9, Python 3.13 vs 3.12). Vì vậy mức giảm (−0.18 tổng, −0.32 no-helmet trên test) **không phải hoàn toàn do rò rỉ**: một phần là do tập test mới khó/khác (xem dưới: video test chỉ có tối đa 1 no-helmet mỗi xe, người đội khăn trùm đầu…), một phần do phương sai khi chỉ có 4–5 video, và không tách riêng được bằng thí nghiệm này (cần huấn luyện lại 3 lớp trên split ngẫu nhiên cũ trong cùng môi trường, hoặc chạy `helmet_best.pt` trên test A – bước cũ sẽ bị rò rỉ vì `helmet_best.pt` đã thấy các frame test A khi huấn luyện).
- Số mới là phép đo **độ khái quát hóa theo video** đầu tiên đáng tin hơn, nhưng vẫn chỉ trên 5 video test của cùng một camera/góc nhìn (từ trên xuống); mỗi video đóng góp nhiều frame liên tiếp nên số mẫu độc lập hiệu dụng nhỏ hơn nhiều so với 214 ảnh. Không có khoảng tin cậy ở đây.
- **Không có claim nào về độ chính xác trên video thật** hay của sự kiện vi phạm; đây chỉ là mAP của detector trên ảnh tĩnh của split A.

## Kiểm tra giả thuyết nhãn (đếm trên nhãn gốc 4 lớp, không sửa gì)
`label_stats.json` (theo ảnh), `label_stats_per_motorcycle.json` (gán đầu vào xe máy theo tâm hộp nằm trong hộp xe).

| | ảnh | no-helmet / ảnh: 0, 1, 2 | ảnh có ≥ 2 no-helmet | xe máy có ≥ 2 no-helmet | đầu (helmet + no-helmet) / ảnh |
|---|---|---|---|---|---|
| split gốc – train / valid / test | 1014 / 278 / 168 | train 458, 448, 108; valid 124, 138, 16; test 68, 89, 11 | 108 / 16 / 11 | – | tối đa 8 / 6 / 7 |
| **split A – train** | 1026 | 495, 400, 131 | **131** | **95** (trên 2102 xe) | 0–8, đỉnh ở 2 (304 ảnh) |
| **split A – val** | 214 | 74, 140, **0** | **0** | **0** | 1–6 |
| **split A – test** | 214 | 81, 133, **0** | **0** | **0** | 1–4 (đỉnh ở 1: 96 ảnh) |
| extra_1080p | 6 | 0, 2, 4 | 4 | – | 2–4 |

- **Có ảnh ≥ 2 no-helmet không?** Có, nhưng chỉ trong train (131 ảnh; 95 xe có 2 no-helmet) và split gốc (108/16/11). **Không có ảnh nào ≥ 3.** Trong **val và test của split A không có ảnh nào có ≥ 2 no-helmet**, và không xe nào có 2 người không mũ: tập đánh giá hoàn toàn thiếu trường hợp "hai người không mũ trên một xe", dù train có 95 xe như vậy → mAP no-helmet trên val/test không phản ánh trường hợp này.
- Số đầu trên mỗi xe (split A, train): 0 đầu: 53 xe, 1 đầu: 1493, 2 đầu: 538, 3 đầu: 18 (xe máy chở 3 người ≈ 0.9%). Val: 21 / 332 / 104 / 14 (14 xe 3 đầu), test: 0 / 227 / 77 / 0 (không có xe 3 đầu ⇒ luật `overload` không có gì để đo trên test).
- Xe có cả đầu đội mũ và đầu không mũ: train 245, val 108, test 77 – trường hợp "mũ + không mũ trên cùng một xe" là chủ yếu, và là nơi hộp đầu chồng nhau.
- Hộp "đầu" không nhất quán về kích thước: nhiều hộp no-helmet (và vài hộp helmet) trong val bao cả đầu lẫn lưng/vai (xem gallery), trong khi hộp helmet khác chỉ bao mũ ⇒ IoU thấp, mAP50-95 thấp (0.45–0.53 no-helmet) dù mAP50 cao (0.84–0.91). Đây là quan sát từ gallery, chưa đo thống kê.

## Gallery lỗi trên val (`error_gallery/`, conf = 0.25, imgsz 640, `best.pt`)
Ghép theo IoU ≥ 0.5 giữa GT và dự đoán lớp đầu (helmet/no-helmet); `val_error_cases.json|csv` có từng ca (id `E00`…`E50`, ảnh, conf, IoU, nhãn của trợ lý + lý do). Ảnh gallery **không commit** (có mặt người; `error_gallery/` nằm trong `.gitignore`).

| nhóm | số ca | lỗi model | không chắc | nhãn thiếu |
|---|---|---|---|---|
| (a) GT no-helmet bị đoán là helmet | 35 (34 + 1 ca IoU 0.3–0.5) | 19 | 16 | 0 |
| (b) GT no-helmet bị bỏ sót hoàn toàn | 5 | 2 | 3 | 0 |
| (c) dự đoán no-helmet không khớp GT no-helmet nào | 11 | 9 | 1 | 1 |
| **tổng** | **51** (trên 46 ảnh) | **30** | **20** | **1** |

GT no-helmet trên val = 140; 40 cái (29%) không được dự đoán đúng ở conf 0.25 (khớp recall 0.771).
**Phân loại trên là của trợ lý sau khi xem từng ô ảnh – bạn duyệt/ sửa.** Định nghĩa: *lỗi model* = nhãn GT hợp lý, hình ảnh rõ, model sai; *không chắc* = ảnh mơ hồ hoặc chưa có quy ước (khăn trùm đầu/hijab, mũ vải/hood, hai đầu chồng trong cùng hộp); *nhãn thiếu* = thấy rõ đối tượng không có hộp GT.
- **Lỗi tập trung theo video, không phải rải đều:** (a) frame_14 19 ca + frame_10 7 + frame_15 8(+1); (b) frame_14 2, frame_15 3; (c) frame_15 5, frame_10 4, frame_14 2. Trong frame_14, 19 ca (a) + 2 ca (b) là *cùng một người ngồi trước, đầu trần, qua nhiều frame liên tiếp* – thực ra là **một đối tượng**, nên số ca nhiều hơn số lỗi độc lập rất nhiều. Các ca (a)/(b) của frame_15 là **một người đội khăn trùm đầu** qua nhiều frame liên tiếp.
- (a) frame_14 (19, *lỗi model*): đầu trần rõ, GT đúng; model đoán helmet vì mũ của người lái nằm chồng trong hộp GT (hộp GT cao gồm cả lưng).
- (a) frame_10 (7, *không chắc*): người trùm mũ vải; mũ bảo hiểm của người thứ hai chồng trong hộp. frame_15 (8 + 1, *không chắc*): khăn trùm đầu, chưa có quy ước (không mũ bảo hiểm ≠ không đầu che).
- (c): 7 dự đoán no-helmet đặt trên **thân/lưng** người ngay dưới một đầu đã có nhãn (thường là đầu đội mũ), 2 trên **đèn hậu ô tô** (E07, E08) (9 ca *lỗi model*), 1 ca nhãn thiếu (E00: người đội mũ bảo hiểm rõ nhưng không có hộp GT helmet), 1 ca định vị (E49: hộp GT dài gồm khăn + lưng).
- Gợi ý (chưa làm): thống nhất quy ước hộp "đầu" (chỉ đầu hay cả vai), quyết định nhãn cho khăn trùm đầu, bổ sung ảnh có ≥ 2 người không mũ vào val/test, thêm ảnh ô tô làm ảnh nền.

## Tệp
`metrics_3cls.json`, `helmet_3cls_splitA/` (args, results.csv, đường cong, ảnh batch, `weights/` rỗng), `eval_3cls_{val,test,extra_1080p}/` (confusion matrix, PR…), `weights/best.pt` (không commit), `label_stats*.json`, `val_error_cases.{json,csv}`, `error_gallery/` (không commit), `scripts/` (các script phân tích đã chạy).
