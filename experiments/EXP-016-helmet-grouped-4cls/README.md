# EXP-016 – huấn luyện lại detector helmet 4 lớp (helmet, motorcycle, no-helmet, plate) trên split A

Người chạy: chủ dự án, trên Kaggle (`results_4cls.zip`, sha256 `f1709e8f4a80c17f55d3f42165fcbcf6b05daea10a733ba701a1767914efd15a`; bản trùng byte ở `<REPO>\results_4cls.zip` và `Downloads\results_4cls.zip`, dùng bản trong repo). Giải nén ở đây; `last.pt` bỏ qua. Phân tích do trợ lý chạy cục bộ (chỉ suy luận; không train, không sửa `data/raw`).

| | |
|---|---|
| mô hình | YOLO11s (`yolo11s.pt`), 4 lớp, split A (EXP-013): train 1026 / val 214 / test 214 ảnh (+ `extra_1080p` 6 ảnh) |
| `weights/best.pt` | sha256 **`6452d6354fbd09cd873b7201efd618238e56c6b9034fcf9cd894f73ca02bb19b`**, 19 160 218 byte; bị `.gitignore` chặn (`*.pt`, và `results_*.zip`), đã kiểm bằng `git check-ignore` |
| môi trường | Kaggle Tesla T4, torch 2.11.0+cu128, Python 3.13.15, ultralytics 8.4.26 (giống EXP-015) |
| trạng thái trong app | **không dùng**; `helmet_best.pt` của app không đổi |

## Quá trình huấn luyện (`helmet_4cls_splitA/results.csv`)
- Dừng ở **epoch 80** (80 dòng), epoch tốt nhất (fitness 0.1·mAP50 + 0.9·mAP50-95) là **55**: dừng do **patience 25** (đúng 25 epoch sau epoch tốt nhất), **không phải** do giới hạn thời gian.
- Tổng thời gian train **3 537 s (0.98 h)**, ≈ 43.7 s/epoch; giới hạn `time=2.5` h (9 000 s) không bị chạm.
- Lần chạy 4cls đầu tiên bị treo (xem `NOTES.md`).

## Kết quả theo lớp (`metrics_4cls.json`; mAP50 là chỉ số chính cho helmet/no-helmet)
**Test (214 ảnh, 5 video)**
| lớp | P (4cls) | R (4cls) | **mAP50 (4cls)** | mAP50-95 (4cls) | mAP50 (3cls, EXP-015) | mAP50-95 (3cls) | Δ mAP50 |
|---|---|---|---|---|---|---|---|
| helmet | 0.777 | 0.973 | **0.929** | 0.577 | 0.962 | 0.623 | -0.033 |
| motorcycle | 0.944 | 0.989 | **0.994** | 0.821 | 0.995 | 0.828 | -0.001 |
| no-helmet | 0.761 | 0.624 | **0.679** | 0.290 | 0.844 | 0.453 | -0.164 |
| plate | 0.947 | 0.982 | **0.991** | 0.658 | – | – | – |
| tất cả (4 lớp / 3 lớp) | 0.857 | 0.892 | 0.898 | 0.586 | 0.933 | 0.635 | (không so được: khác số lớp) |

**Val (214 ảnh, 4 video)**
| lớp | P (4cls) | R (4cls) | **mAP50 (4cls)** | mAP50-95 (4cls) | mAP50 (3cls, EXP-015) | mAP50-95 (3cls) | Δ mAP50 |
|---|---|---|---|---|---|---|---|
| helmet | 0.819 | 0.970 | **0.951** | 0.597 | 0.950 | 0.579 | +0.001 |
| motorcycle | 0.975 | 0.996 | **0.993** | 0.707 | 0.995 | 0.745 | -0.002 |
| no-helmet | 0.972 | 0.742 | **0.911** | 0.540 | 0.913 | 0.529 | -0.002 |
| plate | 0.950 | 0.907 | **0.966** | 0.638 | – | – | – |
| tất cả (4 lớp / 3 lớp) | 0.929 | 0.904 | 0.955 | 0.621 | 0.952 | 0.618 | (không so được: khác số lớp) |

**extra_1080p (6 ảnh 1920×1080, nguồn không rõ) – n = 6, KHÔNG kết luận**
| lớp | P (4cls) | R (4cls) | **mAP50 (4cls)** | mAP50-95 (4cls) | mAP50 (3cls, EXP-015) | mAP50-95 (3cls) | Δ mAP50 |
|---|---|---|---|---|---|---|---|
| helmet | 0.353 | 0.750 | **0.458** | 0.054 | 0.292 | 0.064 | +0.166 |
| motorcycle | 0.974 | 1.000 | **0.995** | 0.494 | 0.995 | 0.483 | +0.000 |
| no-helmet | 0.749 | 0.300 | **0.557** | 0.269 | 0.819 | 0.319 | -0.262 |
| plate | 0.383 | 0.417 | **0.231** | 0.050 | – | – | – |
| tất cả (4 lớp / 3 lớp) | 0.615 | 0.617 | 0.560 | 0.217 | 0.702 | 0.289 | (không so được: khác số lớp) |

### Đọc đúng
- **Lần này đổi nhiều thứ cùng lúc nên chênh lệch với EXP-015 không quy được cho riêng yếu tố nào.** Bạn nêu 2 thứ (thêm lớp plate, `fliplr: 0.0`); so `args.yaml` hai lần chạy còn thấy **thêm 2 khác biệt**: `workers: 2` (EXP-015: 8) và `time: 2.5` (EXP-015: không đặt). Ultralytics khi có `time` sẽ **tính lại số epoch theo thời gian** và co giãn lịch học: từ cột `lr/pg0` suy ra độ dài cosine ≈ **178 epoch** ở EXP-016, so với đúng 150 ở EXP-015 (ước lượng từ lr, giả định lr0 = 0.001429 như EXP-000). Vì vậy lịch học (learning-rate) cũng khác. Các tham số còn lại giống hệt (`diff` args.yaml).
- no-helmet trên test: **mAP50 0.679 (4cls) so với 0.844 (3cls)**, R 0.624 so với 0.744; helmet 0.929 so với 0.962; motorcycle 0.994 so với 0.995. Confusion matrix test (`eval_4cls_test/confusion_matrix.png`): trong 133 no-helmet thật, 4cls tìm đúng **71**, nhầm thành helmet **31**, bỏ sót **31**; EXP-015: 105 đúng, 21 nhầm, 7 bỏ sót. Hai lần chạy cho lớp no-helmet khác nhau nhiều hơn mức ta có cách giải thích; với 5 video test, phương sai theo lần chạy (seed, thứ tự dữ liệu, lịch học) **chưa được đo** và có thể là một phần.
- Val: no-helmet mAP50 0.911 (4cls) so với 0.913 (3cls): gần như bằng nhau, nên chênh lệch lớn chỉ xuất hiện ở test.
- Không có claim nào về độ chính xác trên video thật.

## Lớp plate: 4cls so với `plate_best.pt` (mục đích: đánh giá khả năng gộp model theo G9)
Trên **271 box plate** của test split A (214 ảnh), cùng hàm đánh giá (`tools/eval_ood.py`: AP kiểu COCO, P/R ở conf 0.25 và IoU 0.5; số AP **không** so trực tiếp với mAP của Ultralytics); `scripts/plate_compare.py`, `plate_vs_plate_best*.json`.

| giao thức | model | P | R | AP50 | AP50-95 |
|---|---|---|---|---|---|
| toàn khung (vai trò của model 4 lớp) | 4cls, lớp plate | 0.964 | 0.982 | **0.988** | 0.647 |
| toàn khung | `plate_best.pt` | 0.048 | 0.004 | 0.007 | 0.001 |
| ảnh cắt theo box xe máy GT (vai trò của `plate_best.pt` trong pipeline; 220/271 box plate nằm trong một box xe máy) | 4cls, lớp plate | 0.000 | 0.000 | 0.000 | 0.000 |
| ảnh cắt theo box xe máy GT | `plate_best.pt` | 0.076 | 0.045 | 0.016 | 0.002 |
- 4cls tìm được biển trên toàn khung của miền này (P 0.964, R 0.982); điều này **ủng hộ** khả năng gộp plate vào model một lần chạy **trong miền ảnh của bộ helmet** (nhìn từ trên xuống, biển nhỏ, cùng nhóm video với train).
- `plate_best.pt` gần như **không phát hiện được gì** trên cùng ảnh (R 0.004 toàn khung, 0.045 khi cắt theo xe). **Không được diễn giải là "plate_best.pt tệ"**: nó được dùng và (theo EXP-014) có vẻ huấn luyện trên ảnh 640×640 của camera cổng gửi xe, nơi biển rất lớn; ở ảnh trên cao biển nhỏ nó không phải đối tượng cùng miền. **Nguồn gốc `plate_best.pt` chưa rõ** (EXP-000: notebook không khớp checkpoint), nên không loại trừ được việc nó từng thấy ảnh tương tự; phép so sánh này chỉ cho thấy "model nào dùng được trong miền nào", không phải "model nào tốt hơn".
- Giao thức ảnh cắt làm hỏng 4cls (AP 0): model 4 lớp đã học ở thang toàn khung; cắt rồi phóng to ra 640 là sai thang. Điều này nói rằng nếu gộp, plate phải được phát hiện **trên toàn khung** (ảnh lớn), rồi mới ghép với xe; chất lượng khi biển ở xa/nhỏ (video test: ~26×22 px ở 720p) chưa đo ở đây.
- Chưa đo: 4cls trên `test_video.mp4` cho vai trò biển (không có GT plate ở đó).

## Tệp
`metrics_4cls.json`, `helmet_4cls_splitA/` (args, results.csv, đường cong, confusion matrix; ảnh batch bị bỏ vì có người), `eval_4cls_{val,test,extra_1080p}/`, `weights/best.pt` (không commit), `plate_vs_plate_best*.json`, `scripts/`, `NOTES.md`.
