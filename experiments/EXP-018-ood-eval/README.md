# EXP-018 – bộ đánh giá detector ngoài miền từ `test_video.mp4` (2026-10-19)

Mục đích: đo các detector helmet trên cảnh **khác miền huấn luyện** (camera giám sát nhìn chéo ở nút giao, EXP-017) mà không dùng dữ liệu huấn luyện. Chưa có kết quả: bộ nhãn chưa được sửa tay. Không train, không chạy pipeline.

## Bộ ảnh
- 15 frame cách đều trên toàn clip (tâm của 15 đoạn bằng nhau, cách ~52 frame ≈ 1.75 s; frame 26, 78, … 760), PNG không nén, `data/ood_test_video/images/` (**gitignored**: có người và biển số). Tạo bằng `python tools/make_ood_set.py frames`.
- Cấu hình: `configs/eval/ood_test_video.yaml` (video, ROI, hai model, NMS, lớp).
- Một clip, một camera, 15 frame liền kề về cảnh (cùng nền, cùng nút giao): các frame **không độc lập**; kết quả là "một cảnh này", không đại diện cho các camera khác.

## Vùng đánh giá (ROI) = `[0, 240, 1280, 720]` (2/3 phía dưới khung 1280×720)
Chỉ tính ground truth **và** dự đoán có **tâm** hộp nằm trong ROI. Lý do chọn: từ các phát hiện của cả hai model ở conf 0.15 trên 15 frame (331 đầu sau khi gộp), chiều cao hộp đầu theo vị trí y: y 160–240 ≈ 39 px nhưng conf trung vị chỉ 0.28 (7% ≥ 0.5; người ở xa, mũ không phân biệt được bằng mắt), y 240–480 ≈ 38–51 px, y 480–720 ≈ 63–86 px. 2/3 dưới giữ 286 / 331 đầu (86%) và loại phần xe nhỏ ở xa. ROI được chọn từ kích thước phát hiện, **không** từ nhãn đúng. Ảnh minh họa: `data/ood_test_video/roi_preview.jpg` (vùng ngoài ROI bị làm tối; gitignored vì có người).
Lưu ý thang đo: ở đầu vào 640 px (ảnh 1280 co 0.5×) đầu cao ≈ 19–25 px, trong khi ảnh huấn luyện 3840×2160 co về 640 cho đầu cao ≈ 35 px (trung vị chiều cao hộp đầu ở train split A = 211 px trên ảnh 3840 px, `box_consistency/boxes.csv`): một phần "ngoài miền" là chênh lệch kích thước đối tượng, không chỉ khác cảnh.

## Nhãn gợi ý (pre-label) – chỉ là gợi ý, người gán nhãn phải sửa toàn bộ
- Chạy **cả hai** model (`helmet_best.pt` cũ và `experiments/EXP-015-…/weights/best.pt`) ở conf 0.15, gộp bằng NMS theo lớp (IoU 0.5, giữ hộp có conf cao nhất). **Dùng cả hai model để giảm thiên lệch về một bên**: nhãn gợi ý chỉ từ một model sẽ làm model đó trông tốt hơn khi người sửa nhãn có xu hướng giữ hộp có sẵn.
- Chỉ xuất hộp có tâm trong ROI: 407 hộp (helmet 176, motorcycle 121, no-helmet 110) từ 351 (cũ) + 327 (EXP-015) hộp trước khi gộp. Ở conf 0.15 chắc chắn có nhiều hộp sai (dương tính giả) và thiếu hộp.
- **Người gán nhãn phải kiểm tra và sửa toàn bộ**: xóa hộp sai, thêm hộp thiếu, sửa lớp và vị trí, cho **mọi** đối tượng có tâm trong ROI (phần ngoài ROI không được chấm). Quy tắc lớp đầu và hộp: `docs/LABELING_GUIDE.md` (helmet = chỉ mũ bảo hiểm; khăn, mũ vải, nón lá = no-helmet; box đầu tới vai là được, tới thắt lưng/ba lô là sai). Đầu không nhìn rõ: không vẽ hộp (ảnh tĩnh chưa có cờ ambiguous).
- Gói tải lên Roboflow: `data/ood_test_video/ood_prelabels.zip` (≈ 23 MB, **không commit**): `images/*.png`, `labels/*.txt` (YOLO, 3 lớp: helmet, motorcycle, no-helmet), `data.yaml`. Không có lớp `plate` (không được đánh giá ở đây).
- Tạo bằng `python tools/make_ood_set.py prelabel`.

## Đánh giá (khi đã có nhãn sửa tay)
```
python tools/eval_ood.py --labels <thư mục export YOLO từ Roboflow> --json experiments/EXP-018-ood-eval/results.json
```
Chạy cả hai model của config, báo **trong ROI**, theo lớp: số GT (`n_gt`), số dự đoán, P và R (conf 0.25, IoU 0.5), AP50 và AP50-95. Tên file nhãn của Roboflow (`…_png.rf.<hash>`) được nhận ra; lớp ngoài danh sách (vd. `plate`) bị bỏ.
- **AP ở đây là kiểu COCO (101 điểm, không nội suy tuyến tính về điểm (1, 0))**, không phải công thức `val` của Ultralytics (cho 0.995 với detector hoàn hảo, và cho 0.75 khi recall dừng ở 0.5 thay vì 0.5). Vì vậy **không so trực tiếp** với mAP của EXP-000 / EXP-015; hai model trong cùng lần chạy dùng cùng hàm nên so sánh với nhau được.
- Chỉ số chính cho helmet/no-helmet theo quyết định EXP-015: AP50 (AP50-95 báo kèm và bị giới hạn bởi độ lỏng của box).
- Chỉ ~15 frame của một cảnh: số GT mỗi lớp sẽ nhỏ (xem `n_gt`), nên khoảng bất định lớn; không có khoảng tin cậy ở đây và không có claim nào về độ chính xác trên video khác.
- Kiểm tra công cụ: `tests/test_eval_ood.py` (AP/P/R tính tay, lọc ROI, đọc nhãn Roboflow). Một lần chạy thử CLI trên chính nhãn gợi ý chỉ để xác nhận công cụ chạy được; con số đó **vòng tròn** (chấm model bằng nhãn do chính model tạo ra) nên không ghi và không dùng làm kết quả.
