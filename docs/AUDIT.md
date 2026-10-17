# AUDIT – SentinelAI (phát hiện vi phạm giao thông)

Ngày audit: 2026-10-17. Phạm vi: chỉ đọc. Không sửa code, không train, không cài package.
Quy ước: **[V]** = đã xác minh trực tiếp từ file/lệnh read-only; **[S]** = suy đoán/suy luận.

> **Hạn chế lớn của audit này:** repo chỉ có `app.py` (344 dòng) + `templates/index.html` + 4 file weights + 1 video test.
> **Không có**: `docs/JD.md`, README, requirements, git (`Is a git repository: false`), code/notebook huấn luyện, dataset, log, biểu đồ kết quả, test.
> Mọi thông tin về dữ liệu và huấn luyện dưới đây được **khôi phục từ metadata nhúng trong file `.pt`** (train_args, train_metrics, train_results), không phải từ tài liệu của dự án.

---

## 1. Tóm tắt

SentinelAI hiện là **một demo web (FastAPI + MJPEG) chạy 3 mô hình YOLO11 trên video, kèm vài luật hình học viết cứng** để gắn nhãn vi phạm: không đội mũ bảo hiểm, chở quá 2 người, vượt đèn đỏ, đi ngược chiều (và đi sai làn – đang tắt). Plate detector cắt ảnh biển số làm "bằng chứng". Điểm mạnh lớn nhất: có pipeline end-to-end chạy được, dùng tracking (BoT-SORT qua Ultralytics), HSV để nhận màu đèn, và hai model tự fine-tune trên dữ liệu có vẻ là xe máy/biển số Việt Nam, với mAP50 ≈ 0.99 trên tập val. Tuy nhiên đây **chưa phải một dự án nghiên cứu/kỹ thuật tái lập được**: không có dữ liệu, code train, log, đánh giá mức vi phạm, hay export edge.

**3 vấn đề nghiêm trọng nhất:**
1. **Không có đánh giá mức "vi phạm"** và số mAP báo cáo (0.994/0.995) rất có thể bị thổi phồng (chọn best epoch trên chính tập val, không có test, nguy cơ trùng/leak ảnh Roboflow). Logic vi phạm (đèn đỏ, ngược chiều) là heuristic cứng cho đúng một video, chưa từng được đo precision/recall.
2. **Không tái lập được**: không có dữ liệu, code train, requirements, seed/version dữ liệu; đường dẫn train nằm trên Kaggle (`/kaggle/input/...`) và chỉ biết qua metadata trong `.pt`.
3. **Chưa sẵn sàng Edge**: chạy 3 model PyTorch `.pt` trên mỗi frame (kể cả plate model gọi trên từng vehicle crop), chưa export ONNX/TensorRT/INT8, không đo FPS/latency có hệ thống, không Docker, và có lỗ hổng bảo mật cơ bản (upload path, nguồn video tùy ý, bind 0.0.0.0, không auth).

---

## 2. Hiện trạng chi tiết

### A. Bài toán & phạm vi
| Loại vi phạm | Trạng thái | Bằng chứng |
|---|---|---|
| Không đội mũ bảo hiểm | Có | [V] `app.py:201-214`: đếm frame có "no-helmet" gắn với xe máy, ≥5 → vi phạm (có hysteresis giảm dần). |
| Chở quá 2 người | Có (xấp xỉ) | [V] `app.py:220-222`: `passenger_count` = số **đầu** (helmet/no-helmet) trong box xe máy, >2. Đếm đầu chứ không đếm người. |
| Vượt đèn đỏ | Có (thô) | [V] `app.py:229-243`: hễ đèn đỏ **và** điểm đáy-giữa box nằm trong polygon cố định → vi phạm. Không có vạch dừng, không xét hướng/đường đi qua vạch. |
| Đi ngược chiều | Có (thô) | [V] `app.py:145-151`: ô tô/xe tải/bus có `cy - old_cy < -30` (đi lên trong ảnh) → ngược chiều. Giả định cứng "xe đúng chiều thì đi xuống". Không áp cho xe máy. |
| Đi sai làn | Tắt | [V] `app.py:88-89`: `bike_lane_zone = None`, polygon bị comment, nhãn "FAKE BIKE LANE". |
| Lấn vạch / đỗ sai / nhiều loại khác | Không có | [V] không có code. |

- **Logic vi phạm** = detection + tracking (BoT-SORT mặc định, `track(..., persist=True)`, [V] `app.py:115,172`) + luật hình học/heuristic. Không có state machine đèn, không có lane/ROI cấu hình theo camera, không có khái niệm "sự kiện vi phạm" bền vững (chỉ có list 12 đường dẫn ảnh trong RAM).
- **Input**: video file (`test_video.mp4` mặc định), upload video, URL/RTSP/index camera qua form (`/update_camera`). Không có ảnh tĩnh. [V] `app.py:34,309-335`.

### B. Dữ liệu
**Toàn bộ dữ liệu không có trong repo.** Chỉ biết qua `train_args["data"]` trong checkpoint [V]:

| Model | Đường dẫn data trên Kaggle (từ checkpoint) | Classes |
|---|---|---|
| helmet_best.pt | `/kaggle/input/datasets/tranquangduyenk17ct/helmet-detection-1/helmet plate detection 4.v1i.yolov8/data.yaml` | `{0: helmet, 1: motorcycle, 2: no-helmet}` |
| plate_best.pt | `/kaggle/input/datasets/tranquangduyenk17ct/vietnam-license-plate-v1/data.yaml` | `{0: '0'}` (1 class, tên chưa đặt) |

- Tên `v1i.yolov8` gợi ý export **Roboflow** (YOLOv8 format) [S]. Nguồn gốc gốc, license, số ảnh, số nhãn, phân bố class: **không xác minh được** (không có dataset, không có `data.yaml`/README). "Nhiều tập dữ liệu" như mô tả: repo chỉ cho thấy **2 tập**, không thấy bước gộp/mapping class nào → câu hỏi về nhất quán mapping/trùng lặp **không đánh giá được**.
- Chia train/val/test: checkpoint cho thấy `split: "val"`; **không có bằng chứng về test set**. [V] Chỉ có metrics trên val. Nguy cơ leakage: không kiểm tra được; với Roboflow + ảnh trích từ video, frame liền kề rất hay nằm ở cả train/val [S].
- Augmentation lúc train [V]: mosaic 1.0, mixup 0.1 (helmet) / 0.0 (plate), HSV (0.015/0.7/0.4), fliplr 0.5, scale 0.5, translate 0.1, degrees 10 (helmet) / 0 (plate), randaugment, erasing 0.4, close_mosaic 10. Đây là default Ultralytics + chút chỉnh; không có augmentation riêng cho đêm/mưa/mờ.
- Đặc thù Việt Nam: biển số tên "vietnam-license-plate" và crop mẫu có biển Việt Nam (ví dụ `static/violations/v_1774497419_C_221.jpg` hiển thị biển dạng "65A-768.96") [V]. Mưa/ban đêm/góc camera thực tế: **không có bằng chứng**.

### C. Mô hình & huấn luyện [V từ checkpoint]
| | helmet_best.pt | plate_best.pt |
|---|---|---|
| Kiến trúc | YOLO11s (`yolo11s.yaml`), 9,428,953 tham số | YOLO11n, 2,590,035 tham số |
| Pretrained | `yolo11s.pt` (COCO) | `yolo11n.pt` (COCO) |
| Framework | Ultralytics 8.4.26, PyTorch (venv: torch 2.7.1+cu118) | Ultralytics 8.4.25 |
| Epochs | 150 (patience 25), best epoch = 139 | 50 (patience 100), best epoch = 46 |
| imgsz / batch | 640 / 32 | 640 / 32 |
| Optimizer / LR | `auto`, lr0 0.01, lrf 0.01, **cos_lr True** | `auto`, lr0 0.01, **cos_lr False** |
| Seed | `seed: 0`, `deterministic: True` | cùng |
| Phần cứng | Kaggle (`device: "0"`, GPU cụ thể không ghi) | Kaggle (`device: null`) |
| Tổng thời gian train | 5,229 s ≈ 1.45 h (≈ 35 s/epoch) | 3,274 s ≈ 0.91 h (≈ 65 s/epoch) |
| Ngày | 2026-03-24 | 2026-03-23 |

- Model thứ 3 `yolo11s.pt` là **COCO gốc** (không fine-tune) dùng cho car/truck/bus và traffic light. `yolo11n.pt` nằm trong repo nhưng **không được dùng** trong `app.py` [V].
- Ghi chú: `plate_best.pt` mất 65 s/epoch dù nhỏ hơn 2× → dataset plate có thể lớn hơn hoặc I/O chậm [S].
- **Không có** log thí nghiệm hay so sánh phiên bản: không có results.csv, runs/, TensorBoard/W&B/MLflow. Chỉ còn đường cong `train_results` nhúng trong checkpoint (150/50 dòng). Checkpoint ghi `git: {commit: None}` → không gắn được với phiên bản code.
- Huấn luyện chạy ở Kaggle, không phải trên máy hiện tại (RTX 3050 Laptop 4 GB, `torch.cuda.is_available() == True`) [V].

### D. Đánh giá
Metrics duy nhất tồn tại (từ `train_metrics` cuối, **trên tập val**) [V]:

| Model | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| helmet (3 class gộp) | 0.983 | 0.989 | 0.994 | 0.809 |
| plate (1 class) | 0.994 | 0.991 | 0.995 | 0.752 |

- Không có P/R/AP **theo class**, không confusion matrix, không F1, không PR curve được lưu trong repo. (Chúng có thể có ở Kaggle, nhưng repo không chứa.)
- **Không có đánh giá mức vi phạm** (event-level: bao nhiêu vi phạm thật bị bắt/bỏ sót, bao nhiêu báo nhầm) trên video có ground truth. [V] không có file annotate video/ground truth.
- Chỉ có một video test 1280×720, 30 fps, 787 frame (≈ 26 s) [V]. Thư mục `static/violations/` có 124 ảnh nhưng chỉ **21 kích thước byte khác nhau**, mỗi `M_77`, `M_379`... xuất hiện 4 lần → đó là **cùng một nhóm ~21 vi phạm lặp lại qua nhiều lần chạy cùng một clip** [V]. Nghĩa là "kết quả" thực tế hiện có = vài lần chạy trên một clip 26 s, chưa được kiểm chứng đúng/sai.
- Tốc độ: code có hiển thị FPS trên frame (`app.py:289-293`) nhưng **không có số đo được ghi lại**; mỗi frame chạy `coco.track` + `helmet.track` trên toàn frame + `plate_model` trên **mỗi** crop vi phạm. Chưa đo (cần xin phép chạy). Phần cứng dev: RTX 3050 Laptop 4 GB.
- Điểm yếu (chưa đo, phân tích code) [S]: xe máy là phương tiện chính ở VN nhưng đèn đỏ/ngược chiều **loại xe máy** khỏi các luật đó (`name != "motorcycle"`, `app.py:133`); đèn giao thông phụ thuộc COCO `traffic light` (nhỏ, xa → HSV với ngưỡng chỉ 5 pixel); ban đêm/mưa chưa có dữ liệu.

### E. Pipeline suy luận & OpenCV
- OpenCV dùng cho: đọc video/camera/RTSP (`VideoCapture`), resize cố định 1280×720, vẽ box/polygon/text, `pointPolygonTest` cho ROI, **HSV + `inRange` nhận màu đèn đỏ/xanh** (`app.py:60-73`, không có vàng), crop, `imencode` JPEG để stream, `imwrite` lưu bằng chứng. Hợp lý nhưng toàn bộ là mức cơ bản.
- Tracking: **có** (`model.track(persist=True)` của Ultralytics, BoT-SORT mặc định; `lap` có trong venv). **Không** có đếm đối tượng/đếm lưu lượng.
- **OCR biển số: không có.** `paddleocr 3.4.0` + `paddlepaddle 3.3.0` đã cài trong venv nhưng **không được import** trong `app.py` [V] → ứng viên bị bỏ dở.
- Cấu trúc code: **một file `app.py` duy nhất**, gộp web server + load model + logic vi phạm + vẽ + lưu file trong một generator `generate_frames()` ~220 dòng, dùng nhiều biến global. Không có module/class/test. [V]
- Các lỗi/hạn chế logic đáng chú ý [V, đọc code]:
  - `violation_counters` (đếm no-helmet) **không bao giờ reset** khi đổi nguồn (`/update_camera`, `/upload_video`) trong khi `captured_violator_ids` thì có; `track_history` là local nên reset khi mở stream mới. Track-ID tái sử dụng giữa video khác nhau gây nhiễu.
  - Khi video hết, `cap.set(POS_FRAMES, 0)` nhưng bộ tracker **không reset** → ID vẫn tăng, hành vi "lặp" khó kiểm soát.
  - Chỉ detection **có `boxes.id`** mới được xử lý (`app.py:117,175`): object chưa được gán track bị bỏ qua hoàn toàn.
  - `current_light_state` bị ghi đè bởi **traffic light cuối** trong vòng lặp; không có smoothing theo thời gian; bất kỳ đèn nào (kể cả đèn cho làn khác) quyết định trạng thái toàn khung.
  - Ảnh bằng chứng được cắt từ **frame đã vẽ overlay** (box "PLATE" xanh xuất hiện trong ảnh, thấy rõ ở `v_1774497380_M_77.jpg`) và chỉ là crop biển số ~50–60 px; **không lưu ảnh toàn cảnh, timestamp, loại vi phạm, ID**. Nếu không phát hiện biển → vi phạm không được ghi lại (`app.py:265-282`).
  - Chạy nhiều client `/video_feed` → mỗi client có một generator riêng nhưng dùng chung `cap` → tranh chấp frame, tốn gấp đôi GPU. `continue` khi `cap` chưa mở là busy-loop 100% CPU (`app.py:99`).

### F. Mức độ sẵn sàng cho Edge
- Kích thước [V]: helmet 19,167,578 B (≈18.3 MiB, 9.43 M params); plate 5,471,770 B (≈5.2 MiB, 2.59 M params); COCO yolo11s 19,313,732 B (9.46 M params). FLOPs: **chưa tính** (không chạy; YOLO11s ≈ 21.5 GFLOPs @640 và YOLO11n ≈ 6.5 theo số công bố của Ultralytics [S – cần xác minh]).
- **Chưa** export ONNX / TensorRT / OpenVINO / TFLite; không FP16/INT8/pruning. Không có file `.onnx/.engine/.tflite` trong repo [V]. (`format: torchscript` trong train_args chỉ là giá trị mặc định, không phải bằng chứng đã export.)
- Phụ thuộc gây khó: `torch 2.7.1+cu118` (nặng), Ultralytics (**AGPL-3.0** – rào cản pháp lý nếu đóng gói thương mại/edge), `paddlepaddle` (thừa, không dùng), `opencv_contrib_python` và `opencv_python` **cùng cài** (xung đột), đường dẫn tương đối `"helmet_best.pt"`... và ROI hard-code theo 1280×720.
- **Kiến trúc cần 3 model/frame**: COCO(s) toàn frame + helmet(s) toàn frame + plate(n) trên từng crop. Với Edge (Jetson Nano/Orin Nano, RPi, Hailo...) đây là chi phí lớn; có thể gộp COCO+helmet bằng một model tự train (xe máy/ô tô/người/mũ/đèn), và chỉ chạy plate khi có sự kiện [S].
- Docker/REST/demo app: **có demo web** (FastAPI + template, endpoints `/`, `/video_feed`, `/update_camera`, `/upload_video`, `/api/violations`). **Không** có Dockerfile, requirements.txt, hay API tách khỏi UI.
- Bảo mật [V]: `file.filename` dùng thẳng làm đường dẫn (`app.py:325`, có nguy cơ path traversal/ghi đè), `/update_camera` nhận chuỗi tùy ý đưa vào `cv2.VideoCapture` (đọc file cục bộ/URL tùy ý), `host="0.0.0.0"`, không auth, `reload=True` trong code chạy chính, không giới hạn kích thước upload, ảnh vi phạm không bao giờ bị xóa (124 file tích lũy).

### G. Kỹ thuật phần mềm & tái lập
- Seed cố định: chỉ trong train (`seed: 0, deterministic: True`) [V]. Không có version dữ liệu, version model (tên file `*_best.pt` không có tag), config tách riêng (mọi ngưỡng như conf 0.35/0.55, 30/15/-30, polygon, 5 frame... nằm rải rác trong code).
- Experiment tracking: **không có** (không MLflow/W&B/TensorBoard artefact).
- Test: **không có**. Notebook: không có trong repo. README/requirements/.gitignore: không có.
- Git: **không phải repo git** → không có lịch sử để đọc, không có bằng chứng file lớn bị commit (nhưng sẽ bị nếu `git init` mà không có `.gitignore`: `venv/`, `*.pt`, `*.mp4`, `static/violations/`).
- Secrets: không thấy trong `app.py`/`index.html` [V].
- Tên Kaggle user (`tranquangduyenk17ct`) nằm trong checkpoint → khi công bố weights lên GitHub/HF sẽ lộ metadata đường dẫn; nên xử lý khi export.

---

## 3. Bảng đối chiếu với JD

> **`docs/JD.md` không tồn tại** (thư mục `docs/` chưa có). Vì vậy bảng dưới **không thể đối chiếu từng dòng JD**. Tôi dùng các yêu cầu thường gặp của vị trí AI Engineer Computer Vision và các tiêu chí bạn nêu (OpenCV, mô hình mạnh/chính xác, Edge, báo cáo tái lập). **Hãy gửi JD để tôi cập nhật bảng.**

| Yêu cầu (giả định) | Mức | Bằng chứng |
|---|---|---|
| Phát hiện đối tượng bằng deep learning (YOLO) | Đã đáp ứng | 2 model fine-tune, mAP50≈0.99 trên val (mức tin cậy: xem mục 4). |
| Fine-tune/train model trên dữ liệu tự thu thập/tổng hợp | Một phần | Có train 2 model trên Kaggle, nhưng không có code/dữ liệu/log trong repo. |
| Thành thạo OpenCV | Một phần | Có HSV, ROI, polygon test, I/O; chưa có xử lý ảnh nâng cao (calibration, homography, cải thiện ảnh đêm/mưa). |
| Object tracking | Một phần | BoT-SORT qua Ultralytics, chưa tự cấu hình/đánh giá (MOTA/IDF1). |
| Đánh giá, báo cáo kết quả (metrics, phân tích lỗi) | Chưa có | Không có confusion matrix, per-class, event-level, báo cáo. |
| Tối ưu & triển khai Edge (ONNX/TensorRT/INT8) | Chưa có | Không có artifact export, không đo FPS/latency. |
| Đóng gói/triển khai (Docker, API) | Một phần | Có FastAPI + web UI; không Docker/requirements. |
| Xử lý dữ liệu, annotation, augmentation | Một phần | Chỉ augmentation mặc định; không có pipeline dữ liệu/kiểm chất lượng. |
| MLOps/tracking, tái lập, code sạch, test | Chưa có | Không git, không test, không tracking, một file monolithic. |
| OCR biển số | Chưa có | Có thư viện paddleocr trong venv nhưng không dùng. |

---

## 4. Rủi ro & lỗi tiềm ẩn

1. **Metric có khả năng bị thổi phồng (nói thẳng):** mAP50 0.994 và 0.995 là gần-bão-hòa. Các số này (a) tính trên `val`, cũng là tập dùng để **chọn best.pt** (best epoch 139/150 và 46/50) → lạc quan; (b) không có test set độc lập; (c) dữ liệu Roboflow/frame video dễ có trùng/gần-trùng giữa các split [S]; (d) chỉ một video test 26 s. **Đừng đưa các số này vào CV/portfolio như năng lực tổng quát hóa** trước khi có test độc lập hoặc video chưa từng thấy.
2. **Data leakage:** *chưa xác minh* nhưng rủi ro cao (xem 1c). Cần kiểm tra bằng perceptual hash/scene grouping khi có dữ liệu.
3. **"Độ chính xác vi phạm" chưa được đo:** không biết tỉ lệ báo nhầm. Ví dụ red-light: xe **đang dừng chờ** trong polygon lúc đèn đỏ sẽ bị gắn "RED-LIGHT RUNNER" (`app.py:238-243`). Wrong-way: bất kỳ xe đi "lên" trong ảnh bị coi ngược chiều → khác góc camera là sai hoàn toàn.
4. **Không tái lập:** không dữ liệu, code train, requirements/lockfile, hay seed pipeline; checkpoint `git: None`. Chỉ còn hyperparameters (đã lấy được ở mục C).
5. **Phiên bản/chuẩn bị môi trường:** venv chứa torch+paddle+2 bản OpenCV; `numpy 2.4.3` với paddle/OpenCV có thể xung đột [S].
6. **Pháp lý:** Ultralytics AGPL-3.0 (cả code và weights phát sinh); dataset license **chưa biết** (Roboflow Universe thường CC BY 4.0 nhưng phải kiểm từng tập).
7. **Bảo mật** demo (mục F). Nếu đưa lên public → phải sửa trước.
8. **Quyền riêng tư:** ảnh biển số/người trong `static/violations` và `test_video.mp4` có thể là dữ liệu cá nhân; cần xem xét nếu công bố.

---

## 5. Cơ hội cải tiến (CHƯA phải kế hoạch)

Sắp theo tỷ lệ tác động/công sức (cao → thấp).

| # | Cơ hội | Tác động | Công sức | Phụ thuộc |
|---|---|---|---|---|
| 1 | Tái cấu trúc repo: `git init` + `.gitignore`, README, requirements/lockfile, tách module (detector, tracker, rules, io), config YAML cho ngưỡng/ROI | Cao | Thấp | — |
| 2 | Đo & ghi lại FPS/latency baseline của pipeline hiện tại (PyTorch FP32) trên RTX 3050 | Cao (làm mốc) | Thấp | Cho phép chạy |
| 3 | Export ONNX → TensorRT FP16 / ONNX Runtime; báo cáo trước/sau (latency, mAP, kích thước) | Cao | Trung bình | #2; thiết bị Edge mục tiêu |
| 4 | Bộ test độc lập: tải lại danh sách ảnh/split, chống leakage (hash + group theo video), báo cáo per-class P/R/AP, confusion matrix trên **test** | Cao | Trung bình | Cần dữ liệu gốc |
| 5 | Tạo bộ video đánh giá vi phạm có nhãn thủ công (vài chục clip ngắn, nhiều điều kiện) + metric mức sự kiện (P/R theo loại vi phạm) | Cao | Trung bình–Cao | Cần video, thời gian gán nhãn |
| 6 | Sửa logic luật: vạch dừng + hướng đi qua vạch (thay vì "đang ở trong polygon"), smoothing trạng thái đèn theo thời gian, thêm màu vàng, cấu hình hướng đi thay vì hard-code `dy<-30` | Cao | Trung bình | #5 để kiểm chứng |
| 7 | Bằng chứng đầy đủ: lưu ảnh toàn cảnh gốc (không overlay), timestamp, loại vi phạm, ID, nhật ký JSON/SQLite; dọn file cũ | Trung bình | Thấp | — |
| 8 | OCR biển số (paddleocr đã có trong venv, hoặc mô hình nhẹ hơn), thêm đánh giá CER/độ chính xác chuỗi | Trung bình–Cao (giá trị portfolio) | Trung bình | Crop biển rõ hơn (hiện ~50 px) |
| 9 | Gộp model: một detector tự train (xe máy/ô tô/người/mũ/đèn) thay COCO-s + helmet-s; plate chỉ chạy khi có sự kiện | Cao (cho Edge) | Cao | Dữ liệu + #4 |
| 10 | Bổ sung dữ liệu VN theo điều kiện (đêm, mưa, camera cao/chéo) và đánh giá theo từng điều kiện | Cao | Cao | Nguồn dữ liệu, thời gian |
| 11 | INT8 PTQ/QAT (hiệu chuẩn bằng dữ liệu thật) và so sánh độ giảm mAP | Trung bình | Trung bình | #3 |
| 12 | Experiment tracking (W&B/MLflow/TensorBoard) + model versioning + `data.yaml` kèm dataset card | Trung bình | Thấp–Trung bình | #1 |
| 13 | Vá bảo mật demo (sanitize tên file, whitelist nguồn, giới hạn kích thước, auth, bỏ `reload=True`), Dockerfile | Trung bình | Thấp | — |
| 14 | Đánh giá tracker (BoT-SORT vs ByteTrack), đo ID switches ở cảnh đông xe máy | Thấp–Trung bình | Trung bình | #5 |
| 15 | Loại bỏ phụ thuộc thừa (paddle nếu không dùng, bản OpenCV trùng); xem xét license thay thế (ví dụ RT-DETR/YOLO license khác) nếu thương mại | Thấp–Trung bình | Thấp–Trung bình | Quyết định mục tiêu |

---

## 6. Câu hỏi cho bạn

1. **`docs/JD.md` không có trong repo** – bạn có thể gửi JD (hoặc dán nội dung) để tôi đối chiếu chính xác?
2. **Code/notebook train** (Kaggle notebook?) và **dữ liệu gốc** còn giữ không? Cần: ảnh + nhãn, `data.yaml`, nguồn/license từng tập, số ảnh/nhãn. Dataset Kaggle `tranquangduyenk17ct/helmet-detection-1` và `vietnam-license-plate-v1` có còn truy cập được không?
3. "Huấn luyện trên nhiều tập dữ liệu" – ngoài 2 tập trên còn tập nào? Có bước gộp class nào đã làm ở nơi khác không?
4. Có **kết quả/ảnh/log** từ Kaggle (results.csv, confusion matrix, PR curve, per-class metrics) không?
5. **Thiết bị Edge mục tiêu** là gì (Jetson Nano/Orin/Xavier, Raspberry Pi, Hailo, Intel NUC/OpenVINO, điện thoại)? Có ngân sách FPS/latency/công suất không?
6. Phần cứng dev hiện tại: RTX 3050 Laptop 4 GB – còn GPU/Kaggle/Colab khác để train không? Giới hạn thời gian/tiền?
7. Video test `test_video.mp4` do bạn quay hay lấy từ nguồn nào (license)? Bạn có thêm video thực tế (đêm, mưa, nhiều góc) và có thể gán nhãn không?
8. Phạm vi vi phạm muốn giữ cho portfolio: chỉ **mũ bảo hiểm + chở quá người + đèn đỏ**, hay cần thêm đi sai làn/ngược chiều/OCR biển số? Ưu tiên chiều sâu hay độ rộng?
9. Có yêu cầu về **license** (dự án mở/portfolio vs thương mại) – có chấp nhận AGPL của Ultralytics không?
10. **Thời hạn** nộp hồ sơ và mức thời gian/tuần bạn dành được?
11. Có cho phép tôi chạy các lệnh đo tốn tài nguyên (FPS baseline, tính FLOPs, thử export ONNX) ở bước sau không?
