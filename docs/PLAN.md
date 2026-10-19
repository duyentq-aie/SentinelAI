# PLAN – Nâng cấp SentinelAI

Ngày: 2026-10-17. Trạng thái: **bản nháp chờ duyệt**. Chưa viết code nào.
Đầu vào: [AUDIT.md](AUDIT.md) và câu trả lời của bạn cho mục 6 của audit.

---

## 0. Giả định (vì một số câu trả lời còn trống)

Lúc viết kế hoạch, tôi kiểm tra repo và thấy: **`docs/JD.md` chưa có**, **không có `data/` hay `notebooks/`**. Ba ô `[SỬA: ...]` trong câu trả lời cũng chưa được điền. Kế hoạch dưới đây dùng các giả định sau. Thay đổi giả định nào thì chỉ cần sửa phần ghi chú tương ứng.

| # | Chưa biết | Giả định đang dùng | Nếu khác thì sao |
|---|---|---|---|
| G1 | Nội dung JD | Yêu cầu CV Engineer điển hình (detection, OpenCV, tracking, edge, đánh giá, tái lập) | Khi có JD, tôi cập nhật mục 3 của AUDIT.md và đổi thứ tự ưu tiên nếu JD nhấn mạnh thứ khác (ví dụ MLOps, C++, DeepStream) |
| G2 | Dữ liệu gốc | **Lấy lại được**: hai dataset nằm dưới tài khoản Kaggle của bạn (`tranquangduyenk17ct/...`), nên tải lại bằng `kaggle datasets download` | Nếu không lấy được: bỏ bước re-split. Mọi số detection chỉ đo trên **field test set** tự gán nhãn (mục 3.2) |
| G3 | Tập dữ liệu khác | Chỉ có 2 tập (helmet, plate) | Nếu có thêm thì thêm bước mapping class và dedup chéo giữa các tập ở Phase 1 |
| G4 | Kết quả Kaggle | Không còn | Nếu còn thì lưu vào `experiments/EXP-000-kaggle-original/` làm tham chiếu lịch sử |
| G5 | Nguồn `test_video.mp4` | Không rõ, coi như **video dev**: đã dùng để chỉnh ROI/ngưỡng nên **không bao giờ vào test** | — |
| G6 | Thời hạn | 3 tuần × ~25 h/tuần ≈ **75 h** tổng thời gian của bạn | Ít hơn thì cắt theo "đường cắt tối thiểu" ở mục 2.9 |
| G7 | Thiết bị triển khai (thêm 2026-10-19) | AI chạy **trên camera**; giả định SoC **Rockchip RK3588** (NPU ≈ 6 TOPS INT8) và mức khắt khe **RV1126** (NPU ≈ 2 TOPS). Hai con số TOPS lấy theo đề bài, chưa đối chiếu datasheet. **Không có phần cứng**: proxy vẫn là RTX 3050 Laptop và CPU ONNX (EXP-007/012). **Mọi số tốc độ trên thiết bị phải là ước tính, kèm nguồn dẫn** (datasheet, bảng benchmark công bố của nhà sản xuất hoặc bài đo của bên thứ ba, ghi rõ model, kích thước đầu vào, độ chính xác số học); không quy đổi FPS từ RTX 3050 sang thiết bị (mục 5). | Có thiết bị thật thì đo thật và thay các ước tính; thiết bị khác thì sửa G7 và các ngưỡng G8 |
| G8 | Yêu cầu tốc độ | Luật **helmet** xử lý **≥ 10 FPS** (tốc độ xử lý, đo theo bước (4) của G10). Đây là yêu cầu của thiết bị đích, **khác** thang đo "tỉ lệ so với v0" của P4 (đo trên máy dev). | Nếu thiết bị chỉ đạt ngưỡng thấp hơn thì hạ khung xử lý (cách frame) hoặc đổi cỡ model, rồi đo lại Δ sự kiện |
| G9 | Số model detector | **Tối đa 1 model detector** trên thiết bị: gộp bằng model **4 lớp** (helmet, motorcycle, no-helmet, plate; đúng 4 lớp của dataset helmet, xem DATA_CARD) thay cho 3 model hiện tại (COCO, helmet, plate). **Hệ quả cần quyết định sau:** model 4 lớp không có ô tô, đèn giao thông; luật `red_light` và cảnh ô tô trong pipeline dùng COCO model, nên với giới hạn 1 detector thì `red_light` nằm ngoài yêu cầu (chỉ helmet/overload) hoặc cần thêm lớp/model; chưa quyết. | Nếu cho phép 2 model thì giữ COCO cho `red_light`; bản 3 lớp dùng khi bỏ lớp plate |
| G10 | Thứ tự ưu tiên khi hướng tới Edge (thay thứ tự cũ của P4/P5 cho phần model) | (1) **Train YOLO11n** 3 lớp và 4 lớp trên **split A** (EXP-013), cùng công thức EXP-015 để so với YOLO11s; (2) **đo mất mát độ chính xác FP32 → FP16 → INT8** (PTQ, hiệu chuẩn bằng **ảnh train**, không dùng val/test) cho **cả n và s**, trên **test split A và tập OOD (EXP-018)**, mAP50 là chỉ số chính (EXP-015); (3) **kiểm tra export sang RKNN**: model có chuyển được không và op nào không được hỗ trợ, bằng RKNN-Toolkit2 nếu cài được trên Windows/WSL; **theo hiểu biết hiện tại RKNN-Toolkit2 chạy trên Linux (WSL là đường duy nhất trên Windows), chưa kiểm lại phiên bản hiện hành**; nếu không cài được thì ghi **"chưa kiểm"**, không suy đoán; (4) **đo pipeline ở 10 FPS xử lý**: chạy pipeline với tốc độ xử lý 10 FPS (video 30 fps, mỗi 3 frame lấy 1), báo Δ sự kiện so với chạy đủ tốc độ. **Lưu ý:** cửa sổ luật windowed (W=15, K=5) và hysteresis đang tính **theo frame** (0.5 s ở 30 fps); ở 10 FPS cùng số frame là 1.5 s, nên cần quy đổi tham số về thời gian trước khi so sánh (chưa làm, chưa tinh chỉnh). | Mỗi bước vẫn theo quy tắc chung: một experiment riêng, số liệu trước/sau; mỗi lần train > 30 phút phải hỏi trước, ưu tiên Kaggle; không train trong bước ghi tài liệu này |
| G11 | TensorRT | **Giữ nguyên ngoài phạm vi** (như quyết định 2026-10-18: không có thiết bị NVIDIA Edge). Backend đích của giả định này là RKNN; ONNX Runtime GPU/CPU chỉ còn là proxy | — |

---

## 1. Cấu trúc repo mục tiêu

```
SentinelAI/
├── README.md                  # mô tả, kết quả chính, cách chạy, license AGPL
├── LICENSE                    # AGPL-3.0 (theo Ultralytics)
├── pyproject.toml             # metadata + deps chính (pinned)
├── requirements.txt           # runtime (CPU/GPU tách bằng extras hoặc file riêng)
├── requirements-lock.txt      # pip freeze đầy đủ của môi trường đã dùng đo
├── .gitignore                 # venv/, data/, models/*.pt|*.onnx|*.engine, outputs/, *.mp4
├── Dockerfile                 # image CPU (ONNX Runtime/OpenVINO)
├── configs/
│   ├── pipeline/default.yaml  # model, backend, ngưỡng conf, cửa sổ vote...
│   ├── cameras/<cam_id>.yaml  # ROI, vạch dừng, hướng đi, ROI đèn, độ phân giải
│   ├── train/<exp>.yaml       # hyperparameters cho từng lần train
│   └── eval/<set>.yaml        # định nghĩa bộ đánh giá
├── sentinel/                  # package lõi, không phụ thuộc web
│   ├── io/                    # nguồn video (file/RTSP/webcam), reader có timestamp
│   ├── detection/             # wrapper model (pt/onnx/engine/openvino qua Ultralytics)
│   ├── tracking/              # cấu hình tracker, gộp track các model
│   ├── lights/                # trạng thái đèn: HSV trong ROI + làm mượt theo thời gian
│   ├── rules/                 # helmet.py, overload.py, red_light.py, base.py (Event)
│   ├── ocr/                   # chọn frame tốt nhất, tiền xử lý, nhận dạng, vote, regex VN
│   ├── evidence/              # lưu frame gốc + crop + metadata (SQLite/JSONL)
│   ├── viz/                   # vẽ overlay (tách khỏi logic)
│   ├── pipeline.py            # frame → (tracks, events); chạy headless được
│   └── utils/                 # seed, timing, env info, config loading
├── apps/web/                  # FastAPI + templates + static (demo, gọi sentinel.pipeline)
├── app.py                     # giữ lại làm entrypoint tương thích (gọi apps/web)
├── tools/                     # CLI: benchmark, eval_det, eval_events, eval_ocr,
│                              #      leakage_check, make_splits, export, label_events
├── tests/                     # pytest: luật với track giả lập, matcher, OCR normalize...
├── data/                      # (gitignored) raw/, processed/ ; chỉ commit:
│   ├── splits/*.csv           #   manifest split (đường dẫn + sha256 + group)
│   └── DATA_CARD.md           #   nguồn, license, số lượng, phân bố class
├── eval_sets/
│   ├── events/                # nhãn sự kiện (commit) ; video gitignored
│   ├── field_frames/          # nhãn YOLO cho frame từ video tự quay (commit nếu nhỏ)
│   └── plates_text/           # crop biển + text GT
├── models/
│   ├── registry.yaml          # tên, version, sha256, exp nguồn, data version, metrics
│   └── MODEL_CARD.md          # (weights để ở GitHub Release / HF, không commit)
├── experiments/
│   ├── README.md              # bảng tổng hợp mọi thí nghiệm
│   └── EXP-###-<slug>/        # config.yaml, metrics.json, env.json, notes.md
├── notebooks/                 # notebook train Kaggle (đã xóa output lớn)
└── docs/                      # AUDIT, PLAN, REPORT, JD, hình ảnh báo cáo
```

**Lý do chính**
- **Tách `sentinel/` (lõi) khỏi `apps/web/` (demo).** Nhờ vậy pipeline chạy headless được để benchmark và đánh giá, unit test được mà không cần web server, còn demo chỉ là một lớp mỏng phía trên.
- **Cấu hình theo camera nằm ở `configs/cameras/`.** ROI và vạch dừng là thuộc tính của camera, không phải của code. Đây là điều kiện để chạy video mới mà không phải sửa code.
- **Không commit dữ liệu và weights vào git**, chỉ commit *manifest + sha256*. Tái lập được mà repo vẫn nhẹ. Weights đặt ở GitHub Release.
- **Ghi thí nghiệm bằng thư mục `experiments/` và JSON thay vì MLflow/W&B.** Không cần server, dễ đọc trên GitHub và đủ cho portfolio. Nếu JD yêu cầu MLflow/W&B thì thêm sau, vì mỗi run đã có sẵn config và metrics nên chuyển sang dễ.
- **Data version = sha256 của file manifest split. Model version = sha256 weights + entry trong `registry.yaml`.**
- **Backend model đổi bằng file weights.** Ultralytics nạp được `.pt/.onnx/.engine/openvino` cùng một API, nên đổi backend chỉ là đổi đường dẫn trong config.

---

## 2. Các phase

Thứ tự phụ thuộc: **P0 → (P1 ∥ P2) → P3 → P4 → (P5) → P6**.

Khác với gợi ý của bạn ở một điểm: **P1 và P2 chạy song song.** P1 tốn nhiều công thủ công của bạn (quay và gán nhãn video), còn P2 là việc code. Làm tuần tự thì phần gán nhãn sẽ thành nút cổ chai. Điều kiện để song song được: P2 chỉ *tinh chỉnh* luật trên **dev videos**, và chỉ chạy **test videos** khi đã khóa cấu hình. Ngoài ra, export detector ở P4 có thể bắt đầu sớm (song song với P3), vì không phụ thuộc OCR.

Ước tính thời gian dưới đây là thời gian làm việc (code + chạy + review). Phần thủ công của bạn tính riêng ở mục 6.

### P0 – Nền móng + baseline (4–6 h)
**Mục tiêu:** đóng băng hiện trạng thành số liệu "trước", để mọi thay đổi sau có mốc so sánh.

Việc:
1. `git init`, `.gitignore`, commit đầu tiên với **code hiện tại chưa sửa** (tag `v0-baseline`). Weights và video không vào git.
2. `requirements-lock.txt` từ venv hiện tại, cộng `env.json` (Python, torch, CUDA, driver, GPU, CPU).
3. `tools/benchmark.py`: import `app.generate_frames` **như hiện tại**, chạy N frame của `test_video.mp4` (bỏ 30 frame warmup). Đo FPS end-to-end và latency mean/p50/p95. Bản thứ hai có hook đo thời gian từng stage (COCO, helmet, plate, vẽ, encode). Chạy trong bản sao thư mục tạm để không ghi rác vào `static/`.
4. FLOPs/params cho 3 model bằng `model.info()` ở imgsz 640.
5. **Characterization snapshot:** chạy pipeline cũ trên `test_video.mp4` và lưu danh sách sự kiện (track id, loại, frame) vào `experiments/EXP-001-baseline/events.json`. Đây là mốc hồi quy cho P2.
6. Khung `README.md` và `experiments/README.md`.

File: mới `.gitignore`, `requirements-lock.txt`, `tools/benchmark.py`, `experiments/EXP-001-baseline/*`. **Không sửa `app.py`.**

Hoàn thành khi:
- Có `EXP-001-baseline/metrics.json` gồm FPS end-to-end, latency p50/p95 từng stage, FLOPs/params 3 model, phiên bản môi trường.
- Chạy lại benchmark 3 lần, FPS lệch < 5%.
- `git log` có commit baseline.

Rủi ro: FPS trên laptop dao động do nhiệt/nguồn → ghi chế độ nguồn (cắm sạc, High Performance) và báo trung vị của 3 lần.

### P1 – Dữ liệu trung thực: leakage, test set độc lập, bộ đánh giá sự kiện (6–10 h code; công thủ công xem mục 6)
**Mục tiêu:** có những tập đánh giá mà số liệu trên đó đáng tin.

Việc:
1. Tải lại hai dataset (G2) vào `data/raw/`. Viết `DATA_CARD.md`: nguồn gốc (Roboflow project gốc nếu tìm được), license, số ảnh/nhãn từng split, phân bố class, phân bố kích thước box.
2. `tools/leakage_check.py`:
   - (a) Trùng tuyệt đối bằng sha256.
   - (b) Gần trùng bằng perceptual hash (pHash/dHash, Hamming ≤ ngưỡng) **giữa** các split.
   - (c) Nhóm theo tên gốc Roboflow: phần stem trước `_jpg.rf.<hash>` cho biết ảnh nào là bản augment của cùng ảnh gốc.
   - (d) Lấy mẫu trực quan 30 cặp nghi trùng để xác nhận.
3. `tools/make_splits.py`: chia lại **theo nhóm** (group = ảnh gốc/cụm gần trùng) thành train/val/test 70/15/15, seed cố định, xuất manifest CSV. Data version = sha256 của manifest.
4. **Đánh giá lại model hiện có** (không train) trên test mới và trên field test set. Đây là bằng chứng định lượng cho việc metric val bị thổi phồng hay không.
   *Lưu ý:* model hiện tại đã thấy một phần ảnh của test mới khi train. Vì vậy số này chỉ để *so sánh*, còn **số "thật" cho baseline là field test set**. Re-split chỉ có ý nghĩa đầy đủ cho các model train lại ở P5.
5. **Field test set detection:** frame lấy thưa (khoảng 1 frame/2 s) từ video bạn tự quay, chỉ lấy từ **video test**, gán nhãn 3 class helmet + class plate. Mục tiêu khoảng 300–500 frame, có cả ngày và đêm.
6. **Bộ video đánh giá sự kiện:** định dạng và quy trình ở mục 3. Chia video thành **dev** (được tinh chỉnh) và **test** (khóa, chỉ chạy khi báo cáo), chia theo *buổi quay/vị trí*, không chia theo clip cắt từ cùng một đoạn.
7. `tools/label_events.py`: công cụ OpenCV đơn giản (phát/tua video, phím tắt đánh dấu bắt đầu/kết thúc sự kiện, kéo box key-frame, nhập loại và biển số). Mục đích là giảm công gán nhãn của bạn.

File: `tools/leakage_check.py`, `tools/make_splits.py`, `tools/label_events.py`, `data/splits/*.csv`, `data/DATA_CARD.md`, `eval_sets/**`.

Hoàn thành khi:
- Báo cáo leakage có số cụ thể: % ảnh val/test có bản gần trùng trong train, trước và sau khi re-split; sau re-split là 0 cặp trùng chéo split ở ngưỡng đã chọn.
- Field test set ≥ 300 frame và ≥ 150 instance no-helmet.
- Bộ sự kiện test có, cho mỗi loại vi phạm, **≥ 20 sự kiện GT** (mục tiêu 30+) và ≥ 30 phút video tổng (để đo báo nhầm), trong đó ≥ 25% là ban đêm.

Rủi ro:
- Vượt đèn đỏ hiếm, có thể quay 30 phút chỉ được vài sự kiện. Cách giảm: chọn ngã tư có nhiều xe máy vượt đèn, quay giờ cao điểm. Nếu vẫn < 20 thì báo cáo đúng số lượng kèm khoảng tin cậy rộng, không suy rộng.
- Quay đường công cộng: biển số và mặt người là dữ liệu cá nhân. Video giữ nội bộ, media công bố phải làm mờ (xem P6).
- Nếu không lấy lại được dữ liệu (G2): bỏ bước 2–4, field test set thành nguồn số liệu detection duy nhất.

### P2 – Refactor pipeline + sửa luật + bằng chứng đầy đủ + test (12–16 h)
**Mục tiêu:** pipeline có module rõ ràng, luật vi phạm đúng nghĩa, có test, demo vẫn chạy.

Việc, theo thứ tự (mỗi bước một commit và demo vẫn chạy):
1. **Refactor không đổi hành vi:** tách code sang `sentinel/`, `app.py` gọi lại pipeline. Kiểm bằng **characterization test**: sự kiện sinh ra trên `test_video.mp4` phải trùng snapshot của P0 (cùng loại, cùng frame ± 0; track id có thể khác nhưng số lượng và thời điểm phải khớp).
2. **Config** *(xong 2026-10-17)*: ngưỡng, polygon, đường dẫn model và **tham số tracker** (`configs/trackers/{botsort,bytetrack}.yaml`, mỗi model một đường dẫn tracker) nằm trong `configs/pipeline/default.yaml` và `configs/cameras/test_video.yaml`; bộ nạp `sentinel/config.py` từ chối key lạ và giá trị sai. Nhờ vậy P4 chỉ cần đổi config để thử `gmc_method: none` hoặc ByteTrack.
3. **Sửa lỗi đã thấy trong audit** *(trạng thái/vòng lặp/worker xong 2026-10-17; vá bảo mật còn lại)*:
   - reset trạng thái per-track và tracker khi đổi nguồn hoặc khi video lặp;
   - dùng cả detection chưa có track id cho việc vẽ;
   - bỏ busy-loop;
   - chạy **một worker suy luận duy nhất** phát frame cho nhiều client;
   - vá bảo mật: sanitize tên file, giới hạn kích thước, whitelist nguồn, không `reload=True`, mặc định bind `127.0.0.1`.
4. **Luật mới:**
   - **Không mũ:** gắn đầu với xe máy theo hình học (tâm đầu nằm trong vùng *nửa trên mở rộng* của box xe, chọn xe gần nhất nếu chồng lấn). Bỏ bộ đếm cũ, thay bằng vote theo cửa sổ: trong W frame gần nhất có ≥ K quan sát và tỉ lệ no-helmet ≥ r (W, K, r trong config). Mỗi track sinh **tối đa 1 sự kiện**.
   - **Chở quá người:** số đầu gắn với xe lấy *trung vị* theo cửa sổ, > 2 trong ≥ K frame.
   - **Vượt đèn đỏ:**
     - Trạng thái đèn: lấy từ **ROI đèn cố định** trong config camera (camera giao thông cố định nên vị trí đèn cố định); dùng COCO detector làm phương án dự phòng khi chưa có ROI. HSV đỏ/vàng/xanh, rồi **làm mượt theo thời gian** (đa số trong k frame, có hysteresis). Trạng thái `UNKNOWN` thì không phạt.
     - Vạch dừng: đoạn thẳng 2 điểm, kèm vector hướng đi hợp lệ.
     - Một sự kiện vi phạm xảy ra khi điểm neo (giữa đáy box) **cắt vạch** giữa hai frame liên tiếp (đổi dấu tích có hướng), **cùng chiều** hướng hợp lệ, có hysteresis chống rung (đã ở trước vạch ≥ d px trong ≥ m frame). Tại thời điểm cắt, đèn phải đang ĐỎ và **đã đỏ ≥ t_grace giây**.
     - Áp dụng cho **cả xe máy**. Xe đứng chờ trong vùng không còn bị phạt.
5. **Bằng chứng** *(xong 2026-10-18, 3.1; xem docs/EVIDENCE.md, EXP-003)*: copy frame **gốc** trước khi vẽ. Mỗi sự kiện lưu: ảnh toàn cảnh, crop xe, crop biển (nếu có), JSON metadata (event_id, loại, track_id, frame, timestamp, camera_id, trạng thái đèn, confidence, version model/config) vào SQLite/JSONL. Sự kiện **vẫn được ghi khi không thấy biển** (plate = null). Plate detector **chỉ chạy cho track đã vi phạm** và chỉ trong một số frame giới hạn, không chạy mọi frame như hiện tại.
6. **Đầu ra dự đoán** có cùng schema với nhãn GT (mục 3), để `tools/eval_events.py` dùng trực tiếp.
7. **Tests (pytest):** track giả lập cho cắt vạch (đúng chiều, ngược chiều, rung quanh vạch, đứng yên trong vùng), đèn (đỏ → vàng → xanh, nhiễu 1–2 frame), vote helmet, overload, matcher của evaluator, chống trùng sự kiện. Thêm 1 smoke test end-to-end trên 60 frame (đánh dấu `slow`). GitHub Actions chạy unit test trên CPU, không cần weights.
8. Tinh chỉnh tham số luật **chỉ trên dev videos**. Ghi `EXP-0xx-rules-v2` với P/R/F1 dev trước và sau.

File: `sentinel/**`, `apps/web/**`, `app.py` (thành wrapper mỏng), `configs/**`, `tests/**`, `.github/workflows/test.yml`.

Hoàn thành khi:
- Characterization test pass ở bước 1 (cùng CPU: giống hệt bản gốc kể cả ID; CPU vs GPU: ±3 frame, bỏ qua ID).
- ≥ 20 unit test cho luật và matcher đều pass.
- Demo chạy trên `test_video.mp4` và webcam.
- Có bảng P/R/F1 sự kiện **trên dev** cho luật cũ và luật mới.
- Chạy luật cũ và mới trên **test videos đúng 1 lần** sau khi khóa config, lưu kết quả.

Rủi ro:
- Luật mới có thể *giảm* recall red-light (luật cũ phạt mọi xe trong vùng). Đó là chủ ý; báo cáo trung thực cả P lẫn R.
- Hai model cùng phát hiện xe máy (COCO và helmet) có thể sinh track trùng → dùng **xe máy của helmet model** làm nguồn chính, lọc xe máy của COCO.

### P3 – OCR biển số + đánh giá (6–10 h)
> **Quyết định 2026-10-18:** OCR **giữ trong phạm vi dự án, với điều kiện có footage 1080p** (biển cao ≥ ~40–50 px; xem EXP-008: trên video 720p hiện có hộp biển do detector trả về chỉ ~26×22 px nên đọc được 0/14).
> **Quyết định 2026-10-18 (vòng 2):** (1) **TensorRT / INT8 nằm ngoài phạm vi M1** (không có thiết bị Edge thật; ONNX CPU/GPU FP32/FP16 làm proxy, EXP-007/012). (2) Hysteresis `exit_ratio: 0.3` **bật mặc định**, gắn "chưa tinh chỉnh – chờ dev video" (EXP-005 phụ lục 2; baseline v5). (3) torch cu126 nhanh hơn cu118 ~25% với cùng mã (EXP-012: 16.94 → 20.93 FPS stream, cùng phiên; chưa kiểm tra nguyên nhân/độ lặp lại, **chưa đổi venv chính**). (4) Trước khi push: viết lại email tác giả trong lịch sử thành `<USER>@users.noreply.github.com`, thay đường dẫn home trong env.json bằng `<HOME>`.
> Nếu không quay được footage đủ độ phân giải, OCR bị đưa ra khỏi số liệu báo cáo (code vẫn giữ). Việc đo exact-match / CER chỉ bắt đầu khi có video 1080p và nhãn `plate_text`.
**Mục tiêu:** đọc được biển số cho sự kiện vi phạm và đo được độ chính xác.

Việc:
1. **Chọn frame tốt nhất** trong track: crop biển lớn nhất, độ nét cao nhất (phương sai Laplacian), đủ sáng.
2. **Tiền xử lý OpenCV:** nắn phẳng phối cảnh (`getPerspectiveTransform` từ 4 góc nếu tìm được contour, không thì giữ nguyên box), upscale, CLAHE cho ảnh đêm, tách 2 dòng cho biển xe máy.
3. **Baseline OCR:** PaddleOCR (đã có trong venv) chỉ dùng phần *nhận dạng* trên crop. Ghi lại phiên bản model.
4. **Hậu xử lý:**
   - chuẩn hóa (bỏ `-`, `.`, khoảng trắng; viết hoa);
   - sửa ký tự dễ nhầm theo vị trí (0/O, 8/B, 1/I...);
   - kiểm tra định dạng biển VN bằng regex, ví dụ 2 số tỉnh + seri chữ (± số) + 4–5 số;
   - **vote nhiều frame** trong track (vote theo từng ký tự, có trọng số confidence).
5. `tools/eval_ocr.py`: đánh giá trên (a) **crop biển GT** (đo riêng OCR) và (b) **crop từ detector của pipeline** (đo end-to-end).
6. **Đánh đổi cho Edge:** Paddle nặng. RapidOCR (model PaddleOCR đã chuyển sang ONNX, chạy bằng ONNX Runtime) giúp bỏ hẳn phụ thuộc `paddlepaddle`, nhưng **hoãn tới sau mốc duyệt P3 (D5)**.

Hoàn thành khi:
- Có bảng exact-match accuracy và CER trên ≥ 150 biển GT, tách ngày/đêm và 1 dòng/2 dòng, cho từng cấu hình: raw / + tiền xử lý / + regex / + vote.
- Có latency OCR mỗi sự kiện.

Rủi ro:
- Crop biển ở 720p chỉ khoảng 50 px (đã thấy trong audit), có thể quá nhỏ để đọc. Giảm bằng chọn frame tốt nhất và quay video 1080p. Nếu vẫn kém thì báo cáo đúng giới hạn độ phân giải và không fine-tune OCR (xem mục 5).

### P4 – Tối ưu Edge + benchmark (10–16 h)
**Mục tiêu:** pipeline đầy đủ ≥ 15 FPS ở 720p trên proxy GPU, có bảng đánh đổi độ chính xác/tốc độ trên proxy CPU.
> **Cập nhật 2026-10-19:** hướng triển khai là camera Rockchip (RK3588 / RV1126), ≥ 10 FPS cho luật helmet với tối đa 1 detector; thứ tự việc về model được xếp lại ở giả định **G7–G11** (mục 0). Phần dưới đây là bản ghi cho máy dev và proxy.
Cập nhật 2026-10-17 theo profiling `EXP-001/baseline-v2`.

**Căn cứ đo được (baseline-v2, 135.6 ms/frame, 7.4 FPS, High performance):**
GPU inference chỉ ≈ 46 ms (34%) và GPU rảnh ≈ 83% thời gian. Phần CPU ≈ 54%: tracker BoT-SORT ≈ 40 ms (COCO 21.0 + helmet 19.0, **lớn hơn tổng inference của hai model đó**),
vẽ ≈ 15 ms, JPEG encode ≈ 14.6 ms, đọc + resize ≈ 3 ms; plate ≈ 28.8 ms/frame (chạy lại trên mọi xe vi phạm ở mọi frame).
**Chỉ tăng tốc model (TensorRT/INT8) giảm tối đa ≈ 46 ms → còn ≈ 90 ms/frame ≈ 11 FPS, không đạt mục tiêu 15 FPS.** Vì vậy thứ tự tối ưu là CPU trước, model sau:

| Thứ tự | Thay đổi | Kỳ vọng (từ số đo) | Làm ở |
|---|---|---|---|
| 1 | **Plate chạy theo sự kiện, 1 lần/track** (không chạy lại mỗi frame) | ≤ −28 ms/frame | P2 |
| 2 | **Tracker**: đo BoT-SORT với `gmc_method: none` (camera cố định) và ByteTrack | ≤ −(một phần của ~40 ms) | P4 (chỉ đổi `configs/trackers/*.yaml`, tham số đã có trong config từ P2 bước 2) |
| 3 | **Vẽ**: tránh `copy` + `addWeighted` toàn khung hình mỗi box (vẽ trực tiếp, blend chỉ trên vùng nhãn) | ≤ −15 ms | P4 |
| 4 | **Encode JPEG chỉ khi có client**; luôn báo **FPS headless** và **FPS có stream** riêng | ≤ −14.6 ms headless | P2 (worker) + P4 (đo) |
| 5 | **ONNX Runtime GPU / TensorRT FP16** (TensorRT time-box 2 h, rồi ORT-GPU) | ≤ −(một phần của ~46 ms) | P4 |

**Quy tắc cho từng thay đổi:**
- Mỗi thay đổi là **một experiment riêng** (`EXP-0xx-<slug>`), có số **trước/sau** trên cùng giao thức benchmark (3.5), cùng `--detail`.
- **Thay đổi tracker không được chấp nhận chỉ vì FPS.** Đổi tracker/GMC làm thay đổi gán ID (ID switch), mà luật vi phạm dựa vào track ID (vote helmet, cắt vạch, chống trùng sự kiện).
  Phải kiểm bằng **P/R/F1 sự kiện trên dev videos** (khi có, P1) và so với cấu hình tracker cũ; trong lúc chưa có dev video chỉ được ghi "tạm thời, chưa chấp nhận" và dùng snapshot sự kiện làm kiểm tra phụ (đếm sự kiện, ±3 frame, bỏ qua ID).
- Mọi bước tối ưu có thể đổi kết quả (tracker, giảm imgsz, bỏ frame, INT8) phải báo Δ F1 sự kiện trên **dev** (không dùng test videos để chọn cấu hình).

**Việc còn lại của P4:**
1. `tools/export.py`: export từng model sang ONNX (opset cố định, imgsz 640, static shape, simplify), TensorRT FP16 (`.engine`), OpenVINO FP32 và **INT8** (PTQ bằng NNCF, ảnh hiệu chuẩn từ **train split**, không bao giờ từ test); tùy chọn TensorRT INT8.
2. **Độ chính xác sau export:** mAP trên test/field set cho từng backend/precision, cùng P/R sự kiện trên dev videos.
3. Các tối ưu pipeline khác (đo riêng từng cái): giảm imgsz COCO (640 → 512/480), lọc class COCO ngay trong predictor (`classes=[2,5,7,9]`), tách luồng đọc frame và suy luận, tùy chọn chạy detector cách frame (đo ảnh hưởng tới recall sự kiện trước khi chấp nhận).
4. Benchmark theo giao thức cố định (3.5) cho mọi tổ hợp {backend} × {precision}; mỗi tổ hợp là một experiment.

**Hoàn thành khi:**
- Có bảng: bước tối ưu / backend / precision → kích thước file, latency p50/p95 từng stage, **FPS headless** và **FPS có stream**, Δ mAP, Δ F1 sự kiện (dev).
- **(sửa 2026-10-18)** Báo tốc độ chỉ dưới dạng **tỉ lệ so với v0 đo cùng phiên** (EXP-009: FPS tuyệt đối đổi ~40% theo tải nền CPU, tỉ lệ đổi ≤ 2%); không dùng ngưỡng FPS tuyệt đối. Con số "≥ 15 FPS" trong đề bài ban đầu chỉ được trích kèm trạng thái máy khi đo. Với backend ONNX/TensorRT: Δ F1 sự kiện ≤ 2 điểm so với PyTorch FP32 cùng cấu hình tracker.
- Có cấu hình CPU khuyến nghị kèm FPS đo được.

Rủi ro:
- Cài TensorRT trên Windows cần đúng tổ hợp CUDA/cuDNN với torch cu118. **Time-box 2 h (D4)**; quá thời gian thì dùng ONNX Runtime GPU (CUDA EP, FP16).
- Giả thuyết "phần lớn chi phí tracker là GMC/optical flow" **chưa kiểm chứng**; nếu `gmc_method: none` không giảm đáng kể thì nguyên nhân khác (ví dụ chuyển tensor GPU→CPU, `.cpu().numpy()`, `Results` re-index) và cần profiling sâu hơn.
- Báo cáo **phải ghi rõ** đây là proxy: RTX 3050 không phải Jetson; **không quy đổi** FPS sang Jetson.

### P5 – Cải thiện model dựa trên phân tích lỗi (tùy chọn, 8–15 h + thời gian train)
Chỉ làm nếu P0–P4 và phần lõi P6 đã xong. Hướng cụ thể sẽ do **phân tích lỗi** ở P1–P4 quyết định; dưới đây là các ứng viên có khả năng cao nhất:
- **Train lại trên split theo nhóm** (P1). Đây là điều kiện để test split sạch có ý nghĩa đầy đủ. Cùng hyperparameters cũ để so sánh công bằng.
- **Thêm dữ liệu miền mục tiêu:** frame từ **dev videos** (không bao giờ dùng test videos). Gán nhãn có hỗ trợ của model, nhưng phải duyệt kỹ, vì dùng model gợi ý nhãn dễ khiến người duyệt bỏ qua lỗi của chính model.
- **Vật thể nhỏ (đầu ở xa):** imgsz 832 hoặc 960 khi train, đánh đổi tốc độ, đo lại ở P4.
- **Detector hợp nhất** (xe máy, ô tô, bus/truck, đầu có mũ, đầu không mũ, đèn): bỏ hẳn COCO model, từ 3 model còn 2. Lợi ích lớn nhất cho Edge, nhưng tốn công gán nhãn bổ sung (pseudo-label bằng COCO model + duyệt).

Mỗi lần train > 30 phút: **hỏi trước**. Ưu tiên train trên Kaggle.
Hoàn thành khi: mỗi thay đổi có experiment riêng, có Δ trên **field test set và test videos** (không chỉ val).

### P6 – Báo cáo kỹ thuật + README + Docker + demo (6–8 h)
Việc:
- Viết `docs/REPORT.md` (mục 4).
- README: kết quả chính (một bảng), kiến trúc, quick start, tái lập từng bảng bằng lệnh nào, license AGPL, giới hạn.
- `Dockerfile` CPU (ONNX Runtime/OpenVINO, không cần GPU) với `docker run` chạy demo trên video mẫu. Image GPU chỉ là tùy chọn.
- Video demo ngắn/GIF: làm mờ biển số và mặt của người không liên quan (OpenCV blur từ chính detector).
- `models/registry.yaml` + GitHub Release cho weights; xóa đường dẫn Kaggle cá nhân khỏi metadata checkpoint khi phát hành (nếu bạn muốn).

Hoàn thành khi:
- Trên máy sạch: `git clone` + tải weights + `docker run` (hoặc tạo venv) rồi chạy được demo.
- Mọi con số trong README/REPORT trỏ tới một experiment ID.

### 2.9 Tổng thời gian & đường cắt tối thiểu
| Phase | Ước tính (h) |
|---|---|
| P0 | 4–6 |
| P1 (code) | 6–10 |
| P2 | 12–16 |
| P3 | 6–10 |
| P4 | 8–12 |
| P6 | 6–8 |
| **Lõi (không P5)** | **42–62** |
| P5 (tùy chọn) | 8–15 + train |
| Thủ công của bạn (mục 6) | 20–32 |

Với ~75 h (G6), phần lõi cộng phần thủ công đã sát giới hạn, nên **P5 nhiều khả năng không kịp**. Nếu thiếu thời gian, đường cắt tối thiểu vẫn có câu chuyện portfolio trọn vẹn là: **P0 → P1 (chỉ bộ sự kiện + field set nhỏ) → P2 → P4 (chỉ TensorRT FP16 + ONNX CPU) → P6**, còn P3 rút gọn (PaddleOCR + regex + vote, không RapidOCR).

---

## 3. Thiết kế bộ đánh giá

### 3.1 Nhãn sự kiện vi phạm (ground truth)
Mỗi video có một file `eval_sets/events/<video_id>.json`:
```json
{
  "video_id": "hn_ngatu_A_2026-10-20_night_01",
  "file_sha256": "…",
  "fps": 30, "width": 1920, "height": 1080,
  "split": "test",                 // dev | test (theo buổi quay)
  "camera_config": "configs/cameras/hn_ngatu_A.yaml",
  "conditions": {"time": "night", "weather": "dry", "density": "high"},
  "annotated_ranges": [[0, 27000]],     // frame được gán nhãn đầy đủ (để đếm FP)
  "events": [
    {
      "id": "e001",
      "type": "no_helmet",          // no_helmet | overload | red_light
      "frame_start": 1200, "frame_end": 1380,   // khoảng nhìn thấy vi phạm rõ ràng
      "key_frame": 1290,
      "key_box": [812, 455, 930, 640],          // box XE tại key_frame
      "plate_text": "29B112345",                // null nếu không đọc được bằng mắt
      "difficulty": "normal",                   // normal | hard (che khuất, xa, mờ)
      "notes": ""
    }
  ],
  "ignore_regions": [[[0,0],[300,0],[300,200],[0,200]]]  // vùng không chấm (tùy chọn)
}
```
Quy tắc gán nhãn sẽ có trong `eval_sets/events/GUIDELINES.md`. Các điểm chính:
- Red-light chỉ tính khi xe cắt vạch dừng lúc đèn đã đỏ; xe đã qua vạch từ lúc đèn vàng thì không tính.
- No-helmet tính khi **bất kỳ** người nào trên xe không đội mũ.
- Overload tính khi > 2 người trên xe. Trẻ em vẫn tính, nhưng ghi `notes`, vì luật VN có ngoại lệ; báo cáo sẽ nêu.
- Sự kiện mà người gán nhãn không chắc thì đánh `difficulty: hard`. Báo cáo cho hai số: có và không có các sự kiện hard.

Dự đoán của pipeline có cùng schema (thêm `track_id`, `trigger_frame`, `confidence`, `plate_pred`, `boxes` theo frame).

### 3.2 Field test set detection
Frame YOLO-format lấy từ **test videos** (lấy thưa để giảm tương quan), 4 class: helmet, no-helmet, motorcycle, plate. Có metadata điều kiện (ngày/đêm) để báo cáo theo nhóm.

### 3.3 Khớp dự đoán với GT
Làm riêng cho từng video và từng loại vi phạm:
1. Một dự đoán `p` là **ứng viên** cho GT `g` khi cả hai điều kiện đúng:
   - **Thời gian:** `trigger_frame(p)` ∈ `[frame_start(g) − τ, frame_end(g) + τ]`, với τ = 1 s.
   - **Không gian:** box của track `p` tại frame gần `key_frame(g)` nhất có IoU ≥ 0.3 với `key_box(g)`, hoặc chứa tâm của `key_box`.
   Nếu track không tồn tại quanh `key_frame`, dùng box tại `trigger_frame` so với `key_box`, nới IoU xuống 0.1 (ghi rõ trong báo cáo).
2. Gán **một-một** bằng Hungarian trên chi phí `1 − IoU`.
3. **TP** = cặp đã khớp. **FN** = GT không khớp. **FP** = dự đoán không khớp *nằm trong `annotated_ranges`*.
   - Dự đoán khớp với GT đã được khớp (ví dụ một xe bị đổi ID rồi báo hai lần) **tính là FP**, và được thống kê riêng thành "duplicate rate".
   - Dự đoán nằm trong `ignore_regions` không được chấm.

### 3.4 Metric
- **Sự kiện:** P, R, F1 theo từng loại vi phạm; tách ngày/đêm và mật độ; có/không có sự kiện `hard`. Thêm **FP/giờ video**, vì đây là số người vận hành quan tâm, và **độ trễ phát hiện** (trigger_frame − frame_start, tính bằng giây).
  **Khoảng tin cậy 95%** bằng bootstrap *theo video*. Số sự kiện nhỏ nên phải có CI, không báo một con số trần.
- **Detection:** P/R/AP50/AP50-95 **theo class**, confusion matrix, AP theo kích thước box (nhỏ/vừa/lớn). Chạy trên (i) val gốc (tham chiếu lịch sử), (ii) test sau re-split, (iii) field test. Ba cột đặt cạnh nhau để thấy chênh lệch.
- **OCR:** exact-match accuracy (cả biển, sau chuẩn hóa), CER = Levenshtein / độ dài GT, tách theo 1/2 dòng và ngày/đêm. Đo cả trên crop GT lẫn end-to-end (biển được đọc đúng / tổng sự kiện có `plate_text`).
- **Tốc độ:** xem 3.5.

### 3.5 Giao thức benchmark
- Video cố định (`test_video.mp4` 720p, cùng một clip 1080p tự quay), N = toàn bộ frame, bỏ 30 frame warmup, đọc trước video vào RAM để loại trừ I/O đĩa (đo riêng một bản *có* I/O).
- Đo: latency p50/p95 từng stage (`time.perf_counter`, có `torch.cuda.synchronize` khi dùng GPU), FPS end-to-end có và không có vẽ + JPEG encode, VRAM/RAM đỉnh.
- Lặp 3 lần, báo trung vị. Ghi `env.json` (GPU, driver, CUDA/TensorRT, CPU, chế độ nguồn, phiên bản package).
- Batch = 1 (đúng với streaming). Không dùng số "throughput batch lớn".

---

## 4. Cấu trúc `docs/REPORT.md`

1. **Tóm tắt:** bài toán, kết quả chính (một bảng: F1 sự kiện từng loại, CER OCR, FPS proxy GPU/CPU), giới hạn chính.
2. **Bài toán & phạm vi:** 3 loại vi phạm và OCR; định nghĩa chính xác từng vi phạm; những gì nằm ngoài phạm vi.
3. **Kiến trúc hệ thống:** sơ đồ (frame → detect → track → light state → rules → evidence → OCR); lý do chọn thiết kế.
4. **Dữ liệu:** data card; **phân tích leakage** (số trước/sau); split theo nhóm; field set và bộ sự kiện (thời lượng, điều kiện, số sự kiện), quy trình gán nhãn.
5. **Mô hình & huấn luyện:** cấu hình, phần cứng, thời gian; bảng experiment.
6. **Giao thức đánh giá:** mục 3 rút gọn.
7. **Kết quả**
   - 7.1 Detection: val gốc vs test sạch vs field test (theo class, theo kích thước).
   - 7.2 Sự kiện: luật cũ vs luật mới trên test videos, kèm CI, tách ngày/đêm.
   - 7.3 OCR: ablation tiền xử lý / regex / vote.
   - 7.4 Edge: bảng backend × precision (tốc độ, kích thước, Δ chất lượng); phân rã latency theo stage.
8. **Ablation:** làm mượt đèn, điều kiện cắt vạch, vote helmet, chạy plate theo sự kiện.
9. **Phân tích lỗi:** phân loại FP/FN theo nguyên nhân (che khuất, ban đêm, đổi ID, đèn sai, biển quá nhỏ), có ảnh ví dụ đã làm mờ.
10. **Giới hạn & rủi ro:** proxy hardware, số sự kiện nhỏ, một số vị trí quay, bias dữ liệu, không thay thế quyết định của con người.
11. **Đạo đức, quyền riêng tư, license:** làm mờ, lưu trữ, AGPL-3.0, license dataset.
12. **Tái lập:** bảng "con số → experiment ID → lệnh chạy".
13. **Hướng phát triển.**

---

## 5. Những gì sẽ KHÔNG làm

| Không làm | Lý do |
|---|---|
| Báo cáo ngược chiều, sai làn | Đã loại khỏi phạm vi. Code giữ lại dạng tùy chọn tắt mặc định, không test, không báo cáo. |
| Đổi sang kiến trúc khác / so nhiều phiên bản YOLO | Tốn thời gian train, giá trị portfolio thấp hơn đánh giá trung thực và tối ưu Edge. |
| Tối ưu mAP trên val | Trái nguyên tắc; val gốc có thể đã bị leak. |
| DeepSORT/ReID, tự viết tracker | BoT-SORT/ByteTrack đủ dùng. Chỉ đổi tracker nếu phân tích lỗi cho thấy đổi ID là nguyên nhân chính. |
| Fine-tune OCR / train recognizer riêng | Cần nhiều biển có nhãn text; trước hết đo xem baseline + hậu xử lý được bao nhiêu. |
| QAT (quantization-aware training) | PTQ INT8 đủ để có câu chuyện Edge; QAT tốn công train. |
| Pipeline thuần ONNX Runtime (tự viết pre/post-process + tracker, bỏ Ultralytics/torch) | Đúng hướng cho Edge thật, nhưng là nhiều việc. Chỉ ghi vào hướng phát triển. |
| Quy đổi FPS sang Jetson/RPi | Không có thiết bị thật; quy đổi là bịa số. |
| DVC, MLflow server, Kubernetes, cloud deploy | Quá nặng cho portfolio cá nhân; manifest + sha256 + thư mục experiments là đủ. Sẽ xem lại khi có JD. |
| Viết lại frontend (React...) | Không liên quan JD CV; giữ template hiện tại, chỉ sửa cho đúng. |
| Đa camera, cơ sở dữ liệu vi phạm trung tâm | Ngoài phạm vi. |
| Đổi license / bỏ Ultralytics | Bạn đã chấp nhận AGPL. |

---

## 6. Việc thủ công của bạn

| Việc | Khi nào cần | Ước tính |
|---|---|---|
| Điền các ô `[SỬA]` và thêm `docs/JD.md` | Trước P0/P1 | 0.5 h |
| Tải lại 2 dataset Kaggle (hoặc cấp `kaggle.json` để tôi chạy lệnh tải) | P1 | 0.5 h |
| **Quay video**: ≥ 2 vị trí (có ít nhất 1 ngã tư thấy rõ đèn và vạch dừng), ngày + đêm, 1080p, chân máy cố định, tổng ≥ 60 phút (khoảng 30 dev + 30 test) | Bắt đầu **càng sớm càng tốt** (đường găng) | 4–6 h (gồm di chuyển) |
| Vẽ cấu hình camera (vạch dừng, hướng, ROI đèn) cho mỗi vị trí, bằng công cụ click tôi sẽ viết | P2 | 0.5 h/vị trí |
| **Gán nhãn sự kiện** bằng `tools/label_events.py` (khoảng 1.5–2× thời lượng video) | P1, song song P2 | 8–12 h |
| Gán nhãn field test set (~300–500 frame; CVAT/Label Studio, có nhãn gợi ý từ model, phải duyệt kỹ) | P1 | 5–8 h |
| Gõ text biển số GT cho ≥ 150 crop | P3 | 1.5–2.5 h |
| Duyệt kết quả và quyết định ở cuối mỗi phase; cho phép cài TensorRT/RapidOCR; cho phép train dài | Cuối mỗi phase | 2–3 h |
| **Tổng** | | **≈ 22–33 h** |

Lưu ý về quay video:
- Kiểm tra quy định địa phương về quay nơi công cộng.
- Không công bố video gốc. Media công bố phải làm mờ biển số và mặt người.
- Nếu quay được cảnh dàn dựng với người quen (xe của bạn, không mũ trong khu vực an toàn) thì đó là nguồn biển số có thể công bố.

---

## 7. Quyết định đã duyệt (2026-10-17)

| # | Quyết định |
|---|---|
| D1 | **P1 ∥ P2**: đồng ý. Test videos bị khóa cho tới khi cấu hình đã chốt. |
| D2 | Ghi thí nghiệm bằng `experiments/` + JSON, **không dùng MLflow/W&B**. JD chỉ yêu cầu "ghi lại thí nghiệm, phiên bản model và dữ liệu". |
| D3 | Được cài `kaggle` CLI, OpenVINO/NNCF, ONNX Runtime GPU, imagehash, pytest khi tới bước cần. |
| D4 | **TensorRT time-box tối đa 2 h** (gồm cài đặt và export lần đầu). Quá 2 h thì chuyển sang ONNX Runtime GPU (CUDA EP, FP16) làm proxy GPU, ghi lại lý do vào experiment. |
| D5 | **RapidOCR hoãn** tới sau mốc duyệt P3. P3 dùng PaddleOCR đã có sẵn. |

**Còn mở, cần biết sớm:**
- G2: dữ liệu Kaggle còn tải được không. Kiểm tra ngày 2026-10-17: máy chưa có `kaggle` CLI, chưa có `~/.kaggle/kaggle.json`, chưa có biến `KAGGLE_USERNAME`. Cần bạn tải dataset về hoặc cấp token.
- G6: thời hạn.
- G1: `docs/JD.md` vẫn chưa có trong repo, nên mục 3 của AUDIT.md chưa cập nhật được.
