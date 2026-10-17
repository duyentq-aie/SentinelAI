# EXP-007 – ONNX (đêm 2026-10-18, mục 5)

## Môi trường và giới hạn của máy này
- Xuất: `tools/export_onnx.py` (Ultralytics 8.4.26, opset 17, `simplify=True`, ONNX 1.23.2, onnxslim 0.1.98). Hash, kích thước, input trong `models/onnx/export.json`; file `.onnx` bị `.gitignore`.
- **GPU provider không dùng được.** `onnxruntime-gpu 1.31.0` (bản duy nhất cài được theo quy tắc) cần CUDA 12; máy có CUDA 11.8 (bản của torch `+cu118`). `CUDAExecutionProvider` được liệt kê nhưng không nạp được thư viện → Ultralytics lỗi khi gắn tensor GPU (`no data transfer registered ...`). Theo quy tắc ("lỗi thì onnxruntime CPU") đã gỡ `onnxruntime-gpu` và dùng **`onnxruntime` 1.31.0 (CPU)**. Không cài TensorRT, không cài CUDA 12.
  Hệ quả: backend ONNX trong báo cáo này là **proxy CPU cho thiết bị Edge**, không phải proxy GPU. Thử `onnxruntime-gpu` bản CUDA 11.8 (≤ 1.18) bị loại vì không nằm trong yêu cầu và ABI numpy 2 chưa được kiểm chứng.
- (Ghi chú) Khi export, Ultralytics tự thử `pip install onnxruntime-gpu`; lệnh đó lỗi (file DLL đang bị khoá), không làm hỏng môi trường (đã kiểm tra: `onnxruntime` 1.31.0 CPU vẫn chạy, tập `onnx*` đúng như dự kiến).

## Hình dạng đầu vào quan trọng hơn precision
Export "static 640" vuông (`--imgsz 640`) cho kết quả **khác** PyTorch: PyTorch luôn suy luận dạng chữ nhật cho frame 1280×720 (đầu vào 384×640), ONNX tĩnh 640×640 phải thêm lề vuông → thêm ~67% điểm ảnh, thêm detection ở rìa, chậm hơn. Vì vậy đã xuất thêm bản **tĩnh 384×640** (đúng hình dạng PyTorch dùng) cho COCO và helmet, và bản **dynamic** cho plate (plate chạy trên crop xe có kích thước bất kỳ).

So sánh mức model, CPU, 98 frame (mỗi frame thứ 8 của `test_video.mp4`), `tools/compare_backends.py` — **đây là độ khớp giữa hai backend, không phải độ chính xác** (không có ground truth; `data/raw/` không tồn tại nên không đo được mAP trên val gốc):

| model / file ONNX | detection PyTorch / ONNX | khớp (IoU ≥ 0.5) | chỉ PyTorch / chỉ ONNX | mean IoU | \|Δconf\| mean / max | latency/ảnh PyTorch → ONNX |
|---|---|---|---|---|---|---|
| coco `coco_640.onnx` (vuông) | 2740 / 2899 | 98.5% | 40 / 199 | 0.984 | – | 76.9 → 107.2 ms |
| helmet `helmet_640.onnx` (vuông) | 697 / 726 | 98.6% | 10 / 39 | 0.971 | – | 77.9 → 108.3 ms |
| plate `plate_640.onnx` (vuông) | 146 / 262 | 97.9% | 3 / 119 | 0.909 | – | 38.9 → 43.5 ms |
| **coco `coco_384x640.onnx`** | 2740 / 2740 | **100%** | 0 / 0 | **1.000** | 0.0 / 0.0 (< 5e-6) | 76.2 → **62.5 ms** (−18%) |
| **helmet `helmet_384x640.onnx`** | 697 / 697 | **100%** | 0 / 0 | **1.000** | 0.0 / 0.0 | 77.5 → **63.6 ms** (−18%) |
| **plate `plate_640_dyn.onnx`** | 146 / 146 | **100%** | 0 / 0 | **1.000** | 0.0 / 0.0 | 39.4 → **25.9 ms** (−34%) |

Với đúng hình dạng đầu vào, ONNX FP32 trên CPU cho **cùng kết quả** với PyTorch FP32 CPU (tới sai số làm tròn) và nhanh hơn 18–34% mỗi lần suy luận.
Chưa đo FP16/INT8 (không có GPU provider; INT8 cần dữ liệu hiệu chuẩn từ tập train, không có `data/raw/`). Chưa đo mAP trên tập val gốc.

## Pipeline đầy đủ trên CPU (proxy Edge CPU)
Cấu hình `configs/pipeline/pt_cpu.yaml` và `configs/pipeline/onnx_cpu.yaml` (khác nhau đúng ở đường dẫn trọng số): backend chọn bằng `models.<name>.weights` (`.pt` hoặc `.onnx`).

**Sự kiện:** `tools/record_tracks.py --config configs/pipeline/onnx_cpu.yaml` cho **29 sự kiện giống hệt baseline v3 của PyTorch CPU ở dung sai 0 frame, kể cả ID track** (`diff_events` rỗng: 0 mất / 0 thêm / 0 đổi). Mức sự kiện, ONNX CPU tái tạo PyTorch CPU hoàn toàn trên video này.

**FPS** (`tools/benchmark_pipeline.py`, 2 cấu hình xen kẽ, 3 vòng, CPU Intel i5-11400H 12 luồng, warm-up 20 frame; khoảng min–max):

| backend (CPU) | headless FPS | stream FPS (+ JPEG) | `process` | COCO | helmet | encode |
|---|---|---|---|---|---|---|
| PyTorch FP32 | 4.65 (4.55–4.66) | 4.34 (4.25–4.35) | 212.9 ms | 106.2 ms | 102.3 ms | 15.5 ms |
| **ONNX Runtime FP32** | **4.97 (4.91–4.98)** | **4.52 (4.47–4.53)** | 198.8 ms | 99.9 ms | 93.9 ms | 19.9 ms |
| chênh | +6.9% | +4.1% | −6.6% | −5.9% | −8.2% | |

- **Lợi ích ở mức pipeline (~7%) nhỏ hơn rất nhiều so với mức từng model (−18%)**: trong `.track()`, ~30 ms/model là tracker (BoT-SORT + GMC, xem EXP-006) và chạy trên CPU tranh luồng với suy luận. Trên CPU, tắt GMC (đang là "tạm thời, chưa chấp nhận") sẽ có tác động lớn hơn việc đổi backend.
- Không đạt mục tiêu ≥ 15 FPS ở proxy CPU (~5 FPS). Mục tiêu đó chỉ được đặt cho proxy GPU (đạt ở EXP-006).
- Số đo CPU chưa gồm cấu hình ONNX + `gmc_none`; chưa chạy (còn việc tiếp theo cho EXP-008).

## Kết luận
1. ONNX CPU là backend **tương đương PyTorch về kết quả** (sự kiện giống hệt, detection giống hệt) và nhanh hơn một chút; chọn bằng một dòng config.
2. ONNX Runtime GPU và TensorRT: **không đo được trong phiên này** (CUDA 12 / TensorRT không có, theo quy tắc không cài). Là việc còn lại cho giai đoạn Edge thật.
3. Chưa kiểm định độ chính xác (chỉ có độ khớp backend).
