# EXP-012 – ONNX Runtime GPU (FP32/FP16) vs PyTorch GPU (2026-10-18, venv-edge)

## Summary (English)
- Purpose: compare ONNX Runtime GPU (FP32 / FP16) with PyTorch GPU in a separate venv-edge (torch 2.7.1+cu126, onnxruntime-gpu 1.22.0).
- Result: ratio to v0 in the same round: PyTorch 3.07x, ONNX FP32 3.07x, ONNX FP16 3.35x (stream 22.81 FPS versus 20.93); the main venv (cu118) gave 2.44x; every configuration produced 14 events on 14 tracks with plates.
- Conclusion: FP16 is only about 9% above PyTorch FP32, near the noise level, because the pipeline is CPU-bound; torch cu126 ran about 25% faster than cu118 in the same session (cause not investigated).
- Limits: events were counted, not diffed one by one, so FP16 equivalence is not claimed; TensorRT / INT8 not tried; no real edge device, all numbers are an RTX 3050 Laptop proxy.

**Môi trường:** `venv-edge/` riêng (gitignored; venv chính không bị đụng): torch 2.7.1+cu126, onnxruntime-gpu **1.22.0** (CUDA 12 + cuDNN 9 lấy từ wheel torch cu126), ultralytics 8.4.26. Lưu ý: `onnxruntime-gpu 1.31.0` là bản build cho CUDA 13 (đòi `cublas64_13.dll`) nên không chạy được với torch cu126 → dùng 1.22.0. Không cài CUDA toolkit hệ thống. Thời gian dùng: ~35 phút / 120 phút.
**Phần cứng (proxy, không phải thiết bị Edge thật):** RTX 3050 Laptop 4 GB, laptop Windows 11. Lệnh: `benchmark_pipeline.py --repeats 3 --configs pt_gpu pt_gpu_edge@venv-edge onnx_gpu_fp32@venv-edge onnx_gpu_fp16@venv-edge`; v0 chạy xen kẽ mỗi vòng. ONNX FP16 xuất bằng `tools/export_onnx.py --half --out models/onnx_edge` (từ venv-edge; GPU).

| cấu hình (cùng tracker mặc định, GMC tắt) | stream FPS | headless FPS | tỉ lệ vs v0 cùng vòng |
|---|---|---|---|
| v0 | 6.95 (6.82–7.00) | 7.79 | 1× |
| PyTorch FP32 GPU, venv chính (torch cu118) | 16.94 (16.66–18.09) | 22.29 | 2.44× (2.44–2.58) |
| PyTorch FP32 GPU, venv-edge (torch cu126) | 20.93 (20.26–23.67) | 22.31 | 3.07× (2.92–3.38) |
| ONNX Runtime CUDA FP32 | 21.30 (20.93–25.45) | 22.77 | 3.07× (3.06–3.64) |
| ONNX Runtime CUDA **FP16** | 22.81 (22.70–26.51) | 24.50 | **3.35×** (3.27–3.79) |

Đọc đúng:
- So **cùng môi trường** (venv-edge): ONNX FP32 ≈ PyTorch (3.07× cả hai); FP16 hơn PyTorch FP32 ≈ 9% (3.35× vs 3.07×), nằm gần ngưỡng nhiễu (khoảng chồng nhau một phần). Pipeline bị giới hạn bởi CPU (tracker, vẽ, JPEG), nên tăng tốc model chỉ mang lại ít.
- Hai dòng PyTorch khác nhau 2.44× vs 3.07×: chỉ khác bản torch (cu118 vs cu126) và nhiễu phiên; không điều tra thêm. Không so FPS tuyệt đối giữa các phiên.
- Sự kiện: mọi cấu hình đều 14 sự kiện / 14 track có biển (v0: 20/20 do chính sách cũ) – không có khác biệt đếm giữa PyTorch/ONNX/FP16; **chưa** diff từng sự kiện (việc kế tiếp: `tools/diff_events.py`), nên chưa claim tương đương chính xác của FP16.
- Telemetry: GPU util 37–46%, xung 1750–1940 MHz, không throttle; tải nền khác ~22–24% CPU (Ld9BoxHeadless của trình giả lập đang chạy, không tắt).
- TensorRT / INT8: **không thử** (chưa cài TensorRT; ngoài thời gian ưu tiên). `Edge thật`: không có; mọi số là proxy RTX 3050 Laptop.

## Ghi nhận (2026-10-18, theo quyết định chủ dự án)
torch cu126 (venv-edge) chạy nhanh hơn cu118 (venv chính) ~25% cùng mã, cùng phiên (stream 16.94 → 20.93 FPS; tỉ lệ vs v0 2.44× → 3.07×). Chưa kiểm tra lặp lại hay nguyên nhân (cùng torch 2.7.1, khác bản CUDA/cuDNN/cuBLAS). **Venv chính chưa đổi.** TensorRT/INT8: ngoài phạm vi M1 (PLAN).
