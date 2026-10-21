# EXP-010 – Các bước tối ưu P4 đo dưới dạng tỉ lệ so với v0 (2026-10-18)

Lệnh: `python tools/benchmark_pipeline.py` với 5 cấu hình + v0, 3 vòng xen kẽ, mỗi vòng v0 chạy đầu tiên. Số liệu: `summary.json`, `<cfg>/metrics.json`, `env.json`, `runs/`.

| cấu hình | stream FPS (trung vị, khoảng) | tỉ lệ vs v0 cùng vòng (trung vị, khoảng) |
|---|---|---|
| v0 (code gốc) | 6.82 (6.72–8.68) | 1× |
| p4_before (blend toàn khung) | 8.90 (8.63–9.57) | 1.28× |
| p4a (blend theo ROI) | 9.78 | 1.20× (1.13–1.50) |
| tracker mặc định Ultralytics (GMC sparse) | 10.09 | 1.40× |
| **default** (BoT-SORT, không GMC – TẠM THỜI) | 16.01 | **2.27×** (1.84–2.42) |
| ByteTrack | 16.40 | 2.23× (1.89–2.54) |

## Phiên này nhiễu – đọc cho đúng
- Tiến trình nền `Ld9BoxHeadless.exe` (trình giả lập LDPlayer, ~20% CPU) chạy trong vòng 2–3 (`cpu_other_pct` 4.3 → 22.7 → 21.4); v0 stream theo vòng: 8.68, 6.72, 6.82.
- Vì v0 và cấu hình đo chịu cùng tải trong từng vòng nên tỉ lệ ổn định hơn FPS tuyệt đối, nhưng khoảng vẫn rộng (vd default 1.84–2.42): tải nền đổi **giữa** lượt v0 và lượt cấu hình, không cắt đều.
- Xung nhịp GPU 1500 MHz ở v0/GMC, 1695–1942 MHz ở default/ByteTrack (không throttle) – GPU nhàn rỗi nên xung nhịp không quyết định FPS.
- **Không dùng bảng này để kết luận từng bước nhỏ** (p4a vs p4_before nằm trong nhiễu). Số sạch hơn để trích dẫn: EXP-009 (phiên yên tĩnh, 5 vòng): default **2.42×** (2.40–2.46), tracker GMC **1.46×** (1.45–1.50).
- Phát biểu an toàn: bỏ GMC (mặc định tạm thời) cho ≈ 2.2–2.4× v0; giữ GMC cho ≈ 1.4–1.5× v0. Không có claim "đạt 15 FPS".
