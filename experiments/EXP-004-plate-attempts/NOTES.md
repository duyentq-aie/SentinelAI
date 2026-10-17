# EXP-004 – `max_attempts` 30 so với 90, tắt wrong_way, baseline v2 (đêm 2026-10-18, mục 1)

## Thí nghiệm A/B: `evidence.plate.max_attempts` = 30 so với 90
Cấu hình `configs/pipeline/exp/attempts{30,90}_wrongway_on.yaml` (giống nhau trừ `max_attempts`; wrong_way còn bật để cô lập tác động).
`tools/benchmark_pipeline.py --configs a b`, **xen kẽ** (vòng 1: a rồi b; vòng 2: ...), 3 vòng, GPU, FP32, 720p, High performance.

| | max_attempts 30 | max_attempts 90 |
|---|---|---|
| track vi phạm có biển (GPU) | **20 / 21** | **21 / 21** |
| lần gọi plate / frame | 0.128 | 0.151 |
| FPS headless (trung vị, khoảng) | 10.58 (10.08 – 11.50) | 10.26 (9.60 – 11.29) |
| FPS có stream (trung vị, khoảng) | 9.14 (8.72 – 9.87) | 8.87 (8.30 – 9.71) |
| nhiệt GPU đỉnh mỗi lượt | 65 / 64 / 62 °C | 64 / 63 / 61 °C |

- 90 lần thử tìm lại được biển của track thứ 21 (kết quả giống nhau cả 3 lượt) và chỉ tốn thêm 0.023 lần gọi detector/frame (+18% so với 0.128, vẫn thấp hơn 1.73 của chính sách cũ 11 lần).
- Hai khoảng FPS **chồng lên nhau**; không kết luận được 30 nhanh hơn 90. Chênh trung vị 0.3 FPS nằm trong nhiễu.
- **Quyết định tự chọn:** đặt mặc định `max_attempts: 90` (đảo ngược được: một dòng trong config).
- Lưu ý: FPS ở đây (10.3–10.6) cao hơn EXP-003 (8.9) dù cùng mã pipeline. Hai phiên đo khác thời điểm (GPU/CPU trạng thái khác nhau); **không so số giữa phiên này và EXP-003**, chỉ so trong cùng phiên xen kẽ như bảng trên.

## Tắt `rules.wrong_way` mặc định và baseline v2
Luật ngược chiều nằm ngoài phạm vi báo cáo (PLAN mục 5) nên tắt mặc định (`rules.wrong_way.enabled: false`).
`tests/fixtures/baseline_events_v2.json` (CPU, engine `legacy`, wrong_way tắt, max_attempts 90): **28 sự kiện = 15 nhãn + 13 chụp biển**.
`diff_events` v1_clean → v2: **16 sự kiện mất, 0 thêm, 0 đổi**; 16 sự kiện mất là đúng 8 track WRONG-WAY (C_223, C_261, C_3, C_592, C_735, C_874, C_890, C_1090), mỗi track 1 nhãn + 1 lần chụp biển. Mọi sự kiện no-helmet / overload giữ nguyên frame.
Test: `test_rules_offline.py` kiểm luật legacy (wrong_way bật) tái tạo nhãn của v0 và v1, và (wrong_way tắt) nhãn của v2.

Chưa kiểm định bằng ground truth.
