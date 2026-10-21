# EXP-009 – Vì sao FPS tuyệt đối đổi giữa các phiên đo, và vì sao phải đo kèm v0 (2026-10-18)

## Quy ước đo mới (từ đây trở đi)
`tools/benchmark_pipeline.py` **luôn chạy kèm `v0`** (code gốc `app.py` ở tag `v0-baseline`, không sửa) trong cùng phiên, xen kẽ từng vòng với các cấu hình khác, và báo
**tỉ lệ so với v0 đo trong cùng vòng** (trung vị các vòng + khoảng). FPS tuyệt đối chỉ để tham khảo. Telemetry (`tools/sysprobe.py`) được lấy mẫu mỗi 2 s trong từng lượt:
xung nhịp SM/bộ nhớ GPU, công suất, nhiệt độ, các bit lý do throttle của NVIDIA, "% Processor Performance" của CPU (tần số hiệu dụng; `psutil.cpu_freq()` cố định trên Windows nên không dùng),
tải CPU của **các tiến trình khác** và tiến trình nặng nhất. `env.json` ghi power plan, AC, GPU và các tiến trình đang dùng GPU lúc bắt đầu.

## Thí nghiệm có kiểm soát: tải nền CPU
Cùng cấu hình, cùng mã, hai phiên liền nhau: **yên tĩnh** (5 vòng) và **có 6 tiến trình đốt CPU** (`tools/cpu_burn.py --procs 6`, 6 trên 12 luồng logic; 3 vòng).
`default` = cấu hình mặc định hiện tại (BoT-SORT không GMC); `tracker_botsort_gmc` = cùng cấu hình nhưng tracker mặc định của Ultralytics.

| | stream FPS yên tĩnh | stream FPS có tải nền | đổi | tỉ lệ vs v0 yên tĩnh | tỉ lệ vs v0 có tải |
|---|---|---|---|---|---|
| v0 (code gốc) | 8.61 (8.37–8.70) | 5.28 (5.16–5.33) | **−39%** | – | – |
| default (không GMC) | 20.86 (20.63–21.08) | 12.60 (12.47–12.73) | **−40%** | **2.42×** (2.40–2.46) | **2.39×** (2.36–2.44) |
| tracker mặc định Ultralytics (GMC) | 12.57 (12.49–12.59) | 7.58 (7.54–7.61) | **−40%** | **1.46×** (1.45–1.50) | **1.44×** (1.42–1.46) |

headless (đọc + xử lý + vẽ, không JPEG): v0 9.72 → 6.00 (−38%), default 27.86 → 17.02 (−39%), GMC 14.84 → 8.96 (−40%).

Telemetry hai phiên:
| | yên tĩnh | có tải nền |
|---|---|---|
| CPU performance (tần số hiệu dụng) | **150%** (turbo) | **114%** |
| tải CPU của tiến trình khác | 4% | 55% |
| xung nhịp SM GPU | 1500 MHz (tối đa 2100) | 1500 MHz |
| công suất / nhiệt độ GPU | 19–22 W / ≤ 61 °C | 17–19 W / ≤ 61 °C |
| bit throttle (power cap, thermal, brake…) | không có (ngoài `gpu_idle`) | không có |
| GPU util | 21–32% | 11–20% |

**Kết luận đo được:** (1) pipeline bị giới hạn bởi **CPU**: khi ~50% CPU bị chiếm, tần số hiệu dụng giảm 150% → 114% và mọi cấu hình giảm ~40% FPS, trong khi GPU (xung nhịp, nhiệt, công suất) không đổi và còn rảnh;
(2) **tỉ lệ so với v0 cùng phiên gần như không đổi** (2.42× → 2.39×, 1.46× → 1.44×, chênh ≤ 2%) nên đây là cách báo đáng tin; (3) GPU không phải nguyên nhân của biến thiên giữa phiên trên máy này.

## Còn "7 vs 12 FPS" thì sao? (sửa nhận định trước đây)
Tôi từng viết "cùng mã chạy 7 FPS ở phiên này, 12 FPS ở phiên khác". **Nhận định đó sai một phần**: 7.0–7.4 là pipeline **v0** (EXP-001, 1.73 lần gọi plate/frame, wrong-way bật, vẽ toàn khung) còn 12.05 là pipeline mới ở EXP-006
(luật windowed, wrong-way tắt, 0.018 lần gọi plate/frame). Phần lớn chênh lệch là **khối lượng việc khác nhau**, không phải máy. Phiên đo yên tĩnh hôm nay cho thấy riêng việc đó đã là 1.46× (GMC bật) tới 2.42× (GMC tắt) so với v0.
Biến thiên **thật** giữa phiên với cùng mã vẫn tồn tại, nhưng nhỏ hơn:
- v0: 7.38 (EXP-001, tối 17/10, plan High performance; harness cũ `tools/benchmark.py`, nên chỉ mang tính tham khảo) → 8.61 (phiên yên tĩnh hôm nay, harness mới) = +17%; bản chạy ngắn 60 frame cho 10.04 (không dùng: quá ngắn, khởi động lạnh).
- cùng cấu hình `attempts30` (wrong-way bật): 8.88 (EXP-003, ~00:00) so với 10.58 (EXP-004, ~00:30) = +19%.
**Nguyên nhân của biến thiên này không xác định lại được** vì thời đó chưa có telemetry. Thí nghiệm trên chứng minh một tác nhân đủ lớn (tải nền CPU ~50% → −40%) và loại trừ tác nhân GPU (xung nhịp luôn 1500 MHz — kể cả trong các phiên cũ,
`gpu_min_sm_clock_mhz = 1500` ở mọi EXP — không throttle, nhiệt 58–66 °C). Các tiến trình nền thường thấy lúc đo: `sqlservr.exe`, `MsMpEng.exe` (Defender), `Code.exe`, `claude.exe`, `esrv_svc.exe` (mỗi cái 1–6% tổng CPU).
GPU được dùng: duy nhất RTX 3050 Laptop (`nvidia-smi -L`), chỉ vài tiến trình đồ họa Windows (dwm, WebView, VS Code) có mặt trên GPU lúc bắt đầu, chiếm ~170 MiB.
Những gì **chưa kiểm tra**: chế độ tiết kiệm năng lượng theo tiến trình của Windows (EcoQoS) khi cửa sổ không ở foreground; giới hạn công suất CPU (PL1/PL2) khi nóng dài; nhiệt độ CPU (không đọc được bằng công cụ có sẵn).

## Hệ quả cho các tài liệu
`docs/REPORT.md`, `README.md` bỏ mọi "đạt 15 FPS" và mọi so sánh FPS tuyệt đối giữa phiên; dùng tỉ lệ so với v0 (EXP-010). `docs/MORNING_REPORT.md` và EXP-006 có chú thích đính chính.
