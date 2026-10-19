# Tham số tính theo thời gian (2026-10-19)

Mọi tham số từng đếm bằng **frame** nay tính bằng **giây** hoặc **tỉ lệ**; số frame được suy ra từ tốc độ khung hình mà pipeline thực sự nhận (fps của nguồn / bước nhảy frame) bằng `sentinel/timebase.py`. Lý do: thiết bị đích xử lý ≥ 10 FPS (PLAN G8) và một cửa sổ "15 frame" dài 0.5 s ở 30 FPS nhưng 1.5 s ở 10 FPS.

## Cách hoạt động
- `Pipeline(..., fps=30)`, `Pipeline.set_fps(fps)`: rules, recorder và tracker đọc tốc độ từ đây. `InferenceWorker` gọi `set_fps` với fps danh định của nguồn (`VideoSource.fps`; nguồn không báo fps thì giữ nguyên). Offline: `make_rules(cfg.rules, camera, fps=...)`.
- Làm tròn: nửa lên (`to_frames`), tối thiểu 1 frame. Số phiếu `K = round(min_vote_fraction × W)` trong khoảng 1..W. Ở 10 FPS: W = 5, K = 2 (2/5 = 0.4, thay cho 1/3): lượng tử hoá không tránh được ở cửa sổ ngắn; `min_ratio` (0.5) vẫn áp dụng như cũ.
- **Bất biến ở 30 FPS:** mọi giá trị khớp từng frame với cấu hình cũ (W = 15, K = 5, close_after = 30, 90 lần thử biển số, buffer tracker = 30 ...). Đã kiểm: test đặc trưng v0…v5 đạt; `diff_events` so với baseline v5 = 0 thay đổi; khoảng thời gian sự kiện (start, end, hộp mẫu, frame và số lần thử biển số) trùng 100% giữa trước và sau thay đổi (xem EXP-019).
- Cấu hình cũ viết bằng frame vẫn đọc được: giá trị cũ được hiểu là frame ở 30 FPS (= giây × 30), kèm `UserWarning`. Lệnh chuyển đổi: `python tools/convert_config_to_seconds.py <file> [--write]` (giữ chú thích với yaml dạng khối; yaml dạng `{...}` thì dựng lại file, mất chú thích).
- Tracker: yaml giữ `track_buffer: 30` (frame ở 30 FPS = 1 s). Ở fps khác, một bản sao yaml tạm có `track_buffer` quy đổi được dùng và tracker được tạo lại; ở 30 FPS dùng đúng file gốc.

## Bảng rà soát
| tên | giá trị | đơn vị cũ | mới (trạng thái) |
|---|---|---|---|
| `rules.helmet.window` | 15 | frame (quan sát) | `window_s: 0.5` ✔ |
| `rules.helmet.min_votes` (K) | 5 | frame | `min_vote_fraction: 1/3` = K/W ✔ |
| `rules.helmet.min_ratio` | 0.5 | tỉ lệ | giữ |
| `rules.helmet.exit_ratio` (hysteresis) | 0.3 | tỉ lệ | giữ |
| `rules.helmet.violation_frames` (engine legacy) | 5 | frame | `violation_s: 0.1667` ✔ |
| `rules.overload.window` / `min_votes` | 15 / 5 | frame | `window_s: 0.5` / `min_vote_fraction: 1/3` ✔ |
| `rules.overload.max_passengers` | 2 | người | giữ (không phải thời gian) |
| `rules.wrong_way.history_len` / `min_history` | 30 / 15 | frame | `history_s: 1.0` / `min_history_s: 0.5` ✔ |
| `rules.wrong_way.dy_threshold` | −30 | px trong cả khoảng lịch sử | giữ px (là quãng đường trong `history_s`) |
| `evidence.close_after` | 30 | frame | `close_after_s: 1.0` ✔ |
| `evidence.plate.max_attempts` | 90 | số lần chạy (≈ frame ở 30 FPS) | `search_s: 3.0`; số lần = `search_s × fps / every` ✔ |
| `evidence.plate.every` | 1 | frame | `every_s: 0` (mỗi frame; k > 1 → k/30 s) ✔ |
| `evidence.ocr.min_gap` | 3 | frame | `min_gap_s: 0.1` ✔ |
| `evidence.ocr.candidates` / `reads` / `margin` | 5 / 3 / 2 | số crop / px | giữ |
| `BOX_SAMPLE_EVERY` (hằng trong code) | 5 | frame | `evidence.box_sample_s: 0.1667` ✔ |
| `BOX_SAMPLE_MAX` | 60 | số hộp | giữ |
| `PURGE_EVERY` / `PURGE_AFTER` (windowed) | 300 / 300 | frame | `PURGE_EVERY_S` / `PURGE_AFTER_S` = 10 s ✔ |
| tracker `track_buffer` | 30 | frame ở 30 FPS | giữ trong yaml; quy đổi theo fps khi chạy ✔ |
| tracker `match_thresh`, `new_track_thresh`, `track_high/low_thresh` | 0.8, 0.25, 0.25, 0.1 | tỉ lệ / điểm | giữ |
| debounce đèn giao thông / cooldown sự kiện | — | — | **không tồn tại trong code**: trạng thái đèn lấy từ frame hiện tại; thay cho cooldown là `evidence.one_event_per_track` |
| `web.idle_after` (5 s), `worker.open_retry` (1 s), `read_retry` (0.05 s) | — | giây | đã là giây |
| `worker.reopen_after_failures` | 100 | số lần thử đọc | giữ (đếm lần thất bại, không phải thời gian của video) |
| cửa sổ FPS của HUD (`deque(maxlen=60)`) | 60 | frame mẫu | giữ (chỉ để hiển thị) |
| `evidence.max_records` / `max_dir_mb` / `plate_margin` | 12 / 200 / 10 | số mục / MB / px | giữ |

## Chưa làm / ghi chú
- Chưa tinh chỉnh bất kỳ giá trị nào (vẫn "chưa kiểm định"); chỉ đổi đơn vị.
- Ở 10 FPS chưa đo ảnh hưởng tới sự kiện (PLAN G10 bước 4); test mới chỉ xác nhận cửa sổ dài đúng cùng số giây và thời điểm bật/tắt cờ lệch tối đa một khung hình so với 30 FPS.
- Các cấu hình lịch sử (`configs/pipeline/exp/*`, `ab_plate_v0.yaml`) giữ nguyên dạng frame để đúng với các thí nghiệm đã ghi; chúng đọc được kèm cảnh báo.
