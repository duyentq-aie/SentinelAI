# EXP-019 – tham số tính theo thời gian: kiểm bất biến ở 30 FPS (2026-10-19)

Thay đổi: `sentinel/timebase.py`, cấu hình bằng giây / tỉ lệ, `Pipeline.set_fps` (xem `docs/TIME_BASED_PARAMETERS.md`). Không tinh chỉnh giá trị nào.

Kiểm tra bất biến "ở 30 FPS kết quả giống hệt":
1. **Test đặc trưng** (`tests/test_rules_offline.py`, baseline v0…v5) và toàn bộ test nhanh: 317 passed.
2. **`diff_events` so với baseline v5** (pipeline đầy đủ trên CPU, `tools/record_events.py --impl refactored --device cpu`): `MISSING 0 / ADDED 0 / CHANGED 0`, hai danh sách 29 sự kiện bằng nhau từng phần tử (frame, loại, track, nhãn).
3. **Khoảng thời gian sự kiện** (GPU, `tools/review_run.py`, cùng video và cùng weights, trước = `outputs/review_A`, sau = `outputs/review_A_timebase`): 14 / 14 sự kiện trùng hoàn toàn ở loại, track, `frame_start`, `frame_end`, `trigger_frame`, mọi hộp mẫu (`boxes`), `key_box`, frame và số lần thử của biển số; 14 / 14 sự kiện đặc trưng trùng.
4. **Ở 10 FPS** (test `tests/test_timebase.py`): mọi cửa sổ dài đúng cùng số giây (sai số tối đa nửa khung hình do làm tròn), cờ vi phạm bật và tắt lệch tối đa một khung hình so với 30 FPS, ngân sách tìm biển số là 3 s (30 lần thay cho 90), đóng sự kiện sau 1 s, buffer tracker quy đổi 30 → 10 frame. Chưa đo ảnh hưởng tới sự kiện trên video ở 10 FPS (PLAN G10 bước 4).
