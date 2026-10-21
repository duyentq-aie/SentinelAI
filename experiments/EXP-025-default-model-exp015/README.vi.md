# EXP-025 – model helmet mặc định: `helmet_best.pt` -> EXP-015 `best.pt` (2026-10-20)

**Quyết định (chủ dự án, 2026-10-20):** `configs/pipeline/default.yaml` dùng `experiments/EXP-015-helmet-grouped-3cls/weights/best.pt` (sha256 `20ce0694eb1d8d81d6f579ede9b4d81c1aa1fbdf1d46e9391a63d04960fd7817`). Model cũ `helmet_best.pt` (sha256 `229700319e269ed19057bcedadb08f69881f6672024fdad85b8bde1ef9845f64`) vẫn chọn được bằng `models.helmet.weights: helmet_best.pt`; các config thử nghiệm (`configs/pipeline/exp/*`, `pt_*.yaml`, `ocr.yaml`...) không đổi và vẫn trỏ tới model cũ. Ngưỡng `conf` của helmet (0.55) **không đổi**.

**Lý do:** (1) chỉ số trong miền của EXP-015 đo trên split sạch theo video (không rò rỉ), còn chỉ số của model cũ bị rò rỉ (EXP-011); (2) trên camera ngoài miền (test_video) báo oan giảm khoảng một nửa (EXP-020: 14 -> 7 sự kiện, đều được xác nhận bằng mắt là báo oan). **Chưa chọn EXP-016** (4 lớp) vì lớp no-helmet trong miền của nó đáng ngờ, chờ EXP-016b. Đây KHÔNG phải bằng chứng model mới đúng hơn: cả hai đều báo oan 100% trên clip này; mới chỉ báo oan ít hơn.

## So sánh trên test_video (cùng mã, cùng config, chỉ khác model; `compare.py`, `events_old_vs_exp015.json`, `diff_events.txt`)
- Sự kiện: model cũ **14**, EXP-015 **7**; chung (6), chỉ model cũ (8), chỉ EXP-015 (1). Theo loại: no_helmet 13 -> 7, overload 1 -> 0.
- Khoảng thời gian của các sự kiện chung lệch nhau (thời điểm bắt đầu/kết thúc và độ dài); bảng từng sự kiện kèm giây bắt đầu/kết thúc và thời gian chồng nhau nằm trong `events_old_vs_exp015.json`.
- `diff_events.txt`: `tools/diff_events.py` trên các sự kiện nhãn/biển số của hai lần chạy (các track id khác nhau giữa hai model, nên "missing/added" phản ánh cả đổi ID).
- Mọi sự kiện no_helmet/overload của cả hai model là báo oan (EXP-020), nên bảng này nói về việc báo oan nào biến mất/xuất hiện, không nói về độ chính xác.

## Test, baseline
- Các test đặc trưng dùng fixture track của model cũ (`tests/fixtures/tracks_test_video.jsonl`) **không bị ảnh hưởng** (không tải model). Test chậm `test_perception_unchanged_on_cpu` được ghim vào `helmet_best.pt` (vì fixture ghi bằng model cũ).
- **Baseline v6** mới cho model mặc định mới: `tests/fixtures/tracks_test_video_exp015.jsonl` + `baseline_events_v6_exp015.json` (CPU, mã hiện tại: 15 sự kiện = 8 nhãn + 7 chụp biển số). **v5 giữ nguyên.** Test: `test_default_helmet_model_is_exp015_and_its_fixture_replay_reproduces_the_v6_label_events_exactly`.
- Web app: `/api/status` có `backend.helmet_model` = `{file, sha256}` của model helmet đang chạy (`apps/web/routes.py::backend_info`).
