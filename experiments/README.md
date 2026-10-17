# Experiments

Mỗi thí nghiệm nằm trong `EXP-###-<slug>/` với: `metrics.json`, `env.json` (nếu có đo), `NOTES.md`/config, và
phiên bản dữ liệu/model liên quan. Số trong báo cáo phải trỏ về một experiment ID ở đây.

| ID | Nội dung | Trạng thái |
|---|---|---|
| EXP-000-kaggle-original | Train gốc trên Kaggle (từ metadata checkpoint) | Đã điền từ notebook helmet (val+test theo class, 1014/278/168 ảnh); plate: notebook không khớp checkpoint |
| EXP-001-baseline | Benchmark pipeline gốc `app.py@v0-baseline` (PyTorch FP32), v1 (Balanced) + `baseline-v2/` (High performance, phân rã chi tiết) | P0 xong |
| EXP-002-perception-clean | Models + evidence on the raw frame (3.0b); perception diff, baseline v1 | xong: perception ~không đổi, 23 nhãn giữ, chụp biển 20→21 |
| EXP-003-evidence | Bằng chứng đầy đủ + plate theo sự kiện (3.1); A/B v0 vs event policy | xong: 44 sự kiện giữ nguyên, plate/frame 1.73→0.13, FPS headless 7.29→8.88 (xem NOTES: nhiễu) |
