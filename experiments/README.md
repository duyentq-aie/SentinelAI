# Experiments

Mỗi thí nghiệm nằm trong `EXP-###-<slug>/` với: `metrics.json`, `env.json` (nếu có đo), `NOTES.md`/config, và
phiên bản dữ liệu/model liên quan. Số trong báo cáo phải trỏ về một experiment ID ở đây.

| ID | Nội dung | Trạng thái |
|---|---|---|
| EXP-000-kaggle-original | Train gốc trên Kaggle (từ metadata checkpoint) | Thiếu số per-class/test của notebook |
| EXP-001-baseline | Benchmark pipeline gốc `app.py@v0-baseline` (PyTorch FP32) | P0 |
