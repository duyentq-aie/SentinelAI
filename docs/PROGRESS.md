# PROGRESS – overnight run (started 2026-10-18 00:26, local time)

Rules in force: no push, no Kaggle/token, nothing outside the repo touched, server only on 127.0.0.1, no TensorRT, power plan untouched
(High performance, set earlier by the user's request), allowed installs only: onnx, onnxslim, onnxruntime(-gpu), imagehash.
Each item: own commit(s), fast `pytest` green. Benchmarks: 3 runs, median + range, only compared inside the same harness (`tools/benchmark_pipeline.py`).
No accuracy claims without ground truth ("chưa kiểm định").

| # | item | status | wall time | notes |
|---|---|---|---|---|
| 1 | max_attempts 90, wrong_way off, baseline v2 | done (commit pending with EXP-004 notes) | ~1 h | EXP-004: attempts 90 recovers the 21st plate (21/21 vs 20/21), +0.023 plate calls/frame; FPS within noise. Baseline v2 = 28 events (15 labels + 13 plates): v1 minus the 16 wrong-way events |
| 2 | 3.2 windowed no-helmet / overload | in progress | | rules + 17 unit tests written; CPU run for plate captures running |
| 3 | demo security | done (commit 0aa1d87) | ~40 min | 55 tests; app binds 127.0.0.1, no reload |
| 4 | P4 software optimisations | todo | | |
| 5 | ONNX | todo | | |
| 6 | OCR | todo | | |
| 7 | demo UI | todo | | |
| 8 | hand tools + guides | todo | | |
| 9 | README + REPORT v1 | todo | | |
| 10 | MORNING_REPORT | todo | | |

## Decisions taken without asking (reason, reversibility)
(see per-item sections below)

## Log
