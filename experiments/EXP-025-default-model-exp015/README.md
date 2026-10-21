English translation of README.vi.md (original, Vietnamese).

# EXP-025 – default helmet model: `helmet_best.pt` -> EXP-015 `best.pt` (2026-10-20)

**Decision (project owner, 2026-10-20):** `configs/pipeline/default.yaml` uses `experiments/EXP-015-helmet-grouped-3cls/weights/best.pt` (sha256 `20ce0694eb1d8d81d6f579ede9b4d81c1aa1fbdf1d46e9391a63d04960fd7817`). The old model `helmet_best.pt` (sha256 `229700319e269ed19057bcedadb08f69881f6672024fdad85b8bde1ef9845f64`) can still be selected with `models.helmet.weights: helmet_best.pt`; the experiment configs (`configs/pipeline/exp/*`, `pt_*.yaml`, `ocr.yaml`...) are unchanged and still point to the old model. The helmet `conf` threshold (0.55) is **unchanged**.

**Reason:** (1) the in-domain metrics of EXP-015 were measured on a clean video-grouped split (no leakage), while the metrics of the old model are leaky (EXP-011); (2) on the out-of-domain camera (test_video) false alarms drop by about half (EXP-020: 14 -> 7 events, all confirmed by eye to be false alarms). **EXP-016 is not chosen yet** (4 classes) because its in-domain no-helmet class is doubtful, waiting for EXP-016b. This is NOT evidence that the new model is more correct: both give 100% false alarms on this clip; it only gives fewer false alarms.

## Comparison on test_video (same code, same config, only the model differs; `compare.py`, `events_old_vs_exp015.json`, `diff_events.txt`)
- Events: old model **14**, EXP-015 **7**; common (6), old model only (8), EXP-015 only (1). By type: no_helmet 13 -> 7, overload 1 -> 0.
- The time spans of the common events differ (start/end time and length); the table of every event with start/end seconds and the overlap time is in `events_old_vs_exp015.json`.
- `diff_events.txt`: `tools/diff_events.py` on the label/plate events of the two runs (the track ids differ between the two models, so "missing/added" also reflects the change of IDs).
- Every no_helmet/overload event of both models is a false alarm (EXP-020), so this table is about which false alarms disappear/appear, not about accuracy.

## Tests, baseline
- The characterization tests that use the track fixture of the old model (`tests/fixtures/tracks_test_video.jsonl`) are **not affected** (they load no model). The slow test `test_perception_unchanged_on_cpu` is pinned to `helmet_best.pt` (because the fixture was recorded with the old model).
- A new **baseline v6** for the new default model: `tests/fixtures/tracks_test_video_exp015.jsonl` + `baseline_events_v6_exp015.json` (CPU, current code: 15 events = 8 labels + 7 plate captures). **v5 is kept unchanged.** Test: `test_default_helmet_model_is_exp015_and_its_fixture_replay_reproduces_the_v6_label_events_exactly`.
- Web app: `/api/status` has `backend.helmet_model` = `{file, sha256}` of the helmet model that is running (`apps/web/routes.py::backend_info`).
