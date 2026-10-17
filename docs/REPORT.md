# SentinelAI – technical report v1

Status: **engineering prototype with an honest measurement trail; not a validated violation-detection system.**
Written 2026-10-18 from the experiment folders in `experiments/` (every number below points to one). All speed numbers: RTX 3050 Laptop 4 GB + i5-11400H, Windows 11, PyTorch 2.7.1+cu118, Ultralytics 8.4.26, 1280×720, batch 1.

## 1. Summary
| | |
|---|---|
| What it does | detects no-helmet / >2 riders on motorcycles (and, via the legacy zone rule, red-light running), tracks vehicles, saves evidence per event, optionally reads plate text, serves a demo UI |
| What was improved on the original prototype | perception fixed (models no longer see overlays), rules rewritten as voting over a window, evidence + plate search made event-driven and bounded, security hardened, software optimised (default config +26% headless / +24% stream in the EXP-006 A/B, on top of the plate-policy gain in EXP-003; an unadopted tracker change would give ~2.4×), ONNX backend, OCR, tooling for recording/labelling |
| What is **not** known | the precision/recall of the events (no labelled video), generalisation of the detectors (original test split not verified independent), behaviour at night / rain / other cameras, red-light rule quality (no red-light event in the clip), Edge performance on real hardware |
| Biggest open risks | (1) no ground truth, (2) plate crops are ~25×23 px at 720p so OCR cannot work on this footage, (3) tracker changes that give 2× speed need validation before adoption |

## 2. System
```
video/RTSP/file ──read──> raw frame (resized 1280x720, never drawn on)
                              │
                              ├─ COCO YOLO11s  + tracker ─┐   vehicles, traffic lights (HSV colour of the light box)
                              ├─ helmet YOLO11s + tracker ─┴─> Perception (tracks, light state)    <- sentinel/perception.py
                              │
                              ▼
                    rules (pure function of Perception)                      <- sentinel/rules/{legacy,windowed}.py
                      no-helmet / overload: vote over a sliding window, red light: v0 zone rule
                              │ violating vehicles
                              ▼
        evidence recorder (one event per track and type)                    <- sentinel/evidence.py
          raw frame + vehicle crop + events.jsonl; plate detector only for violating tracks (bounded attempts);
          optional OCR: collect crops -> best crop -> read 3 -> normalise (VN format) -> vote
                              │
        display copy (overlays) ──> JPEG only when a client asks ──> FastAPI MJPEG + /api/events + /api/status    <- apps/web
```
Design points that mattered: models and evidence use the **raw** frame (the v0 code fed drawn overlays into the helmet and plate models, EXP-002); rules never see pixels, so they are tested offline from recorded tracks (`tests/fixtures/tracks_test_video.jsonl`, replay in < 1 s); one inference worker serves every HTTP client.

## 3. Models and data (EXP-000)
- Helmet: YOLO11s fine-tuned on a Roboflow export (1014/278/168 train/val/test images; classes helmet, motorcycle, no-helmet); test mAP50 0.994, mAP50-95 0.812 (per class 0.993/0.995/0.995) — **saturated and probably optimistic**: test ≈ val and the images are frames extracted from video (split independence unchecked).
- Plate: YOLO11n (50 epochs, val mAP50 0.995 / 0.752). **The saved notebook does not correspond to the deployed weights** (10 of 19 training arguments differ; no outputs saved).
- COCO YOLO11s for cars/trucks/buses and traffic lights. Datasets are not in the repo (`data/raw/` absent); their licences are unknown.

## 4. Experiments
| ID | question | answer (same-session comparisons only) |
|---|---|---|
| EXP-000 | what was trained? | helmet notebook == checkpoint; plate notebook != checkpoint |
| EXP-001 | where does the v0 time go? | 135.6 ms/frame (7.4 FPS): COCO 37.2 (inference 12.0, tracker 21.6), helmet 34.8 (12.0 / 19.5), plate 28.8, draw 15.1, JPEG 14.6; GPU idle ~83% → **CPU-bound** |
| EXP-002 | does drawing before inference matter? | not on this clip: 21 005 + 5 164 detections identical; 23 label events identical; plate captures 20 → 21 |
| EXP-003 | plate detector only for violating tracks, bounded | plate calls/frame 1.73 → 0.13; headless 7.29 → 8.88 FPS, stream 6.56 → 7.74 (noisy session, see notes) |
| EXP-004 | max_attempts 30 vs 90 | 90 finds 21/21 plates (30: 20/21) for +0.02 calls/frame; FPS within noise |
| EXP-005 | voting rules vs v0 counters | −1 event (M_1), +2 (M_176 borderline 1-frame, M_702 intermittent heads), 1 overload glitch rejected; timing ±1–4 frames; **unvalidated** |
| EXP-006 | software optimisation | ROI blend +21% headless / +18% stream; JPEG q80 +4–5%; tracker gmc none / ByteTrack ~+90% / +67% (not adopted) |
| EXP-007 | ONNX | rectangular 384×640 export: identical detections to PyTorch, −18…−34% latency per model; pipeline events identical; CPU pipeline +7% FPS; GPU/TensorRT not measurable here |
| EXP-008 | plate OCR | see §8 |

## 5. Performance
Same-session, interleaved A/B (3 rounds; median and range; `tools/benchmark_pipeline.py`), GPU FP32. *headless* = read + process (with drawing); *stream* = + JPEG encode, no HTTP.

| step (EXP-006) | headless FPS | stream FPS | `process` |
|---|---|---|---|
| before (whole-frame blend, JPEG 95) | 12.05 (11.73–13.19) | 10.34 (10.08–11.24) | 80.6 ms |
| + ROI label blending | 14.64 (14.51–15.15) | 12.22 (12.15–12.60) | 66.4 ms |
| + JPEG quality 80 (current default) | 15.22 (14.57–15.25) | 12.85 (12.38–12.89) | 63.9 ms |
| + BoT-SORT `gmc_method: none` (**not default**) | 28.73 (28.55–28.82) | 21.46 (21.30–21.53) | 33.0 ms |
| + ByteTrack instead (**not default**) | 28.21 (23.60–29.20) | 21.07 (17.82–21.71) | 33.5 ms |

Latency now (default config, EXP-006 `p4b`): COCO 31.5 ms (of which tracker ≈ 15), helmet 29.1 ms (tracker ≈ 15), plate detector 0.6, drawing 0.4, read 1.8, JPEG 12.1 (only when a client is watching).
**Absolute numbers are not comparable across experiments** (the machine ran 7 FPS in one session and 12 FPS in another with the same code).
Edge proxy (CPU, EXP-007): PyTorch 4.65 → ONNX Runtime 4.97 FPS headless; the tracker, not the model, dominates.

## 6. Rules and events
- **v0 rules** (kept as `rules.engine: legacy`): running counters, per-frame head count, a car moving "up" the image = wrong-way (switched off by default: out of scope), vehicle bottom-centre in a fixed zone on red = red-light (wrong for a car waiting at the light).
- **Windowed rules** (default): per motorcycle, observations = frames with a head attached; no-helmet when ≥ K=5 votes and ≥ r=50% of the last W=15 observations; overload likewise (> 2 heads). 1 event per (track, type). Parameters are defaults, **not tuned**.
- Event comparison tooling: `tools/diff_events.py` (missing / added / changed between two runs), baselines v0…v3 in `tests/fixtures/` are comparison snapshots, not truth.

## 7. Evidence and security
Per event: raw frame, vehicle crop, plate crop and a JSON line in the ground-truth schema (`docs/EVIDENCE.md`), size-capped folder. The demo binds to 127.0.0.1, validates uploaded file name/type/size, and accepts only whitelisted video sources (tests: `test_security.py`, `test_web_routes.py`).

## 8. Plate OCR (EXP-008)
Chain (`sentinel/ocr/`): best crop of the track (size × sharpness) → tight crop, ×(up to 6) upscale, CLAHE, two-line split → PaddleOCR PP-OCRv5-mobile *recognition* → Vietnamese plate normalisation (position-based repair of O/0, B/8, S/5 …; series letters limited to those that exist) → vote over the 3 best crops. Result goes to `plate_pred` in `events.jsonl`.
- On the test clip it **read 0 of 14 plates** (no valid-format string); the saved crops are ~46×44 px including a 10 px margin, the plate itself ~25×23 px, blurred and compressed. Mean OCR latency 209 ms per event (3 reads, CPU).
- A synthetic probe (`tools/ocr_resolution_probe.py`) shows the chain reads a clean 2-line plate down to ~38 px height, but at 25 px a little blur or JPEG already breaks it: the footage, not the code, is the limit. No accuracy is claimed (no plate ground truth).
- Next: 1080p footage with plate ≥ 40–50 px high, labelled `plate_text`, then exact-match/CER per condition.

## 9. Limitations (honest list)
1. No event-level ground truth; no P/R/F1 anywhere. Tools for recording and labelling exist (`docs/RECORDING_GUIDE.md`, `docs/LABELING_GUIDE.md`) but no data.
2. One 26 s clip, daytime, one camera; dev/test not separated; tracker/rule parameters were never tuned on separate data.
3. Red-light: legacy rule only; the clip has no red-light event and the light is detected in 14 detections. The stop-line rule needs the camera geometry (tool exists) and video.
4. Detector metrics saturated; leakage unchecked; training data and notebooks partly missing (plate).
5. Edge: CPU proxy only; GPU ONNX/TensorRT/INT8 not measured; FP16 untested.
6. Licences: Ultralytics AGPL-3.0; dataset licences unknown. Plates and faces are personal data.

## 10. Reproduce
`README.md` (run), `python -m pytest` (fast suite), `python -m pytest -m slow` (perception regression), the commands under each `experiments/EXP-*/NOTES.md`.
