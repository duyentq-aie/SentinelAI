# SentinelAI – traffic-violation detection on video (portfolio project)

Detects **no helmet**, **motorcycle carrying more than 2 people** and (legacy zone rule, not yet validated) **red-light running** in traffic video,
tracks vehicles, saves **evidence** (raw frame, vehicle crop, plate crop, JSON record) per event and can read the **plate text** (OCR).
A small FastAPI demo streams the annotated video and lists the events.

> **Read this first – what is and is not proven**
> - There is **no ground truth for violation events yet** (no labelled video). No precision/recall of the *system* is claimed anywhere.
>   The detector mAP of ~0.99 (val/test of the original Kaggle datasets) is **saturated and its splits are not verified independent** (frames from video): do not read it as generalisation.
> - Everything measured here is **relative** (before/after on this machine: RTX 3050 Laptop 4 GB, i5-11400H) on **one 26 s clip** (`test_video.mp4`). Absolute FPS depends on the machine state (the original prototype alone measured 7.4–8.6 FPS in different sessions, and about 40% less when half of the CPU is busy with other work, EXP-009): every speed claim is therefore given as a ratio to the original prototype (`v0`) measured in the same session.
> - GPU/Edge numbers are **proxies**: no Jetson/RPi was available. ONNX Runtime GPU and TensorRT were **not** measured (CUDA 12 / TensorRT not installed).
> - The red-light rule is still the v0 zone rule (a stopped car inside the zone on red is flagged). The stop-line + direction rule is planned, not built; the test clip has no red-light event at all.

Full detail: [docs/REPORT.md](docs/REPORT.md) · audit of the starting point: [docs/AUDIT.md](docs/AUDIT.md) · plan: [docs/PLAN.md](docs/PLAN.md) · night-run log: [docs/PROGRESS.md](docs/PROGRESS.md).

## What is in the repo
```
sentinel/            core package (no web dependency)
  pipeline.py        frame -> perceive (YOLO + tracker) -> rules -> draw -> evidence/plates
  rules/             legacy (v0 counters) and windowed (voting) no-helmet / overload rules; pure functions of tracks
  evidence.py        events, evidence files, plate search policy, events.jsonl (schema: docs/EVIDENCE.md)
  ocr/               plate crop quality, OpenCV preprocessing, VN plate normalisation, voting, PaddleOCR recogniser
  config.py          typed config loaded from configs/*.yaml (unknown keys are errors)
  security.py        upload / video-source whitelist for the demo
apps/web/            FastAPI demo (routes, inference worker, server wiring); templates/index.html
configs/             pipeline (default, ocr, onnx_cpu, ...), cameras, trackers, experiment variants (exp/)
tools/               benchmark_pipeline, record_tracks, diff_events, diff_tracks, export_onnx, compare_backends,
                     camera_config (click to draw stop line / ROI), label_events (event labelling GUI)
tests/               ~210 fast tests (rules on synthetic tracks, evidence, security, web routes, OCR pieces ...) + 1 slow
experiments/         every number in the report points to an EXP-xxx folder (notes + metrics.json + env.json)
docs/                AUDIT, PLAN, REPORT, EVIDENCE schema, RECORDING_GUIDE, LABELING_GUIDE, PROGRESS
```
The original single-file prototype is kept at git tag `v0-baseline`.

## Run the demo
```powershell
python -m venv venv; .\venv\Scripts\activate
pip install -r requirements.txt            # torch cu118 wheels: see the comment at the top of that file
# model weights are not in git (*.pt): put helmet_best.pt, plate_best.pt, yolo11s.pt in the repo root; test_video.mp4 too
python app.py                              # http://127.0.0.1:8000/   (loopback only by default; no auto-reload)
$env:SENTINEL_CONFIG = "configs\pipeline\ocr.yaml"; python app.py     # with plate OCR (first run downloads the PaddleOCR model)
```
The page shows the live annotated stream with an **FPS / latency / backend** box, the **event panel** (type, time, raw frame, vehicle, plate crop, OCR text) and plate crops.
Video sources are restricted: files inside `static/uploads` or the repo root, `rtsp://` and camera indexes only if enabled in `configs/pipeline/default.yaml` (`web:`). Uploads are limited in type and size.

### Docker (CPU, ONNX Runtime) - written, **not yet built or run**
`Dockerfile`, `requirements-docker.txt`, `.dockerignore`, `configs/pipeline/docker_cpu.yaml`. Weights, ONNX files and videos are mounted, never baked in. On the dev machine the Docker engine was not running and C: had < 6 GB free, so the image was not built and the demo was not run in a container (nothing was installed to work around it); `tests/test_docker_files.py` only checks the files statically.
```powershell
python tools/export_onnx.py --models coco helmet --imgsz 384 640; python tools/export_onnx.py --models plate --imgsz 640 --dynamic
docker build -t sentinel-cpu .
docker run --rm -p 127.0.0.1:8000:8000 -v ${PWD}\models:/app/models:ro -v ${PWD}	est_video.mp4:/app/test_video.mp4:ro sentinel-cpu
```

## Tests and experiments
```powershell
python -m pytest                           # fast suite (~20 s, no GPU, no models needed)
python -m pytest -m slow                   # re-runs detectors+trackers on CPU and checks perception == tests/fixtures/tracks_test_video.jsonl (5-15 min)
python tools/benchmark_pipeline.py --repeats 3 --configs configs/pipeline/exp/p4b_roi_jpeg80.yaml configs/pipeline/exp/p4c_gmc_none.yaml --out experiments/my_run
python tools/diff_events.py tests/fixtures/baseline_events_v3_windowed.json <events.json>   # what changed between two runs
```
Configs are interleaved by the benchmark tool (round 1: every config, round 2 ...) so drift of the machine hits all of them equally; median and range of 3 runs are reported.

## Data, models, licences
- Models: YOLO11s (COCO, vehicles + traffic lights), YOLO11s fine-tuned for helmet / motorcycle / no-helmet, YOLO11n fine-tuned for licence plates (Kaggle runs; see `experiments/EXP-000-kaggle-original`).
  The notebook for the plate model does **not** match the deployed `plate_best.pt` (documented there).
- **Ultralytics is AGPL-3.0**; this project is a portfolio piece and accepts that (derived code and weights follow AGPL-3.0). Dataset licences: both Roboflow exports *declare* CC BY 4.0 in `data.yaml`, **not independently verified (chưa xác minh)**; see `docs/DATA_CARD.md` (the helmet split leaks: random frame split of 16 videos) – check before any redistribution of data or weights. See `LICENSE` (AGPL-3.0) and `docs/PUBLISH_CHECKLIST.md`.
- Plate text, plate crops and vehicle video are personal data: raw videos and evidence stay out of git (`*.mp4`, `*.pt`, `data/`, `static/violations/` are ignored); blur plates and faces before publishing any image or clip (`docs/RECORDING_GUIDE.md`).

## Direction (2026-10-19)
- **Target: AI on the camera** (assumed Rockchip RK3588, with RV1126 as the harsh case; no hardware, device speeds are sourced estimates only): >= 10 FPS for the helmet rule with a single detector, see PLAN G7-G12.
- **M1 scope for `red_light`: motorcycles only**; the light state is read from a fixed ROI with OpenCV. **Cars are future work**: auto-label a `car` class on the helmet set with the COCO model (then review by hand) and train a 5-class model (helmet, motorcycle, no-helmet, plate, car).
- **Pipeline settings are durations, not frame counts**: windows, close delays, plate-search budget ... are in seconds or ratios and become frame counts from the frame rate the pipeline sees (`docs/TIME_BASED_PARAMETERS.md`); old configs written in frames still load (with a warning, `tools/convert_config_to_seconds.py` converts them).

## Roadmap (not done)
Record + label dev/test videos (`docs/RECORDING_GUIDE.md`, `docs/LABELING_GUIDE.md`) → event-level P/R/F1 (`tools/eval_events.py` is not written yet) → stop-line red-light rule (`tools/camera_config.py` already draws the geometry) →
independent test split for the detectors (leakage check) → GPU/TensorRT/INT8 on real Edge hardware.
