# Evidence & event schema

Implemented by `sentinel/evidence.py` (P2 3.1). Config: `evidence:` in `configs/pipeline/default.yaml`.

## Folder layout (`evidence.dir`, default `static/violations/`)
```
events.jsonl                       append-only log (see below)
<event_id>/frame.jpg               raw frame at the trigger: resized, never drawn on
<event_id>/vehicle.jpg             crop of the offending vehicle at the trigger
<event_id>/plate.jpg               plate crop (raw pixels + plate_margin); only if a plate was found
```
`event_id = <video_id>-r<run>-e<nnnn>`; `run` increases every time the pipeline is reset (new source / video loop), because track ids restart.
An event is **one violation type on one track**: `OVERLOAD & NO-HELMET` opens two events (`overload`, `no_helmet`).
Events are written **even if no plate is found** (`plate.found = false`, `evidence.plate = null`).

## `events.jsonl`
One JSON object per line. An event writes a record with `status: "open"` when it starts (and again when its plate is found),
and `status: "closed"` when the violation has been absent for `close_after` frames or the pipeline resets/flushes.
**Take the last record per `id`.** When `max_dir_mb` is exceeded the oldest event folders are deleted and `{"id": ..., "status": "evicted"}` is appended
(events of the current run that are still open are never evicted). `events.jsonl` itself is never deleted.

Fields share the **ground-truth schema of PLAN section 3.1**, so `tools/eval_events.py` can read predictions and labels with one reader:

| field | GT label | prediction | meaning |
|---|---|---|---|
| `id` | yes | yes | event id |
| `type` | yes | yes | `no_helmet` \| `overload` \| `red_light` (legacy rules also: `wrong_way`, `wrong_lane`) |
| `frame_start`, `frame_end` | yes | yes | first / last frame the violation was flagged (frame index inside the run) |
| `key_frame`, `key_box` | yes | yes | frame of the evidence snapshot and the vehicle box `[x1,y1,x2,y2]` there (resized-frame pixels) |
| `plate_text` | yes | `null` | ground-truth plate string (predictions leave it `null`; the OCR result goes to `plate_pred`) |
| `difficulty`, `notes` | yes | `null`, `""` | annotation fields |
| `schema`, `status` | | yes | `sentinel.event/1`; `open` \| `closed` |
| `frame_size` | | yes | `[w, h]` of the frame the pipeline worked on (1280x720 by default): `key_box`, `boxes`, `plate.box` are in these pixels. Ground-truth labels use the video's native pixels (`width`/`height` of the labels file); the evaluator rescales |
| `track_id`, `trigger_frame` | | yes | tracker id (`M_77` motorcycle from the helmet model, `C_221` car/truck/bus from COCO) and the frame that triggered the event (= `frame_start`) |
| `confidence`, `plate_pred` | | `null` for now | reserved for rule confidence / OCR result |
| `boxes` | | yes | vehicle box sampled every 5 frames while the event is active (`{"frame": box}`, max 60) |
| `plate` | | yes | `{"found": true, "frame", "box" (raw px), "conf", "crop_size": [w,h], "attempts", "file"}` or `{"found": false, "attempts": n}` |
| `evidence` | | yes | paths of `frame` / `vehicle` / `plate`, **relative to the evidence folder** (`null` if absent) |
| `vehicle` | | yes | `{"cls", "box_size": [w,h]}` |
| `light`, `video_id`, `camera_id`, `run`, `created_at` | | yes | traffic-light state at the trigger and provenance |

## Plate search policy (`evidence.plate`)
Run the plate detector only for tracks that are violating **right now**, at most `max_attempts` times per track (0 = unlimited), every `every`-th frame,
and stop at the first plate (`stop_when_found`). A plate found for a track is reused by every later event of the same track.
`max_attempts: 0` + `stop_when_found: false` reproduces the v0 behaviour (detector on every violating frame) — config `configs/pipeline/ab_plate_v0.yaml`,
used as the A/B reference in `experiments/EXP-003-evidence/`.

## Known limits
- Plate text is not read yet (P3); crops average ~54x43 px at 1280x720, which may be too small for OCR.
- The `PLATE` overlay is only drawn on frames where the detector ran (it no longer runs on every frame).
- Track ids are device-specific; events from a GPU run and a CPU run of the same video are not byte-identical.
