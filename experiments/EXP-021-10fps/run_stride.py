"""One pass of test_video.mp4 processing every k-th frame (default 3: 30 -> 10 FPS) with the default pipeline config and fps = source fps / k.
Writes outputs/review_10fps/{evidence/, characterization_events.json}; event frames are stored as indices of the 10-FPS stream AND mapped back to original
video frames (x k). Usage: python experiments/EXP-021-10fps/run_stride.py [--stride 3] [--out outputs/review_10fps]"""
import argparse
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sentinel.config import load_camera_config, load_pipeline_config  # noqa: E402
from sentinel.pipeline import Detectors, Pipeline, SharedState  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--out", default=str(REPO / "outputs" / "review_10fps"))
ap.add_argument("--video", default=str(REPO / "test_video.mp4"))
a = ap.parse_args()
out = Path(a.out)
if out.exists() and any(out.iterdir()):
    sys.exit(f"{out} is not empty: nothing is overwritten or deleted")
(out / "evidence").mkdir(parents=True, exist_ok=True)
cfg = load_pipeline_config(root=REPO)
cfg.evidence.dir = str(out / "evidence")
cap = cv2.VideoCapture(a.video)
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
pipe = Pipeline(Detectors(cfg), SharedState(), cfg, load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml"), fps=src_fps / a.stride)
chars, first, plate_calls, n = [], {}, 0, 0
raw_i = -1
while True:
    raw_i += 1
    if raw_i % a.stride:
        if not cap.grab():
            break
        continue
    ok, frame = cap.read()
    if not ok:
        break
    r = pipe.process(frame)
    plate_calls += r.plate_calls
    n += 1
    for v in r.captures:
        chars.append({"frame": raw_i, "kind": "plate_capture", "track": v})          # original video frame index
    for lab in r.labels:
        first.setdefault(lab, raw_i)
pipe.flush()
for lab, fi in sorted(first.items(), key=lambda x: x[1]):
    chars.append({"frame": fi, "kind": "label_first_seen", "label": lab})
(out / "characterization_events.json").write_text(json.dumps({"impl": f"default config, every {a.stride}-th frame, fps {src_fps / a.stride:g}", "stride": a.stride, "events": chars}, indent=1), encoding="utf-8")
(out / "run_info.json").write_text(json.dumps({"stride": a.stride, "fps": src_fps / a.stride, "frames_processed": n, "plate_detector_calls": plate_calls}, indent=1), encoding="utf-8")
print(f"stride {a.stride}: {n} frames processed, {plate_calls} plate-detector calls -> {out}")
