"""Motorcycle-track stability of the helmet-model tracker (default config: BoT-SORT, GMC off) at 30 FPS and at 10 FPS (every 3rd frame).
Only numbers, nothing is changed. Variants:
  30fps            every frame, tracker buffer 30 frames (1 s)
  10fps            every 3rd frame, buffer rescaled to the same duration (10 frames) - what Pipeline(fps=10) does
  10fps_buffer30   every 3rd frame, buffer left at 30 frames (= 3 s at 10 FPS): separates 'buffer too long' from 'matching worse'
Metrics per variant (motorcycle tracks of the helmet model): unique tracks, median track duration (s), tracks shorter than 1 s, estimated ID changes =
an ended track (not ending at the last frame) followed by a NEW track whose first box centre lies within `0.5 x max(w, h)` of the old track's last centre,
starting within 0.5 s after the old one ended (greedy one-to-one).
Usage: python experiments/EXP-021-10fps/tracking_stability.py"""
import json
import statistics as st
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from sentinel.config import load_pipeline_config  # noqa: E402
from sentinel.pipeline import Detectors  # noqa: E402

VIDEO = REPO / "test_video.mp4"


def centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def run(stride, rescale_buffer):
    cfg = load_pipeline_config(root=REPO)
    det = Detectors(cfg)
    cap0 = cv2.VideoCapture(str(VIDEO))
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    cap0.release()
    if rescale_buffer:
        det.set_fps(fps / stride)
    cap = cv2.VideoCapture(str(VIDEO))
    tracks, i = {}, -1                                             # id -> [(original frame, box)]
    while True:
        i += 1
        if i % stride:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        for t in det.track_helmet(cv2.resize(frame, tuple(cfg.frame_size))):
            if t.cls == "motorcycle":
                tracks.setdefault(t.id, []).append((i, t.box))
    return tracks, fps, n_total


def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


def step_stats(tracks, stride):
    """Between consecutive processed frames of the same track: box IoU and centre displacement (px): how far an object moves between two frames the tracker has to link."""
    ious, disp = [], []
    for v in tracks.values():
        for (f0, b0), (f1, b1) in zip(v, v[1:]):
            if f1 - f0 == stride:
                ious.append(iou(b0, b1))
                (x0, y0), (x1, y1) = centre(b0), centre(b1)
                disp.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
    return {"steps": len(ious), "median_step_iou": round(st.median(ious), 3), "p10_step_iou": round(q(ious, 0.10), 3), "pct_steps_iou_below_0.3": round(100 * sum(1 for x in ious if x < 0.3) / len(ious), 1),
            "median_step_displacement_px": round(st.median(disp), 1), "p90_step_displacement_px": round(q(disp, 0.90), 1)}


def metrics(tracks, fps, n_total, stride):
    dur = {k: (v[-1][0] - v[0][0] + stride) / fps for k, v in tracks.items()}              # seconds covered by the track (first to last detection, inclusive of one step)
    last_frame = max(v[-1][0] for v in tracks.values())
    ended = sorted(((v[-1][0], k) for k, v in tracks.items() if v[-1][0] < last_frame - 2 * stride))
    starts = sorted(((v[0][0], k) for k, v in tracks.items()))
    used, changes = set(), []
    for end_f, ek in ended:
        ex, ey = centre(tracks[ek][-1][1])
        b = tracks[ek][-1][1]
        radius = 0.5 * max(b[2] - b[0], b[3] - b[1])
        best = None
        for sf, sk in starts:
            if sk in used or sk == ek or sf <= end_f or sf > end_f + 0.5 * fps:
                continue
            sx, sy = centre(tracks[sk][0][1])
            d = ((sx - ex) ** 2 + (sy - ey) ** 2) ** 0.5
            if d <= radius and (best is None or d < best[0]):
                best = (d, sk)
        if best:
            used.add(best[1])
            changes.append((ek, best[1]))
    secs = n_total / fps
    return {"unique_tracks": len(tracks), "median_track_duration_s": round(st.median(dur.values()), 2), "mean_track_duration_s": round(st.mean(dur.values()), 2),
            "tracks_shorter_than_1s": sum(1 for d in dur.values() if d < 1.0), "pct_shorter_than_1s": round(100 * sum(1 for d in dur.values() if d < 1.0) / len(dur), 1),
            "estimated_id_changes": len(changes), "id_changes_per_10s_of_video": round(10 * len(changes) / secs, 2),
            **step_stats(tracks, stride), "detections": sum(len(v) for v in tracks.values()), "frames_processed": len({f for v in tracks.values() for f, _ in v})}


out = {}
for name, stride, rescale in (("30fps", 1, False), ("10fps", 3, True), ("10fps_buffer30", 3, False)):
    tr, fps, n = run(stride, rescale)
    out[name] = metrics(tr, fps, n, stride)
    print(name, json.dumps(out[name]))
(Path(__file__).parent / "tracking_stability.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
