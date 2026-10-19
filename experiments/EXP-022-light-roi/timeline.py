"""Traffic-light state over test_video.mp4 read two ways: from the camera light_roi (OpenCV, every frame) and from the COCO detector (recorded tracks of the default
config, tests/fixtures/tracks_test_video.jsonl). Writes light_timeline.csv (committed) and light_strip_<n>.jpg (people visible -> gitignored).
Usage: python experiments/EXP-022-light-roi/timeline.py"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from sentinel.config import load_camera_config  # noqa: E402
from sentinel.lights import detect_traffic_light_color  # noqa: E402
from sentinel.offline import load_tracks  # noqa: E402

OUT = Path(__file__).parent
cam = load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
x1, y1, x2, y2 = cam.light_roi
zone = np.array(cam.violation_zone, np.int32)
zx1, zy1, zx2, zy2 = zone[:, 0].min(), zone[:, 1].min(), zone[:, 0].max(), zone[:, 1].max()


def hsv_counts(crop):
    """Same thresholds as sentinel/lights.py, to show how decisive the call is."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])) + cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
    return int(cv2.countNonZero(red)), int(cv2.countNonZero(green))


_, per = load_tracks(REPO / "tests" / "fixtures" / "tracks_test_video.jsonl")
cap = cv2.VideoCapture(str(REPO / "test_video.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
rows, strip = [], []
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (1280, 720))
    crop = frame[y1:y2, x1:x2]
    state = detect_traffic_light_color(crop)
    r, g = hsv_counts(crop)
    p = per[i] if i < len(per) else None
    coco_det = bool(p and any(t.cls == "traffic light" for t in p.coco))
    rows.append({"frame": i, "time_s": round(i / fps, 3), "roi_state": state, "roi_red_px": r, "roi_green_px": g, "coco_state": p.light if p else "", "coco_detected": int(coco_det)})
    if i % round(0.5 * fps) == 0:
        z = frame[max(0, zy1 - 40):zy2 + 10, zx1:zx2].copy()
        cv2.polylines(z, [zone - [zx1, max(0, zy1 - 40)]], True, (0, 255, 255), 1)
        strip.append((i, state, cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST), cv2.resize(z, (264, int(264 * z.shape[0] / z.shape[1])))))
    i += 1
with open(OUT / "light_timeline.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)

W = 270
pages = []
for p0 in range(0, len(strip), 8):
    cols = []
    for fi, state, c, z in strip[p0:p0 + 8]:
        col = np.full((c.shape[0] + 60 + z.shape[0] + 24, W, 3), 25, np.uint8)
        col[:c.shape[0], :c.shape[1]] = c[:, :W]
        col[c.shape[0] + 4:c.shape[0] + 24, 4:] = 25
        cv2.putText(col, f"{fi / fps:5.1f}s  f{fi}", (4, c.shape[0] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(col, state, (4, c.shape[0] + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if state == "RED" else (0, 255, 0) if state == "GREEN" else (200, 200, 200), 2)
        col[c.shape[0] + 60:c.shape[0] + 60 + z.shape[0], 3:3 + z.shape[1]] = z
        cv2.putText(col, "violation_zone (+40px above)", (4, col.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        cols.append(col)
    h = max(c.shape[0] for c in cols)
    cols = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)) for c in cols]
    f = OUT / f"light_strip_{p0 // 8 + 1}.jpg"
    cv2.imwrite(str(f), np.hstack(cols), [cv2.IMWRITE_JPEG_QUALITY, 90])
    pages.append(f.name)

# transitions
trans, last = [], None
for r in rows:
    if r["roi_state"] != last:
        trans.append((r["frame"], r["time_s"], r["roi_state"]))
        last = r["roi_state"]
import collections
print("frames", len(rows), "| ROI states:", dict(collections.Counter(r["roi_state"] for r in rows)), "| COCO states:", dict(collections.Counter(r["coco_state"] for r in rows)))
print("ROI transitions (frame, s, new state):", trans[:20], "... total", len(trans))
print("pages:", pages)
