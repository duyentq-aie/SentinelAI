"""Strips of the three signal ROIs of test_video (PROPOSAL, not in the camera config): the countdown (light_roi), the left-turn arrow (two candidate housings) and the right-turn arrow, every 0.5 s,
state under each crop (sentinel.lights.detect_traffic_light_color = the classifier the pipeline uses) with the red / green pixel counts.
Writes lane_signals_<page>.jpg (10 times per page, people may be visible in the margin: gitignored), rois_context.jpg, lane_signals.csv (every frame, 3 states) and lane_signals_summary.json
(per ROI: the runs of equal state).  Usage: python experiments/EXP-022-light-roi/lane_signals/make_strips.py"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sentinel.lights import detect_traffic_light_color  # noqa: E402

OUT = Path(__file__).parent
ROIS = {"counter (light_roi)": [590, 0, 632, 22], "left arrow": [466, 0, 530, 16], "right arrow": [845, 3, 898, 16], "left arrow, 2nd housing (alternative)": [684, 0, 748, 15]}      # the two arrow ROIs are PROPOSALS
CELL_W, MARGIN, PER_PAGE = 190, 6, 10


def counts(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])) + cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
    return int(cv2.countNonZero(red)), int(cv2.countNonZero(green))


def cell(frame, roi):
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    state = detect_traffic_light_color(crop)
    r, g = counts(crop)
    X1, Y1, X2, Y2 = max(0, x1 - MARGIN), max(0, y1 - MARGIN), min(frame.shape[1], x2 + MARGIN), min(frame.shape[0], y2 + MARGIN)
    big = frame[Y1:Y2, X1:X2].copy()
    s = CELL_W / big.shape[1]
    big = cv2.resize(big, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(big, (int((x1 - X1) * s), int((y1 - Y1) * s)), (int((x2 - X1) * s) - 1, int((y2 - Y1) * s) - 1), (0, 255, 255), 1)
    return big, state, r, g


def main():
    cap = cv2.VideoCapture(str(REPO / "test_video.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = int(round(fps / 2))                                           # every 0.5 s
    rows, strips, states = [], [], {k: [] for k in ROIS}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (1280, 720))
        row = {"frame": i, "time_s": round(i / fps, 3)}
        cells = {}
        for name, roi in ROIS.items():
            big, st, r, g = cell(frame, roi)
            row[name] = st
            states[name].append((i, st))
            if i % step == 0:
                cells[name] = (big, st, r, g)
        rows.append(row)
        if i % step == 0:
            strips.append((i, cells))
        if i == 0:
            ctx = frame.copy()
            for name, (x1, y1, x2, y2) in ROIS.items():
                cv2.rectangle(ctx, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cv2.putText(ctx, name, (x1 - 20, 40 if "arrow" in name else 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imwrite(str(OUT / "rois_context.jpg"), cv2.resize(ctx[0:90, 400:960], None, fx=2.2, fy=2.2, interpolation=cv2.INTER_NEAREST), [cv2.IMWRITE_JPEG_QUALITY, 92])
        i += 1
    names = list(ROIS)
    row_h = {n: max(c[n][0].shape[0] for _, c in strips) for n in names}
    for page, p in enumerate(range(0, len(strips), PER_PAGE), 1):
        chunk = strips[p:p + PER_PAGE]
        rows_img = []
        for n in names:
            parts = []
            for fi, c in chunk:
                big, st, r, g = c[n]
                t = np.full((row_h[n] + 50, CELL_W, 3), 25, np.uint8)
                t[:big.shape[0]] = big
                col = (0, 0, 255) if st == "RED" else ((0, 255, 0) if st == "GREEN" else (200, 200, 200))
                cv2.putText(t, st, (4, row_h[n] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                cv2.putText(t, f"red {r}  green {g} px", (4, row_h[n] + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                parts.append(t)
            rows_img.append(np.hstack(parts))
        head = np.full((26, rows_img[0].shape[1], 3), 0, np.uint8)
        for k, (fi, _) in enumerate(chunk):
            cv2.putText(head, f"{fi / fps:4.1f}s (f{fi})", (k * CELL_W + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        img = np.vstack([head] + [np.vstack([np.full((22, rows_img[0].shape[1], 3), 60, np.uint8), r]) for r in rows_img])
        for k, n in enumerate(names):
            y = 26 + sum(row_h[m] + 50 + 22 for m in names[:k]) + 16
            cv2.putText(img, n + ("   [PROPOSED ROI " + str(ROIS[n]) + "]" if "arrow" in n else f"   [light_roi {ROIS[n]}]"), (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imwrite(str(OUT / f"lane_signals_{page}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(OUT / "lane_signals.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "time_s"] + names, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    summary = {"rois": ROIS, "frames": len(rows), "fps": fps, "runs": {}}
    for n in names:
        runs, cur = [], None
        for f, st in states[n]:
            if cur is None or cur["state"] != st:
                cur = {"state": st, "first_frame": f, "last_frame": f}
                runs.append(cur)
            cur["last_frame"] = f
        for r in runs:
            r["from_s"], r["to_s"] = round(r["first_frame"] / fps, 2), round(r["last_frame"] / fps, 2)
        summary["runs"][n] = runs
    (OUT / "lane_signals_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["runs"], indent=1))


if __name__ == "__main__":
    main()
