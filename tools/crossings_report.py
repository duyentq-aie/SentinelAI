"""Every stop-line crossing of a video, reported or not (rule red_light = stop_line_crossing, source: COCO motorcycle tracks K_<n>), with numbered contact sheets.

  python tools/crossings_report.py --video test_video.mp4 --helmet-weights helmet_best.pt --out experiments/EXP-022-light-roi/crossings_k

Runs the CURRENT DEFAULT pipeline config (nothing written to it) and the camera config; evidence goes to <out>/<evidence-name> (must be empty). Writes
  crossings.csv / crossings.json   one row per crossing: n, time (s), frame, track, lane, light, reported (yes/no), reason, merged_into, crossing point
  crossings_<page>.jpg             contact sheet, one tile per crossing: vehicle crop at the crossing frame + the whole frame with the stop line, the lanes (cyan = roi, magenta = exempt)
                                   and the crossing point; green frame = reported, yellow = not reported (images show people: do not commit)
  red_in_<lane>.jpg                the crossings on RED in the lane given with --check-lane (default right-turn), larger, for a check by eye
  closest_approach.json, summary.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TH_W, TH_H = 340, 191                     # whole-frame thumbnail in a tile
CROP_W, TILE_H = 200, 230


def overlay(frame, camera, point, scale):
    """Whole frame with the stop line, the lanes and the crossing point, scaled."""
    f = frame.copy()
    for lane in camera.lanes or []:
        col = (255, 0, 255) if lane.kind == "exempt" else (255, 200, 0)
        cv2.polylines(f, [np.array(lane.polygon, np.int32).reshape(-1, 1, 2)], True, col, 3)
    (ax, ay), (bx, by) = camera.stop_line
    cv2.line(f, (int(ax), int(ay)), (int(bx), int(by)), (0, 0, 255), 4)
    cv2.circle(f, (int(point[0]), int(point[1])), 12, (0, 255, 255), 4)
    return cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def crop_of(frame, box, point, height):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    X1, Y1, X2, Y2 = int(max(0, x1 - 0.3 * bw)), int(max(0, y1 - 0.3 * bh)), int(min(frame.shape[1], x2 + 0.3 * bw)), int(min(frame.shape[0], y2 + 0.3 * bh))
    c = frame[Y1:Y2, X1:X2].copy()
    cv2.rectangle(c, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), (0, 255, 255), 2)
    cv2.circle(c, (int(point[0]) - X1, int(point[1]) - Y1), 4, (0, 0, 255), -1)
    return cv2.resize(c, (max(30, int(c.shape[1] * height / c.shape[0])), height))


def tile(frame, camera, r, big=False):
    k = 2 if big else 1
    crop = crop_of(frame, r["box"], r["point"], (TILE_H - 58) * k)[:, :CROP_W * k]
    th = overlay(frame, camera, r["point"], (TH_W * k) / frame.shape[1])
    w = CROP_W * k + th.shape[1]
    t = np.full((TILE_H * k, w, 3), 25, np.uint8)
    t[0:crop.shape[0], 0:crop.shape[1]] = crop
    t[0:th.shape[0], CROP_W * k:] = th
    col = (0, 255, 0) if r["reported"] == "yes" else (0, 255, 255)
    cv2.rectangle(t, (0, 0), (w - 1, TILE_H * k - 1), col, 3)
    for i, txt in enumerate([f"#{r['n']}  {r['time_s']:.2f}s  {r['track']}", f"lane {r['lane'] or '-'}   light {r['light']}   reported: {r['reported']}"]):
        cv2.putText(t, txt, (6, (TILE_H - 36 + 20 * i) * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, (255, 255, 255), k)
    return t


def sheet(tiles, out, prefix, per_row, per_page):
    names = []
    for p in range(0, len(tiles), per_page):
        chunk = tiles[p:p + per_page]
        chunk += [np.zeros_like(tiles[0])] * (-len(chunk) % per_row)
        img = np.vstack([np.hstack(chunk[k:k + per_row]) for k in range(0, len(chunk), per_row)])
        name = f"{prefix}_{p // per_page + 1}.jpg" if len(tiles) > per_page else f"{prefix}.jpg"
        cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        names.append(name)
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--helmet-weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera-config", default=str(REPO / "configs" / "cameras" / "test_video.yaml"))
    ap.add_argument("--evidence-name", default="evidence", help="folder under --out for the evidence of this run (must be empty); the report files are regenerated")
    ap.add_argument("--check-lane", default="right-turn", help="lane whose RED crossings get their own large sheet")
    a = ap.parse_args()

    from sentinel.characterize import read_frames
    from sentinel.config import load_camera_config, load_pipeline_config
    from sentinel.pipeline import Detectors, Pipeline, SharedState

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ev_dir = out / a.evidence_name
    if ev_dir.exists() and any(ev_dir.iterdir()):
        sys.exit(f"{ev_dir} is not empty: nothing is overwritten or deleted (choose another --evidence-name)")
    cfg = load_pipeline_config(root=REPO)
    cfg.models.helmet.weights = a.helmet_weights
    cfg.evidence.dir = str(ev_dir)
    camera = load_camera_config(a.camera_config)
    pipe = Pipeline(Detectors(cfg), SharedState(), cfg, camera)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    rule = pipe.rules.stop_line_rule
    if rule is None or not rule.active:
        sys.exit(f"red-light rule is not active: {pipe.red_light_status()}")
    frames, seen, near = {}, 0, {}              # crossing frames (clean resized); closest approach per K_ track: largest signed distance (negative = before the line)
    for i, frame in enumerate(read_frames(a.video)):
        r = pipe.process(frame)
        for t in r.perception.coco:
            if t.cls == "motorcycle":
                x1, y1, x2, y2 = t.box
                sd = rule.signed_distance(((x1 + x2) / 2.0, float(y2)))
                tid = f"K_{t.id}"
                if tid not in near or sd > near[tid]["s"]:
                    near[tid] = {"track": tid, "frame": i, "s": round(sd, 1), "box": [int(c) for c in t.box], "light": r.perception.light}
        if len(rule.crossings) > seen:
            frames[i] = r.raw.copy()
            seen = len(rule.crossings)
    pipe.flush()
    rows = []
    for n, c in enumerate(rule.crossings, 1):
        rows.append({"n": n, "time_s": round(c["frame"] / fps, 2), "frame": c["frame"], "track": c["id"], "lane": c["lane"] or "", "light": c["light"],
                     "reported": "yes" if c["violation"] else "no", "reason": c["why"], "merged_into": c.get("merged_into") or "", "point_x": c["point"][0], "point_y": c["point"][1],
                     "point": c["point"], "box": c["box"]})
    keys = ["n", "time_s", "frame", "track", "lane", "light", "reported", "reason", "merged_into", "point_x", "point_y"]
    with open(out / "crossings.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(keys)
        for r in rows:
            w.writerow([r[k] for k in keys])
    (out / "crossings.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "point"} for r in rows], indent=1), encoding="utf-8")

    pages = sheet([tile(frames[r["frame"]], camera, r) for r in rows], out, "crossings", 2, 10) if rows else []
    check = [r for r in rows if r["lane"] == a.check_lane and r["light"] == "RED"]
    check_pages = sheet([tile(frames[r["frame"]], camera, r, big=True) for r in check], out, f"red_in_{a.check_lane}", 1, 4) if check else []
    tracks_info = sorted(near.values(), key=lambda d: -d["s"])
    for d in tracks_info:
        lane = rule._lane_of(((d["box"][0] + d["box"][2]) / 2.0, float(d["box"][3])))
        d["lane"] = lane.name if lane else None
    (out / "closest_approach.json").write_text(json.dumps(tracks_info, indent=1), encoding="utf-8")
    events = {}
    ev_file = ev_dir / "events.jsonl"
    if ev_file.exists():
        for line in ev_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                events[e["id"]] = e
    red = [e for e in events.values() if e["type"] == "red_light" and e.get("status") != "evicted"]
    summary = {"coco_motorcycle_tracks": len(near), "closest_approach_px": {"beyond_line": sum(1 for d in near.values() if d["s"] >= 4), "within_20": sum(1 for d in near.values() if d["s"] >= -20),
                                                                         "median": sorted(d["s"] for d in near.values())[len(near) // 2]},
               "crossings": len(rows), "reported": sum(r["reported"] == "yes" for r in rows), "merged": sum(1 for r in rows if r["merged_into"]), "unassigned_crossings": rule.unassigned_crossings,
               "by_lane": {k: sum(1 for r in rows if r["lane"] == k) for k in sorted({r["lane"] for r in rows})},
               "by_light": {k: sum(1 for r in rows if r["light"] == k) for k in sorted({r["light"] for r in rows})},
               "by_lane_and_light": {f"{k[0]}/{k[1]}": sum(1 for r in rows if (r["lane"], r["light"]) == k) for k in sorted({(r["lane"], r["light"]) for r in rows})},
               "red_in_check_lane": [{"n": r["n"], "time_s": r["time_s"], "track": r["track"], "reported": r["reported"]} for r in check],
               "red_light_events": [{"track": e["track_id"], "frames": [e["frame_start"], e["frame_end"]], "linked": e.get("linked")} for e in red],
               "pages": pages, "check_pages": check_pages}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    for r in rows:
        print(r["n"], r["time_s"], r["track"], r["lane"] or "-", r["light"], r["reported"], "|", r["reason"])


if __name__ == "__main__":
    main()
