"""Every stop-line crossing of a video, reported or not (rule red_light = stop_line_crossing), plus a numbered contact sheet of the crossing frames.

  python tools/crossings_report.py --video test_video.mp4 --helmet-weights helmet_best.pt --out experiments/EXP-022-light-roi/crossings

Runs the CURRENT DEFAULT pipeline config (nothing written to it) and the camera config; evidence goes to <out>/<evidence-name>. Writes
  crossings.csv / crossings.json   one row per crossing: n, time (s), frame, track, lane, light, reported (yes/no), reason, bottom-centre point
  crossings_<page>.jpg             contact sheet: one numbered crop of the vehicle at the crossing frame (green = reported, yellow = not); images show people: do not commit
  summary.json                     counts, the red_light events of the run and the lane assigned to every track asked for with --track (default M_257)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--helmet-weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera-config", default=str(REPO / "configs" / "cameras" / "test_video.yaml"))
    ap.add_argument("--evidence-name", default="evidence", help="folder under --out for the evidence of this run (must be empty); the report files are regenerated")
    ap.add_argument("--track", action="append", default=["M_257"])
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
    pipe = Pipeline(Detectors(cfg), SharedState(), cfg, load_camera_config(a.camera_config))
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    rule = pipe.rules.stop_line_rule
    if rule is None or not rule.active:
        sys.exit(f"red-light rule is not active: {pipe.red_light_status()}")
    frames = {}                                                 # frame -> clean resized frame, kept only for crossing frames
    seen = 0
    near = {}                                                   # track -> closest approach to the line (largest signed distance, px; negative = before the line): frame, s, box, light
    for i, frame in enumerate(read_frames(a.video)):
        r = pipe.process(frame)
        for v in r.rules.vehicles:
            if v.id.startswith("M_"):
                x1, y1, x2, y2 = v.coords
                sd = rule.signed_distance(((x1 + x2) / 2.0, float(y2)))
                if v.id not in near or sd > near[v.id]["s"]:
                    near[v.id] = {"track": v.id, "frame": i, "s": round(sd, 1), "box": [int(c) for c in v.coords], "light": r.perception.light, "lane": None}
                near[v.id]["last_frame"] = i
        if len(rule.crossings) > seen:
            frames[i] = r.raw.copy()
            seen = len(rule.crossings)
    pipe.flush()
    for n_ in near.values():
        x1, y1, x2, y2 = n_["box"]
        lane = rule._lane_of(((x1 + x2) / 2.0, float(y2)))
        n_["lane"] = lane.name if lane else None
    tracks_info = sorted(near.values(), key=lambda d: -d["s"])
    rows = []
    for n, c in enumerate(rule.crossings, 1):
        rows.append({"n": n, "time_s": round(c["frame"] / fps, 2), "frame": c["frame"], "track": c["id"], "lane": c["lane"] or "", "light": c["light"],
                     "reported": "yes" if c["violation"] else "no", "reason": c["why"], "point_x": c["point"][0], "point_y": c["point"][1], "box": c["box"]})
    with open(out / "crossings.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        keys = ["n", "time_s", "frame", "track", "lane", "light", "reported", "reason", "point_x", "point_y"]
        w.writerow(keys)
        for r in rows:
            w.writerow([r[k] for k in keys])
    (out / "crossings.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    H, W, per_row, per_page = 230, 200, 5, 20
    vcap = cv2.VideoCapture(a.video)

    def frame_at(fi):
        if fi in frames:
            return frames[fi]
        vcap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = vcap.read()
        return cv2.resize(f, tuple(cfg.frame_size))

    def sheet(items, prefix, label_of):
        tiles = []
        for it in items:
            f = frame_at(it["frame"])
            x1, y1, x2, y2 = it["box"]
            bw, bh = x2 - x1, y2 - y1
            X1, Y1, X2, Y2 = int(max(0, x1 - 0.3 * bw)), int(max(0, y1 - 0.3 * bh)), int(min(f.shape[1], x2 + 0.3 * bw)), int(min(f.shape[0], y2 + 0.3 * bh))
            c = f[Y1:Y2, X1:X2].copy()
            col = (0, 255, 0) if it.get("reported") == "yes" else (0, 255, 255)
            cv2.rectangle(c, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), col, 2)
            px, py = it["point"]
            cv2.circle(c, (int(px) - X1, int(py) - Y1), 4, (0, 0, 255), -1)
            c = cv2.resize(c, (max(30, int(c.shape[1] * (H - 58) / c.shape[0])), H - 58))[:, :W]
            t = np.full((H, W, 3), 25, np.uint8)
            t[0:c.shape[0], 0:c.shape[1]] = c
            for k, txt in enumerate(label_of(it)):
                cv2.putText(t, txt, (4, H - 42 + 14 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col if k == 2 else (255, 255, 255), 1)
            tiles.append(t)
        names = []
        for p in range(0, len(tiles), per_page):
            chunk = tiles[p:p + per_page]
            chunk += [np.zeros_like(tiles[0])] * (-len(chunk) % per_row)
            img = np.vstack([np.hstack(chunk[k:k + per_row]) for k in range(0, len(chunk), per_row)])
            name = f"{prefix}_{p // per_page + 1}.jpg"
            cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            names.append(name)
        return names

    for r in rows:
        r["point"] = [r["point_x"], r["point_y"]]
    pages = sheet(rows, "crossings", lambda r: [f"#{r['n']}  {r['time_s']:.2f}s  {r['track']}", f"lane {r['lane'] or '-'}  light {r['light']}", f"reported: {r['reported']}"])
    # context when few or no crossings: the tracks that came closest to the line (NOT crossings), numbered by closeness
    close = [dict(d, point=[(d["box"][0] + d["box"][2]) / 2, d["box"][3]], n=k + 1) for k, d in enumerate(tracks_info[:20])]
    approach_pages = sheet(close, "closest_approach", lambda d: [f"#{d['n']}  {d['frame'] / fps:.2f}s  {d['track']}", f"s={d['s']} px  lane {d['lane'] or '-'}", f"light {d['light']} (no crossing)"]) if close else []
    events = {}
    ev_file = ev_dir / "events.jsonl"
    if ev_file.exists():
        for line in ev_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                events[e["id"]] = e
    red = [e for e in events.values() if e["type"] == "red_light" and e.get("status") != "evicted"]
    (out / "closest_approach.json").write_text(json.dumps(tracks_info, indent=1), encoding="utf-8")
    summary = {"motorcycle_tracks": len(near), "tracks_closest_approach_px": {"within_20": sum(1 for d in near.values() if d["s"] >= -20), "within_60": sum(1 for d in near.values() if d["s"] >= -60),
                                                                          "within_100": sum(1 for d in near.values() if d["s"] >= -100), "median": sorted(d["s"] for d in near.values())[len(near) // 2]},
               "crossings": len(rows), "reported": sum(r["reported"] == "yes" for r in rows), "unassigned_crossings": rule.unassigned_crossings,
               "by_lane": {k: sum(1 for r in rows if r["lane"] == k) for k in sorted({r["lane"] for r in rows})},
               "by_light": {k: sum(1 for r in rows if r["light"] == k) for k in sorted({r["light"] for r in rows})},
               "red_light_events": [{"track": e["track_id"], "frames": [e["frame_start"], e["frame_end"]]} for e in red], "pages": pages, "closest_approach_pages": approach_pages,
               "closest_approach_of_asked": {t: near.get(t, "track not seen") for t in a.track},
               "tracks_asked": {t: [{k: r[k] for k in ("n", "time_s", "lane", "light", "reported", "reason")} for r in rows if r["track"] == t] or "no crossing of the stop line seen" for t in a.track}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
