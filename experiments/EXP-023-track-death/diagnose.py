"""EXP-023: why do motorcycle tracks of the helmet model die before the stop line on test_video? (diagnosis only, nothing in the pipeline is changed)

Three hypotheses, each tested with numbers:
  a) a region / mask / filter cuts the upper part of the frame           -> `python diagnose.py filters` is a code audit printed by README; here: raw detections above the line prove or refute it
  b) the DETECTOR stops seeing motorcycles near the line (small, far, occluded) -> raw `motorcycle` detections per 40 px band of the box bottom y2, per phase
  c) the TRACKER drops objects the detector still sees                    -> histogram of the y2 where tracks END, and for each ended track: is there an unclaimed detection near its last box in the next frames?
Raw detections are collected without tracker at conf 0.1 (helmet_best.pt and COCO yolo11s.pt); the pipeline conf (0.55 helmet, 0.35 COCO) is a subset of them.
The tracking pass uses Detectors.track_helmet exactly like the pipeline (same conf, same tracker yaml). Phases: RED 0-9.3 s (frames 0-279), then GREEN.
Usage: python experiments/EXP-023-track-death/diagnose.py [--strip]      (writes raw_detections.json, results.json, m257_strip.jpg (gitignored: people are visible))
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from sentinel.characterize import read_frames  # noqa: E402
from sentinel.config import load_camera_config, load_pipeline_config  # noqa: E402
from sentinel.pipeline import Detectors  # noqa: E402

OUT = Path(__file__).parent
VIDEO = REPO / "test_video.mp4"
LINE_Y = 290                      # the stop line is at y ~ 277..302 (stop_line [[400,302],[1040,277]]): "above the line" = y2 < 290
BAND = 40
SWITCH = 280                      # first GREEN frame (EXP-022 timeline)
CONF_LOW = 0.1


def raw_pass(det, cfg):
    """-> {"helmet": [[[x1,y1,x2,y2,conf], ...] per frame], "coco": [...]} motorcycle detections at conf 0.1, no tracker; and the tracked helmet motorcycles at the pipeline conf."""
    raw = {"helmet": [], "coco": []}
    tracks = []
    det.helmet.predictor = None
    for frame in read_frames(str(VIDEO)):
        f = cv2.resize(frame, tuple(cfg.frame_size))
        for key, model in (("helmet", det.helmet), ("coco", det.coco)):
            r = model.predict(f, conf=CONF_LOW, verbose=False, **det.kw)[0]
            names = model.names
            rows = []
            for xyxy, c, k in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(), r.boxes.cls.cpu().tolist()):
                if names[int(k)] == "motorcycle":
                    rows.append([round(v, 1) for v in xyxy] + [round(c, 3)])
            raw[key].append(rows)
    det.helmet.predictor = None
    det.coco.predictor = None
    for frame in read_frames(str(VIDEO)):
        f = cv2.resize(frame, tuple(cfg.frame_size))
        tracks.append([[t.id, *t.box, round(float(t.conf), 3)] for t in det.track_helmet(f) if t.cls == "motorcycle"])
    return raw, tracks


def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


def band_counts(frames, conf_min, lo_frame, hi_frame):
    """detections per frame in each 40 px band of y2 (+ share of frames that have >= 1 detection above the line), for frames [lo, hi)."""
    n = max(1, min(hi_frame, len(frames)) - lo_frame)
    bands = {}
    above_frames = 0
    above = 0
    for fr in frames[lo_frame:hi_frame]:
        has = False
        for d in fr:
            if d[4] >= conf_min:
                b = int(d[3] // BAND) * BAND
                bands[b] = bands.get(b, 0) + 1
                if d[3] < LINE_Y:
                    above += 1
                    has = True
        above_frames += has
    return {"frames": n, "per_band_per_frame": {f"{b}-{b + BAND}": round(v / n, 3) for b, v in sorted(bands.items())}, "dets_above_line": above, "frames_with_det_above_line": above_frames,
            "dets_total": sum(bands.values())}


def track_ends(tracks, raw_helmet_pipeline_conf, conf):
    """Per ended track: y2 of its last box, and whether an unclaimed raw detection (conf >= pipeline conf) overlaps the last box (IoU >= 0.3) in the next 3 frames."""
    last = {}
    first = {}
    for i, fr in enumerate(tracks):
        for t in fr:
            last[t[0]] = (i, t[1:5])
            first.setdefault(t[0], i)
    n = len(tracks)
    ends = []
    for tid, (i, box) in last.items():
        if i >= n - 3:
            continue
        claimed_later = False
        near = None
        for j in range(i + 1, min(n, i + 4)):
            for d in raw_helmet_pipeline_conf[j]:
                if d[4] < conf:
                    continue
                ov = iou(box, d[:4])
                if ov >= 0.3:
                    taken = any(iou(t[1:5], d[:4]) >= 0.5 for t in tracks[j])
                    if taken:
                        claimed_later = True                     # the detector sees it and some track (the same or another id) owns it
                    elif near is None or ov > near[0]:
                        near = (round(ov, 2), j - i)
        ends.append({"track": tid, "first": first[tid], "last": i, "y2": box[3], "len": i - first[tid] + 1, "unclaimed_detection_near": near, "claimed_by_a_track_later": claimed_later})
    return ends


def main():
    cfg = load_pipeline_config(root=REPO)
    det = Detectors(cfg)
    cache = OUT / "raw_detections.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        raw, tracks = d["raw"], d["tracks"]
    else:
        raw, tracks = raw_pass(det, cfg)
        cache.write_text(json.dumps({"raw": raw, "tracks": tracks}), encoding="utf-8")
    n = len(tracks)
    res = {"frames": n, "conf_pipeline": {"helmet": cfg.models.helmet.conf, "coco": cfg.models.coco.conf}, "line_y": LINE_Y, "band_px": BAND, "phases": {"red_0_9.3s": [0, SWITCH], "green_9.3_26s": [SWITCH, n]}}
    for key in ("helmet", "coco"):
        res[f"raw_{key}"] = {}
        for pname, (lo, hi) in res["phases"].items():
            res[f"raw_{key}"][pname] = {"conf_pipeline": band_counts(raw[key], getattr(cfg.models, key).conf, lo, hi),
                                         "conf_0.1": band_counts(raw[key], CONF_LOW, lo, hi)}
    res["tracked_helmet_motorcycles"] = {p: band_counts([[t[1:5] + [t[5]] for t in fr] for fr in tracks], 0.0, lo, hi) for p, (lo, hi) in res["phases"].items()}
    ends = track_ends(tracks, raw["helmet"], cfg.models.helmet.conf)
    hist = {}
    for e in ends:
        b = int(e["y2"] // BAND) * BAND
        hist[f"{b}-{b + BAND}"] = hist.get(f"{b}-{b + BAND}", 0) + 1
    res["track_ends"] = {"n_ended": len(ends), "y2_histogram": dict(sorted(hist.items(), key=lambda kv: int(kv[0].split("-")[0]))),
                         "ended_with_unclaimed_detection_near_in_next_3_frames": sum(1 for e in ends if e["unclaimed_detection_near"]),
                         "ended_where_detector_still_sees_it_but_a_track_owns_it": sum(1 for e in ends if e["claimed_by_a_track_later"] and not e["unclaimed_detection_near"]),
                         "ended_with_no_detection_near": sum(1 for e in ends if not e["unclaimed_detection_near"] and not e["claimed_by_a_track_later"]),
                         "median_len_frames": sorted(e["len"] for e in ends)[len(ends) // 2], "ends": ends}
    # detections near the end of each track at LOW confidence: does lowering the conf recover it?
    low = 0
    for e in ends:
        box_i = next(t[1:5] for t in tracks[e["last"]] if t[0] == e["track"])
        for j in range(e["last"] + 1, min(n, e["last"] + 4)):
            if any(iou(box_i, d[:4]) >= 0.3 and d[4] < cfg.models.helmet.conf for d in raw["helmet"][j]):
                low += 1
                break
    res["track_ends"]["ended_with_only_low_conf_detection_near (0.1 <= conf < pipeline conf)"] = low
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "track_ends"}, indent=1)[:6000])
    print(json.dumps({k: v for k, v in res["track_ends"].items() if k != "ends"}, indent=1))
    if "--strip" in sys.argv:
        strip(raw, tracks, cfg)


def strip(raw, tracks, cfg):
    """M_257 by hand: raw helmet-model motorcycle detections (cyan, conf), raw COCO motorcycles (magenta), tracked boxes (green, id) in frames 210..260 every 5."""
    X1, Y1, X2, Y2 = 560, 120, 1000, 440
    tiles = []
    frames = {}
    want = set(range(210, 261, 5))
    for i, frame in enumerate(read_frames(str(VIDEO))):
        if i in want:
            frames[i] = cv2.resize(frame, tuple(cfg.frame_size))
        if i > 260:
            break
    for i in sorted(want):
        f = frames[i].copy()
        for key, col, off in (("coco", (255, 0, 255), 0), ("helmet", (255, 255, 0), 0)):
            for d in raw[key][i]:
                x1, y1, x2, y2, c = d
                cv2.rectangle(f, (int(x1), int(y1)), (int(x2), int(y2)), col, 1 if c < (cfg.models.helmet.conf if key == "helmet" else cfg.models.coco.conf) else 2)
                cv2.putText(f, f"{c:.2f}", (int(x1), int(y1) - 3 + (12 if key == "coco" else 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        for t in tracks[i]:
            cv2.rectangle(f, (t[1], t[2]), (t[3], t[4]), (0, 255, 0), 3)
            cv2.putText(f, f"M_{t[0]}", (t[1], t[4] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.line(f, (400, 302), (1040, 277), (0, 0, 255), 2)
        c = f[Y1:Y2, X1:X2]
        c = cv2.resize(c, None, fx=1.3, fy=1.3)
        cv2.putText(c, f"frame {i} ({i / 30:.2f}s)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(c)
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    img = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
    cv2.putText(img, "green=track (pipeline conf) cyan=helmet-model raw (thick >= pipeline conf) magenta=COCO raw red=stop line", (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(OUT / "m257_strip.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
