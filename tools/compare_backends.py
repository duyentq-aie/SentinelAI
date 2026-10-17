"""Compare an exported ONNX model with its PyTorch original on the same frames of a video (agreement + latency).

  python tools/compare_backends.py --name helmet --onnx models/onnx/helmet_640.onnx [--device cuda|cpu] [--every 8] [--out file.json]

This is an AGREEMENT check, not an accuracy measurement: there is no ground truth here. When a labelled validation set exists under
data/raw/ the proper check is mAP on it (not available at the time of writing: data/raw/ is absent), see EXP-007 notes.
Metrics: detections per backend, greedy matches (same class, IoU >= 0.5), mean IoU of matches, mean/max |confidence difference|,
unmatched boxes on each side, and per-image latency (mean / p50 / p95) after warm-up.
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from sentinel.perception import Track  # noqa: E402
from sentinel.trackdiff import iou, match  # noqa: E402

SOURCES = {"coco": ("yolo11s.pt", 0.35), "helmet": ("helmet_best.pt", 0.55), "plate": ("plate_best.pt", 0.25)}


def boxes_of(result, names):
    b = result.boxes
    return [Track(i, names[int(c)], tuple(int(v) for v in xy), float(cf))
            for i, (xy, c, cf) in enumerate(zip(b.xyxy.cpu().tolist(), b.cls.cpu().tolist(), b.conf.cpu().tolist()))]


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def run(model, frames, conf, device):
    """-> (list of detections per frame, per-image latency ms list)"""
    for f in frames[:3]:
        model.predict(f, conf=conf, device=device, verbose=False)          # warm-up
    dets, lat = [], []
    for f in frames:
        t0 = time.perf_counter()
        r = model.predict(f, conf=conf, device=device, verbose=False)[0]
        lat.append((time.perf_counter() - t0) * 1000)
        dets.append(boxes_of(r, model.names))
    return dets, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=list(SOURCES))
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--video", default=str(REPO / "test_video.mp4"))
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="max frames (0 = all sampled)")
    ap.add_argument("--out")
    a = ap.parse_args()

    from ultralytics import YOLO
    device = 0 if a.device == "cuda" else "cpu"
    src, conf = SOURCES[a.name]
    cap = cv2.VideoCapture(a.video)
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % a.every == 0:
            frames.append(cv2.resize(f, (1280, 720)))
        i += 1
    if a.limit:
        frames = frames[:a.limit]

    pt = YOLO(str(REPO / src))
    ox = YOLO(a.onnx, task="detect")
    d_pt, l_pt = run(pt, frames, conf, device)
    d_ox, l_ox = run(ox, frames, conf, device)

    matched, ious, dconf, only_pt, only_ox = 0, [], [], 0, 0
    for A, B in zip(d_pt, d_ox):
        pairs, oa, ob = match(A, B, 0.5)
        matched += len(pairs); only_pt += len(oa); only_ox += len(ob)
        ious += [iou(x.box, y.box) for x, y in pairs]
        dconf += [abs(x.conf - y.conf) for x, y in pairs]
    n_pt, n_ox = sum(map(len, d_pt)), sum(map(len, d_ox))
    lat = lambda xs: {"mean_ms": round(st.mean(xs), 2), "p50_ms": round(pct(xs, 50), 2), "p95_ms": round(pct(xs, 95), 2)}
    res = {"model": a.name, "onnx": a.onnx, "device": a.device, "frames": len(frames), "conf": conf,
           "detections": {"pytorch": n_pt, "onnx": n_ox}, "matched": matched,
           "match_rate_vs_pytorch": round(matched / n_pt, 4) if n_pt else None, "unmatched": {"pytorch_only": only_pt, "onnx_only": only_ox},
           "mean_iou_of_matches": round(st.mean(ious), 4) if ious else None,
           "conf_abs_diff": {"mean": round(st.mean(dconf), 5) if dconf else None, "max": round(max(dconf), 5) if dconf else None},
           "latency_per_image": {"pytorch": lat(l_pt), "onnx": lat(l_ox)},
           "note": "agreement between backends on test_video frames (every %d-th), NOT accuracy: no ground truth" % a.every,
           "data_raw_available": (REPO / "data" / "raw").exists()}
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
