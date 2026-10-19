"""Plate boxes of the test split A: the 4-class model (class 3) vs plate_best.pt (the deployed plate model), same evaluation function (tools/eval_ood.py:
COCO-style AP, P/R at conf 0.25 and IoU 0.5). Read-only. Usage: python experiments/EXP-016-helmet-grouped-4cls/scripts/plate_compare.py"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
import eval_ood as eo  # noqa: E402
from ultralytics import YOLO  # noqa: E402

rows = [r for r in csv.DictReader(open(REPO / "experiments/EXP-013-splits/A/manifest.csv", encoding="utf-8")) if r["split"] == "test" and not r["video"].startswith("other:")]
gt, sizes = {}, {}
for r in rows:
    p = REPO / "data/raw/helmet" / r["path"]
    w, h = int(r["width"]), int(r["height"])
    lab = REPO / "data/raw/helmet" / Path(r["path"]).parent.parent / "labels" / (p.stem + ".txt")
    boxes = []
    for line in lab.read_text(encoding="utf-8").splitlines():
        t = line.split()
        if t and int(float(t[0])) == 3:
            x, y, bw, bh = map(float, t[1:5])
            boxes.append(("plate", [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    gt[p.stem] = boxes
    sizes[p.stem] = (w, h, p)
full_roi = [0, 0, 10 ** 6, 10 ** 6]


def run(weights, plate_cls):
    m = YOLO(str(REPO / weights))
    pred = {}
    for stem, (w, h, p) in sizes.items():
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        r = m.predict(img, imgsz=640, conf=0.001, iou=0.7, max_det=300, verbose=False)[0]
        pred[stem] = [("plate", float(s), b) for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()) if int(c) == plate_cls]
    return eo.evaluate(gt, pred, full_roi, ["plate"])["plate"]


out = {"images": len(sizes), "gt_plate_boxes": sum(len(v) for v in gt.values()),
       "4cls_best.pt (class 3)": run("experiments/EXP-016-helmet-grouped-4cls/weights/best.pt", 3),
       "plate_best.pt (deployed plate model)": run("plate_best.pt", 0)}
Path(REPO / "experiments/EXP-016-helmet-grouped-4cls/plate_vs_plate_best.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))


# ---- second protocol: the role plate_best.pt has in the pipeline = detector run on a CROP of the vehicle (here the ground-truth motorcycle boxes, an optimistic
# stand-in for detected vehicles); plate boxes are mapped back to the frame and duplicates from overlapping crops are merged (class-wise NMS, IoU 0.5)
motos = {}
for r in rows:
    p = REPO / "data/raw/helmet" / r["path"]
    lab = REPO / "data/raw/helmet" / Path(r["path"]).parent.parent / "labels" / (p.stem + ".txt")
    w, h = int(r["width"]), int(r["height"])
    mb = []
    for line in lab.read_text(encoding="utf-8").splitlines():
        t = line.split()
        if t and int(float(t[0])) == 1:
            x, y, bw, bh = map(float, t[1:5])
            mb.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    motos[p.stem] = mb


def inside(box, outer):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


gt_reach = {s: [b for b in v if any(inside(b[1], m) for m in motos[s])] for s, v in gt.items()}


def run_crops(weights, plate_cls):
    m = YOLO(str(REPO / weights))
    pred = {}
    for stem, (w, h, p) in sizes.items():
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        found = []
        for mb in motos[stem]:
            x1, y1, x2, y2 = [int(v) for v in (max(0, mb[0]), max(0, mb[1]), min(w, mb[2]), min(h, mb[3]))]
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            r = m.predict(img[y1:y2, x1:x2], imgsz=640, conf=0.001, iou=0.7, max_det=100, verbose=False)[0]
            for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                if int(c) == plate_cls:
                    found.append((int(c), float(s), [b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1], ""))
        keep = []
        for d in sorted(found, key=lambda d: -d[1]):
            if all(eo.iou(k[2], d[2]) < 0.5 for k in keep):
                keep.append(d)
        pred[stem] = [("plate", d[1], d[2]) for d in keep]
    return eo.evaluate(gt_reach, pred, full_roi, ["plate"])["plate"]


out2 = {"protocol": "detector run on crops of the GROUND-TRUTH motorcycle boxes (optimistic stand-in for detected vehicles); GT = plate boxes whose centre lies in such a crop",
        "gt_plate_boxes_reachable": sum(len(v) for v in gt_reach.values()), "gt_plate_boxes_total": out["gt_plate_boxes"],
        "4cls_best.pt (class 3)": run_crops("experiments/EXP-016-helmet-grouped-4cls/weights/best.pt", 3),
        "plate_best.pt (deployed plate model)": run_crops("plate_best.pt", 0)}
Path(REPO / "experiments/EXP-016-helmet-grouped-4cls/plate_vs_plate_best_on_vehicle_crops.json").write_text(json.dumps(out2, indent=1), encoding="utf-8")
print(json.dumps(out2, indent=1))
