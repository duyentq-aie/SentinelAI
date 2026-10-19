"""Evaluate the helmet detectors on the hand-corrected out-of-domain set (EXP-018: 15 frames of test_video.mp4).

  python tools/eval_ood.py --labels <folder with the corrected YOLO labels (Roboflow export)> [--config configs/eval/ood_test_video.yaml] [--json out.json]

Runs EVERY model of the config (old helmet_best.pt and the EXP-015 best.pt by default) on the images and reports, per class and per model, INSIDE the ROI only
(a ground-truth box or a prediction counts only when its centre lies in the ROI of the config):
  n_gt, n_pred (conf >= 0.001), P and R at conf 0.25 / IoU 0.5, AP50 and AP50-95 (COCO-style 101-point interpolation, greedy matching by confidence).
Labels: any folder containing YOLO .txt files (Roboflow renames `x.png` -> `x_png.rf.<hash>.jpg/.txt`; the original stem is recovered) and a data.yaml with class names;
classes that the models do not have (e.g. `plate`) are ignored. An image without a label file counts as having no objects (reported as a warning).
The ground truth is whatever you corrected by hand: this tool measures nothing about the quality of those labels.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
RF_STEM = re.compile(r"^(?P<orig>.+?)_(?:png|jpg|jpeg)\.rf\.[0-9a-f]+$", re.I)
IOU_THRS = np.linspace(0.5, 0.95, 10)


# ------------------------------------------------------------------ io
def original_stem(stem):
    m = RF_STEM.match(stem)
    return m.group("orig") if m else stem


def find_names(labels_dir):
    for p in sorted(Path(labels_dir).rglob("data.yaml")):
        names = yaml.safe_load(p.read_text(encoding="utf-8")).get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        if names:
            return list(names)
    return None


def read_labels(labels_dir, images, classes, names=None):
    """-> ({image stem: [(class name, [x1, y1, x2, y2])]}, warnings). `images` = {stem: (w, h)}. Classes not in `classes` are dropped."""
    names = names or find_names(labels_dir) or classes
    out, warnings = {s: [] for s in images}, []
    seen = set()
    for p in sorted(Path(labels_dir).rglob("*.txt")):
        stem = original_stem(p.stem)
        if stem not in images:
            continue
        seen.add(stem)
        w, h = images[stem]
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            name = names[c] if c < len(names) else None
            if name not in classes:
                continue
            x, y, bw, bh = map(float, parts[1:5])
            out[stem].append((name, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    missing = sorted(set(images) - seen)
    if missing:
        warnings.append(f"{len(missing)} image(s) without a label file (treated as no objects): {missing[:3]}{'...' if len(missing) > 3 else ''}")
    return out, warnings


# ------------------------------------------------------------------ metrics
def in_roi(box, roi):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]


def iou(a, b):
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def compute_ap(recall, precision):
    """COCO-style 101-point AP: mean over recall thresholds r = 0, 0.01 ... 1 of the best precision at recall >= r (0 when that recall is never reached).
    NOT the Ultralytics `val` recipe (which interpolates linearly towards an appended (recall 1, precision 0) point: a perfect detector scores 0.995 there and a
    detector that stops at recall 0.5 scores 0.75 instead of 0.5), so AP here is not directly comparable with the mAP numbers of EXP-000 / EXP-015; both models of this
    evaluation are scored by the same function."""
    env = np.maximum.accumulate(np.asarray(precision, dtype=float)[::-1])[::-1]
    idx = np.searchsorted(np.asarray(recall, dtype=float), np.linspace(0, 1, 101), side="left")
    return float(np.mean([env[i] if i < len(env) else 0.0 for i in idx]))


def match(gt, preds, thr):
    """Greedy matching by confidence inside one image and one class -> list of tp flags (aligned with preds sorted by -conf)."""
    taken, flags = set(), []
    for p in sorted(preds, key=lambda p: -p[0]):
        best, bi = thr, None
        for i, g in enumerate(gt):
            if i not in taken:
                v = iou(p[1], g)
                if v >= best:
                    best, bi = v, i
        flags.append(bi is not None)
        if bi is not None:
            taken.add(bi)
    return flags


def evaluate(gt_by_image, pred_by_image, roi, classes, pr_conf=0.25):
    """gt_by_image {img: [(cls, box)]}, pred_by_image {img: [(cls, conf, box)]} -> {cls: metrics, 'all': mean over classes that have GT}."""
    res = {}
    for c in classes:
        gts = {im: [b for k, b in g if k == c and in_roi(b, roi)] for im, g in gt_by_image.items()}
        prs = {im: [(s, b) for k, s, b in p if k == c and in_roi(b, roi)] for im, p in pred_by_image.items()}
        n_gt, n_pred = sum(map(len, gts.values())), sum(map(len, prs.values()))
        row = {"n_gt": n_gt, "n_pred": n_pred, "precision": None, "recall": None, "ap50": None, "ap50_95": None}
        if n_gt:
            aps = []
            for ti, thr in enumerate(IOU_THRS):
                recs = []                                                    # (conf, tp) over all images
                for im in gts:
                    ps = sorted(prs.get(im, []), key=lambda p: -p[0])
                    for p, f in zip(ps, match(gts[im], ps, thr)):
                        recs.append((p[0], f))
                recs.sort(key=lambda r: -r[0])
                tp = np.array([r[1] for r in recs], dtype=float)
                if len(tp) == 0:
                    aps.append(0.0)
                    continue
                ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
                aps.append(compute_ap(ctp / n_gt, ctp / (ctp + cfp)))
                if ti == 0:
                    row["ap50"] = round(aps[0], 4)
                    k = int((np.array([r[0] for r in recs]) >= pr_conf).sum())
                    row["precision"] = round(float(ctp[k - 1] / k), 4) if k else None
                    row["recall"] = round(float(ctp[k - 1] / n_gt), 4) if k else 0.0
            if row["ap50"] is None:
                row["ap50"] = 0.0
                row["recall"] = 0.0
            row["ap50_95"] = round(float(np.mean(aps)), 4)
        res[c] = row
    have = [r for r in res.values() if r["n_gt"]]
    res["all"] = {"n_gt": sum(r["n_gt"] for r in res.values() if r is not res.get("all")), "n_pred": sum(r["n_pred"] for k, r in res.items() if k != "all"),
                  "precision": None, "recall": None, "ap50": round(float(np.mean([r["ap50"] for r in have])), 4) if have else None,
                  "ap50_95": round(float(np.mean([r["ap50_95"] for r in have])), 4) if have else None, "classes_with_gt": len(have)}
    return res


def predict_dir(model_path, images_dir, classes, conf=0.001, imgsz=640):
    from ultralytics import YOLO
    m = YOLO(str(REPO / model_path) if not Path(model_path).is_absolute() else model_path)
    out = {}
    for p in sorted(Path(images_dir).glob("*.png")):
        r = m.predict(str(p), imgsz=imgsz, conf=conf, iou=0.7, max_det=300, verbose=False)[0]
        out[p.stem] = [(classes[int(c)], float(s), b) for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()) if int(c) < len(classes)]
    return out


def format_table(model, res, roi):
    L = [f"\nmodel: {model}   ROI {roi}   (P/R at conf 0.25, IoU 0.5; AP = COCO-style)", "%-12s %6s %7s %9s %8s %8s %10s" % ("class", "n_gt", "n_pred", "precision", "recall", "AP50", "AP50-95")]
    f = lambda x: "    n/a" if x is None else f"{x:8.3f}"
    for c, r in res.items():
        L.append("%-12s %6d %7d %9s %8s %8s %10s" % (c, r["n_gt"], r["n_pred"], f(r["precision"]), f(r["recall"]), f(r["ap50"]), f(r["ap50_95"])))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="folder with the corrected YOLO labels (searched recursively) and data.yaml")
    ap.add_argument("--config", default=str(REPO / "configs" / "eval" / "ood_test_video.yaml"))
    ap.add_argument("--images", help="override images_dir of the config")
    ap.add_argument("--json")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    images_dir = Path(a.images or REPO / cfg["images_dir"])
    sizes = {}
    for p in sorted(images_dir.glob("*.png")):
        im = cv2.imread(str(p))
        sizes[p.stem] = (im.shape[1], im.shape[0])
    if not sizes:
        sys.exit(f"no images in {images_dir}")
    gt, warnings = read_labels(a.labels, sizes, cfg["classes"])
    for w in warnings:
        print("WARNING:", w)
    n_gt_all = defaultdict(int)
    for g in gt.values():
        for k, b in g:
            n_gt_all[k] += int(in_roi(b, cfg["roi"]))
    print(f"{len(sizes)} images, ground-truth boxes inside the ROI: {dict(n_gt_all)}")
    report = {"roi": cfg["roi"], "images": len(sizes), "gt_boxes_in_roi": dict(n_gt_all), "warnings": warnings, "models": {}}
    for name, path in cfg["models"].items():
        if not (REPO / path).exists():
            print(f"skipping {name}: {path} not found")
            continue
        res = evaluate(gt, predict_dir(path, images_dir, cfg["classes"]), cfg["roi"], cfg["classes"])
        report["models"][name] = res
        print(format_table(name, res, cfg["roi"]))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("->", a.json)


if __name__ == "__main__":
    main()
