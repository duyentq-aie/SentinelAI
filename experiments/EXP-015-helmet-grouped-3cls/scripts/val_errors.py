import csv, json, sys
from pathlib import Path
import numpy as np, cv2
from ultralytics import YOLO
root = Path("data/raw/helmet"); out = Path("experiments/EXP-015-helmet-grouped-3cls")
rows = [r for r in csv.DictReader(open("experiments/EXP-013-splits/A/manifest.csv", encoding="utf-8")) if r["split"] == "valid" and not r["video"].startswith("other:")]
model = YOLO(str(out / "weights" / "best.pt")); NAMES = {0: "helmet", 1: "motorcycle", 2: "no-helmet"}
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0
cases = []
for r in rows:
    p = root / r["path"]; img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR); H, W = img.shape[:2]
    lab = root / Path(r["path"]).parent.parent / "labels" / (p.stem + ".txt")
    gt = []
    for l in lab.read_text().splitlines():
        if not l.strip(): continue
        c, x, y, w, h = int(float(l.split()[0])), *map(float, l.split()[1:5])
        if c in (0, 1, 2): gt.append((c, [(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H]))
    res = model.predict(img, imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
    pr = [(int(c), float(s), b.tolist()) for c, s, b in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy)]
    heads_gt = [(c, b) for c, b in gt if c in (0, 2)]; heads_pr = [(c, s, b) for c, s, b in pr if c in (0, 2)]
    for c, b in heads_gt:
        if c != 2: continue
        ov = sorted(((iou(b, pb), pc, ps, pb) for pc, ps, pb in heads_pr), reverse=True)
        top = ov[0] if ov else None
        if top and top[0] >= 0.5 and top[1] == 2: continue                       # correct
        if top and top[0] >= 0.5 and top[1] == 0: cases.append(dict(kind="a_gt_nohelmet_pred_helmet", img=r["path"], video=r["video"], gt=b, pred=top[3], conf=round(top[2], 3), iou=round(top[0], 2)))
        elif top and top[0] >= 0.3: cases.append(dict(kind="a_gt_nohelmet_pred_helmet" if top[1] == 0 else "other", img=r["path"], video=r["video"], gt=b, pred=top[3], conf=round(top[2], 3), iou=round(top[0], 2), note="low iou"))
        else: cases.append(dict(kind="b_gt_nohelmet_missed", img=r["path"], video=r["video"], gt=b, pred=None, conf=None, iou=round(top[0], 2) if top else 0))
    for c, s, b in heads_pr:
        if c != 2: continue
        best = max(((iou(b, gb), gc) for gc, gb in heads_gt), default=(0, None))
        if best[0] >= 0.5 and best[1] == 2: continue
        cases.append(dict(kind="c_pred_nohelmet_no_gt_nohelmet", img=r["path"], video=r["video"], gt=None, pred=b, conf=round(s, 3), iou=round(best[0], 2), overlaps_gt_class=NAMES.get(best[1]) if best[0] >= 0.3 else None))
json.dump(cases, open(out / "val_error_cases.json", "w"), indent=1)
from collections import Counter
print(len(rows), "val images;", Counter(c["kind"] for c in cases))
print(Counter((c["kind"], c.get("overlaps_gt_class")) for c in cases if c["kind"].startswith("c")))
