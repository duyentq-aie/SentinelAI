"""EXP-024: is a helmet that looks like the rider's clothes more often taken for "no helmet"? (measurement only: nothing in the pipeline is changed, no threshold is touched)

Group A = the heads behind the false no_helmet alarms of EXP-020 (every row was graded by eye: the rider WEARS a helmet). For each no_helmet row one version that reported it is taken
(order below), its helmet model is run on that event's frame (conf = the pipeline's 0.55) and the `no-helmet` detection inside the event's vehicle box (best conf) is the head.
Group B (control) = random heads that the OLD helmet model (helmet_best.pt) classed as `helmet` with conf >= 0.6 on test_video (every 5th frame, at most 2 per frame, seed 0,
not overlapping a group-A head); 60 of them.
Per head: mean colour (CIE Lab, cv2, float) of the upper half of the head box ("hat region") and of the region right below the head ("body region": the head's width +20 % each side,
one head-height deep), and DeltaE76 = Euclidean distance of the two mean Lab colours. Hypothesis: A has a LOWER DeltaE than B (the hat contrasts less with the clothes).
Mann-Whitney U (one-sided A < B, and two-sided) + medians. CRUDE: the regions are estimated from the box, nothing is segmented (hair, face, helmet visor, arms, a passenger or
a bag behind the head all fall into them).
Usage: python experiments/EXP-024-hat-shirt-colour/measure.py   -> heads.csv, results.json, group_A.jpg, group_B.jpg (the images show people: gitignored)
"""
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import grade_rows as gr  # noqa: E402

OUT = Path(__file__).parent
OLD = "helmet_best.pt"
EXP015 = "experiments/EXP-015-helmet-grouped-3cls/weights/best.pt"
EXP016 = "experiments/EXP-016-helmet-grouped-4cls/weights/best.pt"
WEIGHTS = {"v0": OLD, "current+old model": OLD, "current+old model @10FPS": OLD, "EXP-015 3cls": EXP015, "EXP-016 4cls": EXP016}
ORDER = ["current+old model", "v0", "EXP-015 3cls", "EXP-016 4cls", "current+old model @10FPS"]
CONF_A, CONF_B, N_B, SEED = 0.55, 0.6, 60, 0
VIDEO = REPO / "test_video.mp4"


def regions(box, shape):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = shape[:2]
    clip = lambda a, lo, hi: int(min(max(a, lo), hi))
    hat = (clip(x1, 0, W), clip(y1, 0, H), clip(x2, 0, W), clip(y1 + 0.5 * h, 0, H))
    body = (clip(x1 - 0.2 * w, 0, W), clip(y2, 0, H), clip(x2 + 0.2 * w, 0, W), clip(y2 + h, 0, H))
    return hat, body


def mean_lab(img, r):
    x1, y1, x2, y2 = r
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    lab = cv2.cvtColor(img[y1:y2, x1:x2].astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    return lab.reshape(-1, 3).mean(0)


def delta_e(img, box):
    hat, body = regions(box, img.shape)
    a, b = mean_lab(img, hat), mean_lab(img, body)
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b)), a.tolist(), b.tolist()


def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


def detect(model, img, conf):
    r = model.predict(img, conf=conf, verbose=False)[0]
    return [(model.names[int(k)], [int(v) for v in xyxy], float(c)) for xyxy, c, k in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(), r.boxes.cls.cpu().tolist())]


def main():
    from ultralytics import YOLO
    models = {}
    get = lambda w: models.setdefault(w, YOLO(str(REPO / w)))
    frames = gr.Frames(VIDEO)
    rows = [r for r in gr.load_rows() if r["type"] == "no_helmet"]
    events = gr.load_events()
    samples, notes = [], []
    # ---- group A
    for r in rows:
        done = False
        for ver in ORDER:
            e = (r["versions"].get(ver) and gr.row_events(r, events).get(ver))
            if not e:
                continue
            model = get(WEIGHTS[ver])
            x1, y1, x2, y2 = e["box"]
            for df in (0, -1, 1, -2, 2):                                                   # the event frame first, then its neighbours
                img = frames.get(e["frame"] + df)
                heads = [(c, b) for n, b, c in detect(model, img, CONF_A) if n == "no-helmet" and x1 <= (b[0] + b[2]) / 2 <= x2 and y1 <= (b[1] + b[3]) / 2 <= y2]
                if heads:
                    conf, box = max(heads)
                    de = delta_e(img, box)
                    if de:
                        samples.append({"group": "A", "row": r["n"], "frame": e["frame"] + df, "version": ver, "model": WEIGHTS[ver], "box": box, "conf": round(conf, 3), "dE": de[0], "hat_lab": de[1], "body_lab": de[2]})
                        done = True
                        break
            if done:
                break
        if not done:
            notes.append(f"row {r['n']}: no no-helmet head found at conf {CONF_A} in the event box (frames +-2): not in group A")
    a_heads = [(s["frame"], s["box"]) for s in samples]
    # ---- group B
    rng = random.Random(SEED)
    cand = []
    model = get(OLD)
    for fi in range(0, 787, 5):
        img = frames.get(fi)
        hs = [(b, c) for n, b, c in detect(model, img, CONF_B) if n == "helmet"]
        rng.shuffle(hs)
        for b, c in hs[:2]:
            if any(abs(fi - af) <= 15 and iou(b, ab) > 0.3 for af, ab in a_heads):
                continue
            de = delta_e(img, b)
            if de:
                cand.append({"group": "B", "row": "", "frame": fi, "version": "-", "model": OLD, "box": b, "conf": round(c, 3), "dE": de[0], "hat_lab": de[1], "body_lab": de[2]})
    rng.shuffle(cand)
    samples += cand[:N_B]
    A = [s["dE"] for s in samples if s["group"] == "A"]
    B = [s["dE"] for s in samples if s["group"] == "B"]
    less = mannwhitneyu(A, B, alternative="less")
    two = mannwhitneyu(A, B, alternative="two-sided")
    res = {"n_A": len(A), "n_B": len(B), "candidates_B": len(cand), "median_dE_A": round(float(np.median(A)), 2), "median_dE_B": round(float(np.median(B)), 2),
           "mean_dE_A": round(float(np.mean(A)), 2), "mean_dE_B": round(float(np.mean(B)), 2),
           "quartiles_A": [round(float(q), 2) for q in np.percentile(A, [25, 75])], "quartiles_B": [round(float(q), 2) for q in np.percentile(B, [25, 75])],
           "mannwhitney_A_less_than_B": {"U": float(less.statistic), "p": round(float(less.pvalue), 4)}, "mannwhitney_two_sided": {"p": round(float(two.pvalue), 4)},
           "auc_prob_A_lower_than_B": round(float(1 - less.statistic / (len(A) * len(B))), 3), "group_A_versions": {v: sum(1 for s in samples if s["group"] == "A" and s["version"] == v) for v in ORDER},
           "notes": notes + ["overload rows have no no-helmet head and are not in group A"],
           "caveat": "crude: regions come from the box, nothing is segmented; hat region = upper half of the head box, body region = below the head box; DeltaE76 in Lab"}
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    with open(OUT / "heads.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["group", "row", "frame", "version", "model", "conf", "x1", "y1", "x2", "y2", "dE", "hat_L", "hat_a", "hat_b", "body_L", "body_a", "body_b"])
        for s in samples:
            w.writerow([s["group"], s["row"], s["frame"], s["version"], s["model"], s["conf"], *s["box"], round(s["dE"], 2), *[round(v, 1) for v in s["hat_lab"]], *[round(v, 1) for v in s["body_lab"]]])
    for g in "AB":
        sheet(sorted([s for s in samples if s["group"] == g], key=lambda s: s["dE"]), frames, OUT / f"group_{g}.jpg", g)
    print(json.dumps(res, indent=1))


def sheet(items, frames, path, g, cols=6, H=170):
    tiles = []
    for s in items:
        img = frames.get(s["frame"]).copy()
        hat, body = regions(s["box"], img.shape)
        for r, col in ((hat, (0, 255, 255)), (body, (255, 128, 0))):
            cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), col, 1)
        x1, y1, x2, y2 = body
        w = x2 - x1
        X1, Y1, X2, Y2 = max(0, x1 - w // 2), max(0, s["box"][1] - w // 3), min(1280, x2 + w // 2), min(720, y2 + w // 4)
        c = cv2.resize(img[Y1:Y2, X1:X2], (int((X2 - X1) * H / max(1, Y2 - Y1)), H))[:, :170]
        t = np.full((H + 34, 170, 3), 25, np.uint8)
        t[34:34 + c.shape[0], :c.shape[1]] = c
        cv2.putText(t, f"dE {s['dE']:.1f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(t, (f"row {s['row']} " if g == "A" else f"f{s['frame']} ") + f"c{s['conf']:.2f}", (3, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    img = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
