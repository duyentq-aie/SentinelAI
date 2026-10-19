"""Measure how consistent the label boxes are (shape and size relative to the motorcycle) per class and split. Read-only: no label or image is changed.

  python tools/box_consistency.py --data data/raw/helmet --split-dir experiments/EXP-013-splits/A --out experiments/EXP-015-helmet-grouped-3cls/box_consistency \\
         [--classes helmet motorcycle no-helmet plate] [--thresholds helmet:1.45 no-helmet:1.45 --rel-thresholds ...] [--gallery 30 --gallery-dir DIR]

Per box: aspect = height/width in PIXELS (labels are normalised, the image size comes from the manifest) and, for head classes, rel_h = box height / height of the
motorcycle box that contains the box centre (the smallest such motorcycle box; None when no motorcycle contains it).
Output: boxes.csv (one row per box), summary.json (quantiles per class/split, mixture/valley analysis, flagged counts), hist_*.png.
A box is flagged "body-including" when aspect OR rel_h exceeds its threshold: aspect = q90 of the pooled head boxes (the aspect distribution is unimodal, so
there is no valley to cut at); rel_h = Otsu valley (main mode ~0.2-0.25 plus a tail). See summary.json['pooled_heads_valley'].
The flag is a measurement aid for human review, not a verdict about label quality.
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SPLIT_DIR = {"train": "train", "valid": "val", "test": "test"}
EXTRA = "extra_1080p"


def folder_of(r):
    return EXTRA if r["video"].startswith("other:") else SPLIT_DIR[r["split"]]


def read_boxes(data, split_dir, classes):
    """-> list of box dicts for every label line of every image in the manifest."""
    data = Path(data)
    boxes = []
    for r in csv.DictReader(open(Path(split_dir) / "manifest.csv", encoding="utf-8")):
        W, H = int(r["width"]), int(r["height"])
        lab = data / Path(r["path"]).parent.parent / "labels" / (Path(r["path"]).stem + ".txt")
        rows = []
        for line in lab.read_text(encoding="utf-8").splitlines() if lab.exists() else []:
            if line.strip():
                c, x, y, w, h = int(float(line.split()[0])), *map(float, line.split()[1:5])
                rows.append({"cls": classes[c], "x1": (x - w / 2) * W, "y1": (y - h / 2) * H, "x2": (x + w / 2) * W, "y2": (y + h / 2) * H})
        motos = [b for b in rows if b["cls"] == "motorcycle"]
        for b in rows:
            b.update(folder=folder_of(r), video=r["video"], image=r["path"], W=W, H=H)
            b["w"], b["h"] = b["x2"] - b["x1"], b["y2"] - b["y1"]
            b["aspect"] = b["h"] / b["w"] if b["w"] > 0 else None
            b["rel_h"] = b["moto_h"] = None
            if b["cls"] != "motorcycle":
                cx, cy = (b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2
                inside = [m for m in motos if m["x1"] <= cx <= m["x2"] and m["y1"] <= cy <= m["y2"]]
                if inside:
                    m = min(inside, key=lambda m: (m["x2"] - m["x1"]) * (m["y2"] - m["y1"]))
                    b["moto_h"], b["rel_h"] = m["y2"] - m["y1"], b["h"] / (m["y2"] - m["y1"])
            boxes.append(b)
    return boxes


def quantiles(v):
    v = np.array([x for x in v if x is not None], dtype=float)
    if not len(v):
        return None
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(v)), "mean": round(float(v.mean()), 3), "q05": round(float(q[0]), 3), "q25": round(float(q[1]), 3), "median": round(float(q[2]), 3),
            "q75": round(float(q[3]), 3), "q95": round(float(q[4]), 3)}


def valley(values, bins=40):
    """Two-cluster split of a 1-D distribution: Otsu threshold on the histogram + the depth of the valley relative to the two peaks (bimodality hint)."""
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) < 20:
        return None
    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    best, thr = -1, None
    for i in range(1, bins):
        w0, w1 = hist[:i].sum(), hist[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0, m1 = (hist[:i] * centers[:i]).sum() / w0, (hist[i:] * centers[i:]).sum() / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best:
            best, thr = var, edges[i]
    lo, hi = hist[centers < thr], hist[centers >= thr]
    peak_lo, peak_hi = (lo.max() if len(lo) else 0), (hi.max() if len(hi) else 0)
    near = hist[np.argmin(np.abs(centers - thr))]
    return {"otsu_threshold": round(float(thr), 3), "share_above": round(float((v > thr).mean()), 3), "valley_count": int(near), "peak_below": int(peak_lo), "peak_above": int(peak_hi),
            "valley_depth_ratio": round(float(near / max(1, min(peak_lo, peak_hi))), 3)}


def summarize(boxes, classes, thr_aspect, thr_rel):
    out = {"quantiles": {}, "valley": {}, "flagged": {}}
    folders = ["train", "val", "test", EXTRA]
    for c in classes:
        for f in folders + ["all"]:
            bs = [b for b in boxes if b["cls"] == c and (f == "all" or b["folder"] == f)]
            if not bs:
                continue
            out["quantiles"][f"{c}/{f}"] = {"aspect": quantiles([b["aspect"] for b in bs]), "rel_h": quantiles([b["rel_h"] for b in bs]),
                                            "matched_to_motorcycle": sum(b["rel_h"] is not None for b in bs)}
        out["valley"][c] = {"aspect": valley([b["aspect"] for b in boxes if b["cls"] == c]), "rel_h": valley([b["rel_h"] for b in boxes if b["cls"] == c])}
    for c in ("helmet", "no-helmet"):
        for f in folders:
            bs = [b for b in boxes if b["cls"] == c and b["folder"] == f]
            if not bs:
                continue
            fl = [b for b in bs if b["flag"]]
            out["flagged"][f"{c}/{f}"] = {"boxes": len(bs), "flagged": len(fl), "pct": round(100 * len(fl) / len(bs), 1),
                                          "by_aspect": sum(b["flag_aspect"] for b in bs), "by_rel_h": sum(b["flag_rel"] for b in bs),
                                          "unmatched_to_motorcycle": sum(b["rel_h"] is None for b in bs)}
    return out


def histograms(boxes, out_dir, thr_aspect, thr_rel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"helmet": "#1f77b4", "no-helmet": "#d62728", "motorcycle": "#7f7f7f", "plate": "#2ca02c"}
    for key, xlabel, thr, rng in (("aspect", "height / width of the box (pixels)", thr_aspect, (0.2, 3.5)), ("rel_h", "box height / height of the containing motorcycle box", thr_rel, (0, 1.0))):
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=False)
        for ax, f in zip(axes, ("train", "val", "test")):
            for c in (("helmet", "no-helmet") if key == "rel_h" else ("helmet", "no-helmet", "motorcycle", "plate")):
                v = [b[key] for b in boxes if b["cls"] == c and b["folder"] == f and b[key] is not None]
                if v:
                    ax.hist(np.clip(v, *rng), bins=40, range=rng, histtype="step", linewidth=1.6, density=True, color=cols[c], label=f"{c} (n={len(v)})")
            if thr:
                ax.axvline(thr, color="k", linestyle="--", linewidth=1)
            ax.set_title(f"split A – {f}")
            ax.set_xlabel(xlabel)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"hist_{key}.png", dpi=110)
        plt.close(fig)


def gallery(boxes, data, out_dir, n, seed):
    import cv2
    rng = random.Random(seed)
    groups = defaultdict(list)
    for b in boxes:
        if b["cls"] in ("helmet", "no-helmet") and b["folder"] != EXTRA:
            groups[(b["cls"], "body" if b["flag"] else "head_only")].append(b)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    made = {}
    for (c, g), bs in sorted(groups.items()):
        pick = rng.sample(bs, min(n, len(bs)))
        tiles = []
        for b in pick:
            img = cv2.imdecode(np.fromfile(str(Path(data) / b["image"]), np.uint8), cv2.IMREAD_COLOR)
            cx, cy = (b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2
            s = max(b["w"], b["h"]) * 2.2 + 40
            X1, Y1, X2, Y2 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(b["W"], cx + s / 2)), int(min(b["H"], cy + s / 2))
            crop = img[Y1:Y2, X1:X2].copy()
            sc = 220 / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=sc, fy=sc)
            cv2.rectangle(crop, (int((b["x1"] - X1) * sc), int((b["y1"] - Y1) * sc)), (int((b["x2"] - X1) * sc), int((b["y2"] - Y1) * sc)), (0, 255, 0), 2)
            t = np.zeros((246, 220, 3), np.uint8)
            t[:crop.shape[0], :crop.shape[1]] = crop
            cv2.putText(t, f"{b['folder']} {b['video']} a={b['aspect']:.2f} r={b['rel_h'] if b['rel_h'] is None else round(b['rel_h'], 2)}", (2, 240), 0, 0.33, (255, 255, 0), 1)
            tiles.append(t)
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(Path(out_dir) / f"{c}_{g}.jpg"), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        made[f"{c}/{g}"] = {"population": len(bs), "shown": len(pick)}
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", nargs="+", default=["helmet", "motorcycle", "no-helmet", "plate"])
    ap.add_argument("--aspect-threshold", type=float, help="flag boxes with height/width above this (default: Otsu valley of the head classes pooled)")
    ap.add_argument("--rel-threshold", type=float, help="flag boxes with rel_h above this (default: Otsu valley)")
    ap.add_argument("--gallery", type=int, default=0)
    ap.add_argument("--gallery-dir")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    boxes = read_boxes(a.data, a.split_dir, a.classes)
    heads = [b for b in boxes if b["cls"] in ("helmet", "no-helmet")]
    v_a, v_r = valley([b["aspect"] for b in heads]), valley([b["rel_h"] for b in heads])
    asp = [b["aspect"] for b in heads if b["aspect"] is not None]
    q90 = round(float(np.quantile(asp, 0.90)), 3)
    # aspect has no two clusters (valley_depth_ratio ~ 1.0): use the upper tail (q90 of the pooled heads); rel_h has a main mode + tail: use the Otsu valley
    thr_a = a.aspect_threshold or q90
    thr_r = a.rel_threshold or (v_r or {}).get("otsu_threshold")
    for b in boxes:
        b["flag_aspect"] = bool(b["cls"] in ("helmet", "no-helmet") and thr_a and b["aspect"] is not None and b["aspect"] > thr_a)
        b["flag_rel"] = bool(b["cls"] in ("helmet", "no-helmet") and thr_r and b["rel_h"] is not None and b["rel_h"] > thr_r)
        b["flag"] = b["flag_aspect"] or b["flag_rel"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "boxes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["folder", "video", "image", "cls", "w_px", "h_px", "aspect", "moto_h_px", "rel_h", "flag_aspect", "flag_rel", "flag"])
        for b in boxes:
            w.writerow([b["folder"], b["video"], b["image"], b["cls"], round(b["w"], 1), round(b["h"], 1), round(b["aspect"], 3) if b["aspect"] else "", round(b["moto_h"], 1) if b["moto_h"] else "",
                        round(b["rel_h"], 3) if b["rel_h"] else "", int(b["flag_aspect"]), int(b["flag_rel"]), int(b["flag"])])
    summ = summarize(boxes, a.classes, thr_a, thr_r)
    summ["thresholds"] = {"aspect": thr_a, "rel_h": thr_r, "source": "aspect: q90 of helmet+no-helmet pooled (the distribution is unimodal, no valley); rel_h: Otsu valley of the pooled heads; override with --aspect-threshold/--rel-threshold"}
    summ["pooled_heads_valley"] = {"aspect": v_a, "rel_h": v_r}
    histograms(boxes, out, thr_a, thr_r)
    if a.gallery:
        summ["gallery"] = gallery(boxes, a.data, a.gallery_dir or out / "gallery", a.gallery, a.seed)
    (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("thresholds", "pooled_heads_valley", "flagged")}, indent=1))


if __name__ == "__main__":
    main()
