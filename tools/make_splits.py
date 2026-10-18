"""Group-aware train/val/test split proposals for a YOLO-format dataset whose images are frames of a few videos (Roboflow `frame_<video>_<n>`).

  python tools/make_splits.py --data data/raw/helmet --out experiments/EXP-013-splits --classes helmet motorcycle no-helmet plate \\
         --balance no-helmet [--seed 0] [--variants A B C]

A random split of frames leaks (neighbouring frames are almost the same picture, see tools/leakage_check.py). Here every video is ONE group and goes
entirely to one split. With few groups an exact 70/15/15 split is impossible, so for each variant a seeded search (random restarts + local
moves over group assignments) minimises a weighted deviation from the target shares of images and of instances of the balanced class(es).

Read-only on the dataset: nothing is moved, copied or renamed. Output per variant `<out>/<variant>/`:
  manifest.csv      one row per image: split, video, frame, path (relative to --data), sha256, width, height, instances per class
  summary.json      images / instances per split and class, groups per split, targets, objective, seed, tool version, sha256 of manifest.csv
  summary.md        the same as a table
and `<out>/proposals.md` comparing the variants.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leakage_check import IMG, parse_name  # noqa: E402

SPLITS = ("train", "valid", "test")
VERSION = 1
# name -> (target shares train/valid/test, weight of the image-share term, weight of the balanced-class term, weight of "each split has every class")
VARIANTS = {
    "A": {"shares": (0.70, 0.15, 0.15), "w_images": 1.0, "w_balance": 1.0, "desc": "70/15/15, images and balanced class equally important"},
    "B": {"shares": (0.70, 0.15, 0.15), "w_images": 0.3, "w_balance": 3.0, "desc": "70/15/15, balanced class first (split sizes may drift)"},
    "C": {"shares": (0.60, 0.20, 0.20), "w_images": 1.0, "w_balance": 1.0, "desc": "60/20/20, larger val/test for lower metric variance"},
}


def sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def read_dataset(root, classes):
    """-> list of image records with per-class instance counts (labels are read, never written)."""
    root = Path(root)
    rows = []
    for split in SPLITS:
        d = root / split / "images"
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMG:
                continue
            lab = root / split / "labels" / (p.stem + ".txt")
            cnt = Counter()
            if lab.exists():
                for line in lab.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        cnt[int(float(line.split()[0]))] += 1
            img = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            info = parse_name(p.name)
            rows.append({"orig_split": split, "path": str(p.relative_to(root)).replace("\\", "/"), "sha256": sha256(p), "video": info["video"] or "other:" + re.sub(r"_\d+$", "", info["src"]),   # names without <video>_<n>: one group per name stem
                         "frame": info["frame"] if info["frame"] is not None else None, "w": None if img is None else img.shape[1], "h": None if img is None else img.shape[0],
                         "counts": {classes[i] if i < len(classes) else f"class{i}": c for i, c in cnt.items()}})
    return rows


def group_stats(rows, classes):
    g = defaultdict(lambda: {"images": 0, **{c: 0 for c in classes}})
    for r in rows:
        s = g[r["video"]]
        s["images"] += 1
        for c, n in r["counts"].items():
            s[c] = s.get(c, 0) + n
    return dict(g)


def objective(assign, groups, classes, balance, shares, w_images, w_balance):
    tot = Counter()
    per = [Counter() for _ in SPLITS]
    for v, s in assign.items():
        for k, x in groups[v].items():
            per[s][k] += x
            tot[k] += x
    cost = 0.0
    for s in range(3):
        cost += w_images * ((per[s]["images"] / tot["images"] - shares[s]) / shares[s]) ** 2
        for c in balance:
            if tot[c]:
                cost += w_balance * ((per[s][c] / tot[c] - shares[s]) / shares[s]) ** 2
        for c in classes:                                  # every split must contain every class, with a few instances at least
            if tot[c] and per[s][c] < 0.5 * shares[s] * tot[c]:
                cost += 5.0
    return cost


def search(groups, classes, balance, variant, seed, restarts=300):
    """Seeded local search over assignments of whole groups to splits. Deterministic for a given seed."""
    rng = random.Random(seed)
    names = sorted(groups)
    best, best_cost = None, float("inf")
    for _ in range(restarts):
        a = {v: rng.randrange(3) for v in names}
        cost = objective(a, groups, classes, balance, variant["shares"], variant["w_images"], variant["w_balance"])
        improved = True
        while improved:
            improved = False
            for v in rng.sample(names, len(names)):                      # move one group
                for s in range(3):
                    if s != a[v]:
                        old, a[v] = a[v], s
                        c = objective(a, groups, classes, balance, variant["shares"], variant["w_images"], variant["w_balance"])
                        if c < cost - 1e-12:
                            cost, improved = c, True
                        else:
                            a[v] = old
            for v, w in ((x, y) for x in names for y in names if x < y and a[x] != a[y]):      # swap two groups
                a[v], a[w] = a[w], a[v]
                c = objective(a, groups, classes, balance, variant["shares"], variant["w_images"], variant["w_balance"])
                if c < cost - 1e-12:
                    cost, improved = c, True
                else:
                    a[v], a[w] = a[w], a[v]
        if cost < best_cost - 1e-12:
            best, best_cost = dict(a), cost
    return best, best_cost


def summarize(rows, assign, classes):
    out = {s: {"images": 0, "videos": [], "instances": {c: 0 for c in classes}} for s in SPLITS}
    for r in rows:
        s = SPLITS[assign[r["video"]]]
        out[s]["images"] += 1
        for c, n in r["counts"].items():
            out[s]["instances"][c] = out[s]["instances"].get(c, 0) + n
    for v, s in assign.items():
        out[SPLITS[s]]["videos"].append(v)
    for s in SPLITS:
        out[s]["videos"].sort()
    return out


def write_variant(out_dir, name, variant, rows, assign, classes, seed, cost, balance):
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    cols = ["split", "video", "frame", "path", "orig_split", "sha256", "width", "height"] + [f"n_{c}" for c in classes]
    with open(d / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in sorted(rows, key=lambda r: (assign[r["video"]], r["video"], r["frame"] if r["frame"] is not None else -1, r["path"])):
            w.writerow([SPLITS[assign[r["video"]]], r["video"], r["frame"], r["path"], r["orig_split"], r["sha256"], r["w"], r["h"]] + [r["counts"].get(c, 0) for c in classes])
    summ = summarize(rows, assign, classes)
    tot_img = sum(summ[s]["images"] for s in SPLITS)
    data = {"variant": name, "description": variant["desc"], "seed": seed, "tool_version": VERSION, "balanced_classes": balance, "target_shares": variant["shares"],
            "objective": round(cost, 6), "classes": classes, "splits": summ,
            "image_shares": {s: round(summ[s]["images"] / tot_img, 3) for s in SPLITS},
            "manifest_sha256": hashlib.sha256((d / "manifest.csv").read_bytes()).hexdigest(), "groups": {v: SPLITS[s] for v, s in sorted(assign.items())}}
    (d / "summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    lines = [f"# split variant {name}: {variant['desc']} (seed {seed})", "", "| split | images | share | videos | " + " | ".join(classes) + " |", "|---|---|---|---|" + "---|" * len(classes)]
    for s in SPLITS:
        lines.append(f"| {s} | {summ[s]['images']} | {data['image_shares'][s]:.0%} | {len(summ[s]['videos'])} | " + " | ".join(str(summ[s]['instances'].get(c, 0)) for c in classes) + " |")
    lines += ["", "videos: " + "; ".join(f"{s}: {', '.join(summ[s]['videos'])}" for s in SPLITS), f"manifest sha256: `{data['manifest_sha256']}`"]
    (d / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", nargs="+", required=True, help="class names in id order, as in data.yaml")
    ap.add_argument("--balance", nargs="+", default=["no-helmet"], help="class(es) whose instance share should follow the split shares")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    a = ap.parse_args()
    rows = read_dataset(a.data, a.classes)
    groups = group_stats(rows, a.classes)
    print(f"{len(rows)} images, {len(groups)} groups (videos)")
    results = {}
    for name in a.variants:
        assign, cost = search(groups, a.classes, a.balance, VARIANTS[name], a.seed)
        results[name] = write_variant(a.out, name, VARIANTS[name], rows, assign, a.classes, a.seed, cost, a.balance)
        r = results[name]
        print(f"[{name}] {r['description']}: objective {r['objective']}")
        for s in SPLITS:
            print(f"   {s:5s} images {r['splits'][s]['images']:5d} ({r['image_shares'][s]:.0%}) videos {len(r['splits'][s]['videos']):2d} instances {r['splits'][s]['instances']}")
    md = ["# Split proposals (group = video; nothing has been moved)", ""]
    for name, r in results.items():
        md.append(open(Path(a.out) / name / "summary.md", encoding="utf-8").read())
    (Path(a.out) / "proposals.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
