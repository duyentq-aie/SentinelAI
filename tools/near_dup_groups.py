"""Group near-duplicate images of a YOLO dataset by perceptual hash when file names carry no frame/video number (e.g. the plate dataset).

  python tools/near_dup_groups.py --data data/raw/plate --out experiments/EXP-014-plate-groups [--thresholds 2 4 6 8] [--examples 6]

Method: 64-bit difference hash (dHash, 9x8 grayscale) of every image; two images are linked when their Hamming distance is <= T; groups = connected
components (single linkage, so a chain of slowly changing frames is one group, which is what we want for video-like data). For every threshold:
number of groups, group-size distribution, groups spanning several splits, share of val/test images that sit in a group containing a train image.
Read-only. Output: <out>/plate_groups.json (summary per threshold) and <out>/groups_T<k>.csv (image -> group id, split, size) for the chosen thresholds.
dHash is blind to some changes (strong crops, colour grading) and to look-alike scenes: it finds near-duplicates, it does not prove independence.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leakage_check import SPLITS, hamming_matrix, load_split  # noqa: E402


def components(n, hashes, thr, block=512):
    """Union-find over all pairs with Hamming distance <= thr (blockwise, so memory stays small)."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(0, n, block):
        m = hamming_matrix(hashes[i:i + block], hashes)
        for a, b in zip(*np.nonzero(m <= thr)):
            a += i
            if a < b:
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[rb] = ra
    return [find(i) for i in range(n)]


def analyse(rows, thr):
    hashes = np.array([r["hash"] for r in rows], dtype=np.uint64)
    comp = components(len(rows), hashes, thr)
    ids = {c: k for k, c in enumerate(sorted(set(comp), key=lambda c: comp.index(c)))}
    gid = [ids[c] for c in comp]
    members = defaultdict(list)
    for i, g in enumerate(gid):
        members[g].append(i)
    sizes = Counter(len(v) for v in members.values())
    spanning = [g for g, v in members.items() if len({rows[i]["split"] for i in v}) > 1]
    with_train = {s: 0 for s in ("valid", "test")}
    totals = Counter(r["split"] for r in rows)
    for g in spanning:
        for i in members[g]:
            s = rows[i]["split"]
            if s in with_train and any(rows[j]["split"] == "train" for j in members[g]):
                with_train[s] += 1
    largest = sorted((len(v) for v in members.values()), reverse=True)[:5]
    return gid, {"threshold": thr, "groups": len(members), "singletons": sizes.get(1, 0), "groups_of_2_to_5": sum(c for s, c in sizes.items() if 2 <= s <= 5),
                 "groups_of_6_plus": sum(c for s, c in sizes.items() if s >= 6), "largest_groups": largest, "groups_spanning_several_splits": len(spanning),
                 "images_in_spanning_groups": {s: sum(1 for g in spanning for i in members[g] if rows[i]["split"] == s) for s in SPLITS},
                 "pct_val_test_images_whose_group_has_a_train_image": {s: round(100 * with_train[s] / totals[s], 1) for s in with_train}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--csv-for", type=int, nargs="*", default=[4], help="thresholds for which the image->group CSV is written")
    a = ap.parse_args()
    rows = []
    for s in SPLITS:
        for r in load_split(a.data, s):
            if r["hash"] is not None:
                rows.append({**r, "split": s})
    print(f"{len(rows)} images hashed")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"data": a.data, "images": dict(Counter(r["split"] for r in rows)), "thresholds": {}}
    for thr in a.thresholds:
        gid, info = analyse(rows, thr)
        summary["thresholds"][str(thr)] = info
        print(f"T={thr}: {info['groups']} groups ({info['singletons']} singletons), largest {info['largest_groups']}, spanning splits {info['groups_spanning_several_splits']}, "
              f"val/test images with a train twin-group {info['pct_val_test_images_whose_group_has_a_train_image']}")
        if thr in a.csv_for:
            size = Counter(gid)
            with open(out / f"groups_T{thr}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(["path", "split", "sha256", "group", "group_size"])
                for r, g in zip(rows, gid):
                    w.writerow([r["path"].replace("\\", "/"), r["split"], r["sha256"], g, size[g]])
    (out / "plate_groups.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
