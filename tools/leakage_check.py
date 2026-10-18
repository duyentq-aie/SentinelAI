"""Train/val/test leakage check for a YOLO-format dataset (e.g. a Roboflow export). Read-only: does not re-split, rename or delete anything.

  python tools/leakage_check.py --data data/raw/helmet --out experiments/EXP-011-data-leakage/helmet_leakage.json [--examples 12]

What it measures (for every val/test image against the train split unless stated):
  1. byte-identical files (sha256) across splits;
  2. perceptual near-duplicates: 64-bit difference hash (dHash) Hamming distance to the nearest train image;
  3. temporal adjacency: Roboflow names such as frame_10_563_jpg.rf.<hash>.jpg carry <video>_<frame>; how far is the nearest TRAIN frame of the same video?
  4. 'leaky groups': frames of one video are clustered (gap <= --gap frames); a group that appears in more than one split is a leak by construction;
  5. the same source image in several splits (Roboflow `<name>_jpg.rf.<hash>`).
A random split of frames extracted from video leaks: neighbouring frames are nearly the same picture. Fixing it means splitting by group (not done here).
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SPLITS = ("train", "valid", "test")
IMG = {".jpg", ".jpeg", ".png", ".bmp"}
RF = re.compile(r"^(?P<src>.+?)_(?:jpg|jpeg|png)\.rf\.(?P<hash>[0-9a-f]+)\.(?:jpg|jpeg|png)$", re.I)
SEQ = re.compile(r"^(?P<prefix>.*?)(?P<video>\d+)_(?P<frame>\d+)$")           # frame_10_563 -> video "frame_10", frame 563
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def parse_name(name):
    """-> dict(src, video, frame) ; video/frame None when the name has no <video>_<frame> pattern."""
    m = RF.match(name)
    src = m.group("src") if m else Path(name).stem
    s = SEQ.match(src)
    return {"src": src, "video": f"{s.group('prefix')}{s.group('video')}" if s else None, "frame": int(s.group("frame")) if s else None}


def dhash(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def hamming_matrix(a, b):
    """a: (n,) uint64, b: (m,) uint64 -> (n, m) Hamming distances."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    return POPCOUNT[x.view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1)


def load_split(root, split):
    d = Path(root) / split / "images"
    rows = []
    for p in sorted(d.iterdir()) if d.exists() else []:
        if p.suffix.lower() not in IMG:
            continue
        data = p.read_bytes()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        rows.append({"path": str(p.relative_to(root)), "name": p.name, "sha256": hashlib.sha256(data).hexdigest(), "hash": dhash(img) if img is not None else None,
                     "size": None if img is None else [img.shape[1], img.shape[0]], **parse_name(p.name)})
    return rows


def pct(n, d):
    return round(100 * n / d, 1) if d else None


def check(data, gap=15, examples=10):
    rows = {s: load_split(data, s) for s in SPLITS}
    out = {"data": str(data), "images": {s: len(r) for s, r in rows.items()}, "gap_frames_for_groups": gap}

    # 1. exact duplicates
    sha_split = defaultdict(set)
    for s, r in rows.items():
        for x in r:
            sha_split[x["sha256"]].add(s)
    out["identical_files_across_splits"] = sum(1 for v in sha_split.values() if len(v) > 1)
    within = Counter(x["sha256"] for r in rows.values() for x in r)
    out["files_with_a_byte_identical_twin_anywhere"] = sum(1 for c in within.values() if c > 1)

    # 2. dHash near-duplicates against train
    tr = [x for x in rows["train"] if x["hash"] is not None]
    thr = (0, 4, 8, 12)
    out["dhash_nearest_train_hamming"] = {}
    ex = []
    if tr:
        th = np.array([x["hash"] for x in tr], dtype=np.uint64)
        for s in ("valid", "test"):
            q = [x for x in rows[s] if x["hash"] is not None]
            if not q:
                continue
            d = np.empty(len(q), dtype=int)
            nn = np.empty(len(q), dtype=int)
            for i in range(0, len(q), 256):
                m = hamming_matrix(np.array([x["hash"] for x in q[i:i + 256]], dtype=np.uint64), th)
                d[i:i + 256], nn[i:i + 256] = m.min(axis=1), m.argmin(axis=1)
            out["dhash_nearest_train_hamming"][s] = {"n": len(q), "median": float(np.median(d)), "mean": round(float(d.mean()), 2),
                                                     **{f"pct_with_train_neighbour_le_{t}": pct(int((d <= t).sum()), len(q)) for t in thr}}
            for i in np.argsort(d)[: examples // 2]:
                ex.append({"split": s, "image": q[i]["name"], "nearest_train": tr[nn[i]]["name"], "hamming": int(d[i])})
    out["closest_pairs_examples"] = ex

    # 3. temporal adjacency in the same video
    train_frames = defaultdict(list)
    for x in rows["train"]:
        if x["video"] is not None:
            train_frames[x["video"]].append(x["frame"])
    out["temporal"] = {}
    for s in ("valid", "test"):
        q = [x for x in rows[s] if x["video"] is not None]
        if not q:
            out["temporal"][s] = {"n_parsed": 0, "note": "names do not follow <video>_<frame>: temporal check not possible"}
            continue
        dist = []
        for x in q:
            fr = train_frames.get(x["video"])
            dist.append(min(abs(f - x["frame"]) for f in fr) if fr else None)
        known = [d for d in dist if d is not None]
        out["temporal"][s] = {"n_parsed": len(q), "n_unparsed": len(rows[s]) - len(q), "videos": sorted({x["video"] for x in q}),
                              "nearest_train_frame_gap_median": float(np.median(known)) if known else None,
                              **{f"pct_with_train_frame_within_{k}": pct(sum(1 for d in known if d <= k), len(q)) for k in (1, 2, 5, 15, 30)}}

    # 4. groups of consecutive frames spanning several splits
    by_video = defaultdict(list)
    for s, r in rows.items():
        for x in r:
            if x["video"] is not None:
                by_video[x["video"]].append((x["frame"], s))
    groups, leaky = 0, 0
    leaky_images = Counter()
    for v, fl in by_video.items():
        fl.sort()
        cur = [fl[0]]
        for item in fl[1:] + [None]:
            if item is not None and item[0] - cur[-1][0] <= gap:
                cur.append(item)
                continue
            groups += 1
            sp = {s for _, s in cur}
            if len(sp) > 1:
                leaky += 1
                for _, s in cur:
                    leaky_images[s] += 1
            if item is not None:
                cur = [item]
    out["groups"] = {"videos": len(by_video), "groups": groups, "groups_spanning_several_splits": leaky,
                     "images_in_leaky_groups": dict(leaky_images), "total_images_with_video_id": sum(len(v) for v in by_video.values())}

    # 5. same source image in several splits
    src_split = defaultdict(set)
    for s, r in rows.items():
        for x in r:
            src_split[x["src"]].add(s)
    out["same_source_name_in_several_splits"] = sum(1 for v in src_split.values() if len(v) > 1)
    out["image_sizes"] = {f"{w}x{h}": c for (w, h), c in Counter(tuple(x["size"]) for r in rows.values() for x in r if x["size"]).most_common(8)}
    return out


def format_report(r):
    L = [f"data: {r['data']}   images: {r['images']}",
         f"byte-identical files across splits: {r['identical_files_across_splits']}   (files with an identical twin anywhere: {r['files_with_a_byte_identical_twin_anywhere']})",
         f"same source name in several splits: {r['same_source_name_in_several_splits']}"]
    for s, v in r["dhash_nearest_train_hamming"].items():
        L.append(f"dHash {s} -> nearest train: median {v['median']}, " + ", ".join(f"<= {t}: {v[f'pct_with_train_neighbour_le_{t}']}%" for t in (0, 4, 8, 12)))
    for s, v in r["temporal"].items():
        if v.get("n_parsed"):
            L.append(f"temporal {s}: videos {v['videos']}, median gap to a train frame {v['nearest_train_frame_gap_median']}, " +
                     ", ".join(f"within {k}: {v[f'pct_with_train_frame_within_{k}']}%" for k in (1, 2, 5, 15, 30)))
        else:
            L.append(f"temporal {s}: {v['note']}")
    g = r["groups"]
    L.append(f"frame groups (gap <= {r['gap_frames_for_groups']}): {g['groups']} groups from {g['videos']} video(s); {g['groups_spanning_several_splits']} span several splits; images in leaky groups: {g['images_in_leaky_groups']}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset root with train/ valid/ test/ (images/ and labels/)")
    ap.add_argument("--gap", type=int, default=15, help="frames: consecutive frames closer than this form one group")
    ap.add_argument("--examples", type=int, default=10)
    ap.add_argument("--out")
    a = ap.parse_args()
    r = check(Path(a.data), a.gap, a.examples)
    print(format_report(r))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
