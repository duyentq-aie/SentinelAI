"""Build the manual-upload zip of a grouped split (for Kaggle web upload, no CLI). Read-only on the dataset; the zip is written under kaggle_upload/ (gitignored).

  python tools/make_upload_zip.py --data data/raw/helmet --split-dir experiments/EXP-013-splits/A --out kaggle_upload/helmet_split_a.zip

Zip layout:   data.yaml (original classes, original ids) | SPLIT_INFO.md | manifest.csv
              images/{train,val,test,extra_1080p}/*.jpg   labels/{train,val,test,extra_1080p}/*.txt   (labels byte-identical to the originals)
`extra_1080p` = images whose names carry no video id (`testing_images_*`, 1920x1080, unknown source), taken out of the split proposal and kept apart.
Every image is re-hashed against the manifest before it is written; the zip is deterministic (sorted entries, fixed timestamps, no compression of JPEGs).
"""
import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import yaml

SPLIT_DIR = {"train": "train", "valid": "val", "test": "test"}
EXTRA = "extra_1080p"
STAMP = (2026, 1, 1, 0, 0, 0)


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def plan(data, split_dir):
    data = Path(data)
    split_dir = Path(split_dir)
    manifest = split_dir / "manifest.csv"
    summ = json.loads((split_dir / "summary.json").read_text(encoding="utf-8"))
    if sha256(manifest.read_bytes()) != summ["manifest_sha256"]:
        sys.exit("manifest.csv does not match summary.json")
    names = yaml.safe_load((data / "data.yaml").read_text(encoding="utf-8"))["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    if names != summ["classes"]:
        sys.exit(f"class names/order in data.yaml {names} differ from the manifest's {summ['classes']}")
    items = []
    for r in csv.DictReader(open(manifest, encoding="utf-8")):
        folder = EXTRA if r["video"].startswith("other:") else SPLIT_DIR[r["split"]]
        img = data / r["path"]
        lab = data / Path(r["path"]).parent.parent / "labels" / (img.stem + ".txt")
        items.append({"folder": folder, "img": img, "lab": lab, "sha256": r["sha256"], "video": r["video"], "counts": {c: int(r[f"n_{c}"]) for c in names}})
    return names, items, summ


def info_md(names, items, summ):
    folders = ["train", "val", "test", EXTRA]
    lines = ["# SPLIT_INFO – helmet, video-grouped split A (derived from tools/make_splits.py, seed %s)" % summ["seed"], "",
             "Images are frames of videos `frame_1` … `frame_16`; every video is in exactly one folder. `%s` = 6 images named `testing_images_<n>` (1920x1080, unknown source), kept apart." % EXTRA, "",
             "| folder | images | videos | " + " | ".join(f"{c} (id {i})" for i, c in enumerate(names)) + " |", "|---|---|---|" + "---|" * len(names)]
    for f in folders:
        its = [i for i in items if i["folder"] == f]
        tot = Counter()
        for i in its:
            tot.update(i["counts"])
        lines.append(f"| {f} | {len(its)} | {len({i['video'] for i in its})} | " + " | ".join(str(tot[c]) for c in names) + " |")
    lines += ["", "videos per folder:"]
    for f in folders:
        lines.append(f"- {f}: " + ", ".join(sorted({i['video'] for i in items if i['folder'] == f}, key=lambda v: (len(v), v))))
    lines += ["", f"manifest.csv sha256 (split A, EXP-013): `{summ['manifest_sha256']}`",
              "Labels keep the 4 original classes and ids; a 3-class run must drop class id 3 (`plate`) itself. Source: 'helmet plate detection 4' by Rajab, Roboflow Universe 2024, CC BY 4.0 (verified), https://universe.roboflow.com/rajab/helmet-plate-detection-4; changes: split redone by video."]
    return "\n".join(lines) + "\n"


def build(data, split_dir, out):
    out = Path(out)
    if out.exists():
        sys.exit(f"{out} exists: nothing is overwritten or deleted, choose another name")
    names, items, summ = plan(data, split_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = "# paths relative to this file\ntrain: images/train\nval: images/val\ntest: images/test\n\nnc: %d\nnames: %s\n" % (len(names), json.dumps(names))
    entries = {}
    for it in items:
        b = it["img"].read_bytes()
        if sha256(b) != it["sha256"]:
            sys.exit(f"{it['img']} differs from the manifest")
        entries[f"images/{it['folder']}/{it['img'].name}"] = b
        entries[f"labels/{it['folder']}/{it['lab'].name}"] = it["lab"].read_bytes() if it["lab"].exists() else b""
    entries["data.yaml"] = yaml_text.encode("utf-8")
    entries["SPLIT_INFO.md"] = info_md(names, items, summ).encode("utf-8")
    entries["manifest.csv"] = (Path(split_dir) / "manifest.csv").read_bytes()
    tmp = out.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for name in sorted(entries):
            zi = zipfile.ZipInfo(name, STAMP)
            zi.external_attr = 0o644 << 16
            z.writestr(zi, entries[name])
    tmp.rename(out)
    print(f"{len(items)} images -> {out} ({out.stat().st_size / 2**20:.0f} MiB), sha256 {sha256(out.read_bytes())}")
    return names, items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.data, a.split_dir, a.out)


if __name__ == "__main__":
    main()
