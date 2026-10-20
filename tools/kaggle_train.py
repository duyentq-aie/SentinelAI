"""Prepare (and, only on explicit request, launch) the Kaggle re-training of the helmet detector on a video-grouped split.

  python tools/kaggle_train.py prepare --data data/raw/helmet --split-dir experiments/EXP-013-splits/A --out outputs/kaggle/helmet_split_A --slug sentinel-helmet-split-a
  python tools/kaggle_train.py kernel  --dataset-slug sentinel-helmet-split-a --slug sentinel-helmet-train-a --out kaggle/helmet_train_a
  python tools/kaggle_train.py check                                   # is a Kaggle token configured? (existence only, the content is never read or printed)
  python tools/kaggle_train.py push-dataset --dataset-dir ... --user <kaggle-user> --confirm      # NOT run by the assistant
  python tools/kaggle_train.py push-kernel  --kernel-dir ...  --user <kaggle-user> --confirm      # NOT run by the assistant

What `prepare` builds (everything local, gitignored under outputs/): train|valid|test/{images,labels} copied from the dataset according to
manifest.csv, labels reduced to the 3 classes of the old checkpoint (the 4th class `plate` is dropped, see docs/DATA_CARD.md), data.yaml,
manifest.csv, params.json (= the train_args of helmet_best.pt, configs/training/helmet_yolo11s_v1.json) and dataset-metadata.json (private by default).
`kernel` writes the notebook (from kaggle/train_helmet.py) and kernel-metadata.json: private, GPU, internet on (pip install of the pinned ultralytics).
The user name is a placeholder until `--user` is given; the push commands refuse placeholders, a missing token, and missing --confirm.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLACEHOLDER = "<KAGGLE_USER>"
KEEP = {"helmet": 0, "motorcycle": 1, "no-helmet": 2}          # old checkpoint classes, ids unchanged (plate = 3 is dropped)
SPLIT_DIRS = ("train", "valid", "test")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(a):
    data, out = Path(a.data), Path(a.out)
    manifest = Path(a.split_dir) / "manifest.csv"
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    summ = json.loads((Path(a.split_dir) / "summary.json").read_text(encoding="utf-8"))
    if sha256(manifest) != summ["manifest_sha256"]:
        sys.exit("manifest.csv does not match summary.json (edited after it was generated?)")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty: pick a new directory (nothing is deleted or overwritten)")
    dropped = 0
    for r in rows:
        src_img = data / r["path"]
        if sha256(src_img) != r["sha256"]:
            sys.exit(f"{r['path']} differs from the manifest")
        dst_img = out / r["split"] / "images" / src_img.name
        dst_lab = out / r["split"] / "labels" / (src_img.stem + ".txt")
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lab.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img)
        src_lab = data / Path(r["path"]).parent.parent / "labels" / (src_img.stem + ".txt")
        lines = []
        for line in src_lab.read_text(encoding="utf-8").splitlines() if src_lab.exists() else []:
            if line.strip():
                if int(float(line.split()[0])) in KEEP.values():
                    lines.append(line)
                else:
                    dropped += 1
        dst_lab.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    shutil.copy2(manifest, out / "manifest.csv")
    params = json.loads((REPO / "configs" / "training" / "helmet_yolo11s_v1.json").read_text(encoding="utf-8"))
    (out / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    (out / "data.yaml").write_text("path: .\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 3\nnames: [helmet, motorcycle, no-helmet]\n", encoding="utf-8")
    meta = {"title": a.slug.replace("-", " "), "id": f"{PLACEHOLDER}/{a.slug}",
            "licenses": [{"name": "other"}],
            "description": "Helmet/motorcycle/no-helmet detection, video-grouped split. Source images: 'helmet plate detection 4' by Rajab, Roboflow Universe 2024 (CC BY 4.0, verified), https://universe.roboflow.com/rajab/helmet-plate-detection-4; "
                           "changes: split redone by video. Attribution to the original author applies. Private; for re-training the SentinelAI helmet detector."}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{len(rows)} images -> {out}; {dropped} `plate` label lines dropped; classes {list(KEEP)}")


def kernel(a):
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty: pick a new directory")
    out.mkdir(parents=True, exist_ok=True)
    src = (REPO / "kaggle" / "train_helmet.py").read_text(encoding="utf-8")
    cells = []
    for block in re.split(r"(?m)^# %%", src)[1:]:
        md = block.startswith(" [markdown]")
        body = block.split("\n", 1)[1].rstrip("\n") if "\n" in block else ""
        lines = [re.sub(r"^# ?", "", l, count=1) for l in body.split("\n")] if md else body.split("\n")
        cell = {"cell_type": "markdown" if md else "code", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
        if not md:
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (out / "train_helmet.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    kmeta = {"id": f"{PLACEHOLDER}/{a.slug}", "title": a.slug.replace("-", " "), "code_file": "train_helmet.ipynb", "language": "python", "kernel_type": "notebook",
             "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [f"{PLACEHOLDER}/{a.dataset_slug}"], "competition_sources": [], "kernel_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(kmeta, indent=2), encoding="utf-8")
    print(f"notebook + kernel-metadata.json -> {out} (private, GPU, dataset source {PLACEHOLDER}/{a.dataset_slug})")


def token_present():
    """Existence check only: the token file/env content is never read or printed."""
    home = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (home / "kaggle.json").exists() or bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) or bool(os.environ.get("KAGGLE_API_TOKEN"))


def guarded(a, what):
    if not a.confirm:
        sys.exit(f"{what}: refusing without --confirm (this would upload to / run on Kaggle)")
    if not a.user or "<" in a.user:
        sys.exit("--user must be your real Kaggle user name (the metadata still has the placeholder)")
    if not token_present():
        sys.exit("no Kaggle token found (~/.kaggle/kaggle.json or KAGGLE_* variables): stop here, nothing was sent")
    if shutil.which("kaggle") is None:
        sys.exit("the `kaggle` CLI is not installed (pip install kaggle in a separate environment)")


def fill_user(path, user):
    p = Path(path)
    t = p.read_text(encoding="utf-8")
    p.write_text(t.replace(PLACEHOLDER, user), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", required=True)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--slug", required=True)
    k = sub.add_parser("kernel")
    k.add_argument("--dataset-slug", required=True, help="slug given to `prepare --slug`")
    k.add_argument("--slug", required=True)
    k.add_argument("--out", required=True)
    sub.add_parser("check")
    for name in ("push-dataset", "push-kernel"):
        q = sub.add_parser(name)
        q.add_argument("--dataset-dir" if name == "push-dataset" else "--kernel-dir", dest="dir", required=True)
        q.add_argument("--user")
        q.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a)
    elif a.cmd == "kernel":
        kernel(a)
    elif a.cmd == "check":
        print("Kaggle token configured:", token_present(), "| kaggle CLI installed:", shutil.which("kaggle") is not None)
    else:
        guarded(a, a.cmd)
        work = REPO / "outputs" / "kaggle" / "_push" / Path(a.dir).name            # the user name is filled into a COPY (committed files keep the placeholder)
        if a.cmd == "push-kernel":
            if work.exists():
                sys.exit(f"{work} exists already: remove it yourself, nothing is deleted here")
            shutil.copytree(a.dir, work)
            target = work
        else:
            target = Path(a.dir)                                                    # the dataset directory is already under outputs/ (gitignored)
        meta = target / ("dataset-metadata.json" if a.cmd == "push-dataset" else "kernel-metadata.json")
        fill_user(meta, a.user)
        cmd = ["kaggle", "datasets", "create", "-p", str(target), "--dir-mode", "zip"] if a.cmd == "push-dataset" else ["kaggle", "kernels", "push", "-p", str(target)]
        sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
