"""Export the reviewed events as labelled crops in YOLO format: the data of the NEXT training round (positives and hard negatives).

  python tools/export_reviews.py [--reviews static/violations/reviews.jsonl] [--evidence static/violations] [--out data/review_export]

For every event whose LATEST decision is `confirmed` or `rejected` (`unclear` is left out) one crop of the vehicle (event key box + a margin) is written:
  images/positives/<id>.jpg        confirmed: the vehicle really did the violation; labels/positives/<id>.txt has ONE line "<class> cx cy w h" (normalised, in the crop):
                                   the whole vehicle box, class = the violation type (names in dataset.yaml: no_helmet, overload, red_light)
  images/hard_negatives/<id>.jpg   rejected: the detector said "violation" and a person said no; labels/hard_negatives/<id>.txt is EMPTY (YOLO background image).
                                   The manifest says which type was falsely reported (`event_type`)
  manifest.csv, dataset.yaml, README.md
Sources: live events (evidence folder: <id>/frame.jpg + key_box of events.jsonl) and imported reviews (e.g. EXP-020: crop of `frame`/`box` from the video `video`).
IMPORTANT: these are VEHICLE-level crops. The head boxes (helmet / no-helmet) are NOT in the event records, so a head-level training set needs one more labelling pass
(model-assisted pre-labels + review). A hard negative is empty for ALL classes: do not train a helmet class on it blindly.
LEAKAGE: crops of the same video / camera are near-duplicates of each other. When this data is used to train, SPLIT IT BY VIDEO / CAMERA (the `group` column of manifest.csv), never by crop or
by frame, and keep the groups of the validation/test sets out of training. Nothing is deleted: the output folder must be new or empty.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
CLASSES = ["no_helmet", "overload", "red_light"]
README = """# review_export

Labelled vehicle crops made from reviewed events by `tools/export_reviews.py`: `images/positives` (confirmed violations, one YOLO label line each), `images/hard_negatives`
(rejected = false alarms, empty label file = background). `manifest.csv` has one row per crop (event, decision, type, model + sha256 that made the event, `group`, note).

**Split by video / camera, never by crop.** Crops of one video or camera are near-duplicates (same scene, same people, neighbouring frames). When training, put whole groups
(`group` column) into train OR validation OR test, and keep the groups of your validation/test sets out of training, otherwise the scores are inflated (see EXP-011 for what that
cost the first helmet model).

These are vehicle-level crops: head boxes are not recorded in the events. A hard negative has an empty label for every class.
"""


def latest_reviews(path):
    last = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("event_id") and r.get("decision"):
                last[r["event_id"]] = r
    return last


def live_events(evidence_dir):
    last = {}
    f = Path(evidence_dir) / "events.jsonl"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("id") and r.get("status") != "evicted":
                last[r["id"]] = r
    return last


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


class Videos:
    def __init__(self, root):
        self.root, self.caps = Path(root), {}

    def frame(self, name, index, size=(1280, 720)):
        cap = self.caps.get(name)
        if cap is None:
            cap = self.caps[name] = cv2.VideoCapture(str(self.root / name))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, f = cap.read()
        return cv2.resize(f, size) if ok else None


def crop_with_margin(img, box, margin):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = img.shape[:2]
    X1, Y1, X2, Y2 = int(max(0, x1 - margin * w)), int(max(0, y1 - margin * h)), int(min(W, x2 + margin * w)), int(min(H, y2 + margin * h))
    if X2 - X1 < 8 or Y2 - Y1 < 8:
        return None, None
    inner = [max(0, x1 - X1), max(0, y1 - Y1), min(X2 - X1, x2 - X1), min(Y2 - Y1, y2 - Y1)]
    return img[Y1:Y2, X1:X2].copy(), inner


def yolo_line(cls, inner, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = inner
    return f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def export(reviews, evidence_dir, out, video_root=REPO, margin=0.15):
    """-> summary dict. Raises FileExistsError when `out` is not empty (nothing is overwritten)."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: nothing is overwritten or deleted (choose a new --out)")
    latest = latest_reviews(reviews)
    events = live_events(evidence_dir)
    videos = Videos(video_root)
    rows, skipped = [], {"unclear": 0, "no_image": 0}
    for eid, r in sorted(latest.items()):
        if r["decision"] not in ("confirmed", "rejected"):
            skipped["unclear"] += 1
            continue
        img, box = None, None
        if r.get("video") and r.get("box") is not None and r.get("frame") is not None:                       # imported review: crop from the video
            frame = videos.frame(r["video"], int(r["frame"]))
            img, box = frame, r["box"]
            src = {"video_id": r.get("video_id") or Path(r["video"]).stem, "camera_id": r.get("camera_id") or "", "frame": r["frame"], "run": r.get("run")}
        elif eid in events:                                                                                    # live event: the saved evidence frame
            e = events[eid]
            p = Path(evidence_dir) / eid / "frame.jpg"
            img = cv2.imread(str(p)) if p.exists() else None
            box = e.get("key_box")
            src = {"video_id": e.get("video_id") or "", "camera_id": e.get("camera_id") or "", "frame": e.get("trigger_frame"), "run": e.get("run")}
        if img is None or box is None:
            skipped["no_image"] += 1
            continue
        crop, inner = crop_with_margin(img, box, margin)
        if crop is None:
            skipped["no_image"] += 1
            continue
        pos = r["decision"] == "confirmed"
        bucket = "positives" if pos else "hard_negatives"
        name = safe(eid)
        (out / "images" / bucket).mkdir(parents=True, exist_ok=True)
        (out / "labels" / bucket).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "images" / bucket / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        etype = r.get("event_type") or ""
        label = yolo_line(CLASSES.index(etype), inner, crop.shape) if pos and etype in CLASSES else ""
        (out / "labels" / bucket / f"{name}.txt").write_text(label + ("\n" if label else ""), encoding="utf-8", newline="\n")
        m = r.get("model") or {}
        rows.append({"file": f"images/{bucket}/{name}.jpg", "decision": r["decision"], "event_type": etype, "group": src["video_id"] or src["camera_id"] or "unknown",
                     "video_id": src["video_id"], "camera_id": src["camera_id"], "event_id": eid, "run": src["run"], "track": r.get("track_id"), "frame": src["frame"],
                     "crop_w": crop.shape[1], "crop_h": crop.shape[0], "model_file": m.get("file"), "model_sha256": m.get("sha256"), "reviewed_at": r.get("reviewed_at"),
                     "note": r.get("note", ""), "source": r.get("source", "live")})
    out.mkdir(parents=True, exist_ok=True)
    keys = ["file", "decision", "event_type", "group", "video_id", "camera_id", "event_id", "run", "track", "frame", "crop_w", "crop_h", "model_file", "model_sha256", "reviewed_at", "note", "source"]
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (out / "dataset.yaml").write_text("# vehicle-level crops of reviewed events; split by `group` of manifest.csv (video / camera), never by crop\npath: .\ntrain: images/positives\n"
                                      "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)), encoding="utf-8", newline="\n")
    (out / "README.md").write_text(README, encoding="utf-8", newline="\n")
    return {"positives": sum(1 for r in rows if r["decision"] == "confirmed"), "hard_negatives": sum(1 for r in rows if r["decision"] == "rejected"),
            "groups": sorted({r["group"] for r in rows}), "skipped": skipped, "out": str(out)}


def main():
    from sentinel.config import load_pipeline_config
    ev = REPO / load_pipeline_config().evidence.dir
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviews", default=str(ev / "reviews.jsonl"))
    ap.add_argument("--evidence", default=str(ev))
    ap.add_argument("--out", default=str(REPO / "data" / "review_export"))
    ap.add_argument("--video-root", default=str(REPO))
    ap.add_argument("--margin", type=float, default=0.15)
    a = ap.parse_args()
    try:
        print(json.dumps(export(a.reviews, a.evidence, a.out, a.video_root, a.margin), indent=1))
    except FileExistsError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
