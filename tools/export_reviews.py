"""Export the reviewed events as detector training data in YOLO format: whole frames with the CORRECTED label of the head that the model got wrong (or right).

  python tools/export_reviews.py [--reviews static/violations/reviews.jsonl] [--evidence static/violations] [--out data/review_export]

What a review decision means for a no_helmet event (the model called a head "no-helmet"), see apps/web/review.py:
  confirmed            -> the right label is no-helmet
  rejected             -> the model was wrong; the right label is HELMET (default)
  rejected + not_head  -> not a head at all / the wrong person: the right label is background
  unclear              -> not exported
Output (a NEW or empty folder; nothing is overwritten or deleted):
  images/<id>.jpg, labels/<id>.txt   the WHOLE frame (not a vehicle crop) + ONE label line "<class> cx cy w h" (normalised): the head box with the corrected label (helmet = 0, no-helmet = 2,
                                     the class ids of the 3-class helmet model; motorcycle = 1 is not written). The head box is the one stored in the event (`head_box`) or, for imported events
                                     (EXP-020), the reconstructed one (head_source says which). manifest.csv has one row per image.
  needs_head_box.csv                 no_helmet reviews (confirmed / rejected) that have no head box (events made before head_box was stored): NO image and NO label are written for them
  background.csv                     rejected + not_head: the head box region is background; no label file is written (an empty label file would turn the WHOLE frame into background)
  metadata_only.csv                  overload / red_light reviews: metadata only, no detector label (overload is a count of riders, red light is a line crossing: neither is a head box)
  duplicates.csv, conflicts.csv      the same head reviewed more than once (several model versions, +-5 frames, IoU > 0.5): one image is kept; when the corrected labels disagree none is exported
  dataset.yaml, README.md
An empty label file is NEVER written for a frame that has a person's head in it (see README.md of the export for why).
PARTIAL LABELS: an image carries only the head that was reviewed; the other riders in the frame are NOT labelled. Pre-label them (model-assisted, then review) before training, or you teach the
detector that they are background.
LEAKAGE: images of the same video / camera are near-duplicates. SPLIT BY VIDEO / CAMERA (the `group` column), never by image or frame.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from apps.web.review import corrected_label  # noqa: E402

CLASS_ID = {"helmet": 0, "no-helmet": 2}
MIN_IOU, MAX_DFRAME = 0.5, 5
README = """# review_export

Whole frames of reviewed `no_helmet` events with the **corrected label of the head** in YOLO format (`images/`, `labels/`, one label line per image). Class ids are those of the
3-class helmet model: 0 helmet, 1 motorcycle (never written here), 2 no-helmet.

## What the label is
| review | right label | exported as |
|---|---|---|
| confirmed | no-helmet | `labels/<id>.txt`: class 2 + the head box |
| rejected | **helmet** | `labels/<id>.txt`: class 0 + the head box |
| rejected, "not a head / wrong person" | background | listed in `background.csv`, no label file |
| unclear | - | not exported |
| overload / red_light (any decision) | - | `metadata_only.csv`, no detector label |
| no head box stored (events from before head_box existed) | - | `needs_head_box.csv`, no label file |

## Rejecting a no-helmet event means "relabel it HELMET", NOT "background"
The head in a rejected event is a person who WEARS a helmet. If its label were an empty file, the trainer would treat the whole image as background, so the correct helmet
detection on that head would be counted as a false positive and the model would be pushed to NOT detect the helmet there, the opposite of what the review says. The right correction is
the label `helmet` on that head. Only a detection that is not a head at all (a bag, a shadow, the wrong person) is background, and that case is listed in `background.csv` instead of being
written as an empty label.

## Partial labels
An image carries ONE head label, the reviewed one. The other riders in the same frame are unlabelled. Training on it as is would teach the detector that they are background. Pre-label the other
heads (model-assisted, then reviewed by a person) or use the images only after that.

## Split by video / camera, never by image
Images of one video or camera are near-duplicates (same scene, same people, neighbouring frames). Put whole groups (the `group` column of `manifest.csv`) into train OR validation OR test, and keep
the groups of your validation/test sets out of training; otherwise the scores are inflated (EXP-011).

`head_source`: `event` = the head box stored in the event record; `reconstructed:...` = rebuilt after the fact (EXP-020: from the recorded detections of the same model, or by running it again on
the frame), see tools/import_exp020_reviews.py.
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
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


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


def yolo_line(cls, box, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))
    return f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def write_csv(path, rows, keys):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def export(reviews, evidence_dir, out, video_root=REPO):
    """-> summary dict. Raises FileExistsError when `out` is not empty."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: nothing is overwritten or deleted (choose a new --out)")
    latest = latest_reviews(reviews)
    events = live_events(evidence_dir)
    videos = Videos(video_root)
    cand, needs, background, meta, summary = [], [], [], [], {"unclear": 0, "no_image": 0}
    for eid, r in sorted(latest.items()):
        if r["decision"] == "unclear":
            summary["unclear"] += 1
            continue
        etype = r.get("event_type") or (events.get(eid) or {}).get("type") or ""
        e = events.get(eid) or {}
        m = r.get("model") or {}
        base = {"event_id": eid, "event_type": etype, "decision": r["decision"], "track": r.get("track_id"), "video_id": r.get("video_id") or e.get("video_id") or "",
                "camera_id": r.get("camera_id") or e.get("camera_id") or "", "model_file": m.get("file"), "model_sha256": m.get("sha256"), "reviewed_at": r.get("reviewed_at"),
                "note": r.get("note", ""), "source": r.get("source", "live")}
        base["group"] = base["video_id"] or base["camera_id"] or "unknown"
        if etype != "no_helmet":
            meta.append({**base, "frame": r.get("frame", e.get("trigger_frame")), "box": r.get("box", e.get("key_box"))})
            continue
        label = r.get("corrected_label") or corrected_label(etype, r["decision"], bool(r.get("not_head")))
        head = r.get("head_box") or e.get("head_box")
        rec = r.get("head_reconstructed")
        head_source = f"reconstructed:{rec['method']}" if (r.get("head_box") and rec) else "event"
        frame_no = r.get("head_frame") if r.get("head_frame") is not None else (e.get("trigger_frame") if e else r.get("frame"))
        where = {"frame": frame_no, "frame_file": f"{eid}/frame.jpg" if e else "", "video": r.get("video", "")}
        if label == "background":
            background.append({**base, **where, "head_box": head, "corrected_label": label})
            continue
        if not head:
            needs.append({**base, **where, "corrected_label": label})
            continue
        cand.append({**base, **where, "head_box": [int(v) for v in head], "corrected_label": label, "head_source": head_source, "frame": frame_no, "live": bool(e) and not r.get("video")})
    # the same head reviewed more than once (several versions of one rider): keep one image, drop disagreements
    kept, dups, conflicts = [], [], []
    for c in cand:
        same = next((k for k in kept if k["group"] == c["group"] and abs(int(k["frame"]) - int(c["frame"])) <= MAX_DFRAME and iou(k["head_box"], c["head_box"]) > MIN_IOU), None)
        if same is None:
            kept.append(c)
        elif same["corrected_label"] == c["corrected_label"]:
            dups.append({"event_id": c["event_id"], "duplicate_of": same["event_id"], "label": c["corrected_label"]})
        else:
            conflicts.append({"event_id": c["event_id"], "other_event_id": same["event_id"], "label": c["corrected_label"], "other_label": same["corrected_label"]})
            same["conflict"] = True
    for k in kept:
        if k.get("conflict"):
            conflicts.append({"event_id": k["event_id"], "other_event_id": "(see the conflicting rows)", "label": k["corrected_label"], "other_label": "(differs)"})
    rows = []
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for c in kept:
        if c.get("conflict"):
            continue
        if c["live"]:
            p = Path(evidence_dir) / c["event_id"] / "frame.jpg"
            img = cv2.imread(str(p)) if p.exists() else None
        else:
            img = videos.frame(c["video"], int(c["frame"])) if c["video"] else None
        if img is None:
            summary["no_image"] += 1
            needs.append({**{k: c.get(k) for k in ("event_id", "event_type", "decision", "track", "video_id", "camera_id", "model_file", "model_sha256", "reviewed_at", "note", "source", "group", "frame",
                                                    "frame_file", "video", "corrected_label")}, "note": "image not available"})
            continue
        name = safe(c["event_id"])
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / f"{name}.txt").write_text(yolo_line(CLASS_ID[c["corrected_label"]], c["head_box"], img.shape) + "\n", encoding="utf-8", newline="\n")
        rows.append({**c, "file": f"images/{name}.jpg", "label": c["corrected_label"], "head_box": " ".join(map(str, c["head_box"])), "partial_labels": "yes"})
    mkeys = ["file", "label", "event_id", "event_type", "decision", "group", "video_id", "camera_id", "frame", "head_box", "head_source", "model_file", "model_sha256", "reviewed_at", "note", "source", "partial_labels"]
    common = ["event_id", "event_type", "decision", "corrected_label", "track", "group", "video_id", "camera_id", "frame", "frame_file", "video", "head_box", "model_file", "model_sha256", "reviewed_at", "note", "source"]
    write_csv(out / "manifest.csv", rows, mkeys)
    write_csv(out / "needs_head_box.csv", needs, common)
    write_csv(out / "background.csv", background, common)
    write_csv(out / "metadata_only.csv", meta, ["event_id", "event_type", "decision", "track", "group", "video_id", "camera_id", "frame", "box", "model_file", "model_sha256", "reviewed_at", "note", "source"])
    write_csv(out / "duplicates.csv", dups, ["event_id", "duplicate_of", "label"])
    write_csv(out / "conflicts.csv", conflicts, ["event_id", "other_event_id", "label", "other_label"])
    (out / "dataset.yaml").write_text("# whole frames, ONE reviewed head per image (partial labels!); split by `group` of manifest.csv (video / camera), never by image\npath: .\ntrain: images\n"
                                      "names:\n  0: helmet\n  1: motorcycle\n  2: no-helmet\n", encoding="utf-8", newline="\n")
    (out / "README.md").write_text(README, encoding="utf-8", newline="\n")
    return {"images": len(rows), "helmet": sum(1 for r in rows if r["label"] == "helmet"), "no-helmet": sum(1 for r in rows if r["label"] == "no-helmet"), "needs_head_box": len(needs),
            "background": len(background), "metadata_only": len(meta), "duplicates": len(dups), "conflicts": len(conflicts), "skipped": summary, "groups": sorted({r["group"] for r in rows}),
            "out": str(out)}


def main():
    from sentinel.config import load_pipeline_config
    ev = REPO / load_pipeline_config().evidence.dir
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviews", default=str(ev / "reviews.jsonl"))
    ap.add_argument("--evidence", default=str(ev))
    ap.add_argument("--out", default=str(REPO / "data" / "review_export"))
    ap.add_argument("--video-root", default=str(REPO))
    a = ap.parse_args()
    try:
        print(json.dumps(export(a.reviews, a.evidence, a.out, a.video_root), indent=1))
    except FileExistsError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
