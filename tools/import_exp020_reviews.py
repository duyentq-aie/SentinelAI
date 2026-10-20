"""Import the by-eye grading of EXP-020 (tools/grade_rows.py -> grading_rows.csv) as review decisions.

  python tools/import_exp020_reviews.py [--out static/violations/reviews.jsonl]

The 23 graded rows (no_helmet / overload; wrong_way skipped) become one review line per version that reported an event in the row (50 events in all: v0 14, current + old model 14,
EXP-015 7, EXP-016 5, current + old model @10 FPS 10), because precision per model needs one decision per model event. Every one is `rejected` (all 23 riders wear a helmet, graded by
the project owner on 2026-10-20). Lines are APPENDED to --out and an event_id that is already there is skipped, so the tool can be run again. The same lines are kept in
experiments/EXP-020-regression-diag/reviews_exp020.jsonl (written only if missing) as the committed source of this data.

event_id = exp020/<version>/<track>/<type> (these events are not in a live events.jsonl: they come from the review runs of EXP-020). `video`, `frame`, `box` locate the vehicle in
test_video.mp4 (resized 1280x720). `model` = the helmet model file and sha256 of that version.

Revision 2 (2026-10-20): the head. A no_helmet review means "this head was called no-helmet; the right label is helmet" (corrected_label), so the head box is needed. The EXP-020
events do not store it, so it is RECONSTRUCTED: the no-helmet head the same model detected (conf 0.55, the pipeline's) whose centre lies in the event's vehicle box, at the event frame
or the nearest frame within +-3 (closest in time, then best conf). Source of the detections: the recorded perception fixture of that model (tests/fixtures/tracks_test_video.jsonl =
the old helmet_best.pt, covers v0, current + old model and @10 FPS; tracks_test_video_exp015.jsonl = EXP-015) -> method "fixture:<file>"; EXP-016 has no fixture, so its model
is run again on that frame -> method "model_rerun:<weights>". Fields: head_box, head_class ("no-helmet"), head_frame, head_reconstructed {method, frame, event_frame}. Not found -> head_box null
(the export lists the event in needs_head_box.csv). Overload rows have no head. A newer `import_rev` appends a NEWER line for an event already in reviews.jsonl (append-only, nothing edited).
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import grade_rows as gr  # noqa: E402
from apps.web import review  # noqa: E402
from sentinel.modelinfo import file_sha256  # noqa: E402

SLUG = {"v0": "v0", "current+old model": "current-old", "EXP-015 3cls": "exp015", "EXP-016 4cls": "exp016", "current+old model @10FPS": "current-old-10fps"}
HELMET = {"v0": "helmet_best.pt", "current+old model": "helmet_best.pt", "current+old model @10FPS": "helmet_best.pt",
          "EXP-015 3cls": "experiments/EXP-015-helmet-grouped-3cls/weights/best.pt", "EXP-016 4cls": "experiments/EXP-016-helmet-grouped-4cls/weights/best.pt"}
DECISION = {"dung": "confirmed", "sai": "rejected", "khong_ro": "unclear"}
SOURCE = REPO / "experiments" / "EXP-020-regression-diag" / "reviews_exp020.jsonl"
IMPORT_REV = 2
FIXTURE = {"v0": "tests/fixtures/tracks_test_video.jsonl", "current+old model": "tests/fixtures/tracks_test_video.jsonl", "current+old model @10FPS": "tests/fixtures/tracks_test_video.jsonl",
           "EXP-015 3cls": "tests/fixtures/tracks_test_video_exp015.jsonl", "EXP-016 4cls": None}
WINDOW = 3


def head_from_perceptions(perceptions, frame, box, window=WINDOW):
    """The no-helmet head (centre inside the vehicle box) of `perceptions` (list of Perception, one per frame) at `frame` or the closest frame within +-window. -> (box, frame) or None."""
    x1, y1, x2, y2 = box
    for d in sorted(range(-window, window + 1), key=lambda d: (abs(d), d)):
        f = frame + d
        if not 0 <= f < len(perceptions):
            continue
        heads = [(t.conf, list(t.box)) for t in perceptions[f].helmet if t.cls == "no-helmet" and x1 <= (t.box[0] + t.box[2]) / 2 <= x2 and y1 <= (t.box[1] + t.box[3]) / 2 <= y2]
        if heads:
            return max(heads)[1], f
    return None


def make_head_finder(conf=0.55):
    """-> finder(version, frame, box) -> {"box", "frame", "method"} or None; loads each fixture / model once."""
    from sentinel.offline import load_tracks
    cache = {}

    def finder(version, frame, box):
        fx = FIXTURE[version]
        if fx:
            if fx not in cache:
                cache[fx] = load_tracks(REPO / fx)[1]
            r = head_from_perceptions(cache[fx], frame, box)
            return {"box": [int(v) for v in r[0]], "frame": r[1], "method": f"fixture:{Path(fx).name}"} if r else None
        w = HELMET[version]                                                                         # no fixture: run the model again on the frames around the event
        if w not in cache:
            from ultralytics import YOLO
            import cv2
            cache[w] = (YOLO(str(REPO / w)), cv2.VideoCapture(str(REPO / "test_video.mp4")))
        model, cap = cache[w]
        import cv2
        x1, y1, x2, y2 = box
        for d in sorted(range(-WINDOW, WINDOW + 1), key=lambda d: (abs(d), d)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame + d)
            ok, img = cap.read()
            if not ok:
                continue
            res = model.predict(cv2.resize(img, (1280, 720)), conf=conf, verbose=False)[0]
            heads = [(float(c), [int(v) for v in b]) for b, c, k in zip(res.boxes.xyxy.cpu().tolist(), res.boxes.conf.cpu().tolist(), res.boxes.cls.cpu().tolist())
                     if model.names[int(k)] == "no-helmet" and x1 <= (b[0] + b[2]) / 2 <= x2 and y1 <= (b[1] + b[3]) / 2 <= y2]
            if heads:
                return {"box": max(heads)[1], "frame": frame + d, "method": f"model_rerun:{w}"}
        return None
    return finder


def build_records(rows, grades, events, reviewed_at, sha=file_sha256, finder=None):
    """rows: gr.load_rows(); grades: gr.read_grades(); events: gr.load_events(); finder: make_head_finder() (None = no head reconstruction). -> [review record]"""
    out = []
    for r in rows:
        g = grades.get(r["n"])
        if not gr.is_complete(r, g):
            continue
        per = gr.verdicts_of(r, g)
        for ver, e in gr.row_events(r, events).items():
            w = HELMET[ver]
            head = finder(ver, int(e["frame"]), e["box"]) if (finder and r["type"] == "no_helmet") else None
            decision = DECISION[per[ver]]
            out.append({"schema": "sentinel.review/1", "event_id": f"exp020/{SLUG[ver]}/{e['track']}/{r['type']}", "decision": decision,
                        "note": f"EXP-020 by-eye grading, row {r['n']}" + (": the rider wears a helmet" if per[ver] == "sai" else ""),
                        "reviewed_at": reviewed_at, "event_type": r["type"], "track_id": e["track"], "run": None, "video_id": "test_video", "camera_id": "test_video",
                        "model": {"role": "helmet", "file": w, "sha256": sha(REPO / w), "source": "exp020 version " + ver},
                        "source": "exp020", "version": ver, "row": r["n"], "video": "test_video.mp4", "frame": int(e["frame"]), "box": [int(x) for x in e["box"]], "import_rev": IMPORT_REV,
                        "corrected_label": review.corrected_label(r["type"], decision), "not_head": False,
                        "head_box": head["box"] if head else None, "head_class": "no-helmet" if head else None, "head_frame": head["frame"] if head else None,
                        "head_reconstructed": ({"method": head["method"], "frame": head["frame"], "event_frame": int(e["frame"])} if head else None)})
    return out


def append_missing(path, records):
    """Append the records that are not in `path` yet, or whose last line there has an older `import_rev` (a NEWER line is appended; nothing is edited). -> number appended."""
    path = Path(path)
    have = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                have[r["event_id"]] = r.get("import_rev", 1)
            except (ValueError, KeyError):
                pass
    new = [r for r in records if have.get(r["event_id"], 0) < r.get("import_rev", 1)]
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(new)


def main():
    from sentinel.config import load_pipeline_config
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="reviews.jsonl to append to (default: <evidence dir of the default config>/reviews.jsonl)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else REPO / load_pipeline_config().evidence.dir / "reviews.jsonl"
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(gr.OUT_CSV.stat().st_mtime))
    recs = build_records(gr.load_rows(), gr.read_grades(), gr.load_events(), stamp, finder=make_head_finder())
    if True:                                                                                           # the committed source file = the current revision of the imported lines
        SOURCE.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in recs), encoding="utf-8", newline="\n")
    print(f"{len(recs)} review lines built from {len(gr.read_grades())} graded rows; {append_missing(out, recs)} appended to {out}")


if __name__ == "__main__":
    main()
