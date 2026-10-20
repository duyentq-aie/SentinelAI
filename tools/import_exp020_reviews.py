"""Import the by-eye grading of EXP-020 (tools/grade_rows.py -> grading_rows.csv) as review decisions.

  python tools/import_exp020_reviews.py [--out static/violations/reviews.jsonl]

The 23 graded rows (no_helmet / overload; wrong_way skipped) become one review line per version that reported an event in the row (50 events in all: v0 14, current + old model 14,
EXP-015 7, EXP-016 5, current + old model @10 FPS 10), because precision per model needs one decision per model event. Every one is `rejected` (all 23 riders wear a helmet, graded by
the project owner on 2026-10-20). Lines are APPENDED to --out and an event_id that is already there is skipped, so the tool can be run again. The same lines are kept in
experiments/EXP-020-regression-diag/reviews_exp020.jsonl (written only if missing) as the committed source of this data.

event_id = exp020/<version>/<track>/<type> (these events are not in a live events.jsonl: they come from the review runs of EXP-020). `video`, `frame`, `box` locate the vehicle in
test_video.mp4 (resized 1280x720), which is what tools/export_reviews.py crops. `model` = the helmet model file and sha256 of that version.
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
from sentinel.modelinfo import file_sha256  # noqa: E402

SLUG = {"v0": "v0", "current+old model": "current-old", "EXP-015 3cls": "exp015", "EXP-016 4cls": "exp016", "current+old model @10FPS": "current-old-10fps"}
HELMET = {"v0": "helmet_best.pt", "current+old model": "helmet_best.pt", "current+old model @10FPS": "helmet_best.pt",
          "EXP-015 3cls": "experiments/EXP-015-helmet-grouped-3cls/weights/best.pt", "EXP-016 4cls": "experiments/EXP-016-helmet-grouped-4cls/weights/best.pt"}
DECISION = {"dung": "confirmed", "sai": "rejected", "khong_ro": "unclear"}
SOURCE = REPO / "experiments" / "EXP-020-regression-diag" / "reviews_exp020.jsonl"


def build_records(rows, grades, events, reviewed_at, sha=file_sha256):
    """rows: gr.load_rows(); grades: gr.read_grades(); events: gr.load_events(). -> [review record]"""
    out = []
    for r in rows:
        g = grades.get(r["n"])
        if not gr.is_complete(r, g):
            continue
        per = gr.verdicts_of(r, g)
        for ver, e in gr.row_events(r, events).items():
            w = HELMET[ver]
            out.append({"schema": "sentinel.review/1", "event_id": f"exp020/{SLUG[ver]}/{e['track']}/{r['type']}", "decision": DECISION[per[ver]],
                        "note": f"EXP-020 by-eye grading, row {r['n']}" + (": the rider wears a helmet" if per[ver] == "sai" else ""),
                        "reviewed_at": reviewed_at, "event_type": r["type"], "track_id": e["track"], "run": None, "video_id": "test_video", "camera_id": "test_video",
                        "model": {"role": "helmet", "file": w, "sha256": sha(REPO / w), "source": "exp020 version " + ver},
                        "source": "exp020", "version": ver, "row": r["n"], "video": "test_video.mp4", "frame": int(e["frame"]), "box": [int(x) for x in e["box"]]})
    return out


def append_missing(path, records):
    """Append the records whose event_id is not in `path` yet. -> number appended."""
    path = Path(path)
    have = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                have.add(json.loads(line)["event_id"])
            except (ValueError, KeyError):
                pass
    new = [r for r in records if r["event_id"] not in have]
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
    recs = build_records(gr.load_rows(), gr.read_grades(), gr.load_events(), stamp)
    if not SOURCE.exists():
        SOURCE.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in recs), encoding="utf-8", newline="\n")
    print(f"{len(recs)} review lines built from {len(gr.read_grades())} graded rows; {append_missing(out, recs)} appended to {out}")


if __name__ == "__main__":
    main()
