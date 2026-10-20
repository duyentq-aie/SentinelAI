"""EXP-025: events of the old default helmet model (helmet_best.pt) vs the new default (EXP-015 best.pt) on test_video, same code, same config otherwise (both runs: tools/review_run.py).
  python experiments/EXP-025-default-model-exp015/compare.py outputs/review_old_now outputs/review_exp015_now
Writes diff_events.txt (tools/diff_events.py on the label/plate events of both runs) and events_old_vs_exp015.json: every event with its time span (frames, seconds) in each run and whether
the other run has a related event (same type, within 30 frames, boxes overlapping by IoU >= 0.2 or one centre inside the other, the rule of the contact sheets).
All 23 no_helmet/overload events of the old model and the ones of EXP-015 were graded wrong by eye (EXP-020): this table is about WHICH false alarms disappear / appear, not about accuracy."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from contact_sheet_versions import load_run, related  # noqa: E402

OUT = Path(__file__).parent
FPS = 30.0


def main(old, new):
    old, new = Path(old), Path(new)
    A, B = load_run(old), load_run(new)
    rows, used = [], set()
    for a in sorted(A, key=lambda e: e["frame"]):
        best = max(((related(a, b), i) for i, b in enumerate(B) if i not in used), default=(0, None))
        if best[0] > 0:
            used.add(best[1])
        rows.append({"type": a["type"], "old": a, "new": B[best[1]] if best[0] > 0 else None})
    for i, b in enumerate(B):
        if i not in used:
            rows.append({"type": b["type"], "old": None, "new": b})
    span = lambda e: None if e is None else {"track": e["track"], "frames": [e["start"], e["end"]], "seconds": [round(e["start"] / FPS, 2), round(e["end"] / FPS, 2)], "duration_s": round((e["end"] - e["start"] + 1) / FPS, 2)}
    out = []
    for r in sorted(rows, key=lambda r: (r["old"] or r["new"])["frame"]):
        o, n = span(r["old"]), span(r["new"])
        ov = None
        if o and n:
            lo, hi = max(o["frames"][0], n["frames"][0]), min(o["frames"][1], n["frames"][1])
            ov = round(max(0, hi - lo + 1) / FPS, 2)
        out.append({"type": r["type"], "old_model": o, "exp015": n, "overlap_s": ov, "status": "both" if o and n else ("only old" if o else "only EXP-015")})
    summary = {"events_old": len(A), "events_exp015": len(B), "both": sum(1 for r in out if r["status"] == "both"), "only_old": sum(1 for r in out if r["status"] == "only old"),
               "only_exp015": sum(1 for r in out if r["status"] == "only EXP-015"),
               "by_type": {t: {"old": sum(1 for e in A if e["type"] == t), "exp015": sum(1 for e in B if e["type"] == t)} for t in sorted({e["type"] for e in A + B})},
               "clip_seconds": round(787 / FPS, 3)}
    (OUT / "events_old_vs_exp015.json").write_text(json.dumps({"summary": summary, "events": out}, indent=1), encoding="utf-8")
    d = subprocess.run([sys.executable, str(REPO / "tools" / "diff_events.py"), str(old / "characterization_events.json"), str(new / "characterization_events.json")], capture_output=True, text=True)
    (OUT / "diff_events.txt").write_text(d.stdout + d.stderr, encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for r in out:
        print(r["status"], r["type"], r["old_model"] and r["old_model"]["seconds"], r["exp015"] and r["exp015"]["seconds"], "overlap", r["overlap_s"])


if __name__ == "__main__":
    main(*sys.argv[1:3])
