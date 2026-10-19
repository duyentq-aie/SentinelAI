"""Compare two review runs (tools/review_run.py): side-by-side video, diff_events, and an id-independent comparison of event intervals.

  python tools/review_compare.py --a outputs/review_A --b outputs/review_B --out outputs/review_AB

Writes <out>/compare_AB.mp4 (A | B, same frame index = same timeline, each half 960x540), <out>/diff_report.md and <out>/diff.json.
Two comparisons:
  1. tools/diff_events.py logic (sentinel.characterize.diff_events) on the characterization events (label_first_seen / plate_capture), tolerance 3 frames; entities
     are track ids (M_77), which can differ between runs because the helmet model (and so its tracker) differs: a change of id shows up as missing + added.
  2. Interval comparison on events.jsonl, independent of ids: an event of A and an event of B of the SAME TYPE are matched (Hungarian on 1 - IoU of the vehicle
     key box) when their frame ranges overlap within --tol frames (default 15 = 0.5 s) and the boxes overlap (IoU >= 0.3 or one contains the other's centre).
     Reports only-A, only-B, and matched pairs with the start / end shift (same idea as the v4 -> v5 interval comparison of EXP-005).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from review_run import last_records  # noqa: E402
from sentinel.characterize import diff_events  # noqa: E402


def iou(a, b):
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def contains_centre(outer, inner):
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def match_intervals(A, B, tol=15, min_iou=0.3):
    """A, B: lists of event records (events.jsonl). -> (pairs [(a, b)], only_a, only_b). Matching is per type, one-to-one."""
    pairs, used_a, used_b = [], set(), set()
    for typ in sorted({e["type"] for e in A} | {e["type"] for e in B}):
        ia = [i for i, e in enumerate(A) if e["type"] == typ]
        ib = [j for j, e in enumerate(B) if e["type"] == typ]
        cost = np.full((len(ia), len(ib)), 10.0)
        for x, i in enumerate(ia):
            for y, j in enumerate(ib):
                a, b = A[i], B[j]
                if a["frame_start"] - tol <= b["frame_end"] and b["frame_start"] - tol <= a["frame_end"]:
                    v = iou(a["key_box"], b["key_box"])
                    if v >= min_iou or contains_centre(a["key_box"], b["key_box"]) or contains_centre(b["key_box"], a["key_box"]):
                        cost[x, y] = 1.0 - max(v, 1e-3)
        if len(ia) and len(ib):
            for x, y in zip(*linear_sum_assignment(cost)):
                if cost[x, y] < 10.0:
                    pairs.append((A[ia[x]], B[ib[y]]))
                    used_a.add(ia[x])
                    used_b.add(ib[y])
    return pairs, [e for i, e in enumerate(A) if i not in used_a], [e for j, e in enumerate(B) if j not in used_b]


def side_by_side(va, vb, out, fps):
    ca, cb = cv2.VideoCapture(str(va)), cv2.VideoCapture(str(vb))
    w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 540))
    n = 0
    while True:
        oka, fa = ca.read()
        okb, fb = cb.read()
        if not (oka and okb):
            break
        w.write(np.hstack([cv2.resize(fa, (960, 540)), cv2.resize(fb, (960, 540))]))
        n += 1
    w.release()
    return n


def fmt_ev(e, fps):
    return f"{e['type']} {e['track_id']} frames {e['frame_start']}-{e['frame_end']} (t = {e['trigger_frame'] / fps:.2f} s)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=int, default=15)
    a = ap.parse_args()
    A_dir, B_dir, out = Path(a.a), Path(a.b), Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty: nothing is overwritten or deleted")
    out.mkdir(parents=True, exist_ok=True)
    ia, ib = json.loads((A_dir / "run_info.json").read_text(encoding="utf-8")), json.loads((B_dir / "run_info.json").read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(str(next(A_dir.glob("review_*.mp4"))))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    n = side_by_side(next(A_dir.glob("review_*.mp4")), next(B_dir.glob("review_*.mp4")), out / "compare_AB.mp4", fps)

    ca = json.loads((A_dir / "characterization_events.json").read_text(encoding="utf-8"))["events"]
    cb = json.loads((B_dir / "characterization_events.json").read_text(encoding="utf-8"))["events"]
    d3 = diff_events(ca, cb, 3)
    RA, RB = list(last_records(A_dir / "evidence" / "events.jsonl").values()), list(last_records(B_dir / "evidence" / "events.jsonl").values())
    pairs, only_a, only_b = match_intervals(RA, RB, a.tol)
    (out / "diff.json").write_text(json.dumps({"diff_events_tol3": d3, "intervals": {"matched": [{"a": x["id"], "b": y["id"]} for x, y in pairs], "only_a": [e["id"] for e in only_a],
                                                                                     "only_b": [e["id"] for e in only_b], "tol_frames": a.tol}}, indent=1), encoding="utf-8")

    L = ["# Comparison A vs B (EXP-015 review runs)", "",
         f"A = `{ia['helmet_weights']}` ({len(RA)} events), B = `{ib['helmet_weights']}` ({len(RB)} events). Same video, same default config (hysteresis on), only the helmet weights differ. "
         f"Side-by-side video: `compare_AB.mp4` ({n} frames, A left | B right). **Both runs are unvalidated: no ground truth exists, so 'only in A / only in B' says nothing about which is right.**", "",
         "## 1. `diff_events` (characterization events, tolerance 3 frames; entity = track id)",
         f"missing in B: {len(d3['missing'])}, added in B: {len(d3['added'])}, changed: {len(d3['changed'])}", ""]
    for e in d3["missing"]:
        L.append(f"- only A: frame {e['frame']} {e['kind']} {e.get('track') or e.get('label')}")
    for e in d3["added"]:
        L.append(f"- only B: frame {e['frame']} {e['kind']} {e.get('track') or e.get('label')}")
    for c in d3["changed"]:
        L.append(f"- changed {c['kind']} {c['entity']}: " + "; ".join(c["why"]))
    L += ["", "Track ids differ between the runs (different helmet model = different tracker input), so this comparison mostly reports id changes; section 2 does not depend on ids.", "",
          f"## 2. Event intervals, independent of ids (same type, frame ranges within ±{a.tol} frames, boxes overlapping)",
          f"matched {len(pairs)} | only in A {len(only_a)} | only in B {len(only_b)}", "", "### matched (start / end shift = B − A, frames)",
          "| type | A | B | start shift | end shift |", "|---|---|---|---|---|"]
    for x, y in sorted(pairs, key=lambda p: p[0]["frame_start"]):
        L.append(f"| {x['type']} | {x['track_id']} {x['frame_start']}–{x['frame_end']} | {y['track_id']} {y['frame_start']}–{y['frame_end']} | {y['frame_start'] - x['frame_start']:+d} | {y['frame_end'] - x['frame_end']:+d} |")
    L += ["", "### only in A"] + [f"- {fmt_ev(e, fps)}" for e in sorted(only_a, key=lambda e: e["frame_start"])]
    L += ["", "### only in B"] + [f"- {fmt_ev(e, fps)}" for e in sorted(only_b, key=lambda e: e["frame_start"])]
    (out / "diff_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
