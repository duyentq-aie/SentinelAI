"""Frame-by-frame trace of the windowed no-helmet / overload rule for one motorcycle track, from a recorded tracks file.

  python tools/rule_trace.py --track 176 [--tracks tests/fixtures/tracks_test_video_gmc_sparse.jsonl] [--from 95 --to 125]

For every frame in which the helmet model sees the track: heads attached / no-helmet heads, the vote window as the rule sees it
(observations, votes, ratio) and the verdict. Also prints the verdict intervals (= what becomes frame_start..frame_end of an event) and
what a hysteresis (exit threshold lower than the entry threshold) would do to them. Used to analyse M_176 (EXP-005, `docs/PROGRESS.md`).
"""
import argparse
import sys
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from sentinel.config import load_pipeline_config  # noqa: E402
from sentinel.offline import load_tracks  # noqa: E402


def heads_of(p, tid):
    m = next((t for t in p.helmet if t.cls == "motorcycle" and t.id == tid), None)
    if m is None:
        return None
    mx1, my1, mx2, my2 = m.box
    return [t for t in p.helmet if t.cls in ("helmet", "no-helmet") and mx1 <= (t.box[0] + t.box[2]) / 2 <= mx2 and my1 <= (t.box[1] + t.box[3]) / 2 <= my2]


def trace(perceptions, tid, rule, exit_ratio=None):
    """Rows for every frame with the track; `exit_ratio` = hysteresis (flag stays on until ratio < exit_ratio or votes < min_votes)."""
    win = deque(maxlen=rule.window)
    rows, on_h = [], False
    for i, p in enumerate(perceptions):
        hs = heads_of(p, tid)
        if hs is None:
            continue
        if hs:
            win.append(any(h.cls == "no-helmet" for h in hs))
        votes, n = sum(win), len(win)
        ratio = votes / n if n else 0.0
        plain = n > 0 and votes >= rule.min_votes and ratio >= rule.min_ratio
        if exit_ratio is not None:
            on_h = (plain or (on_h and votes >= rule.min_votes and ratio >= exit_ratio))
        rows.append({"frame": i, "heads": len(hs), "no_helmet_heads": sum(h.cls == "no-helmet" for h in hs), "obs": n, "votes": votes,
                     "ratio": round(ratio, 3), "verdict": plain, "verdict_hysteresis": on_h if exit_ratio is not None else None})
    return rows


def intervals(rows, key):
    out, start, last = [], None, None
    for r in rows:
        if r[key] and start is None:
            start = r["frame"]
        if r[key]:
            last = r["frame"]
        if not r[key] and start is not None:
            out.append((start, last)); start = None
    if start is not None:
        out.append((start, last))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=int, required=True, help="helmet-model track id (the number in M_<id>)")
    ap.add_argument("--tracks", default=str(REPO / "tests" / "fixtures" / "tracks_test_video_gmc_sparse.jsonl"))
    ap.add_argument("--from", dest="lo", type=int, default=0)
    ap.add_argument("--to", dest="hi", type=int, default=10**9)
    ap.add_argument("--exit-ratio", type=float, default=0.3, help="exit threshold for the hysteresis what-if")
    a = ap.parse_args()
    rule = load_pipeline_config().rules.helmet
    _, per = load_tracks(a.tracks)
    rows = trace(per, a.track, rule, a.exit_ratio)
    print(f"track M_{a.track}: W={rule.window} K={rule.min_votes} r={rule.min_ratio}   ({len(rows)} frames seen)")
    print("frame heads bad  obs votes ratio  verdict  verdict(hysteresis exit<%.2f)" % a.exit_ratio)
    for r in rows:
        if a.lo <= r["frame"] <= a.hi:
            print(f"{r['frame']:5d} {r['heads']:5d} {r['no_helmet_heads']:3d} {r['obs']:4d} {r['votes']:5d} {r['ratio']:5.2f}  {'ON ' if r['verdict'] else '.  '}     {'ON' if r['verdict_hysteresis'] else '.'}")
    print("verdict intervals (-> event frame_start..frame_end):", intervals(rows, "verdict"))
    print("with hysteresis                                  :", intervals(rows, "verdict_hysteresis"))


if __name__ == "__main__":
    main()
