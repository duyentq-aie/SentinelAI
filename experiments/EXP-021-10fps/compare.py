"""30 FPS (outputs/review_30fps_calls, every frame) vs 10 FPS (outputs/review_10fps, every 3rd frame): diff_events and event intervals (original frame indices).
Usage: python experiments/EXP-021-10fps/compare.py   -> experiments/EXP-021-10fps/compare.json and compare.md"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from review_compare import match_intervals  # noqa: E402
from review_run import last_records  # noqa: E402
from sentinel.characterize import diff_events  # noqa: E402

A, B = REPO / "outputs" / "review_30fps_calls", REPO / "outputs" / "review_10fps"
K = json.loads((B / "run_info.json").read_text())["stride"]
ca = json.loads((A / "characterization_events.json").read_text())["events"]
cb = json.loads((B / "characterization_events.json").read_text())["events"]
d = {tol: diff_events(ca, cb, tol) for tol in (3, 6)}


def records(folder, scale):
    out = []
    for r in last_records(folder / "evidence" / "events.jsonl").values():
        r = dict(r)
        for k in ("frame_start", "frame_end", "trigger_frame"):
            r[k] = r[k] * scale
        out.append(r)
    return out


RA, RB = records(A, 1), records(B, K)
pairs, only_a, only_b = match_intervals(RA, RB, 15)
rows = [{"type": x["type"], "a": f"{x['track_id']} {x['frame_start']}-{x['frame_end']}", "b": f"{y['track_id']} {y['frame_start']}-{y['frame_end']}",
         "start_shift_frames": y["frame_start"] - x["frame_start"], "end_shift_frames": y["frame_end"] - x["frame_end"],
         "plate_attempts": [(x.get("plate") or {}).get("attempts"), (y.get("plate") or {}).get("attempts")],
         "plate_found_frame": [(x.get("plate") or {}).get("frame"), (y.get("plate") or {}).get("frame", None) and (y.get("plate") or {}).get("frame") * K]} for x, y in pairs]
res = {"stride": K, "events_30fps": len(RA), "events_10fps": len(RB), "matched": len(pairs), "only_30fps": [f"{e['type']} {e['track_id']} {e['frame_start']}-{e['frame_end']}" for e in only_a],
       "only_10fps": [f"{e['type']} {e['track_id']} {e['frame_start']}-{e['frame_end']}" for e in only_b], "matched_pairs": rows,
       "diff_events": {str(t): {"missing_in_10fps": len(v["missing"]), "added_in_10fps": len(v["added"]), "changed": len(v["changed"])} for t, v in d.items()},
       "plate_detector_calls": {"30fps": json.loads((A / "run_info.json").read_text())["plate_detector_calls"], "10fps": json.loads((B / "run_info.json").read_text())["plate_detector_calls"]}}
(Path(__file__).parent / "compare.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "matched_pairs"}, indent=1))
for r in rows:
    print(r)
