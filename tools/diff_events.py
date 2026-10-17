"""List the violation events lost / added / changed between two runs.

  python tools/diff_events.py A.json B.json [--tol 3] [--json]

A and B are event files: either {"events": [...]} (tools/record_events.py, EXP-xxx/events*.json) or a bare list.
Exit code 0 when the runs are identical within --tol, 1 otherwise (so it can gate scripts; *not* a pass/fail for rule changes,
which are expected to differ from the v0 baseline and are explained in the experiment notes).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentinel.characterize import diff_events  # noqa: E402


def load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["events"] if isinstance(d, dict) else d


def fmt(e):
    what = e.get("track") or e.get("label")
    return f"frame {e['frame']:>4}  {e['kind']:<16} {what}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=int, default=3, help="frame tolerance for 'moved' (default 3)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    d = diff_events(a, b, args.tol)
    if args.json:
        print(json.dumps(d, indent=2))
    else:
        print(f"A: {args.a} ({len(a)} events)\nB: {args.b} ({len(b)} events)\n")
        print(f"MISSING in B ({len(d['missing'])}):")
        for e in d["missing"]: print("  -", fmt(e))
        print(f"ADDED in B ({len(d['added'])}):")
        for e in d["added"]: print("  +", fmt(e))
        print(f"CHANGED ({len(d['changed'])}):")
        for c in d["changed"]:
            print(f"  ~ {c['kind']} {c['entity']}: " + "; ".join(c["why"]))
        if not any(d.values()):
            print("\nidentical (within tolerance)")
    sys.exit(0 if not any(d.values()) else 1)


if __name__ == "__main__":
    main()
