"""Evaluate predicted violation events against labelled ground truth (matching + P/R/F1 per type + FP/hour + latency + bootstrap CI by video).

  python tools/eval_events.py --labels eval_sets/events/*.json --pred static/violations/events.jsonl [--split test] [--json report.json]

Prediction frames must be frame indices of the SAME video (run the pipeline on each labelled video from its first frame, every frame).
Evaluate the `test` split only once, with the configuration frozen on `dev` (see docs/LABELING_GUIDE.md); the tool warns when splits are mixed.
Definitions: sentinel/evaluation.py (docstring) and PLAN section 3.3-3.4.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from sentinel import evaluation  # noqa: E402


def expand(patterns):
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p))
        out += hits if hits else ([p] if Path(p).exists() else [])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", required=True, help="label files (glob allowed)")
    ap.add_argument("--pred", nargs="+", required=True, help="events.jsonl file(s) written by the pipeline")
    ap.add_argument("--run", type=int, default=0, help="which pipeline run of events.jsonl to evaluate (0 = first pass; -1 = all runs)")
    ap.add_argument("--split", nargs="*", choices=["dev", "test"], help="only these label splits")
    ap.add_argument("--tau", type=float, default=1.0, help="time tolerance in seconds around the labelled interval")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--relaxed-iou", type=float, default=0.1, help="IoU used when the predicted track has no box near the labelled key frame (reported as 'relaxed_matches')")
    ap.add_argument("--exclude-hard", action="store_true", help="do not score events labelled difficulty=hard")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write the full report here")
    a = ap.parse_args()

    label_files, pred_files = expand(a.labels), expand(a.pred)
    if not label_files or not pred_files:
        sys.exit("no label or prediction files found")
    labels = evaluation.load_labels(label_files)
    preds = evaluation.load_predictions(pred_files, None if a.run < 0 else a.run)
    rep = evaluation.evaluate(labels, preds, a.tau, a.iou, a.relaxed_iou, not a.exclude_hard, a.bootstrap, a.seed, a.split)
    print(evaluation.format_report(rep))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\nfull report -> {a.json}")


if __name__ == "__main__":
    main()
