"""Report how perception changed between two recorded tracks files.

  python tools/diff_tracks.py tests/fixtures/tracks_test_video_v0_overlay.jsonl tests/fixtures/tracks_test_video.jsonl [--json OUT]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentinel.offline import load_tracks  # noqa: E402
from sentinel.trackdiff import diff_tracks, format_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--min-iou", type=float, default=0.5)
    ap.add_argument("--json", help="also write the full result here")
    args = ap.parse_args()
    _, a = load_tracks(args.a)
    _, b = load_tracks(args.b)
    d = diff_tracks(a, b, args.min_iou)
    print(format_report(d, Path(args.a).name, Path(args.b).name))
    if args.json:
        Path(args.json).write_text(json.dumps(d, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
