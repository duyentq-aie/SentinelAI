"""Run the full pipeline once and record per-frame perception (tracks + light state) and the resulting events.

  python tools/record_tracks.py --device cpu \
      --tracks tests/fixtures/tracks_test_video.jsonl --events experiments/EXP-001-baseline/events_refactor_check.json

The tracks file is the input of the offline rule tests (sentinel/offline.py): rule tests replay it in seconds without any model.
Metadata records sha256 of the video, the three weights and the tracker yamls, so a stale fixture can be detected.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "sentinel", "configs"], cwd=REPO, text=True).strip())
        return rev + ("+uncommitted" if dirty else "")
    except Exception:
        return None


def fixture_meta(cfg, video, device):
    import torch
    import ultralytics
    return {
        "git": git_state(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video": {"file": Path(video).name, "sha256": sha256(video)},
        "models": {n: {"file": getattr(cfg.models, n).weights, "sha256": sha256(cfg.path(getattr(cfg.models, n).weights))}
                   for n in ("coco", "helmet", "plate")},
        "trackers": {n: {"file": getattr(cfg.models, n).tracker, "sha256": sha256(cfg.path(getattr(cfg.models, n).tracker))}
                     for n in ("coco", "helmet")},
        "conf": {n: getattr(cfg.models, n).conf for n in ("coco", "helmet", "plate")},
        "frame_size": cfg.frame_size, "device": device or "default",
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "note": "models see the clean resized frame (no overlays) since P2 3.0b; track ids are device-specific, this file was recorded on CPU",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--video", default=str(REPO / "test_video.mp4"))
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--events")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    from sentinel.characterize import read_frames
    from sentinel.config import load_camera_config, load_pipeline_config
    from sentinel.offline import write_tracks
    from sentinel.pipeline import Detectors, Pipeline, SharedState

    cfg = load_pipeline_config(a.config, root=REPO, device=a.device)
    work = Path(tempfile.mkdtemp(prefix="sentinel_rec_"))
    (work / "ev").mkdir()
    cfg.evidence.dir = str(work / "ev")
    camera = load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
    pipe = Pipeline(Detectors(cfg), SharedState(), cfg, camera)

    perceptions, events, first = [], [], {}
    for i, frame in enumerate(read_frames(a.video)):
        r = pipe.process(frame)
        perceptions.append(r.perception)
        events += [{"frame": i, "kind": "plate_capture", "track": v} for v in r.captures]
        for lab in r.labels:
            first.setdefault(lab, i)
    events += [{"frame": fi, "kind": "label_first_seen", "label": lab} for lab, fi in sorted(first.items(), key=lambda x: x[1])]
    shutil.rmtree(work, ignore_errors=True)

    Path(a.tracks).parent.mkdir(parents=True, exist_ok=True)
    write_tracks(a.tracks, fixture_meta(cfg, a.video, a.device), perceptions)
    print(f"{len(perceptions)} frames -> {a.tracks} ({Path(a.tracks).stat().st_size / 1024:.0f} KiB)")
    if a.events:
        Path(a.events).write_text(json.dumps({"impl": "refactored-3.0", "device": a.device, "video": Path(a.video).name,
                                              "events": events}, indent=2), encoding="utf-8")
        print(f"{len(events)} events -> {a.events}")


if __name__ == "__main__":
    main()
