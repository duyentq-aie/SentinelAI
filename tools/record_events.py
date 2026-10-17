"""Record the violation-event snapshot of a pipeline implementation on a video.

  python tools/record_events.py --impl original   --device cpu --out experiments/EXP-001-baseline/events_cpu_original.json
  python tools/record_events.py --impl refactored --device cpu --out experiments/EXP-001-baseline/events_cpu_refactored.json

'original' runs app.py as committed at git tag v0-baseline (extracted to a temp dir, not modified).
--device cpu works for both by hiding CUDA (CUDA_VISIBLE_DEVICES=-1), so no code differs between the two runs.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def run_original(video, tag="v0-baseline"):
    import cv2
    work = Path(tempfile.mkdtemp(prefix="sentinel_orig_"))
    (work / "app.py").write_text(
        subprocess.check_output(["git", "show", f"{tag}:app.py"], cwd=REPO, text=True, encoding="utf-8"), encoding="utf-8")
    for f in ["helmet_best.pt", "plate_best.pt", "yolo11s.pt"]:
        shutil.copy2(REPO / f, work / f)
    shutil.copy2(video, work / "test_video.mp4")
    shutil.copytree(REPO / "templates", work / "templates")
    os.chdir(work)
    sys.path.insert(0, str(work))
    import app
    labels, cur = [], {"i": 0}
    orig_box = app.draw_sleek_box

    def box(img, x1, y1, x2, y2, label, color, thickness=1):
        if any(r in label for r in ("NO-HELMET", "OVERLOAD", "RED-LIGHT", "WRONG-WAY", "WRONG LANE")):
            labels.append((cur["i"], label))
        return orig_box(img, x1, y1, x2, y2, label, color, thickness)
    app.draw_sleek_box = box
    n = int(app.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gen, events, seen = app.generate_frames(), [], set()
    for i in range(n):
        cur["i"] = i
        next(gen)
        for v in app.captured_violator_ids - seen:
            events.append({"frame": i, "kind": "plate_capture", "track": v})
        seen = set(app.captured_violator_ids)
    first = {}
    for fi, lab in labels:
        first.setdefault(lab, fi)
    for lab, fi in sorted(first.items(), key=lambda x: x[1]):
        events.append({"frame": fi, "kind": "label_first_seen", "label": lab})
    shutil.rmtree(work, ignore_errors=True)
    return events


def run_refactored(video, device):
    from sentinel.characterize import collect_events, read_frames
    from sentinel.config import load_camera_config, load_pipeline_config
    from sentinel.pipeline import Detectors, Pipeline, SharedState
    work = Path(tempfile.mkdtemp(prefix="sentinel_ref_"))
    (work / "ev").mkdir()
    cfg = load_pipeline_config(root=REPO, device=device)
    cfg.evidence.dir = str(work / "ev")
    camera = load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
    events = collect_events(Pipeline(Detectors(cfg), SharedState(), cfg, camera), read_frames(video))
    shutil.rmtree(work, ignore_errors=True)
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["original", "refactored"], required=True)
    ap.add_argument("--device", default=None, help="cpu to hide CUDA; omit for default")
    ap.add_argument("--video", default=str(REPO / "test_video.mp4"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    out = Path(a.out).resolve()
    video = Path(a.video).resolve()
    events = run_original(video) if a.impl == "original" else run_refactored(video, a.device)
    out.write_text(json.dumps({"impl": a.impl, "device": a.device or "default", "video": video.name,
                               "events": events}, indent=2), encoding="utf-8")
    print(f"{a.impl}: {len(events)} events -> {out}")


if __name__ == "__main__":
    main()
