"""Run the pipeline on a video and write everything a human needs to review by eye (outputs/ is gitignored: people and plates are visible).

  python tools/review_run.py --video test_video.mp4 --label A --helmet-weights helmet_best.pt --out outputs/review_A
  python tools/review_run.py --video test_video.mp4 --label B --helmet-weights experiments/EXP-015-helmet-grouped-3cls/weights/best.pt --out outputs/review_B

Uses the CURRENT DEFAULT pipeline config (configs/pipeline/default.yaml, nothing is written to it); only the helmet weights are replaced in memory.
Output folder:
  review_<label>.mp4          annotated video: boxes, track ids (ID:M_77 ...), HELMET / NO-HELMET head labels, red banner while a violation event is open
  evidence/                   events.jsonl + one folder per event (frame.jpg, vehicle.jpg, plate.jpg)
  events_summary.md           table: time (s), track id, type, evidence path
  characterization_events.json  label_first_seen / plate_capture events (the format of tools/diff_events.py and the v0..v5 baselines)
  run_info.json               weights, sha256, config hash, frame count (no timing: speed is measured by tools/benchmark_pipeline.py as a ratio to v0)
The pipeline's own 'FPS:' overlay (an absolute number) is painted over.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def last_records(events_jsonl):
    """events.jsonl -> {id: last record}, evicted dropped (take the last record per id, docs/EVIDENCE.md)."""
    last = {}
    for line in Path(events_jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            last[r["id"]] = r
    return {k: v for k, v in last.items() if v.get("status") != "evicted"}


def summary_md(label, weights, records, fps, evidence_dir):
    L = [f"# Events – run {label} (helmet weights: `{weights}`)", "",
         f"{len(records)} events. Time = trigger frame / {fps:g} fps. Evidence paths are relative to `{evidence_dir.parent.name}/`.", "",
         "| time (s) | frames | track | type | evidence |", "|---|---|---|---|---|"]
    for r in sorted(records.values(), key=lambda r: (r["trigger_frame"], r["id"])):
        ev = (r.get("evidence") or {}).get("frame")
        L.append(f"| {r['trigger_frame'] / fps:.2f} | {r['frame_start']}–{r['frame_end']} | {r['track_id']} | {r['type']} | "
                 f"{('evidence/' + ev) if ev else '-'} |")
    return "\n".join(L) + "\n"


def banner(img, open_events, label):
    h, w = img.shape[:2]
    if open_events:
        cv2.rectangle(img, (0, h - 46), (w, h), (0, 0, 200), -1)
        text = "VIOLATION: " + "  |  ".join(f"{t.upper()} {tid}" for tid, t in open_events)
        cv2.putText(img, text[:110], (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(img, (0, 0), (400, 34), (0, 0, 0), -1)
    cv2.putText(img, f"run {label}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--helmet-weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera-config", default=str(REPO / "configs" / "cameras" / "test_video.yaml"))
    a = ap.parse_args()

    from sentinel.characterize import read_frames
    from sentinel.config import load_camera_config, load_pipeline_config
    from sentinel.pipeline import Detectors, Pipeline, SharedState

    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty: nothing is overwritten or deleted (choose another --out)")
    ev_dir = out / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_pipeline_config(root=REPO)                                    # the default config, as shipped
    cfg_text = (REPO / "configs" / "pipeline" / "default.yaml").read_bytes()
    cfg.models.helmet.weights = a.helmet_weights                              # in memory only
    cfg.evidence.dir = str(ev_dir)
    camera = load_camera_config(a.camera_config)
    pipe = Pipeline(Detectors(cfg), SharedState(), cfg, camera)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    w, h = cfg.frame_size
    vw = cv2.VideoWriter(str(out / f"review_{a.label}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    chars, first, n = [], {}, 0
    for i, frame in enumerate(read_frames(a.video)):
        r = pipe.process(frame)
        for v in r.captures:
            chars.append({"frame": i, "kind": "plate_capture", "track": v})
        for lab in r.labels:
            first.setdefault(lab, i)
        disp = r.frame
        disp[60:105, 40:230] = r.raw[60:105, 40:230]                           # remove the pipeline's absolute 'FPS: n' text
        banner(disp, [(e.track_id, e.type) for e in pipe.recorder._open.values()], a.label)
        cv2.putText(disp, f"t={i / fps:5.2f}s  frame {i}", (w - 330, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        vw.write(disp)
        n += 1
    pipe.flush()
    vw.release()
    for lab, fi in sorted(first.items(), key=lambda x: x[1]):
        chars.append({"frame": fi, "kind": "label_first_seen", "label": lab})
    (out / "characterization_events.json").write_text(json.dumps({"impl": f"review run {a.label}", "helmet_weights": a.helmet_weights, "events": chars}, indent=1), encoding="utf-8")
    records = last_records(ev_dir / "events.jsonl") if (ev_dir / "events.jsonl").exists() else {}
    (out / "events_summary.md").write_text(summary_md(a.label, a.helmet_weights, records, fps, ev_dir), encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps({"label": a.label, "helmet_weights": a.helmet_weights, "helmet_weights_sha256": sha256(REPO / a.helmet_weights), "video": Path(a.video).name,
                                                   "video_sha256": sha256(a.video), "frames": n, "default_config_sha256": hashlib.sha256(cfg_text).hexdigest(),
                                                   "rules": {"engine": cfg.rules.engine, "helmet_exit_ratio": cfg.rules.helmet.exit_ratio}, "events": len(records)}, indent=2), encoding="utf-8")
    print(f"run {a.label}: {n} frames, {len(records)} events -> {out}")


if __name__ == "__main__":
    main()
