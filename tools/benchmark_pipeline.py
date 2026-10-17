"""Benchmark the current sentinel Pipeline (not the v0 app.py; for that see tools/benchmark.py).

  python tools/benchmark_pipeline.py --repeats 3 --out experiments/EXP-003-evidence/bench_event_policy [--config configs/pipeline/ab_plate_v0.yaml]

Protocol: one pass over test_video.mp4 per repeat (fresh process), first --warmup frames dropped, device = config default (GPU).
  headless FPS = frames / (read + Pipeline.process)               <- no JPEG encoding, nobody watching
  stream   FPS = frames / (read + Pipeline.process + JPEG encode) <- what the MJPEG endpoint does per frame (HTTP transfer not included)
Also reported: p50/p95 per stage, plate-detector calls per frame, evidence statistics (events, plates found, mean crop sizes).
"""
import argparse
import json
import statistics as st
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from benchmark import env_info, gpu_query, summarize  # noqa: E402


def run_once(config, warmup, out_json):
    import threading

    import cv2
    import torch

    from sentinel.config import load_camera_config, load_pipeline_config
    from sentinel.pipeline import Detectors, Pipeline, SharedState

    cfg = load_pipeline_config(config, root=REPO)
    work = Path(tempfile.mkdtemp(prefix="sentinel_bp_"))
    cfg.evidence.dir = str(work / "ev")
    camera = load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
    det = Detectors(cfg)
    pipe = Pipeline(det, SharedState(), cfg, camera)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    acc = {}

    def add(k, ms):
        acc[k] = acc.get(k, 0.0) + ms

    def timed(obj, name, key, gpu=False):
        orig = getattr(obj, name)

        def w(*a, **kw):
            if gpu: sync()
            t0 = time.perf_counter()
            r = orig(*a, **kw)
            if gpu: sync()
            add(key, (time.perf_counter() - t0) * 1000)
            return r
        setattr(obj, name, w)

    timed(det, "track_coco", "coco", True)
    timed(det, "track_helmet", "helmet", True)
    timed(det, "plate_boxes", "plate_detector", True)
    timed(pipe.rules, "step", "rules")
    timed(pipe, "_draw", "draw")
    timed(pipe.recorder, "update", "evidence_incl_plate")
    pipe.recorder.plate_boxes = det.plate_boxes            # the recorder captured the unwrapped bound method at construction

    samples, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            g = gpu_query()
            if g: samples.append(g)
            stop.wait(2)
    th = threading.Thread(target=poll, daemon=True); th.start()

    rows, plate_calls, events_opened, plates_found = [], 0, 0, 0
    cap = cv2.VideoCapture(str(REPO / "test_video.mp4"))
    i = -1
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        read = (time.perf_counter() - t0) * 1000
        if not ok:
            break
        i += 1
        acc.clear()
        t0 = time.perf_counter()
        r = pipe.process(frame)
        sync()
        proc = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        cv2.imencode(".jpg", r.frame)
        enc = (time.perf_counter() - t1) * 1000
        if i < warmup:
            continue
        acc.update(process=proc, encode=enc, read=read)
        acc["evidence_io_and_bookkeeping"] = max(0.0, acc.get("evidence_incl_plate", 0.0) - acc.get("plate_detector", 0.0))
        rows.append(dict(acc))
        plate_calls += r.plate_calls; events_opened += len(r.new_events); plates_found += len(r.captures)
    cap.release()
    stop.set(); th.join(timeout=3)
    pipe.flush()
    series = {k: [row.get(k, 0.0) for row in rows] for k in {k for row in rows for k in row}}

    n = len(series["process"])
    headless_s = (sum(series["process"]) + sum(series["read"])) / 1000
    stream_s = headless_s + sum(series["encode"]) / 1000
    ev_file = Path(cfg.evidence.dir) / "events.jsonl"
    last = {}
    for line in (ev_file.read_text(encoding="utf-8").splitlines() if ev_file.exists() else []):
        r = json.loads(line); last[r["id"]] = r
    evs = [e for e in last.values() if e.get("status") != "evicted"]
    plates = [e["plate"] for e in evs if e.get("plate", {}).get("found")]
    res = {
        "n_frames": n, "fps_headless": round(n / headless_s, 2), "fps_stream": round(n / stream_s, 2),
        "stages": {k: summarize(v) for k, v in series.items() if k != "evidence_incl_plate"},
        "plate_calls": plate_calls, "plate_calls_per_frame": round(plate_calls / n, 3),
        "evidence": {
            "events": len(evs), "events_by_type": {t: sum(1 for e in evs if e["type"] == t) for t in sorted({e["type"] for e in evs})},
            "tracks_with_event": len({e["track_id"] for e in evs}),
            "tracks_with_plate": len({e["track_id"] for e in evs if e.get("plate", {}).get("found")}),
            "plates_found_first_time": plates_found, "events_opened_counter": events_opened,
            "mean_vehicle_crop_wh": [round(st.mean(e["vehicle"]["box_size"][i] for e in evs), 1) for i in (0, 1)] if evs else None,
            "mean_plate_crop_wh": [round(st.mean(p["crop_size"][i] for p in plates), 1) for i in (0, 1)] if plates else None,
            "mean_plate_attempts_until_found": round(st.mean(p["attempts"] for p in plates), 2) if plates else None,
            "evidence_dir_mb": round(sum(f.stat().st_size for f in Path(cfg.evidence.dir).rglob("*") if f.is_file()) / 2**20, 2),
        },
        "gpu_mean_util_pct": round(st.mean(x["util_pct"] for x in samples), 1) if samples else None,
        "gpu_peak_temp_c": max((x["temp_c"] for x in samples), default=None),
    }
    Path(out_json).write_text(json.dumps(res, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        run_once(a.config, a.warmup, a.child); return

    if not a.out:
        ap.error("--out is required")
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for r in range(a.repeats):
        tmp = out / f"_run{r + 1}.json"
        print(f"[bench] run {r + 1}/{a.repeats}", flush=True)
        subprocess.check_call([sys.executable, __file__, "--child", str(tmp), "--warmup", str(a.warmup)] + (["--config", a.config] if a.config else []))
        runs.append(json.loads(tmp.read_text(encoding="utf-8")))
        time.sleep(5)
    med = lambda f: st.median(f(r) for r in runs)
    spread = lambda key: round((max(r[key] for r in runs) - min(r[key] for r in runs)) / st.median(r[key] for r in runs) * 100, 2)
    summary = {
        "config": a.config or "configs/pipeline/default.yaml",
        "fps_headless_runs": [r["fps_headless"] for r in runs], "fps_headless_median": med(lambda r: r["fps_headless"]), "fps_headless_spread_pct": spread("fps_headless"),
        "fps_stream_runs": [r["fps_stream"] for r in runs], "fps_stream_median": med(lambda r: r["fps_stream"]), "fps_stream_spread_pct": spread("fps_stream"),
        "plate_calls_per_frame_median": med(lambda r: r["plate_calls_per_frame"]),
        "stages_median_of_runs": {k: {m: round(st.median(r["stages"][k][m] for r in runs), 2) for m in ("mean_ms", "p50_ms", "p95_ms")} for k in runs[0]["stages"]},
        "evidence_first_run": runs[0]["evidence"], "gpu_mean_util_pct_per_run": [r["gpu_mean_util_pct"] for r in runs],
        "gpu_peak_temp_c_per_run": [r["gpu_peak_temp_c"] for r in runs],
    }
    (out / "metrics.json").write_text(json.dumps({"summary": summary, "runs": runs}, indent=2), encoding="utf-8")
    (out / "env.json").write_text(json.dumps(env_info(), indent=2), encoding="utf-8")
    for r in range(a.repeats):
        (out / f"_run{r + 1}.json").unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
