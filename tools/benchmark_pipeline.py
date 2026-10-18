"""Benchmark the sentinel Pipeline against the ORIGINAL prototype (git tag v0-baseline) in the SAME session.

  python tools/benchmark_pipeline.py --repeats 3 --configs configs/pipeline/exp/p4b_roi_jpeg80.yaml configs/pipeline/exp/p4c_gmc_none.yaml --out experiments/EXP-0xx
  python tools/benchmark_pipeline.py --configs configs/pipeline/onnx_gpu_fp16.yaml@venv-edge/Scripts/python.exe ...     # a config run by another interpreter

Every session also runs `v0` (the original app.py, unmodified, extracted from the tag) and reports each config as a RATIO to v0 measured in the
same round, because absolute FPS on this laptop changes a lot between sessions (7 vs 12 FPS for identical code). `--no-v0` disables that (do not).

Protocol: round 1 runs every config once (fresh process each), round 2 the next, ... (interleaved, so drift of the machine hits all of them equally).
First --warmup frames dropped; one pass over test_video.mp4. All configs use this same harness; numbers from other harnesses are not comparable.
  headless FPS = frames / (read + process incl. drawing, no JPEG)     stream FPS = frames / (read + process + JPEG encode, no HTTP)
  v0: process+drawing+encode happen inside its generator; its encode time is measured by wrapping cv2.imencode, so headless = total - encode.
Telemetry (tools/sysprobe.py) is sampled during every run: GPU clocks / power / temperature / throttle reasons, CPU performance %, load of OTHER processes.
"""
import argparse
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import sysprobe  # noqa: E402
from benchmark import env_info, summarize  # noqa: E402

WORK = REPO / "outputs" / "bench_work"          # scratch space on the repo volume (git-ignored; nothing is deleted by this tool)


def _finish(res, sampler):
    res["system"] = sysprobe.summarize(sampler.stop())
    g = res["system"]
    res["gpu_mean_util_pct"] = (g.get("gpu_util_pct") or {}).get("median")
    res["gpu_peak_temp_c"] = (g.get("gpu_temp_c") or {}).get("max")
    return res


# --------------------------------------------------------------------------- sentinel pipeline child
def run_once(config, warmup, out_json, limit=0):
    import cv2
    import statistics
    import torch

    from sentinel.config import load_camera_config, load_pipeline_config
    from sentinel.pipeline import Detectors, Pipeline, SharedState

    cfg = load_pipeline_config(config, root=REPO)
    work = WORK / f"ev_{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    cfg.evidence.dir = str(work)
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

    sampler = sysprobe.Sampler(2.0)
    sampler.start()

    rows, plate_calls, events_opened, plates_found = [], 0, 0, 0
    cap = cv2.VideoCapture(str(REPO / "test_video.mp4"))
    i = -1
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        read = (time.perf_counter() - t0) * 1000
        if not ok or (limit and i + 1 >= limit):
            break
        i += 1
        acc.clear()
        t0 = time.perf_counter()
        r = pipe.process(frame)
        sync()
        proc = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        cv2.imencode(".jpg", r.frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.web.jpeg_quality])
        enc = (time.perf_counter() - t1) * 1000
        if i < warmup:
            continue
        acc.update(process=proc, encode=enc, read=read)
        acc["evidence_io_and_bookkeeping"] = max(0.0, acc.get("evidence_incl_plate", 0.0) - acc.get("plate_detector", 0.0))
        rows.append(dict(acc))
        plate_calls += r.plate_calls; events_opened += len(r.new_events); plates_found += len(r.captures)
    cap.release()
    pipe.flush()
    series = {k: [row.get(k, 0.0) for row in rows] for k in {k for row in rows for k in row}}

    n = len(series["process"])
    headless_s = (sum(series["process"]) + sum(series["read"])) / 1000
    stream_s = headless_s + sum(series["encode"]) / 1000
    ev_file = Path(cfg.evidence.dir) / "events.jsonl"
    last = {}
    for line in (ev_file.read_text(encoding="utf-8").splitlines() if ev_file.exists() else []):
        rec = json.loads(line)
        last[rec["id"]] = rec
    evs = [e for e in last.values() if e.get("status") != "evicted"]
    plates = [e["plate"] for e in evs if e.get("plate", {}).get("found")]
    res = {
        "kind": "pipeline", "config": config, "n_frames": n, "fps_headless": round(n / headless_s, 2), "fps_stream": round(n / stream_s, 2),
        "stages": {k: summarize(v) for k, v in series.items() if k != "evidence_incl_plate"},
        "plate_calls": plate_calls, "plate_calls_per_frame": round(plate_calls / n, 3),
        "evidence": {
            "events": len(evs), "events_by_type": {t: sum(1 for e in evs if e["type"] == t) for t in sorted({e["type"] for e in evs})},
            "tracks_with_event": len({e["track_id"] for e in evs}),
            "tracks_with_plate": len({e["track_id"] for e in evs if e.get("plate", {}).get("found")}),
            "plates_found_first_time": plates_found, "events_opened_counter": events_opened,
            "mean_vehicle_crop_wh": [round(statistics.mean(e["vehicle"]["box_size"][k] for e in evs), 1) for k in (0, 1)] if evs else None,
            "mean_plate_crop_wh": [round(statistics.mean(p["crop_size"][k] for p in plates), 1) for k in (0, 1)] if plates else None,
            "mean_plate_box_wh": [round(statistics.mean(p["box"][k + 2] - p["box"][k] for p in plates), 1) for k in (0, 1)] if plates else None,
            "evidence_dir_mb": round(sum(f.stat().st_size for f in Path(cfg.evidence.dir).rglob("*") if f.is_file()) / 2**20, 2),
        },
    }
    Path(out_json).write_text(json.dumps(_finish(res, sampler), indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- v0 child (original app.py from the tag)
def run_once_v0(warmup, out_json, limit=0):
    import cv2
    import torch

    work = WORK / "v0"
    (work / "templates").mkdir(parents=True, exist_ok=True)
    (work / "static" / "violations").mkdir(parents=True, exist_ok=True)
    (work / "static" / "uploads").mkdir(parents=True, exist_ok=True)
    for f in ("helmet_best.pt", "plate_best.pt", "yolo11s.pt", "test_video.mp4"):
        dst = work / f
        if not dst.exists():
            try:
                os.link(REPO / f, dst)                  # hard link: no extra disk space, nothing to delete
            except OSError:
                shutil.copy2(REPO / f, dst)
    shutil.copy2(REPO / "templates" / "index.html", work / "templates" / "index.html")
    (work / "app.py").write_text(subprocess.check_output(["git", "show", "v0-baseline:app.py"], cwd=REPO, text=True, encoding="utf-8"), encoding="utf-8")
    os.chdir(work)
    sys.path.insert(0, str(work))
    import app                                           # the original, unmodified

    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    enc_ms, read_ms, cur = [], [], {"enc": 0.0, "plate": 0}
    orig_imencode = cv2.imencode

    def imencode(*a, **k):
        t0 = time.perf_counter()
        r = orig_imencode(*a, **k)
        cur["enc"] += (time.perf_counter() - t0) * 1000
        return r
    cv2.imencode = imencode

    class CapProxy:
        def __init__(self, c): self._c = c
        def isOpened(self): return self._c.isOpened()
        def set(self, *a): return self._c.set(*a)
        def release(self): return self._c.release()
        def read(self):
            t0 = time.perf_counter(); r = self._c.read(); read_ms.append((time.perf_counter() - t0) * 1000); return r
    app.cap = CapProxy(app.cap)
    orig_predict = app.plate_model.predict

    def predict(*a, **k):
        cur["plate"] += 1
        return orig_predict(*a, **k)
    app.plate_model.predict = predict

    total_frames = int(app.cap._c.get(cv2.CAP_PROP_FRAME_COUNT))
    if limit:
        total_frames = min(total_frames, limit)
    gen = app.generate_frames()
    sampler = sysprobe.Sampler(2.0)
    sampler.start()
    tot, enc, plate_calls = [], [], 0
    for i in range(total_frames):
        cur["enc"], cur["plate"] = 0.0, 0
        sync(); t0 = time.perf_counter()
        next(gen)
        sync(); dt = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            tot.append(dt); enc.append(cur["enc"]); plate_calls += cur["plate"]
    n = len(tot)
    proc = [t - e for t, e in zip(tot, enc)]
    captured = len(app.captured_violator_ids)
    res = {"kind": "v0", "config": "v0-baseline (original app.py)", "n_frames": n,
           "fps_headless": round(n / (sum(proc) / 1000), 2), "fps_stream": round(n / (sum(tot) / 1000), 2),
           "stages": {"process": summarize(proc), "encode": summarize(enc), "read": summarize(read_ms[warmup:])},
           "plate_calls": plate_calls, "plate_calls_per_frame": round(plate_calls / n, 3),
           "evidence": {"events": captured, "events_by_type": {}, "tracks_with_event": captured, "tracks_with_plate": captured,
                        "note": "v0 only records plate captures (no event notion): counts are captures"}}
    Path(out_json).write_text(json.dumps(_finish(res, sampler), indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- parent
def _med_range(xs):
    return {"median": round(st.median(xs), 3), "min": round(min(xs), 3), "max": round(max(xs), 3)}


def summarize_runs(label, runs):
    sysm = [r.get("system", {}) for r in runs]
    pick = lambda key, sub="median": [s[key][sub] for s in sysm if s.get(key)]
    return {
        "config": label, "n_runs": len(runs),
        "fps_headless": {**_med_range([r["fps_headless"] for r in runs]), "runs": [r["fps_headless"] for r in runs]},
        "fps_stream": {**_med_range([r["fps_stream"] for r in runs]), "runs": [r["fps_stream"] for r in runs]},
        "plate_calls_per_frame_median": st.median(r["plate_calls_per_frame"] for r in runs),
        "stages_median_of_runs": {k: {m: round(st.median(r["stages"][k][m] for r in runs), 2) for m in ("mean_ms", "p50_ms", "p95_ms")} for k in runs[0]["stages"]},
        "evidence_first_run": runs[0]["evidence"],
        "telemetry_median_of_runs": {
            "gpu_sm_clock_mhz": round(st.median(pick("gpu_sm_clock_mhz")), 0) if pick("gpu_sm_clock_mhz") else None,
            "gpu_power_w": round(st.median(pick("gpu_power_w")), 1) if pick("gpu_power_w") else None,
            "gpu_temp_c_max": max(pick("gpu_temp_c", "max")) if pick("gpu_temp_c", "max") else None,
            "gpu_util_pct": round(st.median(pick("gpu_util_pct")), 0) if pick("gpu_util_pct") else None,
            "cpu_perf_pct": round(st.median(pick("cpu_perf_pct")), 0) if pick("cpu_perf_pct") else None,
            "cpu_other_pct": round(st.median(pick("cpu_other_pct")), 1) if pick("cpu_other_pct") else None,
            "throttle_fraction_max": {k: max(s.get("gpu_throttle_reason_fraction", {}).get(k, 0) for s in sysm)
                                      for k in (sysm[0].get("gpu_throttle_reason_fraction") or {}) if k != "gpu_idle"},
            "heaviest_other_processes": {k: v for s in sysm for k, v in (s.get("heaviest_other_processes_peak_pct") or {}).items()},
        },
    }


def ratios_vs_v0(runs, stems):
    out = {}
    for stem in stems:
        if stem == "v0":
            continue
        rows = {}
        for key in ("fps_headless", "fps_stream"):
            per_round = [b[key] / a[key] for a, b in zip(runs["v0"], runs[stem]) if a[key] > 0]
            rows[key] = {**_med_range(per_round), "per_round": [round(x, 3) for x in per_round]}
        out[stem] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--configs", "--config", nargs="*", default=None, help="config.yaml or config.yaml@python_exe; run INTERLEAVED with v0")
    ap.add_argument("--no-v0", action="store_true", help="do not run the v0 reference (not recommended: ratios need it)")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames (smoke tests only)")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--v0", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        if a.v0:
            run_once_v0(a.warmup, a.child, a.limit)
        else:
            run_once(a.configs[0] if a.configs else None, a.warmup, a.child, a.limit)
        return
    if not a.out:
        ap.error("--out is required")

    entries = []                                           # (stem, config path or None, python exe)
    if not a.no_v0:
        entries.append(("v0", None, sys.executable))
    for c in (a.configs or [None]):
        path, _, py = (c or "").partition("@")
        entries.append((Path(path).stem if path else "default", path or None, str(REPO / py) if py else sys.executable))
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    static = sysprobe.static_info()
    runs = {e[0]: [] for e in entries}
    for r in range(a.repeats):
        for stem, cfg, py in entries:
            (out / "runs").mkdir(exist_ok=True)
            tmp = out / "runs" / f"{stem}_{r + 1}.json"
            print(f"[bench] round {r + 1}/{a.repeats}: {stem}", flush=True)
            cmd = [py, __file__, "--child", str(tmp), "--warmup", str(a.warmup)] + (["--limit", str(a.limit)] if a.limit else []) + (["--v0"] if stem == "v0" else (["--configs", cfg] if cfg else []))
            subprocess.check_call(cmd)
            runs[stem].append(json.loads(tmp.read_text(encoding="utf-8")))
            time.sleep(5)
    labels = {e[0]: (e[1] or e[0]) for e in entries}
    summaries = {s: summarize_runs(labels[s], runs[s]) for s in runs}
    for s in runs:
        d = out / s
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text(json.dumps({"summary": summaries[s], "runs": runs[s]}, indent=2), encoding="utf-8")
    ratios = ratios_vs_v0(runs, list(runs)) if "v0" in runs else {}
    env_text = json.dumps({**env_info(), "system_static": static}, indent=2)
    for home in {str(Path.home()), str(Path.home()).replace("\\", "\\\\")}:          # published files must not carry the user's home path
        env_text = env_text.replace(home, "<HOME>")
    (out / "env.json").write_text(env_text, encoding="utf-8")
    table = {s: {"headless_fps": summaries[s]["fps_headless"]["median"], "headless_range": [summaries[s]["fps_headless"]["min"], summaries[s]["fps_headless"]["max"]],
                 "stream_fps": summaries[s]["fps_stream"]["median"], "stream_range": [summaries[s]["fps_stream"]["min"], summaries[s]["fps_stream"]["max"]],
                 "ratio_vs_v0": ratios.get(s), "plate_calls_per_frame": summaries[s]["plate_calls_per_frame_median"],
                 "tracks_with_event/with_plate": [summaries[s]["evidence_first_run"]["tracks_with_event"], summaries[s]["evidence_first_run"]["tracks_with_plate"]],
                 "telemetry": summaries[s]["telemetry_median_of_runs"]} for s in summaries}
    (out / "summary.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    print("\n%-26s %-24s %-24s %s" % ("config", "headless FPS (range)", "stream FPS (range)", "stream ratio vs v0 (per-round median, range)"))
    for s, t in table.items():
        rv = (t["ratio_vs_v0"] or {}).get("fps_stream")
        print("%-26s %6.2f (%5.2f-%5.2f)      %6.2f (%5.2f-%5.2f)      %s" % (
            s, t["headless_fps"], *t["headless_range"], t["stream_fps"], *t["stream_range"],
            "%.2fx (%.2f-%.2f)" % (rv["median"], rv["min"], rv["max"]) if rv else "(reference)"))


if __name__ == "__main__":
    main()
