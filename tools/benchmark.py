"""P0 baseline benchmark for the ORIGINAL app.py pipeline (no changes to app.py).

Imports app.generate_frames, patches model/imencode/read calls with timers, and
runs it for one pass over the test video (warm-up frames excluded). Each repeat
runs in its own subprocess so global state (trackers, counters) starts fresh.

    python tools/benchmark.py --repeats 3 --out experiments/EXP-001-baseline
"""
import argparse
import ctypes
import json
import os
import platform
import shutil
import statistics as st
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = ["helmet_best.pt", "plate_best.pt", "yolo11s.pt"]
VIDEO = "test_video.mp4"
REASONS = ["NO-HELMET", "OVERLOAD", "RED-LIGHT", "WRONG-WAY", "WRONG LANE"]


# --------------------------------------------------------------------- helpers
def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize(ms):
    return {"n": len(ms), "mean_ms": round(st.mean(ms), 2) if ms else None,
            "p50_ms": round(pct(ms, 50), 2) if ms else None,
            "p95_ms": round(pct(ms, 95), 2) if ms else None}


def gpu_query():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=10).strip().split(",")
        t, clk, pw, ut, mem = [float(x) for x in out]
        return {"temp_c": t, "sm_clock_mhz": clk, "power_w": pw, "util_pct": ut, "mem_used_mib": mem}
    except Exception:
        return None


def power_state():
    """AC/battery + active Windows power scheme (best effort)."""
    info = {}
    try:
        class S(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        s = S()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
        info["ac_line"] = {0: "battery", 1: "plugged_in"}.get(s.ACLineStatus, "unknown")
        info["battery_pct"] = s.BatteryLifePercent
    except Exception as e:
        info["ac_line_error"] = str(e)
    try:
        info["power_scheme"] = subprocess.check_output(["powercfg", "/getactivescheme"], text=True, timeout=10).strip()
    except Exception as e:
        info["power_scheme_error"] = str(e)
    return info


def env_info():
    import torch, cv2, numpy, ultralytics
    info = {
        "python": platform.python_version(), "os": platform.platform(),
        "cpu": platform.processor(), "cpu_count": os.cpu_count(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "opencv": cv2.__version__, "numpy": numpy.__version__, "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20) if torch.cuda.is_available() else None,
        "power": power_state(),
    }
    try:
        info["nvidia_driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    except Exception:
        pass
    return info


# --------------------------------------------------------------- one run (child)
def run_once(n_warmup, out_json, detail=False):
    import cv2
    import torch

    work = REPO / "outputs" / "work" / f"bench_{os.getpid()}"            # scratch under outputs/ (git-ignored); nothing is deleted
    work.mkdir(parents=True, exist_ok=True)
    for f in WEIGHTS + [VIDEO]:
        shutil.copy2(REPO / f, work / f)           # app.py writes to ./static under cwd -> keep repo clean
    shutil.copytree(REPO / "templates", work / "templates")
    # Always benchmark the ORIGINAL app.py as committed at tag v0-baseline (the repo's app.py is now a thin wrapper).
    (work / "app.py").write_text(subprocess.check_output(
        ["git", "show", "v0-baseline:app.py"], cwd=REPO, text=True, encoding="utf-8"), encoding="utf-8")
    os.chdir(work)
    sys.path.insert(0, str(work))
    import app                                      # original, unmodified app.py

    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    t = {"coco": [], "helmet": [], "plate": [], "read": [], "encode": []}
    labels = []  # (frame_idx, label) for violation labels
    cur = {"i": -1}

    def timed(obj, attr, key):
        orig = getattr(obj, attr)

        def w(*a, **k):
            sync(); t0 = time.perf_counter()
            r = orig(*a, **k)
            sync(); t[key].append((time.perf_counter() - t0) * 1000)
            return r
        setattr(obj, attr, w)

    if not detail:
        timed(app.standard_coco_model, "track", "coco")
        timed(app.helmet_model, "track", "helmet")
        timed(app.plate_model, "predict", "plate")      # YOLO.__call__ -> self.predict

    orig_imencode = cv2.imencode

    def imencode(*a, **k):
        t0 = time.perf_counter(); r = orig_imencode(*a, **k); t["encode"].append((time.perf_counter() - t0) * 1000); return r
    if not detail:
        cv2.imencode = imencode

    orig_box = app.draw_sleek_box

    def box(img, x1, y1, x2, y2, label, color, thickness=1):
        if any(r in label for r in REASONS):
            labels.append((cur["i"], label))
        return orig_box(img, x1, y1, x2, y2, label, color, thickness)
    if not detail:
        app.draw_sleek_box = box

    class CapProxy:
        def __init__(self, c): self._c = c
        def isOpened(self): return self._c.isOpened()
        def set(self, *a): return self._c.set(*a)
        def release(self): return self._c.release()
        def read(self):
            t0 = time.perf_counter(); r = self._c.read(); t["read"].append((time.perf_counter() - t0) * 1000); return r
    app.cap = CapProxy(app.cap)

    # ---- detail mode: split every model call into speed parts + tracker, and account for CPU-side glue
    series = {}                     # key -> list of per-frame ms
    acc = {}                        # key -> ms accumulated in current frame
    in_draw = [False]

    def add(key, ms):
        acc[key] = acc.get(key, 0.0) + ms

    if detail:
        import ultralytics.trackers.track as tt
        orig_cb = tt.on_predict_postprocess_end
        cb_owner = {"name": None}

        def cb(predictor, persist=False):
            for tr in predictor.trackers:               # wrap tracker.update once (pure BoT-SORT update time)
                if not getattr(tr, "_timed", False):
                    ou = tr.update
                    def upd(*a, _ou=ou, _o=cb_owner, **k):
                        t0 = time.perf_counter(); r = _ou(*a, **k)
                        add(_o["name"] + "_tracker_update", (time.perf_counter() - t0) * 1000); return r
                    tr.update = upd; tr._timed = True
            t0 = time.perf_counter()
            orig_cb(predictor, persist=persist)
            add(cb_owner["name"] + "_tracker_cb", (time.perf_counter() - t0) * 1000)
        tt.on_predict_postprocess_end = cb              # looked up at register_tracker() time -> must precede first track()

        def detail_wrap(obj, attr, name):
            orig = getattr(obj, attr)
            def w(*a, **k):
                cb_owner["name"] = name
                sync(); t0 = time.perf_counter()
                r = orig(*a, **k)
                sync(); add(name + "_wall", (time.perf_counter() - t0) * 1000)
                sp = r[0].speed
                add(name + "_calls", 1); add(name + "_pre", sp["preprocess"]); add(name + "_inf", sp["inference"]); add(name + "_post", sp["postprocess"])
                return r
            setattr(obj, attr, w)
        detail_wrap(app.standard_coco_model, "track", "coco")
        detail_wrap(app.helmet_model, "track", "helmet")
        detail_wrap(app.plate_model, "predict", "plate")   # applied on top of nothing else: v1 timers not installed in detail mode

        class CvProxy:
            def __init__(self, m): self._m = m
            def __getattr__(self, n):
                f = getattr(self._m, n)
                if n in ("resize", "polylines", "putText", "pointPolygonTest", "imwrite", "imencode") and callable(f):
                    def w(*a, _f=f, _n=n, **k):
                        if in_draw[0]: return _f(*a, **k)
                        t0 = time.perf_counter(); r = _f(*a, **k)
                        add("cv_" + _n, (time.perf_counter() - t0) * 1000); return r
                    return w
                return f
        app.cv2 = CvProxy(cv2)

        ob = app.draw_sleek_box
        def dbox(*a, **k):
            in_draw[0] = True; t0 = time.perf_counter()
            try:
                lab = a[5] if len(a) > 5 else k.get("label", "")
                if any(r in lab for r in REASONS): labels.append((cur["i"], lab))
                return ob(*a, **k)
            finally: in_draw[0] = False; add("draw_sleek_box", (time.perf_counter() - t0) * 1000)
        app.draw_sleek_box = dbox
        ol = app.detect_traffic_light_color
        def dl(*a, **k):
            t0 = time.perf_counter(); r = ol(*a, **k); add("light_hsv", (time.perf_counter() - t0) * 1000); return r
        app.detect_traffic_light_color = dl

    total_frames = int(app.cap._c.get(cv2.CAP_PROP_FRAME_COUNT))
    n_measure = total_frames - n_warmup              # exactly one pass, no loop wrap
    gen = app.generate_frames()

    samples, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            g = gpu_query()
            if g: samples.append(g)
            stop.wait(2)
    th = threading.Thread(target=poll, daemon=True); th.start()
    pre = gpu_query()

    frame_ms, captured_seen = [], set()
    cpu0 = None
    events = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(n_warmup + n_measure):
        cur["i"] = i
        acc.clear()
        t0 = time.perf_counter()
        next(gen)
        dt = (time.perf_counter() - t0) * 1000
        if i == n_warmup - 1:
            for k in t: t[k].clear()                  # drop warm-up from stage timers
            cpu0 = time.process_time()
        if i >= n_warmup:
            frame_ms.append(dt)
            if detail:
                acc["read"] = t["read"][-1]
                acc["frame_total"] = dt
                for key in set(series) | set(acc): series.setdefault(key, [0.0] * (len(frame_ms) - 1)).append(acc.get(key, 0.0))
        for vid in app.captured_violator_ids - captured_seen:
            events.append({"frame": i, "kind": "plate_capture", "track": vid})
        captured_seen = set(app.captured_violator_ids)
    stop.set(); th.join(timeout=3)
    post = gpu_query()

    # first appearance of each (track, reason)
    first = {}
    for fi, lab in labels:
        if fi < 0: continue
        first.setdefault(lab, fi)
    for lab, fi in sorted(first.items(), key=lambda x: x[1]):
        events.append({"frame": fi, "kind": "label_first_seen", "label": lab})

    total_s = sum(frame_ms) / 1000
    cpu_s = time.process_time() - cpu0
    stages = {k: summarize(v) for k, v in t.items()}
    stages["frame_total"] = summarize(frame_ms)
    res = {
        "n_frames_measured": len(frame_ms), "n_warmup": n_warmup, "video_total_frames": total_frames,
        "fps_end_to_end": round(len(frame_ms) / total_s, 2),
        "stages": stages,
        "plate_calls": int(sum(series["plate_calls"])) if detail and "plate_calls" in series else len(t["plate"]),
        "plate_calls_per_frame": round((sum(series["plate_calls"]) if detail and "plate_calls" in series else len(t["plate"])) / len(frame_ms), 3),
        "gpu_peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "gpu_before": pre, "gpu_after": post,
        "gpu_peak_temp_c": max((s["temp_c"] for s in samples), default=None),
        "gpu_min_sm_clock_mhz": min((s["sm_clock_mhz"] for s in samples), default=None),
        "gpu_max_power_w": max((s["power_w"] for s in samples), default=None),
        "events": events,
        "process_cpu_seconds": round(cpu_s, 2), "wall_seconds": round(total_s, 2),
        "process_cpu_over_wall": round(cpu_s / total_s, 3),
        "gpu_mean_util_pct": round(st.mean(x["util_pct"] for x in samples), 1) if samples else None,
    }
    if detail:
        EXCL = ["read", "coco_wall", "helmet_wall", "plate_wall", "draw_sleek_box", "cv_polylines", "cv_putText",
                "cv_resize", "cv_imwrite", "cv_imencode", "cv_pointPolygonTest", "light_hsv"]
        n = len(frame_ms)
        for key in EXCL: series.setdefault(key, [0.0] * n)
        series["unallocated"] = [series["frame_total"][j] - sum(series[k][j] for k in EXCL) for j in range(n)]
        for m_ in ("coco", "helmet"):
            series[m_ + "_track_overhead"] = [series[m_ + "_wall"][j] - series[m_ + "_pre"][j] - series[m_ + "_inf"][j]
                                              - series[m_ + "_post"][j] - series.get(m_ + "_tracker_cb", [0.0] * n)[j]
                                              for j in range(n)]
        series["plate_predict_overhead"] = [series["plate_wall"][j] - series["plate_pre"][j] - series["plate_inf"][j]
                                            - series["plate_post"][j] for j in range(n)] if "plate_pre" in series else [0.0] * n
        res["detail"] = {k: summarize(v) for k, v in series.items()}
        res["plate_nonzero_frames_pct"] = round(100 * sum(1 for x in series["plate_wall"] if x > 0) / n, 1)
    Path(out_json).write_text(json.dumps(res, indent=2), encoding="utf-8")


# ---------------------------------------------------------- model complexity
def model_complexity():
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    out = {}
    for f in WEIGHTS:
        m = YOLO(str(REPO / f))
        params = sum(p.numel() for p in m.model.parameters())
        out[f] = {"params": params, "gflops_640": round(get_flops(m.model, 640), 2),
                  "file_bytes": (REPO / f).stat().st_size, "classes": len(m.names)}
    return out


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--out", default="experiments/EXP-001-baseline")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--detail", action="store_true", help="split model calls (pre/inf/post/tracker) and CPU-side glue")
    a = ap.parse_args()
    if a.child:
        run_once(a.warmup, a.child, a.detail); return

    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for r in range(a.repeats):
        tmp = out / f"_run{r + 1}.json"
        print(f"[bench] run {r + 1}/{a.repeats} ...", flush=True)
        subprocess.check_call([sys.executable, __file__, "--child", str(tmp), "--warmup", str(a.warmup)]
                              + (["--detail"] if a.detail else []))
        runs.append(json.loads(tmp.read_text(encoding="utf-8")))
        time.sleep(5)
    fps = [x["fps_end_to_end"] for x in runs]
    med_idx = sorted(range(len(runs)), key=lambda i: fps[i])[len(runs) // 2]
    stage_names = [k for k, v in runs[0]["stages"].items() if v["p50_ms"] is not None]   # v1 timers are empty in --detail mode
    summary = {
        "fps_runs": fps, "fps_median": st.median(fps),
        "fps_spread_pct": round((max(fps) - min(fps)) / st.median(fps) * 100, 2),
        "stages_median_of_runs": {k: {m: round(st.median([r["stages"][k][m] for r in runs]), 2)
                                      for m in ("mean_ms", "p50_ms", "p95_ms")} for k in stage_names},
        "plate_calls_per_frame_median": st.median([r["plate_calls_per_frame"] for r in runs]),
        "gpu_peak_alloc_mib_max": max(r["gpu_peak_alloc_mib"] for r in runs),
        "gpu_peak_temp_c_per_run": [r["gpu_peak_temp_c"] for r in runs],
        "gpu_min_sm_clock_mhz_per_run": [r["gpu_min_sm_clock_mhz"] for r in runs],
        "gpu_max_power_w_per_run": [r["gpu_max_power_w"] for r in runs],
    }
    if a.detail:
        summary["detail_median_of_runs"] = {k: {m: round(st.median([r["detail"][k][m] for r in runs]), 2)
                                                for m in ("mean_ms", "p50_ms", "p95_ms")} for k in runs[0]["detail"]}
        for key in ("process_cpu_over_wall", "gpu_mean_util_pct", "plate_nonzero_frames_pct"):
            summary[key + "_per_run"] = [r.get(key) for r in runs]
    metrics = {"experiment": "EXP-001-baseline" + ("/baseline-v2" if a.detail else ""), "pipeline": "app.py @ v0-baseline (unmodified), PyTorch FP32, 1280x720 resize",
               "video": VIDEO, "protocol": "one full pass of the video after warm-up; each repeat in a fresh process; "
               "frame time = time for next(generate_frames()) (includes read, 3 models, draw, evidence save, JPEG encode)",
               "summary": summary, "model_complexity": model_complexity(), "runs": runs}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "env.json").write_text(json.dumps(env_info(), indent=2), encoding="utf-8")
    ev = {"note": "characterization snapshot from run with median FPS; track ids/timing are not bit-exact across runs "
                  "(GPU non-determinism + tracker); compare counts and approximate frames.",
          "events": runs[med_idx]["events"], "events_per_run_counts": [len(r["events"]) for r in runs]}
    (out / "events.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
    for r in range(a.repeats):
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
