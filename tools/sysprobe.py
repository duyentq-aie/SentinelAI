"""System telemetry sampled while a benchmark runs (to explain why the same code is 7 FPS in one session and 12 in another).

Per sample (every `interval` s): GPU SM/memory clock, board power, temperature, utilisation, P-state and the NVIDIA clock-throttle reason bits;
CPU: Windows "% Processor Performance" (real frequency relative to base, via typeperf; psutil.cpu_freq() is constant on Windows), whole-machine CPU
load, load of OTHER processes (machine minus the benchmark's own process tree) and the heaviest other processes.
Everything is best effort: a missing tool gives None, never an exception.
"""
import os
import statistics as st
import subprocess
import threading
import time

import psutil

GPU_FIELDS = ["clocks.sm", "clocks.max.sm", "clocks.mem", "power.draw", "power.limit", "temperature.gpu", "utilization.gpu", "pstate",
              "clocks_event_reasons.active", "memory.used"]
# NVIDIA clock-event reason bits (nvidia-smi -q -d PERFORMANCE)
REASON_BITS = {0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
               0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake_slowdown"}


def _run(cmd, timeout=10):
    try:
        return subprocess.check_output(cmd, text=True, timeout=timeout, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _num(x):
    try:
        return float(str(x).split()[0])
    except Exception:
        return None


def gpu_sample():
    out = _run(["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"])
    if not out:
        return {}
    v = [x.strip() for x in out.splitlines()[0].split(",")]
    d = dict(zip(GPU_FIELDS, v))
    try:
        reasons = int(d["clocks_event_reasons.active"], 16)
    except Exception:
        reasons = None
    return {"sm_clock_mhz": _num(d["clocks.sm"]), "sm_clock_max_mhz": _num(d["clocks.max.sm"]), "mem_clock_mhz": _num(d["clocks.mem"]),
            "power_w": _num(d["power.draw"]), "power_limit_w": _num(d["power.limit"]), "temp_c": _num(d["temperature.gpu"]),
            "util_pct": _num(d["utilization.gpu"]), "pstate": d["pstate"], "throttle_reasons": reasons, "mem_used_mib": _num(d["memory.used"])}


def cpu_performance_pct():
    """Windows: effective CPU frequency as % of base ('% Processor Performance', > 100 = turbo). None elsewhere."""
    out = _run(["typeperf", r"\Processor Information(_Total)\% Processor Performance", "-sc", "1"], timeout=15)
    if not out:
        return None
    for line in out.splitlines():
        parts = [p.strip().strip('"') for p in line.split('","')]
        if len(parts) >= 2:
            try:
                return float(parts[-1].strip('"'))
            except ValueError:
                continue
    return None


def _tree(pid):
    try:
        p = psutil.Process(pid)
        return {pid} | {c.pid for c in p.children(recursive=True)}
    except Exception:
        return {pid}


class Sampler(threading.Thread):
    def __init__(self, interval=2.0, cpu_perf_every=3):
        super().__init__(daemon=True)
        self.interval, self.cpu_perf_every = interval, cpu_perf_every
        self.samples, self._stop_ev = [], threading.Event()
        self._own = _tree(os.getpid())
        psutil.cpu_percent(None)
        for p in psutil.process_iter():                        # prime the per-process CPU counters
            try:
                p.cpu_percent(None)
            except Exception:
                pass

    def run(self):
        i = 0
        while not self._stop_ev.wait(self.interval):
            s = gpu_sample()
            s["t"] = time.time()
            s["cpu_total_pct"] = psutil.cpu_percent(None)
            own = other = 0.0
            heavy = []
            self._own = _tree(os.getpid())
            for p in psutil.process_iter(["pid", "name"]):
                try:
                    c = p.cpu_percent(None)
                except Exception:
                    continue
                if p.info["pid"] in self._own or p.info["pid"] == 0:
                    own += c
                else:
                    other += c
                    heavy.append((c, p.info["name"]))
            n = psutil.cpu_count() or 1
            s["cpu_own_pct"], s["cpu_other_pct"] = round(own / n, 1), round(other / n, 1)        # % of the whole machine
            s["top_other"] = [(round(c / n, 1), nme) for c, nme in sorted(heavy, reverse=True)[:3] if c > 0]
            if i % self.cpu_perf_every == 0:
                s["cpu_perf_pct"] = cpu_performance_pct()
            self.samples.append(s)
            i += 1

    def stop(self):
        self._stop_ev.set()
        self.join(timeout=20)
        return self.samples


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"median": round(st.median(xs), 2), "min": round(min(xs), 2), "max": round(max(xs), 2)}


def summarize(samples):
    if not samples:
        return {"n_samples": 0}
    reasons = [s.get("throttle_reasons") for s in samples if s.get("throttle_reasons") is not None]
    frac = {name: round(sum(1 for r in reasons if r & bit) / len(reasons), 2) for bit, name in REASON_BITS.items()} if reasons else {}
    others = {}
    for s in samples:
        for c, nme in s.get("top_other", []):
            others[nme] = max(others.get(nme, 0), c)
    return {"n_samples": len(samples),
            "gpu_sm_clock_mhz": _stats([s.get("sm_clock_mhz") for s in samples]), "gpu_sm_clock_max_mhz": next((s["sm_clock_max_mhz"] for s in samples if s.get("sm_clock_max_mhz")), None),
            "gpu_mem_clock_mhz": _stats([s.get("mem_clock_mhz") for s in samples]),
            "gpu_power_w": _stats([s.get("power_w") for s in samples]), "gpu_power_limit_w": next((s["power_limit_w"] for s in samples if s.get("power_limit_w") is not None), None),
            "gpu_temp_c": _stats([s.get("temp_c") for s in samples]), "gpu_util_pct": _stats([s.get("util_pct") for s in samples]),
            "gpu_throttle_reason_fraction": frac,
            "cpu_perf_pct": _stats([s.get("cpu_perf_pct") for s in samples]),
            "cpu_total_pct": _stats([s.get("cpu_total_pct") for s in samples]),
            "cpu_own_pct": _stats([s.get("cpu_own_pct") for s in samples]),
            "cpu_other_pct": _stats([s.get("cpu_other_pct") for s in samples]),
            "heaviest_other_processes_peak_pct": dict(sorted(others.items(), key=lambda kv: -kv[1])[:5])}


def static_info():
    """Facts that do not change during a run."""
    info = {"gpus": _run(["nvidia-smi", "-L"]), "power_plan": _run(["powercfg", "/getactivescheme"]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cpu_count": psutil.cpu_count(), "cpu_count_physical": psutil.cpu_count(logical=False)}
    try:
        b = psutil.sensors_battery()
        info["ac_power"] = None if b is None else bool(b.power_plugged)
        info["battery_pct"] = None if b is None else b.percent
    except Exception:
        pass
    apps = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    info["gpu_compute_apps_at_start"] = apps.splitlines() if apps else []
    mem = _run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"])
    info["gpu_memory_used_total_at_start"] = mem
    return info


if __name__ == "__main__":
    import json
    import sys
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10
    s = Sampler(interval=1.0, cpu_perf_every=2)
    s.start()
    time.sleep(secs)
    print(json.dumps({"static": static_info(), "summary": summarize(s.stop())}, indent=2))
