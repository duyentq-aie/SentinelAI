English translation of NOTES.vi.md (original, Vietnamese).

# EXP-009 – Why absolute FPS changes between measurement sessions, and why v0 must be measured alongside (2026-10-18)

## New measurement convention (from here on)
`tools/benchmark_pipeline.py` **always runs `v0`** (the original `app.py` code at tag `v0-baseline`, unmodified) in the same session, interleaved round by round with the other configurations, and reports the
**ratio to v0 measured in the same round** (median of the rounds + range). Absolute FPS is for reference only. Telemetry (`tools/sysprobe.py`) is sampled every 2 s during each run:
GPU SM/memory clocks, power, temperature, NVIDIA throttle reason bits, CPU "% Processor Performance" (effective frequency; `psutil.cpu_freq()` is fixed on Windows so it is not used),
CPU load of **other processes** and the heaviest process. `env.json` records the power plan, AC, GPU and the processes that use the GPU at the start.

## Controlled experiment: CPU background load
Same configuration, same code, two consecutive sessions: **quiet** (5 rounds) and **with 6 CPU-burning processes** (`tools/cpu_burn.py --procs 6`, 6 of the 12 logical threads; 3 rounds).
`default` = the current default configuration (BoT-SORT without GMC); `tracker_botsort_gmc` = the same configuration but the Ultralytics default tracker.

| | stream FPS quiet | stream FPS with background load | change | ratio vs v0 quiet | ratio vs v0 with load |
|---|---|---|---|---|---|
| v0 (original code) | 8.61 (8.37–8.70) | 5.28 (5.16–5.33) | **−39%** | – | – |
| default (no GMC) | 20.86 (20.63–21.08) | 12.60 (12.47–12.73) | **−40%** | **2.42×** (2.40–2.46) | **2.39×** (2.36–2.44) |
| Ultralytics default tracker (GMC) | 12.57 (12.49–12.59) | 7.58 (7.54–7.61) | **−40%** | **1.46×** (1.45–1.50) | **1.44×** (1.42–1.46) |

headless (read + process + draw, no JPEG): v0 9.72 → 6.00 (−38%), default 27.86 → 17.02 (−39%), GMC 14.84 → 8.96 (−40%).

Telemetry of the two sessions:
| | quiet | with background load |
|---|---|---|
| CPU performance (effective frequency) | **150%** (turbo) | **114%** |
| CPU load of other processes | 4% | 55% |
| GPU SM clock | 1500 MHz (max 2100) | 1500 MHz |
| GPU power / temperature | 19–22 W / ≤ 61 °C | 17–19 W / ≤ 61 °C |
| throttle bits (power cap, thermal, brake…) | none (apart from `gpu_idle`) | none |
| GPU util | 21–32% | 11–20% |

**Measured conclusion:** (1) the pipeline is limited by the **CPU**: when ~50% of the CPU is taken, the effective frequency drops 150% → 114% and every configuration loses ~40% FPS, while the GPU (clock, temperature, power) does not change and stays idle;
(2) **the ratio to v0 in the same session is almost unchanged** (2.42× → 2.39×, 1.46× → 1.44×, difference ≤ 2%), so this is the trustworthy way to report; (3) the GPU is not the cause of the variation between sessions on this machine.

## What about "7 vs 12 FPS"? (correction of an earlier statement)
I once wrote "the same code runs at 7 FPS in this session, 12 FPS in another". **That statement was partly wrong**: 7.0–7.4 is the **v0** pipeline (EXP-001, 1.73 plate calls/frame, wrong-way on, whole-frame drawing) while 12.05 is the new pipeline in EXP-006
(windowed rules, wrong-way off, 0.018 plate calls/frame). Most of the difference is **different amounts of work**, not the machine. Today's quiet session shows that this alone is 1.46× (GMC on) up to 2.42× (GMC off) over v0.
A **real** variation between sessions with the same code still exists, but it is smaller:
- v0: 7.38 (EXP-001, evening of 17/10, High performance plan; old harness `tools/benchmark.py`, so for reference only) → 8.61 (today's quiet session, new harness) = +17%; a short 60-frame run gave 10.04 (not used: too short, cold start).
- the same `attempts30` configuration (wrong-way on): 8.88 (EXP-003, ~00:00) versus 10.58 (EXP-004, ~00:30) = +19%.
**The cause of this variation cannot be determined again** because there was no telemetry then. The experiment above proves one factor that is large enough (CPU background load ~50% → −40%) and rules out the GPU factor (the clock was always 1500 MHz — including in the old sessions,
`gpu_min_sm_clock_mhz = 1500` in every EXP — no throttling, temperature 58–66 °C). Background processes commonly seen during measurements: `sqlservr.exe`, `MsMpEng.exe` (Defender), `Code.exe`, `claude.exe`, `esrv_svc.exe` (each 1–6% of the total CPU).
GPU in use: only the RTX 3050 Laptop (`nvidia-smi -L`), only a few Windows graphics processes (dwm, WebView, VS Code) were on the GPU at the start, taking ~170 MiB.
What has **not been checked**: Windows per-process power saving (EcoQoS) when the window is not in the foreground; CPU power limits (PL1/PL2) during long hot periods; CPU temperature (cannot be read with the available tools).

## Consequences for the documents
`docs/REPORT.md` and `README.md` drop every "reaches 15 FPS" and every comparison of absolute FPS between sessions; they use ratios to v0 (EXP-010). `docs/MORNING_REPORT.md` and EXP-006 carry correction notes.
