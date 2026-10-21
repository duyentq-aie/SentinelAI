English translation of NOTES.vi.md (original, Vietnamese).

# EXP-010 – P4 optimisation steps measured as ratios to v0 (2026-10-18)

Command: `python tools/benchmark_pipeline.py` with 5 configurations + v0, 3 interleaved rounds, v0 runs first in each round. Data: `summary.json`, `<cfg>/metrics.json`, `env.json`, `runs/`.

| configuration | stream FPS (median, range) | ratio vs v0 of the same round (median, range) |
|---|---|---|
| v0 (original code) | 6.82 (6.72–8.68) | 1× |
| p4_before (whole-frame blend) | 8.90 (8.63–9.57) | 1.28× |
| p4a (ROI blend) | 9.78 | 1.20× (1.13–1.50) |
| Ultralytics default tracker (GMC sparse) | 10.09 | 1.40× |
| **default** (BoT-SORT, no GMC – TEMPORARY) | 16.01 | **2.27×** (1.84–2.42) |
| ByteTrack | 16.40 | 2.23× (1.89–2.54) |

## This session is noisy – read it accordingly
- The background process `Ld9BoxHeadless.exe` (the LDPlayer emulator, ~20% CPU) ran during rounds 2–3 (`cpu_other_pct` 4.3 → 22.7 → 21.4); v0 stream by round: 8.68, 6.72, 6.82.
- Because v0 and the measured configuration are subject to the same load within each round, the ratio is more stable than absolute FPS, but the range is still wide (e.g. default 1.84–2.42): the background load changes **between** the v0 run and the configuration run, it does not cut evenly.
- GPU clock 1500 MHz for v0/GMC, 1695–1942 MHz for default/ByteTrack (no throttling) – the GPU is idle, so the clock does not decide the FPS.
- **Do not use this table to conclude anything about the small steps** (p4a vs p4_before is within the noise). Cleaner numbers to quote: EXP-009 (quiet session, 5 rounds): default **2.42×** (2.40–2.46), GMC tracker **1.46×** (1.45–1.50).
- Safe statement: dropping GMC (the temporary default) gives ≈ 2.2–2.4× v0; keeping GMC gives ≈ 1.4–1.5× v0. There is no claim of "reaching 15 FPS".
