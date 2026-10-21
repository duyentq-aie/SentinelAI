English translation of README.vi.md (original, Vietnamese).

# EXP-021 – processing at 10 FPS: `test_video.mp4` with stride 3 (30 → 10 FPS), 2026-10-19

Current default configuration (hysteresis on, GMC off), model `helmet_best.pt`; the pipeline is told fps = 30 / 3 = 10 (`Pipeline(fps=10)`): every parameter expressed in seconds is converted again (W = 5, K = 2, event close 10 frames, tracker buffer 10 ...; `docs/TIME_BASED_PARAMETERS.md`). Every 3rd frame is taken (263 frames). There is no ground truth yet: the comparisons below are about **differences** from the 30 FPS run, not about right/wrong.
Files: `run_stride.py` (run), `compare.py` (comparison), `compare.json`, `bench_summary.json`, `bench_stride1|3/` (benchmark `tools/benchmark_pipeline.py --stride`). Images/videos are not included.

## Events: 10 FPS versus 30 FPS (same video, same weights)
| | 30 FPS | 10 FPS |
|---|---|---|
| events | 14 (13 no_helmet + 1 overload) | 10 (9 no_helmet + 1 overload) |
| matched (same type, time span within ±15 original frames, overlapping boxes) | 7 | 7 |
| only at 30 FPS | 7: M_129 (71–146), M_175 (112–117), M_523 (451–484), M_573 (474–496), M_634 (533–588), M_707 (554–622), M_782 (582–595) | – |
| only at 10 FPS | – | 3: M_372 (426–456), M_659 (471–483), M_725 (498–534) |

`diff_events` (tolerance 3 or 6 original frames) reports 28 missing / 20 added / 1 changed: **not readable** because the tracker changes IDs when the fps changes (buffer, stride), so every event is under a different name (`M_153` → `M_127`...). The matching by time span and box is the one that means something.

Time spans of the 7 matched events (original frames, 10 FPS − 30 FPS): start −5…+16 frames (6 of the 7 events differ by ≤ 3 frames, i.e. ≤ 0.1 s), end −7…+17 frames:
| type | 30 FPS | 10 FPS | start offset | end offset |
|---|---|---|---|---|
| no_helmet | M_77 35–89 | M_69 36–87 | +1 | −2 |
| no_helmet | M_153 104–118 | M_127 99–135 | −5 | +17 |
| no_helmet | M_299 290–333 | M_392 306–333 | +16 | 0 |
| no_helmet | M_380 374–424 | M_505 375–423 | +1 | −1 |
| no_helmet | M_482 471–474 | M_620 474–474 | +3 | 0 |
| no_helmet | M_681 528–570 | M_869 531–570 | +3 | 0 |
| overload | M_350 328–337 | M_468 330–330 | +2 | −7 |

Read correctly: at 10 FPS **7 events are lost and 3 are added**. The two events that exist only at 30 FPS are M_175 and M_707, two flags that the EXP-020 analysis found only in the current version (not in v0) and that look like helmeted people by eye; M_129, M_523, M_573, M_634, M_782 are flags that also exist in v0. There are no correct labels, so it cannot be concluded that 10 FPS is "better" or "worse". The causes (not separable here): K/W = 2/5 instead of 1/3 (stricter when the window is full: 3/5 versus 8/15), the quantisation of `exit_ratio`, and the tracker changing IDs/breaking tracks when 2 of 3 frames are lost.

## Plates
- Number of plate detector calls over the whole run: **14 (30 FPS) versus 10 (10 FPS)**: every event finds the plate on the first attempt (`plate.attempts` = 1 for all 7 matched events, at both speeds). The maximum budget of 3 s = 90 attempts at 30 FPS, 30 attempts at 10 FPS is not reached.
- The frame where the plate is found differs by −5…+16 original frames from 30 FPS (the same offset as at the event start).

## Processing speed (no absolute FPS; ratio to v0 of the same session, 3 interleaved rounds, v0 always runs every frame)
| session | frames processed per run | `process` time per frame (p50) | v0 `process` (p50) | stream ratio vs v0 | headless ratio vs v0 |
|---|---|---|---|---|---|
| 30 FPS (stride 1) | 757 | 48–49 ms | 151–153 ms | **2.39×** (2.38–2.46) | 2.81× |
| 10 FPS (stride 3) | 233 | 48–51 ms | 150–156 ms | **2.46×** (2.29–2.52) | 2.89× |

- **The cost per processed frame is unchanged** (≈ 49 ms for both, the ratio to v0 is the same within the noise): the stride does not make the pipeline faster per frame, it only makes **fewer frames** (263 versus 787 per run), so the total computing time per second of video drops about 3 times.
- **The measurement session had background load** (`cpu_other` ≈ 35%): the ratio to v0 is more stable than absolute FPS; see EXP-009 about the noise.
- The speed on the target device (G7) cannot be inferred: this is an RTX 3050 Laptop + the dev machine CPU.

## Stability of motorcycle tracking: 30 FPS and 10 FPS (`tracking_stability.py`, `tracking_stability.json`; numbers only, nothing was changed)
The tracker of the helmet model (BoT-SORT, GMC off, default configuration), `motorcycle` class tracks on `test_video.mp4`. Three variants: 30 FPS (buffer 30 frames = 1 s); 10 FPS with the buffer converted to the same duration (10 frames, exactly as `Pipeline(fps=10)`); 10 FPS with the buffer left at 30 frames (= 3 s) to separate the effect of the buffer.

| | 30 FPS | 10 FPS (buffer 10) | 10 FPS (buffer 30) |
|---|---|---|---|
| unique tracks | 63 | 49 | 45 |
| median track length (s) | 1.37 | 1.30 | 1.80 |
| tracks shorter than 1 s | 23 (36.5%) | 15 (30.6%) | 7 (15.6%) |
| estimated number of ID changes¹ | 2 | 0 | 0 |
| box IoU between two consecutive frames of a track (median / 10th percentile) | 0.957 / 0.895 | 0.860 / 0.764 | 0.861 / 0.764 |
| centre displacement between two frames (px, median / 90th percentile) | 2.8 / 8.5 | 10.6 / 21.7 | 10.5 / 21.7 |
| steps with IoU < 0.3 | 0% | 0% | 0% |
| detections per processed frame | 3.3 | 3.2 | 3.2 |
¹ a track ends (not at the end of the video) and then a NEW track starts within ≤ 0.5 s, the centre of its first box is ≤ 0.5 × the longer side of the box from the centre of the last box of the old track, matched one-to-one; a heuristic, **low power** (only 2 cases at 30 FPS).

**The hypothesis "at 10 FPS the vehicles move farther so the tracker matches worse": not supported on this clip.** The displacement between two frames grows as predicted (≈ 3.8×: 2.8 → 10.6 px) but it only lowers the consecutive IoU from 0.96 to 0.86, still far from the matching threshold; no step is below 0.3. The number of unique tracks **drops** (49 versus 63), the share of tracks shorter than 1 s does not rise (30.6% versus 36.5%) and no ID change is detected. Note: this scene is mostly a queue of vehicles waiting at a red light and then moving slowly, so the motion is small; the conclusion does not apply to fast traffic. Keeping the 30-frame buffer at 10 FPS gives longer tracks (median 1.80 s) and fewer short tracks (15.6%), but these numbers do not say whether a longer buffer increases ID mix-ups (there are no ID labels).
Consequence: the 7 events lost and 3 events added at 10 FPS (section above) **cannot be explained by broken/changed tracker IDs**; the remaining causes (not separable here): K/W = 2/5, the quantisation of the window, fewer observations per vehicle.
