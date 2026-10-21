English translation of README.vi.md (original, Vietnamese).

# EXP-023 – why the motorcycle tracks of the helmet model die before the stop line (test_video) – diagnosis, nothing was changed

Question (from EXP-022): the `stop_line_crossing` rule saw 0 line crossings on 63 motorcycle tracks; the tracks almost never reach the line (median closest distance -168 px). Code: `diagnose.py` (raw numbers and results: `raw_detections.json`, `results.json`), `coco_source_check.py` (what-if, `coco_source_crossings.json`), the M_257 strip image `m257_strip.jpg` (not committed: people are visible). Default pipeline, 787 frames, 30 FPS; RED phase = frames 0–279 (0–9.3 s), GREEN = from frame 280. "Above the line" = box bottom y < 290 (the line is at y 277–302). Raw detection = `predict` without a tracker.

## (a) Is there any ROI / mask / position filter that cuts the upper part of the frame? **No.**
Check of the code and config:
- `frame_size: [1280, 720]`: every frame is only resized, no crop, no mask (`Pipeline.process`).
- `Detectors._track_kw`: only passes `conf`, tracker, device; no `classes`, custom `imgsz`, ROI.
- `sentinel/perception.py::tracks_from_results`: passes the box through, no position filter.
- The only places that filter by position are the rules: `violation_zone` (old zone rule), `wrong_lane_zone` (null), the stop line/lanes of the new rule, and the assignment of heads to a motorcycle box (head inside the box). Nowhere is a box dropped by position.
- Numerical evidence: COCO (same frames, same pipeline) sees 0.7–0.9 motorcycles per frame in the y band 200–280 at conf 0.35 (GREEN phase; RED phase: `results.json`), so the upper part of the frame is **not cut**.
**Hypothesis (a) is rejected.**

## (b) Does the detector see motorcycles above the line? For the helmet model: **almost not**; for COCO: yes
Number of raw `motorcycle` detections (counted by box, bottom y < 290):

| model, conf | RED phase (280 frames) | GREEN phase (507 frames) |
|---|---|---|
| helmet_best.pt, 0.55 (pipeline) | **0** / 811 detections | 13 / 1673 |
| helmet_best.pt, 0.1 | 10 / 2942 | 57 / 4475 |
| COCO yolo11s.pt, 0.35 (pipeline) | 831 / 3053 (258 frames have ≥1) | 1025 / 6315 (428 frames have ≥1) |
| COCO yolo11s.pt, 0.1 | 3033 / 10105 | 5522 / 17746 |

The distribution by 40 px band (detections per frame) is in `results.json`; for example GREEN phase, pipeline conf, band 240–280: helmet 0.002, COCO 0.68; band 200–240: COCO 0.93 (helmet none); RED phase, conf 0.1, band 200–240: COCO 6.3. Pipeline tracks (helmet, 0.55): **0 detections above the line in the RED phase, 7 in the GREEN phase**.
Even lowering conf to 0.1, the helmet model has only 10 + 57 detections above the line in 787 frames (COCO: 8555 at conf 0.1). **Hypothesis (b) is true for the helmet model**: it does not detect motorcycles from the line upwards. (The cause was not checked: vehicles small/far compared with the training data, the default 640 resolution of Ultralytics; no guessing.)

## (c) Does the tracker drop objects that the detector still sees? **Rarely (≈5%)**
61 tracks end (not counting the 2 tracks still alive in the last 3 frames of the video). The bottom y where tracks end (`results.json`): 280–320: 6, 320–360: 13, 360–400: 5, 400–440: 5, 440–480: 10, 480–520: 7, 520–560: 1, 560–600: 5, 640–760: 9: **spread out gradually, no sharp threshold** (not like a mask/cut at one fixed y). For each ended track, the next 3 frames were examined (IoU ≥ 0.3 with the last box):
- 58/61: **no detection at the pipeline conf (0.55)** near the last box. Among them **56/61 still have a low-confidence detection** (0.1 ≤ conf < 0.55) nearby: the track dies because the detector falls below the 0.55 threshold, not because the tracker matched wrongly.
- 2/61: the detection is still there but already belongs to another track (ID change/merge).
- 1/61: there is a detection that nobody took (tracker matching error).
**Hypothesis (c) explains ~3–5%** of the dead tracks; most (56/61 ≈ 92%) is conf < 0.55 of the helmet model, together with the helmet model not seeing motorcycles in the area near the line (b).

## M_257 checked by hand (`m257_strip.jpg`, frames 210–260 every 5)
- Helmet model: the M_257 box only lives in frames 215–221 because conf fluctuates 0.28 (f210) → 0.57 → 0.69 → 0.13 (f225): it falls below 0.55 so the track dies. The 111×130 px box is **off-centre and much larger** than the rider (~35×70): the bottom-centre point of the helmet box is a poor approximation of the vehicle position.
- COCO: sees that rider in **every frame 220–280**, conf 0.50–0.60, tight box. Bottom y: 336 (f220) → 333 → 330 → 328 → 325 → 321 → 318 → 316 → 312 (f260) → 309 → 305 → 302 (f275) → 298 (f280): **~0.6 px/frame ≈ 19 px/s**, the vehicle creeps slowly up to the line (the line is at y ≈ 287 at x ≈ 783) while the light is red, and **has not crossed the line when the light turns green (f280)**. This is a vehicle waiting for the light, **not a red-light-running case**; the description "moving" in the first version of the EXP-022 README was wrong and has been corrected. The old zone rule reported it by mistake; the new rule is right not to report it.

## What-if: the new rule takes the COCO motorcycle tracks instead of the helmet ones (`coco_source_check.py`, the pipeline was not changed)
Same tracker yaml, conf 0.35, approved geometry, ROI light (EXP-022): 242 tracks (IDs break more), **99 tracks are beyond the line at some moment** (helmet: 2), **13 line crossings**: 9 on GREEN (not reported because the light is green or the lane is exempt), 4 on RED **all in the right-turn lane (exempt)**: **0 reported**. By lane: through 5 (all 5 on green), right-turn 7, left-turn 1 (on green). There are no real labels so it cannot be said how correct these crossings are, it only shows that the COCO track source **reaches the line** while the helmet source does not.

## Conclusion (by evidence)
| hypothesis | explains | evidence |
|---|---|---|
| (a) mask/ROI/position filter | 0% | not in code/config; COCO sees motorcycles above the line in the same frames |
| (b) detector (helmet model) does not see / low conf near the line and above | most: 56/61 tracks (≈92%) die from conf < 0.55 near the last box; above the line only 0–13 detections at the pipeline conf (COCO: 831–1025) | tables (b), (c) |
| (c) tracker | ≈3–5% (3/61) | section (c) |
Secondary: the helmet model box is wide/off-centre (M_257: 111×130 versus a rider ~35×70), so the bottom-centre point is unreliable even when the track is alive.

## Proposed fix: approved and done (2026-10-19)
The project owner approved: the `stop_line_crossing` rule uses the COCO motorcycle tracks (`K_`), links the reason with the helmet track, merges events when an ID breaks; **no** lowering of `models.helmet.conf`, no training, no change to the helmet/overload rules. Details, the linking criterion (IoU >= 0.3 or centre inside the box), the merge parameters (1 s, 40 px) and the result on test_video: section "Red-light rule on COCO tracks" of `experiments/EXP-022-light-roi/README.md`. The error that was fixed is that helmet boxes/tracks **do not reach the line**; M_257 is still a vehicle waiting (the old zone rule reported it falsely, the new rule is right not to report it).

**Remaining limit:** the helmet model almost does not see motorcycles from the stop line upwards (13 versus 1000+ COCO detections); the deeper cause (vehicle size in the training data versus test_video) is **not checked**, left until there are labels. Direction for later (not done): pseudo-label far motorcycles with COCO into the training data (review a sample by hand), then retrain (see `docs/PLAN.md`, G9).
