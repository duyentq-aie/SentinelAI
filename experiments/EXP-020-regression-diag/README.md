English translation of README.vi.md (original, Vietnamese).

# EXP-020 – diagnosis of "more false alarms, blurrier plate images" after the refactor (2026-10-19)

No images are included (people and plates are visible). The images are in `outputs/regression_diag/` (gitignored): `contact4_1..6.jpg` (contact sheet of 4 versions, 28 rows), `contact_sheet_1..5.jpg` (v0 versus current), `plates_v0_vs_current_1..3.jpg`. Original report: `outputs/regression_diag/REPORT.md`.

## Main findings
1. **v0 fed the helmet model and the plate model frames that had already been drawn on**: the original `app.py` draws on the very frame that goes into the model (traffic-light box, labels, vehicle boxes), so the helmet model sees the traffic-light box and the plate model sees the violation boxes. **The current version fixes this** (perception runs on the original frame `raw`, drawing goes on a copy `display`, since step 3.0b). On this clip that **does not change the number of no-helmet tracks** (2×2 table below: perception explains none of the differences), but it does change the stored plate crop images: the v0 crop contains the green `PLATE` box and a label drawn on it, the current crop is clean.
2. **The difference in the number of no-helmet flags comes from the rules, not from perception**: the 2×2 table (offline replay of the recorded tracks; wrong-way off):

| perception | rules | no-helmet tracks | overload tracks | flagged track-frames |
|---|---|---|---|---|
| v0 perception (overlay-contaminated, default tra | legacy | 12 | 2 | 314 |
| v0 perception (overlay-contaminated, default tra | windowed (no hysteresis) | 13 | 1 | 350 |
| v0 perception (overlay-contaminated, default tra | windowed + hysteresis 0.3 (current default) | 13 | 1 | 389 |
| clean frames + GMC on (3.0b fix, default until 1 | legacy | 12 | 2 | 314 |
| clean frames + GMC on (3.0b fix, default until 1 | windowed (no hysteresis) | 13 | 1 | 350 |
| clean frames + GMC on (3.0b fix, default until 1 | windowed + hysteresis 0.3 (current default) | 13 | 1 | 389 |
| current default (clean + GMC off) | legacy | 12 | 2 | 314 |
| current default (clean + GMC off) | windowed (no hysteresis) | 13 | 1 | 350 |
| current default (clean + GMC off) | windowed + hysteresis 0.3 (current default) | 13 | 1 | 389 |

   Column "rules": `legacy` = v0; `windowed (no hysteresis)`; `windowed + hysteresis 0.3` = the current default. Windowed adds 1 track and +36 track-frames (+11%); hysteresis adds +39 (+11%): in total the red boxes are held longer by ≈ 24%.
3. **The baseline level of false alarms is the same in every version**: 11 of the 13 current no-helmet events also exist in v0; many flags are people wearing a helmet.
4. **Video looping**: both v0 (3 rounds without reset: 12 tracks and 20 plate files per round, new ids every round) and the current version repeat the same vehicles in each round; the current version shows it more clearly because it has an event panel (fixed below).
5. **Plate images**: the same crop size (median area 1928 versus 1941 px), the same JPEG q95, cut from the same 1280×720 frame; sharpness inside the plate box: v0 2264, current 2962 (median Laplacian); there is no evidence that the stored image is blurrier. What can make the plate look blurrier when viewed in the web UI is the **low-quality MJPEG stream** (see below) and the **CSS that enlarges thumbnails**.

## Events of the 4 versions on the same video (one run, GPU, the same current default configuration)
| version | no_helmet | overload | wrong_way | total events (rows) |
|---|---|---|---|---|
| v0 | 12 | 2 | 8 | 22 |
| current+old model | 13 | 1 | 0 | 14 |
| EXP-015 3cls | 7 | 0 | 0 | 7 |
| EXP-016 4cls | 5 | 0 | 0 | 5 |

"current + old model" = `helmet_best.pt`; "EXP-015 3cls" and "EXP-016 4cls" = the current pipeline with the corresponding helmet model (EXP-016 is only used in the helmet role; the plate is still `plate_best.pt`). Matched by time ±1 s and overlapping boxes; 28 rows, the old row numbers are kept (1–24), new rows 25–28. **There are no right/wrong labels yet**: `grading_template.csv` is for you to grade (right / wrong / unclear) for each version; fewer events does not mean fewer false alarms.

## Changes in the web application (same batch)
- `web.jpeg_quality` default **90** (was 80). Measured on 12 frames with plates (1280×720, PSNR against the original frame), median:

| quality | frame size | PSNR plate region | PSNR whole frame |
|---|---|---|---|
| 80 | 192.7 KB | 35.2 dB | 36.5 dB |
| 90 | 270.3 KB | 38.9 dB | 39.4 dB |
| 95 | 374.0 KB | 42.4 dB | 41.8 dB |

  Level 90 trades +40% bandwidth over 80 for +3.7 dB in the plate region; level 95 almost doubles the size of 80. It only affects the stream; the evidence images are always ≥ 95.
- **The panel and the gallery only show the stored evidence images** (`frame.jpg`, `vehicle.jpg`, `plate.jpg`, q ≥ 95, from `static/violations`), never taken from the stream frames (checked in code: `/api/violations` and `/api/events` return file URLs; the stream only supplies the video element). **Thumbnail CSS**: before, the plate thumbnail in the panel used `object-fit: cover` (cutting off part of the 46×40 px crop when enlarged ~2.5×) and the gallery enlarged ~3× with the browser's smoothing (blurry). Now plates use `object-fit: contain` and `image-rendering: pixelated`. (This is a change based on reading the code and CSS, not yet compared with screenshots.)
- **Event panel**: every card says `run N`; by default only the events of the current run (one pass over the video) are shown, with a "show all runs" button; API `/api/events?scope=current|all`, `/api/status` has `run`. Tested.

## Update: the 5th version "current + old model @10 FPS" (2026-10-19)
`contact5_1..7.jpg` (in `outputs/regression_diag/`, gitignored) and `event_rows_5_versions.json`: the same 28 old rows (row numbers kept) plus a 5th column for the current pipeline + `helmet_best.pt` processed at 10 FPS (every 3rd frame, EXP-021); matched by time ±1 s + IoU; **3 new rows 29–31** (events of the 10 FPS version that match no old row: M_372 @14.2 s, M_620 @15.8 s, M_725 @16.6 s). `grading_template.csv` has an extra column for the 5th version (the "verdict" column is left empty for you to grade). Number of events: v0 22, current + old model 14, EXP-015 7, EXP-016 5, 10 FPS 10.

## Conclusion (confirmed by eye, 2026-10-20)
**Every no_helmet and overload event that the 5 versions report on test_video is a false alarm.** The project owner confirmed that all 23 people in the graded rows wear a helmet (`grading_rows.csv`: 23/23 rows "wrong"). The clip (26 s, morning rush hour) **has no violation among the events that were reported**.

**Precision is no longer a suitable measure for this clip** (every version is 0/n, so they cannot be compared). The measure used: **false alarms per minute** = number of events / clip duration (787 frames / 30 FPS = 26.233 s = 0.4372 min):
| version | false alarms (events) | false alarms / minute |
|---|---|---|
| v0 (original code) | 14 | 32.0 |
| current + old model | 14 | 32.0 |
| current + old model @10 FPS | 10 | 22.9 |
| EXP-015 (3 classes) | 7 | 16.0 |
| EXP-016 (4 classes) | 5 | 11.4 |
(Only no_helmet and overload; wrong_way is off. An event is a (track, type) pair: one person can produce several events if the track breaks.)

**Recall cannot be measured.** The clip has no recorded real violation, so it is not known how many the 5 versions miss; **there may still be people without a helmet that all 5 versions do not report**, which has not been checked (it needs a look at the whole clip or a video with real violations).

**Base rate:** when violations are rare, even a small probability of error on each person who wears a helmet means that false alarms still dominate real violations. Illustration (assumed, NOT measured): sensitivity 80%, the rate of mistaking a helmeted person for no-helmet is 1% / 2% / 5%; the precision of an alarm (the share of alarms that are real violations) by the share of people without a helmet among the vehicles passing:
| real no-helmet share | mistaken 1% | mistaken 2% | mistaken 5% |
|---|---|---|---|
| 23.6% (share of no-helmet among the head labels of the test set: 111 / (359 + 111)) | 96% | 93% | 83% |
| 5% | 81% | 68% | 46% |
| 1% | 45% | 29% | 14% |
**The mAP on the test set does not reflect this**: the test set of the dataset has no-helmet as 23.6% of the head labels (DATA_CARD), much higher than a real street, so the in-domain metrics (EXP-015: mAP50 0.934, no-helmet 0.844) give a higher alarm precision than in reality on the street. On this out-of-domain camera, the measured error rate is very large compared with the real violation rate (zero in what was observed).

Limits: one 26 s clip, the rules (window, voting, hysteresis) were once tuned on this very clip, one camera angle, rush hour; no parameter was tuned from this result. The table "Precision by eye" below is generated automatically by `tools/grade_rows.py`: every cell is 0 because every row is wrong, read the conclusion above.

<!-- grade_rows:start -->
## Precision by eye (`tools/grade_rows.py`)

Graded by the project owner by eye, 23 of 23 rows (no_helmet / overload; wrong_way rows skipped: that rule is off). The verdict of a row applies to every version that has an event in it, unless the row was marked as different people (then each version was graded on its own). Precision = right / (right + wrong), unclear left out.

| version | events graded | right | wrong | unclear | precision | relative coverage |
|---|---|---|---|---|---|---|
| v0 | 14 | 0 | 14 | 0 | 0.00 (0/14) | 0/0 |
| current+old model | 14 | 0 | 14 | 0 | 0.00 (0/14) | 0/0 |
| EXP-015 3cls | 7 | 0 | 7 | 0 | 0.00 (0/7) | 0/0 |
| EXP-016 4cls | 5 | 0 | 5 | 0 | 0.00 (0/5) | 0/0 |
| current+old model @10FPS | 10 | 0 | 10 | 0 | 0.00 (0/10) | 0/0 |

Relative coverage = among the 0 rows where at least one version was graded right, how many each version caught. **It is NOT recall**: a violation that all five versions missed is in nobody's denominator, so the true number of violations in the clip is unknown.

**Read with care:** n is small (every precision above rests on a handful of events), the whole evidence is ONE 26-second clip, and the rules (windows, votes, hysteresis, the ratios) were adjusted on this very clip, so the numbers say little about new footage. The versions differ in model AND settings (see the table above), so a difference between them is not attributed to one cause. The default model is not changed by this table.
<!-- grade_rows:end -->
