English translation of NOTES.vi.md (original, Vietnamese).

# EXP-008 – Plate OCR, basic version (night of 2026-10-18, section 6)

**No accuracy claim** (there is no plate ground truth). The practical conclusion of the experiment: *the OCR chain works mechanically, but on the project's 720p video the plates are too small to read*.

## Processing chain (`sentinel/ocr/`)
1. **Pick the best crop of a track** (`quality.py`): score = 0.5·size + 0.5·sharpness (Laplacian variance), each component is capped (4× / 3× the reference value). The plate detector keeps running until `ocr.candidates` (5) crops at least `min_gap` (3) frames apart are collected, or `max_attempts` is reached, or the violation ends. `plate.jpg` is overwritten with the best crop.
2. **OpenCV preprocessing** (`preprocess.py`): crop tight to the plate box (`ocr.margin` = 2 px, instead of 10 px of background), enlarge bicubic to 128 px high (at most ×6), CLAHE on the L channel, **split a two-line plate** (motorcycle) at the row with the least "ink" around the middle.
3. **Recognition** (`engine.py`): PaddleOCR 3.4 `TextRecognition(PP-OCRv5_mobile_rec)` (recognition only, no detection), line by line, then joined. The model is downloaded from Hugging Face on first use (cache outside the repo: `~/.paddlex/`).
4. **Normalisation + Vietnamese plate format** (`normalize.py`): remove separator characters, fix characters that are easy to confuse by position (O→0, B↔8, S↔5, …) with a pattern of 8–10 characters; the series letter is only taken from `ABCDEFGHKLMNPSTUVXYZ` (Vietnamese plates do not use I, J, O, Q, R, W); regex `^\d{2}[letters]{1,2}\d?\d{4,5}$`. It only checks the shape of the string, not whether the plate exists or was read correctly.
5. **Vote** (`vote.py`): read the `reads` (3) best crops; the most supported length wins, then a per-character vote weighted by confidence.
6. The result is written to `events.jsonl` in `plate_pred` (`docs/EVIDENCE.md`); PaddleOCR errors are contained (the event stays complete).
Enable with `configs/pipeline/ocr.yaml` (off by default).

Tests: unit tests for each block (`tests/test_ocr_pure.py`: crop quality, preprocessing, normalisation/character fixing, vote, PlateOCR) and 9 tests for the integration in the recorder (`tests/test_evidence_ocr.py`) with a fake recogniser (number of crops collected, best crop replaces `plate.jpg`, OCR runs once per track, an error does not crash the pipeline, reset). These tests do **not** use the real PaddleOCR.

## Result on the test video (CPU, `record_tracks --config configs/pipeline/ocr.yaml`)
- Events identical to baseline v3 (empty diff): turning OCR on changes neither the rules nor the events.
- **14 events, 14 have `plate_pred`; 0 have a string in the plate format.** Size of the stored plate crop: 38–51 × 36–49 px (mean 46×44, **including a 10 px border of background**; the plate box returned by the detector measures ≈ 26×22 px (EXP-010 `mean_plate_box_wh`); 46×44 is the size of the stored crop, not of the plate).
- The raw recogniser output is fragments or empty (`''`, `'0'`, `'M'`, `'BD'`, some Chinese characters). When the crop is 64 px high with a 10 px border, the PP-OCRv5 model mostly just "invents" Chinese characters (first run: 0/14 valid); after cropping tight to the plate and enlarging to 128 px, part of the output turns into Latin letters/digits (a few Chinese characters remain) but does not form a valid plate (0/14).
  (A few preprocessing variants were tried on the 14 saved crops with a temporary script, not committed; parameters were **not** chosen to "increase the number of valid strings" because there is no ground truth, that would be tuning on noise. Two changes were kept — a tight crop and a larger enlargement — for reasons of principle: fewer background pixels and characters large enough for the recogniser.)
- **OCR latency / event** (3 reads, CPU, PaddleOCR mobile): mean **209 ms**, median 220 ms, min 68, max 377 ms. It runs in the inference thread (blocking the pipeline ~0.2 s per event). Model loading: ~11 s the first time (download) / a few seconds when cached.
- Examples of "could not read": all 14 events of this clip. Montage of the crops: `outputs/ocr_examples/plate_crops_montage.jpg` (the `outputs/` folder is gitignored because it holds real plates).

## Probe with synthetic plates (`tools/ocr_resolution_probe.py`, `resolution_probe.json`)
A two-line plate "29B1 / 12345" drawn with OpenCV, shrunk and degraded (Gaussian blur, JPEG q25), run through the whole OCR chain. **This is not accuracy on real images** (one font, bold strokes, clean):

| height (px) | clean | blur 0.8 | blur 1.2 | JPEG 25 | blur 0.8 + JPEG 25 |
|---|---|---|---|---|---|
| 100 | correct | correct | correct | correct | correct |
| 50 | correct | correct | correct (after fixing 8→B) | correct | correct (after fixing 8→B) |
| 38 | correct | correct | correct (after fixing) | correct (after fixing) | **wrong** (20B112345) |
| 25 | correct | **wrong** | **invalid** | **wrong** | **wrong** |

With a clean plate the chain (preprocessing + PaddleOCR + character fixing + vote) reads correctly; at ~25 px high, blur 0.8 or JPEG alone is enough to break it. The real plates in this clip are small, blurred, JPEG-compressed and tilted, so failing to read them is reasonable.

## Conclusion and what to do
1. **With a 720p camera at this distance OCR is not feasible.** 1080p+ video or a closer camera is needed (plate ≥ ~40–50 px high) — recorded in `docs/RECORDING_GUIDE.md`. The OCR model/post-processing should not be changed before there is data of sufficient resolution and plate labels (`label_events.py` already has the `plate_text` field).
2. When labels exist: measure exact-match and CER on GT crops and end-to-end (PLAN section 3.4), separating day/night, 1/2 lines.
3. The latency of ~0.2 s/event suggests moving OCR to its own thread if it is turned on for a demo.
