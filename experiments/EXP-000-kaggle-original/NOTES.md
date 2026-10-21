English translation of NOTES.vi.md (original, Vietnamese).

# EXP-000 – Original training on Kaggle (history)

Source: the two notebooks in `notebooks/` (the saved outputs were read) compared with the metadata embedded in `helmet_best.pt` / `plate_best.pt`
(`train_args`, `train_metrics`, `train_results`). Full numbers: `metrics.json`; parameter comparison table: `notebook_vs_checkpoint.json`.

## Helmet – the notebook MATCHES the checkpoint ✅
- All 108 train parameters of the checkpoint equal the `engine/trainer:` line in the notebook log, except 3 harmless differences:
  `device` (0 vs "0", only the type differs), `save_dir` (not stored in the checkpoint), `warmup_bias_lr` (0.1 in the log, 0.0 in the checkpoint — the value was changed by the warm-up schedule).
- The final result on val matches the checkpoint (P 0.983, R 0.989, mAP50 0.994, mAP50-95 0.809); the time 1.453 h matches 5229 s.
- Training environment: Kaggle, **Tesla T4 (14.9 GB)**, Python 3.12.12, torch 2.9.0+cu126, Ultralytics 8.4.26, YOLO11s (9,428,953 parameters, 21.6 GFLOPs), AdamW chosen automatically (lr 0.001429), 150 epochs, AMP.
- **Data split** (from the label-scan log): train **1014** images (2 background images), val **278** images (1 background, 1384 instances), test **168** images (2 background, 818 instances). 1460 images in total.
- **Val per class (best.pt):**

| class | images | instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| all | 278 | 1384 | 0.983 | 0.989 | 0.994 | 0.809 |
| helmet | 262 | 606 | 0.987 | 0.994 | 0.993 | 0.793 |
| motorcycle | 276 | 608 | 0.985 | 0.995 | 0.995 | 0.882 |
| no-helmet | 154 | 170 | 0.977 | 0.976 | 0.993 | 0.751 |

- **Test per class (best.pt, `split='test'`):**

| class | images | instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| all | 168 | 818 | 0.987 | 0.996 | 0.994 | 0.812 |
| helmet | 164 | 359 | 0.980 | 1.000 | 0.993 | 0.791 |
| motorcycle | 166 | 348 | 0.991 | 0.989 | 0.995 | 0.878 |
| no-helmet | 100 | 111 | 0.991 | 0.998 | 0.995 | 0.768 |

- Speed reported in the notebook: 2.9 ms inference/image (val) on the T4 — **not** comparable with the pipeline FPS of EXP-001 (different hardware, model only, no tracker/drawing).
- An inconsistency in the notebook: the markdown in section 2 still says `yolo11n` and `epochs=50` while the code and the checkpoint are `yolo11s` / 150 epochs (the old caption was not corrected).

### Correction to the AUDIT
The AUDIT (sections 2.B/2.D) once said "no evidence about a test set". **Partly wrong**: the helmet dataset **does** have a test split (168 images) and the notebook evaluates on it; that result was simply not stored in the repo until now.

### How to read these numbers (still a warning)
- The test (0.994 / 0.812) almost **equals** val (0.994 / 0.809): mAP50 is saturated (0.993–0.995 for all 3 classes) → this scale no longer distinguishes quality.
- The sample test image name in the notebook (`frame_10_579_jpg`) shows that the images are **frames extracted from video**. If the splits were made per frame instead of per video, adjacent frames can be in both train and test → **the independence of the test has not been verified** (that check belongs to P1). This is a reasoned guess, not yet a conclusion.
- Imbalance: `no-helmet` has only 170 instances on val (28% of helmet/motorcycle); it is also the class with the lowest mAP50-95 (0.751 val / 0.768 test) and the most important class for the violation rules.

## Plate – the notebook does NOT match the checkpoint ❌
This confirms your suspicion. 10 of the 19 `train()` parameters written explicitly in `license-plate-detector.ipynb` differ from `plate_best.pt`:

| parameter | notebook | `plate_best.pt` |
|---|---|---|
| model | yolo11**s** | yolo11**n** |
| epochs | 150 | 50 |
| patience | 30 | 100 |
| name | plate_model_pro | license_plate_model |
| degrees | 15.0 | 0.0 |
| perspective | 0.001 | 0.0 |
| shear | 2.0 | 0.0 |
| scale | 0.6 | 0.5 |
| fliplr | 0.0 | 0.5 |
| cos_lr | True | False |

The same: data (`vietnam-license-plate-v1/data.yaml`), imgsz 640, batch 32, project, mosaic 1.0, hsv_h/s/v, optimizer auto.
- The notebook has **no saved output** → no metrics, no number of images/classes for plate.
- Interpretation: `plate_best.pt` (2026-03-23, Ultralytics 8.4.25) comes from an earlier run with the default configuration that the notebook no longer has;
  the notebook that is stored now is the later "pro" configuration (yolo11s, plate-specific augmentation, `fliplr=0` which is sensible for plates) but **the weights of that configuration are not the file being deployed**.
- Consequences: (1) the only plate numbers that can be trusted remain the metadata of `plate_best.pt` itself (val mAP50 0.995 / mAP50-95 0.752, combined, no test);
  (2) the plate retraining in P5 should use exactly the notebook configuration and be treated as a new model (not a reproduction);
  (3) `fliplr=0.5` in the checkpoint is training the plate model on mirrored plate images — this may affect OCR quality later.
