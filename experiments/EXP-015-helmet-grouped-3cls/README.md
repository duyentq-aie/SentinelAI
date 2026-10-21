English translation of README.vi.md (original, Vietnamese).

# EXP-015 – retraining the helmet detector (3 classes) on the video-grouped split (split A)

Run by: the project owner, on Kaggle (`results_3cls.zip`, extracted here; `last.pt` skipped on request). The analysis below was run by the assistant on the local machine (inference only, no training, `data/raw` not modified).

| | |
|---|---|
| model / recipe | YOLO11s from `yolo11s.pt`, exactly the `train_args` of `helmet_best.pt` (150 epochs, patience 25, batch 32, imgsz 640, seed 0, AdamW via `optimizer: auto`); `helmet_3cls_splitA/args.yaml` |
| data | split A (`experiments/EXP-013-splits/A`, manifest sha256 `fe103d4e…2279`): train 1026 / val 214 / test 214 images (+ `extra_1080p` 6 images), 3 classes (`plate` filtered out) |
| environment | Kaggle Tesla T4, torch 2.11.0+cu128, Python 3.13.15, ultralytics 8.4.26 (old numbers: torch 2.9.0, Python 3.12.12, same ultralytics) |
| finish | early stop at epoch 83 (patience 25), best at epoch 58; 2936 s |
| `weights/best.pt` | sha256 **`20ce0694eb1d8d81d6f579ede9b4d81c1aa1fbdf1d46e9391a63d04960fd7817`**, 19 159 770 bytes. Blocked by `.gitignore` (`*.pt`, checked with `git check-ignore`). `last.pt` was not extracted. |
| `results_3cls.zip` | sha256 `0efd2ae46460f5e91036645b4930aade987c20c9e9bd8e204734d6d9da735ebf` |
| status in the app | **not used**: the app's `helmet_best.pt` was not touched; changing the pipeline's model is a separate step waiting for your approval. |

## Results per class (`metrics_3cls.json`, default `val` conf)
Column "old" = `helmet_best.pt`, mAP50-95 on the **old test** (EXP-000): helmet 0.791 / motorcycle 0.878 / no-helmet 0.768 (overall 0.812).

**Test (214 images, 5 videos: frame_4, 6, 9, 13, 16)** – P / R / mAP50 / mAP50-95

| class | instances | P | R | mAP50 | mAP50-95 | mAP50-95 old | difference |
|---|---|---|---|---|---|---|---|
| helmet | 258 | 0.866 | 0.985 | 0.962 | 0.623 | 0.791 | −0.168 |
| motorcycle | 304 | 0.999 | 0.990 | 0.995 | 0.828 | 0.878 | −0.050 |
| no-helmet | 133 | 0.833 | 0.744 | 0.844 | 0.453 | 0.768 | −0.315 |
| **all** | 695 | 0.900 | 0.906 | 0.934 | **0.635** | 0.812 | −0.177 |

**Val (214 images, 4 videos: frame_10, 11, 14, 15)** – old (old val, 278 images): helmet 0.793 / motorcycle 0.882 / no-helmet 0.751 / overall 0.809

| class | instances | P | R | mAP50 | mAP50-95 | mAP50-95 old | difference |
|---|---|---|---|---|---|---|---|
| helmet | 468 | 0.772 | 0.954 | 0.950 | 0.579 | 0.793 | −0.214 |
| motorcycle | 471 | 0.991 | 0.981 | 0.995 | 0.745 | 0.882 | −0.137 |
| no-helmet | 140 | 0.913 | 0.771 | 0.913 | 0.529 | 0.751 | −0.222 |
| **all** | 1079 | 0.892 | 0.902 | 0.952 | 0.618 | 0.809 | −0.191 |

**extra_1080p (6 images of 1920×1080, unknown source) – n = 6, NO conclusion** (helmet has 4 instances, no-helmet 10)

| class | instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| helmet | 4 | 0.358 | 0.250 | 0.292 | 0.064 |
| motorcycle | 14 | 0.993 | 1.000 | 0.995 | 0.483 |
| no-helmet | 10 | 0.985 | 0.600 | 0.819 | 0.319 |
| all | 28 | 0.778 | 0.617 | 0.702 | 0.289 |

### Reading the numbers correctly
- **The old numbers cannot be compared directly with the new ones.** The old numbers were measured on a *random per-frame split with leakage* (EXP-011: ~86% of the test images have a train image within ≤ 4 dHash bits), on *a different set of test images* (168 images, frames mixed with train). The new numbers are measured on videos from which the model has seen no frame, on *a different set of images*, with a *different training environment* (torch 2.11 vs 2.9, Python 3.13 vs 3.12). So the drop (−0.18 overall, −0.32 no-helmet on test) is **not entirely due to leakage**: part of it is because the new test set is harder/different (see below: the test videos have at most 1 no-helmet per vehicle, people wearing head scarves…), part is variance from having only 4–5 videos, and it cannot be separated with this experiment (it would need retraining 3 classes on the old random split in the same environment, or running `helmet_best.pt` on test A – the old one would be leaky because `helmet_best.pt` saw the test-A frames during training).
- The new numbers are the first more trustworthy measurement of **generalisation across videos**, but still only on 5 test videos of the same camera/viewpoint (from above); each video contributes many consecutive frames so the effective number of independent samples is much smaller than 214 images. There is no confidence interval here.
- **There is no claim about accuracy on real video** or about violation events; this is only the mAP of the detector on the still images of split A.

## Checking label hypotheses (counted on the original 4-class labels, nothing changed)
`label_stats.json` (per image), `label_stats_per_motorcycle.json` (heads assigned to a motorcycle by the box centre lying inside the vehicle box).

| | images | no-helmet / image: 0, 1, 2 | images with ≥ 2 no-helmet | motorcycles with ≥ 2 no-helmet | heads (helmet + no-helmet) / image |
|---|---|---|---|---|---|
| original split – train / valid / test | 1014 / 278 / 168 | train 458, 448, 108; valid 124, 138, 16; test 68, 89, 11 | 108 / 16 / 11 | – | at most 8 / 6 / 7 |
| **split A – train** | 1026 | 495, 400, 131 | **131** | **95** (of 2102 vehicles) | 0–8, peak at 2 (304 images) |
| **split A – val** | 214 | 74, 140, **0** | **0** | **0** | 1–6 |
| **split A – test** | 214 | 81, 133, **0** | **0** | **0** | 1–4 (peak at 1: 96 images) |
| extra_1080p | 6 | 0, 2, 4 | 4 | – | 2–4 |

- **Are there images with ≥ 2 no-helmet?** Yes, but only in train (131 images; 95 vehicles have 2 no-helmet) and in the original split (108/16/11). **No image has ≥ 3.** In **val and test of split A no image has ≥ 2 no-helmet**, and no vehicle has 2 people without a helmet: the evaluation sets completely lack the case "two people without a helmet on one vehicle", although train has 95 such vehicles → the no-helmet mAP on val/test does not reflect this case.
- Number of heads per vehicle (split A, train): 0 heads: 53 vehicles, 1 head: 1493, 2 heads: 538, 3 heads: 18 (motorcycles carrying 3 people ≈ 0.9%). Val: 21 / 332 / 104 / 14 (14 vehicles with 3 heads), test: 0 / 227 / 77 / 0 (no vehicle with 3 heads ⇒ the `overload` rule has nothing to measure on test).
- Vehicles with both a helmeted head and a head without a helmet: train 245, val 108, test 77 – the case "helmet + no helmet on the same vehicle" is the main one, and it is where the head boxes overlap.
- The "head" boxes are inconsistent in size: many no-helmet boxes (and a few helmet boxes) in val cover both the head and the back/shoulders (see the gallery), while other helmet boxes cover only the helmet ⇒ low IoU, low mAP50-95 (0.45–0.53 for no-helmet) although mAP50 is high (0.84–0.91). This is an observation from the gallery, not yet measured statistically.

## Error gallery on val (`error_gallery/`, conf = 0.25, imgsz 640, `best.pt`)
Matched by IoU ≥ 0.5 between GT and the predictions of the head classes (helmet/no-helmet); `val_error_cases.json|csv` has every case (id `E00`…`E50`, image, conf, IoU, the assistant's label + reason). The gallery images are **not committed** (people are visible; `error_gallery/` is in `.gitignore`).

| group | cases | model error | unsure | missing label |
|---|---|---|---|---|
| (a) GT no-helmet predicted as helmet | 35 (34 + 1 case with IoU 0.3–0.5) | 19 | 16 | 0 |
| (b) GT no-helmet missed completely | 5 | 2 | 3 | 0 |
| (c) predicted no-helmet matching no GT no-helmet | 11 | 9 | 1 | 1 |
| **total** | **51** (in 46 images) | **30** | **20** | **1** |

GT no-helmet on val = 140; 40 of them (29%) are not predicted correctly at conf 0.25 (consistent with recall 0.771).
**The classification above is the assistant's, after looking at every image tile – you review/correct it.** Definitions: *model error* = the GT label is reasonable, the image is clear, the model is wrong; *unsure* = the image is ambiguous or there is no convention yet (head scarf/hijab, cloth cap/hood, two heads overlapping in the same box); *missing label* = an object is clearly visible without a GT box.
- **The errors are concentrated by video, not spread evenly:** (a) frame_14 19 cases + frame_10 7 + frame_15 8(+1); (b) frame_14 2, frame_15 3; (c) frame_15 5, frame_10 4, frame_14 2. In frame_14, 19 cases (a) + 2 cases (b) are *the same person sitting in front, bare-headed, over many consecutive frames* – really **one subject**, so the number of cases is much larger than the number of independent errors. The (a)/(b) cases of frame_15 are **one person wearing a head scarf** over many consecutive frames.
- (a) frame_14 (19, *model error*): bare head clearly visible, the GT is right; the model predicts helmet because the rider's helmet lies overlapping inside the GT box (the tall GT box includes the back).
- (a) frame_10 (7, *unsure*): a person wearing a cloth cap; the helmet of the second person overlaps inside the box. frame_15 (8 + 1, *unsure*): head scarf, no convention yet (no helmet ≠ no head covering).
- (c): 7 no-helmet predictions placed on the **torso/back** of a person right below a head that already has a label (often a helmeted head), 2 on a **car tail light** (E07, E08) (9 cases of *model error*), 1 missing-label case (E00: a person clearly wearing a helmet but no GT helmet box), 1 localisation case (E49: the long GT box includes the scarf + back).
- Suggestions (not done): agree on a convention for the "head" box (head only or including shoulders), decide the label for head scarves, add images with ≥ 2 people without a helmet to val/test, add car images as background images.

## Files
`metrics_3cls.json`, `helmet_3cls_splitA/` (args, results.csv, curves, batch images, empty `weights/`), `eval_3cls_{val,test,extra_1080p}/` (confusion matrix, PR…), `weights/best.pt` (not committed), `label_stats*.json`, `val_error_cases.{json,csv}`, `error_gallery/` (not committed), `scripts/` (the analysis scripts that were run).


## Update 2026-10-19

### Three conclusions to remember
1. **The hypothesis "every image has only 1 no-helmet" is rejected for the training data.** Split A train has **131 / 1026 images with exactly 2 no-helmet** (12.8%) and **95 vehicles with 2 people without a helmet**; original split: 108/1014, 16/278, 11/168 images. No image has ≥ 3. *But* it is true for val and test of split A (0 images with ≥ 2, 0 vehicles with 2 people without a helmet): that is a gap of the evaluation set, not of the labels in general. These numbers do not prove the cause of the low no-helmet recall (0.74–0.77).
2. **The errors are concentrated in frame_14.** 23 / 51 error cases of val (19 (a) + 2 (b) + 2 (c)) are in frame_14 (one bare-headed person sitting in front over many consecutive frames, tall GT box). In val, 24/34 of the no-helmet boxes in frame_14 are marked "covers the torso" (see below).
3. **The test has no vehicle with 3 people** (0 vehicles with 3 heads; val has 14, train 18): the `overload` rule is **not evaluated** on this split; the mAP says nothing about it.

### Box consistency (`tools/box_consistency.py`, `box_consistency/`)
Measured on all 4 classes of split A. Height/width in pixels (image size from the manifest); relative height = head box height / height of the motorcycle box that contains the centre of that box (94–98% of the head boxes could be matched, the rest lie in no vehicle box). `boxes.csv` = every box; `hist_aspect.png`, `hist_rel_h.png` = histograms per class/split; `summary.json` = percentiles.
- **The height/width distribution has no two clusters** (a single continuous peak, `valley_depth_ratio` = 1.0). Median helmet 1.32 / 1.39 / 1.36 (train/val/test), no-helmet 1.36 / **1.47** / 1.41; motorcycle ≈ 1.32, plate ≈ 0.59. A head box seen from above is nearly square if it covers only the head, so a median of 1.3–1.4 means most boxes are already longer than the head region.
- **The relative height has one main peak + a tail:** the peak at ≈ 0.2–0.25 of the vehicle height (median helmet 0.23, no-helmet 0.22 – almost equal), the tail extends to 0.6; Otsu gives the valley at **0.303**.
- **Proposed thresholds (based on the distribution, not checked with people):** height/width > **1.514** (the 90th percentile of helmet + no-helmet combined; since there is no valley this is an "upper tail", not a cluster boundary) **or** relative height > **0.303** (Otsu valley) ⇒ flag "box covers the torso".

| class | split | boxes | flagged (%) | due to height/width | due to relative height | not matched to a vehicle |
|---|---|---|---|---|---|---|
| helmet | train | 2105 | 404 (19.2%) | 174 | 281 | 120 |
| helmet | val | 468 | 90 (19.2%) | 57 | 44 | 23 |
| helmet | test | 258 | 33 (12.8%) | 15 | 21 | 8 |
| helmet | extra_1080p | 4 | 4 (100.0%) | 0 | 4 | 0 |
| no-helmet | train | 662 | 117 (17.7%) | 64 | 68 | 24 |
| no-helmet | val | 140 | 63 (45.0%) | 55 | 13 | 3 |
| no-helmet | test | 133 | 24 (18.0%) | 17 | 7 | 2 |
| no-helmet | extra_1080p | 10 | 3 (30.0%) | 0 | 3 | 0 |

- no-helmet **val: 45.0%** flagged versus 17.7% (train) and 18.0% (test); by val video: frame_14 24/34, frame_10 14/30, frame_15 25/76. Median height/width on val 1.47. The difference shows that the "covers the torso" level is uneven between videos and agrees with the val errors piling up in frame_14. Helmet 12.8–19.2%.
- Hand-reviewed gallery (30 random boxes with seed 0 per group: helmet/no-helmet × flagged/not flagged): `box_consistency/gallery/*.jpg` (**not committed**, people are visible). Observation by eye on 120 boxes: the flagged group often covers down to the waist/backpack; the *not* flagged group of no-helmet also **mostly covers the shoulders** (not only the head); the unflagged helmet mostly covers the helmet and a little of the shoulders.
- **Conclusion (observation + measurement):** the boxes of both classes vary continuously from "almost only the head" to "head + shoulders + back", no-helmet is a little taller than helmet and deviates a lot in val (frame_14). There is no clear cluster so the thresholds above are only a review tool. **Relabelling is not proposed** at this step: it is not known how much of the low mAP50-95 is due to the boxes (versus the model) – a measurement is needed (for example re-evaluating with a looser IoU or on a set with normalised boxes) before deciding. The new rule for head labels: `docs/LABELING_GUIDE.md`.


### Decision 2026-10-19: no relabelling; the main metric for helmet / no-helmet is mAP50
*This decision was made after looking at the data* (the measurements and the gallery above), not a plan made beforehand; it is not pre-registered.
- **The existing boxes are not relabelled.** Reasons: (1) the shape and size of the boxes change **continuously**, single peak, no clear cluster (height/width: `valley_depth_ratio` 1.0; relative height: one peak + tail), so there is no natural boundary to say which box is "wrong"; (2) cost: 3 780 head boxes (2 835 helmet + 945 no-helmet) in split A, i.e. about 3 000–3 800 boxes to look at and redraw; (3) it is not known how much of the low mAP50-95 is due to loose boxes. The rule for **new** labels: `docs/LABELING_GUIDE.md` (head box = the head region, up to the shoulders allowed; covering the waist/backpack is wrong; old data is not corrected retroactively).
- **The main metric for `helmet` / `no-helmet` = mAP50** (test: helmet 0.962, no-helmet 0.844, all 0.934; val: 0.950 / 0.913 / 0.952). **mAP50-95 is still reported** (test 0.623 / 0.453 / 0.635) **with the note: it is limited by the looseness/inconsistency of the boxes** (a high IoU demands tightly fitting boxes, but the labels do not fit tightly), so the mAP50-95 of these two classes is lower than the real detection quality and is not used to compare models class by class. `motorcycle` keeps both metrics.
- There is no number yet showing that normalising the boxes would change the conclusion; if it is needed later, the starting point is to re-evaluate with a looser IoU or on a subset of reviewed boxes.
