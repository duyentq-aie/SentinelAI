English translation of README.vi.md (original, Vietnamese).

# EXP-016 – retraining the helmet detector with 4 classes (helmet, motorcycle, no-helmet, plate) on split A

Run by: the project owner, on Kaggle (`results_4cls.zip`, sha256 `f1709e8f4a80c17f55d3f42165fcbcf6b05daea10a733ba701a1767914efd15a`; a byte-identical copy is at `<REPO>\results_4cls.zip` and `Downloads\results_4cls.zip`, the copy in the repo was used). Extracted here; `last.pt` skipped. The analysis was run by the assistant locally (inference only; no training, `data/raw` not modified).

| | |
|---|---|
| model | YOLO11s (`yolo11s.pt`), 4 classes, split A (EXP-013): train 1026 / val 214 / test 214 images (+ `extra_1080p` 6 images) |
| `weights/best.pt` | sha256 **`6452d6354fbd09cd873b7201efd618238e56c6b9034fcf9cd894f73ca02bb19b`**, 19 160 218 bytes; blocked by `.gitignore` (`*.pt`, and `results_*.zip`), checked with `git check-ignore` |
| environment | Kaggle Tesla T4, torch 2.11.0+cu128, Python 3.13.15, ultralytics 8.4.26 (same as EXP-015) |
| status in the app | **not used**; the app's `helmet_best.pt` is unchanged |

## Training run (`helmet_4cls_splitA/results.csv`)
- Stopped at **epoch 80** (80 rows), the best epoch (fitness 0.1·mAP50 + 0.9·mAP50-95) is **55**: stopped by **patience 25** (exactly 25 epochs after the best epoch), **not** by the time limit.
- Total training time **3 537 s (0.98 h)**, ≈ 43.7 s/epoch; the `time=2.5` h (9 000 s) limit was not reached.
- The first 4cls run hung (see `NOTES.md`).

## Results per class (`metrics_4cls.json`; mAP50 is the main metric for helmet/no-helmet)
**Test (214 images, 5 videos)**
| class | P (4cls) | R (4cls) | **mAP50 (4cls)** | mAP50-95 (4cls) | mAP50 (3cls, EXP-015) | mAP50-95 (3cls) | Δ mAP50 |
|---|---|---|---|---|---|---|---|
| helmet | 0.777 | 0.973 | **0.929** | 0.577 | 0.962 | 0.623 | -0.033 |
| motorcycle | 0.944 | 0.989 | **0.994** | 0.821 | 0.995 | 0.828 | -0.001 |
| no-helmet | 0.761 | 0.624 | **0.679** | 0.290 | 0.844 | 0.453 | -0.164 |
| plate | 0.947 | 0.982 | **0.991** | 0.658 | – | – | – |
| all (4 classes / 3 classes) | 0.857 | 0.892 | 0.898 | 0.586 | 0.933 | 0.635 | (not comparable: different number of classes) |

**Val (214 images, 4 videos)**
| class | P (4cls) | R (4cls) | **mAP50 (4cls)** | mAP50-95 (4cls) | mAP50 (3cls, EXP-015) | mAP50-95 (3cls) | Δ mAP50 |
|---|---|---|---|---|---|---|---|
| helmet | 0.819 | 0.970 | **0.951** | 0.597 | 0.950 | 0.579 | +0.001 |
| motorcycle | 0.975 | 0.996 | **0.993** | 0.707 | 0.995 | 0.745 | -0.002 |
| no-helmet | 0.972 | 0.742 | **0.911** | 0.540 | 0.913 | 0.529 | -0.002 |
| plate | 0.950 | 0.907 | **0.966** | 0.638 | – | – | – |
| all (4 classes / 3 classes) | 0.929 | 0.904 | 0.955 | 0.621 | 0.952 | 0.618 | (not comparable: different number of classes) |

**extra_1080p (6 images of 1920×1080, unknown source) – n = 6, NO conclusion**
| class | P (4cls) | R (4cls) | **mAP50 (4cls)** | mAP50-95 (4cls) | mAP50 (3cls, EXP-015) | mAP50-95 (3cls) | Δ mAP50 |
|---|---|---|---|---|---|---|---|
| helmet | 0.353 | 0.750 | **0.458** | 0.054 | 0.292 | 0.064 | +0.166 |
| motorcycle | 0.974 | 1.000 | **0.995** | 0.494 | 0.995 | 0.483 | +0.000 |
| no-helmet | 0.749 | 0.300 | **0.557** | 0.269 | 0.819 | 0.319 | -0.262 |
| plate | 0.383 | 0.417 | **0.231** | 0.050 | – | – | – |
| all (4 classes / 3 classes) | 0.615 | 0.617 | 0.560 | 0.217 | 0.702 | 0.289 | (not comparable: different number of classes) |

### Differences from EXP-015 (complete, from the `args.yaml` of both runs; every other parameter is identical)
| parameter | EXP-015 (3cls) | EXP-016 (4cls) | possible effect |
|---|---|---|---|
| classes | 3 | **4 (plate added)** | adds a small target, changes the label distribution and the loss |
| `fliplr` | 0.5 | **0.0** | removes one augmentation (horizontal flip) for every class |
| `time` | not set | **2.5** | Ultralytics recomputes the number of epochs, the cosine schedule is ≈ **178 epochs** long (inferred from `lr/pg0`) instead of 150 |
| `workers` | 8 | **2** | the order and randomisation of data loading may differ (the algorithm is unchanged) |
Stop: patience 25 at epoch 80 (EXP-015: 83).

**Important note:** no-helmet on **val is almost equal** (mAP50 0.911 versus 0.913); the large difference appears only on **test** (0.679 versus 0.844). With one run per configuration and 5 test videos, **the real effect of the four changes cannot yet be told apart from the variation between runs** (seed, data order, learning schedule). **Plan: EXP-016b** = 4cls with `fliplr=0.5` and `time` not set (the notebook `kaggle/helmet_train_split_a.ipynb` was edited accordingly; `workers=2` is kept), to separate the plate-class change from the other three changes; repeat with another seed if the run-to-run variation needs to be estimated.

### Reading it correctly
- **This run changed many things at once, so the difference from EXP-015 cannot be attributed to any single factor.** You named 2 (the added plate class, `fliplr: 0.0`); comparing the `args.yaml` of the two runs shows **2 more differences**: `workers: 2` (EXP-015: 8) and `time: 2.5` (EXP-015: not set). When `time` is set, Ultralytics **recomputes the number of epochs from the time per epoch** and rescales the learning schedule: from the `lr/pg0` column the cosine length is ≈ **178 epochs** in EXP-016, versus exactly 150 in EXP-015 (an estimate from lr, assuming lr0 = 0.001429 as in EXP-000). So the learning-rate schedule also differs. The remaining parameters are identical (`diff` of args.yaml).
- no-helmet on test: **mAP50 0.679 (4cls) versus 0.844 (3cls)**, R 0.624 versus 0.744; helmet 0.929 versus 0.962; motorcycle 0.994 versus 0.995. Test confusion matrix (`eval_4cls_test/confusion_matrix.png`): of the 133 real no-helmet, 4cls finds **71** correctly, confuses **31** with helmet and misses **31**; EXP-015: 105 correct, 21 confused, 7 missed. The two runs differ on the no-helmet class by more than we can explain; with 5 test videos, the variance between runs (seed, data order, learning schedule) **has not been measured** and may be part of it.
- Val: no-helmet mAP50 0.911 (4cls) versus 0.913 (3cls): almost equal, so the large difference appears only on test.
- No claim is made about accuracy on real video.

## Plate class: 4cls versus `plate_best.pt` (purpose: judge whether the models can be merged, per G9)
On the **271 plate boxes** of the split A test (214 images), with the same evaluation function (`tools/eval_ood.py`: COCO-style AP, P/R at conf 0.25 and IoU 0.5; the AP numbers are **not** directly comparable with the Ultralytics mAP); `scripts/plate_compare.py`, `plate_vs_plate_best*.json`.

| protocol | model | P | R | AP50 | AP50-95 |
|---|---|---|---|---|---|
| whole frame (the role of the 4-class model) | 4cls, plate class | 0.964 | 0.982 | **0.988** | 0.647 |
| whole frame | `plate_best.pt` | 0.048 | 0.004 | 0.007 | 0.001 |
| image cropped to the GT motorcycle box (the role of `plate_best.pt` in the pipeline; 220/271 plate boxes lie inside a motorcycle box) | 4cls, plate class | 0.000 | 0.000 | 0.000 | 0.000 |
| image cropped to the GT motorcycle box | `plate_best.pt` | 0.076 | 0.045 | 0.016 | 0.002 |
- 4cls finds plates on the whole frame in this domain (P 0.964, R 0.982); this **supports** the possibility of merging the plate into a one-pass model **within the image domain of the helmet set** (seen from above, small plates, the same group of videos as train).
- `plate_best.pt` detects almost **nothing** on the same images (R 0.004 whole frame, 0.045 when cropped to the vehicle). **This must not be read as "plate_best.pt is bad"**: it is used and (per EXP-014) seems to be trained on 640×640 images of a fixed parking-gate camera, where plates are very large; on top-down images with small plates it is not an object of the same domain. **The origin of `plate_best.pt` is unclear** (EXP-000: the notebook does not match the checkpoint), so it cannot be excluded that it has seen similar images; this comparison only shows "which model is usable in which domain", not "which model is better".
- The cropped-image protocol breaks 4cls (AP 0): the 4-class model learned at the whole-frame scale; cropping and enlarging to 640 is the wrong scale. This says that if merged, the plate must be detected **on the whole frame** (the large image), and only then matched to a vehicle; the quality when the plate is far/small (test video: ~26×22 px at 720p) was not measured here.
- Not measured: 4cls on `test_video.mp4` for the plate role (there is no plate GT there).

## Files
`metrics_4cls.json`, `helmet_4cls_splitA/` (args, results.csv, curves, confusion matrix; the batch images were dropped because people are visible), `eval_4cls_{val,test,extra_1080p}/`, `weights/best.pt` (not committed), `plate_vs_plate_best*.json`, `scripts/`, `NOTES.md`.
