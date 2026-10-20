# models-v1 – helmet models of SentinelAI

Two YOLO11s helmet models trained on a **video-grouped, leak-free split** (split A, EXP-013). Weights are not in git; they are attached to this release.

| file | what | experiment | sha256 |
|---|---|---|---|
| `helmet_3cls_exp015.pt` | classes: helmet, motorcycle, no-helmet. **DEFAULT helmet model of the app since 2026-10-20** (`configs/pipeline/default.yaml`) | `experiments/EXP-015-helmet-grouped-3cls/` (weights: `.../weights/best.pt`) | `20ce0694eb1d8d81d6f579ede9b4d81c1aa1fbdf1d46e9391a63d04960fd7817` |
| `helmet_4cls_exp016.pt` | classes: helmet, motorcycle, no-helmet, plate. NOT the default: its in-domain no-helmet score is doubtful (waiting for EXP-016b) | `experiments/EXP-016-helmet-grouped-4cls/` (weights: `.../weights/best.pt`) | `6452d6354fbd09cd873b7201efd618238e56c6b9034fcf9cd894f73ca02bb19b` |

## Test mAP50 per class (in-domain test split A: 214 images from videos never seen in training; Ultralytics 8.4.26, Tesla T4)
| model | all | helmet | motorcycle | no-helmet | plate |
|---|---|---|---|---|---|
| EXP-015 (3 classes) | **0.934** | 0.962 | 0.995 | 0.844 | - |
| EXP-016 (4 classes) | 0.898 | 0.929 | 0.994 | 0.679 | 0.991 |

mAP50-95 (all classes): EXP-015 0.635, EXP-016 0.586 (the boxes of the dataset are loose, which caps mAP50-95; boxes were not re-labelled). Source: `metrics_3cls.json` / `metrics_4cls.json` in the experiment folders.
**These are in-domain numbers.** On an out-of-domain camera (test_video, 26 s) every no-helmet / overload alarm of both models was a false alarm (EXP-020); the test split has 23.6% no-helmet heads, far more than a street, so its scores overstate the precision of an alarm.

## Use
Download both files and `SHA256SUMS`, then `python tools/verify_models.py <folder>`. Put `helmet_3cls_exp015.pt` at `experiments/EXP-015-helmet-grouped-3cls/weights/best.pt` (the path in `default.yaml`), or point `models.helmet.weights` at any path.

## Licence and data attribution
- Code and weights: **AGPL-3.0** (Ultralytics YOLO is AGPL-3.0). Copyright (C) 2026 Trần Quang Duyên.
- **Training data (both models are derivative works of it):** "helmet plate detection 4" by **Rajab**, Roboflow Universe, 2024, https://universe.roboflow.com/rajab/helmet-plate-detection-4, licensed under **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/); checked on the Roboflow Universe page (1,460 images, 4 classes).
- **Changes:** the data was re-split by video (split A, EXP-013) for training and testing; EXP-015 (3 classes) drops the `plate` class, EXP-016 keeps all 4; the boxes were not relabelled. The original images are not redistributed here.
- The plate class of EXP-016 was trained on the plate labels of the same dataset. `plate_best.pt` (a separate detector) is not part of this release: it is not verified that it was trained on the CC BY plate dataset "vietnamese license plate" (school, 2023, https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0).
- The files contain only `/kaggle/working/...` training paths as metadata.

```bibtex
@misc{ helmet-plate-detection-4_dataset,
    title = { helmet plate detection 4 Dataset },
    type = { Open Source Dataset },
    author = { Rajab },
    howpublished = { \url{ https://universe.roboflow.com/rajab/helmet-plate-detection-4 } },
    url = { https://universe.roboflow.com/rajab/helmet-plate-detection-4 },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2024 },
}
```
