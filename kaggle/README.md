# Kaggle re-training of the helmet detector (PREPARED, NOT RUN)

Status 2026-10-18: no Kaggle token on this machine (`python tools/kaggle_train.py check` -> False) and the `kaggle` CLI is not installed, so nothing was uploaded or launched.

Recipe: YOLO11s from `yolo11s.pt`, the exact `train_args` of `helmet_best.pt` (`configs/training/helmet_yolo11s_v1.json`, ultralytics 8.4.26: 150 epochs, patience 25, batch 32, imgsz 640, seed 0, deterministic, AdamW via `optimizer: auto`, cos_lr, mixup 0.1, ...), 3 classes (helmet, motorcycle, no-helmet; `plate` dropped, as in the old training copy), on the split chosen from `experiments/EXP-013-splits/`.

Steps once a split (A/B/C) is chosen and a token exists:
1. `python tools/kaggle_train.py prepare --data data/raw/helmet --split-dir experiments/EXP-013-splits/A --out outputs/kaggle/helmet_split_A --slug sentinel-helmet-split-a`
2. `python tools/kaggle_train.py push-dataset --dataset-dir outputs/kaggle/helmet_split_A --user <kaggle-user> --confirm`  (creates a private dataset)
3. `python tools/kaggle_train.py push-kernel --kernel-dir kaggle/helmet_train_a --user <kaggle-user> --confirm`  (private, GPU; edit `SMOKE = True` in the notebook first for a 2-epoch plumbing check)
4. Download `helmet_v2_best.pt`, `result.json`, `runs/` and record them in a new EXP folder. Do NOT overwrite `helmet_best.pt`.

The user name is `<KAGGLE_USER>` in the committed files; `push-*` fill it into a copy under `outputs/kaggle/_push/`.
Expect different numbers from EXP-000: the old val/test leaked (random frame split), so the new test mAP should be lower and is the first honest one.
