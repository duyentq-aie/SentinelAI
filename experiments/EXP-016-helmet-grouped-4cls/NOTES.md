English translation of NOTES.vi.md (original, Vietnamese).

# NOTES – EXP-016 (4cls)

## The first run hung
The first 4cls run on Kaggle **hung for more than 3.9 hours with 0 B of output** and was stopped by hand. No log or result of that run was saved; the cause is unknown (suspected data-loading congestion with `workers: 8` on the Kaggle machine, **not verified**). The rerun (the results here) set `workers: 2` and `time: 2.5`, both to avoid the congestion and to cap the time.

## Configuration of this run
The `zip` contains no training log, only `args.yaml` and `results.csv`, so the configuration below is read from `helmet_4cls_splitA/args.yaml`:
- model `yolo11s.pt`, split A data with 4 classes, `epochs: 150`, `patience: 25`, `batch: 32`, `imgsz: 640`, `seed: 0`, `deterministic: true`, optimizer `auto`, `cos_lr: true`, `close_mosaic: 10`, `mosaic: 1.0`, `mixup: 0.1`, `degrees: 10.0`;
- **different from EXP-015:** `fliplr: 0.0` (EXP-015: 0.5), `workers: 2` (8), `time: 2.5` (not set), `name: helmet_4cls_splitA`; 4 classes instead of 3.
- Consequence of `time`: Ultralytics recomputes the number of epochs from the time per epoch, so the cosine schedule is ≈ 178 epochs long (inferred from `lr/pg0`) instead of 150; the run stopped early at epoch 80 by patience, before the schedule ended.
- The notebook used for this run is **not in the repo** (it was not sent), so it could not be compared cell by cell; see `kaggle/` for the newer parts.
