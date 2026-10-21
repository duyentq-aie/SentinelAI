English translation of NOTES.vi.md (original, Vietnamese).

# EXP-011 – data leakage check of the detector datasets (2026-10-18)

Commands:
```
python tools/leakage_check.py --data data/raw/helmet --out experiments/EXP-011-data-leakage/helmet_leakage.json
python tools/leakage_check.py --data data/raw/plate  --out experiments/EXP-011-data-leakage/plate_leakage.json
```
Summary of the results and their interpretation: `docs/DATA_CARD.md`. Helmet: 100% of the valid/test images are in groups of consecutive frames that also appear in another split; 75.6% of the test images have a train frame of the same video at most 1 frame away. Plate: the adjacent-frame check was not possible (the names carry no frame number); 102 files are byte-identical within the same split.
No re-split, no training, `data/raw` not modified.
