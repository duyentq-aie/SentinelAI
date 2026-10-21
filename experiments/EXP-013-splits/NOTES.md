English translation of NOTES.vi.md (original, Vietnamese).

# EXP-013 – proposals for re-splitting helmet by video (2026-10-18)

**Nothing was re-split**: `tools/make_splits.py` only reads the dataset and writes a manifest (it does not move/copy/rename images). Choose a variant first, then create the split folders.
Command: `python tools/make_splits.py --data data/raw/helmet --out experiments/EXP-013-splits --classes helmet motorcycle no-helmet plate --balance no-helmet --seed 0`

## Groups
17 groups: 16 videos (`frame_1` … `frame_16`, 21–343 images) + 1 group `other:testing_images` (6 images of 1920×1080 named `testing_images_<n>`, unknown source; currently all 6 are in train). Each group goes whole into one split. This differs from the 16 groups you mentioned: the 6 1080p images do not belong to any numbered video.
Seeded search (minimises the deviation from the target proportions of the number of images and of `no-helmet` instances; penalises a split that lacks a class); the same seed gives the same result (test).

## Three variants (seed 0; instance counts include `plate` because the dataset has 4 classes, see DATA_CARD)
| | train | valid | test | no-helmet (train/val/test) | note |
|---|---|---|---|---|---|
| **A** 70/15/15, images and no-helmet balanced equally | 1026 images, 7 groups | 214, 4 groups | 220, 6 groups (includes the 6 1080p images) | 662 / 140 / 143 | the default I propose |
| **B** 70/15/15, priority to no-helmet | 1014, 8 groups | 226, 3 groups | 220, 6 groups | 660 / 142 / 143 | almost the same as A (differs by 2 videos); valid has only 3 videos → high variance |
| **C** 60/20/20, larger val/test | 875, 7 groups | 294, 6 groups | 291, 4 groups | 565 / 187 / 193 | less train data, more stable measurements; test has `frame_2`, `frame_3`, `frame_7` + 1080p |
Detail per class/video: `A|B|C/summary.md`, `summary.json`, `manifest.csv` (sha256 of every image + sha256 of the manifest). Compared with the old split (no-helmet 664/170/111) all three are better balanced.

## Notes
- Remaining similarity between different videos (dHash to the nearest train image, `cross_video_dhash.json`): test A 26.8% ≤ 4, B 24.5%, **C 9.6%**; the old random split was 86%. The cameras are static at the same place so videos still look alike; C the least, but it is also the variant with the least train data.
- Only 16 videos: the test result depends strongly on which videos fall into test; a bootstrap CI over videos should be reported, and self-recorded video remains the more trustworthy independent test.
- The 6 1080p images (other source) fall into test in all three variants because they are treated as one group: small effect (≈ 2–3% of the test images) but worth knowing.
- Every variant only solves adjacent-frame leakage; none guarantees that the videos do not show the same scene.
