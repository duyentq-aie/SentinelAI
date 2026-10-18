import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import near_dup_groups as nd  # noqa: E402
from leakage_check import load_split  # noqa: E402


def img(seed):
    rng = np.random.default_rng(seed)
    return cv2.resize(rng.integers(0, 255, (9, 12, 3), dtype=np.uint8), (120, 90), interpolation=cv2.INTER_CUBIC)


def put(root, split, name, im):
    d = Path(root) / split / "images"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / name), im)


def rows_of(root):
    return [{**r, "split": s} for s in ("train", "valid", "test") for r in load_split(root, s) if r["hash"] is not None]


def test_brightness_variant_across_splits_is_grouped_and_different_picture_is_not(tmp_path):
    a = img(1)
    put(tmp_path, "train", "a_jpg.rf.1.jpg", a)
    put(tmp_path, "test", "a_dark_jpg.rf.2.jpg", np.clip(a.astype(int) * 0.6, 0, 255).astype(np.uint8))   # augmented copy in another split
    put(tmp_path, "valid", "b_jpg.rf.3.jpg", img(2))
    put(tmp_path, "train", "c_jpg.rf.4.jpg", img(3))
    rows = rows_of(tmp_path)
    gid, info = nd.analyse(rows, 4)
    by = {r["name"]: g for r, g in zip(rows, gid)}
    assert by["a_jpg.rf.1.jpg"] == by["a_dark_jpg.rf.2.jpg"] and len({by["b_jpg.rf.3.jpg"], by["c_jpg.rf.4.jpg"], by["a_jpg.rf.1.jpg"]}) == 3
    assert info["groups"] == 3 and info["groups_spanning_several_splits"] == 1
    assert info["pct_val_test_images_whose_group_has_a_train_image"] == {"valid": 0.0, "test": 100.0}


def test_chain_of_slowly_changing_frames_is_one_group(tmp_path):
    base = img(5).astype(int)
    for k in range(6):
        put(tmp_path, ("train", "valid", "test")[k % 3], f"f{k}_jpg.rf.{k}.jpg", np.clip(base + k * 2, 0, 255).astype(np.uint8))
    _, info = nd.analyse(rows_of(tmp_path), 4)
    assert info["groups"] == 1 and info["largest_groups"] == [6]
