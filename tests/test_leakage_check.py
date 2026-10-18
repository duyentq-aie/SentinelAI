import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import leakage_check as lc  # noqa: E402


def img(seed, size=(120, 90)):
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.integers(0, 255, (9, 12, 3), dtype=np.uint8), size, interpolation=cv2.INTER_CUBIC)
    return base


def put(root, split, name, im):
    d = Path(root) / split / "images"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / name), im)


def test_parse_roboflow_names():
    assert lc.parse_name("frame_10_563_jpg.rf.ec45d19f262915fce4a4d77caefe12f8.jpg") == {"src": "frame_10_563", "video": "frame_10", "frame": 563}
    assert lc.parse_name("clip7_1200_jpg.rf.abc123.jpg")["video"] == "clip7" and lc.parse_name("clip7_1200_jpg.rf.abc123.jpg")["frame"] == 1200
    assert lc.parse_name("random_name.jpg") == {"src": "random_name", "video": None, "frame": None}


def test_dhash_is_stable_for_small_changes_and_differs_for_other_pictures():
    a = img(1)
    noisy = np.clip(a.astype(int) + np.random.default_rng(0).integers(-3, 4, a.shape), 0, 255).astype(np.uint8)
    d = lc.hamming_matrix(np.array([lc.dhash(a)], dtype=np.uint64), np.array([lc.dhash(noisy), lc.dhash(img(2))], dtype=np.uint64))[0]
    assert d[0] <= 6 and d[1] >= 16 and lc.hamming_matrix(np.array([5], dtype=np.uint64), np.array([5], dtype=np.uint64))[0, 0] == 0


def test_a_random_frame_split_of_one_video_is_reported_as_leaking(tmp_path):
    base = img(3)
    for f in range(100, 120):                                            # 20 consecutive frames of one video, each a small variation of the same picture
        im = np.clip(base.astype(int) + (f - 100), 0, 255).astype(np.uint8)
        split = "test" if f % 10 == 5 else ("valid" if f % 10 == 7 else "train")
        put(tmp_path, split, f"vid_3_{f}_jpg.rf.{f:08x}.jpg", im)
    r = lc.check(tmp_path, gap=5)
    assert r["images"] == {"train": 16, "valid": 2, "test": 2}
    assert r["temporal"]["test"]["pct_with_train_frame_within_5"] == 100.0 and r["temporal"]["test"]["pct_with_train_frame_within_1"] == 100.0
    assert r["dhash_nearest_train_hamming"]["test"]["pct_with_train_neighbour_le_8"] == 100.0
    assert r["groups"]["groups"] == 1 and r["groups"]["groups_spanning_several_splits"] == 1
    assert "test" in r["groups"]["images_in_leaky_groups"]


def test_an_independent_split_by_video_is_clean(tmp_path):
    for k, split in ((1, "train"), (2, "train"), (3, "valid"), (4, "test")):
        for f in range(10):
            put(tmp_path, split, f"cam{k}_{f}_jpg.rf.{k}{f:07x}.jpg", np.clip(img(10 * k).astype(int) + f, 0, 255).astype(np.uint8))
    r = lc.check(tmp_path, gap=5)
    assert r["groups"]["groups_spanning_several_splits"] == 0
    assert r["temporal"]["test"]["pct_with_train_frame_within_30"] == 0.0      # no train frame of the same video
    assert r["dhash_nearest_train_hamming"]["test"]["pct_with_train_neighbour_le_4"] == 0.0


def test_byte_identical_files_across_splits_are_found(tmp_path):
    im = img(5)
    put(tmp_path, "train", "a_jpg.rf.1.jpg", im)
    put(tmp_path, "test", "a_copy_jpg.rf.2.jpg", im)
    put(tmp_path, "valid", "other_jpg.rf.3.jpg", img(6))
    r = lc.check(tmp_path)
    assert r["identical_files_across_splits"] == 1 and r["files_with_a_byte_identical_twin_anywhere"] == 1
    assert "byte-identical files across splits: 1" in lc.format_report(r)


def test_names_without_frame_numbers_get_an_explicit_note(tmp_path):
    put(tmp_path, "train", "x.jpg", img(1))
    put(tmp_path, "test", "y.jpg", img(2))
    r = lc.check(tmp_path)
    assert "not possible" in r["temporal"]["test"]["note"]
