import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import make_splits as ms  # noqa: E402

CLASSES = ["helmet", "motorcycle", "no-helmet"]


def build(tmp_path, n_videos=8, per_video=12):
    rng = np.random.default_rng(0)
    for v in range(1, n_videos + 1):
        for f in range(per_video):
            split = ("train", "valid", "test")[(v + f) % 3]                    # deliberately leaky original split
            d = tmp_path / split
            (d / "images").mkdir(parents=True, exist_ok=True)
            (d / "labels").mkdir(parents=True, exist_ok=True)
            name = f"frame_{v}_{f}_jpg.rf.{v:02d}{f:02d}"
            cv2.imwrite(str(d / "images" / (name + ".jpg")), rng.integers(0, 255, (24, 32, 3), dtype=np.uint8))
            lines = ["0 0.5 0.5 0.1 0.1", "1 0.5 0.5 0.2 0.2"] + (["2 0.4 0.4 0.1 0.1"] if (v + f) % 4 == 0 else [])
            (d / "labels" / (name + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def run(tmp_path, seed=0):
    rows = ms.read_dataset(tmp_path, CLASSES)
    groups = ms.group_stats(rows, CLASSES)
    assign, cost = ms.search(groups, CLASSES, ["no-helmet"], ms.VARIANTS["A"], seed, restarts=40)
    return rows, groups, assign, cost


def test_whole_videos_go_to_one_split_and_every_split_has_every_class(tmp_path):
    rows, groups, assign, _ = run(build(tmp_path))
    assert set(assign) == {f"frame_{v}" for v in range(1, 9)}
    summ = ms.summarize(rows, assign, CLASSES)
    assert sum(summ[s]["images"] for s in ms.SPLITS) == 96
    for s in ms.SPLITS:
        assert all(summ[s]["instances"][c] > 0 for c in CLASSES), (s, summ[s])
    seen = {}
    for r in rows:
        seen.setdefault(r["video"], set()).add(ms.SPLITS[assign[r["video"]]])
    assert all(len(v) == 1 for v in seen.values())


def test_search_is_deterministic_for_a_seed(tmp_path):
    root = build(tmp_path)
    assert run(root, 3)[2] == run(root, 3)[2]


def test_manifest_has_hashes_and_summary_hash_matches_file(tmp_path):
    root = build(tmp_path / "data")
    rows, groups, assign, cost = run(root)
    out = tmp_path / "out"
    info = ms.write_variant(out, "A", ms.VARIANTS["A"], rows, assign, CLASSES, 0, cost, ["no-helmet"])
    lines = list(csv.DictReader(open(out / "A" / "manifest.csv", encoding="utf-8")))
    assert len(lines) == 96 and all(len(r["sha256"]) == 64 for r in lines)
    assert info["manifest_sha256"] == json.loads((out / "A" / "summary.json").read_text(encoding="utf-8"))["manifest_sha256"]
    import hashlib
    assert hashlib.sha256((out / "A" / "manifest.csv").read_bytes()).hexdigest() == info["manifest_sha256"]
    assert not list((root).rglob("*.csv"))                                    # the dataset directory is untouched


def test_names_without_video_ids_are_grouped_by_stem(tmp_path):
    d = tmp_path / "train"
    (d / "images").mkdir(parents=True)
    (d / "labels").mkdir()
    for i in (1, 2):
        cv2.imwrite(str(d / "images" / f"testing_images_{i}_jpg.rf.abc{i}.jpg"), np.zeros((8, 8, 3), np.uint8))
    rows = ms.read_dataset(tmp_path, CLASSES)
    assert {r["video"] for r in rows} == {"other:testing_images"}
