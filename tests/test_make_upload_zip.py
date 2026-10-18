import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import make_splits as ms  # noqa: E402
import make_upload_zip as mz  # noqa: E402

CLASSES = ["helmet", "motorcycle", "no-helmet", "plate"]


def build_raw(root):
    rng = np.random.default_rng(2)
    (root).mkdir(parents=True)
    (root / "data.yaml").write_text("nc: 4\nnames: ['helmet', 'motorcycle', 'no-helmet', 'plate']\n", encoding="utf-8")
    names = [f"frame_{v}_{f}_jpg.rf.{v}{f}" for v in range(1, 7) for f in range(5)] + ["testing_images_1_jpg.rf.zz1", "testing_images_2_jpg.rf.zz2"]
    for k, n in enumerate(names):
        d = root / ("train", "valid", "test")[k % 3]
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "images" / (n + ".jpg")), rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
        (d / "labels" / (n + ".txt")).write_text("0 .5 .5 .1 .1\n3 .5 .5 .05 .05\n" + ("2 .4 .4 .1 .1\n" if k % 2 else ""), encoding="utf-8")


def make(tmp_path):
    raw = tmp_path / "raw"
    build_raw(raw)
    rows = ms.read_dataset(raw, CLASSES)
    groups = ms.group_stats(rows, CLASSES)
    assign, cost = ms.search(groups, CLASSES, ["no-helmet"], ms.VARIANTS["A"], 0, restarts=30)
    ms.write_variant(tmp_path / "sp", "A", ms.VARIANTS["A"], rows, assign, CLASSES, 0, cost, ["no-helmet"])
    return raw, tmp_path / "sp" / "A"


def test_zip_layout_four_classes_extra_folder_and_untouched_sources(tmp_path):
    raw, sp = make(tmp_path)
    before = {p: p.read_bytes() for p in raw.rglob("*") if p.is_file()}
    out = tmp_path / "up" / "x.zip"
    mz.build(raw, sp, out)
    z = zipfile.ZipFile(out)
    names = z.namelist()
    cfg = yaml.safe_load(z.read("data.yaml"))
    assert cfg["names"] == CLASSES and cfg["nc"] == 4 and cfg["val"] == "images/val"
    extra = [n for n in names if n.startswith("images/extra_1080p/")]
    assert sorted(Path(n).name for n in extra) == ["testing_images_1_jpg.rf.zz1.jpg", "testing_images_2_jpg.rf.zz2.jpg"]
    assert {n.split("/")[1] for n in names if n.startswith("images/")} <= {"train", "val", "test", "extra_1080p"}
    for n in (n for n in names if n.startswith("images/")):
        lab = n.replace("images/", "labels/", 1).rsplit(".", 1)[0] + ".txt"
        assert lab in names
    assert "SPLIT_INFO.md" in names and "manifest.csv" in names and b"sha256" in z.read("SPLIT_INFO.md")
    assert any(l.startswith("3 ") for n in names if n.startswith("labels/") for l in z.read(n).decode().splitlines())      # plate labels kept
    assert before == {p: p.read_bytes() for p in raw.rglob("*") if p.is_file()}


def test_zip_is_deterministic_and_not_overwritten(tmp_path):
    raw, sp = make(tmp_path)
    mz.build(raw, sp, tmp_path / "a.zip")
    mz.build(raw, sp, tmp_path / "b.zip")
    assert (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()
    try:
        mz.build(raw, sp, tmp_path / "a.zip")
        assert False
    except SystemExit as e:
        assert "exists" in str(e)
