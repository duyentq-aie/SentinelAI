import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import box_consistency as bc  # noqa: E402

CLASSES = ["helmet", "motorcycle", "no-helmet", "plate"]


def setup(tmp_path):
    raw = tmp_path / "raw"
    (raw / "train" / "images").mkdir(parents=True)
    (raw / "train" / "labels").mkdir()
    cv2.imwrite(str(raw / "train" / "images" / "frame_1_0_jpg.rf.a.jpg"), np.zeros((200, 400, 3), np.uint8))
    # image 400x200 px. moto: x .5 y .5 w .5 h 1.0 -> 200x200 px; helmet 40x40 px (aspect 1, rel_h .2); no-helmet 40x100 px (aspect 2.5, rel_h .5); plate outside the moto
    (raw / "train" / "labels" / "frame_1_0_jpg.rf.a.txt").write_text(
        "1 0.5 0.5 0.5 1.0\n0 0.5 0.3 0.1 0.2\n2 0.55 0.6 0.1 0.5\n3 0.9 0.9 0.05 0.05\n", encoding="utf-8")
    sp = tmp_path / "split"
    sp.mkdir()
    with open(sp / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["split", "video", "frame", "path", "orig_split", "sha256", "width", "height"])
        w.writerow(["train", "frame_1", 0, "train/images/frame_1_0_jpg.rf.a.jpg", "train", "x", 400, 200])
    return raw, sp


def test_aspect_is_measured_in_pixels_and_heads_are_matched_to_the_containing_motorcycle(tmp_path):
    raw, sp = setup(tmp_path)
    boxes = {b["cls"]: b for b in bc.read_boxes(raw, sp, CLASSES)}
    assert round(boxes["helmet"]["aspect"], 3) == 1.0 and round(boxes["helmet"]["rel_h"], 3) == 0.2
    assert round(boxes["no-helmet"]["aspect"], 3) == 2.5 and round(boxes["no-helmet"]["rel_h"], 3) == 0.5
    assert boxes["motorcycle"]["rel_h"] is None and boxes["plate"]["rel_h"] is None            # plate centre is outside the moto box
    assert all(b["folder"] == "train" for b in boxes.values())


def test_quantiles_and_valley_on_a_two_cluster_sample():
    rng = np.random.default_rng(0)
    v = list(rng.normal(1.0, 0.05, 300)) + list(rng.normal(2.0, 0.05, 300))
    q = bc.quantiles(v)
    assert q["n"] == 600 and 1.0 < q["median"] < 2.0
    va = bc.valley(v)
    assert 1.1 < va["otsu_threshold"] < 1.9 and va["valley_depth_ratio"] < 0.2          # clear valley
    uni = bc.valley(list(rng.normal(1.35, 0.2, 600)))
    assert uni["valley_depth_ratio"] > 0.6                                               # unimodal: no valley


def test_main_flags_tall_boxes_and_writes_outputs(tmp_path, monkeypatch):
    raw, sp = setup(tmp_path)
    import json
    (sp / "summary.json").write_text(json.dumps({}), encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["box_consistency.py", "--data", str(raw), "--split-dir", str(sp), "--out", str(out), "--aspect-threshold", "1.5", "--rel-threshold", "0.3"])
    bc.main()
    rows = list(csv.DictReader(open(out / "boxes.csv", encoding="utf-8")))
    flags = {r["cls"]: r["flag"] for r in rows}
    assert flags["no-helmet"] == "1" and flags["helmet"] == "0" and flags["motorcycle"] == "0"
    assert (out / "summary.json").exists() and (out / "hist_aspect.png").exists() and (out / "hist_rel_h.png").exists()
