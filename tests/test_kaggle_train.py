import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import kaggle_train as kt  # noqa: E402
import make_splits as ms  # noqa: E402

CLASSES = ["helmet", "motorcycle", "no-helmet", "plate"]


def dataset(tmp_path):
    rng = np.random.default_rng(1)
    for v in range(1, 7):
        for f in range(6):
            split = ("train", "valid", "test")[(v + f) % 3]
            d = tmp_path / split
            (d / "images").mkdir(parents=True, exist_ok=True)
            (d / "labels").mkdir(parents=True, exist_ok=True)
            n = f"frame_{v}_{f}_jpg.rf.{v}{f}"
            cv2.imwrite(str(d / "images" / (n + ".jpg")), rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
            (d / "labels" / (n + ".txt")).write_text("0 .5 .5 .1 .1\n1 .5 .5 .2 .2\n3 .5 .5 .05 .05\n" + ("2 .4 .4 .1 .1\n" if (v + f) % 3 == 0 else ""), encoding="utf-8")


def make_split(tmp_path):
    rows = ms.read_dataset(tmp_path / "raw", CLASSES)
    groups = ms.group_stats(rows, CLASSES)
    assign, cost = ms.search(groups, CLASSES, ["no-helmet"], ms.VARIANTS["A"], 0, restarts=30)
    ms.write_variant(tmp_path / "splits", "A", ms.VARIANTS["A"], rows, assign, CLASSES, 0, cost, ["no-helmet"])
    return tmp_path / "splits" / "A"


class A:
    def __init__(self, **k):
        self.__dict__.update(k)


def test_prepare_drops_plate_keeps_three_classes_and_copies_by_manifest(tmp_path):
    dataset(tmp_path / "raw")
    sd = make_split(tmp_path)
    out = tmp_path / "out"
    kt.prepare(A(data=str(tmp_path / "raw"), split_dir=str(sd), out=str(out), slug="sentinel-helmet-split-a"))
    labs = [l for p in out.glob("*/labels/*.txt") for l in p.read_text(encoding="utf-8").splitlines()]
    assert labs and {int(l.split()[0]) for l in labs} <= {0, 1, 2}
    assert len(list(out.glob("*/images/*.jpg"))) == 36
    meta = json.loads((out / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"].startswith("<KAGGLE_USER>/") and (out / "params.json").exists()
    rows = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
    for r in rows:
        assert (out / r["split"] / "images" / Path(r["path"]).name).exists()


def test_prepare_refuses_a_non_empty_output_and_a_tampered_manifest(tmp_path):
    dataset(tmp_path / "raw")
    sd = make_split(tmp_path)
    (tmp_path / "o").mkdir()
    (tmp_path / "o" / "x").write_text("1", encoding="utf-8")
    try:
        kt.prepare(A(data=str(tmp_path / "raw"), split_dir=str(sd), out=str(tmp_path / "o"), slug="s"))
        assert False
    except SystemExit as e:
        assert "not empty" in str(e)
    (sd / "manifest.csv").write_text((sd / "manifest.csv").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    try:
        kt.prepare(A(data=str(tmp_path / "raw"), split_dir=str(sd), out=str(tmp_path / "o2"), slug="s"))
        assert False
    except SystemExit as e:
        assert "manifest" in str(e)


def test_kernel_notebook_is_valid_private_gpu_and_has_no_user_name(tmp_path):
    out = tmp_path / "k"
    kt.kernel(A(dataset_slug="sentinel-helmet-split-a", slug="sentinel-helmet-train-a", out=str(out)))
    nb = json.loads((out / "train_helmet.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and any(c["cell_type"] == "code" for c in nb["cells"]) and any("params.json" in "".join(c["source"]) for c in nb["cells"])
    km = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert km["is_private"] is True and km["enable_gpu"] is True and km["id"].startswith("<KAGGLE_USER>/") and km["dataset_sources"] == ["<KAGGLE_USER>/sentinel-helmet-split-a"]
    for code in (c for c in nb["cells"] if c["cell_type"] == "code"):
        compile("".join(code["source"]), "cell", "exec")                   # every cell is valid Python


def test_push_refuses_without_confirm_placeholder_user_or_token(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "nothing"))
    for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    for args in (A(confirm=False, user="me"), A(confirm=True, user="<KAGGLE_USER>"), A(confirm=True, user=None), A(confirm=True, user="me")):
        try:
            kt.guarded(args, "push-dataset")
            assert False, args.__dict__
        except SystemExit as e:
            assert str(e)
    assert kt.token_present() is False


def test_committed_training_params_match_the_old_checkpoint_recipe():
    d = json.loads((REPO / "configs" / "training" / "helmet_yolo11s_v1.json").read_text(encoding="utf-8"))
    p = d["params"]
    assert d["meta"]["base_model"] == "yolo11s.pt" and len(d["meta"]["names"]) == 3 and d["meta"]["ultralytics_version"] == "8.4.26"
    assert (p["epochs"], p["patience"], p["batch"], p["imgsz"], p["seed"], p["cos_lr"], p["close_mosaic"], p["mixup"], p["degrees"]) == (150, 25, 32, 640, 0, True, 10, 0.1, 10.0)
    assert "tranquang" not in json.dumps(d).lower()
