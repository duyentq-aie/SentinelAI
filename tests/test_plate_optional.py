"""The plate layer is optional: no plate file (plate_best.pt is not published) or models.plate.enabled: false -> the app starts, events are recorded, no plate search, /api/status says why."""
import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import sentinel.pipeline as pl
from sentinel.config import REPO, load_pipeline_config
from sentinel.modelinfo import plate_layer_status
from test_pipeline_fake import FakeDetectors, frame, make, moto_no_helmet
from test_worker import make as make_worker


def cfg_with_plate(tmp_path, exists=True, enabled=True):
    cfg = load_pipeline_config(root=tmp_path)
    cfg.models.plate.weights = "plate_best.pt"
    cfg.models.plate.enabled = enabled
    if exists:
        (tmp_path / "plate_best.pt").write_bytes(b"x")
    return cfg


def test_plate_layer_status_for_the_three_cases(tmp_path):
    (tmp_path / "a").mkdir()
    assert plate_layer_status(cfg_with_plate(tmp_path / "a")) == {"state": "enabled", "reason": None}
    (tmp_path / "b").mkdir()
    missing = plate_layer_status(cfg_with_plate(tmp_path / "b", exists=False))
    assert missing["state"] == "disabled" and "not found" in missing["reason"] and "plate_best.pt" in missing["reason"] and "not published" in missing["reason"]
    (tmp_path / "c").mkdir()
    off = plate_layer_status(cfg_with_plate(tmp_path / "c", enabled=False))                       # the file is there, the switch is off
    assert off["state"] == "disabled" and "enabled is false" in off["reason"]


def test_only_the_plate_model_can_be_switched_off(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("models:\n  plate: {weights: plate_best.pt, conf: 0.35, enabled: false}\n", encoding="utf-8")
    assert load_pipeline_config(f).models.plate.enabled is False
    for name in ("coco", "helmet"):
        f.write_text(f"models:\n  {name}: {{weights: a.pt, conf: 0.5, enabled: false}}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="only the plate layer is optional"):
            load_pipeline_config(f)
    assert load_pipeline_config().models.plate.enabled is True                                       # default: on (when the file exists)


class FakeYOLO:
    loaded = []

    def __init__(self, path):
        FakeYOLO.loaded.append(Path(path).name)
        self.predictor = None
        self.names = {}

    def __call__(self, *a, **k):
        raise AssertionError("the plate model must not run")


def test_detectors_start_without_the_plate_file_and_never_run_a_plate_model(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "YOLO", FakeYOLO)
    FakeYOLO.loaded = []
    cfg = load_pipeline_config(root=tmp_path)
    cfg.models.helmet.weights, cfg.models.coco.weights, cfg.models.plate.weights = "h.pt", "c.pt", "plate_best.pt"           # plate_best.pt does not exist
    d = pl.Detectors(cfg)
    assert sorted(FakeYOLO.loaded) == ["c.pt", "h.pt"] and d.plate is None
    assert "plate_best.pt" in d.plate_disabled_reason
    import numpy as np
    assert d.plate_boxes(np.zeros((50, 50, 3), np.uint8)) == []                                      # no model: no boxes, no crash


def test_detectors_load_the_plate_model_when_it_exists_and_not_when_the_switch_is_off(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "YOLO", FakeYOLO)
    cfg = load_pipeline_config(root=tmp_path)
    cfg.models.helmet.weights, cfg.models.coco.weights, cfg.models.plate.weights = "h.pt", "c.pt", "plate_best.pt"
    (tmp_path / "plate_best.pt").write_bytes(b"x")
    FakeYOLO.loaded = []
    d = pl.Detectors(cfg)
    assert sorted(FakeYOLO.loaded) == ["c.pt", "h.pt", "plate_best.pt"] and d.plate_disabled_reason is None
    cfg.models.plate.enabled = False
    FakeYOLO.loaded = []
    d = pl.Detectors(cfg)
    assert sorted(FakeYOLO.loaded) == ["c.pt", "h.pt"] and d.plate is None and "enabled is false" in d.plate_disabled_reason


class NoPlateDetectors(FakeDetectors):
    plate_disabled_reason = "plate model file not found: plate_best.pt (plate_best.pt is not published)"

    def plate_boxes(self, crop):
        raise AssertionError("the plate layer is off: nothing may call the plate detector")


def test_pipeline_without_a_plate_layer_records_events_searches_no_plate_and_logs_it_once(tmp_path, capsys):
    from sentinel.config import load_camera_config
    from sentinel.pipeline import Pipeline, SharedState
    cfg = load_pipeline_config()
    cfg.evidence.dir = str(tmp_path / "ev")
    det = NoPlateDetectors(moto_no_helmet)
    p = Pipeline(det, SharedState(), cfg, load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml"))
    out = capsys.readouterr().out
    assert out.count("[plate] plate layer DISABLED") == 1 and "[!]" not in out                      # one calm info line, no warning marker
    outs = [p.process(frame()) for _ in range(8)]
    p.flush()
    assert sum(len(o.new_events) for o in outs) == 1 and all(o.plate_calls == 0 and o.captures == [] for o in outs)
    rec = [json.loads(l) for l in (tmp_path / "ev" / "events.jsonl").read_text(encoding="utf-8").splitlines()][-1]
    assert rec["type"] == "no_helmet" and rec["plate"] == {"found": False, "attempts": 0, "disabled": True} and rec["evidence"]["plate"] is None
    assert p.plate_status() == {"state": "disabled", "reason": det.plate_disabled_reason}


def test_pipeline_with_a_plate_layer_is_unchanged(tmp_path):
    p, det = make(tmp_path)
    for _ in range(8):
        p.process(frame())
    assert p.plate_status() == {"state": "enabled", "reason": None} and det.plate_calls == 1


def test_api_status_says_plate_disabled_with_the_reason_and_the_hud_has_a_neutral_line():
    w, pipe = make_worker()
    pipe.plate_status = lambda: {"state": "disabled", "reason": "plate model file not found: plate_best.pt"}
    st = w.status()
    assert st["plate"] == {"state": "disabled", "reason": "plate model file not found: plate_best.pt"}
    w.stop()
    html = (REPO / "templates" / "index.html").read_text(encoding="utf-8")
    assert "line('plate'" in html and "'off'" in html and "plate layer off" in html and "alert(" not in html.split("fetchStatus")[1].split("setInterval")[0]


def test_the_app_wiring_accepts_a_config_without_a_plate_file(tmp_path):
    """create_app / backend_info only look at config and files: a missing plate file is fine."""
    from apps.web.routes import backend_info
    cfg = load_pipeline_config(root=tmp_path)
    cfg.models.plate.weights = "missing_plate.pt"
    info = backend_info(cfg)
    assert info["helmet_model"]["file"].endswith("best.pt") and "missing_plate.pt" in info["models"]
