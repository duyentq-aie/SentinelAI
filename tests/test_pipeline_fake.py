"""Pipeline end-to-end with fake detectors (no models): rule engine -> evidence -> drawing, config-driven."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sentinel.config import REPO, load_camera_config, load_pipeline_config
from sentinel.perception import Track
from sentinel.pipeline import Detectors, Pipeline, SharedState


class FakeDetectors(Detectors):
    def __init__(self, script):
        self.script = script           # frame index -> (coco tracks, helmet tracks)
        self.i = -1
        self.plate_calls = 0
        self.coco = self.helmet = self.plate = SimpleNamespace(predictor=None)

    def track_coco(self, frame):
        self.i += 1
        return [Track(*t) if not isinstance(t, Track) else t for t in self.script(self.i)[0]]

    def track_helmet(self, frame):
        return [Track(*t) if not isinstance(t, Track) else t for t in self.script(self.i)[1]]

    def plate_boxes(self, crop):
        self.plate_calls += 1
        h, w = crop.shape[:2]
        return [(w // 4, h // 2, w // 2, h * 3 // 4, 0.9)]


def moto_no_helmet(i):
    """A motorcycle (helmet-model track 5) whose rider has no helmet on every frame."""
    return ([], [Track(5, "motorcycle", (400, 300, 700, 650), 0.9), Track(0, "no-helmet", (500, 320, 560, 380), 0.9)])


def make(tmp_path, script=moto_no_helmet, **cfg_kw):
    cfg = load_pipeline_config()
    cfg.evidence.dir = str(tmp_path / "ev")
    for k, v in cfg_kw.items():
        scope, name = k.split("__")
        setattr(getattr(cfg, scope), name, v)
    cam = load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
    det = FakeDetectors(script)
    return Pipeline(det, SharedState(), cfg, cam), det


def frame():
    return np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)


def test_no_helmet_event_with_evidence_after_the_vote(tmp_path):
    p, det = make(tmp_path)
    outs = [p.process(frame()) for _ in range(8)]
    assert [o.rules.vehicles[0].violator for o in outs] == [False] * 4 + [True] * 4          # 5th vote
    assert sum(len(o.new_events) for o in outs) == 1 and outs[4].new_events
    p.flush()
    lines = [json.loads(l) for l in (tmp_path / "ev" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["status"] == "closed" and lines[-1]["type"] == "no_helmet" and lines[-1]["track_id"] == "M_5"
    assert lines[-1]["plate"]["found"] is True and det.plate_calls == 1                       # first attempt finds the fake plate, then stops
    assert outs[4].frame.shape == outs[4].raw.shape == (720, 1280, 3)                         # resized to cfg.frame_size


def test_raw_frame_is_never_drawn_on_but_display_is(tmp_path):
    p, _ = make(tmp_path)
    src = frame()
    out = p.process(src)
    import cv2
    assert np.array_equal(out.raw, cv2.resize(src, (1280, 720)))
    assert not np.array_equal(out.frame, out.raw)


def test_blend_mode_does_not_change_the_pixels(tmp_path):
    a, _ = make(tmp_path / "a", render__blend="full")
    b, _ = make(tmp_path / "b", render__blend="roi")
    for _ in range(7):
        fa, fb = a.process(frame()), b.process(frame())
        # the FPS counter differs by timing only; compare everything below it
        assert np.abs(fa.frame[120:].astype(int) - fb.frame[120:].astype(int)).max() <= 1       # whole-frame blend adds +-1 noise elsewhere
        assert fa.labels == fb.labels and bool(fa.new_events) == bool(fb.new_events)


def test_engine_switch_changes_the_verdict(tmp_path):
    legacy, _ = make(tmp_path / "l", rules__engine="legacy")
    windowed, _ = make(tmp_path / "w")
    lv = [legacy.process(frame()).rules.vehicles[0].violator for _ in range(6)]
    wv = [windowed.process(frame()).rules.vehicles[0].violator for _ in range(6)]
    assert lv == wv == [False] * 4 + [True] * 2                                               # both need 5 consecutive bad frames here
