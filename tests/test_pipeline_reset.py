"""State must not leak across sources / video loops (audit section 2.E)."""
from types import SimpleNamespace

from sentinel.config import load_camera_config, load_pipeline_config, REPO
from sentinel.pipeline import Detectors, Pipeline, SharedState


class FakeTracker:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


def fake_detectors():
    d = Detectors.__new__(Detectors)             # skip loading weights
    d.coco = SimpleNamespace(predictor=SimpleNamespace(trackers=[FakeTracker()]))
    d.helmet = SimpleNamespace(predictor=SimpleNamespace(trackers=[FakeTracker()]))
    d.plate = SimpleNamespace()                  # plate has no tracker
    d.plate_boxes = lambda crop: []
    return d


def make(tmp_path):
    cfg = load_pipeline_config()
    cfg.evidence.dir = str(tmp_path)
    cam = load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
    state = SharedState()
    return Pipeline(fake_detectors(), state, cfg, cam), state


def test_reset_clears_every_id_keyed_state_and_keeps_gallery(tmp_path):
    p, s = make(tmp_path)
    p.rules.track_history[7].append((1, 2))
    p.prev_frame_time = 123.0
    p.rules.violation_counters[5] = 4
    p.recorder._plates["M_5"] = {"found": True}
    p.recorder._attempts["M_5"] = 3
    p.frame_idx = 99
    s.violation_records.append("/static/violations/a.jpg")

    p.reset()

    assert not p.rules.track_history and p.prev_frame_time == 0
    assert p.rules.violation_counters == {} and p.recorder._plates == {} and p.recorder._attempts == {}
    assert p.frame_idx == 0 and p.recorder.run == 1
    assert s.violation_records == ["/static/violations/a.jpg"]          # gallery survives
    assert p.d.coco.predictor.trackers[0].resets == 1
    assert p.d.helmet.predictor.trackers[0].resets == 1


def test_reset_before_first_inference_is_safe(tmp_path):
    d = Detectors.__new__(Detectors)
    d.plate_boxes = lambda crop: []
    d.coco = SimpleNamespace(predictor=None)     # models not used yet: no predictor, no trackers
    d.helmet = SimpleNamespace()                 # not even the attribute
    d.plate = SimpleNamespace()
    cfg = load_pipeline_config()
    cfg.evidence.dir = str(tmp_path)
    p = Pipeline(d, SharedState(), cfg, load_camera_config(REPO / "configs/cameras/test_video.yaml"))
    p.reset()                                    # must not raise
