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
    return d


def make():
    cfg = load_pipeline_config()
    cam = load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
    state = SharedState()
    return Pipeline(fake_detectors(), state, cfg, cam), state


def test_reset_clears_every_id_keyed_state_and_keeps_gallery():
    p, s = make()
    p.rules.track_history[7].append((1, 2))
    p.prev_frame_time = 123.0
    p.rules.violation_counters[5] = 4
    s.captured_violator_ids.add("M_5")
    s.violation_records.append("/static/violations/a.jpg")

    p.reset()

    assert not p.rules.track_history and p.prev_frame_time == 0
    assert p.rules.violation_counters == {} and s.captured_violator_ids == set()
    assert s.violation_records == ["/static/violations/a.jpg"]          # gallery survives
    assert p.d.coco.predictor.trackers[0].resets == 1
    assert p.d.helmet.predictor.trackers[0].resets == 1


def test_reset_before_first_inference_is_safe():
    d = Detectors.__new__(Detectors)
    d.coco = SimpleNamespace(predictor=None)     # models not used yet: no predictor, no trackers
    d.helmet = SimpleNamespace()                 # not even the attribute
    d.plate = SimpleNamespace()
    cfg = load_pipeline_config()
    p = Pipeline(d, SharedState(), cfg, load_camera_config(REPO / "configs/cameras/test_video.yaml"))
    p.reset()                                    # must not raise
