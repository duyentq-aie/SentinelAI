"""Traffic-light state: from the camera's light_roi when it has one (OpenCV on a fixed region), otherwise from the COCO detector as before."""
import numpy as np

from sentinel.config import REPO, load_camera_config, load_pipeline_config
from sentinel.perception import Track
from sentinel.pipeline import Pipeline, SharedState
from test_pipeline_fake import FakeDetectors


def pipeline(tmp_path, light_roi, coco):
    cfg = load_pipeline_config()
    cfg.evidence.dir = str(tmp_path / "ev")
    cam = load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
    cam.light_roi = light_roi
    return Pipeline(FakeDetectors(lambda i: (coco, [])), SharedState(), cfg, cam)


def frame_with(red_at=None, green_at=None):
    f = np.full((720, 1280, 3), 90, np.uint8)
    for box, bgr in ((red_at, (0, 0, 255)), (green_at, (0, 255, 0))):
        if box:
            f[box[1]:box[3], box[0]:box[2]] = bgr
    return f


def test_without_light_roi_the_coco_detector_gives_the_state(tmp_path, capsys):
    p = pipeline(tmp_path, None, [Track(7, "traffic light", (1100, 150, 1140, 190), 0.9)])
    assert p.light_source == "coco" and "source: coco" in capsys.readouterr().out
    r = p.process(frame_with(red_at=(1100, 150, 1140, 190)))
    assert r.perception.light == "RED" and r.perception.coco[0].state == "RED"


def test_with_light_roi_the_state_comes_from_the_roi_and_the_coco_light_is_ignored(tmp_path, capsys):
    coco_light = Track(7, "traffic light", (1100, 150, 1140, 190), 0.9)               # COCO sees a GREEN light somewhere else ...
    p = pipeline(tmp_path, [590, 0, 630, 20], [coco_light])
    assert p.light_source == "roi" and "source: roi" in capsys.readouterr().out
    r = p.process(frame_with(red_at=(590, 0, 630, 20), green_at=(1100, 150, 1140, 190)))      # ... while the ROI is RED
    assert r.perception.light == "RED"
    r = p.process(frame_with(green_at=(590, 0, 630, 20)))
    assert r.perception.light == "GREEN"


def test_light_roi_state_works_without_any_coco_detection_and_is_unknown_for_a_dull_or_off_frame_roi(tmp_path):
    p = pipeline(tmp_path, [590, 0, 630, 20], [])
    assert p.process(frame_with(red_at=(590, 0, 630, 20))).perception.light == "RED"        # no detector involved
    assert p.process(frame_with()).perception.light == "UNKNOWN"                             # grey ROI
    p2 = pipeline(tmp_path, [5000, 5000, 5100, 5100], [])
    assert p2.process(frame_with()).perception.light == "UNKNOWN"                            # ROI outside the frame: no crash


def test_red_light_rule_logic_is_unchanged_by_the_source_of_the_light():
    from sentinel.perception import Perception
    from sentinel.rules.windowed import make_rules
    cam = load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
    cfg = load_pipeline_config().rules
    cfg.red_light.mode = "legacy_zone"   # the v0 zone rule; the stop-line rule has its own tests
    rules = make_rules(cfg, cam)
    p = Perception(helmet=[Track(1, "motorcycle", (300, 200, 400, 370), 0.9)], light="RED")        # bottom-centre (350, 370) inside the violation zone
    assert [v.reason for v in rules.step(p).vehicles] == ["RED-LIGHT RUNNER"]
