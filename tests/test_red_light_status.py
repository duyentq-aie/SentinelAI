"""An INACTIVE red-light rule must be visible: log line at start-up, HUD line, field in /api/status. Plus the approved test_video geometry."""
import numpy as np

from sentinel.config import REPO, load_camera_config, load_pipeline_config
from sentinel.pipeline import Pipeline, SharedState
from sentinel.rules.windowed import make_rules
from test_pipeline_fake import FakeDetectors
from test_worker import make

CAM_FILE = REPO / "configs" / "cameras" / "test_video.yaml"


def pipeline(tmp_path, **cam_changes):
    cfg = load_pipeline_config()
    cfg.evidence.dir = str(tmp_path / "ev")
    cam = load_camera_config(CAM_FILE)
    for k, v in cam_changes.items():
        setattr(cam, k, v)
    return Pipeline(FakeDetectors(lambda i: ([], [])), SharedState(), cfg, cam)


def hud_pixels(pipe, frame):
    return pipe.process(frame).frame[110:140, 40:700]


def test_the_approved_test_video_geometry_is_loaded_and_the_rule_is_active(tmp_path, capsys):
    pipe = pipeline(tmp_path)
    cam = pipe.camera
    assert cam.stop_line == [[400, 302], [1040, 277]] and cam.direction == [[650, 430], [665, 200]]
    assert [(l.name, l.signal) for l in cam.lanes] == [("left-turn", "exempt"), ("through", "shared"), ("right-turn", "exempt")]
    st = pipe.red_light_status()
    assert st["state"] == "active" and st["reasons"] == [] and st["lanes"] == 3
    assert "INACTIVE" not in capsys.readouterr().out


def test_missing_stop_line_or_light_roi_is_reported_everywhere(tmp_path, capsys):
    for field in ("stop_line", "light_roi"):
        pipe = pipeline(tmp_path, **{field: None})
        out = capsys.readouterr().out
        first = [l for l in out.splitlines() if l.strip()][0]
        assert "INACTIVE" in out and f"no camera.{field}" in out
        st = pipe.red_light_status()
        assert st["state"] == "inactive" and f"no camera.{field}" in st["reasons"]
        frame = np.full((720, 1280, 3), 90, np.uint8)
        with_hud = pipe.process(frame).frame[110:140, 40:700]
        assert (with_hud != frame[110:140, 40:700]).any()                              # the HUD line is painted
    ok = pipeline(tmp_path)
    frame = np.full((720, 1280, 3), 90, np.uint8)
    assert not (ok.process(frame).frame[110:140, 40:700] != frame[110:140, 40:700]).any()   # nothing painted when the rule is active


def test_the_first_log_line_of_a_start_up_without_a_stop_line_is_the_warning(tmp_path, capsys):
    pipeline(tmp_path, stop_line=None)
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert "INACTIVE" in lines[0]


def test_disabled_and_legacy_modes_are_not_reported_as_inactive(tmp_path):
    cfg = load_pipeline_config()
    cfg.evidence.dir = str(tmp_path / "ev")
    cfg.rules.red_light.mode = "legacy_zone"
    pipe = Pipeline(FakeDetectors(lambda i: ([], [])), SharedState(), cfg, load_camera_config(CAM_FILE))
    assert pipe.red_light_status()["state"] == "legacy_zone"
    cfg2 = load_pipeline_config()
    cfg2.evidence.dir = str(tmp_path / "ev2")
    cfg2.rules.red_light.enabled = False
    assert Pipeline(FakeDetectors(lambda i: ([], [])), SharedState(), cfg2, load_camera_config(CAM_FILE)).red_light_status()["state"] == "disabled"


def test_api_status_carries_the_red_light_state(tmp_path):
    w, pipe = make()
    pipe.red_light_status = lambda: {"state": "inactive", "mode": "stop_line_crossing", "reasons": ["no camera.stop_line"]}
    assert w.status()["red_light"]["state"] == "inactive"
    w2, _ = make()
    assert w2.status()["red_light"] is None                                           # a pipeline without the method: field present, null
    w.stop()
    w2.stop()
