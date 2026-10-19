"""Per-lane signals: {roi: [...]} (the lane's own light), "shared" (light_roi; old spelling "roi") and "exempt"; read, judged, logged, drawn and reported per lane."""
import numpy as np
import pytest
import yaml

from sentinel.config import Lane, load_camera_config, load_pipeline_config
from sentinel.perception import Perception, Track
from sentinel.pipeline import Pipeline, SharedState
from sentinel.rules.windowed import make_rules
from test_pipeline_fake import FakeDetectors
from test_stopline_rule import APPROACH, LINE, UP, camera

ARROW = [100, 5, 130, 20]                                        # the own light of the "turn" lane (an arrow)
TURN = [[100, 100], [300, 100], [300, 500], [100, 500]]
GO = [[300, 100], [500, 100], [500, 500], [300, 500]]


def lanes(turn_signal):
    return [Lane("turn", TURN, turn_signal), Lane("go", GO, "shared")]


def run(cam, x, light, lane_lights, **kw):
    cfg = load_pipeline_config().rules
    r = make_rules(cfg, cam)
    got = []
    for y in APPROACH:
        res = r.step(Perception(coco=[Track(1, "motorcycle", (x - 30, y - 70, x + 30, y), 0.9)], light=light, lane_lights=lane_lights))
        got.append(any(v.id == "K_1" and v.violator for v in res.vehicles))
    return any(got), r.stop_line_rule.crossings


def test_the_three_forms_parse_from_yaml_and_the_old_roi_means_shared(tmp_path):
    base = {"camera_id": "c", "source": "c.mp4", "violation_zone": [[0, 0], [5, 0], [5, 5]], "stop_line": LINE, "direction": UP,
            "lanes": [{"name": "own", "polygon": TURN, "signal": {"roi": ARROW}}, {"name": "sh", "polygon": GO, "signal": "shared"},
                      {"name": "ex", "polygon": [[0, 0], [9, 0], [9, 9]], "signal": "exempt"}, {"name": "old", "polygon": [[20, 0], [29, 0], [29, 9]], "signal": "roi"},
                      {"name": "default", "polygon": [[40, 0], [49, 0], [49, 9]]}]}
    f = tmp_path / "cam.yaml"
    f.write_text(yaml.safe_dump(base), encoding="utf-8")
    cam = load_camera_config(f)
    assert [(l.name, l.kind, l.roi) for l in cam.lanes] == [("own", "own", ARROW), ("sh", "shared", None), ("ex", "exempt", None), ("old", "shared", None), ("default", "shared", None)]
    assert cam.lanes[0].signal == {"roi": ARROW} and cam.lanes[3].signal == "shared"


@pytest.mark.parametrize("signal", [{"roi": [1, 2, 3]}, {"roi": [5, 5, 5, 9]}, {"roi": [9, 1, 5, 5]}, {"roi": "x"}, {"roi": ARROW, "extra": 1}, {}, "green", "own", 5])
def test_invalid_lane_signals_are_rejected(signal):
    with pytest.raises(ValueError):
        camera([Lane("a", TURN, signal)])


def test_own_light_lane_is_judged_by_its_own_state_not_by_the_shared_one():
    cam = camera(lanes({"roi": ARROW}))
    ok, crossings = run(cam, 200, "GREEN", {"turn": "RED"})                 # shared light GREEN, own arrow RED: violation
    assert ok and crossings[0]["light"] == "RED" and crossings[0]["lane"] == "turn"
    ok, crossings = run(cam, 200, "RED", {"turn": "GREEN"})                 # shared RED, own arrow GREEN: no violation
    assert not ok and crossings[0]["light"] == "GREEN" and "light is GREEN (lane 'turn' own light)" in crossings[0]["why"]
    ok, crossings = run(cam, 200, "RED", {"turn": "UNKNOWN"})               # own light unreadable: precision first, not reported
    assert not ok and crossings[0]["light"] == "UNKNOWN"
    ok, _ = run(cam, 200, "RED", {})                                        # own light missing from the frame: UNKNOWN as well
    assert not ok


def test_shared_lane_uses_light_roi_and_exempt_lane_never_reports():
    cam = camera(lanes({"roi": ARROW}))
    assert run(cam, 400, "RED", {"turn": "GREEN"})[0]                       # "go" is shared: RED
    assert not run(cam, 400, "GREEN", {"turn": "RED"})[0]                   # shared GREEN, the arrow of another lane does not matter
    cam = camera(lanes("exempt"))
    assert not run(cam, 200, "RED", {"turn": "RED"})[0]
    cam = camera(lanes("roi"))                                              # old spelling = shared
    assert run(cam, 200, "RED", {})[0] and not run(cam, 200, "GREEN", {})[0]


def light_frame(arrow=None, counter=None):
    f = np.full((720, 1280, 3), 90, np.uint8)
    if arrow:
        f[ARROW[1]:ARROW[3], ARROW[0]:ARROW[2]] = {"RED": (0, 0, 255), "GREEN": (0, 255, 0)}[arrow]
    if counter:
        f[0:22, 590:632] = {"RED": (0, 0, 255), "GREEN": (0, 255, 0)}[counter]
    return f


def pipeline(tmp_path, turn_signal):
    cfg = load_pipeline_config()
    cfg.evidence.dir = str(tmp_path / "ev")
    cam = camera(lanes(turn_signal))
    cam.light_roi = [590, 0, 632, 22]
    return Pipeline(FakeDetectors(lambda i: ([], [])), SharedState(), cfg, cam)


def test_pipeline_reads_each_lane_signal_logs_changes_draws_and_reports_status(tmp_path, capsys):
    pipe = pipeline(tmp_path, {"roi": ARROW})
    out = capsys.readouterr().out
    assert "lane 'turn': signal own light, roi [100, 5, 130, 20]" in out and "lane 'go': signal shared (light_roi)" in out
    r = pipe.process(light_frame(arrow="RED", counter="GREEN"))
    assert r.perception.light == "GREEN" and r.perception.lane_lights == {"turn": "RED"}
    log = capsys.readouterr().out
    assert "lane 'turn' signal - -> RED" in log and "lane 'go' signal - -> GREEN" in log
    st = pipe.red_light_status()["lane_signals"]
    assert st == {"turn": {"signal": "own", "roi": ARROW, "state": "RED"}, "go": {"signal": "shared", "roi": None, "state": "GREEN"}}
    pipe.process(light_frame(arrow="RED", counter="GREEN"))
    assert capsys.readouterr().out == ""                                    # no change, no log line
    pipe.process(light_frame(arrow="GREEN", counter="GREEN"))
    assert "lane 'turn' signal RED -> GREEN" in capsys.readouterr().out
    hud = pipe.process(light_frame(arrow="GREEN", counter="GREEN")).frame[145:170, 40:900]
    assert (hud != light_frame(arrow="GREEN", counter="GREEN")[145:170, 40:900]).any()          # the "LANE SIGNALS" line is painted


def test_exempt_lane_has_no_state_and_status_says_so(tmp_path):
    pipe = pipeline(tmp_path, "exempt")
    pipe.process(light_frame(counter="RED"))
    assert pipe.red_light_status()["lane_signals"]["turn"] == {"signal": "exempt", "roi": None, "state": None}
    assert pipe.process(light_frame(counter="RED")).perception.lane_lights == {}


def test_api_status_carries_the_lane_signals(tmp_path):
    from test_worker import make
    w, pipe = make()
    pipe.red_light_status = lambda: {"state": "active", "lane_signals": {"turn": {"signal": "own", "roi": ARROW, "state": "RED"}}}
    assert w.status()["red_light"]["lane_signals"]["turn"]["state"] == "RED"
    w.stop()
