"""Red-light rule 'stop_line_crossing' on synthetic tracks (no video, no models)."""
import json

import numpy as np
import pytest
import yaml

from sentinel.config import REPO, CameraConfig, Lane, PipelineConfig, RedLightRule, load_camera_config, load_pipeline_config
from sentinel.evidence import types_of
from sentinel.perception import Perception, Track
from sentinel.rules.stopline import StopLineCrossing, join_reason
from sentinel.rules.windowed import make_rules

# Stop line y = 300 (x 100..500); vehicles go UP the image (decreasing y): before the line y > 300, beyond it y < 300.
LINE = [[100, 300], [500, 300]]
UP = [[300, 400], [300, 200]]
LEFT = Lane("left-turn", [[100, 100], [300, 100], [300, 500], [100, 500]], "exempt")
THROUGH = Lane("through", [[300, 100], [500, 100], [500, 500], [300, 500]], "roi")
APPROACH = list(range(420, 280, -5))        # bottom-centre y from 420 to 285: crosses the line (dead zone 296..304)


def camera(lanes=None, stop_line=LINE, direction=UP, light_roi=(590, 0, 632, 22)):
    return CameraConfig(camera_id="t", source="t.mp4", violation_zone=[[0, 0], [10, 0], [10, 10]],
                        stop_line=stop_line, direction=direction, lanes=lanes, light_roi=light_roi).validate()


def rules(cam=None, fps=30, **red):
    cfg = load_pipeline_config().rules
    for k, v in red.items():
        setattr(cfg.red_light, k, v)
    return make_rules(cfg, cam or camera([LEFT, THROUGH]), fps)


def frame(y, x=400, tid=1, light="RED", extra=()):
    return Perception(helmet=[Track(tid, "motorcycle", (x - 30, y - 100, x + 30, y), 0.9), *extra], light=light)


def run(r, ys, x=400, light="RED", tid=1):
    """Feed one motorcycle at bottom-centre (x, y) per frame; -> [(reason, violator)] after each frame."""
    out = []
    for y in ys:
        v = {v.id: v for v in r.step(frame(y, x, tid, light)).vehicles}[f"M_{tid}"]
        out.append((v.reason, v.violator))
    return out


def flags(res):
    return [violator for _, violator in res]


def test_crossing_the_line_in_the_direction_of_travel_while_red_is_a_violation():
    res = run(rules(), APPROACH)
    first = flags(res).index(True)
    assert APPROACH[first] <= 296 < APPROACH[first - 1]            # flagged at the first observation beyond the dead zone
    assert not any(flags(res)[:first]) and res[first][0] == "RED-LIGHT RUNNER"


def test_crossing_on_green_is_not_a_violation():
    assert not any(flags(run(rules(), APPROACH, light="GREEN")))
    assert not any(flags(run(rules(), APPROACH, light="UNKNOWN")))


def test_standing_still_before_on_or_just_over_the_line_while_red_is_not_a_violation():
    for y in (420, 310, 300, 298, 302, 250):
        assert not any(flags(run(rules(), [y] * 90))), y


def test_travelling_against_the_direction_is_not_a_violation():
    assert not any(flags(run(rules(), list(reversed(APPROACH)))))


def test_a_vehicle_already_beyond_the_line_when_the_light_turns_red_is_not_a_violation():
    r = rules()
    res = run(r, list(range(420, 250, -5)), light="GREEN") + run(r, list(range(250, 150, -5)), light="RED")
    assert not any(flags(res))                                       # crossed on green, still moving when it turns red
    assert not any(flags(run(rules(), [250] * 5 + list(range(250, 150, -5)))))   # first seen beyond the line, light already red


def test_a_crossing_in_an_exempt_lane_is_not_a_violation_but_in_the_roi_lane_it_is():
    assert not any(flags(run(rules(), APPROACH, x=200)))             # left-turn lane (exempt)
    assert any(flags(run(rules(), APPROACH, x=400)))                 # through lane (shared signal)


def test_a_crossing_outside_every_lane_is_not_reported_but_counted():
    r = rules(camera([Lane("narrow", [[380, 100], [420, 100], [420, 500], [380, 500]], "roi")]))
    assert not any(flags(run(r, APPROACH, x=450)))
    assert r.stop_line_rule.unassigned_crossings == 1


def test_a_crossing_beyond_the_ends_of_the_segment_is_ignored():
    assert not any(flags(run(rules(camera(None)), APPROACH, x=800)))


def test_jitter_around_the_line_gives_at_most_one_event():
    r = rules()
    assert not any(flags(run(r, [302, 298, 303, 297, 301, 299] * 10)))       # inside the dead zone: no crossing at all
    r = rules()
    res = run(r, [420] + [310, 290] * 40)                                   # large jitter: many crossings while RED
    rising = sum(1 for a, b in zip([False] + flags(res), flags(res)) if b and not a)
    assert rising == 1 and sum(flags(res)) == r.stop_line_rule._hold         # one verdict per track, held hold_s, never re-armed


def test_a_jittering_track_emits_only_the_red_light_reason():
    res = run(rules(), [420] + [310, 290] * 40)
    assert {reason for reason, violator in res if violator} == {"RED-LIGHT RUNNER"} and types_of("RED-LIGHT RUNNER") == ["red_light"]


def test_the_verdict_is_held_for_hold_s_seconds_at_any_frame_rate():
    for fps in (30, 10):
        r = rules(fps=fps)
        res = run(r, [420] + [280] * 80)
        assert sum(flags(res)) == round(r.stop_line_rule.cfg.hold_s * fps)


def test_cars_are_not_judged_and_reasons_are_combined_with_other_violations():
    r = rules()
    for y in APPROACH:
        res = r.step(Perception(coco=[Track(7, "car", (370, y - 100, 430, y), 0.9)], light="RED"))
    assert not any(v.violator for v in res.vehicles)
    assert types_of("NO-HELMET & RED-LIGHT RUNNER") == ["no_helmet", "red_light"]
    assert join_reason("NO-HELMET", "RED-LIGHT RUNNER") == "NO-HELMET & RED-LIGHT RUNNER"
    assert join_reason("", "RED-LIGHT RUNNER") == "RED-LIGHT RUNNER"
    assert join_reason("RED-LIGHT RUNNER", "RED-LIGHT RUNNER") == "RED-LIGHT RUNNER"


def test_direction_can_be_any_orientation_and_the_normal_follows_it():
    cam = camera(None, stop_line=[[300, 100], [300, 500]], direction=[[500, 300], [100, 300]])   # vertical line, travel to the LEFT
    r = StopLineCrossing(cam, RedLightRule(), log=lambda m: None)
    assert r.signed_distance((250, 300)) > 0 > r.signed_distance((350, 300))


def test_missing_geometry_or_lanes_is_logged_never_silent():
    logs = []
    r = StopLineCrossing(camera(None, stop_line=None, direction=None), RedLightRule(), log=logs.append)
    assert not r.active and any("INACTIVE" in m for m in logs)
    logs.clear()
    r = StopLineCrossing(camera(None), RedLightRule(), log=logs.append)
    assert r.active and any("no camera.lanes" in m and "WHOLE stop line" in m for m in logs)
    logs.clear()
    StopLineCrossing(camera([THROUGH]), RedLightRule(), log=logs.append)
    assert logs == []
    StopLineCrossing(camera(None, direction=[[100, 300], [500, 300]]), RedLightRule(), log=logs.append)
    assert any("parallel" in m for m in logs)
    assert any(flags(run(rules(camera(None)), APPROACH, x=200)))     # no lanes: the shared signal applies to the whole line


def test_lanes_are_read_from_the_camera_yaml_and_validated(tmp_path):
    base = {"camera_id": "c", "source": "c.mp4", "violation_zone": [[0, 0], [5, 0], [5, 5]], "stop_line": LINE, "direction": UP,
            "lanes": [{"name": "turn", "polygon": [[0, 0], [10, 0], [10, 10]], "signal": "exempt"}, {"name": "go", "polygon": [[20, 0], [30, 0], [30, 10]]}]}
    f = tmp_path / "cam.yaml"
    f.write_text(yaml.safe_dump(base), encoding="utf-8")
    cam = load_camera_config(f)
    assert [(l.name, l.signal) for l in cam.lanes] == [("turn", "exempt"), ("go", "roi")] and isinstance(cam.lanes[0], Lane)
    bad_lanes = ([{"name": "a", "polygon": [[0, 0], [1, 1]]}],
                 [{"name": "a", "polygon": [[0, 0], [1, 0], [1, 1]], "signal": "green"}],
                 [{"name": "a", "polygon": [[0, 0], [1, 0], [1, 1]]}, {"name": "a", "polygon": [[0, 0], [1, 0], [1, 1]]}])
    for lanes in bad_lanes:
        f.write_text(yaml.safe_dump({**base, "lanes": lanes}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_camera_config(f)
    f.write_text(yaml.safe_dump({k: v for k, v in base.items() if k != "lanes"}), encoding="utf-8")
    assert load_camera_config(f).lanes is None


def test_mode_is_stop_line_crossing_by_default_and_legacy_zone_stays_selectable(tmp_path):
    assert load_pipeline_config().rules.red_light.mode == "stop_line_crossing"
    cfg = load_pipeline_config().rules
    cfg.red_light.mode = "legacy_zone"
    cam = load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
    assert make_rules(cfg, cam).stop_line_rule is None
    cfg.red_light.mode = "whatever"
    with pytest.raises(ValueError):
        PipelineConfig(rules=cfg).validate()
