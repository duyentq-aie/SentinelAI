"""Unit tests of the legacy (v0) rules on synthetic perceptions: pins today's behaviour, quirks included."""
from sentinel.config import REPO, load_camera_config, load_pipeline_config
from sentinel.perception import Perception, Track
from sentinel.rules.legacy import LegacyRules, vehicle_label, violation_labels

CAM = load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")


def rules(wrong_way=False):
    cfg = load_pipeline_config().rules
    cfg.red_light.mode = "legacy_zone"   # these tests pin the v0 zone rule; the default mode is stop_line_crossing (tests/test_stopline_rule.py)
    cfg.wrong_way.enabled = wrong_way                # default config has the legacy wrong-way rule off
    return LegacyRules(cfg, CAM)


def moto(tid, box=(100, 100, 200, 300)):
    return Track(tid, "motorcycle", box, 0.9)


def head(kind, box=(130, 110, 170, 150)):
    return Track(0, kind, box, 0.9)


def vehicles(result):
    return {v.id: v for v in result.vehicles}


def test_no_helmet_needs_five_frames():
    r = rules()
    p = Perception(helmet=[moto(1), head("no-helmet")])
    for _ in range(4):
        assert not vehicles(r.step(p))["M_1"].violator            # counter 1..4
    assert vehicles(r.step(p))["M_1"].reason == "NO-HELMET"       # 5th frame


def test_no_helmet_counter_drops_below_threshold_after_a_clean_frame():
    r = rules()
    bad = Perception(helmet=[moto(1), head("no-helmet")])
    for _ in range(5):
        r.step(bad)
    res = r.step(Perception(helmet=[moto(1), head("helmet")]))    # counter 5 -> 4
    assert not vehicles(res)["M_1"].violator


def test_overload_counts_heads_not_people():
    r = rules()
    p = Perception(helmet=[moto(2), head("helmet", (110, 110, 130, 130)), head("helmet", (140, 110, 160, 130)),
                           head("helmet", (170, 110, 190, 130))])
    res = r.step(p)
    assert vehicles(res)["M_2"].reason == "OVERLOAD"
    assert len(res.heads) == 3


def test_overload_and_no_helmet_combine():
    r = rules()
    p = Perception(helmet=[moto(3), head("no-helmet", (110, 110, 130, 130)), head("helmet", (140, 110, 160, 130)),
                           head("helmet", (170, 110, 190, 130))])
    for _ in range(5):
        res = r.step(p)
    assert vehicles(res)["M_3"].reason == "OVERLOAD & NO-HELMET"


def test_head_outside_motorcycle_is_not_attached():
    res = rules().step(Perception(helmet=[moto(4), head("no-helmet", (500, 500, 540, 540))]))
    assert res.heads == [] and not vehicles(res)["M_4"].violator


def car_at(y):
    return Perception(coco=[Track(7, "car", (400, y, 500, y + 60), 0.9)])


def test_wrong_way_is_off_by_default():
    r = rules()
    for i in range(30):
        res = r.step(car_at(600 - 5 * i))
    assert not vehicles(res)["C_7"].violator


def test_wrong_way_car_moving_up_the_image():
    r = rules(True)
    res = None
    for i in range(16):                                           # 16 positions, moving up 5 px/frame -> dy = -75 after 16
        res = r.step(car_at(600 - 5 * i))
    assert vehicles(res)["C_7"].reason == "WRONG-WAY"


def test_car_moving_down_is_fine():
    r = rules()
    for i in range(30):
        res = r.step(car_at(100 + 5 * i))
    assert not vehicles(res)["C_7"].violator


def test_wrong_way_needs_15_positions():
    r = rules(True)
    for i in range(14):
        res = r.step(car_at(600 - 20 * i))
    assert not vehicles(res)["C_7"].violator


def test_coco_motorcycle_is_ignored_by_the_coco_branch():
    res = rules().step(Perception(coco=[Track(9, "motorcycle", (100, 100, 200, 300), 0.9)]))
    assert res.vehicles == []


def test_red_light_zone_flags_anything_inside_including_stopped_vehicles():
    r = rules()
    # bottom-centre (450, 300) is inside the zone [[250,250],[750,250],[820,380],[150,380]]
    stopped = Perception(coco=[Track(5, "car", (400, 200, 500, 300), 0.9)], light="RED")
    for _ in range(3):
        res = r.step(stopped)
    assert vehicles(res)["C_5"].reason == "RED-LIGHT RUNNER"      # legacy quirk: a car waiting at the light is a "runner"
    assert vehicle_label(vehicles(res)["C_5"]) == "ID:C_5 RED-LIGHT"


def test_red_light_needs_a_red_light():
    for state in ("GREEN", "UNKNOWN"):
        res = rules().step(Perception(coco=[Track(5, "car", (400, 200, 500, 300), 0.9)], light=state))
        assert not vehicles(res)["C_5"].violator


def test_labels_match_what_is_drawn():
    r = rules()
    p = Perception(helmet=[moto(1), head("no-helmet")], coco=[Track(5, "car", (400, 200, 500, 300), 0.9)], light="RED")
    for _ in range(5):
        res = r.step(p)
    assert violation_labels(res) == ["NO-HELMET", "ID:C_5 RED-LIGHT", "ID:M_1 NO-HELMET"]


def test_reset_clears_counters_and_history():
    r = rules()
    for _ in range(5):
        r.step(Perception(helmet=[moto(1), head("no-helmet")]))
    r.reset()
    assert not vehicles(r.step(Perception(helmet=[moto(1), head("no-helmet")])))["M_1"].violator
