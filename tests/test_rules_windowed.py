"""Windowed (voting) no-helmet / overload rules on synthetic tracks: glitches, occlusion, ratio, window expiry, independence of tracks."""
import pytest

from sentinel.config import REPO, load_camera_config, load_pipeline_config
from sentinel.perception import Perception, Track
from sentinel.rules.legacy import LegacyRules
from sentinel.rules.windowed import PURGE_AFTER, PURGE_EVERY, WindowedRules, make_rules

CAM = load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
MOTO = (100, 100, 400, 500)


def rules(**kw):
    cfg = load_pipeline_config().rules
    for k, v in kw.items():
        scope, name = k.split("__")
        setattr(getattr(cfg, scope), name, v)
    return WindowedRules(cfg, CAM)


def frame(heads=(), tid=1):
    """heads: iterable of 'helmet' / 'no-helmet'; each placed inside the motorcycle box."""
    hs = [Track(0, kind, (120 + 40 * i, 120, 150 + 40 * i, 160), 0.9) for i, kind in enumerate(heads)]
    return Perception(helmet=[Track(tid, "motorcycle", MOTO, 0.9)] + hs)


def verdicts(r, seq, tid=1):
    """Run a sequence of head-lists; return the reason ('' = no violation) after every frame."""
    out = []
    for heads in seq:
        res = r.step(frame(heads, tid))
        out.append({v.id: v for v in res.vehicles}[f"M_{tid}"].reason)
    return out


OK, BAD, NONE = ("helmet",), ("no-helmet",), ()


def test_steady_no_helmet_triggers_at_the_fifth_vote():
    v = verdicts(rules(), [BAD] * 8)
    assert v[:4] == [""] * 4 and v[4:] == ["NO-HELMET"] * 4


def test_one_or_two_frame_glitches_never_trigger():
    seq = ([OK] * 6 + [BAD] + [OK] * 6 + [BAD, BAD] + [OK] * 10) * 3
    assert set(verdicts(rules(), seq)) == {""}


def test_temporary_occlusion_does_not_dilute_the_vote_and_does_not_reset_it():
    seq = [BAD] * 3 + [NONE] * 6 + [BAD] * 3
    v = verdicts(rules(), seq)
    assert v[:10] == [""] * 10                            # 3 votes, then 6 occluded frames (not observations): nothing yet
    assert v[10] == "NO-HELMET"                           # votes 4 and 5 arrive at frames 9 and 10: the 3 earlier votes still count


def test_verdict_holds_through_a_later_occlusion():
    v = verdicts(rules(), [BAD] * 6 + [NONE] * 20)
    assert v[5] == "NO-HELMET" and set(v[6:]) == {"NO-HELMET"}


def test_ratio_must_be_reached_not_just_the_vote_count():
    # 1 bad in 3 -> ratio .33 < .5 : never flagged even after 15 frames (5 votes inside the 15-frame window)
    v = verdicts(rules(), [BAD, OK, OK] * 6)
    assert set(v) == {""}
    # 2 bad in 3 -> ratio .67 >= .5 -> flagged once 5 votes are in
    v = verdicts(rules(), [BAD, BAD, OK] * 6)
    assert v[6] == "NO-HELMET" and v.count("") == 6


def test_old_votes_expire_when_the_rider_puts_the_helmet_on():
    v = verdicts(rules(), [BAD] * 6 + [OK] * 15)
    assert v[5] == "NO-HELMET"
    first_clear = next(i for i in range(5, len(v)) if v[i] == "")        # first "no violation" AFTER the flag was raised
    assert v[-1] == "" and first_clear > 5                # once enough helmeted observations arrive the vote ratio falls below r


def test_overload_needs_sustained_three_heads():
    three = ("helmet", "helmet", "helmet")
    assert set(verdicts(rules(), [OK, three, OK, three, OK] * 4)) == {""}           # momentary 3rd head
    v = verdicts(rules(), [three] * 7)
    assert v[:4] == [""] * 4 and v[4] == "OVERLOAD"


def test_two_riders_is_not_overload():
    assert set(verdicts(rules(), [("helmet", "helmet")] * 20)) == {""}


def test_overload_and_no_helmet_combine():
    seq = [("no-helmet", "helmet", "helmet")] * 6
    assert verdicts(rules(), seq)[-1] == "OVERLOAD & NO-HELMET"


def test_tracks_are_independent():
    r = rules()
    for _ in range(6):
        res = r.step(Perception(helmet=[Track(1, "motorcycle", MOTO, .9), Track(0, "no-helmet", (120, 120, 150, 160), .9),
                                        Track(2, "motorcycle", (600, 100, 900, 500), .9), Track(0, "helmet", (620, 120, 650, 160), .9)]))
    got = {v.id: v.reason for v in res.vehicles}
    assert got == {"M_1": "NO-HELMET", "M_2": ""}


def test_parameters_come_from_config():
    r = rules(helmet__min_votes=3, helmet__min_ratio=0.5, helmet__window=5)
    assert verdicts(r, [BAD] * 3)[2] == "NO-HELMET"


def test_window_parameters_bound_the_memory_of_votes():
    r = rules(helmet__window=6, helmet__min_votes=4, helmet__min_ratio=0.5, overload__window=6, overload__min_votes=4)
    v = verdicts(r, [BAD] * 4 + [OK] * 6)
    assert v[3] == "NO-HELMET" and v[-1] == ""            # 6-observation window forgets the 4 votes after 6 helmeted frames


def test_reset_forgets_everything():
    r = rules()
    verdicts(r, [BAD] * 6)
    r.reset()
    assert verdicts(r, [BAD] * 4)[-1] == ""


def test_windows_of_vanished_tracks_are_purged():
    r = rules()
    verdicts(r, [BAD] * 3, tid=77)
    for _ in range(PURGE_EVERY + PURGE_AFTER):
        r.step(Perception())
    assert 77 not in r._obs and 77 not in r._last_seen


def test_default_engine_is_windowed_and_legacy_is_selectable():
    cfg = load_pipeline_config().rules
    assert isinstance(make_rules(cfg, CAM), WindowedRules)
    cfg.engine = "legacy"
    assert type(make_rules(cfg, CAM)) is LegacyRules
    cfg.engine = "nope"
    with pytest.raises(ValueError):
        make_rules(cfg, CAM)


def test_invalid_window_config_is_rejected(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("rules:\n  helmet: {window: 4, min_votes: 5}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pipeline_config(f)
    f.write_text("rules:\n  engine: weird\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pipeline_config(f)


# ---- the M_176 pattern (EXP-005 / tools/rule_trace.py): the 5th vote lands exactly on ratio == r, the next clean frame drops it below
M176 = [BAD, BAD, OK, OK, OK, BAD, BAD, OK, NONE, NONE, OK, BAD] + [OK] * 8


def flagged_frames(r, seq):
    return [i for i, v in enumerate(verdicts(r, seq)) if v]


def test_borderline_vote_creates_a_one_frame_verdict_by_default():
    assert flagged_frames(rules(), M176) == [11]                  # 5 votes / 10 observations = 0.5 >= r at frame 11, 5/11 at frame 12


def test_hysteresis_keeps_the_flag_until_the_ratio_falls_below_exit_ratio():
    r = rules(helmet__exit_ratio=0.3)
    assert flagged_frames(r, M176) == [11, 12, 13, 14, 15, 16]    # ratio .50 .46 .42 .39 .36 .33 ... then < .3 at frame 17


def test_hysteresis_does_not_change_when_a_track_is_first_flagged():
    base = verdicts(rules(), [BAD] * 8 + [OK] * 20)
    hyst = verdicts(rules(helmet__exit_ratio=0.3), [BAD] * 8 + [OK] * 20)
    assert [i for i, v in enumerate(base) if v][0] == [i for i, v in enumerate(hyst) if v][0] == 4
    assert sum(1 for v in hyst if v) >= sum(1 for v in base if v)


def test_exit_ratio_must_not_exceed_min_ratio(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("rules:\n  helmet: {exit_ratio: 0.9}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exit_ratio"):
        load_pipeline_config(f)
    f.write_text("rules:\n  overload: {exit_ratio: 0.3}\n", encoding="utf-8")
    assert load_pipeline_config(f).rules.overload.exit_ratio == 0.3


def test_reset_clears_the_hysteresis_state():
    r = rules(helmet__exit_ratio=0.3)
    verdicts(r, [BAD] * 6)
    r.reset()
    assert not r._on and verdicts(r, [OK] * 3) == [""] * 3
