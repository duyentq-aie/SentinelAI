"""Event comparison helpers + the slow perception-regression test.

Perception (tracks) is what must not change while rules are rewritten. The slow test re-runs the detectors+trackers on
test_video.mp4 (CPU, ~5-17 min) and compares every frame to tests/fixtures/tracks_test_video.jsonl. Rule behaviour is
tested offline from that fixture (tests/test_rules_offline.py), in seconds.
"""
import tempfile
from pathlib import Path

import pytest

from sentinel.characterize import compare_events, diff_events, read_frames
from sentinel.offline import load_tracks
from sentinel.perception import tracks_equal

REPO = Path(__file__).resolve().parent.parent
VIDEO = REPO / "test_video.mp4"
TRACKS = REPO / "tests" / "fixtures" / "tracks_test_video.jsonl"
WEIGHTS = ["helmet_best.pt", "plate_best.pt", "yolo11s.pt"]

needs_assets = pytest.mark.skipif(
    not (VIDEO.exists() and TRACKS.exists() and all((REPO / w).exists() for w in WEIGHTS)),
    reason="needs test_video.mp4, model weights and the tracks fixture")


def ev(frame, kind, **kw):
    return {"frame": frame, "kind": kind, **kw}


def test_compare_events_detects_drift():
    ref = [ev(10, "plate_capture", track="M_1")]
    assert compare_events(ref, [ev(13, "plate_capture", track="M_1")], tol=3) == []
    assert compare_events(ref, [ev(14, "plate_capture", track="M_1")], tol=3)   # too late
    assert compare_events(ref, [], tol=3)                                       # missing
    assert compare_events(ref, ref + [ev(50, "plate_capture", track="M_2")])    # extra


def test_compare_events_ignore_ids():
    ref = [ev(10, "label_first_seen", label="ID:M_128 NO-HELMET")]
    got = [ev(11, "label_first_seen", label="ID:M_129 NO-HELMET")]
    assert compare_events(ref, got, tol=3)
    assert compare_events(ref, got, tol=3, ignore_ids=True) == []
    assert compare_events(ref, [ev(11, "label_first_seen", label="ID:M_129 OVERLOAD")], tol=3, ignore_ids=True)


def test_diff_events_lists_missing_added_changed():
    a = [ev(10, "label_first_seen", label="ID:M_1 NO-HELMET"), ev(20, "label_first_seen", label="ID:C_2 WRONG-WAY"),
         ev(30, "plate_capture", track="M_1"), ev(40, "label_first_seen", label="ID:M_5 OVERLOAD")]
    b = [ev(10, "label_first_seen", label="ID:M_1 NO-HELMET"),                       # identical
         ev(25, "label_first_seen", label="ID:M_5 OVERLOAD & NO-HELMET"),            # reason changed
         ev(30, "plate_capture", track="M_1"), ev(50, "label_first_seen", label="ID:C_9 RED-LIGHT")]   # C_2 lost, C_9 new
    d = diff_events(a, b, tol=3)
    assert [e["label"] for e in d["missing"]] == ["ID:C_2 WRONG-WAY"]
    assert [e["label"] for e in d["added"]] == ["ID:C_9 RED-LIGHT"]
    assert len(d["changed"]) == 1 and d["changed"][0]["entity"] == "M_5"
    assert any("ID:M_5 OVERLOAD'" in w for w in d["changed"][0]["why"])
    assert diff_events(a, a) == {"missing": [], "added": [], "changed": []}


def test_diff_events_frame_move_beyond_tolerance_is_a_change():
    a = [ev(10, "label_first_seen", label="ID:M_1 NO-HELMET")]
    assert diff_events(a, [ev(12, "label_first_seen", label="ID:M_1 NO-HELMET")], tol=3)["changed"] == []
    assert diff_events(a, [ev(20, "label_first_seen", label="ID:M_1 NO-HELMET")], tol=3)["changed"]


@pytest.mark.slow
@needs_assets
def test_perception_unchanged_on_cpu():
    from sentinel.config import load_camera_config, load_pipeline_config
    from sentinel.pipeline import Detectors, Pipeline, SharedState
    meta, expected = load_tracks(TRACKS)
    cfg = load_pipeline_config(root=REPO, device="cpu")
    camera = load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
    bad, n = [], 0
    with tempfile.TemporaryDirectory() as ev_dir:
        cfg.evidence.dir = ev_dir
        pipe = Pipeline(Detectors(cfg), SharedState(), cfg, camera)
        for i, frame in enumerate(read_frames(VIDEO)):
            n = i + 1
            got = pipe.process(frame).perception
            if i >= len(expected) or not (tracks_equal(got.coco, expected[i].coco) and tracks_equal(got.helmet, expected[i].helmet)
                                          and got.light == expected[i].light):
                bad.append(i)
    assert n == len(expected), f"video has {n} frames, fixture {len(expected)}"
    assert not bad, f"perception differs from the fixture on {len(bad)} frames, first: {bad[:10]}"
