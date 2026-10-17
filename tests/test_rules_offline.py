"""Replay the legacy rules from the recorded tracks (no models, no video): fast, deterministic.

The baseline events are COMPARISON snapshots (see tools/diff_events.py). The legacy rules are a verbatim port, so they must reproduce
the label events of both snapshots exactly: v0 (recorded on the old overlay-contaminated perception, fixture *_v0_overlay) and v1_clean
(perception on raw frames, since 3.0b). The new rules of steps 3.2/3.3 are expected to differ and are judged by their own unit tests
and the diff report.
"""
import hashlib
import json
import time
from pathlib import Path

import pytest

from sentinel.characterize import diff_events
from sentinel.config import load_camera_config, load_pipeline_config
from sentinel.offline import load_tracks, run_rules
from sentinel.rules.legacy import LegacyRules, violation_labels

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tests" / "fixtures"
TRACKS = FIX / "tracks_test_video.jsonl"
TRACKS_V0 = FIX / "tracks_test_video_v0_overlay.jsonl"
BASELINE_V0 = FIX / "baseline_events_v0_cpu.json"
BASELINE = FIX / "baseline_events_v1_clean.json"

pytestmark = pytest.mark.skipif(not TRACKS.exists(), reason="tracks fixture not recorded (tools/record_tracks.py)")


@pytest.fixture(scope="module")
def tracks():
    return load_tracks(TRACKS)


def test_fixture_metadata_is_complete(tracks):
    meta, perceptions = tracks
    assert meta["video"]["file"] == "test_video.mp4" and len(meta["video"]["sha256"]) == 64
    assert set(meta["models"]) == {"coco", "helmet", "plate"} and all(len(m["sha256"]) == 64 for m in meta["models"].values())
    assert meta["track_fields"] == ["id", "cls", "x1", "y1", "x2", "y2", "conf", "state"]
    assert len(perceptions) == 787


@pytest.fixture(scope="module", params=[TRACKS, TRACKS_V0], ids=["clean", "v0_overlay"])
def any_tracks(request):
    return load_tracks(request.param)


@pytest.mark.parametrize("name", ["video", "models"])
def test_fixture_matches_local_assets_when_present(any_tracks, name):
    """A stale fixture (other video / other weights) would make every rule test meaningless."""
    meta, _ = any_tracks
    items = {"video": {"test_video.mp4": meta["video"]["sha256"]},
             "models": {m["file"]: m["sha256"] for m in meta["models"].values()}}[name]
    for fname, sha in items.items():
        f = REPO / fname
        if not f.exists():
            pytest.skip(f"{fname} not available locally")
        assert hashlib.sha256(f.read_bytes()).hexdigest() == sha, f"{fname} differs from the one the fixture was recorded with"


def _replay(perceptions, wrong_way=True):
    """wrong_way=True reproduces baselines v0/v1 (they were recorded with the legacy wrong-way rule on)."""
    cfg, cam = load_pipeline_config(), load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
    cfg.rules.wrong_way.enabled = wrong_way
    t0 = time.perf_counter()
    got = run_rules(perceptions, LegacyRules(cfg.rules, cam), violation_labels)
    assert time.perf_counter() - t0 < 30                                   # offline rule replay budget
    return got


def _labels(path):
    return [e for e in json.loads(path.read_text(encoding="utf-8"))["events"] if e["kind"] == "label_first_seen"]


def test_legacy_rules_reproduce_the_v1_clean_label_events_exactly(tracks):
    d = diff_events(_labels(BASELINE), _replay(tracks[1]), tol=0)
    assert d == {"missing": [], "added": [], "changed": []}, d


@pytest.mark.skipif(not TRACKS_V0.exists(), reason="v0 overlay fixture missing")
def test_legacy_rules_reproduce_the_v0_label_events_exactly_on_the_old_fixture():
    d = diff_events(_labels(BASELINE_V0), _replay(load_tracks(TRACKS_V0)[1]), tol=0)
    assert d == {"missing": [], "added": [], "changed": []}, d


def test_label_events_are_the_same_in_v0_and_v1_clean():
    """The 3.0b perception fix changed plate captures only; the 23 violation labels are unchanged."""
    assert diff_events(_labels(BASELINE_V0), _labels(BASELINE), tol=0) == {"missing": [], "added": [], "changed": []}


def test_baseline_snapshot_shapes():
    for path, expected in ((BASELINE_V0, (20, 23)), (BASELINE, (21, 23))):
        d = json.loads(path.read_text(encoding="utf-8"))
        kinds = [e["kind"] for e in d["events"]]
        assert (kinds.count("plate_capture"), kinds.count("label_first_seen")) == expected and d["video_sha256"], path.name
