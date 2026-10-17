"""Replay the legacy rules from the recorded tracks (no models, no video): fast, deterministic.

The v0 baseline events are a COMPARISON snapshot (see tools/diff_events.py). The legacy rules are a verbatim port, so they must
reproduce it exactly; the new rules of steps 3.2/3.3 are expected to differ and are judged by their own unit tests + the diff report.
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
BASELINE = FIX / "baseline_events_v0_cpu.json"

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


@pytest.mark.parametrize("name", ["video", "models"])
def test_fixture_matches_local_assets_when_present(tracks, name):
    """A stale fixture (other video / other weights) would make every rule test meaningless."""
    meta, _ = tracks
    items = {"video": {"test_video.mp4": meta["video"]["sha256"]},
             "models": {m["file"]: m["sha256"] for m in meta["models"].values()}}[name]
    for fname, sha in items.items():
        f = REPO / fname
        if not f.exists():
            pytest.skip(f"{fname} not available locally")
        assert hashlib.sha256(f.read_bytes()).hexdigest() == sha, f"{fname} differs from the one the fixture was recorded with"


def test_legacy_rules_reproduce_the_v0_label_events_exactly(tracks):
    _, perceptions = tracks
    cfg, cam = load_pipeline_config(), load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
    t0 = time.perf_counter()
    got = run_rules(perceptions, LegacyRules(cfg.rules, cam), violation_labels)
    assert time.perf_counter() - t0 < 30                                   # offline rule replay budget
    base = [e for e in json.loads(BASELINE.read_text(encoding="utf-8"))["events"] if e["kind"] == "label_first_seen"]
    d = diff_events(base, got, tol=0)
    assert d == {"missing": [], "added": [], "changed": []}, d


def test_baseline_snapshot_shape():
    d = json.loads(BASELINE.read_text(encoding="utf-8"))
    kinds = [e["kind"] for e in d["events"]]
    assert (kinds.count("plate_capture"), kinds.count("label_first_seen")) == (20, 23) and d["video_sha256"]
