"""Characterization tests for the P2 step-1 refactor.

The refactored pipeline must reproduce the violation events of the original app.py:
  * vs the GPU snapshot experiments/EXP-001-baseline/events.json, within +/-3 frames, track ids ignored
    (Ultralytics assigns different ids on CPU vs GPU because detections differ numerically);
  * vs the CPU snapshot of the ORIGINAL code (events_cpu_original.json, recorded by tools/record_events.py
    from git tag v0-baseline): identical, track ids included, tolerance 0 frames.
Both run the full test video on CPU, so they are slow (marked `slow`; ~5 min, shared via one fixture).
"""
import json
import tempfile
from pathlib import Path

import pytest

from sentinel.characterize import collect_events, compare_events, read_frames

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments" / "EXP-001-baseline"
VIDEO = REPO / "test_video.mp4"
GPU_SNAPSHOT = EXP / "events.json"
CPU_SNAPSHOT = EXP / "events_cpu_original.json"
WEIGHTS = ["helmet_best.pt", "plate_best.pt", "yolo11s.pt"]

needs_assets = pytest.mark.skipif(
    not (VIDEO.exists() and GPU_SNAPSHOT.exists() and all((REPO / w).exists() for w in WEIGHTS)),
    reason="needs test_video.mp4, model weights and the EXP-001 events snapshot")


def test_compare_events_detects_drift():
    ref = [{"frame": 10, "kind": "plate_capture", "track": "M_1"}]
    assert compare_events(ref, [{"frame": 13, "kind": "plate_capture", "track": "M_1"}], tol=3) == []
    assert compare_events(ref, [{"frame": 14, "kind": "plate_capture", "track": "M_1"}], tol=3)   # too late
    assert compare_events(ref, [], tol=3)                                                        # missing
    assert compare_events(ref, ref + [{"frame": 50, "kind": "plate_capture", "track": "M_2"}])   # extra


def test_compare_events_ignore_ids():
    ref = [{"frame": 10, "kind": "label_first_seen", "label": "ID:M_128 NO-HELMET"}]
    got = [{"frame": 11, "kind": "label_first_seen", "label": "ID:M_129 NO-HELMET"}]
    assert compare_events(ref, got, tol=3)                                   # ids differ -> mismatch
    assert compare_events(ref, got, tol=3, ignore_ids=True) == []            # same type/time -> ok
    other = [{"frame": 11, "kind": "label_first_seen", "label": "ID:M_129 OVERLOAD"}]
    assert compare_events(ref, other, tol=3, ignore_ids=True)                # different violation -> mismatch


@pytest.fixture(scope="module")
def refactored_cpu_events():
    from sentinel.pipeline import Detectors, Pipeline, SharedState
    with tempfile.TemporaryDirectory() as ev_dir:
        pipe = Pipeline(Detectors(weights_dir=REPO, device="cpu"), SharedState(), evidence_dir=ev_dir)
        return collect_events(pipe, read_frames(VIDEO))


@pytest.mark.slow
@needs_assets
def test_matches_gpu_baseline_within_3_frames_ignoring_ids(refactored_cpu_events):
    ref = json.loads(GPU_SNAPSHOT.read_text(encoding="utf-8"))["events"]
    problems = compare_events(ref, refactored_cpu_events, tol=3, ignore_ids=True)
    assert not problems, "\n".join(problems)


@pytest.mark.slow
@needs_assets
@pytest.mark.skipif(not CPU_SNAPSHOT.exists(), reason="run tools/record_events.py --impl original --device cpu first")
def test_identical_to_original_code_on_same_device(refactored_cpu_events):
    ref = json.loads(CPU_SNAPSHOT.read_text(encoding="utf-8"))["events"]
    problems = compare_events(ref, refactored_cpu_events, tol=0)
    assert not problems, "\n".join(problems)
