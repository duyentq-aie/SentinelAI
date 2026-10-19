"""Durations in the config are seconds / ratios; frame counts come from the frame rate the pipeline sees. At 30 FPS they are the old constants."""
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from sentinel.config import REPO, EvidenceCfg, PlateCfg, load_camera_config, load_pipeline_config
from sentinel.evidence import EvidenceRecorder
from sentinel.perception import Perception, Track
from sentinel.pipeline import Detectors
from sentinel.rules.legacy import Vehicle
from sentinel.rules.windowed import PURGE_AFTER_S, PURGE_EVERY_S, WindowedRules
from sentinel.timebase import to_frames, votes_needed, window_votes

sys.path.insert(0, str(REPO / "tools"))
import convert_config_to_seconds as conv  # noqa: E402

CAM = load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
MOTO = (100, 100, 400, 500)


def cfg():
    return load_pipeline_config()


# ---- arithmetic
def test_to_frames_and_votes_are_round_half_up_and_exact_at_the_reference_rate():
    assert [to_frames(s, 30) for s in (0.5, 1.0, 3.0, 0.1, 5 / 30, 10.0)] == [15, 30, 90, 3, 5, 300]
    assert [to_frames(s, 10) for s in (0.5, 1.0, 3.0, 0.1, 10.0)] == [5, 10, 30, 1, 100]
    assert to_frames(0.0, 10) == 1 and to_frames(0.0, 10, minimum=0) == 0                      # a zero duration never becomes "no frames" by accident
    assert (votes_needed(1 / 3, 15), votes_needed(0.333333, 15), votes_needed(1 / 3, 5), votes_needed(1 / 3, 30), votes_needed(1.0, 4)) == (5, 5, 2, 10, 4)


def test_default_config_gives_exactly_the_old_frame_counts_at_30_fps():
    c = cfg()
    assert window_votes(c.rules.helmet, 30) == (15, 5) and window_votes(c.rules.overload, 30) == (15, 5)
    r = WindowedRules(c.rules, CAM, fps=30)
    assert (r._wh, r._kh, r._wo, r._ko) == (15, 5, 15, 5)
    assert (r._violation_frames, r._history_len, r._min_history, r._purge_every, r._purge_after) == (5, 30, 15, 300, 300)
    rec = EvidenceRecorder(c.evidence, lambda crop: [], "v", "c", fps=30)
    assert (rec._close_after, rec._box_every, rec._every, rec._max_attempts, rec._min_gap) == (30, 5, 1, 90, 3)


# ---- other frame rates: the same number of SECONDS
@pytest.mark.parametrize("fps", [10, 15, 24, 25, 60])
def test_every_window_lasts_the_same_seconds_at_other_frame_rates(fps):
    c = cfg()
    r = WindowedRules(c.rules, CAM, fps=fps)
    tol = 0.5 / fps + 1e-9                                      # frame counts are integers: at most half a frame of rounding error
    assert abs(r._wh / fps - c.rules.helmet.window_s) <= tol and abs(r._wo / fps - c.rules.overload.window_s) <= tol
    assert abs(r._history_len / fps - c.rules.wrong_way.history_s) <= tol and abs(r._min_history / fps - c.rules.wrong_way.min_history_s) <= tol
    assert abs(r._purge_every / fps - PURGE_EVERY_S) <= tol and abs(r._purge_after / fps - PURGE_AFTER_S) <= tol
    assert abs(r._violation_frames / fps - c.rules.helmet.violation_s) <= tol
    rec = EvidenceRecorder(c.evidence, lambda crop: [], "v", "c", fps=fps)
    assert abs(rec._close_after / fps - c.evidence.close_after_s) <= tol
    assert abs(rec._box_every / fps - c.evidence.box_sample_s) <= tol
    assert abs(rec._min_gap / fps - c.evidence.ocr.min_gap_s) <= max(tol, 1 / fps)                # 0.1 s is below one frame at 10 FPS: the minimum is one frame
    assert abs(rec._max_attempts * rec._every / fps - c.evidence.plate.search_s) <= tol * rec._every + 1e-9


def test_at_10_fps_the_default_windows_are_5_frames_and_the_vote_threshold_is_the_same_fraction():
    c = cfg()
    assert window_votes(c.rules.helmet, 10) == (5, 2)           # 0.5 s = 5 frames; K / W = 1/3 -> 2 of 5 (rounded), min_ratio 0.5 still applies
    r = WindowedRules(c.rules, CAM, fps=10)
    assert (r._wh, r._kh, r._history_len, r._min_history, r._purge_every) == (5, 2, 10, 5, 100)


def frame(heads=(), tid=1):
    hs = [Track(0, kind, (120 + 40 * i, 120, 150 + 40 * i, 160), 0.9) for i, kind in enumerate(heads)]
    return Perception(helmet=[Track(tid, "motorcycle", MOTO, 0.9)] + hs)


def verdict_times(fps, seq_seconds):
    """seq_seconds: list of (head kind or None, seconds). Returns (time of the first flag, time the flag is gone again) in seconds."""
    r = WindowedRules(cfg().rules, CAM, fps=fps)
    first = last_on = None
    t = 0
    for kind, secs in seq_seconds:
        for _ in range(round(secs * fps)):
            res = r.step(frame((kind,) if kind else ()))
            on = {v.id: v for v in res.vehicles}["M_1"].violator
            t += 1
            if on and first is None:
                first = t / fps
            if on:
                last_on = t / fps
    return first, last_on


def test_a_steady_violation_is_flagged_after_the_same_time_and_a_clean_stretch_clears_it_after_the_same_time():
    seq = [("no-helmet", 1.0), ("helmet", 1.0)]
    f30, l30 = verdict_times(30, seq)
    for fps in (10, 15, 60):
        f, l = verdict_times(fps, seq)
        assert abs(f - f30) <= 1.5 / fps + 1e-9, (fps, f, f30)          # flagged within one frame period (rounding of K)
        assert abs(l - l30) <= 1.5 / fps + 1e-9, (fps, l, l30)          # and cleared with the same lag after the helmets appear
    assert f30 == pytest.approx(5 / 30)                                 # unchanged 30 FPS behaviour: the 5th vote


def test_a_short_glitch_is_ignored_at_every_frame_rate():
    seq = [("helmet", 0.5), ("no-helmet", 0.06), ("helmet", 1.0)]       # ~2 frames at 30 FPS, 1 frame at 10 FPS: below K at the same relative scale
    for fps in (10, 30, 60):
        assert verdict_times(fps, seq)[0] is None


# ---- evidence recorder and trackers follow the frame rate
def run_recorder(tmp_path, fps, violating_seconds, quiet_seconds, **plate):
    cfg_e = EvidenceCfg(dir=str(tmp_path), plate=PlateCfg(**plate))
    calls = []
    rec = EvidenceRecorder(cfg_e, lambda crop: calls.append(1) or [], "v", "c", fps=fps)
    raw = np.full((720, 1280, 3), 40, np.uint8)
    veh = Vehicle("M_1", (100, 100, 300, 400), "motorcycle", True, "NO-HELMET")
    n = round(violating_seconds * fps)
    for i in range(n):
        rec.update(i, raw, [veh], "UNKNOWN")
    closed_at = None
    for j in range(round(quiet_seconds * fps)):
        rec.update(n + j, raw, [], "UNKNOWN")
        if closed_at is None and not rec._open:
            closed_at = (j + 1) / fps
    return len(calls), closed_at


def test_plate_search_budget_and_event_close_delay_are_the_same_seconds_at_10_and_30_fps(tmp_path):
    runs30, close30 = run_recorder(tmp_path / "a", 30, 6.0, 3.0)
    runs10, close10 = run_recorder(tmp_path / "b", 10, 6.0, 3.0)
    assert runs30 == 90 and runs10 == 30                                 # 3 s of search: every frame is tried
    assert close30 == pytest.approx(1.0) and close10 == pytest.approx(1.0)


def test_tracker_buffer_keeps_its_duration(tmp_path):
    d = Detectors.__new__(Detectors)
    d.cfg = cfg()
    d.cfg.root = REPO
    tracker = "configs/trackers/botsort_gmc_none.yaml"
    d.fps = 30.0
    assert d._tracker_path(tracker) == REPO / tracker                    # 30 FPS: the original file, untouched
    for fps, frames in ((10.0, 10), (60.0, 60), (15.0, 15)):
        d.fps = fps
        got = yaml.safe_load(d._tracker_path(tracker).read_text(encoding="utf-8"))
        assert got["track_buffer"] == frames and got["tracker_type"] == "botsort" and got["gmc_method"] == "none"
    assert yaml.safe_load((REPO / tracker).read_text(encoding="utf-8"))["track_buffer"] == 30      # the shipped file is never modified


def test_set_fps_on_detectors_drops_the_tracker_only_when_the_rate_changes():
    d = Detectors.__new__(Detectors)
    d.coco, d.helmet = SimpleNamespace(predictor="p1"), SimpleNamespace(predictor="p2")
    d.set_fps(30)
    assert d.coco.predictor == "p1"                                      # nothing changes at the reference rate
    d.set_fps(10)
    assert d.coco.predictor is None and d.helmet.predictor is None and d.fps == 10.0


def test_pipeline_set_fps_reconfigures_rules_recorder_and_detectors(tmp_path):
    from sentinel.pipeline import Pipeline, SharedState

    class Dets(Detectors):
        def __init__(self):
            self.coco = self.helmet = self.plate = SimpleNamespace(predictor=None)

    c = cfg()
    c.evidence.dir = str(tmp_path)
    p = Pipeline(Dets(), SharedState(), c, CAM)
    assert (p.rules._wh, p.recorder._close_after, p.d.fps) == (15, 30, 30.0)
    p.set_fps(10)
    assert (p.rules._wh, p.rules._kh, p.recorder._close_after, p.d.fps, p.fps) == (5, 2, 10, 10.0, 10.0)
    p.set_fps(30)
    assert (p.rules._wh, p.rules._kh, p.recorder._close_after) == (15, 5, 30)


def test_the_worker_tells_the_pipeline_the_frame_rate_of_each_source():
    from apps.web.worker import InferenceWorker

    class Src:
        is_file = True

        def __init__(self, spec):
            self.fps = 12.5 if "slow" in spec else None

        def is_open(self):
            return True

        def read(self):
            return True, np.zeros((4, 4, 3), np.uint8)

        def release(self):
            pass

    seen = []
    pipe = SimpleNamespace(process=lambda f: SimpleNamespace(frame=f), reset=lambda: None, set_fps=lambda fps: seen.append(fps))
    w = InferenceWorker(pipe, "slow.mp4", source_factory=Src, encode=lambda f: b"x")
    assert w.next_jpeg(0, timeout=2) is not None
    w.stop()
    assert seen == [12.5]                                                # a source without a known fps leaves the pipeline as it is


# ---- legacy configs and the converter
LEGACY = """
rules:
  engine: windowed
  helmet: {violation_frames: 5, window: 15, min_votes: 5, min_ratio: 0.5}
  overload: {max_passengers: 2, window: 15, min_votes: 5, min_ratio: 0.5}
  wrong_way: {enabled: false, history_len: 30, min_history: 15, dy_threshold: -30}
evidence:
  close_after: 30
  plate: {max_attempts: 90, stop_when_found: true, every: 1}
  ocr: {min_gap: 3}
"""


def test_a_config_written_in_frames_still_loads_with_a_warning_and_means_frames_at_30_fps(tmp_path):
    f = tmp_path / "old.yaml"
    f.write_text(LEGACY, encoding="utf-8")
    with pytest.warns(UserWarning, match="frame-based"):
        c = load_pipeline_config(f)
    assert window_votes(c.rules.helmet, 30) == (15, 5)                   # identical to the new default at 30 FPS
    assert window_votes(c.rules.helmet, 10) == (5, 2)                    # and it follows the frame rate like the new keys
    rec = EvidenceRecorder(c.evidence, lambda crop: [], "v", "c", fps=30)
    assert (rec._close_after, rec._max_attempts, rec._every, rec._min_gap) == (30, 90, 1, 3)
    rec10 = EvidenceRecorder(c.evidence, lambda crop: [], "v", "c", fps=10)
    assert rec10._close_after == 10 and rec10._max_attempts == 30        # legacy frame values are durations at 30 FPS (1 s, 3 s): at 10 FPS they stay 1 s and 3 s


def test_the_new_default_config_loads_without_any_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_pipeline_config()
        for name in ("docker_cpu", "ocr", "onnx_cpu", "pt_cpu", "pt_gpu"):
            load_pipeline_config(REPO / "configs" / "pipeline" / f"{name}.yaml")


LEGACY_BLOCK = """
rules:
  engine: windowed
  helmet:
    violation_frames: 5
    window: 15              # W
    min_votes: 5            # K
    min_ratio: 0.5
  overload:
    max_passengers: 2
    window: 15
    min_votes: 5
    min_ratio: 0.5
  wrong_way:
    enabled: false
    history_len: 30
    min_history: 15
    dy_threshold: -30
evidence:
  close_after: 30           # frames
  plate:
    max_attempts: 90
    stop_when_found: true
    every: 1
  ocr:
    min_gap: 3
"""


def same_resolution(old_file, new_file):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b = load_pipeline_config(old_file), load_pipeline_config(new_file)
    for fps in (30, 10, 60):
        assert window_votes(a.rules.helmet, fps) == window_votes(b.rules.helmet, fps)
        ra, rb = EvidenceRecorder(a.evidence, lambda c: [], "v", "c", fps=fps), EvidenceRecorder(b.evidence, lambda c: [], "v", "c", fps=fps)
        assert (ra._close_after, ra._max_attempts, ra._every, ra._min_gap) == (rb._close_after, rb._max_attempts, rb._every, rb._min_gap)


def test_converter_rewrites_frames_to_seconds_keeps_comments_and_is_idempotent(tmp_path):
    new, changes = conv.convert_text(LEGACY_BLOCK)
    assert len(changes) == 11 and "window_s: 0.5              # W" in new and "close_after_s: 1           # frames" in new and "search_s: 3" in new
    old_f, new_f = tmp_path / "old.yaml", tmp_path / "new.yaml"
    old_f.write_text(LEGACY_BLOCK, encoding="utf-8")
    new_f.write_text(new, encoding="utf-8")
    again, changes2 = conv.convert_text(new)
    assert changes2 == {} and again == new                                         # nothing left to convert
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_pipeline_config(new_f)                                                 # no legacy key left: no warning
    same_resolution(old_f, new_f)


def test_converter_handles_flow_style_mappings_by_rebuilding_the_file(tmp_path):
    new, changes = conv.convert_text(LEGACY)                                        # LEGACY uses inline {...} mappings: no line to edit
    assert changes and "window:" not in new and "window_s" in new
    old_f, new_f = tmp_path / "old.yaml", tmp_path / "new.yaml"
    old_f.write_text(LEGACY, encoding="utf-8")
    new_f.write_text(new, encoding="utf-8")
    same_resolution(old_f, new_f)


def test_the_shipped_configs_convert_to_the_same_values_at_30_fps():
    """Every config under configs/pipeline resolves to identical frame counts at 30 FPS whether it is written in frames or in seconds."""
    for f in sorted((REPO / "configs" / "pipeline").rglob("*.yaml")):
        raw = f.read_text(encoding="utf-8")
        new, _ = conv.convert_text(raw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = load_pipeline_config(f)
        d = yaml.safe_load(new)
        # resolve both through the same arithmetic
        ra = WindowedRules(a.rules, CAM, fps=30)
        from sentinel.config import _build, PipelineConfig
        b = _build(PipelineConfig, d, str(f)).validate()
        rb = WindowedRules(b.rules, CAM, fps=30)
        assert (ra._wh, ra._kh, ra._wo, ra._ko, ra._violation_frames, ra._history_len, ra._min_history) == (rb._wh, rb._kh, rb._wo, rb._ko, rb._violation_frames, rb._history_len, rb._min_history), f.name
        ea, eb = EvidenceRecorder(a.evidence, lambda c: [], "v", "c", fps=30), EvidenceRecorder(b.evidence, lambda c: [], "v", "c", fps=30)
        assert (ea._close_after, ea._max_attempts, ea._every, ea._min_gap) == (eb._close_after, eb._max_attempts, eb._every, eb._min_gap), f.name
