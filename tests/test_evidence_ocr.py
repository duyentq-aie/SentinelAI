"""OCR inside the evidence recorder: candidate crops per track, best-quality crop kept, one OCR run per track, failures contained."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel.config import EvidenceCfg, OcrCfg, PlateCfg
from sentinel.evidence import EvidenceRecorder
from sentinel.ocr.engine import OcrUnavailable, PlateOCR
from sentinel.rules.legacy import Vehicle

BOX = (100, 100, 400, 500)


class Rec:
    """Fake text recogniser: returns the scripted plate for every line; counts calls."""
    def __init__(self, text="29B112345", score=0.9, raises=None):
        self.text, self.score, self.raises, self.calls = text, score, raises, 0

    def __call__(self, img):
        self.calls += 1
        if self.raises:
            raise self.raises
        return self.text, self.score


def frame_with_plate(sharp):
    """720p frame whose 'plate' region (x 180..260, y 300..340) is either high-contrast noise (sharp) or flat grey (blurred)."""
    f = np.full((720, 1280, 3), 60, np.uint8)
    f[300:340, 180:260] = np.random.default_rng(1).integers(0, 255, (40, 80, 3), np.uint8) if sharp else 128
    return f


def make(tmp_path, rec=None, ocr_kw=None, plate_kw=None, **ev_kw):
    plate_boxes = lambda crop: [(80, 200, 160, 240, 0.9)]       # the vehicle crop starts at (100, 100): plate at x 180..260, y 300..340
    cfg = EvidenceCfg(dir=str(tmp_path), plate_margin=0, plate=PlateCfg(**(plate_kw or {})), ocr=OcrCfg(enabled=True, **(ocr_kw or {})), **ev_kw)
    rec = rec or Rec()
    return EvidenceRecorder(cfg, plate_boxes, "vid", "cam", ocr=PlateOCR(rec, reads=3)), rec


def veh(reason="NO-HELMET", id="M_1"):
    return Vehicle(id, BOX, "motorcycle", True, reason)


def last(tmp_path):
    out = {}
    for line in (Path(tmp_path) / "events.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["id"]] = r
    return out


def test_disabled_by_default_leaves_plate_pred_empty(tmp_path):
    cfg = EvidenceCfg(dir=str(tmp_path))
    rec = EvidenceRecorder(cfg, lambda c: [(80, 200, 160, 240, 0.9)], "vid", "cam")
    rec.update(0, frame_with_plate(True), [veh()], "UNKNOWN")
    rec.flush()
    assert next(iter(last(tmp_path).values()))["plate_pred"] is None


def test_search_continues_until_enough_candidates_then_ocr_runs_once(tmp_path):
    rec_text = Rec()
    r, _ = make(tmp_path, rec_text, ocr_kw={"candidates": 4, "min_gap": 1, "reads": 3})
    calls = []
    orig = r.plate_boxes
    r.plate_boxes = lambda crop: (calls.append(1), orig(crop))[1]
    for i in range(10):
        r.update(i, frame_with_plate(sharp=(i == 2)), [veh()], "UNKNOWN")
    r.flush()
    ev = next(iter(last(tmp_path).values()))
    assert len(calls) == 4                                          # stopped after 4 candidates (not at the first plate, not at 10)
    assert rec_text.calls == 3 and ev["plate_pred"]["text"] == "29B112345" and ev["plate_pred"]["reads"] == 3
    assert ev["plate_pred"]["format_ok"] is True and ev["plate_pred"]["latency_ms"] >= 0 and ev["plate_text"] is None


def test_plate_jpg_is_replaced_by_the_best_quality_candidate(tmp_path):
    r, _ = make(tmp_path, ocr_kw={"candidates": 4, "min_gap": 1, "margin": 0})        # margin 0: no background edges to confuse the sharpness score
    for i in range(6):
        r.update(i, frame_with_plate(sharp=(i == 2)), [veh()], "UNKNOWN")          # only frame 2 has a sharp plate
    r.flush()
    ev = next(iter(last(tmp_path).values()))
    assert ev["plate"]["frame"] == 2                                  # the best crop's frame, not the first one
    img = cv2.imread(str(tmp_path / ev["evidence"]["plate"]))
    assert img.shape[:2] == (40, 80) and img.std() > 40               # the noisy (sharp) crop, not the flat one


def test_ocr_runs_when_the_violation_ends_before_enough_candidates(tmp_path):
    rec_text = Rec()
    r, _ = make(tmp_path, rec_text, ocr_kw={"candidates": 5, "min_gap": 1}, close_after=3)
    for i in range(2):
        r.update(i, frame_with_plate(True), [veh()], "UNKNOWN")
    for i in range(2, 8):
        r.update(i, frame_with_plate(True), [], "UNKNOWN")                    # vehicle stops violating -> event closes after 3 frames
    ev = next(iter(last(tmp_path).values()))
    assert ev["status"] == "closed" and ev["plate_pred"]["reads"] == 2 and rec_text.calls == 2


def test_ocr_runs_when_attempts_are_exhausted(tmp_path):
    rec_text = Rec()
    r, _ = make(tmp_path, rec_text, ocr_kw={"candidates": 5, "min_gap": 1}, plate_kw={"max_attempts": 3})
    for i in range(10):
        r.update(i, frame_with_plate(True), [veh()], "UNKNOWN")
    ev = next(iter(last(tmp_path).values()))
    assert ev["plate_pred"] is not None and ev["plate_pred"]["reads"] == 3             # 3 attempts -> 3 candidates -> read


def test_a_second_event_of_the_same_track_reuses_the_reading(tmp_path):
    rec_text = Rec()
    r, _ = make(tmp_path, rec_text, ocr_kw={"candidates": 2, "min_gap": 1, "reads": 2})
    for i in range(4):
        r.update(i, frame_with_plate(True), [veh("NO-HELMET")], "UNKNOWN")
    calls_before = rec_text.calls
    r.update(4, frame_with_plate(True), [veh("OVERLOAD & NO-HELMET")], "UNKNOWN")
    r.flush()
    evs = last(tmp_path).values()
    assert rec_text.calls == calls_before and len(evs) == 2 and all(e["plate_pred"]["text"] == "29B112345" for e in evs)


def test_an_unavailable_ocr_engine_is_contained_and_not_retried(tmp_path):
    rec_text = Rec(raises=OcrUnavailable("no model"))
    r, _ = make(tmp_path, rec_text, ocr_kw={"candidates": 2, "min_gap": 1})
    for i in range(8):
        r.update(i, frame_with_plate(True), [veh()], "UNKNOWN")                    # must not raise
    r.flush()
    ev = next(iter(last(tmp_path).values()))
    assert ev["plate_pred"]["text"] is None and "no model" in ev["plate_pred"]["error"]
    assert ev["plate"]["found"] is True                                              # evidence is still complete
    assert rec_text.calls == 1                                                       # tried once, then OCR switched off


def test_unreadable_plate_gives_text_none_not_an_error(tmp_path):
    r, _ = make(tmp_path, Rec(text="???", score=0.1), ocr_kw={"candidates": 2, "min_gap": 1})
    for i in range(4):
        r.update(i, frame_with_plate(True), [veh()], "UNKNOWN")
    r.flush()
    pred = next(iter(last(tmp_path).values()))["plate_pred"]
    assert pred["text"] is None and pred["format_ok"] is False and "error" not in pred


def test_reset_forgets_candidates_and_readings(tmp_path):
    rec_text = Rec()
    r, _ = make(tmp_path, rec_text, ocr_kw={"candidates": 2, "min_gap": 1, "reads": 2})
    for i in range(3):
        r.update(i, frame_with_plate(True), [veh()], "UNKNOWN")
    r.reset()
    n = rec_text.calls
    for i in range(3):
        r.update(i, frame_with_plate(True), [veh()], "UNKNOWN")                    # same track id after the reset = a different vehicle
    assert rec_text.calls > n
