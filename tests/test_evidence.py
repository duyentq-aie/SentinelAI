"""EvidenceRecorder: events are always written, plates are searched only for violating tracks and only a bounded number of times."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel.config import EvidenceCfg, PlateCfg
from sentinel.evidence import GT_FIELDS, PREDICTION_FIELDS, EvidenceRecorder, enforce_dir_limit
from sentinel.rules.legacy import Vehicle


def make(tmp_path, plate_results=None, **kw):
    """plate_results: callable(call_no, crop) -> boxes; default never finds a plate."""
    calls = []

    def plate_boxes(crop):
        calls.append(crop.shape)
        return plate_results(len(calls), crop) if plate_results else []

    plate_kw = kw.pop("plate", {})
    cfg = EvidenceCfg(dir=str(tmp_path), plate=PlateCfg(**plate_kw), **kw)
    return EvidenceRecorder(cfg, plate_boxes, video_id="vid", camera_id="cam"), calls


def raw_frame(fill=40):
    f = np.full((720, 1280, 3), fill, np.uint8)
    return f


def veh(id="M_1", reason="NO-HELMET", box=(100, 100, 300, 400), type="motorcycle", violator=True):
    return Vehicle(id, box, type, violator, reason if violator else "")


def records(tmp_path):
    return [json.loads(l) for l in (Path(tmp_path) / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def last_by_id(tmp_path):
    out = {}
    for r in records(tmp_path):
        out[r["id"]] = r
    return out


def test_event_is_written_even_without_a_plate_and_has_all_files(tmp_path):
    rec, calls = make(tmp_path, plate={"max_attempts": 3})
    for i in range(5):
        rec.update(i, raw_frame(), [veh()], "UNKNOWN")
    rec.flush()
    (ev,) = last_by_id(tmp_path).values()
    assert ev["status"] == "closed" and ev["type"] == "no_helmet" and ev["track_id"] == "M_1"
    assert ev["plate"] == {"found": False, "attempts": 3} and ev["evidence"]["plate"] is None
    d = tmp_path / ev["id"]
    assert (d / "frame.jpg").exists() and (d / "vehicle.jpg").exists() and not (d / "plate.jpg").exists()
    assert ev["frame_start"] == 0 and ev["frame_end"] == 4 and len(calls) == 3        # stopped after max_attempts


def test_record_schema_contains_ground_truth_and_prediction_fields(tmp_path):
    rec, _ = make(tmp_path)
    rec.update(7, raw_frame(), [veh(box=(100, 100, 300, 400))], "RED")
    rec.flush()
    ev = last_by_id(tmp_path)[next(iter(last_by_id(tmp_path)))]
    assert all(k in ev for k in GT_FIELDS + PREDICTION_FIELDS), set(GT_FIELDS + PREDICTION_FIELDS) - set(ev)
    assert ev["key_frame"] == 7 == ev["trigger_frame"] and ev["key_box"] == [100, 100, 300, 400]
    assert ev["plate_text"] is None and ev["difficulty"] is None and ev["light"] == "RED"
    assert ev["vehicle"] == {"cls": "motorcycle", "box_size": [200, 300]}


def test_plate_detector_runs_only_for_violating_tracks(tmp_path):
    rec, calls = make(tmp_path)
    for i in range(10):
        rec.update(i, raw_frame(), [veh("M_2", violator=False), veh("C_3", "WRONG-WAY", (400, 100, 600, 300), "car", False)], "UNKNOWN")
    assert calls == [] and not (tmp_path / "events.jsonl").exists()


def test_plate_search_stops_at_first_plate_and_crop_is_raw_pixels(tmp_path):
    raw = raw_frame(40)
    raw[200:230, 150:230] = (0, 200, 0)                       # a green "plate" at x 150..230, y 200..230
    found = lambda n, crop: [(50, 100, 130, 130, 0.9)] if n == 2 else []   # crop starts at (100, 100)
    rec, calls = make(tmp_path, found, plate={"max_attempts": 30}, plate_margin=0)
    res = [rec.update(i, raw, [veh()], "UNKNOWN") for i in range(8)]
    rec.flush()
    assert len(calls) == 2                                    # found on the 2nd run, then never again
    assert [r.plates_found for r in res].count(["M_1"]) == 1 and sum(r.plate_calls for r in res) == 2
    ev = next(iter(last_by_id(tmp_path).values()))
    assert ev["plate"]["found"] and ev["plate"]["frame"] == 1 and ev["plate"]["box"] == [150, 200, 230, 230]
    assert ev["plate"]["crop_size"] == [80, 30] and ev["evidence"]["plate"] == f"{ev['id']}/plate.jpg"      # relative to the folder
    img = cv2.imread(str(tmp_path / ev["evidence"]["plate"]))
    assert img.shape[:2] == (30, 80) and img[..., 1].mean() > 150          # clean green pixels, no overlay


def test_v0_mode_runs_on_every_violating_frame(tmp_path):
    found = lambda n, crop: [(1, 1, 20, 10, 0.9)]
    rec, calls = make(tmp_path, found, plate={"max_attempts": 0, "stop_when_found": False})
    res = [rec.update(i, raw_frame(), [veh()], "UNKNOWN") for i in range(6)]
    assert len(calls) == 6 and sum(len(r.plates_found) for r in res) == 1      # still only one capture per track
    assert all(r.plate_boxes for r in res)                                      # boxes still reported for drawing every frame


def test_every_k_frames(tmp_path):
    rec, calls = make(tmp_path, plate={"every": 3, "max_attempts": 0})
    for i in range(9):
        rec.update(i, raw_frame(), [veh()], "UNKNOWN")
    assert len(calls) == 3


def test_overload_and_no_helmet_make_two_events_sharing_one_plate_search(tmp_path):
    found = lambda n, crop: [(1, 1, 20, 10, 0.9)]
    rec, calls = make(tmp_path, found)
    for i in range(3):
        rec.update(i, raw_frame(), [veh(reason="OVERLOAD & NO-HELMET")], "UNKNOWN")
    rec.flush()
    evs = last_by_id(tmp_path).values()
    assert sorted(e["type"] for e in evs) == ["no_helmet", "overload"]
    assert len(calls) == 1 and all(e["plate"]["found"] for e in evs)


def test_escalation_opens_new_event_and_reuses_the_known_plate(tmp_path):
    found = lambda n, crop: [(1, 1, 20, 10, 0.9)]
    rec, calls = make(tmp_path, found)
    rec.update(0, raw_frame(), [veh(reason="NO-HELMET")], "UNKNOWN")
    rec.update(1, raw_frame(), [veh(reason="NO-HELMET")], "UNKNOWN")
    r = rec.update(2, raw_frame(), [veh(reason="OVERLOAD & NO-HELMET")], "UNKNOWN")
    rec.flush()
    assert len(r.new_events) == 1 and len(calls) == 1
    evs = {e["type"]: e for e in last_by_id(tmp_path).values()}
    assert evs["overload"]["frame_start"] == 2 and evs["overload"]["plate"]["found"] and evs["no_helmet"]["frame_start"] == 0


def test_event_closes_after_the_violation_stops_and_reopens_as_a_new_event_when_allowed(tmp_path):
    rec, _ = make(tmp_path, close_after=5, one_event_per_track=False)
    for i in range(3):
        rec.update(i, raw_frame(), [veh()], "UNKNOWN")
    for i in range(3, 9):
        rec.update(i, raw_frame(), [veh(violator=False)], "UNKNOWN")
    assert [r["status"] for r in records(tmp_path)] == ["open", "closed"]
    rec.update(20, raw_frame(), [veh()], "UNKNOWN")
    assert len(last_by_id(tmp_path)) == 2


def test_reset_closes_events_and_forgets_tracks(tmp_path):
    found = lambda n, crop: [(1, 1, 20, 10, 0.9)]
    rec, calls = make(tmp_path, found)
    rec.update(0, raw_frame(), [veh()], "UNKNOWN")
    rec.reset()
    assert last_by_id(tmp_path)[next(iter(last_by_id(tmp_path)))]["status"] == "closed" and rec.run == 1
    rec.update(0, raw_frame(), [veh()], "UNKNOWN")                       # same track id after a reset is a different vehicle
    assert len(calls) == 2 and len(last_by_id(tmp_path)) == 2
    assert any(i.startswith("vid-r1-") for i in last_by_id(tmp_path))


def test_dir_size_limit_deletes_oldest_and_logs_eviction(tmp_path):
    evicted = []
    cfg = EvidenceCfg(dir=str(tmp_path), max_dir_mb=1)
    rec = EvidenceRecorder(cfg, lambda c: [], "vid", "cam", on_evicted=evicted.extend)
    noise = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)        # incompressible: ~1 MB JPEG each
    for k in range(6):
        rec.update(k * 100, noise, [veh(f"M_{k}")], "UNKNOWN")
        rec.update(k * 100 + 1, noise, [], "UNKNOWN")
        time.sleep(0.02)                                                  # distinct mtimes
    rec.flush()
    left = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert 0 < len(left) < 6 and evicted
    assert "vid-r0-e0001" not in left and f"vid-r0-e0006" in left        # the oldest went first, the newest stays
    assert (tmp_path / "events.jsonl").exists()                          # the log is never evicted
    assert any(r.get("status") == "evicted" for r in records(tmp_path))
    assert sum(f.stat().st_size for f in tmp_path.rglob("*") if f.is_file()) < 1.2 * 1024 * 1024 * 3   # roughly bounded


def test_enforce_dir_limit_protects_named_entries(tmp_path):
    (tmp_path / "events.jsonl").write_bytes(b"x" * 2000)
    for n in ("a", "b"):
        (tmp_path / n).mkdir()
        (tmp_path / n / "f").write_bytes(b"y" * 2000)
        time.sleep(0.02)
    assert enforce_dir_limit(tmp_path, 3000, protect=("events.jsonl", "a")) == ["b"]
    assert enforce_dir_limit(tmp_path, 0) == []                          # 0 = unlimited


def test_one_event_per_track_and_type_even_if_the_verdict_flickers(tmp_path):
    rec, _ = make(tmp_path, close_after=3)                      # one_event_per_track defaults to True
    for start in (0, 20, 40):                                    # violation comes back twice after the event closed
        for i in range(start, start + 3):
            rec.update(i, raw_frame(), [veh()], "UNKNOWN")
        for i in range(start + 3, start + 10):
            rec.update(i, raw_frame(), [veh(violator=False)], "UNKNOWN")
    rec.flush()
    assert len(last_by_id(tmp_path)) == 1
    rec.update(0, raw_frame(), [veh("M_2")], "UNKNOWN")           # a different track still gets its own event
    rec.update(1, raw_frame(), [veh("M_1", "OVERLOAD")], "UNKNOWN")   # and another TYPE on the same track too
    rec.flush()
    assert sorted((e["track_id"], e["type"]) for e in last_by_id(tmp_path).values()) == [("M_1", "no_helmet"), ("M_1", "overload"), ("M_2", "no_helmet")]


def test_reset_allows_the_same_track_id_again(tmp_path):
    rec, _ = make(tmp_path)
    rec.update(0, raw_frame(), [veh()], "UNKNOWN")
    rec.reset()
    rec.update(0, raw_frame(), [veh()], "UNKNOWN")                # ids restarted: a new vehicle
    assert len(last_by_id(tmp_path)) == 2
