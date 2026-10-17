import json

import pytest

from sentinel import labeling
from sentinel.evidence import GT_FIELDS


def fresh():
    return labeling.new_labels("clip1", 30.0, 1920, 1080, "dev", annotated_ranges=[[0, 3000]])


def commit(s, box=(100, 100, 300, 400), start=10, end=60, key=30, plate="29-B1 123.45", difficulty="normal"):
    s.mark_start(start); s.mark_end(end); s.set_key(key, box)
    return s.commit(plate, difficulty)


def test_session_builds_events_with_the_gt_schema_and_incrementing_ids():
    s = labeling.LabelSession(fresh())
    a = commit(s)
    s.cycle_type()
    b = commit(s, box=(500, 200, 600, 300), start=100, end=140, key=120, plate="", difficulty="hard")
    assert [a["id"], b["id"]] == ["e001", "e002"] and (a["type"], b["type"]) == ("no_helmet", "overload")
    assert all(k in a for k in GT_FIELDS)
    assert a["plate_text"] == "29B112345" and b["plate_text"] is None and b["difficulty"] == "hard"


def test_box_corners_are_normalised_whatever_the_drag_direction():
    s = labeling.LabelSession(fresh())
    ev = commit(s, box=(300, 400, 100, 100))
    assert ev["key_box"] == [100, 100, 300, 400]


def test_commit_requires_a_complete_draft_and_a_valid_event():
    s = labeling.LabelSession(fresh())
    with pytest.raises(ValueError):
        s.commit("29B112345")                                   # nothing marked
    s.mark_start(10); s.mark_end(50); s.set_key(20, (0, 0, 5000, 5000))        # box outside the 1920x1080 frame
    with pytest.raises(ValueError):
        s.commit()
    assert s.labels["events"] == []                              # a rejected event is not kept
    s.set_key(5000, (10, 10, 50, 50))                            # key frame outside every annotated range
    with pytest.raises(ValueError):
        s.commit()


def test_start_end_key_are_reordered_so_that_start_le_key_le_end():
    s = labeling.LabelSession(fresh())
    ev = commit(s, start=80, end=40, key=60)
    assert ev["frame_start"] == 40 and ev["frame_end"] == 80 and ev["key_frame"] == 60


def test_undo_and_ranges():
    s = labeling.LabelSession(fresh())
    commit(s)
    assert s.undo()["id"] == "e001" and s.undo() is None
    assert s.toggle_range(100) is None and s.toggle_range(50) == [50, 100]
    assert s.labels["annotated_ranges"][-1] == [50, 100]
    s.toggle_range(7); assert s.toggle_range(7) is None          # zero-length range is ignored


def test_save_load_roundtrip_is_atomic_sorted_and_validated(tmp_path):
    s = labeling.LabelSession(fresh())
    commit(s, start=200, end=260, key=230)
    commit(s, box=(10, 10, 60, 60), start=10, end=40, key=20)
    out = tmp_path / "sub" / "clip1.json"
    labeling.save(out, s.labels)
    loaded = labeling.load(out)
    assert [e["frame_start"] for e in loaded["events"]] == [10, 200]
    assert not list(out.parent.glob("*.tmp"))
    assert json.loads(out.read_text(encoding="utf-8"))["video_id"] == "clip1"
    bad = dict(loaded, events=loaded["events"] + [dict(loaded["events"][0])])          # duplicate id
    with pytest.raises(ValueError):
        labeling.save(out, bad)
    assert labeling.load(out)["events"] == loaded["events"]                             # the failed save left the file intact


@pytest.mark.parametrize("mutate,fragment", [
    (lambda e: e.update(type="speeding"), "type must be"),
    (lambda e: e.update(plate_text="29-B1"), "plate_text"),
    (lambda e: e.update(difficulty="easy"), "difficulty"),
    (lambda e: e.update(key_box=[5, 5, 5, 50]), "key_box"),
    (lambda e: e.update(frame_end=1, frame_start=50), "frame_start <= key_frame"),
])
def test_validate_reports_problems(mutate, fragment):
    s = labeling.LabelSession(fresh())
    commit(s)
    mutate(s.labels["events"][0])
    assert any(fragment in p for p in labeling.validate(s.labels))


def test_normalize_plate_text_and_video_hash(tmp_path):
    assert labeling.normalize_plate_text(" 51F-123.45 ") == "51F12345" and labeling.normalize_plate_text("") is None and labeling.normalize_plate_text(None) is None
    f = tmp_path / "v.bin"
    f.write_bytes(b"abc")
    assert labeling.video_sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_a_labels_file_without_annotated_ranges_accepts_events_anywhere():
    labels = labeling.new_labels("c", 30, 1280, 720)                       # no ranges declared yet
    s = labeling.LabelSession(labels)
    assert commit(s)["id"] == "e001"
