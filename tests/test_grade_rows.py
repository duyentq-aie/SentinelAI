"""tools/grade_rows.py: the pure parts (rows, csv, resume, statistics, README section); the OpenCV window itself is not tested."""
import sys

import pytest

from sentinel.config import REPO

sys.path.insert(0, str(REPO / "tools"))
import grade_rows as g  # noqa: E402


def row(n, versions, type="no_helmet", t=1.0):
    return {"n": n, "time_s": t, "type": type, "versions": {v: ({"track": f"M_{n}", "frame": 10} if v in versions else None) for v in g.VERSIONS}}


ROWS = [row(1, ["v0"]), row(2, ["v0", "current+old model"]), row(3, ["EXP-015 3cls", "EXP-016 4cls"]), row(4, g.VERSIONS)]


def test_rows_of_the_real_template_are_the_23_non_wrong_way_rows():
    rows = g.load_rows()
    assert len(rows) == 23 and {r["type"] for r in rows} == {"no_helmet", "overload"}
    assert [(r["time_s"], r["n"]) for r in rows] == sorted((r["time_s"], r["n"]) for r in rows)


def test_csv_roundtrip_and_atomic_overwrite(tmp_path):
    p = tmp_path / "grading_rows.csv"
    assert g.read_grades(p) == {}
    grades = {2: {"truth": "sai", "note": ""}, 1: {"truth": "split", "note": "different people; v0=dung|EXP-015 3cls=sai"}}
    g.write_grades(grades, p)
    assert p.read_text(encoding="utf-8").splitlines()[0] == "row,truth,note"
    assert g.read_grades(p) == grades and not list(tmp_path.glob("*.tmp"))


def test_split_rows_are_complete_only_when_every_version_with_an_event_has_a_verdict():
    r = row(5, ["v0", "EXP-015 3cls"])
    assert not g.is_complete(r, None)
    assert g.is_complete(r, {"truth": "dung", "note": ""})
    assert not g.is_complete(r, {"truth": "split", "note": "different people; v0=dung"})
    assert g.is_complete(r, {"truth": "split", "note": "different people; v0=dung|EXP-015 3cls=sai"})
    assert g.split_note("different people; v0=dung|EXP-015 3cls=sai") == {"v0": "dung", "EXP-015 3cls": "sai"}


def test_precision_counts_only_versions_with_an_event_and_leaves_unclear_out():
    grades = {1: {"truth": "dung", "note": ""}, 2: {"truth": "sai", "note": ""}, 3: {"truth": "khong_ro", "note": ""}, 4: {"truth": "dung", "note": ""}}
    st = g.stats(ROWS, grades)
    v = st["versions"]
    assert (v["v0"]["dung"], v["v0"]["sai"], v["v0"]["khong_ro"], v["v0"]["precision"]) == (2, 1, 0, round(2 / 3, 3))      # rows 1 (dung), 2 (sai), 4 (dung)
    assert (v["current+old model"]["dung"], v["current+old model"]["sai"], v["current+old model"]["precision"]) == (1, 1, 0.5)
    assert (v["EXP-015 3cls"]["dung"], v["EXP-015 3cls"]["khong_ro"], v["EXP-015 3cls"]["precision"]) == (1, 1, 1.0)         # unclear left out of the ratio
    assert v["EXP-016 4cls"]["events"] == 2 and st["rows_graded"] == 4


def test_precision_is_none_when_a_version_has_no_graded_event():
    st = g.stats([row(1, ["v0"])], {1: {"truth": "khong_ro", "note": ""}})
    assert st["versions"]["v0"]["precision"] is None and st["versions"]["EXP-016 4cls"]["events"] == 0


def test_relative_coverage_counts_rows_graded_right_and_is_not_recall():
    grades = {1: {"truth": "dung", "note": ""}, 2: {"truth": "dung", "note": ""}, 3: {"truth": "sai", "note": ""},
              4: {"truth": "split", "note": "different people; v0=dung|current+old model=sai|EXP-015 3cls=sai|EXP-016 4cls=sai|current+old model @10FPS=sai"}}
    st = g.stats(ROWS, grades)
    assert st["true_rows"] == 3                                                       # rows 1, 2 and 4 (one version right)
    assert st["versions"]["v0"]["covered_true_rows"] == 3 and st["versions"]["current+old model"]["covered_true_rows"] == 1
    assert st["versions"]["EXP-015 3cls"]["covered_true_rows"] == 0
    md = g.render_markdown(st)
    assert "NOT recall" in md and "ONE 26-second clip" in md and "adjusted on this very clip" in md and "n is small" in md


def test_readme_section_is_inserted_once_and_replaced_on_rerun(tmp_path):
    p = tmp_path / "README.md"
    p.write_text("# EXP\n\ntext\n", encoding="utf-8")
    st1 = g.stats(ROWS, {1: {"truth": "dung", "note": ""}})
    g.write_readme(st1, p)
    g.write_readme(g.stats(ROWS, {1: {"truth": "sai", "note": ""}}), p)
    s = p.read_text(encoding="utf-8")
    assert s.count(g.MARK[0]) == 1 and s.startswith("# EXP\n\ntext\n") and "| v0 | 1 | 0 | 1 | 0 | 0.00 (0/1)" in s
