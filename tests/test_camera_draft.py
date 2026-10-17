import numpy as np
import pytest
import yaml

from sentinel.camera_draft import CameraDraft


def drawn():
    d = CameraDraft("cam1", "clip.mp4")
    d.set_mode("stop_line"); d.click(100, 500); d.click(900, 520)
    d.set_mode("direction"); d.click(500, 600); d.click(500, 400)
    d.set_mode("light_roi"); d.click(1130, 190); d.click(1110, 140)               # corners in any order
    d.set_mode("violation_zone")
    for p in ((250, 250), (750, 250), (820, 380), (150, 380)):
        d.click(*p)
    return d


def test_complete_draft_has_no_problems_and_serialises_normalised_values():
    d = drawn()
    assert d.problems() == []
    out = d.to_dict()
    assert out["stop_line"] == [[100, 500], [900, 520]] and out["direction"] == [[500, 600], [500, 400]]
    assert out["light_roi"] == [1110, 140, 1130, 190]                          # x1 <= x2, y1 <= y2
    assert out["violation_zone"][0] == [250, 250] and out["wrong_lane_zone"] is None and out["frame_size"] == [1280, 720]


def test_clicks_are_clamped_to_the_frame_and_a_third_click_restarts_a_two_point_shape():
    d = CameraDraft("c", "v", 1280, 720)
    d.set_mode("stop_line"); d.click(-5, 9999); d.click(5000, -1)
    assert d.points["stop_line"] == [[0, 719], [1279, 0]]
    d.click(10, 10)
    assert d.points["stop_line"] == [[10, 10]]                                  # started over


def test_undo_clear_and_mode_validation():
    d = drawn()
    d.set_mode("violation_zone"); assert d.undo() == [150, 380]; assert len(d.points["violation_zone"]) == 3
    d.clear(); assert d.points["violation_zone"] == []
    with pytest.raises(ValueError):
        d.set_mode("nope")


@pytest.mark.parametrize("edit,fragment", [
    (lambda d: d.points.update(stop_line=[[1, 1], [1, 1]]), "identical"),
    (lambda d: d.points.update(direction=[[5, 5]]), "exactly 2"),
    (lambda d: d.points.update(light_roi=[[10, 10], [10, 50]]), "empty"),
    (lambda d: d.points.update(violation_zone=[[0, 0], [5, 5]]), "at least 3"),
])
def test_problems_are_reported_and_block_saving(tmp_path, edit, fragment):
    d = drawn()
    edit(d)
    assert any(fragment in p for p in d.problems())
    with pytest.raises(ValueError):
        d.save(tmp_path / "c.yaml")
    assert not (tmp_path / "c.yaml").exists()


def test_save_and_reload_roundtrip_and_pipeline_loadable(tmp_path):
    d = drawn()
    f = tmp_path / "cams" / "cam1.yaml"
    d.save(f)
    again = CameraDraft.from_file(f)
    assert again.to_dict() == d.to_dict()
    assert yaml.safe_load(f.read_text(encoding="utf-8"))["camera_id"] == "cam1"


def test_the_pipeline_config_loader_accepts_a_drawn_file(tmp_path):
    from sentinel.config import load_camera_config
    f = tmp_path / "cam1.yaml"
    drawn().save(f)
    cam = load_camera_config(f)
    assert cam.camera_id == "cam1" and cam.stop_line == [[100, 500], [900, 520]] and cam.light_roi == [1110, 140, 1130, 190]


def test_overlay_draws_without_a_window():
    from tools.camera_config import draw_overlay
    img = np.zeros((720, 1280, 3), np.uint8)
    out = draw_overlay(img, drawn())
    assert out.shape == img.shape and out.any() and not img.any()             # drew on a copy
