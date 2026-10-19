from pathlib import Path

import pytest
import yaml

from sentinel.config import REPO, CameraConfig, load_camera_config, load_pipeline_config
from sentinel.timebase import to_frames


def test_default_config_matches_original_constants():
    """The values the original app.py hard-coded; characterization tests depend on them."""
    c = load_pipeline_config()
    assert c.frame_size == [1280, 720]
    assert (c.models.coco.conf, c.models.helmet.conf, c.models.plate.conf) == (0.35, 0.55, 0.35)
    assert to_frames(c.rules.helmet.violation_s, 30) == 5                         # durations are seconds; at 30 FPS they are the old frame counts
    assert c.rules.overload.max_passengers == 2
    ww = c.rules.wrong_way
    assert (to_frames(ww.history_s, 30), to_frames(ww.min_history_s, 30), ww.dy_threshold) == (30, 15, -30)
    assert ww.enabled is False                       # off by default since baseline v2 (out of the reported scope)
    assert round(c.evidence.plate.search_s * 30) == 90 and c.evidence.plate.every_s == 0
    assert c.evidence.dir == "static/violations" and c.evidence.plate_margin == 10 and c.evidence.max_records == 12


def test_camera_config_polygon_and_defaults():
    cam = load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
    assert cam.violation_zone == [[250, 250], [750, 250], [820, 380], [150, 380]]
    assert cam.wrong_lane_zone is None
    poly = CameraConfig.polygon(cam.violation_zone)
    assert poly.shape == (4, 1, 2) and str(poly.dtype) == "int32"
    assert CameraConfig.polygon(None) is None


def test_tracker_configs_are_exposed_and_resolve():
    c = load_pipeline_config()
    for m in (c.models.coco, c.models.helmet):
        p = c.path(m.tracker)
        assert p.exists(), p
        assert yaml.safe_load(p.read_text(encoding="utf-8"))["tracker_type"] in ("botsort", "bytetrack")
    assert c.models.plate.tracker is None
    assert yaml.safe_load((REPO / "configs/trackers/botsort.yaml").read_text(encoding="utf-8"))["gmc_method"] == "sparseOptFlow"


def test_unknown_keys_are_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("rules:\n  helmet:\n    violation_framez: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="violation_framez"):
        load_pipeline_config(bad)


@pytest.mark.parametrize("text", [
    "models:\n  coco: {weights: a.pt, conf: 0}\n",
    "frame_size: [1280]\n",
    "rules:\n  wrong_way: {history_len: 10, min_history: 20}\n",
    "rules:\n  helmet: {violation_frames: 0}\n",
])
def test_invalid_values_are_rejected(tmp_path, text):
    f = tmp_path / "bad.yaml"
    f.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_pipeline_config(f)


def test_camera_requires_a_valid_zone(tmp_path):
    f = tmp_path / "cam.yaml"
    f.write_text("camera_id: x\nviolation_zone: [[0, 0], [1, 1]]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_camera_config(f)


def test_device_override_and_relative_paths():
    c = load_pipeline_config(device="cpu", root=Path("/somewhere"))
    assert c.device == "cpu" and c.path("yolo11s.pt") == Path("/somewhere/yolo11s.pt")
    assert load_pipeline_config().device is None


def test_web_defaults_are_restrictive():
    w = load_pipeline_config().web
    assert w.host == "127.0.0.1" and w.allow_rtsp is False and w.allow_camera_index is False
    assert w.max_upload_mb <= 1000 and ".exe" not in w.allowed_extensions and ".py" not in w.allowed_extensions


def test_entrypoint_does_not_expose_the_server_or_autoreload():
    src = (REPO / "app.py").read_text(encoding="utf-8")
    assert "0.0.0.0" not in src and "reload=True" not in src


def test_default_tracker_is_botsort_without_gmc_and_marked_temporary():
    c = load_pipeline_config()
    assert c.models.coco.tracker == c.models.helmet.tracker == "configs/trackers/botsort_gmc_none.yaml"
    assert yaml.safe_load((REPO / c.models.coco.tracker).read_text(encoding="utf-8"))["gmc_method"] == "none"
    assert "TEMPORARY" in (REPO / "configs/pipeline/default.yaml").read_text(encoding="utf-8")
    ref = load_pipeline_config(REPO / "configs/pipeline/exp/tracker_botsort_gmc.yaml")
    assert yaml.safe_load((REPO / ref.models.coco.tracker).read_text(encoding="utf-8"))["gmc_method"] == "sparseOptFlow"


def test_saved_evidence_is_never_stored_at_stream_quality(tmp_path):
    c = load_pipeline_config()
    assert c.evidence.jpeg_quality >= 95 and c.web.jpeg_quality == 80 and c.evidence.jpeg_quality > c.web.jpeg_quality
    f = tmp_path / "low.yaml"
    f.write_text("evidence:\n  jpeg_quality: 90\n", encoding="utf-8")
    with pytest.raises(ValueError, match="95"):
        load_pipeline_config(f)


def test_every_shipped_pipeline_config_validates():
    """configs/pipeline/**/*.yaml are used by experiments and tools: a rule change (e.g. evidence quality >= 95) must not silently break them."""
    files = sorted((REPO / "configs" / "pipeline").rglob("*.yaml"))
    assert len(files) >= 10
    for f in files:
        load_pipeline_config(f)
