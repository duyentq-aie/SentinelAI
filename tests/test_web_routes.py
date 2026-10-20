"""HTTP-level tests of the demo routes with a fake worker (no models, no GPU)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.web.routes import Context, create_app, read_events


class FakeWorker:
    def __init__(self):
        self.sources = []

    def set_source(self, s):
        self.sources.append(s)
        return str(s)

    def status(self):
        return {"running": False, "source": "x", "frames_processed": 0, "loops": 0}

    def next_jpeg(self, last_id=0, timeout=2.0):
        return None


@pytest.fixture
def env(tmp_path):
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    (tmp_path / "static" / "uploads").mkdir(parents=True)
    (tmp_path / "static" / "uploads" / "a.mp4").write_bytes(b"x")
    (tmp_path / "secret.mp4").write_bytes(b"x")
    web = SimpleNamespace(max_upload_mb=1, upload_dir="static/uploads", allowed_extensions=[".mp4", ".avi"],
                          allowed_source_dirs=["static/uploads"], allow_rtsp=False, allow_camera_index=False)
    worker = FakeWorker()
    ctx = Context(worker=worker, records=[], web=web, root=tmp_path, evidence_dir=tmp_path / "static" / "violations")
    return TestClient(create_app(ctx)), worker, tmp_path


def test_index_and_status(env):
    c, _, _ = env
    assert c.get("/").status_code == 200 and c.get("/api/status").json()["running"] is False


def test_upload_is_sanitised_and_becomes_the_source(env):
    c, worker, root = env
    r = c.post("/upload_video", files={"file": ("../../evil name.mp4", b"abc", "video/mp4")})
    assert r.status_code == 200
    name = r.json()["filename"]
    assert "/" not in name and ".." not in name and name.endswith("_evil_name.mp4")
    assert (root / "static" / "uploads" / name).read_bytes() == b"abc"
    assert worker.sources == [str(root / "static" / "uploads" / name)]
    assert not (root.parent / "evil name.mp4").exists()


def test_upload_wrong_type_is_400_and_does_not_change_the_source(env):
    c, worker, _ = env
    r = c.post("/upload_video", files={"file": ("shell.py", b"print(1)", "text/plain")})
    assert r.status_code == 400 and worker.sources == []


def test_upload_too_large_is_413_and_leaves_no_file(env):
    c, worker, root = env
    before = set((root / "static" / "uploads").iterdir())
    r = c.post("/upload_video", files={"file": ("big.mp4", b"x" * (2 * 1024 * 1024), "video/mp4")})
    assert r.status_code == 413 and worker.sources == [] and set((root / "static" / "uploads").iterdir()) == before


@pytest.mark.parametrize("src", ["../secret.mp4", "secret.mp4", "http://example.com/a.mp4", "rtsp://cam/1", "0", "static/uploads/missing.mp4",
                                 "C:/Windows/system32/config.mp4"])
def test_update_camera_rejects_everything_outside_the_whitelist(env, src):
    c, worker, _ = env
    r = c.post("/update_camera", data={"new_source": src})
    assert r.status_code == 400 and worker.sources == []


def test_update_camera_accepts_an_uploaded_file(env):
    c, worker, root = env
    r = c.post("/update_camera", data={"new_source": "static/uploads/a.mp4"})
    assert r.status_code == 200 and worker.sources == [str((root / "static/uploads/a.mp4").resolve())]


def test_events_endpoint_returns_last_record_per_id_newest_first_without_evicted(env):
    c, _, root = env
    d = root / "static" / "violations"
    d.mkdir(parents=True)
    lines = [{"id": "e1", "status": "open"}, {"id": "e2", "status": "open"}, {"id": "e1", "status": "closed"}, {"id": "e3", "status": "open"},
             {"id": "e2", "status": "evicted"}]
    (d / "events.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json\n", encoding="utf-8")
    evs = c.get("/api/events").json()["events"]
    assert [(e["id"], e["status"]) for e in evs] == [("e3", "open"), ("e1", "closed")]
    assert read_events(root / "nowhere") == []


def test_shutdown_hook_runs_when_the_app_stops(tmp_path):
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "index.html").write_text("x", encoding="utf-8")
    called = []
    web = SimpleNamespace(max_upload_mb=1, upload_dir="static/uploads", allowed_extensions=[".mp4"], allowed_source_dirs=["."],
                          allow_rtsp=False, allow_camera_index=False)
    app = create_app(Context(worker=FakeWorker(), records=[], web=web, root=tmp_path, evidence_dir=tmp_path, on_shutdown=lambda: called.append(1)))
    with TestClient(app) as c:
        assert c.get("/api/status").status_code == 200
    assert called == [1]


def test_backend_info_describes_runtime_and_device():
    from apps.web.routes import backend_info
    from sentinel.config import load_pipeline_config
    cfg = load_pipeline_config()
    cfg.device = "cpu"
    info = backend_info(cfg)
    assert info["runtime"] == "pytorch" and info["device"] == "cpu" and info["rules"] == "windowed" and "yolo11s.pt" in info["models"]
    cfg.models.coco.weights = "m/coco.onnx"; cfg.models.helmet.weights = "m/h.ONNX"; cfg.models.plate.weights = "m/p.onnx"
    assert backend_info(cfg)["runtime"] == "onnxruntime"
    cfg.models.plate.weights = "plate_best.pt"
    assert backend_info(cfg)["runtime"] == "mixed"


def test_events_get_browser_urls_only_when_the_evidence_folder_is_served(env, tmp_path):
    from apps.web.routes import with_urls
    ev = [{"id": "e1", "evidence": {"frame": "e1/frame.jpg", "vehicle": "e1/vehicle.jpg", "plate": None}}]
    assert with_urls([dict(e) for e in ev], "/static/violations/")[0]["urls"] == {"frame": "/static/violations/e1/frame.jpg",
                                                                                 "vehicle": "/static/violations/e1/vehicle.jpg", "plate": None}
    assert with_urls([dict(e) for e in ev], None)[0]["urls"] == {"frame": None, "vehicle": None, "plate": None}


def test_index_page_has_the_panels_and_never_injects_server_data_as_html():
    html = (Path(__file__).resolve().parent.parent / "templates" / "index.html").read_text(encoding="utf-8")
    for needle in ('id="camera_input"', 'id="video_upload"', 'id="events"', 'id="hud"', "/api/events", "/api/status", "/update_camera", "/upload_video"):
        assert needle in html, needle
    assert "innerHTML" not in html                       # event fields are rendered with textContent only


# ---- events panel: the current run by default, every run on request; the run number is shown
def write_events(root, records):
    d = root / "static" / "violations"
    d.mkdir(parents=True, exist_ok=True)
    (d / "events.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def test_events_default_to_the_current_run_and_all_runs_on_request(env):
    c, worker, root = env
    worker.status = lambda: {"running": True, "source": "x", "frames_processed": 5, "loops": 2, "run": 3}
    write_events(root, [{"id": f"e{i}", "run": run, "status": "closed", "type": "no_helmet", "track_id": f"M_{i}"} for i, run in enumerate([1, 1, 2, 3, 3, 3])])
    cur = c.get("/api/events").json()
    assert cur["scope"] == "current" and cur["current_run"] == 3 and [e["id"] for e in cur["events"]] == ["e5", "e4", "e3"]
    allr = c.get("/api/events?scope=all").json()
    assert allr["scope"] == "all" and len(allr["events"]) == 6 and {e["run"] for e in allr["events"]} == {1, 2, 3}
    assert all("run" in e for e in allr["events"])


def test_without_a_known_run_every_event_is_listed(env):
    c, worker, root = env
    write_events(root, [{"id": "a", "run": 1, "status": "closed"}, {"id": "b", "run": 2, "status": "closed"}])
    assert len(c.get("/api/events").json()["events"]) == 2                       # FakeWorker.status() has no 'run': nothing to filter by


def test_the_page_shows_run_numbers_with_a_show_all_button_and_does_not_smooth_small_crops():
    html = (Path(__file__).resolve().parent.parent / "templates" / "index.html").read_text(encoding="utf-8")
    assert "scope=" in html and "toggleEventScope" in html and "show all runs" in html and "'run ' +" in html
    assert "image-rendering: pixelated" in html and "plate-thumb" in html and "object-fit: contain" in html
    assert "/video_feed" in html and "api/events" in html                           # evidence thumbnails come from the saved files, the stream only feeds the video element


def test_backend_info_names_the_helmet_model_and_its_sha256(tmp_path):
    import hashlib
    from apps.web.routes import backend_info
    from sentinel.config import load_pipeline_config
    cfg = load_pipeline_config(root=tmp_path)
    w = tmp_path / "m" / "h.pt"
    w.parent.mkdir()
    w.write_bytes(b"weights")
    cfg.models.helmet.weights = "m/h.pt"
    info = backend_info(cfg)["helmet_model"]
    assert info == {"file": "m/h.pt", "sha256": hashlib.sha256(b"weights").hexdigest()}
    cfg.models.helmet.weights = "m/missing.pt"
    assert backend_info(cfg)["helmet_model"]["sha256"] is None                 # not there: null, no crash
    d = load_pipeline_config()
    assert backend_info(d)["helmet_model"]["file"] == "experiments/EXP-015-helmet-grouped-3cls/weights/best.pt"
