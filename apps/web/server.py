"""Real server wiring: loads models, builds the pipeline + inference worker, and hands them to apps.web.routes.create_app.

`uvicorn app:app` (see app.py) imports `app` from here. Everything is created at import time because the models must be loaded once.
"""
import os
from pathlib import Path

from apps.web.routes import Context, backend_info, create_app
from apps.web.worker import InferenceWorker
from sentinel.config import REPO, load_camera_config, load_pipeline_config
from sentinel.pipeline import Detectors, Pipeline, SharedState

PIPELINE_CONFIG = os.environ.get("SENTINEL_CONFIG", str(REPO / "configs" / "pipeline" / "default.yaml"))
CAMERA_CONFIG = os.environ.get("SENTINEL_CAMERA", str(REPO / "configs" / "cameras" / "test_video.yaml"))

cfg = load_pipeline_config(PIPELINE_CONFIG)
camera = load_camera_config(CAMERA_CONFIG)

os.makedirs(cfg.evidence.dir, exist_ok=True)
os.makedirs(REPO / cfg.web.upload_dir, exist_ok=True)

print("[*] Loading AI Models (Anti-Spam Tracking Edition)...")
state = SharedState()
pipeline = Pipeline(Detectors(cfg), state, cfg, camera)
worker = InferenceWorker(pipeline, camera.source, jpeg_quality=cfg.web.jpeg_quality, backend=backend_info(cfg))


def _shutdown():
    worker.stop()
    pipeline.flush()                          # close open evidence events so events.jsonl is complete


def _evidence_url():
    """/static/... URL of the evidence folder when it lives under static/ (the only folder the app serves), else None."""
    try:
        return "/static/" + Path(cfg.evidence.dir).resolve().relative_to((REPO / "static").resolve()).as_posix()
    except ValueError:
        return None


app = create_app(Context(worker=worker, records=state.violation_records, web=cfg.web, root=REPO,
                         evidence_dir=Path(cfg.evidence.dir), on_shutdown=_shutdown, evidence_url=_evidence_url()))
