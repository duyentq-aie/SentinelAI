"""Real server wiring: loads models, builds the pipeline + inference worker, and hands them to apps.web.routes.create_app.

`uvicorn app:app` (see app.py) imports `app` from here. Everything is created at import time because the models must be loaded once.
"""
import os
from pathlib import Path

from apps.web.routes import Context, create_app
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
worker = InferenceWorker(pipeline, camera.source)


def _shutdown():
    worker.stop()
    pipeline.flush()                          # close open evidence events so events.jsonl is complete


app = create_app(Context(worker=worker, records=state.violation_records, web=cfg.web, root=REPO,
                         evidence_dir=Path(cfg.evidence.dir), on_shutdown=_shutdown))
