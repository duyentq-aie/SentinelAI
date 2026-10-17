"""FastAPI demo server (MJPEG stream + violation gallery).

One InferenceWorker serves every client. Security hardening of the upload/camera endpoints is a
later P2 step; they are unchanged here.
"""
import os
import shutil

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from apps.web.worker import InferenceWorker
from sentinel.config import REPO, load_camera_config, load_pipeline_config
from sentinel.pipeline import Detectors, Pipeline, SharedState

PIPELINE_CONFIG = os.environ.get("SENTINEL_CONFIG", str(REPO / "configs" / "pipeline" / "default.yaml"))
CAMERA_CONFIG = os.environ.get("SENTINEL_CAMERA", str(REPO / "configs" / "cameras" / "test_video.yaml"))

cfg = load_pipeline_config(PIPELINE_CONFIG)
camera = load_camera_config(CAMERA_CONFIG)

os.makedirs(cfg.evidence.dir, exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

print("[*] Loading AI Models (Anti-Spam Tracking Edition)...")
state = SharedState()
pipeline = Pipeline(Detectors(cfg), state, cfg, camera)
worker = InferenceWorker(pipeline, camera.source)


def generate_frames():
    last_id = 0
    while True:
        item = worker.next_jpeg(last_id)
        if item is None:                      # no new frame yet (source starting / reconnecting)
            continue
        last_id, jpeg = item
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/update_camera")
async def update_camera(new_source: str = Form(...)):
    source = new_source.strip()
    return {"status": "success", "source": worker.set_source(int(source) if source.isdigit() else source)}


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    file_location = f"static/uploads/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    worker.set_source(file_location)
    return {"status": "success", "filename": file.filename}


@app.get("/api/violations")
async def get_violations():
    return {"images": state.violation_records}


@app.get("/api/status")
async def get_status():
    return worker.status()
