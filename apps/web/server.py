"""FastAPI demo server (MJPEG stream + violation gallery).

Behaviour-preserving port of the web part of the old app.py. Known issues in this
layer (unsanitised upload name, arbitrary camera source, shared capture, ...) are
deliberately kept for now and fixed in a later P2 step.
"""
import os
import shutil
import threading

import cv2
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sentinel.pipeline import EVIDENCE_DIR, Detectors, Pipeline, SharedState

os.makedirs("static/violations", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

print("[*] Loading AI Models (Anti-Spam Tracking Edition)...")
detectors = Detectors()
state = SharedState()

current_source = "test_video.mp4"
cap = cv2.VideoCapture(current_source)
lock = threading.Lock()


def generate_frames():
    global cap, current_source
    pipeline = Pipeline(detectors, state, evidence_dir=EVIDENCE_DIR)

    while True:
        with lock:
            if cap is None or not cap.isOpened(): continue
            success, frame = cap.read()
            if not success:
                if isinstance(current_source, str) and not current_source.startswith("http"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        result = pipeline.process(frame)
        ret, buffer = cv2.imencode('.jpg', result.frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/update_camera")
async def update_camera(new_source: str = Form(...)):
    global cap, current_source
    with lock:
        if cap is not None: cap.release()
        if new_source.strip().isdigit():
            current_source = int(new_source.strip())
        else:
            current_source = new_source.strip()
        cap = cv2.VideoCapture(current_source)
        state.captured_violator_ids.clear()
    return {"status": "success", "source": str(current_source)}


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    global cap, current_source
    file_location = f"static/uploads/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with lock:
        if cap is not None: cap.release()
        current_source = file_location
        cap = cv2.VideoCapture(current_source)
        state.captured_violator_ids.clear()

    return {"status": "success", "filename": file.filename}


@app.get("/api/violations")
async def get_violations():
    return {"images": state.violation_records}
