"""FastAPI routes of the demo. Built by `create_app` from plain objects so tests can use fakes (no models, no GPU)."""
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sentinel.security import SourceRejected, UploadRejected, UploadTooLarge, save_upload, validate_source


@dataclass
class Context:
    worker: Any            # InferenceWorker-like: set_source(src) -> str, next_jpeg(last_id), status() -> dict
    records: list          # gallery of plate crops (SharedState.violation_records)
    web: Any               # WebCfg
    root: Path             # repo root (templates/, static/, allowed source folders are relative to it)
    evidence_dir: Path     # where events.jsonl lives
    on_shutdown: Any = None


def read_events(evidence_dir, limit=30):
    """Last record per event id from events.jsonl, newest first, evicted events dropped."""
    f = Path(evidence_dir) / "events.jsonl"
    if not f.exists():
        return []
    last, order = {}, []
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r["id"] not in last:
            order.append(r["id"])
        last[r["id"]] = r
    evs = [last[i] for i in order if last[i].get("status") != "evicted"]
    return list(reversed(evs))[:limit]


def create_app(ctx: Context) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        yield
        if ctx.on_shutdown:
            ctx.on_shutdown()

    app = FastAPI(lifespan=lifespan)
    root = Path(ctx.root)
    (root / "static").mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(root / "static")), name="static")
    templates = Jinja2Templates(directory=str(root / "templates"))
    max_bytes = ctx.web.max_upload_mb * 1024 * 1024

    def generate_frames():
        last_id = 0
        while True:
            item = ctx.worker.next_jpeg(last_id)
            if item is None:
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
        try:
            source = validate_source(new_source, ctx.web, root)
        except SourceRejected as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"status": "success", "source": ctx.worker.set_source(source)}

    @app.post("/upload_video")
    async def upload_video(file: UploadFile = File(...)):
        try:
            saved = save_upload(file.file, file.filename, root / ctx.web.upload_dir, max_bytes, ctx.web.allowed_extensions)
        except UploadTooLarge as e:
            raise HTTPException(status_code=413, detail=str(e))
        except UploadRejected as e:
            raise HTTPException(status_code=400, detail=str(e))
        ctx.worker.set_source(str(saved))
        return {"status": "success", "filename": saved.name}

    @app.get("/api/violations")
    async def get_violations():
        return {"images": ctx.records}

    @app.get("/api/status")
    async def get_status():
        return ctx.worker.status()

    @app.get("/api/events")
    async def get_events(limit: int = 30):
        return {"events": read_events(ctx.evidence_dir, max(1, min(limit, 200)))}

    return app
