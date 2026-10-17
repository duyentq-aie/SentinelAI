"""Entrypoint kept for compatibility: `python app.py` / `uvicorn app:app`.

The code lives in sentinel/ (pipeline) and apps/web/ (FastAPI demo). The original single-file version is tagged `v0-baseline`.
Binds to 127.0.0.1 by default (web.host in configs/pipeline/default.yaml); there is no auto-reload.
"""
import uvicorn

from apps.web.server import app, cfg  # noqa: F401  (app is re-exported for `uvicorn app:app`)

if __name__ == "__main__":
    print(f"[*] Starting SentinelAI FastApi Server on http://{cfg.web.host}:{cfg.web.port}/")
    uvicorn.run(app, host=cfg.web.host, port=cfg.web.port, reload=False)
