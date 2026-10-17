"""Entrypoint kept for compatibility: `python app.py` / `uvicorn app:app`.

The code now lives in sentinel/ (pipeline) and apps/web/ (FastAPI demo).
The original single-file version is tagged `v0-baseline`.
"""
import uvicorn

from apps.web.server import app  # noqa: F401  (re-exported for uvicorn)

if __name__ == "__main__":
    print("[*] Starting SentinelAI FastApi Server...")
    # http://127.0.0.1:8000/
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
