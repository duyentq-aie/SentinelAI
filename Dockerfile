# SentinelAI demo, CPU only (ONNX Runtime). Not baked into the image: model weights, ONNX files, videos (mounted at run time).
#
#   docker build -t sentinel-cpu .
#   docker run --rm -p 127.0.0.1:8000:8000 ^
#       -v %cd%\models:/app/models:ro -v %cd%\test_video.mp4:/app/test_video.mp4:ro sentinel-cpu
#   -> http://127.0.0.1:8000/     (publish on 127.0.0.1 only: the app inside the container listens on 0.0.0.0)
#
# models/onnx/*.onnx come from:  python tools/export_onnx.py --models coco helmet --imgsz 384 640 ; python tools/export_onnx.py --models plate --imgsz 640 --dynamic
# NOTE (2026-10-18): this file has NOT been built or run: Docker Desktop's engine was not running on the dev machine and its disk (C:) had < 6 GB free.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SENTINEL_CONFIG=/app/configs/pipeline/docker_cpu.yaml

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU wheels of torch (Ultralytics imports torch even when it only runs ONNX models), then the rest
RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt \
    && pip install --no-deps ultralytics==8.4.26

# code and configs only (see .dockerignore)
COPY app.py ./
COPY sentinel ./sentinel
COPY apps ./apps
COPY configs ./configs
COPY templates ./templates

RUN useradd --create-home --uid 1000 sentinel \
    && mkdir -p /app/static/uploads /app/static/violations /app/models /app/outputs \
    && chown -R sentinel:sentinel /app
USER sentinel

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=4)"

CMD ["python", "app.py"]
