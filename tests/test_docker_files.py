"""The Dockerfile could not be built on the dev machine (engine off): these tests check the files statically so they cannot rot silently."""
import re

from sentinel.config import REPO, load_pipeline_config


def test_dockerfile_copies_only_existing_code_and_runs_unprivileged():
    d = (REPO / "Dockerfile").read_text(encoding="utf-8")
    for src in re.findall(r"^COPY (\S+) ", d, re.M):
        if src != "requirements-docker.txt":
            assert (REPO / src).exists(), src
    assert (REPO / "requirements-docker.txt").exists()
    assert "USER sentinel" in d and "SENTINEL_CONFIG=/app/configs/pipeline/docker_cpu.yaml" in d
    assert "NOT been built" in d                                  # honest status stays in the file until someone builds it


def test_dockerignore_keeps_weights_videos_data_and_secrets_out():
    ign = (REPO / ".dockerignore").read_text(encoding="utf-8").split()
    for must in ("*.pt", "*.onnx", "*.mp4", "data", ".git", "outputs"):
        assert must in ign, must


def test_docker_config_is_cpu_onnx_and_validates():
    c = load_pipeline_config(REPO / "configs" / "pipeline" / "docker_cpu.yaml")
    assert c.device == "cpu" and c.web.host == "0.0.0.0"
    assert all(str(m.weights).endswith(".onnx") for m in (c.models.coco, c.models.helmet, c.models.plate))


def test_requirements_docker_pins_the_runtime():
    r = (REPO / "requirements-docker.txt").read_text(encoding="utf-8")
    assert "onnxruntime==" in r and "opencv-python-headless" in r and "onnxruntime-gpu" not in r
