"""Which model files produced a result: file name + sha256 (cached by path, size and mtime so building many pipelines does not rehash the weights)."""
import hashlib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=64)
def _sha(path, size, mtime):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path):
    """sha256 of a file, or None when it is not there."""
    try:
        st = Path(path).stat()
        return _sha(str(path), st.st_size, st.st_mtime_ns)
    except OSError:
        return None


def plate_layer_status(cfg):
    """The plate layer is OPTIONAL: {"state": "enabled" | "disabled", "reason": None | str}. Disabled when models.plate.enabled is false or the weights file does not exist."""
    m = cfg.models.plate
    if not m.enabled:
        return {"state": "disabled", "reason": "models.plate.enabled is false"}
    if not Path(cfg.path(m.weights)).is_file():
        return {"state": "disabled", "reason": f"plate model file not found: {m.weights} (plate_best.pt is not published)"}
    return {"state": "enabled", "reason": None}


def models_info(cfg):
    """{"helmet": {"file", "sha256"}, "coco": {...}, "plate": {...}} of a PipelineConfig (sha256 None when the file is missing)."""
    return {n: {"file": str(getattr(cfg.models, n).weights), "sha256": file_sha256(cfg.path(getattr(cfg.models, n).weights))} for n in ("helmet", "coco", "plate")}
