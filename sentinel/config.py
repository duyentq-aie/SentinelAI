"""Typed configuration loaded from configs/*.yaml (unknown keys are errors, values are validated)."""
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent


@dataclass
class ModelCfg:
    weights: str
    conf: float
    tracker: Optional[str] = None      # path to an Ultralytics tracker yaml; None = no tracking


@dataclass
class ModelsCfg:
    coco: ModelCfg = field(default_factory=lambda: ModelCfg("yolo11s.pt", 0.35, "configs/trackers/botsort.yaml"))
    helmet: ModelCfg = field(default_factory=lambda: ModelCfg("helmet_best.pt", 0.55, "configs/trackers/botsort.yaml"))
    plate: ModelCfg = field(default_factory=lambda: ModelCfg("plate_best.pt", 0.35))


@dataclass
class HelmetRule:
    violation_frames: int = 5


@dataclass
class OverloadRule:
    max_passengers: int = 2


@dataclass
class WrongWayRule:
    enabled: bool = False
    history_len: int = 30
    min_history: int = 15
    dy_threshold: float = -30


@dataclass
class RedLightRule:
    enabled: bool = True


@dataclass
class RulesCfg:
    helmet: HelmetRule = field(default_factory=HelmetRule)
    overload: OverloadRule = field(default_factory=OverloadRule)
    wrong_way: WrongWayRule = field(default_factory=WrongWayRule)
    red_light: RedLightRule = field(default_factory=RedLightRule)


@dataclass
class PlateCfg:
    max_attempts: int = 90          # plate-detector runs per violating track; 0 = unlimited
    stop_when_found: bool = True    # False = keep detecting on every violating frame (v0 behaviour, used as the A/B reference)
    every: int = 1                  # try on every k-th frame of a violating track


@dataclass
class EvidenceCfg:
    dir: str = "static/violations"
    plate_margin: int = 10
    max_records: int = 12           # gallery length kept in memory
    max_dir_mb: int = 200           # oldest evidence is deleted when the folder exceeds this; 0 = unlimited
    jpeg_quality: int = 90
    close_after: int = 30           # frames without the violation before an event is closed
    plate: PlateCfg = field(default_factory=PlateCfg)


@dataclass
class WebCfg:
    """Demo server. Defaults are deliberately restrictive: loopback only, no RTSP, no camera index, uploads limited in type and size."""
    host: str = "127.0.0.1"
    port: int = 8000
    upload_dir: str = "static/uploads"
    max_upload_mb: int = 500
    allowed_extensions: List[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"])
    allowed_source_dirs: List[str] = field(default_factory=lambda: ["static/uploads", "."])   # "." = repo root (test_video.mp4)
    allow_rtsp: bool = False
    allow_camera_index: bool = False


@dataclass
class PipelineConfig:
    device: Optional[str] = None
    frame_size: List[int] = field(default_factory=lambda: [1280, 720])
    models: ModelsCfg = field(default_factory=ModelsCfg)
    rules: RulesCfg = field(default_factory=RulesCfg)
    evidence: EvidenceCfg = field(default_factory=EvidenceCfg)
    web: WebCfg = field(default_factory=WebCfg)
    root: Path = REPO                  # base for relative weights/tracker paths (not read from yaml)

    def validate(self):
        w, h = self.frame_size if len(self.frame_size) == 2 else (0, 0)
        if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
            raise ValueError(f"frame_size must be [width, height] positive ints, got {self.frame_size}")
        for name in ("coco", "helmet", "plate"):
            m = getattr(self.models, name)
            if not 0 < m.conf <= 1:
                raise ValueError(f"models.{name}.conf must be in (0, 1], got {m.conf}")
        r = self.rules
        if r.helmet.violation_frames < 1:
            raise ValueError("rules.helmet.violation_frames must be >= 1")
        if r.overload.max_passengers < 1:
            raise ValueError("rules.overload.max_passengers must be >= 1")
        ww = r.wrong_way
        if not 1 <= ww.min_history <= ww.history_len:
            raise ValueError("rules.wrong_way needs 1 <= min_history <= history_len")
        ev = self.evidence
        if ev.max_records < 1 or ev.plate_margin < 0:
            raise ValueError("evidence.max_records must be >= 1 and plate_margin >= 0")
        if ev.max_dir_mb < 0 or not 1 <= ev.jpeg_quality <= 100 or ev.close_after < 1:
            raise ValueError("evidence: max_dir_mb >= 0, 1 <= jpeg_quality <= 100, close_after >= 1")
        if ev.plate.max_attempts < 0 or ev.plate.every < 1:
            raise ValueError("evidence.plate: max_attempts >= 0 and every >= 1")
        w = self.web
        if not 1 <= w.port <= 65535 or w.max_upload_mb < 1 or not w.allowed_extensions or not w.allowed_source_dirs:
            raise ValueError("web: port in 1..65535, max_upload_mb >= 1, allowed_extensions and allowed_source_dirs non-empty")
        if any(not e.startswith(".") for e in w.allowed_extensions):
            raise ValueError("web.allowed_extensions entries must start with '.'")
        return self

    def path(self, p):
        """Resolve a config path (weights, tracker yaml) against the repo root; absolute paths are kept."""
        p = Path(p)
        return p if p.is_absolute() else self.root / p


@dataclass
class CameraConfig:
    camera_id: str = "camera"
    source: str = ""
    violation_zone: List[List[int]] = field(default_factory=list)
    wrong_lane_zone: Optional[List[List[int]]] = None

    def validate(self):
        for name in ("violation_zone", "wrong_lane_zone"):
            z = getattr(self, name)
            if z is None:
                continue
            if len(z) < 3 or any(len(p) != 2 for p in z):
                raise ValueError(f"{name} needs >= 3 [x, y] points, got {z}")
        if not self.violation_zone:
            raise ValueError("violation_zone is required")
        return self

    @staticmethod
    def polygon(points):
        return None if points is None else np.array(points, np.int32).reshape((-1, 1, 2))


def _build(cls, data, where):
    data = data or {}
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(data).__name__}")
    known = {f.name for f in fields(cls) if f.name != "root"}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(known)}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            t = hints[f.name]
            kwargs[f.name] = _build(t, data[f.name], f"{where}.{f.name}") if is_dataclass(t) else data[f.name]
    return cls(**kwargs)


def load_pipeline_config(path=None, root=REPO, device="__keep__"):
    path = Path(path) if path else REPO / "configs" / "pipeline" / "default.yaml"
    cfg = _build(PipelineConfig, yaml.safe_load(path.read_text(encoding="utf-8")), str(path))
    cfg.root = Path(root)
    if device != "__keep__":
        cfg.device = device
    return cfg.validate()


def load_camera_config(path):
    path = Path(path)
    return _build(CameraConfig, yaml.safe_load(path.read_text(encoding="utf-8")), str(path)).validate()
