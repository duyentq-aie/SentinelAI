"""Typed configuration loaded from configs/*.yaml (unknown keys are errors, values are validated)."""
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, List, Optional

import warnings

import numpy as np
import yaml

from sentinel.timebase import REFERENCE_FPS, seconds_of

REPO = Path(__file__).resolve().parent.parent


@dataclass
class ModelCfg:
    weights: str
    conf: float
    tracker: Optional[str] = None      # path to an Ultralytics tracker yaml; None = no tracking
    enabled: bool = True               # only models.plate may be switched off (the plate layer is optional); a missing plate file disables it as well


@dataclass
class ModelsCfg:
    coco: ModelCfg = field(default_factory=lambda: ModelCfg("yolo11s.pt", 0.35, "configs/trackers/botsort.yaml"))
    helmet: ModelCfg = field(default_factory=lambda: ModelCfg("helmet_best.pt", 0.55, "configs/trackers/botsort.yaml"))
    plate: ModelCfg = field(default_factory=lambda: ModelCfg("plate_best.pt", 0.35))


@dataclass
class HelmetRule:
    """Durations are in SECONDS and turned into frames from the processing fps (sentinel/timebase.py). The integer fields marked LEGACY are the old
    frame-based settings: when present they win and mean frames at 30 FPS (the loader warns)."""
    violation_s: float = 5 / 30   # legacy engine: net no-helmet time (+1 seen / -1 not seen per frame) before flagging
    window_s: float = 0.5         # windowed engine: span of observations kept (observation = frame with a head attached to the motorcycle)
    min_vote_fraction: float = 1 / 3   # K / W: no-helmet votes needed inside the window, as a fraction of the window
    min_ratio: float = 0.5        # votes / observations needed in the window (already a ratio)
    exit_ratio: Optional[float] = None   # hysteresis: once flagged, stay flagged while ratio >= exit_ratio (None = same as min_ratio = no hysteresis)
    violation_frames: Optional[int] = None   # LEGACY (frames at 30 FPS)
    window: Optional[int] = None             # LEGACY (frames at 30 FPS)
    min_votes: Optional[int] = None          # LEGACY (K in frames; converted to K / W)


@dataclass
class OverloadRule:
    max_passengers: int = 2       # more heads than this on one motorcycle = overload vote
    window_s: float = 0.5
    min_vote_fraction: float = 1 / 3
    min_ratio: float = 0.5
    exit_ratio: Optional[float] = None
    window: Optional[int] = None             # LEGACY
    min_votes: Optional[int] = None          # LEGACY


@dataclass
class WrongWayRule:
    enabled: bool = False
    history_s: float = 1.0        # span of positions kept per track
    min_history_s: float = 0.5    # span needed before judging
    dy_threshold: float = -30     # px moved "up" the image over the history span (pixels, not time)
    history_len: Optional[int] = None        # LEGACY (frames at 30 FPS)
    min_history: Optional[int] = None        # LEGACY


@dataclass
class RedLightRule:
    enabled: bool = True
    mode: str = "stop_line_crossing"   # "stop_line_crossing": a motorcycle crossing the stop line (camera.stop_line / direction) while its lane's light is RED;
                                       # "legacy_zone": the v0 rule (bottom-centre inside violation_zone while RED: also flags stopped vehicles, knows no lanes)
    jitter_px: float = 4.0             # stop_line_crossing: dead zone of +-jitter_px around the line (detector jitter is not a crossing)
    hold_s: float = 1.0                # stop_line_crossing: the verdict is held this long after a crossing so the recorder opens an event with evidence
    merge_s: float = 1.0               # stop_line_crossing: two reportable crossings in the same lane less than merge_s apart ...
    merge_px: float = 40.0             # ... and whose crossing points are less than merge_px apart are ONE event (a track that broke in two: ID switch)
    link_iou: float = 0.3              # stop_line_crossing: a red-light track K_ is linked to a helmet-model track M_ when their boxes overlap with IoU >= link_iou, or one box centre lies inside the other box


@dataclass
class RulesCfg:
    engine: str = "windowed"      # "windowed" (P2 3.2 voting) or "legacy" (v0 counters, kept for baseline replays)
    helmet: HelmetRule = field(default_factory=HelmetRule)
    overload: OverloadRule = field(default_factory=OverloadRule)
    wrong_way: WrongWayRule = field(default_factory=WrongWayRule)
    red_light: RedLightRule = field(default_factory=RedLightRule)


@dataclass
class PlateCfg:
    search_s: float = 3.0           # plate search budget per violating track, in seconds of violation (runs allowed = search_s * fps / every); 0 = unlimited
    stop_when_found: bool = True    # False = keep detecting on every violating frame (v0 behaviour, used as the A/B reference)
    every_s: float = 0.0            # minimum time between two detector runs on a track (0 = every frame)
    max_attempts: Optional[int] = None       # LEGACY: runs per track (frames at 30 FPS when every = 1); 0 = unlimited
    every: Optional[int] = None              # LEGACY: every k-th frame


@dataclass
class OcrCfg:
    enabled: bool = False             # off by default: needs PaddleOCR (heavy import, ~model download on first use)
    candidates: int = 5               # plate crops collected per track before reading (the detector keeps running until then)
    min_gap_s: float = 0.1            # time between two collected crops (different views of the same plate)
    min_gap: Optional[int] = None     # LEGACY (frames at 30 FPS)
    reads: int = 3                    # best crops (by size x sharpness) actually read; the readings are voted
    margin: int = 2                   # px of background kept around the plate box in the crops that are READ (plate.jpg keeps evidence.plate_margin)
    model: str = "PP-OCRv5_mobile_rec"


@dataclass
class EvidenceCfg:
    dir: str = "static/violations"
    plate_margin: int = 10
    max_records: int = 12           # gallery length kept in memory
    max_dir_mb: int = 200           # oldest evidence is deleted when the folder exceeds this; 0 = unlimited
    jpeg_quality: int = 95            # saved evidence images; validate() enforces >= 95 (the MJPEG stream uses web.jpeg_quality)
    close_after_s: float = 1.0      # time without the violation before an event is closed
    box_sample_s: float = 5 / 30    # time between the vehicle boxes stored in event["boxes"]
    close_after: Optional[int] = None        # LEGACY (frames at 30 FPS)
    one_event_per_track: bool = True  # at most one event per (track, type) per run, even if the verdict flickers
    plate: PlateCfg = field(default_factory=PlateCfg)
    ocr: OcrCfg = field(default_factory=OcrCfg)


@dataclass
class RenderCfg:
    blend: str = "roi"            # label-tag blending: "roi" (only the tag rectangle) or "full" (v0: copy + blend the whole frame)


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
    jpeg_quality: int = 90        # MJPEG stream quality only (cv2 default is 95); saved evidence is always >= 95 (evidence.jpeg_quality). 80/90/95: EXP-020


@dataclass
class PipelineConfig:
    device: Optional[str] = None
    frame_size: List[int] = field(default_factory=lambda: [1280, 720])
    models: ModelsCfg = field(default_factory=ModelsCfg)
    rules: RulesCfg = field(default_factory=RulesCfg)
    evidence: EvidenceCfg = field(default_factory=EvidenceCfg)
    web: WebCfg = field(default_factory=WebCfg)
    render: RenderCfg = field(default_factory=RenderCfg)
    root: Path = REPO                  # base for relative weights/tracker paths (not read from yaml)

    def validate(self):
        w, h = self.frame_size if len(self.frame_size) == 2 else (0, 0)
        if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
            raise ValueError(f"frame_size must be [width, height] positive ints, got {self.frame_size}")
        for name in ("coco", "helmet", "plate"):
            m = getattr(self.models, name)
            if not 0 < m.conf <= 1:
                raise ValueError(f"models.{name}.conf must be in (0, 1], got {m.conf}")
            if not m.enabled and name != "plate":
                raise ValueError(f"models.{name}.enabled: false is not allowed, only the plate layer is optional")
        r = self.rules
        if r.engine not in ("legacy", "windowed"):
            raise ValueError(f"rules.engine must be 'legacy' or 'windowed', got {r.engine!r}")
        if (r.helmet.violation_frames is not None and r.helmet.violation_frames < 1) or r.helmet.violation_s <= 0:
            raise ValueError("rules.helmet.violation_s (or legacy violation_frames) must be positive")
        if r.overload.max_passengers < 1:
            raise ValueError("rules.overload.max_passengers must be >= 1")
        for name, rule in (("helmet", r.helmet), ("overload", r.overload)):
            if rule.window is not None or rule.min_votes is not None:
                w_ref = rule.window if rule.window is not None else round(rule.window_s * REFERENCE_FPS)
                k_ref = rule.min_votes if rule.min_votes is not None else 1
                if not 1 <= k_ref <= w_ref:
                    raise ValueError(f"rules.{name}: need 1 <= min_votes <= window")
            if rule.window_s <= 0 or not 0 < rule.min_vote_fraction <= 1:
                raise ValueError(f"rules.{name}: window_s must be > 0 and min_vote_fraction in (0, 1]")
            if not 0 < rule.min_ratio <= 1:
                raise ValueError(f"rules.{name}.min_ratio must be in (0, 1]")
            if rule.exit_ratio is not None and not 0 < rule.exit_ratio <= rule.min_ratio:
                raise ValueError(f"rules.{name}.exit_ratio must be in (0, min_ratio]")
        if r.red_light.mode not in ("stop_line_crossing", "legacy_zone"):
            raise ValueError("rules.red_light.mode must be 'stop_line_crossing' or 'legacy_zone'")
        if r.red_light.jitter_px < 0 or r.red_light.hold_s <= 0:
            raise ValueError("rules.red_light: jitter_px >= 0 and hold_s > 0")
        if r.red_light.merge_s < 0 or r.red_light.merge_px < 0 or not 0 <= r.red_light.link_iou <= 1:
            raise ValueError("rules.red_light: merge_s >= 0, merge_px >= 0 and 0 <= link_iou <= 1")
        ww = r.wrong_way
        hist, minh = seconds_of(ww.history_s, ww.history_len), seconds_of(ww.min_history_s, ww.min_history)
        if not 0 < minh <= hist:
            raise ValueError("rules.wrong_way needs 0 < min_history <= history (history_s / min_history_s)")
        ev = self.evidence
        if ev.max_records < 1 or ev.plate_margin < 0:
            raise ValueError("evidence.max_records must be >= 1 and plate_margin >= 0")
        if ev.max_dir_mb < 0 or not 95 <= ev.jpeg_quality <= 100 or seconds_of(ev.close_after_s, ev.close_after) <= 0 or ev.box_sample_s <= 0:
            raise ValueError("evidence: max_dir_mb >= 0, 95 <= jpeg_quality <= 100 (evidence is not stored at stream quality), close_after_s > 0, box_sample_s > 0")
        o = ev.ocr
        if o.margin < 0 or o.candidates < 1 or o.reads < 1 or seconds_of(o.min_gap_s, o.min_gap) <= 0 or o.reads > o.candidates:
            raise ValueError("evidence.ocr: candidates >= 1, min_gap_s > 0 and 1 <= reads <= candidates")
        pl = ev.plate
        if pl.search_s < 0 or pl.every_s < 0 or (pl.max_attempts is not None and pl.max_attempts < 0) or (pl.every is not None and pl.every < 1):
            raise ValueError("evidence.plate: search_s >= 0, every_s >= 0 (legacy: max_attempts >= 0, every >= 1)")
        w = self.web
        if not 1 <= w.port <= 65535 or w.max_upload_mb < 1 or not w.allowed_extensions or not w.allowed_source_dirs:
            raise ValueError("web: port in 1..65535, max_upload_mb >= 1, allowed_extensions and allowed_source_dirs non-empty")
        if not 1 <= w.jpeg_quality <= 100:
            raise ValueError("web.jpeg_quality must be in 1..100")
        if self.render.blend not in ("roi", "full"):
            raise ValueError("render.blend must be 'roi' or 'full'")
        if any(not e.startswith(".") for e in w.allowed_extensions):
            raise ValueError("web.allowed_extensions entries must start with '.'")
        return self

    def path(self, p):
        """Resolve a config path (weights, tracker yaml) against the repo root; absolute paths are kept."""
        p = Path(p)
        return p if p.is_absolute() else self.root / p


@dataclass
class Lane:
    """A lane of the approach for the stop-line red-light rule. `signal` is one of
      {roi: [x1, y1, x2, y2]}  the lane's own light (e.g. an arrow), read from that region with the same HSV classifier as light_roi
      "shared"                 the camera's light_roi (the old spelling "roi" means the same)
      "exempt"                 the lane has no light: the red-light rule does not apply (precision first)."""
    name: str
    polygon: List[List[int]]
    signal: Any = "shared"

    @property
    def kind(self):
        """"own" | "shared" | "exempt"."""
        return "own" if isinstance(self.signal, dict) else ("exempt" if self.signal == "exempt" else "shared")

    @property
    def roi(self):
        """[x1, y1, x2, y2] of the lane's own light, or None."""
        return list(self.signal["roi"]) if isinstance(self.signal, dict) else None


@dataclass
class CameraConfig:
    camera_id: str = "camera"
    source: str = ""
    violation_zone: List[List[int]] = field(default_factory=list)
    wrong_lane_zone: Optional[List[List[int]]] = None
    frame_size: List[int] = field(default_factory=lambda: [1280, 720])   # coordinates below are in this (resized) frame
    # drawn with tools/camera_config.py; used by the stop-line red-light rule (P2 3.3), stored and validated until then
    stop_line: Optional[List[List[int]]] = None       # two points
    direction: Optional[List[List[int]]] = None       # arrow tail -> head: legal direction of travel across the stop line
    light_roi: Optional[List[int]] = None             # [x1, y1, x2, y2] around the traffic light that governs this approach
    lanes: Optional[List[Lane]] = None                # lanes of the approach (stop_line_crossing); None = the shared signal applies to the whole stop line

    def validate(self):
        for name in ("violation_zone", "wrong_lane_zone"):
            z = getattr(self, name)
            if z is None:
                continue
            if len(z) < 3 or any(len(p) != 2 for p in z):
                raise ValueError(f"{name} needs >= 3 [x, y] points, got {z}")
        if not self.violation_zone:
            raise ValueError("violation_zone is required")
        for name in ("stop_line", "direction"):
            v = getattr(self, name)
            if v is not None and (len(v) != 2 or any(len(p) != 2 for p in v) or v[0] == v[1]):
                raise ValueError(f"{name} needs two distinct [x, y] points, got {v}")
        r = self.light_roi
        if r is not None and (len(r) != 4 or not (r[0] < r[2] and r[1] < r[3])):
            raise ValueError(f"light_roi must be [x1, y1, x2, y2] with x1 < x2 and y1 < y2, got {r}")
        if len(self.frame_size) != 2 or min(self.frame_size) < 1:
            raise ValueError("frame_size must be [width, height]")
        names = [l.name for l in self.lanes or []]
        if len(set(names)) != len(names):
            raise ValueError(f"lane names must be unique, got {names}")
        for l in self.lanes or []:
            if len(l.polygon) < 3 or any(len(p) != 2 for p in l.polygon):
                raise ValueError(f"lane {l.name!r}: polygon needs >= 3 [x, y] points")
            if isinstance(l.signal, dict):
                r = l.signal.get("roi")
                if set(l.signal) != {"roi"} or not isinstance(r, (list, tuple)) or len(r) != 4 or not (r[0] < r[2] and r[1] < r[3]):
                    raise ValueError(f"lane {l.name!r}: signal {{roi: [x1, y1, x2, y2]}} needs exactly the key 'roi' with x1 < x2 and y1 < y2, got {l.signal!r}")
            elif l.signal not in ("shared", "roi", "exempt"):
                raise ValueError(f"lane {l.name!r}: signal must be {{roi: [x1, y1, x2, y2]}}, 'shared' or 'exempt', got {l.signal!r}")
            elif l.signal == "roi":
                l.signal = "shared"                                # old spelling
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
            if is_dataclass(t):
                kwargs[f.name] = _build(t, data[f.name], f"{where}.{f.name}")
            elif typing.get_origin(t) is typing.Union and any(typing.get_origin(a) is list and a.__args__ and is_dataclass(a.__args__[0]) for a in typing.get_args(t)):
                item = next(a for a in typing.get_args(t) if typing.get_origin(a) is list).__args__[0]
                kwargs[f.name] = None if data[f.name] is None else [_build(item, x, f"{where}.{f.name}[{i}]") for i, x in enumerate(data[f.name])]
            else:
                kwargs[f.name] = data[f.name]
    return cls(**kwargs)


LEGACY_FRAME_KEYS = {"rules.helmet": ("violation_frames", "window", "min_votes"), "rules.overload": ("window", "min_votes"), "rules.wrong_way": ("history_len", "min_history"),
                     "evidence": ("close_after",), "evidence.plate": ("max_attempts", "every"), "evidence.ocr": ("min_gap",)}


def legacy_frame_keys(data):
    """Dotted names of the old frame-based keys present in a raw yaml mapping."""
    found = []
    for scope, keys in LEGACY_FRAME_KEYS.items():
        node = data
        for part in scope.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict):
            found += [f"{scope}.{k}" for k in keys if k in node]
    return found


def load_pipeline_config(path=None, root=REPO, device="__keep__"):
    path = Path(path) if path else REPO / "configs" / "pipeline" / "default.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    old = legacy_frame_keys(data or {})
    if old:
        warnings.warn(f"{path.name}: frame-based setting(s) {old} are read as frames at {REFERENCE_FPS:g} FPS; use the seconds / fraction keys "
                      f"(window_s, min_vote_fraction, close_after_s, search_s, every_s, min_gap_s, history_s, min_history_s, violation_s) so they follow the real frame rate",
                      UserWarning, stacklevel=2)
    cfg = _build(PipelineConfig, data, str(path))
    cfg.root = Path(root)
    if device != "__keep__":
        cfg.device = device
    return cfg.validate()


def load_camera_config(path):
    path = Path(path)
    return _build(CameraConfig, yaml.safe_load(path.read_text(encoding="utf-8")), str(path)).validate()
