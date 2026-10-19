"""Frame -> annotated frame pipeline.

Stages: perceive (detect + track + traffic-light colour) -> rules (pure function of Perception) -> draw -> plate evidence.
P2 step 1 ported the old ``generate_frames()`` unchanged; step 2 moved settings to configs/; step 3.0 split perception from rules
so the rules can be replayed offline from a recorded tracks file (see sentinel/offline.py).

Two frames per step (since 3.0b): `raw` (resized camera frame, never drawn on) feeds every model and the evidence crops;
`display` (a copy) is what gets overlays and is streamed. Before 3.0b the helmet model saw traffic-light boxes and the plate model
saw violation overlays, because everything was drawn on one frame (fixture tracks_test_video_v0_overlay.jsonl keeps that behaviour).
"""
import functools
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from sentinel.config import CameraConfig, PipelineConfig
from sentinel.evidence import EvidenceRecorder
from sentinel.lights import detect_traffic_light_color
from sentinel.perception import Perception, tracks_from_results
from sentinel.rules.legacy import RuleResult, vehicle_label, violation_labels
from sentinel.rules.windowed import make_rules
from sentinel.timebase import REFERENCE_FPS, to_frames
from sentinel.viz import draw_sleek_box as _draw_box


def build_ocr(cfg):
    """PaddleOCR-based PlateOCR from the config, or None (with a printed reason) if PaddleOCR cannot be loaded: evidence still works."""
    from sentinel.ocr.engine import OcrUnavailable, PaddleRecognizer, PlateOCR
    rec = PaddleRecognizer(cfg.evidence.ocr.model)
    try:
        rec._load()                                       # fail now, not in the middle of the first violation
    except OcrUnavailable as e:
        print(f"[!] OCR disabled: {e}")
        return None
    return PlateOCR(rec, reads=cfg.evidence.ocr.reads, model_name=cfg.evidence.ocr.model)


class Detectors:
    """The three YOLO models, built from PipelineConfig."""
    fps = REFERENCE_FPS                          # frame rate the trackers are configured for (class defaults so test doubles need not call __init__)
    _derived = None

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.helmet = YOLO(str(cfg.path(cfg.models.helmet.weights)))
        self.coco = YOLO(str(cfg.path(cfg.models.coco.weights)))
        self.plate = YOLO(str(cfg.path(cfg.models.plate.weights)))
        self.kw = {} if cfg.device is None else {"device": cfg.device}
        self._derived = {}                       # (tracker path, fps) -> tracker yaml with track_buffer rescaled

    def _tracker_path(self, tracker):
        """The tracker yaml to hand to Ultralytics. `track_buffer` is a frame count in the yaml (frames at 30 FPS = 1 s); at another frame rate a copy
        with the same DURATION is used. At 30 FPS the original file is used unchanged."""
        p = self.cfg.path(tracker)
        if abs(self.fps - REFERENCE_FPS) < 1e-9:
            return p
        key = (str(p), self.fps)
        if self._derived is None:
            self._derived = {}
        if key not in self._derived:
            text = p.read_text(encoding="utf-8")
            m = re.search(r"^track_buffer:\s*(\d+)", text, re.M)
            if m:
                frames = to_frames(int(m.group(1)) / REFERENCE_FPS, self.fps)
                text = text[:m.start()] + f"track_buffer: {frames}" + text[m.end():]
            d = Path(tempfile.mkdtemp(prefix="sentinel_tracker_"))
            out = d / p.name
            out.write_text(text, encoding="utf-8")
            self._derived[key] = out
        return self._derived[key]

    def set_fps(self, fps):
        """Frame rate the pipeline sees (source fps / stride). Re-creates the trackers (at the next frame) when their buffer has to change."""
        fps = float(fps)
        if abs(fps - self.fps) < 1e-9:
            return
        self.fps = fps
        for model in (self.coco, self.helmet):
            model.predictor = None               # the tracker is built from the yaml again on the next track() call

    def _track_kw(self, m):
        kw = dict(conf=m.conf, verbose=False, **self.kw)
        if m.tracker:
            kw["tracker"] = str(self._tracker_path(m.tracker))
        return kw

    def track_coco(self, frame):
        return tracks_from_results(self.coco.track(frame, persist=True, **self._track_kw(self.cfg.models.coco)), self.coco.names)

    def track_helmet(self, frame):
        return tracks_from_results(self.helmet.track(frame, persist=True, **self._track_kw(self.cfg.models.helmet)), self.helmet.names)

    def plate_boxes(self, crop):
        """Plate detections in a vehicle crop, best first: [(x1, y1, x2, y2, conf)] relative to the crop."""
        boxes = self.plate(crop, stream=False, **self._track_kw(self.cfg.models.plate))[0].boxes
        return [(*map(int, xyxy), float(conf)) for xyxy, conf in zip(boxes.xyxy, boxes.conf)]

    def reset_trackers(self):
        """Forget all tracks and restart ids (the predictor and the model stay loaded)."""
        for model in (self.coco, self.helmet):
            for tr in getattr(getattr(model, "predictor", None), "trackers", []) or []:
                tr.reset()


@dataclass
class SharedState:
    """Gallery of plate crops for the web UI (survives Pipeline.reset())."""
    violation_records: list = field(default_factory=list)


@dataclass
class FrameResult:
    frame: np.ndarray            # display frame (overlays) for streaming
    labels: list      # violation-looking labels drawn on this frame (characterization tests)
    captures: list    # track ids whose plate was found (first time) on this frame
    plate_calls: int = 0          # plate-detector runs on this frame
    new_events: list = field(default_factory=list)   # evidence event ids opened on this frame
    perception: Perception = None
    rules: RuleResult = None
    raw: np.ndarray = None       # clean resized frame (what the models saw; source of evidence crops)


class Pipeline:
    def __init__(self, detectors, state, cfg: PipelineConfig, camera: CameraConfig, ocr=None, fps=REFERENCE_FPS):
        self.d = detectors
        self.s = state
        self.cfg = cfg
        self.camera = camera
        self.fps = float(fps)
        self.rules = make_rules(cfg.rules, camera, self.fps)
        st = self.red_light_status()
        if st["state"] == "inactive":                       # first lines of the log of every run
            print(f"[!] RED-LIGHT RULE INACTIVE: {', '.join(st['reasons'])}. No red_light events will be reported for camera {camera.camera_id!r}.")
        # where the traffic-light colour comes from: a fixed region of the camera config (light_roi, read with OpenCV) or the COCO detector's `traffic light` boxes
        self.light_source = "roi" if camera.light_roi else "coco"
        print(f"[lights] traffic-light source: {self.light_source}" + (f" (light_roi {camera.light_roi})" if camera.light_roi else " (COCO detector; no light_roi in the camera config)"))
        if hasattr(detectors, "set_fps"):
            detectors.set_fps(self.fps)
        self._box = functools.partial(_draw_box, blend=cfg.render.blend)
        self.frame_idx = 0
        if ocr is None and cfg.evidence.ocr.enabled:
            ocr = build_ocr(cfg)
        self.recorder = EvidenceRecorder(cfg.evidence, detectors.plate_boxes, video_id=Path(camera.source).stem or camera.camera_id,
                                         camera_id=camera.camera_id, on_plate_saved=self._gallery_add, on_evicted=self._gallery_prune, ocr=ocr, fps=self.fps)
        self.prev_frame_time = 0
        self.violation_zone = CameraConfig.polygon(camera.violation_zone)
        self.bike_lane_zone = CameraConfig.polygon(camera.wrong_lane_zone)

    def set_fps(self, fps):
        """Tell the pipeline the frame rate of the frames it receives (source fps / stride): all durations of the config become frame counts at that rate.
        Call it when the source is (re)opened, before reset()."""
        fps = float(fps)
        if abs(fps - self.fps) < 1e-9:
            return
        self.fps = fps
        if hasattr(self.d, "set_fps"):
            self.d.set_fps(fps)
        self.rules.set_fps(fps)
        self.recorder.set_fps(fps)

    def reset(self):
        """Call when the source changes or a video loops: ids restart, so every id-keyed state must go."""
        self.rules.reset()
        self.recorder.reset()
        self.frame_idx = 0
        self.prev_frame_time = 0
        self.d.reset_trackers()

    def flush(self):
        """Close open evidence events (end of video / shutdown)."""
        self.recorder.flush()

    def _gallery_add(self, ev, path):
        recs = self.s.violation_records
        recs.insert(0, "/" + Path(path).as_posix())
        del recs[self.cfg.evidence.max_records:]

    def _gallery_prune(self, removed):
        self.s.violation_records[:] = [r for r in self.s.violation_records if not any(f"/{n}/" in r for n in removed)]

    # ---- stage 1: perception
    def _read_light_roi(self, raw, display):
        """State of the traffic light from the camera's fixed light_roi (no detector needed); '' when the ROI is outside the frame."""
        x1, y1, x2, y2 = self.camera.light_roi
        h, w = raw.shape[:2]
        crop = raw[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0:
            return "UNKNOWN"
        state = detect_traffic_light_color(crop)
        self._box(display, x1, y1, x2, y2, f"LIGHT: {state}", (0, 0, 255) if state == "RED" else (0, 255, 0), 2)
        return state

    def _perceive(self, raw, display):
        p = Perception()
        p.coco = self.d.track_coco(raw)
        if self.light_source == "roi":
            p.light = self._read_light_roi(raw, display)         # the COCO traffic-light boxes are ignored for the state
        for t in p.coco:
            if t.cls == 'traffic light' and self.light_source == "coco":
                x1, y1, x2, y2 = t.box
                crop = raw[y1:y2, x1:x2]
                if crop.size > 0:
                    t.state = p.light = detect_traffic_light_color(crop)
                    color = (0, 0, 255) if t.state == "RED" else (0, 255, 0)
                    self._box(display, x1, y1, x2, y2, f"LIGHT: {t.state}", color, 2)
        p.helmet = self.d.track_helmet(raw)
        return p

    def red_light_status(self):
        """{"state": "active" | "inactive" | "disabled" | "legacy_zone", "mode", "reasons": [...], ...}: shown on the HUD and in /api/status."""
        r = self.cfg.rules.red_light
        if not r.enabled:
            return {"state": "disabled", "mode": r.mode, "reasons": []}
        slr = getattr(self.rules, "stop_line_rule", None)
        if r.mode == "legacy_zone" or slr is None:
            return {"state": "legacy_zone", "mode": r.mode, "reasons": []}
        st = slr.status()
        return {"state": "active" if st["active"] else "inactive", "mode": r.mode, "reasons": st["inactive_reasons"], "lanes": st["lanes"],
                "crossings": st["crossings"], "unassigned_crossings": st["unassigned_crossings"]}

    # ---- stage 3: drawing (same order as the original code)
    def _draw(self, frame, p, result):
        for hbox, h_type in result.heads:
            hx1, hy1, hx2, hy2 = hbox
            self._box(frame, hx1, hy1, hx2, hy2, h_type.upper(), (255, 144, 30) if h_type == "helmet" else (0, 0, 255))

        zone_color = (0, 0, 255) if p.light == "RED" else (0, 255, 0)
        if self.cfg.rules.red_light.mode == "legacy_zone":
            cv2.polylines(frame, [self.violation_zone], isClosed=True, color=zone_color, thickness=2)
        elif self.camera.stop_line:                                              # stop_line_crossing: the line (coloured by the light) and the lanes (grey = exempt)
            (ax, ay), (bx, by) = self.camera.stop_line
            cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), zone_color, 3)
            for lane in self.camera.lanes or []:
                cv2.polylines(frame, [CameraConfig.polygon(lane.polygon)], isClosed=True, color=(160, 160, 160) if lane.signal == "exempt" else (255, 200, 0), thickness=1)
        cv2.putText(frame, f"TRAFFIC STATUS: {p.light}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, zone_color, 3)
        rl = self.red_light_status()
        if rl["state"] == "inactive":
            cv2.putText(frame, f"RED-LIGHT RULE INACTIVE ({'; '.join(rl['reasons'])})", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for v in result.vehicles:
            x1, y1, x2, y2 = v.coords
            if v.violator:
                self._box(frame, x1, y1, x2, y2, vehicle_label(v), (0, 0, 255), 2)
            else:
                self._box(frame, x1, y1, x2, y2, vehicle_label(v), (0, 165, 255) if v.type == "motorcycle" else (255, 255, 0))

    def process(self, frame):
        raw = cv2.resize(frame, tuple(self.cfg.frame_size))      # models + evidence only ever see this
        display = raw.copy()                                     # overlays go here

        if self.bike_lane_zone is not None:
            cv2.polylines(display, [self.bike_lane_zone], isClosed=True, color=(0, 165, 255), thickness=2)
            cv2.putText(display, "FAKE BIKE LANE", (750, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        p = self._perceive(raw, display)
        result = self.rules.step(p)
        self._draw(display, p, result)
        up = self.recorder.update(self.frame_idx, raw, result.vehicles, p.light)
        for x1, y1, x2, y2 in up.plate_boxes:
            self._box(display, x1, y1, x2, y2, "PLATE", (0, 255, 0), thickness=2)
        self.frame_idx += 1

        new_frame_time = time.perf_counter()                # time.time() has ~15 ms resolution on Windows: two fast frames could share a tick (division by zero)
        dt = new_frame_time - self.prev_frame_time
        fps = 1 / dt if self.prev_frame_time > 0 and dt > 0 else 0
        self.prev_frame_time = new_frame_time
        cv2.putText(display, f"FPS: {int(fps)}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        return FrameResult(frame=display, labels=violation_labels(result), captures=up.plates_found, perception=p, rules=result, raw=raw,
                           plate_calls=up.plate_calls, new_events=up.new_events)
