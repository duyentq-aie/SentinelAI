"""Frame -> annotated frame pipeline.

Stages: perceive (detect + track + traffic-light colour) -> rules (pure function of Perception) -> draw -> plate evidence.
P2 step 1 ported the old ``generate_frames()`` unchanged; step 2 moved settings to configs/; step 3.0 split perception from rules
so the rules can be replayed offline from a recorded tracks file (see sentinel/offline.py).

Two frames per step (since 3.0b): `raw` (resized camera frame, never drawn on) feeds every model and the evidence crops;
`display` (a copy) is what gets overlays and is streamed. Before 3.0b the helmet model saw traffic-light boxes and the plate model
saw violation overlays, because everything was drawn on one frame (fixture tracks_test_video_v0_overlay.jsonl keeps that behaviour).
"""
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
from sentinel.rules.legacy import LegacyRules, RuleResult, vehicle_label, violation_labels
from sentinel.viz import draw_sleek_box


class Detectors:
    """The three YOLO models, built from PipelineConfig."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.helmet = YOLO(str(cfg.path(cfg.models.helmet.weights)))
        self.coco = YOLO(str(cfg.path(cfg.models.coco.weights)))
        self.plate = YOLO(str(cfg.path(cfg.models.plate.weights)))
        self.kw = {} if cfg.device is None else {"device": cfg.device}

    def _track_kw(self, m):
        kw = dict(conf=m.conf, verbose=False, **self.kw)
        if m.tracker:
            kw["tracker"] = str(self.cfg.path(m.tracker))
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
    def __init__(self, detectors, state, cfg: PipelineConfig, camera: CameraConfig):
        self.d = detectors
        self.s = state
        self.cfg = cfg
        self.camera = camera
        self.rules = LegacyRules(cfg.rules, camera)
        self.frame_idx = 0
        self.recorder = EvidenceRecorder(cfg.evidence, detectors.plate_boxes, video_id=Path(camera.source).stem or camera.camera_id,
                                         camera_id=camera.camera_id, on_plate_saved=self._gallery_add, on_evicted=self._gallery_prune)
        self.prev_frame_time = 0
        self.violation_zone = CameraConfig.polygon(camera.violation_zone)
        self.bike_lane_zone = CameraConfig.polygon(camera.wrong_lane_zone)

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
    def _perceive(self, raw, display):
        p = Perception()
        p.coco = self.d.track_coco(raw)
        for t in p.coco:
            if t.cls == 'traffic light':
                x1, y1, x2, y2 = t.box
                crop = raw[y1:y2, x1:x2]
                if crop.size > 0:
                    t.state = p.light = detect_traffic_light_color(crop)
                    color = (0, 0, 255) if t.state == "RED" else (0, 255, 0)
                    draw_sleek_box(display, x1, y1, x2, y2, f"LIGHT: {t.state}", color, 2)
        p.helmet = self.d.track_helmet(raw)
        return p

    # ---- stage 3: drawing (same order as the original code)
    def _draw(self, frame, p, result):
        for hbox, h_type in result.heads:
            hx1, hy1, hx2, hy2 = hbox
            draw_sleek_box(frame, hx1, hy1, hx2, hy2, h_type.upper(), (255, 144, 30) if h_type == "helmet" else (0, 0, 255))

        zone_color = (0, 0, 255) if p.light == "RED" else (0, 255, 0)
        cv2.polylines(frame, [self.violation_zone], isClosed=True, color=zone_color, thickness=2)
        cv2.putText(frame, f"TRAFFIC STATUS: {p.light}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, zone_color, 3)

        for v in result.vehicles:
            x1, y1, x2, y2 = v.coords
            if v.violator:
                draw_sleek_box(frame, x1, y1, x2, y2, vehicle_label(v), (0, 0, 255), 2)
            else:
                draw_sleek_box(frame, x1, y1, x2, y2, vehicle_label(v), (0, 165, 255) if v.type == "motorcycle" else (255, 255, 0))

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
            draw_sleek_box(display, x1, y1, x2, y2, "PLATE", (0, 255, 0), thickness=2)
        self.frame_idx += 1

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = new_frame_time
        cv2.putText(display, f"FPS: {int(fps)}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        return FrameResult(frame=display, labels=violation_labels(result), captures=up.plates_found, perception=p, rules=result, raw=raw,
                           plate_calls=up.plate_calls, new_events=up.new_events)
