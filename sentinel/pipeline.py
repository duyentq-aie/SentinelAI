"""Frame -> annotated frame pipeline.

Stages: perceive (detect + track + traffic-light colour) -> rules (pure function of Perception) -> draw -> plate evidence.
P2 step 1 ported the old ``generate_frames()`` unchanged; step 2 moved settings to configs/; step 3.0 split perception from rules
so the rules can be replayed offline from a recorded tracks file (see sentinel/offline.py).

Legacy quirk kept on purpose (perception must stay identical until it is deliberately changed): traffic-light boxes are drawn on the
frame *before* the helmet model runs, and violation overlays are drawn before the plate model sees the crop.
"""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

from sentinel.config import CameraConfig, PipelineConfig
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

    def detect_plate(self, crop):
        return self.plate(crop, stream=False, **self._track_kw(self.cfg.models.plate))

    def reset_trackers(self):
        """Forget all tracks and restart ids (the predictor and the model stay loaded)."""
        for model in (self.coco, self.helmet):
            for tr in getattr(getattr(model, "predictor", None), "trackers", []) or []:
                tr.reset()


@dataclass
class SharedState:
    """Evidence state. captured ids are cleared by Pipeline.reset(); the gallery list survives resets."""
    violation_records: list = field(default_factory=list)
    captured_violator_ids: set = field(default_factory=set)


@dataclass
class FrameResult:
    frame: np.ndarray
    labels: list      # violation-looking labels drawn on this frame (characterization tests)
    captures: list    # track ids whose plate evidence was saved on this frame
    perception: Perception = None
    rules: RuleResult = None


class Pipeline:
    def __init__(self, detectors, state, cfg: PipelineConfig, camera: CameraConfig):
        self.d = detectors
        self.s = state
        self.cfg = cfg
        self.camera = camera
        self.rules = LegacyRules(cfg.rules, camera)
        self.prev_frame_time = 0
        self.violation_zone = CameraConfig.polygon(camera.violation_zone)
        self.bike_lane_zone = CameraConfig.polygon(camera.wrong_lane_zone)

    def reset(self):
        """Call when the source changes or a video loops: ids restart, so every id-keyed state must go."""
        self.rules.reset()
        self.prev_frame_time = 0
        self.s.captured_violator_ids.clear()
        self.d.reset_trackers()

    # ---- stage 1: perception
    def _perceive(self, frame):
        p = Perception()
        p.coco = self.d.track_coco(frame)
        for t in p.coco:
            if t.cls == 'traffic light':
                x1, y1, x2, y2 = t.box
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    t.state = p.light = detect_traffic_light_color(crop)
                    color = (0, 0, 255) if t.state == "RED" else (0, 255, 0)
                    draw_sleek_box(frame, x1, y1, x2, y2, f"LIGHT: {t.state}", color, 2)   # legacy: drawn before the helmet model runs
        p.helmet = self.d.track_helmet(frame)
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

    # ---- stage 4: plates + evidence (still run for every violator on every frame, as in the original)
    def _plates(self, frame, vehicles):
        s, ev, captures = self.s, self.cfg.evidence, []
        for v in vehicles:
            if not v.violator:
                continue
            vx1, vy1, vx2, vy2 = v.coords
            vx1, vy1 = max(0, vx1), max(0, vy1)
            vx2, vy2 = min(frame.shape[1], vx2), min(frame.shape[0], vy2)
            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            if vehicle_crop.size == 0:
                continue
            plate_results = self.d.detect_plate(vehicle_crop)

            for p_box in plate_results[0].boxes:
                px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                rx1, ry1 = vx1 + px1, vy1 + py1
                rx2, ry2 = vx1 + px2, vy1 + py2

                draw_sleek_box(frame, rx1, ry1, rx2, ry2, "PLATE", (0, 255, 0), thickness=2)

                if v.id not in s.captured_violator_ids:
                    margin = ev.plate_margin
                    crop_y1, crop_y2 = max(0, ry1 - margin), min(frame.shape[0], ry2 + margin)
                    crop_x1, crop_x2 = max(0, rx1 - margin), min(frame.shape[1], rx2 + margin)
                    plate_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                    if plate_img.size > 0:
                        filename = f"v_{int(time.time())}_{v.id}.jpg"
                        filepath = f"{ev.dir}/{filename}"
                        cv2.imwrite(filepath, plate_img)
                        s.violation_records.insert(0, f"/{filepath}")

                        s.captured_violator_ids.add(v.id)
                        captures.append(v.id)

                        if len(s.violation_records) > ev.max_records:
                            s.violation_records.pop()
        return captures

    def process(self, frame):
        frame = cv2.resize(frame, tuple(self.cfg.frame_size))

        if self.bike_lane_zone is not None:
            cv2.polylines(frame, [self.bike_lane_zone], isClosed=True, color=(0, 165, 255), thickness=2)
            cv2.putText(frame, "FAKE BIKE LANE", (750, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        p = self._perceive(frame)
        result = self.rules.step(p)
        self._draw(frame, p, result)
        captures = self._plates(frame, result.vehicles)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        return FrameResult(frame=frame, labels=violation_labels(result), captures=captures, perception=p, rules=result)
