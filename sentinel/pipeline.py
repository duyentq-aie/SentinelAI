"""Frame -> annotated frame pipeline.

P2 step 1 ported the old ``generate_frames()`` without changing behaviour; step 2 moved its
thresholds/polygons/tracker settings into configs/ and added ``Pipeline.reset()`` so tracker and
per-track state no longer leak across sources or video loops. The detection rules themselves are
unchanged (they are rewritten in later steps), which the characterization tests verify.
"""
import time
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

from sentinel.config import CameraConfig, PipelineConfig
from sentinel.lights import detect_traffic_light_color
from sentinel.viz import draw_sleek_box

VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']
VIOLATION_LABELS = ("NO-HELMET", "OVERLOAD", "RED-LIGHT", "WRONG-WAY", "WRONG LANE")


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
        return self.coco.track(frame, persist=True, **self._track_kw(self.cfg.models.coco))

    def track_helmet(self, frame):
        return self.helmet.track(frame, persist=True, **self._track_kw(self.cfg.models.helmet))

    def detect_plate(self, crop):
        return self.plate(crop, stream=False, **self._track_kw(self.cfg.models.plate))

    def reset_trackers(self):
        """Forget all tracks and restart ids (the predictor and the model stay loaded)."""
        for model in (self.coco, self.helmet):
            for tr in getattr(getattr(model, "predictor", None), "trackers", []) or []:
                tr.reset()


@dataclass
class SharedState:
    """Per-source state: cleared by Pipeline.reset(). The gallery list survives resets."""
    violation_counters: dict = field(default_factory=dict)
    violation_records: list = field(default_factory=list)
    captured_violator_ids: set = field(default_factory=set)


@dataclass
class FrameResult:
    frame: np.ndarray
    labels: list      # violation-looking labels drawn on this frame (characterization tests)
    captures: list    # track ids whose plate evidence was saved on this frame


class Pipeline:
    def __init__(self, detectors, state, cfg: PipelineConfig, camera: CameraConfig):
        self.d = detectors
        self.s = state
        self.cfg = cfg
        self.camera = camera
        self.track_history = defaultdict(lambda: [])
        self.prev_frame_time = 0
        self.violation_zone = CameraConfig.polygon(camera.violation_zone)
        self.bike_lane_zone = CameraConfig.polygon(camera.wrong_lane_zone)
        self._labels = []

    def reset(self):
        """Call when the source changes or a video loops: ids restart, so every id-keyed state must go."""
        self.track_history.clear()
        self.prev_frame_time = 0
        self.s.violation_counters.clear()
        self.s.captured_violator_ids.clear()
        self.d.reset_trackers()

    def _box(self, img, x1, y1, x2, y2, label, color, thickness=1):
        if any(r in label for r in VIOLATION_LABELS):
            self._labels.append(label)
        return draw_sleek_box(img, x1, y1, x2, y2, label, color, thickness)

    def process(self, frame):
        d, s = self.d, self.s
        rules, ev = self.cfg.rules, self.cfg.evidence
        self._labels = []
        captures = []
        track_history = self.track_history
        bike_lane_zone = self.bike_lane_zone
        violation_zone = self.violation_zone

        frame = cv2.resize(frame, tuple(self.cfg.frame_size))
        boxes_to_process = []
        current_light_state = "UNKNOWN"

        if bike_lane_zone is not None:
            cv2.polylines(frame, [bike_lane_zone], isClosed=True, color=(0, 165, 255), thickness=2)
            cv2.putText(frame, "FAKE BIKE LANE", (750, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # --- STAGE 1: COCO VEHICLES & TRAFFIC LIGHTS ---
        coco_results = d.track_coco(frame)

        if coco_results[0].boxes.id is not None:
            boxes = coco_results[0].boxes.xyxy.cpu()
            track_ids = coco_results[0].boxes.id.int().cpu().tolist()
            clss = coco_results[0].boxes.cls.cpu().tolist()

            for box, t_id, cls in zip(boxes, track_ids, clss):
                x1, y1, x2, y2 = map(int, box)
                name = d.coco.names[int(cls)]

                if name == 'traffic light':
                    tl_crop = frame[y1:y2, x1:x2]
                    if tl_crop.size > 0:
                        current_light_state = detect_traffic_light_color(tl_crop)
                        color = (0, 0, 255) if current_light_state == "RED" else (0, 255, 0)
                        frame = self._box(frame, x1, y1, x2, y2, f"LIGHT: {current_light_state}", color, 2)

                elif name in VEHICLE_CLASSES and name != "motorcycle":
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    track = track_history[t_id]
                    track.append((cx, cy))
                    if len(track) > rules.wrong_way.history_len: track.pop(0)

                    is_violator = False
                    violation_reason = ""

                    # WRONG-WAY: any car moving "up" in the image (legacy rule, configurable)
                    if rules.wrong_way.enabled and len(track) >= rules.wrong_way.min_history:
                        old_cx, old_cy = track[0]
                        dy = cy - old_cy

                        if dy < rules.wrong_way.dy_threshold:
                            is_violator = True
                            violation_reason = "WRONG-WAY"

                    # WRONG-LANE (cars/trucks/buses only), active only if the camera defines a zone
                    if bike_lane_zone is not None:
                        bottom_center = (int((x1 + x2) / 2), int(y2))
                        if name in ['car', 'truck', 'bus']:
                            if cv2.pointPolygonTest(bike_lane_zone, bottom_center, False) >= 0:
                                is_violator = True
                                violation_reason = "WRONG LANE & WAY" if violation_reason == "WRONG-WAY" else "WRONG LANE"

                    boxes_to_process.append({
                        "id": f"C_{t_id}", "coords": (x1, y1, x2, y2),
                        "type": name, "violator": is_violator, "reason": violation_reason
                    })

        # --- STAGE 1.5: HELMETS, MOTORCYCLES & OVERLOAD ---
        helmet_results = d.track_helmet(frame)
        motorcycles, heads = [], []

        if helmet_results[0].boxes.id is not None:
            boxes = helmet_results[0].boxes.xyxy.cpu()
            track_ids = helmet_results[0].boxes.id.int().cpu().tolist()
            clss = helmet_results[0].boxes.cls.cpu().tolist()

            for box, t_id, cls in zip(boxes, track_ids, clss):
                x1, y1, x2, y2 = map(int, box)
                name = d.helmet.names[int(cls)]

                if name == "motorcycle":
                    motorcycles.append((x1, y1, x2, y2, t_id))
                elif name in ["helmet", "no-helmet"]:
                    heads.append((x1, y1, x2, y2, name))

        for mx1, my1, mx2, my2, m_id in motorcycles:
            is_violator = False
            violation_reason = ""
            passenger_count = 0
            associated_heads = []
            current_frame_no_helmet = False

            for hx1, hy1, hx2, hy2, h_type in heads:
                cx, cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
                if mx1 <= cx <= mx2 and my1 <= cy <= my2:
                    passenger_count += 1
                    associated_heads.append((hx1, hy1, hx2, hy2, h_type))
                    if h_type == "no-helmet":
                        current_frame_no_helmet = True

            if m_id not in s.violation_counters:
                s.violation_counters[m_id] = 0

            if current_frame_no_helmet:
                s.violation_counters[m_id] += 1
            else:
                s.violation_counters[m_id] = max(0, s.violation_counters[m_id] - 1)

            if s.violation_counters[m_id] >= rules.helmet.violation_frames:
                is_violator = True
                violation_reason = "NO-HELMET"

            for hx1, hy1, hx2, hy2, h_type in associated_heads:
                head_color = (255, 144, 30) if h_type == "helmet" else (0, 0, 255)
                frame = self._box(frame, hx1, hy1, hx2, hy2, h_type.upper(), head_color)

            if passenger_count > rules.overload.max_passengers:
                is_violator = True
                violation_reason = "OVERLOAD & NO-HELMET" if violation_reason == "NO-HELMET" else "OVERLOAD"

            boxes_to_process.append({
                "id": f"M_{m_id}", "coords": (mx1, my1, mx2, my2),
                "type": "motorcycle", "violator": is_violator, "reason": violation_reason
            })

        # --- RED LIGHT VIOLATION CHECK ---
        zone_color = (0, 0, 255) if current_light_state == "RED" else (0, 255, 0)
        cv2.polylines(frame, [violation_zone], isClosed=True, color=zone_color, thickness=2)
        cv2.putText(frame, f"TRAFFIC STATUS: {current_light_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, zone_color, 3)

        for idx, vehicle in enumerate(boxes_to_process):
            vx1, vy1, vx2, vy2 = vehicle["coords"]
            v_id = vehicle["id"]

            if rules.red_light.enabled and current_light_state == "RED":
                bottom_center = (int((vx1 + vx2) / 2), int(vy2))
                if cv2.pointPolygonTest(violation_zone, bottom_center, False) >= 0:
                    boxes_to_process[idx]["violator"] = True
                    boxes_to_process[idx]["reason"] = "RED-LIGHT RUNNER"
                    frame = self._box(frame, vx1, vy1, vx2, vy2, f"ID:{v_id} RED-LIGHT", (0, 0, 255), 2)

            if not boxes_to_process[idx]["violator"]:
                box_color = (0, 165, 255) if vehicle["type"] == "motorcycle" else (255, 255, 0)
                frame = self._box(frame, vx1, vy1, vx2, vy2, f"ID:{v_id} {vehicle['type'].upper()}", box_color)
            elif boxes_to_process[idx]["reason"] != "RED-LIGHT RUNNER":
                frame = self._box(frame, vx1, vy1, vx2, vy2, f"ID:{v_id} {boxes_to_process[idx]['reason']}", (0, 0, 255), 2)

        # --- STAGE 2: LICENSE PLATES (ANTI-SPAM CAPTURE) ---
        for vehicle in boxes_to_process:
            vx1, vy1, vx2, vy2 = vehicle["coords"]
            is_violator = vehicle["violator"]
            v_id = vehicle["id"]

            if is_violator:
                vx1, vy1 = max(0, vx1), max(0, vy1)
                vx2, vy2 = min(frame.shape[1], vx2), min(frame.shape[0], vy2)
                vehicle_crop = frame[vy1:vy2, vx1:vx2]

                if vehicle_crop.size > 0:
                    plate_results = d.detect_plate(vehicle_crop)

                    for p_box in plate_results[0].boxes:
                        px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                        rx1, ry1 = vx1 + px1, vy1 + py1
                        rx2, ry2 = vx1 + px2, vy1 + py2

                        frame = self._box(frame, rx1, ry1, rx2, ry2, "PLATE", (0, 255, 0), thickness=2)

                        if v_id not in s.captured_violator_ids:
                            margin = ev.plate_margin
                            crop_y1, crop_y2 = max(0, ry1 - margin), min(frame.shape[0], ry2 + margin)
                            crop_x1, crop_x2 = max(0, rx1 - margin), min(frame.shape[1], rx2 + margin)
                            plate_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                            if plate_img.size > 0:
                                filename = f"v_{int(time.time())}_{v_id}.jpg"
                                filepath = f"{ev.dir}/{filename}"
                                cv2.imwrite(filepath, plate_img)
                                s.violation_records.insert(0, f"/{filepath}")

                                s.captured_violator_ids.add(v_id)
                                captures.append(v_id)

                                if len(s.violation_records) > ev.max_records:
                                    s.violation_records.pop()

        # --- FPS OVERLAY ---
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        return FrameResult(frame=frame, labels=self._labels, captures=captures)
