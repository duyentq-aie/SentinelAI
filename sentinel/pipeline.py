"""Frame -> annotated frame pipeline.

Step 1 of the P2 refactor: a behaviour-preserving port of the old
``generate_frames()`` from app.py, including its known quirks (they are fixed in
later steps). Thresholds are still constants here; they move to YAML in step 2.
"""
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from sentinel.lights import detect_traffic_light_color
from sentinel.viz import draw_sleek_box

VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']
VIOLATION_LABELS = ("NO-HELMET", "OVERLOAD", "RED-LIGHT", "WRONG-WAY", "WRONG LANE")
EVIDENCE_DIR = "static/violations"


class Detectors:
    """The three YOLO models. device=None keeps Ultralytics' default device choice."""

    def __init__(self, weights_dir=".", device=None):
        wd = Path(weights_dir)
        self.helmet = YOLO(str(wd / "helmet_best.pt"))
        self.coco = YOLO(str(wd / "yolo11s.pt"))
        self.plate = YOLO(str(wd / "plate_best.pt"))
        self.kw = {} if device is None else {"device": device}


@dataclass
class SharedState:
    """State that, as in the original app.py, outlives a single video stream."""
    violation_counters: dict = field(default_factory=dict)
    violation_records: list = field(default_factory=list)
    captured_violator_ids: set = field(default_factory=set)


@dataclass
class FrameResult:
    frame: np.ndarray
    labels: list      # violation-looking labels drawn on this frame (characterization tests)
    captures: list    # track ids whose plate evidence was saved on this frame


class Pipeline:
    """Per-stream processor. Holds per-stream state (track_history, fps timer)."""

    def __init__(self, detectors, state, evidence_dir=EVIDENCE_DIR):
        self.d = detectors
        self.s = state
        self.evidence_dir = evidence_dir
        self.track_history = defaultdict(lambda: [])
        self.prev_frame_time = 0
        self.violation_zone = np.array([[250, 250], [750, 250], [820, 380], [150, 380]], np.int32).reshape((-1, 1, 2))
        # WRONG LANE logic exists but is disabled, as in the original.
        self.bike_lane_zone = None
        self._labels = []

    def _box(self, img, x1, y1, x2, y2, label, color, thickness=1):
        if any(r in label for r in VIOLATION_LABELS):
            self._labels.append(label)
        return draw_sleek_box(img, x1, y1, x2, y2, label, color, thickness)

    def process(self, frame):
        d, s = self.d, self.s
        self._labels = []
        captures = []
        track_history = self.track_history
        bike_lane_zone = self.bike_lane_zone
        violation_zone = self.violation_zone

        frame = cv2.resize(frame, (1280, 720))
        boxes_to_process = []
        current_light_state = "UNKNOWN"

        if bike_lane_zone is not None:
            cv2.polylines(frame, [bike_lane_zone], isClosed=True, color=(0, 165, 255), thickness=2)
            cv2.putText(frame, "FAKE BIKE LANE", (750, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # --- STAGE 1: COCO VEHICLES & TRAFFIC LIGHTS ---
        coco_results = d.coco.track(frame, persist=True, conf=0.35, verbose=False, **d.kw)

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
                    if len(track) > 30: track.pop(0)

                    is_violator = False
                    violation_reason = ""

                    # WRONG-WAY: any car moving "up" in the image
                    if len(track) >= 15:
                        old_cx, old_cy = track[0]
                        dy = cy - old_cy

                        if dy < -30:
                            is_violator = True
                            violation_reason = "WRONG-WAY"

                    # WRONG-LANE (cars/trucks/buses only)
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
        helmet_results = d.helmet.track(frame, persist=True, conf=0.55, verbose=False, **d.kw)
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

            if s.violation_counters[m_id] >= 5:
                is_violator = True
                violation_reason = "NO-HELMET"

            for hx1, hy1, hx2, hy2, h_type in associated_heads:
                head_color = (255, 144, 30) if h_type == "helmet" else (0, 0, 255)
                frame = self._box(frame, hx1, hy1, hx2, hy2, h_type.upper(), head_color)

            if passenger_count > 2:
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

            if current_light_state == "RED":
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
                    plate_results = d.plate(vehicle_crop, stream=False, conf=0.35, verbose=False, **d.kw)

                    for p_box in plate_results[0].boxes:
                        px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                        rx1, ry1 = vx1 + px1, vy1 + py1
                        rx2, ry2 = vx1 + px2, vy1 + py2

                        frame = self._box(frame, rx1, ry1, rx2, ry2, "PLATE", (0, 255, 0), thickness=2)

                        if v_id not in s.captured_violator_ids:
                            margin = 10
                            crop_y1, crop_y2 = max(0, ry1 - margin), min(frame.shape[0], ry2 + margin)
                            crop_x1, crop_x2 = max(0, rx1 - margin), min(frame.shape[1], rx2 + margin)
                            plate_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                            if plate_img.size > 0:
                                filename = f"v_{int(time.time())}_{v_id}.jpg"
                                filepath = f"{self.evidence_dir}/{filename}"
                                cv2.imwrite(filepath, plate_img)
                                s.violation_records.insert(0, f"/{filepath}")

                                s.captured_violator_ids.add(v_id)
                                captures.append(v_id)

                                if len(s.violation_records) > 12:
                                    s.violation_records.pop()

        # --- FPS OVERLAY ---
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        return FrameResult(frame=frame, labels=self._labels, captures=captures)
