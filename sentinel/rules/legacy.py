"""The original (v0-baseline) violation rules, ported verbatim onto Perception.

Behaviour is intentionally unchanged, quirks included (see AUDIT section 2.E): wrong-way = any car moving up the
image, red light = vehicle bottom-centre inside a fixed zone while the light is RED, no-helmet = running counter.
They are replaced by the new rules in P2 steps 3.2/3.3; this module stays as the reference for diffs.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import cv2

from sentinel.config import CameraConfig
from sentinel.perception import Perception

VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']
VIOLATION_LABELS = ("NO-HELMET", "OVERLOAD", "RED-LIGHT", "WRONG-WAY", "WRONG LANE")


@dataclass
class Vehicle:
    id: str                                  # "C_<track>" (COCO vehicle) or "M_<track>" (helmet-model motorcycle)
    coords: Tuple[int, int, int, int]
    type: str
    violator: bool = False
    reason: str = ""


@dataclass
class RuleResult:
    vehicles: List[Vehicle]
    heads: List[Tuple[Tuple[int, int, int, int], str]]    # heads attached to a motorcycle, in drawing order: (box, "helmet"|"no-helmet")


def vehicle_label(v):
    """Label drawn on a vehicle box (also what characterization events are made of)."""
    if v.reason == "RED-LIGHT RUNNER":
        return f"ID:{v.id} RED-LIGHT"
    if not v.violator:
        return f"ID:{v.id} {v.type.upper()}"
    return f"ID:{v.id} {v.reason}"


def violation_labels(result):
    """All violation-looking labels the frame would show: head labels first, then vehicle labels."""
    labels = [h_type.upper() for _, h_type in result.heads] + [vehicle_label(v) for v in result.vehicles]
    return [l for l in labels if any(r in l for r in VIOLATION_LABELS)]


class LegacyRules:
    def __init__(self, rules_cfg, camera):
        self.cfg = rules_cfg
        self.violation_zone = CameraConfig.polygon(camera.violation_zone)
        self.bike_lane_zone = CameraConfig.polygon(camera.wrong_lane_zone)
        self.track_history = defaultdict(lambda: [])
        self.violation_counters = {}

    def reset(self):
        self.track_history.clear()
        self.violation_counters.clear()

    def step(self, p: Perception) -> RuleResult:
        rules = self.cfg
        vehicles: List[Vehicle] = []
        heads_drawn = []

        # --- COCO vehicles (cars/trucks/buses; COCO motorcycles are ignored) ---
        for t in p.coco:
            if t.cls in VEHICLE_CLASSES and t.cls != "motorcycle":
                x1, y1, x2, y2 = t.box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                track = self.track_history[t.id]
                track.append((cx, cy))
                if len(track) > rules.wrong_way.history_len: track.pop(0)

                violator, reason = False, ""
                if rules.wrong_way.enabled and len(track) >= rules.wrong_way.min_history:
                    old_cx, old_cy = track[0]
                    if cy - old_cy < rules.wrong_way.dy_threshold:
                        violator, reason = True, "WRONG-WAY"

                if self.bike_lane_zone is not None and t.cls in ['car', 'truck', 'bus']:
                    if cv2.pointPolygonTest(self.bike_lane_zone, (int((x1 + x2) / 2), int(y2)), False) >= 0:
                        violator = True
                        reason = "WRONG LANE & WAY" if reason == "WRONG-WAY" else "WRONG LANE"

                vehicles.append(Vehicle(f"C_{t.id}", t.box, t.cls, violator, reason))

        # --- helmet model: motorcycles with their heads ---
        motorcycles = [t for t in p.helmet if t.cls == "motorcycle"]
        heads = [(t.box, t.cls) for t in p.helmet if t.cls in ("helmet", "no-helmet")]

        for m in motorcycles:
            mx1, my1, mx2, my2 = m.box
            violator, reason = False, ""
            associated, no_helmet_now = [], False
            for hbox, h_type in heads:
                hx1, hy1, hx2, hy2 = hbox
                cx, cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
                if mx1 <= cx <= mx2 and my1 <= cy <= my2:
                    associated.append((hbox, h_type))
                    if h_type == "no-helmet":
                        no_helmet_now = True
            passengers = len(associated)

            c = self.violation_counters.get(m.id, 0)
            self.violation_counters[m.id] = c + 1 if no_helmet_now else max(0, c - 1)
            if self.violation_counters[m.id] >= rules.helmet.violation_frames:
                violator, reason = True, "NO-HELMET"

            heads_drawn.extend(associated)

            if passengers > rules.overload.max_passengers:
                violator = True
                reason = "OVERLOAD & NO-HELMET" if reason == "NO-HELMET" else "OVERLOAD"

            vehicles.append(Vehicle(f"M_{m.id}", m.box, "motorcycle", violator, reason))

        # --- red light: bottom-centre inside the zone while the light is RED ---
        if rules.red_light.enabled and p.light == "RED":
            for v in vehicles:
                vx1, vy1, vx2, vy2 = v.coords
                if cv2.pointPolygonTest(self.violation_zone, (int((vx1 + vx2) / 2), int(vy2)), False) >= 0:
                    v.violator, v.reason = True, "RED-LIGHT RUNNER"

        return RuleResult(vehicles, heads_drawn)
