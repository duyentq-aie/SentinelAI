"""Perception output: what the detectors/trackers saw in one frame.

Rules consume Perception only (no pixels), so they can be replayed offline from a recorded tracks file.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Track:
    id: int
    cls: str
    box: Tuple[int, int, int, int]          # x1, y1, x2, y2 in resized-frame pixels
    conf: float
    state: Optional[str] = None             # traffic lights only: RED / GREEN / UNKNOWN from the HSV classifier


@dataclass
class Perception:
    coco: List[Track] = field(default_factory=list)       # COCO model tracks (vehicles, traffic lights, ...)
    helmet: List[Track] = field(default_factory=list)     # helmet model tracks (motorcycle, helmet, no-helmet)
    lane_lights: dict = field(default_factory=dict)          # lane name -> state of that lane's own light ({roi: ...} signal): RED | GREEN | UNKNOWN
    light: str = "UNKNOWN"                                # state of the last traffic light with a usable crop

    # ---- compact JSON form: [id, cls, x1, y1, x2, y2, conf, state]
    def to_json(self, frame_idx):
        row = lambda t: [t.id, t.cls, *t.box, round(float(t.conf), 4), t.state]
        return {"f": frame_idx, "light": self.light, "coco": [row(t) for t in self.coco], "helmet": [row(t) for t in self.helmet]}

    @staticmethod
    def from_json(d):
        mk = lambda r: Track(r[0], r[1], (r[2], r[3], r[4], r[5]), r[6], r[7])
        return Perception(coco=[mk(r) for r in d["coco"]], helmet=[mk(r) for r in d["helmet"]], light=d["light"])


TRACK_FIELDS = ["id", "cls", "x1", "y1", "x2", "y2", "conf", "state"]


def tracks_from_results(results, names):
    """Ultralytics track() results -> [Track]. Detections without a track id are dropped (as in the original code)."""
    boxes = results[0].boxes
    if boxes.id is None:
        return []
    out = []
    for box, t_id, cls, conf in zip(boxes.xyxy.cpu(), boxes.id.int().cpu().tolist(),
                                    boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
        x1, y1, x2, y2 = map(int, box)
        out.append(Track(t_id, names[int(cls)], (x1, y1, x2, y2), conf))
    return out


def tracks_equal(a, b, box_tol=2, conf_tol=0.02):
    """Compare two [Track] lists: same ids/classes in the same order, boxes within box_tol px, conf within conf_tol."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if (x.id, x.cls, x.state) != (y.id, y.cls, y.state):
            return False
        if any(abs(p - q) > box_tol for p, q in zip(x.box, y.box)) or abs(x.conf - y.conf) > conf_tol:
            return False
    return True
