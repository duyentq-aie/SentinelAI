"""Draft of a camera config (stop line, travel direction, traffic-light ROI, red-light zone): the logic behind tools/camera_config.py.

Writes configs/cameras/<id>.yaml. `stop_line`, `direction` and `light_roi` are consumed by the stop-line red-light rule (P2 3.3);
until that rule exists they are stored and validated but not used by the pipeline.
"""
from pathlib import Path

import yaml

MODES = ("stop_line", "direction", "light_roi", "violation_zone", "wrong_lane_zone")
POINTS_NEEDED = {"stop_line": 2, "direction": 2, "light_roi": 2, "violation_zone": None, "wrong_lane_zone": None}   # None = open polygon (>= 3)


def _pt(p, w, h):
    x, y = int(round(p[0])), int(round(p[1]))
    return [min(max(x, 0), w - 1), min(max(y, 0), h - 1)]


class CameraDraft:
    def __init__(self, camera_id, source, width=1280, height=720):
        self.camera_id, self.source, self.w, self.h = camera_id, source, width, height
        self.points = {m: [] for m in MODES}
        self.mode = "stop_line"

    # ---- editing
    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode

    def click(self, x, y):
        need = POINTS_NEEDED[self.mode]
        pts = self.points[self.mode]
        if need is not None and len(pts) >= need:
            pts.clear()                                   # a new click after a complete shape starts it over
        pts.append(_pt((x, y), self.w, self.h))

    def undo(self):
        pts = self.points[self.mode]
        return pts.pop() if pts else None

    def clear(self):
        self.points[self.mode].clear()

    # ---- completeness / output
    def problems(self):
        p = []
        for m in ("stop_line", "direction", "light_roi"):
            if self.points[m] and len(self.points[m]) != 2:
                p.append(f"{m}: needs exactly 2 points")
        if self.points["stop_line"] and len(self.points["stop_line"]) == 2 and self.points["stop_line"][0] == self.points["stop_line"][1]:
            p.append("stop_line: the two points are identical")
        if self.points["direction"] and len(self.points["direction"]) == 2 and self.points["direction"][0] == self.points["direction"][1]:
            p.append("direction: the arrow has zero length")
        lr = self.points["light_roi"]
        if len(lr) == 2 and (lr[0][0] == lr[1][0] or lr[0][1] == lr[1][1]):
            p.append("light_roi: the rectangle is empty")
        for m in ("violation_zone", "wrong_lane_zone"):
            if self.points[m] and len(self.points[m]) < 3:
                p.append(f"{m}: needs at least 3 points")
        if len(self.points["violation_zone"]) < 3:
            p.append("violation_zone is required by the pipeline (>= 3 points); draw it or keep the old one")
        return p

    def to_dict(self):
        d = {"camera_id": self.camera_id, "source": self.source, "frame_size": [self.w, self.h]}
        d["violation_zone"] = self.points["violation_zone"]
        d["wrong_lane_zone"] = self.points["wrong_lane_zone"] or None
        if len(self.points["stop_line"]) == 2:
            d["stop_line"] = self.points["stop_line"]
        if len(self.points["direction"]) == 2:
            d["direction"] = self.points["direction"]         # arrow tail -> head = direction of legal travel across the stop line
        if len(self.points["light_roi"]) == 2:
            (x1, y1), (x2, y2) = self.points["light_roi"]
            d["light_roi"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        return d

    def save(self, path):
        problems = [p for p in self.problems() if not p.startswith("violation_zone is required")] if self.points["violation_zone"] else self.problems()
        if problems:
            raise ValueError("; ".join(problems))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# written by tools/camera_config.py; points are [x, y] in resized-frame pixels\n"
                        + yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=None), encoding="utf-8")

    @classmethod
    def from_file(cls, path):
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        w, h = d.get("frame_size", [1280, 720])
        c = cls(d.get("camera_id", Path(path).stem), d.get("source", ""), w, h)
        c.points["violation_zone"] = [list(p) for p in d.get("violation_zone") or []]
        c.points["wrong_lane_zone"] = [list(p) for p in d.get("wrong_lane_zone") or []]
        c.points["stop_line"] = [list(p) for p in d.get("stop_line") or []]
        c.points["direction"] = [list(p) for p in d.get("direction") or []]
        r = d.get("light_roi")
        c.points["light_roi"] = [[r[0], r[1]], [r[2], r[3]]] if r else []
        return c
