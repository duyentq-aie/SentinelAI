"""Red-light rule "stop_line_crossing": a motorcycle violates when the bottom-centre of its track CROSSES the stop line in the direction of travel
while the signal of its lane is RED.

Geometry. The stop line is a segment A-B; the direction of travel is a vector (camera.direction, tail -> head). The signed distance of a point P to the line is
s = n . (P - A), n = unit normal of the line oriented so that n . direction > 0: s < 0 before the line, s > 0 beyond it. Each track keeps a SIDE with a dead zone
of +-jitter_px (hysteresis, against detector jitter): side = -1 when s <= -jitter, +1 when s >= +jitter, otherwise unchanged. A crossing is the side flipping from -1 to +1.
It is a violation only if, in that very frame, the light is RED, the crossing point lies on the segment and its lane is not "exempt".
Not violations: a stationary vehicle (no flip), a vehicle already beyond the line when the light turned red (no flip while RED), travelling the other way (+1 -> -1),
jitter inside the dead zone, a crossing on green. At most one violation per track and run.
The verdict is held for `hold_s` seconds after the crossing so the evidence recorder opens one event with the vehicle box and plate search.

Lanes (camera.lanes): polygons with signal "roi" (the shared light_roi applies) or "exempt" (no red-light rule, e.g. a right-turn lane with its own signal). A vehicle belongs to the
lane that contains the crossing point. With NO lanes configured the shared signal applies to the whole line (a warning is logged). With lanes configured, a crossing outside every
lane is not reported (it cannot be attributed to a signal) and is counted in `unassigned_crossings`.
"""
import math

import cv2

from sentinel.timebase import REFERENCE_FPS, to_frames

PURGE_S = 10.0           # forget tracks not seen for this long
SEGMENT_MARGIN = 0.02    # a crossing may lie this fraction beyond the ends of the segment


def join_reason(existing, new):
    """Reasons are ' & '-joined so a vehicle can carry several violations at once (the recorder splits them again)."""
    return new if not existing else (existing if new in existing.split(" & ") else f"{existing} & {new}")


class StopLineCrossing:
    REASON = "RED-LIGHT RUNNER"

    def __init__(self, camera, red_cfg, fps=REFERENCE_FPS, log=print):
        self.cfg, self.log = red_cfg, log
        self.lanes = list(camera.lanes or [])
        self.active = bool(camera.stop_line and camera.direction)
        self.warnings = []
        if not self.active:
            self._warn("red_light: stop_line_crossing needs camera.stop_line and camera.direction; the rule is INACTIVE (no red-light events)")
        else:
            (ax, ay), (bx, by) = camera.stop_line
            (tx, ty), (hx, hy) = camera.direction
            self.A = (float(ax), float(ay))
            self.L = (float(bx - ax), float(by - ay))
            self.length = math.hypot(*self.L)
            d = (hx - tx, hy - ty)
            dn = math.hypot(*d)
            n = (-self.L[1] / self.length, self.L[0] / self.length)
            if abs(n[0] * d[0] + n[1] * d[1]) < 0.2 * dn:
                self.active = False
                self._warn("red_light: the direction of travel is (nearly) parallel to the stop line; the rule is INACTIVE")
            else:
                self.n = n if n[0] * d[0] + n[1] * d[1] > 0 else (-n[0], -n[1])
        if self.active and not self.lanes:
            self._warn("red_light: no camera.lanes configured: the shared light_roi signal applies to the WHOLE stop line")
        self.lane_polys = [(l, _poly(l.polygon)) for l in self.lanes]
        self.set_fps(fps)
        self.reset()

    def _warn(self, msg):
        self.warnings.append(msg)
        self.log(f"[!] {msg}")

    def set_fps(self, fps):
        self._hold = to_frames(self.cfg.hold_s, fps)
        self._purge = to_frames(PURGE_S, fps)

    def reset(self):
        self._side, self._fired, self._hold_until, self._last_seen = {}, set(), {}, {}
        self._frame = 0
        self.unassigned_crossings = 0
        self.crossings = []                      # (frame, vehicle id, lane name or None, light, violation?) of every crossing seen: for tests and diagnostics

    def signed_distance(self, point):
        return self.n[0] * (point[0] - self.A[0]) + self.n[1] * (point[1] - self.A[1])

    def _lane_of(self, point):
        for lane, poly in self.lane_polys:
            if cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0:
                return lane
        return None

    def update(self, light, vehicles):
        """Mark violating motorcycles in place (vehicles: sentinel.rules.legacy.Vehicle list). Called once per frame."""
        self._frame += 1
        if not self.active:
            return
        h = self.cfg.jitter_px
        for v in vehicles:
            if not v.id.startswith("M_"):
                continue
            x1, y1, x2, y2 = v.coords
            p = ((x1 + x2) / 2.0, float(y2))
            s = self.signed_distance(p)
            self._last_seen[v.id] = self._frame
            prev = self._side.get(v.id, 0)
            side = -1 if s <= -h else (1 if s >= h else prev)
            self._side[v.id] = side
            if prev == -1 and side == 1:
                t = ((p[0] - self.A[0]) * self.L[0] + (p[1] - self.A[1]) * self.L[1]) / (self.length ** 2)
                if -SEGMENT_MARGIN <= t <= 1 + SEGMENT_MARGIN:
                    lane = self._lane_of(p) if self.lane_polys else None
                    violation = False
                    if self.lane_polys and lane is None:
                        self.unassigned_crossings += 1
                    elif (lane is None or lane.signal == "roi") and light == "RED" and v.id not in self._fired:
                        violation = True
                        self._fired.add(v.id)
                        self._hold_until[v.id] = self._frame + self._hold - 1
                    self.crossings.append((self._frame, v.id, getattr(lane, "name", None), light, violation))
            if self._hold_until.get(v.id, -1) >= self._frame:
                v.violator = True
                v.reason = join_reason(v.reason, self.REASON)
        if self._frame % self._purge == 0:
            for vid in [k for k, f in self._last_seen.items() if self._frame - f > self._purge]:
                for d in (self._side, self._hold_until, self._last_seen):
                    d.pop(vid, None)
                self._fired.discard(vid)


def _poly(points):
    import numpy as np
    return np.array(points, np.int32).reshape((-1, 1, 2))
