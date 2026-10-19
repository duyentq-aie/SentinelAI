"""Red-light rule "stop_line_crossing": a motorcycle violates when the bottom-centre of its COCO-model track (id K_<n>) CROSSES the stop line in the direction of travel
while the signal of its lane is RED.

Geometry. The stop line is a segment A-B; the direction of travel is a vector (camera.direction, tail -> head). The signed distance of a point P to the line is
s = n . (P - A), n = unit normal of the line oriented so that n . direction > 0: s < 0 before the line, s > 0 beyond it. Each track keeps a SIDE with a dead zone
of +-jitter_px (hysteresis, against detector jitter): side = -1 when s <= -jitter, +1 when s >= +jitter, otherwise unchanged. A crossing is the side flipping from -1 to +1.
It is a violation only if, in that very frame, the light is RED, the crossing point lies on the segment and its lane is not "exempt".
Not violations: a stationary vehicle (no flip), a vehicle already beyond the line when the light turned red (no flip while RED), travelling the other way (+1 -> -1),
jitter inside the dead zone, a crossing on green. At most one violation per track and run.
The verdict is held for `hold_s` seconds after the crossing so the evidence recorder opens one event with the vehicle box and plate search.

Source of the tracks (EXP-023): the helmet model almost never sees motorcycles at and above the stop line (13 vs 1000+ detections of the COCO model in the pha-xanh part of test_video), so the
rule uses the COCO model's motorcycle tracks (ids K_<n>). The evidence of a red_light event is that K_ track. A K_ track is LINKED to the helmet-model track M_ it overlaps (IoU >= link_iou, or one box
centre inside the other box; best match wins): the event record carries {"linked": {"track": "M_9", "reason": "NO-HELMET"}} and the drawn label shows both reasons. The helmet/overload verdicts
and their events stay those of the M_ track.
ID breaks: two reportable crossings in the same lane less than merge_s seconds apart whose crossing points are less than merge_px apart are one event (the second is recorded as merged, not reported).
A track that breaks BEFORE the line and restarts beyond it is not a crossing (no side flip): a known limit.

Lanes (camera.lanes): polygons whose signal is {roi: [x1, y1, x2, y2]} (the lane's own light, e.g. an arrow: its state arrives as `lane_lights`), "shared" (the camera's light_roi; the old
spelling "roi" is read as "shared") or "exempt" (no light: no red-light rule). A vehicle belongs to the
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
        self.active = bool(camera.stop_line and camera.direction and camera.light_roi)
        self.warnings = []
        self.inactive_reasons = [f"no camera.{k}" for k in ("stop_line", "direction", "light_roi") if not getattr(camera, k)]
        if not self.active:
            self._warn(f"red_light: stop_line_crossing needs camera.stop_line, camera.direction and camera.light_roi ({', '.join(self.inactive_reasons)}); the rule is INACTIVE (no red-light events)")
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
                self.inactive_reasons.append("direction parallel to the stop line")
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

    def status(self):
        """For the HUD and /api/status: whether the rule can report anything, and what it has counted instead of reporting."""
        return {"mode": "stop_line_crossing", "active": self.active, "inactive_reasons": list(self.inactive_reasons), "lanes": len(self.lanes),
                "crossings": len(self.crossings), "unassigned_crossings": self.unassigned_crossings}

    def set_fps(self, fps):
        self._hold = to_frames(self.cfg.hold_s, fps)
        self._merge = to_frames(self.cfg.merge_s, fps) if self.cfg.merge_s > 0 else 0
        self._purge = to_frames(PURGE_S, fps)

    def reset(self):
        self._side, self._fired, self._hold_until, self._last_seen = {}, set(), {}, {}
        self._reported = []                      # (frame, lane name, crossing point, track id) of the reportable crossings of the last merge_s seconds
        self._frame = 0
        self.unassigned_crossings = 0
        self.crossings = []                      # one dict per crossing seen (frame, id, lane, light, violation, why, box, point): reported or not, for the crossings report

    def signed_distance(self, point):
        return self.n[0] * (point[0] - self.A[0]) + self.n[1] * (point[1] - self.A[1])

    def _lane_of(self, point):
        for lane, poly in self.lane_polys:
            if cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0:
                return lane
        return None

    def _link(self, bike, vehicles):
        """The helmet-model track (M_) that is the same rider as this COCO track (K_): IoU >= link_iou, or the centre of one box inside the other box. The best match
        (highest IoU, then nearest centres) wins. -> Vehicle or None."""
        best, key = None, None
        bx1, by1, bx2, by2 = bike.coords
        bc = ((bx1 + bx2) / 2, (by1 + by2) / 2)
        for m in vehicles:
            if not m.id.startswith("M_"):
                continue
            mx1, my1, mx2, my2 = m.coords
            mc = ((mx1 + mx2) / 2, (my1 + my2) / 2)
            ov = iou(bike.coords, m.coords)
            inside = (mx1 <= bc[0] <= mx2 and my1 <= bc[1] <= my2) or (bx1 <= mc[0] <= bx2 and by1 <= mc[1] <= by2)
            if ov >= self.cfg.link_iou or inside:
                k = (-ov, math.hypot(bc[0] - mc[0], bc[1] - mc[1]))
                if key is None or k < key:
                    best, key = m, k
        return best

    def _merge_target(self, lane_name, point):
        """A reported crossing of the same lane less than merge_s ago and merge_px away: this one is the same event (the track broke in two)."""
        for fr, ln, pt, vid in self._reported:
            if ln == lane_name and self._frame - fr < self._merge and math.hypot(point[0] - pt[0], point[1] - pt[1]) < self.cfg.merge_px:
                return vid
        return None

    def lane_state(self, lane, light, lane_lights):
        """State of the signal that governs `lane`: its own light, the shared light, or None for an exempt lane."""
        if lane.kind == "own":
            return (lane_lights or {}).get(lane.name, "UNKNOWN")
        return None if lane.kind == "exempt" else light

    def update(self, light, bikes, vehicles=(), lane_lights=None):
        """Judge the motorcycle tracks `bikes` (Vehicle list, ids "K_<COCO track>") of this frame; `vehicles` are the other vehicles of the frame (the helmet-model M_ tracks are
        linked to a violating bike). -> the violating bikes (violator=True, reason RED-LIGHT RUNNER, linked/linked_reason set), to be appended to the frame's vehicles."""
        self._frame += 1
        out = []
        if not self.active:
            return out
        h = self.cfg.jitter_px
        self._reported = [r for r in self._reported if self._frame - r[0] < self._merge]
        for v in bikes:
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
                    lane_name = getattr(lane, "name", None)
                    state = light if lane is None else (self.lane_state(lane, light, lane_lights) or light)       # the signal that governs this crossing
                    violation, merged = False, None
                    if self.lane_polys and lane is None:
                        self.unassigned_crossings += 1
                        why = "crossing point in no lane: not attributable to a signal (precision first: counted, not reported)"
                    elif lane is not None and lane.kind == "exempt":
                        why = f"lane {lane.name!r} is exempt (no light of its own to judge by): no red-light rule"
                    elif state != "RED":
                        why = f"light is {state}" + (f" (lane {lane.name!r} own light)" if lane is not None and lane.kind == "own" else "")
                    elif v.id in self._fired:
                        why = "track already reported"
                    else:
                        merged = self._merge_target(lane_name, p)
                        if merged is not None:
                            self._fired.add(v.id)
                            why = (f"same event as the crossing of {merged} in lane {lane_name!r} less than {self.cfg.merge_s:g} s and {self.cfg.merge_px:g} px earlier "
                                   f"(a track that broke in two): not a second event")
                        else:
                            violation = True
                            self._fired.add(v.id)
                            self._hold_until[v.id] = self._frame + self._hold - 1
                            self._reported.append((self._frame, lane_name, p, v.id))
                            why = f"crossed the stop line on RED in lane {lane.name!r}" if lane else "crossed the stop line on RED (no lanes configured)"
                    self.crossings.append({"frame": self._frame - 1, "id": v.id, "lane": lane_name, "light": state, "violation": violation, "merged_into": merged,
                                           "why": why, "box": [int(c) for c in v.coords], "point": [round(p[0], 1), round(p[1], 1)]})
                else:
                    self.crossings.append({"frame": self._frame - 1, "id": v.id, "lane": None, "light": light, "violation": False, "merged_into": None,
                                           "why": "crossed the line's extension beyond the ends of the segment: ignored",
                                           "box": [int(c) for c in v.coords], "point": [round(p[0], 1), round(p[1], 1)]})
            if self._hold_until.get(v.id, -1) >= self._frame:
                v.violator, v.reason = True, self.REASON
                m = self._link(v, vehicles)
                if m is not None:
                    v.linked = m.id
                    v.linked_reason = m.reason if m.violator else ""
                out.append(v)
        if self._frame % self._purge == 0:
            for vid in [k for k, f in self._last_seen.items() if self._frame - f > self._purge]:
                for d in (self._side, self._hold_until, self._last_seen):
                    d.pop(vid, None)
                self._fired.discard(vid)
        return out


def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


def _poly(points):
    import numpy as np
    return np.array(points, np.int32).reshape((-1, 1, 2))
