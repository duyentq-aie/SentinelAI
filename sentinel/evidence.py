"""Evidence for violation events.

Per event (one violation *type* on one track):
    <dir>/<event_id>/frame.jpg     the raw frame at the trigger (never drawn on)
    <dir>/<event_id>/vehicle.jpg   crop of the offending vehicle at the trigger
    <dir>/<event_id>/plate.jpg     plate crop (raw pixels + margin), only if a plate was found
    <dir>/events.jsonl             append-only log: a record when the event opens and one when it closes
                                   (take the LAST record per id); plus {"id", "status": "evicted"} when the size limit deletes its files.

Event records use the ground-truth schema of PLAN section 3.1 (id, type, frame_start, frame_end, key_frame, key_box, plate_text,
difficulty, notes) plus prediction-only fields (track_id, trigger_frame, confidence, plate_pred, boxes, plate, evidence, ...),
so tools/eval_events.py can read predictions and ground truth with the same code. An event is written even when no plate is found.

The plate detector runs only for tracks that are violating right now, at most `plate.max_attempts` times per track, and stops at the
first plate (unless plate.stop_when_found is false: the v0 behaviour, kept as an A/B reference).
"""
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

SCHEMA = "sentinel.event/1"
GT_FIELDS = ("id", "type", "frame_start", "frame_end", "key_frame", "key_box", "plate_text", "difficulty", "notes")
PREDICTION_FIELDS = ("schema", "status", "frame_size", "track_id", "trigger_frame", "confidence", "plate_pred", "boxes", "plate", "evidence",
                     "vehicle", "light", "video_id", "camera_id", "run", "created_at")

TYPES_OF_REASON = {
    "NO-HELMET": ["no_helmet"],
    "OVERLOAD": ["overload"],
    "OVERLOAD & NO-HELMET": ["overload", "no_helmet"],
    "RED-LIGHT RUNNER": ["red_light"],
    "WRONG-WAY": ["wrong_way"],
    "WRONG LANE": ["wrong_lane"],
    "WRONG LANE & WAY": ["wrong_lane", "wrong_way"],
}
BOX_SAMPLE_EVERY = 5      # frames between boxes stored in event["boxes"]
BOX_SAMPLE_MAX = 60


@dataclass
class _Event:
    id: str
    type: str
    track_id: str
    run: int
    frame_start: int
    key_box: list
    vehicle_cls: str
    light: str
    created_at: str
    frame_end: int = 0
    boxes: dict = field(default_factory=dict)
    plate: dict = None
    last_seen: int = 0
    closed: bool = False
    written: bool = False
    frame_size: list = None


@dataclass
class UpdateResult:
    new_events: list = field(default_factory=list)      # ids opened this frame
    plates_found: list = field(default_factory=list)    # track ids whose plate was found this frame (first time)
    plate_calls: int = 0                                # plate-detector runs this frame
    plate_boxes: list = field(default_factory=list)     # (x1, y1, x2, y2) absolute, for drawing


def dir_size(root):
    return sum(f.stat().st_size for f in Path(root).rglob("*") if f.is_file())


def enforce_dir_limit(root, max_bytes, protect=("events.jsonl",)):
    """Delete the oldest entries (files or event folders) under root until it fits. -> list of removed entry names."""
    root, removed = Path(root), []
    if max_bytes <= 0 or not root.exists():
        return removed
    entries = [e for e in root.iterdir() if e.name not in protect]
    total = dir_size(root)
    for e in sorted(entries, key=lambda e: e.stat().st_mtime):
        if total <= max_bytes:
            break
        size = dir_size(e) if e.is_dir() else e.stat().st_size
        shutil.rmtree(e, ignore_errors=True) if e.is_dir() else e.unlink(missing_ok=True)
        total -= size
        removed.append(e.name)
    return removed


class EvidenceRecorder:
    def __init__(self, cfg, plate_boxes, video_id, camera_id, on_plate_saved=None, on_evicted=None, clock=time.time):
        """cfg: EvidenceCfg. plate_boxes(crop_bgr) -> [(x1, y1, x2, y2, conf)] relative to the crop, best first."""
        self.cfg, self.plate_boxes = cfg, plate_boxes
        self.video_id, self.camera_id = video_id, camera_id
        self.root = Path(cfg.dir)
        self.on_plate_saved, self.on_evicted, self.clock = on_plate_saved, on_evicted, clock
        self.run, self._n = 0, 0
        self._open = {}            # (track_id, type) -> _Event
        self._done = set()         # (track_id, type) that already had an event this run (one_event_per_track)
        self._plates = {}          # track_id -> plate dict (found)
        self._attempts = {}        # track_id -> detector runs
        self._violating_since = {} # track_id -> first frame of the current violating stretch
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- lifecycle
    def reset(self):
        """Source changed / video looped: ids restart, so close everything and forget id-keyed state."""
        self.flush()
        self._plates.clear(); self._attempts.clear(); self._violating_since.clear(); self._done.clear()
        self.run += 1

    def flush(self):
        for ev in list(self._open.values()):
            self._close(ev)

    # ---- per frame
    def update(self, frame_idx, raw, vehicles, light):
        res, cfg = UpdateResult(), self.cfg
        violating = {v.id: v for v in vehicles if v.violator}
        for tid in list(self._violating_since):
            if tid not in violating:
                del self._violating_since[tid]

        for v in violating.values():
            self._violating_since.setdefault(v.id, frame_idx)
            for etype in TYPES_OF_REASON.get(v.reason, [v.reason.lower().replace(" ", "_")]):
                ev = self._open.get((v.id, etype))
                if ev is None and cfg.one_event_per_track and (v.id, etype) in self._done:
                    continue                      # this track already had its event of this type
                if ev is None:
                    ev = self._start(frame_idx, raw, v, etype, light)
                    res.new_events.append(ev.id)
                ev.last_seen = ev.frame_end = frame_idx
                if (frame_idx - ev.frame_start) % BOX_SAMPLE_EVERY == 0 and len(ev.boxes) < BOX_SAMPLE_MAX:
                    ev.boxes[str(frame_idx)] = list(v.coords)

            self._maybe_detect_plate(frame_idx, raw, v, res)

        for ev in list(self._open.values()):
            if frame_idx - ev.last_seen >= cfg.close_after:
                self._close(ev)
        return res

    # ---- plates
    def _maybe_detect_plate(self, frame_idx, raw, v, res):
        cfg = self.cfg.plate
        found = v.id in self._plates
        if found and cfg.stop_when_found:
            return
        attempts = self._attempts.get(v.id, 0)
        if cfg.max_attempts and attempts >= cfg.max_attempts:
            self._finish_plate_search(v.id)
            return
        if (frame_idx - self._violating_since[v.id]) % cfg.every:
            return
        vx1, vy1, vx2, vy2 = v.coords
        vx1, vy1 = max(0, vx1), max(0, vy1)
        vx2, vy2 = min(raw.shape[1], vx2), min(raw.shape[0], vy2)
        crop = raw[vy1:vy2, vx1:vx2]
        if crop.size == 0:
            return
        self._attempts[v.id] = attempts + 1
        res.plate_calls += 1
        boxes = self.plate_boxes(crop)
        for (px1, py1, px2, py2, _c) in boxes:
            res.plate_boxes.append((vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2))
        if boxes and not found:
            px1, py1, px2, py2, conf = boxes[0]
            rx1, ry1, rx2, ry2 = vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2
            m = self.cfg.plate_margin
            y1, y2 = max(0, ry1 - m), min(raw.shape[0], ry2 + m)
            x1, x2 = max(0, rx1 - m), min(raw.shape[1], rx2 + m)
            img = raw[y1:y2, x1:x2]
            if img.size:
                self._plates[v.id] = {"found": True, "frame": frame_idx, "box": [rx1, ry1, rx2, ry2], "conf": round(float(conf), 4),
                                      "crop_size": [int(img.shape[1]), int(img.shape[0])], "attempts": self._attempts[v.id], "_img": img}
                res.plates_found.append(v.id)
                for ev in self._open.values():
                    if ev.track_id == v.id:
                        self._attach_plate(ev)
        elif not boxes and cfg.max_attempts and self._attempts[v.id] >= cfg.max_attempts:
            self._finish_plate_search(v.id)

    def _finish_plate_search(self, track_id):
        for ev in self._open.values():
            if ev.track_id == track_id and ev.plate is None:
                ev.plate = {"found": False, "attempts": self._attempts.get(track_id, 0)}

    def _attach_plate(self, ev):
        info = self._plates[ev.track_id]
        d = self.root / ev.id
        d.mkdir(parents=True, exist_ok=True)
        path = d / "plate.jpg"
        cv2.imwrite(str(path), info["_img"], [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        ev.plate = {k: v for k, v in info.items() if k != "_img"}
        ev.plate["file"] = self._rel(path)
        if self.on_plate_saved:
            self.on_plate_saved(ev, path)
        if ev.written:
            self._write(ev, "open")           # last record wins: makes the plate visible even if the process dies before close
        self._limit()

    # ---- events
    def _start(self, frame_idx, raw, v, etype, light):
        self._n += 1
        ev = _Event(id=f"{self.video_id}-r{self.run}-e{self._n:04d}", type=etype, track_id=v.id, run=self.run, frame_start=frame_idx,
                    key_box=list(v.coords), vehicle_cls=v.type, light=light,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.clock())))
        ev.frame_end = ev.last_seen = frame_idx
        ev.frame_size = [int(raw.shape[1]), int(raw.shape[0])]
        ev.boxes[str(frame_idx)] = list(v.coords)
        d = self.root / ev.id
        d.mkdir(parents=True, exist_ok=True)
        q = [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality]
        cv2.imwrite(str(d / "frame.jpg"), raw, q)
        x1, y1, x2, y2 = (max(0, v.coords[0]), max(0, v.coords[1]), min(raw.shape[1], v.coords[2]), min(raw.shape[0], v.coords[3]))
        if x2 > x1 and y2 > y1:
            cv2.imwrite(str(d / "vehicle.jpg"), raw[y1:y2, x1:x2], q)
        self._open[(v.id, etype)] = ev
        if v.id in self._plates:
            self._attach_plate(ev)
        self._write(ev, "open")
        self._limit()
        return ev

    def _close(self, ev):
        if ev.closed:
            return
        if ev.plate is None:
            ev.plate = {"found": False, "attempts": self._attempts.get(ev.track_id, 0)}
        ev.closed = True
        self._done.add((ev.track_id, ev.type))
        self._open.pop((ev.track_id, ev.type), None)
        self._write(ev, "closed")

    def _rel(self, path):
        """Paths in events.jsonl are relative to the evidence folder, so the folder can be moved or served as is."""
        return Path(path).relative_to(self.root).as_posix()

    def record(self, ev, status):
        d = self.root / ev.id
        files = {"frame": d / "frame.jpg", "vehicle": d / "vehicle.jpg", "plate": d / "plate.jpg"}
        out = {
            "id": ev.id, "type": ev.type, "frame_start": ev.frame_start, "frame_end": ev.frame_end, "key_frame": ev.frame_start,
            "key_box": ev.key_box, "plate_text": None, "difficulty": None, "notes": "",
            "schema": SCHEMA, "status": status, "frame_size": ev.frame_size, "track_id": ev.track_id, "trigger_frame": ev.frame_start, "confidence": None,
            "plate_pred": None, "boxes": ev.boxes if status != "open" else {str(ev.frame_start): ev.key_box},
            "plate": ev.plate if ev.plate is not None else {"found": False, "attempts": self._attempts.get(ev.track_id, 0), "searching": True},
            "evidence": {k: (self._rel(p) if p.exists() else None) for k, p in files.items()},
            "vehicle": {"cls": ev.vehicle_cls, "box_size": [ev.key_box[2] - ev.key_box[0], ev.key_box[3] - ev.key_box[1]]},
            "light": ev.light, "video_id": self.video_id, "camera_id": self.camera_id, "run": ev.run, "created_at": ev.created_at,
        }
        return out

    def _write(self, ev, status):
        ev.written = True
        with open(self.root / "events.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(self.record(ev, status), separators=(",", ":")) + "\n")

    def _limit(self):
        protect = ("events.jsonl", *[e.id for e in self._open.values()])      # never delete the evidence of an open event
        removed = enforce_dir_limit(self.root, self.cfg.max_dir_mb * 1024 * 1024, protect)
        if removed:
            with open(self.root / "events.jsonl", "a", encoding="utf-8", newline="\n") as f:
                for name in removed:
                    f.write(json.dumps({"id": name, "status": "evicted", "reason": "evidence dir size limit"}) + "\n")
            if self.on_evicted:
                self.on_evicted(removed)
