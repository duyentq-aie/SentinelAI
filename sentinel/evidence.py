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

The plate detector runs only for tracks that are violating right now, for at most `plate.search_s` seconds of violation per track (`max_attempts` runs at 30 FPS), and stops at the
first plate (unless plate.stop_when_found is false: the v0 behaviour, kept as an A/B reference).
"""
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from sentinel.ocr import quality
from sentinel.timebase import REFERENCE_FPS, seconds_of, to_frames
from sentinel.ocr.engine import OcrUnavailable

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
BOX_SAMPLE_MAX = 60
SINGLE_TYPE = {"NO-HELMET": "no_helmet", "OVERLOAD": "overload", "RED-LIGHT RUNNER": "red_light", "WRONG-WAY": "wrong_way", "WRONG LANE": "wrong_lane"}


def types_of(reason):
    """Event types of a vehicle's reason: a known combined reason, or several single reasons joined with ' & ' (e.g. 'NO-HELMET & RED-LIGHT RUNNER'), else a slug."""
    if reason in TYPES_OF_REASON:
        return TYPES_OF_REASON[reason]
    parts = reason.split(" & ")
    if all(p in SINGLE_TYPE for p in parts):
        return list(dict.fromkeys(SINGLE_TYPE[p] for p in parts))
    return [reason.lower().replace(" ", "_")]


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
    plate_pred: dict = None
    head_box: list = None                     # no_helmet events: the box of the no-helmet head that was attached to the vehicle in the key frame (null when it was not detected that frame / not applicable)
    head_class: str = None                    # ... and the class the model gave it ("no-helmet")
    linked: dict = None                       # red_light events only: {"track": "M_<id>", "reason": "NO-HELMET"} of the overlapping helmet-model track


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
    def __init__(self, cfg, plate_boxes, video_id, camera_id, on_plate_saved=None, on_evicted=None, clock=time.time, ocr=None, fps=REFERENCE_FPS, models=None):
        """cfg: EvidenceCfg. plate_boxes(crop_bgr) -> [(x1, y1, x2, y2, conf)] relative to the crop, best first.
        ocr: PlateOCR or None (None = plates are only cropped, never read)."""
        self.cfg, self.plate_boxes, self.ocr = cfg, plate_boxes, ocr
        self.models = models                      # {"helmet": {"file", "sha256"}, "coco": ..., "plate": ...}: written into every event record (which model made the event; the review queue reads it)
        self.set_fps(fps)
        self.ocr_latencies_ms = []
        self.video_id, self.camera_id = video_id, camera_id
        self.root = Path(cfg.dir)
        self.on_plate_saved, self.on_evicted, self.clock = on_plate_saved, on_evicted, clock
        self.run, self._n = 0, 0
        self._open = {}            # (track_id, type) -> _Event
        self._done = set()         # (track_id, type) that already had an event this run (one_event_per_track)
        self._plates = {}          # track_id -> plate dict (found)
        self._attempts = {}        # track_id -> detector runs
        self._violating_since = {} # track_id -> first frame of the current violating stretch
        self._cands = {}           # track_id -> plate crop candidates for OCR [{img, frame, box, conf, attempts}]
        self._last_cand = {}       # track_id -> frame of the last collected candidate
        self._pred = {}            # track_id -> plate_pred (OCR result, or an error record)
        self.root.mkdir(parents=True, exist_ok=True)

    def set_fps(self, fps):
        """Durations of the config (seconds) -> frame counts at the frame rate the pipeline sees."""
        self.fps = float(fps)
        c, p, o = self.cfg, self.cfg.plate, self.cfg.ocr
        self._close_after = to_frames(seconds_of(c.close_after_s, c.close_after), self.fps)
        self._box_every = to_frames(c.box_sample_s, self.fps)
        self._every = to_frames(p.every_s if p.every is None else (p.every / REFERENCE_FPS if p.every > 1 else 0.0), self.fps)   # legacy `every` k = k frames at 30 FPS (1 = every frame)
        search_s = p.search_s if p.max_attempts is None else p.max_attempts * (p.every or 1) / REFERENCE_FPS   # legacy runs -> seconds of search
        self._max_attempts = 0 if search_s == 0 else max(1, int(round(search_s * self.fps / self._every + 1e-9)))
        self._min_gap = to_frames(seconds_of(o.min_gap_s, o.min_gap), self.fps)

    # ---- lifecycle
    def reset(self):
        """Source changed / video looped: ids restart, so close everything and forget id-keyed state."""
        self.flush()
        self._plates.clear(); self._attempts.clear(); self._violating_since.clear(); self._done.clear()
        self._cands.clear(); self._last_cand.clear(); self._pred.clear()
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
            for etype in types_of(v.reason):
                ev = self._open.get((v.id, etype))
                if ev is None and cfg.one_event_per_track and (v.id, etype) in self._done:
                    continue                      # this track already had its event of this type
                if ev is None:
                    ev = self._start(frame_idx, raw, v, etype, light)
                    res.new_events.append(ev.id)
                ev.last_seen = ev.frame_end = frame_idx
                if (frame_idx - ev.frame_start) % self._box_every == 0 and len(ev.boxes) < BOX_SAMPLE_MAX:
                    ev.boxes[str(frame_idx)] = list(v.coords)

            self._maybe_detect_plate(frame_idx, raw, v, res)

        for ev in list(self._open.values()):
            if frame_idx - ev.last_seen >= self._close_after:
                self._close(ev)
        return res

    # ---- plates
    def _collecting(self, tid):
        """OCR wants more crops of this track before reading."""
        return self.ocr is not None and tid not in self._pred and len(self._cands.get(tid, [])) < self.cfg.ocr.candidates

    def _maybe_detect_plate(self, frame_idx, raw, v, res):
        cfg = self.cfg.plate
        found = v.id in self._plates
        if found and cfg.stop_when_found and not self._collecting(v.id):
            return
        attempts = self._attempts.get(v.id, 0)
        if self._max_attempts and attempts >= self._max_attempts:
            self._finish_plate_search(v.id)
            return
        if (frame_idx - self._violating_since[v.id]) % self._every:
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
        if boxes:
            px1, py1, px2, py2, conf = boxes[0]
            rx1, ry1, rx2, ry2 = vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2
            m = self.cfg.plate_margin
            y1, y2 = max(0, ry1 - m), min(raw.shape[0], ry2 + m)
            x1, x2 = max(0, rx1 - m), min(raw.shape[1], rx2 + m)
            img = raw[y1:y2, x1:x2]
            if img.size:
                om = self.cfg.ocr.margin
                tight = raw[max(0, ry1 - om):min(raw.shape[0], ry2 + om), max(0, rx1 - om):min(raw.shape[1], rx2 + om)]
                info = {"found": True, "frame": frame_idx, "box": [rx1, ry1, rx2, ry2], "conf": round(float(conf), 4),
                        "crop_size": [int(img.shape[1]), int(img.shape[0])], "attempts": self._attempts[v.id], "_img": img, "_tight": tight}
                if not found:
                    self._plates[v.id] = info
                    res.plates_found.append(v.id)
                    for ev in self._open.values():
                        if ev.track_id == v.id:
                            self._attach_plate(ev)
                if self.ocr is not None and v.id not in self._pred:
                    self._collect(v.id, info)
        if (not boxes or v.id in self._plates) and self._max_attempts and self._attempts[v.id] >= self._max_attempts:
            self._finish_plate_search(v.id)

    def _collect(self, tid, info):
        o = self.cfg.ocr
        if info["frame"] - self._last_cand.get(tid, -10**9) < self._min_gap:
            return
        self._last_cand[tid] = info["frame"]
        self._cands.setdefault(tid, []).append(info)
        if len(self._cands[tid]) >= o.candidates:
            self._run_ocr(tid)

    def _run_ocr(self, tid):
        """Pick the best crop (size x sharpness) of the track as THE plate crop, read the best few and vote. Never raises."""
        cands = self._cands.get(tid)
        if not cands or tid in self._pred or self.ocr is None:
            return
        best = cands[quality.best([c["_tight"] for c in cands], 1)[0]]
        self._plates[tid] = dict(best)
        for ev in self._open.values():
            if ev.track_id == tid:
                self._attach_plate(ev, notify=False)                 # rewrite plate.jpg with the best crop
        try:
            pred = self.ocr.read_track([c["_tight"] for c in cands])
            self.ocr_latencies_ms.append(pred["latency_ms"])
        except OcrUnavailable as e:
            pred = {"text": None, "format_ok": False, "error": str(e)}
            self.ocr = None                                          # do not retry (and do not stall) on every violating track
        except Exception as e:                                       # a bad crop must not stop the pipeline
            pred = {"text": None, "format_ok": False, "error": f"{type(e).__name__}: {e}"}
        self._pred[tid] = pred
        for ev in self._open.values():
            if ev.track_id == tid:
                ev.plate_pred = pred
                if ev.written:
                    self._write(ev, "open")

    def _finish_plate_search(self, track_id):
        for ev in self._open.values():
            if ev.track_id == track_id and ev.plate is None:
                ev.plate = {"found": False, "attempts": self._attempts.get(track_id, 0)}
        self._run_ocr(track_id)                                      # whatever crops were collected are read now

    def _attach_plate(self, ev, notify=True):
        info = self._plates[ev.track_id]
        d = self.root / ev.id
        d.mkdir(parents=True, exist_ok=True)
        path = d / "plate.jpg"
        cv2.imwrite(str(path), info["_img"], [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        ev.plate = {k: v for k, v in info.items() if not k.startswith("_")}
        ev.plate["file"] = self._rel(path)
        if self.on_plate_saved and notify:
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
        if etype == "no_helmet":
            bad = [b for b, c in getattr(v, "heads", []) if c == "no-helmet"]
            if bad:
                ev.head_box, ev.head_class = [int(x) for x in max(bad, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))], "no-helmet"       # several: the biggest one
        if getattr(v, "linked", ""):
            ev.linked = {"track": v.linked, "reason": v.linked_reason}
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
        ev.plate_pred = self._pred.get(v.id)
        self._write(ev, "open")
        self._limit()
        return ev

    def _close(self, ev):
        if ev.closed:
            return
        if not any(e.track_id == ev.track_id for e in self._open.values() if e is not ev):
            self._run_ocr(ev.track_id)                               # last open event of this vehicle: read what was collected
        if ev.plate_pred is None:
            ev.plate_pred = self._pred.get(ev.track_id)
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
            "plate_pred": ev.plate_pred, "boxes": ev.boxes if status != "open" else {str(ev.frame_start): ev.key_box},
            "plate": ev.plate if ev.plate is not None else {"found": False, "attempts": self._attempts.get(ev.track_id, 0), "searching": True},
            "evidence": {k: (self._rel(p) if p.exists() else None) for k, p in files.items()},
            "vehicle": {"cls": ev.vehicle_cls, "box_size": [ev.key_box[2] - ev.key_box[0], ev.key_box[3] - ev.key_box[1]]},
            "light": ev.light, "video_id": self.video_id, "camera_id": self.camera_id, "run": ev.run, "created_at": ev.created_at,
        }
        out["head_box"], out["head_class"] = ev.head_box, ev.head_class
        if self.models:
            out["models"] = self.models
        if ev.linked:
            out["linked"] = ev.linked
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
