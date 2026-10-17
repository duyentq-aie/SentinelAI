"""Ground-truth event labels (eval_sets/events/<video_id>.json) and the logic behind tools/label_events.py (no GUI here, so it is testable).

Schema = PLAN section 3.1 / docs/EVIDENCE.md (GT fields): a labels file describes one video; each event is one violation type on one vehicle.
"""
import hashlib
import json
import os
import re
from pathlib import Path

EVENT_TYPES = ("no_helmet", "overload", "red_light")
DIFFICULTIES = ("normal", "hard")
SPLITS = ("dev", "test")
PLATE_RE = re.compile(r"^[0-9A-Z]{5,12}$")          # normalised plate text: digits + capital letters, no separators


def video_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def normalize_plate_text(text):
    """'29-B1 123.45' -> '29B112345'; '' / None -> None."""
    if text is None:
        return None
    t = re.sub(r"[\s.\-_·]", "", str(text)).upper()
    return t or None


def new_labels(video_id, fps, width, height, split="dev", file_sha256=None, camera_config=None, conditions=None, annotated_ranges=None):
    return {"video_id": video_id, "file_sha256": file_sha256, "fps": fps, "width": width, "height": height, "split": split,
            "camera_config": camera_config, "conditions": conditions or {"time": "day", "weather": "dry", "density": "medium"},
            "annotated_ranges": annotated_ranges if annotated_ranges is not None else [], "events": [], "ignore_regions": []}


def validate(labels):
    """-> list of problems (empty = valid)."""
    p = []
    for k in ("video_id", "fps", "width", "height", "split", "annotated_ranges", "events"):
        if k not in labels:
            p.append(f"missing key {k!r}")
    if p:
        return p
    if labels["split"] not in SPLITS:
        p.append(f"split must be one of {SPLITS}")
    for k in ("fps", "width", "height"):
        if not (isinstance(labels[k], (int, float)) and labels[k] > 0):
            p.append(f"{k} must be positive")
    for r in labels["annotated_ranges"]:
        if not (isinstance(r, (list, tuple)) and len(r) == 2 and 0 <= r[0] < r[1]):
            p.append(f"bad annotated range {r}")
    seen = set()
    for e in labels["events"]:
        eid = e.get("id")
        if not eid or eid in seen:
            p.append(f"missing or duplicate event id {eid!r}")
        seen.add(eid)
        if e.get("type") not in EVENT_TYPES:
            p.append(f"{eid}: type must be one of {EVENT_TYPES}")
        fs, fe, kf = e.get("frame_start"), e.get("frame_end"), e.get("key_frame")
        if not all(isinstance(x, int) and x >= 0 for x in (fs, fe, kf)) or not fs <= kf <= fe:
            p.append(f"{eid}: need integers frame_start <= key_frame <= frame_end")
        box = e.get("key_box")
        if not (isinstance(box, (list, tuple)) and len(box) == 4 and box[0] < box[2] and box[1] < box[3]):
            p.append(f"{eid}: key_box must be [x1, y1, x2, y2] with x1 < x2 and y1 < y2")
        elif labels["width"] and (box[0] < 0 or box[1] < 0 or box[2] > labels["width"] or box[3] > labels["height"]):
            p.append(f"{eid}: key_box outside the {labels['width']}x{labels['height']} frame")
        pt = e.get("plate_text")
        if pt is not None and not PLATE_RE.match(pt):
            p.append(f"{eid}: plate_text {pt!r} must be 5-12 capital letters/digits without separators (or null)")
        if e.get("difficulty") not in DIFFICULTIES:
            p.append(f"{eid}: difficulty must be one of {DIFFICULTIES}")
        if labels["annotated_ranges"] and not any(a <= kf <= b for a, b in labels["annotated_ranges"]):
            p.append(f"{eid}: key_frame {kf} is outside every annotated range (false positives are only counted inside them)")
    return p


def save(path, labels):
    """Validate, then write atomically (events sorted by frame_start)."""
    labels = dict(labels, events=sorted(labels["events"], key=lambda e: (e["frame_start"], e["id"])))
    problems = validate(labels)
    if problems:
        raise ValueError("invalid labels: " + "; ".join(problems))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load(path):
    labels = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate(labels)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return labels


class LabelSession:
    """State machine behind the labelling GUI: mark start / end / key frame + box, pick a type, commit, undo."""

    def __init__(self, labels):
        self.labels = labels
        self.type_idx = 0
        self.draft = {}
        self._range_open = None

    @property
    def type(self):
        return EVENT_TYPES[self.type_idx]

    def cycle_type(self):
        self.type_idx = (self.type_idx + 1) % len(EVENT_TYPES)
        return self.type

    def mark_start(self, frame):
        self.draft["frame_start"] = int(frame)

    def mark_end(self, frame):
        self.draft["frame_end"] = int(frame)

    def set_key(self, frame, box):
        x1, y1, x2, y2 = box
        self.draft["key_frame"] = int(frame)
        self.draft["key_box"] = [int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))]

    def ready(self):
        return all(k in self.draft for k in ("frame_start", "frame_end", "key_frame", "key_box"))

    def next_id(self):
        n = 1 + max([int(re.sub(r"\D", "", e["id"]) or 0) for e in self.labels["events"]] + [0])
        return f"e{n:03d}"

    def commit(self, plate_text=None, difficulty="normal", notes=""):
        if not self.ready():
            raise ValueError("set start, end and key frame + box first")
        d = self.draft
        ev = {"id": self.next_id(), "type": self.type, "frame_start": min(d["frame_start"], d["key_frame"], d["frame_end"]),
              "frame_end": max(d["frame_end"], d["key_frame"], d["frame_start"]), "key_frame": d["key_frame"], "key_box": d["key_box"],
              "plate_text": normalize_plate_text(plate_text), "difficulty": difficulty, "notes": notes}
        self.labels["events"].append(ev)
        problems = validate(self.labels)
        if problems:
            self.labels["events"].pop()
            raise ValueError("; ".join(problems))
        self.draft = {}
        return ev

    def undo(self):
        return self.labels["events"].pop() if self.labels["events"] else None

    def toggle_range(self, frame):
        """First call opens an annotated range at `frame`, second call closes it."""
        if self._range_open is None:
            self._range_open = int(frame)
            return None
        a, b = sorted((self._range_open, int(frame)))
        self._range_open = None
        if b > a:
            self.labels["annotated_ranges"].append([a, b])
            return [a, b]
        return None
