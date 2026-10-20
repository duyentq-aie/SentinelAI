"""Human-in-the-loop review of events: reviews.jsonl (append-only) next to events.jsonl.

An event is `pending` until somebody decides: confirmed | rejected | unclear. Only `confirmed` events count as violations (event panel, plate gallery). A decision is
APPENDED as a new line {schema, event_id, decision, note, reviewed_at, event_type, track_id, run, model{role, file, sha256}, ...}; reviewing an event again appends a newer line
and never edits an older one: the LAST line of an event is its decision, the earlier lines are its history.
Precision (per violation type and per model) is computed from these decisions only: confirmed / (confirmed + rejected), `unclear` left out.

What a decision MEANS for a no_helmet event (the model said "this head has no helmet"), written as `corrected_label` = the right label of the head:
  confirmed -> "no-helmet"    rejected -> "helmet" (the model was wrong: the head wears a helmet; NOT "background")
  rejected + not_head -> "background" (special case: it is not a head at all / the wrong person)    unclear -> null (not exported)
Other event types (overload, red_light) have no corrected_label: their decisions are metadata only.
"""
import json
import re
import threading
import time
from pathlib import Path

DECISIONS = ("confirmed", "rejected", "unclear")
SCHEMA = "sentinel.review/1"
EVENT_ID = re.compile(r"^[A-Za-z0-9._:/-]{1,160}$")
NOTE_MAX = 300
MODEL_OF_TYPE = {"red_light": "coco"}              # which model produced the event of this type; every other type comes from the helmet model


def corrected_label(event_type, decision, not_head=False):
    """The right label of the head of a reviewed no_helmet event (see the module docstring); None for every other case."""
    if event_type != "no_helmet" or decision == "unclear":
        return None
    if decision == "confirmed":
        return "no-helmet"
    return "background" if not_head else "helmet"


def clean_note(note):
    """Short free text: control characters removed, whitespace collapsed, at most NOTE_MAX characters."""
    note = re.sub(r"[\x00-\x1f\x7f]+", " ", note or "")
    return re.sub(r"\s+", " ", note).strip()[:NOTE_MAX]


def model_of(event):
    """{"role", "file", "sha256", "source"} of the model that generated an event, as stamped in the event record when it was made. An event from before the stamp existed has
    NO known model (file / sha256 null): guessing "the model running now" could attribute an old event to the wrong model."""
    role = MODEL_OF_TYPE.get(event.get("type"), "helmet")
    stamped = (event.get("models") or {}).get(role)
    if stamped:
        return {"role": role, "file": stamped.get("file"), "sha256": stamped.get("sha256"), "source": "event"}
    return {"role": role, "file": None, "sha256": None, "source": "unknown: the event record has no model stamp (made before events were stamped)"}


class ReviewStore:
    def __init__(self, evidence_dir, read_events, clock=time.time):
        self.dir = Path(evidence_dir)
        self.path = self.dir / "reviews.jsonl"
        self.read_events, self.clock = read_events, clock
        self._lock = threading.Lock()

    # ---- storage
    def history(self):
        """Every line of reviews.jsonl in file order (a damaged line is skipped, never fatal)."""
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("event_id") and r.get("decision") in DECISIONS:
                out.append(r)
        return out

    def latest(self):
        """{event_id: its last review line}"""
        return {r["event_id"]: r for r in self.history()}

    def events(self):
        """All events of the evidence folder, oldest first: [{... event record ...}]"""
        return list(reversed(self.read_events(self.dir, 10 ** 9)))

    def status_of(self, event_id, latest=None):
        r = (latest if latest is not None else self.latest()).get(event_id)
        return r["decision"] if r else "pending"

    # ---- queue
    def next_pending(self):
        """Oldest pending event, closed ones first (their evidence is final); an event that is still open is offered only when no closed one is waiting. -> (event or None, n pending)."""
        latest = self.latest()
        pend = [e for e in self.events() if e["id"] not in latest]
        closed = [e for e in pend if e.get("status") == "closed"]
        return (closed or pend or [None])[0], len(pend)

    def submit(self, event_id, decision, note="", not_head=False):
        """Append a decision. -> the stored record. KeyError: unknown event; ValueError: bad decision / id / not_head."""
        if not isinstance(event_id, str) or not EVENT_ID.match(event_id):
            raise ValueError("event_id must be 1-160 characters of [A-Za-z0-9._:/-]")
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {list(DECISIONS)}")
        event = next((e for e in self.events() if e["id"] == event_id), None)
        if event is None:
            raise KeyError(event_id)
        if not_head and (decision != "rejected" or event.get("type") != "no_helmet"):
            raise ValueError("not_head is only for a rejected no_helmet event")
        rec = {"schema": SCHEMA, "event_id": event_id, "decision": decision, "note": clean_note(note),
               "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.clock())), "event_type": event.get("type"), "track_id": event.get("track_id"),
               "run": event.get("run"), "video_id": event.get("video_id"), "camera_id": event.get("camera_id"), "model": model_of(event),
               "corrected_label": corrected_label(event.get("type"), decision, not_head), "not_head": bool(not_head), "head_box": event.get("head_box"), "head_class": event.get("head_class")}
        self.append(rec)
        return rec

    def append(self, rec):
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                f.flush()

    # ---- statistics
    def stats(self):
        """Counts and precision computed from the decisions only (latest decision of every event; imported reviews included)."""
        latest = self.latest()
        events = self.events()
        known = {e["id"] for e in events}
        pending = sum(1 for e in events if e["id"] not in latest)
        counts = {d: sum(1 for r in latest.values() if r["decision"] == d) for d in DECISIONS}

        def table(key):
            groups = {}
            for r in latest.values():
                g = groups.setdefault(key(r), {"confirmed": 0, "rejected": 0, "unclear": 0})
                g[r["decision"]] += 1
            for g in groups.values():
                n = g["confirmed"] + g["rejected"]
                g["precision"] = round(g["confirmed"] / n, 3) if n else None
                g["decided"] = n
            return dict(sorted(groups.items()))

        def model_key(r):
            m = r.get("model") or {}
            return f"{m.get('file') or '?'} ({(m.get('sha256') or 'no sha256')[:12]})"

        return {"pending": pending, **counts, "reviewed_events": len(latest), "review_lines": len(self.history()), "imported": sum(1 for r in latest.values() if r.get("source")),
                "reviewed_of_current_events": len(known & set(latest)), "corrected_labels": {k: sum(1 for r in latest.values() if r.get("corrected_label") == k) for k in ("no-helmet", "helmet", "background")},
                "by_type": table(lambda r: r.get("event_type") or "?"), "by_model": table(model_key),
                "note": "precision = confirmed / (confirmed + rejected) from the review decisions only; unclear is left out; pending events are not in it"}
