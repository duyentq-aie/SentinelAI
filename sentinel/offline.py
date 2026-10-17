"""Replay violation rules offline from a recorded tracks file (no models, no video).

Tracks file (tests/fixtures/tracks_test_video.jsonl): line 1 = {"meta": {...}}, then one JSON object per frame
{"f", "light", "coco": [[id, cls, x1, y1, x2, y2, conf, state], ...], "helmet": [...]} (see Perception.to_json).
"""
import json
from pathlib import Path

from sentinel.perception import TRACK_FIELDS, Perception


def write_tracks(path, meta, perceptions):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"meta": {**meta, "track_fields": TRACK_FIELDS}}) + "\n")
        for i, p in enumerate(perceptions):
            f.write(json.dumps(p.to_json(i), separators=(",", ":")) + "\n")


def load_tracks(path):
    """-> (meta, [Perception])"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0])["meta"]
    return meta, [Perception.from_json(json.loads(l)) for l in lines[1:] if l.strip()]


def run_rules(perceptions, rules, label_fn):
    """Feed every frame to `rules` and return the violation events in the characterization schema.

    Only label events are reproducible offline: plate captures need the plate model on pixels.
    """
    first = {}
    for i, p in enumerate(perceptions):
        for lab in label_fn(rules.step(p)):
            first.setdefault(lab, i)
    return [{"frame": fi, "kind": "label_first_seen", "label": lab} for lab, fi in sorted(first.items(), key=lambda x: x[1])]
