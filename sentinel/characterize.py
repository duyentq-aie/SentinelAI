"""Helpers for characterization tests: record pipeline events and compare snapshots.

An *event* is one of
  {"frame": i, "kind": "plate_capture", "track": "M_77"}       evidence saved for that track
  {"frame": i, "kind": "label_first_seen", "label": "ID:M_77 NO-HELMET"}   first frame a violation-looking label was drawn
This is the same schema as experiments/EXP-001-baseline/events.json.
"""
import re

import cv2


def read_frames(path, limit=None):
    cap = cv2.VideoCapture(str(path))
    n = 0
    try:
        while limit is None or n < limit:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            n += 1
    finally:
        cap.release()


def collect_events(pipeline, frames):
    events, first = [], {}
    for i, frame in enumerate(frames):
        r = pipeline.process(frame)
        for v in r.captures:
            events.append({"frame": i, "kind": "plate_capture", "track": v})
        for lab in r.labels:
            first.setdefault(lab, i)
    for lab, fi in sorted(first.items(), key=lambda x: x[1]):
        events.append({"frame": fi, "kind": "label_first_seen", "label": lab})
    return events


def _key(e, ignore_ids=False):
    k = e.get("track") or e.get("label")
    if ignore_ids:
        k = re.sub(r"_\d+", "_#", k)   # M_128 -> M_#, "ID:C_221 WRONG-WAY" -> "ID:C_# WRONG-WAY"
    return (e["kind"], k)


def compare_events(ref, got, tol=3, ignore_ids=False):
    """Return a list of human-readable problems (empty list == match within +/-tol frames).

    ignore_ids=True is for cross-device comparisons (CPU vs GPU): Ultralytics assigns track ids
    from numerically different detections, so ids shift by a few while type/time still agree.
    Same-device comparisons should keep ids (ignore_ids=False).
    """
    problems = []
    got_by_key = {}
    for e in got:
        got_by_key.setdefault(_key(e, ignore_ids), []).append(e["frame"])
    ref_by_key = {}
    for e in ref:
        ref_by_key.setdefault(_key(e, ignore_ids), []).append(e["frame"])
    for k, frames in ref_by_key.items():
        gf = sorted(got_by_key.get(k, []))
        for f in sorted(frames):
            hit = [g for g in gf if abs(g - f) <= tol]
            if not hit:
                problems.append(f"missing {k} near frame {f} (got frames {gf})")
            else:
                gf.remove(hit[0])
    for k, frames in got_by_key.items():
        if len(frames) > len(ref_by_key.get(k, [])):
            problems.append(f"extra {k} at frames {frames}")
    return problems


_ENTITY = re.compile(r"[MC]_\d+")


def _entity(e):
    """Track id the event is about ('M_77'); falls back to the whole label for events without one (e.g. a bare head label)."""
    text = e.get("track") or e.get("label") or ""
    m = _ENTITY.search(text)
    return m.group(0) if m else text


def diff_events(a, b, tol=3):
    """Compare two event lists keyed by (kind, track entity).

    -> {"missing": events only in a, "added": events only in b, "changed": [{"entity", "kind", "a": [...], "b": [...], "why"}]}
    'changed' = same (kind, entity) in both, but the label (e.g. violation reason) differs or the frame moved by more than tol.
    """
    def group(evs):
        g = {}
        for e in evs:
            g.setdefault((e["kind"], _entity(e)), []).append(e)
        return g
    ga, gb = group(a), group(b)
    out = {"missing": [], "added": [], "changed": []}
    for k in sorted(set(ga) | set(gb), key=lambda k: min(e["frame"] for e in ga.get(k, gb.get(k)))):
        if k not in gb:
            out["missing"] += ga[k]
        elif k not in ga:
            out["added"] += gb[k]
        else:
            la = {e.get("label"): e["frame"] for e in ga[k]}
            lb = {e.get("label"): e["frame"] for e in gb[k]}
            why = []
            for lab in sorted(set(la) - set(lb), key=str):
                why.append(f"label only in A: {lab!r}@{la[lab]}")
            for lab in sorted(set(lb) - set(la), key=str):
                why.append(f"label only in B: {lab!r}@{lb[lab]}")
            for lab in set(la) & set(lb):
                if abs(la[lab] - lb[lab]) > tol:
                    why.append(f"frame moved for {lab!r}: {la[lab]} -> {lb[lab]}")
            if k[0] == "plate_capture":
                fa, fb = ga[k][0]["frame"], gb[k][0]["frame"]
                if abs(fa - fb) > tol:
                    why.append(f"plate capture frame moved: {fa} -> {fb}")
            if why:
                out["changed"].append({"kind": k[0], "entity": k[1], "a": ga[k], "b": gb[k], "why": why})
    return out
