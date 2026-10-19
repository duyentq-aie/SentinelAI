"""Event-level evaluation of predicted violation events against hand-labelled ground truth (PLAN sections 3.3 / 3.4).

Ground truth: eval_sets/events/<video_id>.json (sentinel.labeling schema). Predictions: events.jsonl written by the pipeline (docs/EVIDENCE.md).

Matching, per video and per violation type (never across types):
  candidate pair (gt g, prediction p) if
    time   : p.trigger_frame lies in [g.frame_start - tau*fps, g.frame_end + tau*fps]            (tau = 1 s by default)
    space  : IoU(g.key_box, p's vehicle box at the frame closest to g.key_frame) >= iou, or p's box contains the centre of g.key_box;
             fallback ("relaxed"): IoU(g.key_box, p.key_box at its trigger) >= relaxed_iou when the track has no box near g.key_frame
  optimal one-to-one assignment (Hungarian, cost = 1 - IoU)  ->  TP = matched pair,  FN = unmatched g,
  FP = unmatched p lying inside an annotated range and outside every ignore region (duplicates of an already matched g are FP and counted as duplicates).
Metrics: P / R / F1 per type and micro-averaged, FP per annotated hour, detection latency of TPs, plate exact-match / CER of TPs that have a labelled plate,
bootstrap 95% CI resampling VIDEOS (the unit that is independent), and breakdowns by conditions (day/night ...) and with/without `hard` events.
Pred boxes are rescaled from the pipeline's frame_size to the native video size of the labels.

Ambiguous ground truth (`ambiguous: true` on an event, set when the annotator could not tell whether the violation happened): the event is matched with the
SAME criteria (time tolerance + IoU), but only AFTER the normal events have taken their predictions (an ambiguous event never steals a TP). A prediction that
matches an ambiguous event is dropped from the score: neither TP nor FP (its extra duplicates too). An ambiguous event without a matching prediction is not a FN.
The counts `n_ambiguous_matched` / `n_ambiguous_total` are reported next to the metrics. This is the same idea as the `difficult` flag of PASCAL VOC and
`iscrowd` of COCO: detections on such objects are ignored instead of rewarded or punished.
"""
import json
import random
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from sentinel import labeling

BIG = 10.0                      # assignment cost of a non-candidate pair


# ------------------------------------------------------------------ io
def load_predictions(paths, run=0):
    """events.jsonl file(s) -> {video_id: [event records]}: last record per id, evicted dropped, only events of `run` (None = every run)."""
    last = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            last[(str(p), r["id"])] = r
    out = {}
    for r in last.values():
        if r.get("status") == "evicted" or (run is not None and r.get("run") != run):
            continue
        out.setdefault(r.get("video_id", "?"), []).append(r)
    return out


def load_labels(paths):
    out = {}
    for p in paths:
        lab = labeling.load(p)
        out[lab["video_id"]] = lab
    return out


# ------------------------------------------------------------------ geometry
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def contains_centre(outer, inner):
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def scale_box(box, src_wh, dst_wh):
    if not src_wh or not dst_wh or tuple(src_wh) == tuple(dst_wh):
        return list(box)
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def in_polygon(pt, poly):
    x, y, inside = pt[0], pt[1], False
    n = len(poly)
    for i in range(n):
        x1, y1, x2, y2 = poly[i][0], poly[i][1], poly[(i + 1) % n][0], poly[(i + 1) % n][1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def pred_boxes(p, dst_wh):
    """{frame: box} of a predicted event in native video pixels (sampled boxes + the box at the trigger)."""
    src = p.get("frame_size")
    boxes = {int(k): scale_box(v, src, dst_wh) for k, v in (p.get("boxes") or {}).items()}
    boxes.setdefault(p["trigger_frame"], scale_box(p["key_box"], src, dst_wh))
    return boxes


def pair_score(g, p, boxes, tau_frames, iou_thr, relaxed_iou):
    """-> (iou, relaxed) if (g, p) is a candidate pair else None."""
    t = p["trigger_frame"]
    if not (g["frame_start"] - tau_frames <= t <= g["frame_end"] + tau_frames):
        return None
    near = min(boxes, key=lambda f: abs(f - g["key_frame"]))
    has_box_near = abs(near - g["key_frame"]) <= max(tau_frames, 5) and (p["frame_start"] - tau_frames <= g["key_frame"] <= p["frame_end"] + tau_frames)
    if has_box_near:
        v = iou(g["key_box"], boxes[near])
        if v >= iou_thr or contains_centre(boxes[near], g["key_box"]):
            return max(v, 1e-6), False
        return None
    v = iou(g["key_box"], boxes[p["trigger_frame"]])
    return (max(v, 1e-6), True) if v >= relaxed_iou else None


# ------------------------------------------------------------------ one video
@dataclass
class VideoResult:
    video_id: str
    split: str
    conditions: dict
    hours: float
    counts: dict = field(default_factory=dict)       # type -> {"tp","fp","fn","dup","relaxed","excluded_fp","hard_excluded"}
    latencies: list = field(default_factory=list)    # seconds, TPs
    matches: list = field(default_factory=list)      # dicts for the report
    plate: list = field(default_factory=list)        # (gt text, pred text or None)


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluate_video(labels, preds, tau=1.0, iou_thr=0.3, relaxed_iou=0.1, include_hard=True):
    fps = float(labels["fps"])
    wh = (labels["width"], labels["height"])
    tau_f = int(round(tau * fps))
    ranges = [tuple(r) for r in labels.get("annotated_ranges") or []]
    ignores = labels.get("ignore_regions") or []
    hours = sum((b - a) / fps for a, b in ranges) / 3600.0
    res = VideoResult(labels["video_id"], labels.get("split", "?"), labels.get("conditions") or {}, hours)
    for typ in labeling.EVENT_TYPES:
        G = [e for e in labels["events"] if e["type"] == typ]
        P = [e for e in preds if e["type"] == typ]
        PB = [pred_boxes(p, wh) for p in P]
        amb = [i for i, g in enumerate(G) if g.get("ambiguous")]
        cost = np.full((len(G), len(P)), BIG)
        info = {}
        for i, g in enumerate(G):
            for j, p in enumerate(P):
                sc = pair_score(g, p, PB[j], tau_f, iou_thr, relaxed_iou)
                if sc:
                    cost[i, j] = 1.0 - sc[0]
                    info[(i, j)] = sc
        def assign(rows, cols):
            """Hungarian assignment restricted to the given ground-truth rows and prediction columns -> [(i, j)] of candidate pairs."""
            if not rows or not cols:
                return []
            sub = cost[np.ix_(rows, cols)]
            return [(rows[a], cols[b]) for a, b in zip(*linear_sum_assignment(sub)) if sub[a, b] < BIG]

        normal = [i for i in range(len(G)) if i not in amb]
        pairs = assign(normal, list(range(len(P))))                                  # normal events first ...
        free = [j for j in range(len(P)) if j not in {j for _, j in pairs}]
        amb_pairs = assign(amb, free)                                                 # ... ambiguous ones only get the leftovers
        matched_g, matched_p = {i for i, _ in pairs}, {j for _, j in pairs}
        amb_matched = {i for i, _ in amb_pairs}
        amb_preds = {j for _, j in amb_pairs}
        c = {"tp": 0, "fp": 0, "fn": 0, "dup": 0, "relaxed": 0, "excluded_fp": 0, "hard_excluded": 0, "amb_matched": len(amb_matched), "amb_total": len(amb), "amb_excluded_pred": 0}
        for i, j in pairs:
            g, p = G[i], P[j]
            if not include_hard and g.get("difficulty") == "hard":
                c["hard_excluded"] += 1
                continue
            c["tp"] += 1
            c["relaxed"] += int(info[(i, j)][1])
            lat = (p["trigger_frame"] - g["frame_start"]) / fps
            res.latencies.append(lat)
            res.matches.append({"type": typ, "gt": g["id"], "pred": p["id"], "iou": round(info[(i, j)][0], 3), "relaxed": info[(i, j)][1], "latency_s": round(lat, 3)})
            if g.get("plate_text"):
                pp = p.get("plate_pred") or {}
                res.plate.append((g["plate_text"], pp.get("text")))
        for i, g in enumerate(G):
            if i not in matched_g and i not in amb:                  # an unmatched ambiguous event is not a missed detection
                if not include_hard and g.get("difficulty") == "hard":
                    c["hard_excluded"] += 1
                else:
                    c["fn"] += 1
        for j, p in enumerate(P):
            if j in matched_p:
                continue
            if j in amb_preds or any((i, j) in info for i in amb):        # matches an ambiguous event (or duplicates a prediction that did): not scored
                c["amb_excluded_pred"] += 1
                continue
            kc = scale_box(p["key_box"], p.get("frame_size"), wh)
            centre = ((kc[0] + kc[2]) / 2, (kc[1] + kc[3]) / 2)
            if ranges and not any(a <= p["trigger_frame"] <= b for a, b in ranges):
                c["excluded_fp"] += 1                                  # outside every annotated range: unknown, not counted
            elif any(in_polygon(centre, poly) for poly in ignores):
                c["excluded_fp"] += 1
            elif not include_hard and any((i, j) in info and G[i].get("difficulty") == "hard" for i in range(len(G))):
                c["hard_excluded"] += 1                                # would have matched a hard event we are not scoring
            else:
                c["fp"] += 1
                if any((i, j) in info for i in matched_g):
                    c["dup"] += 1                                      # a second prediction for an event that already has one
        res.counts[typ] = c
    return res


# ------------------------------------------------------------------ aggregation
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": None if p is None else round(p, 4), "recall": None if r is None else round(r, 4), "f1": None if f is None else round(f, 4)}


def total(results, typ=None):
    t = {"tp": 0, "fp": 0, "fn": 0, "dup": 0, "relaxed": 0, "excluded_fp": 0, "hard_excluded": 0, "amb_matched": 0, "amb_total": 0, "amb_excluded_pred": 0}
    for r in results:
        for k, c in r.counts.items():
            if typ in (None, k):
                for key in t:
                    t[key] += c[key]
    return t


def plate_metrics(results):
    pairs = [x for r in results for x in r.plate]
    if not pairs:
        return None
    exact = sum(1 for g, p in pairs if p == g)
    read = [(g, p) for g, p in pairs if p]
    cer_all = st.mean(lev(p or "", g) / len(g) for g, p in pairs)
    cer_read = st.mean(lev(p, g) / len(g) for g, p in read) if read else None
    return {"n_tp_with_plate_label": len(pairs), "pred_text_available": len(read), "exact_match": round(exact / len(pairs), 4),
            "cer_all_counting_unread_as_full_error": round(cer_all, 4), "cer_on_read_only": None if cer_read is None else round(cer_read, 4)}


def bootstrap(results, typ=None, n=1000, seed=0):
    """95% percentile CI of micro P/R/F1 resampling videos with replacement. None when fewer than 2 videos."""
    if len(results) < 2:
        return None
    rng = random.Random(seed)
    vals = {"precision": [], "recall": [], "f1": []}
    for _ in range(n):
        s = prf(*(total([rng.choice(results) for _ in results], typ)[k] for k in ("tp", "fp", "fn")))
        for k in vals:
            if s[k] is not None:
                vals[k].append(s[k])
    def ci(xs):
        if len(xs) < 20:
            return None
        xs = sorted(xs)
        return [round(xs[int(0.025 * (len(xs) - 1))], 4), round(xs[int(0.975 * (len(xs) - 1))], 4)]
    return {k: ci(v) for k, v in vals.items()} | {"n_boot": n, "unit": "video"}


def summarize(results, n_boot=1000, seed=0):
    out = {"videos": [r.video_id for r in results], "splits": sorted({r.split for r in results}), "annotated_hours": round(sum(r.hours for r in results), 4), "types": {}}
    for typ in list(labeling.EVENT_TYPES) + [None]:
        t = total(results, typ)
        row = prf(t["tp"], t["fp"], t["fn"]) | {"duplicates": t["dup"], "relaxed_matches": t["relaxed"], "fp_excluded_outside_annotated_or_ignored": t["excluded_fp"],
                                                 "hard_not_scored": t["hard_excluded"], "n_ambiguous_matched": t["amb_matched"], "n_ambiguous_total": t["amb_total"],
                                                 "predictions_dropped_for_ambiguous": t["amb_excluded_pred"]}
        h = out["annotated_hours"]
        row["fp_per_hour"] = round(t["fp"] / h, 2) if h > 0 else None
        row["duplicate_rate"] = round(t["dup"] / (t["tp"] + t["fp"]), 4) if t["tp"] + t["fp"] else None
        row["ci95_bootstrap_by_video"] = bootstrap(results, typ, n_boot, seed)
        out["types"]["all" if typ is None else typ] = row
    lats = [x for r in results for x in r.latencies]
    out["detection_latency_s"] = {"n": len(lats), "mean": round(st.mean(lats), 3), "median": round(st.median(lats), 3), "min": round(min(lats), 3), "max": round(max(lats), 3)} if lats else None
    out["plate"] = plate_metrics(results)
    return out


def evaluate(labels_by_video, preds_by_video, tau=1.0, iou_thr=0.3, relaxed_iou=0.1, include_hard=True, n_boot=1000, seed=0, splits=None):
    """-> report dict (overall + by condition). Videos without predictions are evaluated with no prediction (all FN)."""
    results = []
    for vid, lab in sorted(labels_by_video.items()):
        if splits and lab.get("split") not in splits:
            continue
        results.append(evaluate_video(lab, preds_by_video.get(vid, []), tau, iou_thr, relaxed_iou, include_hard))
    report = {"settings": {"tau_s": tau, "iou": iou_thr, "relaxed_iou": relaxed_iou, "include_hard": include_hard, "splits": splits, "bootstrap": n_boot, "seed": seed},
              "overall": summarize(results, n_boot, seed), "by_condition": {},
              "unlabelled_prediction_videos": sorted(set(preds_by_video) - set(labels_by_video)),
              "matches": [dict(m, video=r.video_id) for r in results for m in r.matches]}
    keys = {k for r in results for k in r.conditions}
    for k in sorted(keys):
        for v in sorted({str(r.conditions[k]) for r in results if k in r.conditions}):
            sub = [r for r in results if str(r.conditions.get(k)) == v]
            if sub and len(sub) < len(results):
                report["by_condition"][f"{k}={v}"] = summarize(sub, n_boot, seed)
    if not include_hard:
        report["note"] = "events labelled difficulty=hard (and predictions that match them) are not scored"
    return report


def format_report(rep):
    o = rep["overall"]
    L = [f"videos: {', '.join(o['videos'])}  | splits: {', '.join(o['splits'])}  | annotated: {o['annotated_hours']:.3f} h",
         f"matching: tau={rep['settings']['tau_s']} s, IoU>={rep['settings']['iou']} (relaxed {rep['settings']['relaxed_iou']}), hard events {'scored' if rep['settings']['include_hard'] else 'NOT scored'}"]
    if len(o["splits"]) > 1:
        L.append("WARNING: dev and test videos are mixed in this evaluation; report test only.")

    def table(title, s):
        out = [f"\n{title}", "%-10s %4s %4s %4s  %9s %9s %9s  %8s %6s  %s" % ("type", "TP", "FP", "FN", "precision", "recall", "F1", "FP/hour", "dup", "F1 95% CI (video bootstrap)")]
        for typ, r in s["types"].items():
            f = lambda x: "  n/a" if x is None else f"{x:9.3f}"
            ci = (r["ci95_bootstrap_by_video"] or {}).get("f1")
            out.append("%-10s %4d %4d %4d  %s %s %s  %8s %6d  %s" % (typ, r["tp"], r["fp"], r["fn"], f(r["precision"]), f(r["recall"]), f(r["f1"]),
                       "n/a" if r["fp_per_hour"] is None else f"{r['fp_per_hour']:.1f}", r["duplicates"], f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "n/a (< 2 videos or degenerate)"))
        a = s["types"]["all"]
        if a["n_ambiguous_total"]:
            out.append(f"ambiguous ground truth (not scored): {a['n_ambiguous_matched']} of {a['n_ambiguous_total']} matched by a prediction; "
                       f"{a['predictions_dropped_for_ambiguous']} prediction(s) dropped from FP/TP because of them")
        if s["detection_latency_s"]:
            d = s["detection_latency_s"]
            out.append(f"detection latency (trigger - frame_start) over {d['n']} TPs: mean {d['mean']} s, median {d['median']} s, range [{d['min']}, {d['max']}]")
        if s["plate"]:
            out.append(f"plate: {s['plate']}")
        return out
    L += table("OVERALL", o)
    for k, s in rep["by_condition"].items():
        L += table(f"condition {k}", s)
    if rep["unlabelled_prediction_videos"]:
        L.append(f"\nignored predictions for videos without labels: {rep['unlabelled_prediction_videos']}")
    return "\n".join(L)
