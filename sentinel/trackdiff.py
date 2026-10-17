"""Compare two recorded tracks files frame by frame (what changed in perception, and where).

Detections are matched per frame by class + IoU (ids are ignored for matching, because ids are exactly what may change).
"""
from collections import Counter, defaultdict

MODELS = ("coco", "helmet")
GRID = (4, 3)                      # columns x rows over the frame, for the "where" histogram
FRAME = (1280, 720)
NEAR_PX = 60                       # "near a traffic light" = within this many px of its box


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def box_distance(a, b):
    """Gap in px between two boxes (0 if they touch/overlap)."""
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def match(ta, tb, min_iou=0.5):
    """Greedy best-IoU matching within the same class. -> (pairs, only_a, only_b)"""
    cand = sorted(((iou(x.box, y.box), i, j) for i, x in enumerate(ta) for j, y in enumerate(tb) if x.cls == y.cls), reverse=True)
    used_a, used_b, pairs = set(), set(), []
    for v, i, j in cand:
        if v < min_iou:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); pairs.append((ta[i], tb[j]))
    return pairs, [t for i, t in enumerate(ta) if i not in used_a], [t for j, t in enumerate(tb) if j not in used_b]


def cell(box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return (min(GRID[0] - 1, int(cx / FRAME[0] * GRID[0])), min(GRID[1] - 1, int(cy / FRAME[1] * GRID[1])))


def diff_tracks(a, b, min_iou=0.5):
    n = min(len(a), len(b))
    out = {"frames_a": len(a), "frames_b": len(b), "frames_compared": n, "models": {}, "light": {}, "light_boxes": []}
    light_boxes = Counter()
    for pa, pb in zip(a[:n], b[:n]):
        for t in pa.coco + pb.coco:
            if t.cls == "traffic light":
                light_boxes[tuple(t.box)] += 1
    # coarse union of every place a traffic light was ever seen (rounded to 20 px) -> "light ROI"
    lights = [list(k) for k, _ in light_boxes.most_common(8)]
    out["light_boxes"] = lights

    for m in MODELS:
        agg = {"detections_a": 0, "detections_b": 0, "matched": 0, "lost": 0, "new": 0, "id_changed": 0, "frames_with_change": 0,
               "by_class": defaultdict(lambda: Counter()), "grid_changed": Counter(), "near_light_changed": 0, "changed_total": 0,
               "near_light_all_detections": 0, "all_detections": 0, "box_shift_px_mean": 0.0, "box_shift_px_max": 0.0, "first_change_frame": None}
        shifts = []
        for i, (pa, pb) in enumerate(zip(a[:n], b[:n])):
            ta, tb = getattr(pa, m), getattr(pb, m)
            pairs, only_a, only_b = match(ta, tb, min_iou)
            agg["detections_a"] += len(ta); agg["detections_b"] += len(tb)
            agg["matched"] += len(pairs); agg["lost"] += len(only_a); agg["new"] += len(only_b)
            changed_here = len(only_a) + len(only_b) + sum(1 for x, y in pairs if x.id != y.id)
            agg["id_changed"] += sum(1 for x, y in pairs if x.id != y.id)
            if changed_here and agg["first_change_frame"] is None:
                agg["first_change_frame"] = i
            agg["frames_with_change"] += 1 if changed_here else 0
            for x, y in pairs:
                shifts.append(max(abs(p - q) for p, q in zip(x.box, y.box)))
            lbs = [t.box for t in pa.coco + pb.coco if t.cls == "traffic light"]
            for t in ta + tb:
                near = any(box_distance(t.box, lb) <= NEAR_PX for lb in lbs)
                agg["all_detections"] += 1
                agg["near_light_all_detections"] += near
            for t, kind in [(t, "lost") for t in only_a] + [(t, "new") for t in only_b] + [(y, "id_changed") for x, y in pairs if x.id != y.id]:
                agg["by_class"][t.cls][kind] += 1
                agg["grid_changed"][cell(t.box)] += 1
                agg["changed_total"] += 1
                agg["near_light_changed"] += any(box_distance(t.box, lb) <= NEAR_PX for lb in lbs)
        agg["box_shift_px_mean"] = round(sum(shifts) / len(shifts), 2) if shifts else 0.0
        agg["box_shift_px_max"] = max(shifts) if shifts else 0.0
        agg["by_class"] = {k: dict(v) for k, v in agg["by_class"].items()}
        agg["grid_changed"] = {f"col{c}_row{r}": v for (c, r), v in sorted(agg["grid_changed"].items())}
        out["models"][m] = agg
    la, lb = Counter(p.light for p in a[:n]), Counter(p.light for p in b[:n])
    out["id_consistency"] = id_consistency(a, b, min_iou=min_iou)
    out["light"] = {"frames_state_differs": sum(1 for pa, pb in zip(a[:n], b[:n]) if pa.light != pb.light),
                    "states_a": dict(la), "states_b": dict(lb)}
    return out


def id_consistency(a, b, min_frames=5, min_iou=0.5):
    """How differently two trackers number the SAME objects (raw id equality is meaningless: numbering offsets are arbitrary).

    Detections are matched per frame (class + IoU); for every model:
      split  = tracks of A whose detections are spread over >1 id of B (second id covers >= min_frames frames)  -> B fragments / switches ids
      merged = tracks of B that cover >1 id of A (same rule)                                                    -> B glues different objects of A
    """
    out = {}
    n = min(len(a), len(b))
    for m in MODELS:
        fwd, rev, ids_a, ids_b = defaultdict(Counter), defaultdict(Counter), set(), set()
        for pa, pb in zip(a[:n], b[:n]):
            ta, tb = getattr(pa, m), getattr(pb, m)
            ids_a |= {t.id for t in ta}; ids_b |= {t.id for t in tb}
            for x, y in match(ta, tb, min_iou)[0]:
                fwd[x.id][y.id] += 1
                rev[y.id][x.id] += 1
        second = lambda c: c.most_common(2)[1][1] if len(c) > 1 else 0
        out[m] = {"tracks_a": len(ids_a), "tracks_b": len(ids_b), "matched_tracks_a": len(fwd), "matched_tracks_b": len(rev),
                  "split": sum(1 for c in fwd.values() if second(c) >= min_frames),
                  "merged": sum(1 for c in rev.values() if second(c) >= min_frames)}
    return out


def format_report(d, name_a="A", name_b="B"):
    L = [f"tracks diff: {name_a} ({d['frames_a']} frames)  vs  {name_b} ({d['frames_b']} frames); compared {d['frames_compared']}",
         f"traffic-light boxes seen (x1,y1,x2,y2; most frequent first): {d['light_boxes']}",
         f"traffic-light state differs on {d['light']['frames_state_differs']} frames; states A={d['light']['states_a']} B={d['light']['states_b']}"]
    for m, g in d["models"].items():
        pct = lambda x, y: f"{100 * x / y:.1f}%" if y else "n/a"
        ic = d["id_consistency"][m]
        L += [f"\n[{m}] ids: {ic['tracks_a']} tracks in A, {ic['tracks_b']} in B; A-tracks split over >1 B-id: {ic['split']}/{ic['matched_tracks_a']}; "
              f"B-tracks merging >1 A-id: {ic['merged']}/{ic['matched_tracks_b']}  (numbering offsets are not counted)"]
        L += [f"[{m}] detections A={g['detections_a']} B={g['detections_b']} | matched={g['matched']} lost(only A)={g['lost']} new(only B)={g['new']} "
              f"id-changed={g['id_changed']}",
              f"   frames with any change: {g['frames_with_change']}/{d['frames_compared']} ({pct(g['frames_with_change'], d['frames_compared'])}), first at frame {g['first_change_frame']}",
              f"   matched boxes moved: mean {g['box_shift_px_mean']} px, max {g['box_shift_px_max']} px",
              f"   changed detections near a traffic light (<= {NEAR_PX}px): {g['near_light_changed']}/{g['changed_total']}; "
              f"for reference, {g['near_light_all_detections']}/{g['all_detections']} of ALL detections are near a light",
              f"   by class: {g['by_class']}", f"   where (grid {GRID[0]}x{GRID[1]}, col/row): {g['grid_changed']}"]
    return "\n".join(L)
