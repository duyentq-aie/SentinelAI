from sentinel.perception import Perception, Track
from sentinel.trackdiff import box_distance, diff_tracks, iou, match


def T(i, cls, box, conf=0.9):
    return Track(i, cls, box, conf)


def test_iou_and_distance():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9
    assert box_distance((0, 0, 10, 10), (5, 5, 20, 20)) == 0
    assert box_distance((0, 0, 10, 10), (13, 14, 20, 20)) == 5.0


def test_match_is_per_class_and_ignores_ids():
    pairs, only_a, only_b = match([T(1, "car", (0, 0, 100, 100))], [T(9, "car", (2, 2, 100, 100)), T(2, "bus", (0, 0, 100, 100))])
    assert len(pairs) == 1 and pairs[0][0].id == 1 and pairs[0][1].id == 9
    assert only_a == [] and [t.cls for t in only_b] == ["bus"]


def test_identical_tracks_have_no_change():
    p = Perception(coco=[T(1, "car", (0, 0, 100, 100)), T(2, "traffic light", (500, 20, 520, 60))], helmet=[T(3, "motorcycle", (10, 10, 90, 90))], light="RED")
    d = diff_tracks([p, p], [p, p])
    for m in ("coco", "helmet"):
        g = d["models"][m]
        assert (g["lost"], g["new"], g["id_changed"], g["frames_with_change"]) == (0, 0, 0, 0)
    assert d["light"]["frames_state_differs"] == 0


def test_lost_new_idchange_and_nearness_to_light_are_reported():
    light = T(7, "traffic light", (500, 20, 520, 60))
    a = Perception(coco=[light], helmet=[T(1, "no-helmet", (510, 70, 540, 100)), T(2, "helmet", (900, 500, 950, 560)), T(3, "motorcycle", (100, 300, 200, 400))])
    b = Perception(coco=[light], helmet=[T(5, "motorcycle", (100, 300, 200, 400)), T(8, "helmet", (900, 500, 950, 560))])
    d = diff_tracks([a], [b])
    h = d["models"]["helmet"]
    assert h["lost"] == 1 and h["new"] == 0 and h["id_changed"] == 2 and h["matched"] == 2
    assert h["by_class"]["no-helmet"] == {"lost": 1}                    # the head just under the light was lost
    assert h["near_light_changed"] == 1 and h["changed_total"] == 3     # only that head is within 60 px of the light box
    assert d["light_boxes"] == [[500, 20, 520, 60]]


def test_id_consistency_ignores_numbering_offsets_and_counts_splits_and_merges():
    from sentinel.trackdiff import id_consistency
    box = lambda x: (x, 100, x + 80, 200)
    frames = lambda ids: [Perception(helmet=[T(i, "motorcycle", box(100 + 10 * k)) for i in ids(k)]) for k in range(12)]
    a = frames(lambda k: [1])
    shifted = frames(lambda k: [51])                       # same object, different numbering
    split = frames(lambda k: [51] if k < 6 else [52])      # tracker lost the object and re-created it half way
    assert id_consistency(a, shifted)["helmet"]["split"] == 0 and id_consistency(a, shifted)["helmet"]["merged"] == 0
    r = id_consistency(a, split)["helmet"]
    assert (r["split"], r["merged"], r["tracks_a"], r["tracks_b"]) == (1, 0, 1, 2)
    m = id_consistency(split, a)["helmet"]                 # reverse: B glues two A tracks
    assert (m["split"], m["merged"]) == (0, 1)
