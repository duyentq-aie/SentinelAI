"""Event evaluator on simulated ground truth / predictions (no video, no models)."""
import json

import pytest

from sentinel import evaluation as ev
from sentinel import labeling

FPS, W, H = 30.0, 1920, 1080


def labels(events=(), ranges=((0, 30000),), video="v1", split="test", **kw):
    lab = labeling.new_labels(video, FPS, W, H, split, annotated_ranges=[list(r) for r in ranges])
    lab["events"] = list(events)
    lab.update(kw)
    return lab


def gt(i, typ="no_helmet", start=100, end=160, key=130, box=(600, 300, 800, 700), plate=None, difficulty="normal"):
    return {"id": f"e{i:03d}", "type": typ, "frame_start": start, "frame_end": end, "key_frame": key, "key_box": list(box),
            "plate_text": plate, "difficulty": difficulty, "notes": ""}


def pred(i, typ="no_helmet", trigger=105, box_small=(400, 200, 533, 467), end=170, plate_pred=None, video="v1"):
    """box in the pipeline's 1280x720 frame: (x * 2/3, y * 2/3) of the 1920x1080 GT box by default (600,300,800,700 -> 400,200,533,467)."""
    return {"id": f"p{i:03d}", "type": typ, "frame_start": trigger, "frame_end": end, "key_frame": trigger, "trigger_frame": trigger, "key_box": list(box_small),
            "boxes": {str(trigger): list(box_small)}, "frame_size": [1280, 720], "video_id": video, "plate_pred": plate_pred}


def run(lab, preds, **kw):
    return ev.evaluate_video(lab, preds, **kw)


def counts(res, typ="no_helmet"):
    return res.counts[typ]


def test_perfect_match_with_box_rescaling_and_latency():
    r = run(labels([gt(1)]), [pred(1, trigger=115)])
    assert (counts(r)["tp"], counts(r)["fp"], counts(r)["fn"]) == (1, 0, 0)
    assert r.latencies == [pytest.approx(0.5)]                       # (115 - 100) / 30 fps
    assert r.matches[0]["iou"] > 0.95 and not r.matches[0]["relaxed"]


def test_missed_event_is_a_false_negative_and_a_spurious_one_a_false_positive():
    r = run(labels([gt(1)]), [pred(1, trigger=5000, box_small=(10, 10, 90, 90), end=5100)])
    assert (counts(r)["tp"], counts(r)["fp"], counts(r)["fn"]) == (0, 1, 1)


def test_time_tolerance_is_one_second_by_default():
    ok = run(labels([gt(1)]), [pred(1, trigger=100 - 30)])                # exactly tau before frame_start
    late = run(labels([gt(1)]), [pred(1, trigger=160 + 31, end=300)])      # one frame beyond frame_end + tau
    assert counts(ok)["tp"] == 1 and counts(late)["tp"] == 0 and counts(late)["fp"] == 1
    assert counts(run(labels([gt(1)]), [pred(1, trigger=100 - 60)], tau=2.0))["tp"] == 1


def test_wrong_place_is_not_a_match():
    r = run(labels([gt(1)]), [pred(1, box_small=(900, 100, 1000, 300))])
    assert (counts(r)["tp"], counts(r)["fp"], counts(r)["fn"]) == (0, 1, 1)


def test_types_are_never_mixed():
    r = run(labels([gt(1, typ="no_helmet")]), [pred(1, typ="overload")])
    assert (counts(r)["tp"], counts(r)["fn"]) == (0, 1) and counts(r, "overload")["fp"] == 1


def test_hungarian_gives_each_gt_its_own_prediction():
    g1, g2 = gt(1, box=(600, 300, 800, 700)), gt(2, start=110, end=170, key=140, box=(700, 300, 900, 700))
    p1 = pred(1, box_small=(450, 200, 600, 467))        # overlaps both ground truths, a bit more with g2
    p2 = pred(2, trigger=112, box_small=(400, 200, 533, 467))
    r = run(labels([g1, g2]), [p1, p2])
    assert (counts(r)["tp"], counts(r)["fp"], counts(r)["fn"]) == (2, 0, 0)
    assert sorted(m["gt"] for m in r.matches) == ["e001", "e002"] and sorted(m["pred"] for m in r.matches) == ["p001", "p002"]


def test_duplicate_prediction_for_one_event_is_a_false_positive_and_counted_as_duplicate():
    r = run(labels([gt(1)]), [pred(1, trigger=105), pred(2, trigger=108)])
    c = counts(r)
    assert (c["tp"], c["fp"], c["fn"], c["dup"]) == (1, 1, 0, 1)
    s = ev.summarize([r], n_boot=10)
    assert s["types"]["no_helmet"]["duplicate_rate"] == 0.5


def test_false_positives_only_count_inside_annotated_ranges():
    far = pred(1, trigger=9000, box_small=(10, 10, 90, 90), end=9100)
    inside = run(labels([], ranges=((8000, 10000),)), [far])
    outside = run(labels([], ranges=((0, 1000),)), [far])
    assert counts(inside)["fp"] == 1 and counts(outside)["fp"] == 0 and counts(outside)["excluded_fp"] == 1


def test_ignore_regions_exclude_false_positives():
    lab = labels([], ignore_regions=[[[0, 0], [300, 0], [300, 300], [0, 300]]])
    r = run(lab, [pred(1, trigger=500, box_small=(10, 10, 100, 100), end=520)])      # centre (55,55)*1.5 = (82, 82): inside the polygon
    assert counts(r)["fp"] == 0 and counts(r)["excluded_fp"] == 1


def test_hard_events_can_be_left_out_of_the_score():
    lab = labels([gt(1), gt(2, start=1000, end=1060, key=1030, box=(100, 100, 300, 400), difficulty="hard")])
    p = [pred(1)]
    with_hard = run(lab, p, include_hard=True)
    without = run(lab, p, include_hard=False)
    assert (counts(with_hard)["tp"], counts(with_hard)["fn"]) == (1, 1)
    assert (counts(without)["tp"], counts(without)["fn"], counts(without)["hard_excluded"]) == (1, 0, 1)
    # a prediction that matches a hard event is neither TP nor FP when hard events are not scored
    both = run(lab, [pred(1), pred(2, trigger=1010, box_small=(67, 67, 200, 267), end=1070)], include_hard=False)
    assert (counts(both)["tp"], counts(both)["fp"]) == (1, 0)


def test_relaxed_match_when_the_track_has_no_box_near_the_key_frame():
    p = pred(1, trigger=100 - 20, end=100 - 10)                     # track ended before the labelled key frame (130): only its trigger box exists
    r = run(labels([gt(1)]), [p])
    assert counts(r)["tp"] == 1 and counts(r)["relaxed"] == 1 and r.matches[0]["relaxed"] is True


def test_plate_exact_match_and_cer():
    lab = labels([gt(1, plate="29B112345"), gt(2, start=1000, end=1060, key=1030, box=(100, 100, 300, 400), plate="30A12345")])
    p = [pred(1, plate_pred={"text": "29B112345"}), pred(2, trigger=1010, box_small=(67, 67, 200, 267), end=1070, plate_pred={"text": None})]
    s = ev.summarize([run(lab, p)], n_boot=5)
    pl = s["plate"]
    assert pl["n_tp_with_plate_label"] == 2 and pl["exact_match"] == 0.5 and pl["pred_text_available"] == 1
    assert pl["cer_on_read_only"] == 0.0 and pl["cer_all_counting_unread_as_full_error"] == 0.5


def test_aggregate_prf_fp_per_hour_and_latency():
    r1 = run(labels([gt(1), gt(2, start=2000, end=2060, key=2030, box=(100, 100, 300, 400))], ranges=((0, 108000),)),     # 1 hour at 30 fps
             [pred(1), pred(2, trigger=4000, box_small=(10, 10, 90, 90), end=4100)])
    s = ev.summarize([r1], n_boot=5)
    t = s["types"]["no_helmet"]
    assert (t["tp"], t["fp"], t["fn"]) == (1, 1, 1) and t["precision"] == 0.5 and t["recall"] == 0.5 and t["f1"] == 0.5
    assert s["annotated_hours"] == pytest.approx(1.0) and t["fp_per_hour"] == 1.0
    assert s["detection_latency_s"]["n"] == 1 and s["types"]["all"]["tp"] == 1


def test_bootstrap_resamples_videos_and_is_deterministic():
    good = run(labels([gt(1)], video="a"), [pred(1, video="a")])
    bad = run(labels([gt(1)], video="b"), [])
    ci1 = ev.bootstrap([good, bad], "no_helmet", n=300, seed=7)
    ci2 = ev.bootstrap([good, bad], "no_helmet", n=300, seed=7)
    assert ci1 == ci2 and ci1["unit"] == "video"
    lo, hi = ci1["recall"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0                           # the point estimate (1 of 2 events) lies inside the interval
    assert ev.bootstrap([good], "no_helmet") is None               # one video: no interval is claimed


def test_end_to_end_with_files_conditions_and_unlabelled_videos(tmp_path):
    lab_a = labels([gt(1)], video="a", conditions={"time": "day", "weather": "dry"})
    lab_b = labels([gt(1)], video="b", conditions={"time": "night", "weather": "dry"})
    labeling.save(tmp_path / "a.json", lab_a)
    labeling.save(tmp_path / "b.json", lab_b)
    lines = [dict(pred(1, video="a"), status="open", run=0), dict(pred(1, video="a"), status="closed", run=0),     # last record per id wins
             dict(pred(9, video="b", trigger=5000, box_small=(10, 10, 80, 80), end=5100), status="closed", run=0),
             dict(pred(2, video="zzz"), status="closed", run=0), dict(pred(3, video="a"), status="closed", run=1),   # other video / other run
             {"id": "p003", "status": "evicted"}]
    f = tmp_path / "events.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    rep = ev.evaluate(ev.load_labels(sorted(tmp_path.glob("*.json"))), ev.load_predictions([f], run=0), n_boot=20)
    o = rep["overall"]["types"]["no_helmet"]
    assert (o["tp"], o["fp"], o["fn"]) == (1, 1, 1)
    assert rep["unlabelled_prediction_videos"] == ["zzz"]
    assert set(rep["by_condition"]) == {"time=day", "time=night"}
    assert rep["by_condition"]["time=day"]["types"]["no_helmet"]["tp"] == 1 and rep["by_condition"]["time=night"]["types"]["no_helmet"]["fn"] == 1
    text = ev.format_report(rep)
    assert "OVERALL" in text and "no_helmet" in text and "condition time=night" in text


def test_mixed_splits_trigger_a_warning_and_split_filter_works():
    a, b = labels([gt(1)], video="a", split="dev"), labels([gt(1)], video="b", split="test")
    both = ev.evaluate({"a": a, "b": b}, {}, n_boot=5)
    assert "WARNING" in ev.format_report(both)
    only_test = ev.evaluate({"a": a, "b": b}, {}, n_boot=5, splits=["test"])
    assert only_test["overall"]["videos"] == ["b"] and "WARNING" not in ev.format_report(only_test)


def test_levenshtein():
    assert ev.lev("29B112345", "29B112345") == 0 and ev.lev("29B112345", "29B11234") == 1 and ev.lev("", "abc") == 3 and ev.lev("kitten", "sitting") == 3


# ---- ambiguous ground truth: one normal event N (frames 100-160) and one ambiguous event A (frames 1000-1060), same type, different places and times
N = gt(1)
A = dict(gt(2, start=1000, end=1060, key=1030, box=(100, 100, 300, 400)), ambiguous=True)
P_N = pred(1)                                                         # hits N
P_A = pred(2, trigger=1010, box_small=(67, 67, 200, 267), end=1070)   # hits A
P_X = pred(3, trigger=4000, box_small=(10, 10, 90, 90), end=4100)     # hits nothing


def amb_summary(preds):
    r = run(labels([N, A]), preds)
    return r, ev.summarize([r], n_boot=5)["types"]["no_helmet"]


def test_ambiguous_case_1_prediction_hits_the_ambiguous_event_it_is_neither_tp_nor_fp():
    r, t = amb_summary([P_N, P_A])
    assert (counts(r)["tp"], counts(r)["fp"], counts(r)["fn"]) == (1, 0, 0)
    assert (t["precision"], t["recall"], t["f1"]) == (1.0, 1.0, 1.0)             # by hand: TP 1, FP 0, FN 0
    assert (t["n_ambiguous_matched"], t["n_ambiguous_total"], t["predictions_dropped_for_ambiguous"]) == (1, 1, 1)
    assert [m["gt"] for m in r.matches] == ["e001"] and len(r.latencies) == 1    # only the normal event produces a match / latency


def test_ambiguous_case_2_missing_the_ambiguous_event_costs_nothing():
    r, t = amb_summary([P_N])
    assert (counts(r)["tp"], counts(r)["fp"], counts(r)["fn"]) == (1, 0, 0)       # not a false negative
    assert (t["precision"], t["recall"], t["f1"]) == (1.0, 1.0, 1.0)
    assert (t["n_ambiguous_matched"], t["n_ambiguous_total"]) == (0, 1)


def test_ambiguous_case_3_hitting_the_normal_event_is_scored_as_usual_with_a_false_positive_mixed_in():
    r, t = amb_summary([P_N, P_X])
    assert (counts(r)["tp"], counts(r)["fp"], counts(r)["fn"]) == (1, 1, 0)
    assert t["precision"] == 0.5 and t["recall"] == 1.0 and t["f1"] == pytest.approx(0.6667, abs=1e-4)      # by hand: P 1/2, R 1/1, F1 2*.5*1/1.5
    assert (t["n_ambiguous_matched"], t["n_ambiguous_total"]) == (0, 1)


def test_ambiguous_hit_does_not_rescue_a_missed_normal_event():
    r, t = amb_summary([P_A])                                                    # only the ambiguous event is hit
    assert (counts(r)["tp"], counts(r)["fp"], counts(r)["fn"]) == (0, 0, 1)
    assert t["precision"] is None and t["recall"] == 0.0 and t["f1"] is None     # TP 0, FP 0, FN 1: precision undefined, recall 0


def test_ambiguous_event_never_steals_a_prediction_from_a_normal_event_and_duplicates_are_not_fp():
    amb_overlap = dict(gt(2, start=100, end=160, key=130, box=(610, 310, 810, 710)), ambiguous=True)      # same place and time as N, slightly shifted
    r = run(labels([N, amb_overlap]), [P_N])
    assert (counts(r)["tp"], counts(r)["fp"], counts(r)["fn"]) == (1, 0, 0) and [m["gt"] for m in r.matches] == ["e001"]
    both = run(labels([N, amb_overlap]), [P_N, pred(2, trigger=108)])            # two predictions for two overlapping events: one TP, one ambiguous match
    assert (counts(both)["tp"], counts(both)["fp"]) == (1, 0) and counts(both)["amb_matched"] == 1
    dup = run(labels([A]), [P_A, pred(5, trigger=1012, box_small=(67, 67, 200, 267), end=1070)])           # a second prediction for the ambiguous event
    assert (counts(dup)["fp"], counts(dup)["tp"], counts(dup)["amb_excluded_pred"]) == (0, 0, 2)                  # the matched prediction and its duplicate are both dropped


def test_ambiguous_counts_appear_in_the_report_text_and_overall_row():
    rep = ev.evaluate({"v1": labels([N, A])}, {"v1": [P_N, P_A]}, n_boot=5)
    assert rep["overall"]["types"]["all"]["n_ambiguous_matched"] == 1 and rep["overall"]["types"]["all"]["n_ambiguous_total"] == 1
    assert "ambiguous ground truth (not scored): 1 of 1 matched" in ev.format_report(rep)
    assert "ambiguous" not in ev.format_report(ev.evaluate({"v1": labels([N])}, {"v1": [P_N]}, n_boot=5))        # no line when there is none


# ---- undefined precision (no prediction counted) through every output path
def two_videos(pred_b=True):
    a = run(labels([gt(1)], video="a"), [])                                     # GT, no prediction at all: TP 0, FP 0, FN 1 -> precision undefined
    b = run(labels([gt(1)], video="b"), [pred(1, video="b")] if pred_b else [])
    return a, b


def test_bootstrap_drops_resamples_with_undefined_precision_and_reports_how_many():
    """Also a regression test: tp, fp and fn of one resample must come from the SAME draw (they once came from three independent draws, so the
    dropped count would not match the replayed draws below)."""
    import random
    a, b = two_videos()
    n, seed = 300, 3
    rng = random.Random(seed)
    only_a = sum(1 for _ in range(n) if [rng.choice([a, b]) for _ in range(2)] == [a, a])      # replays the bootstrap's own draws (both draws are always made)
    boot = ev.bootstrap([a, b], "no_helmet", n=n, seed=seed)
    assert only_a > 0
    assert boot["n_resamples_dropped"]["precision"] == boot["n_resamples_dropped"]["f1"] == only_a
    assert boot["n_resamples_dropped"]["recall"] == 0                                         # recall is defined whenever a GT event is drawn
    assert boot["n_resamples_used"]["precision"] + boot["n_resamples_dropped"]["precision"] == n
    assert boot["precision"] == [1.0, 1.0]                                                    # every usable resample contains the single TP: never silently 0
    assert boot["recall"][0] == 0.0 and boot["recall"][1] == 1.0                              # a dropped resample is not mixed into the recall interval either


def test_bootstrap_with_no_prediction_in_any_video_gives_null_intervals_not_zeros_or_a_crash():
    a, b = two_videos(pred_b=False)
    boot = ev.bootstrap([a, b], "no_helmet", n=200, seed=1)
    assert boot["precision"] is None and boot["f1"] is None
    assert boot["n_resamples_dropped"]["precision"] == boot["n_resamples_dropped"]["f1"] == 200 and boot["n_resamples_used"]["precision"] == 0
    assert boot["recall"] == [0.0, 0.0]                                                       # recall is defined (0 of 2) and is really 0
    empty = ev.bootstrap([run(labels([], video="x"), []), run(labels([], video="y"), [])], "no_helmet", n=50)   # nothing labelled, nothing predicted
    assert empty["precision"] is None and empty["recall"] is None and empty["n_resamples_dropped"] == {"precision": 50, "recall": 50, "f1": 50}


def test_undefined_precision_is_null_in_json_n_a_in_text_and_the_dropped_resamples_are_stated():
    rep = ev.evaluate({"a": labels([gt(1)], video="a"), "b": labels([gt(1)], video="b")}, {"b": [pred(1, video="b")]}, n_boot=200, seed=3)
    only_a = ev.evaluate({"a": labels([gt(1)], video="a")}, {}, n_boot=50)                      # one video, no prediction: precision undefined everywhere
    for r in (rep, only_a):
        text = json.dumps(r, allow_nan=False)                                                   # would raise on NaN / Infinity
        assert "NaN" not in text and json.loads(text)["overall"]["types"]["all"] is not None
        assert isinstance(ev.format_report(r), str)
    row = only_a["overall"]["types"]["no_helmet"]
    assert row["precision"] is None and row["f1"] is None and row["recall"] == 0.0 and row["ci95_bootstrap_by_video"] is None      # 1 video: no CI claimed
    txt = ev.format_report(only_a)
    line = next(l for l in txt.splitlines() if l.startswith("no_helmet"))
    assert "n/a" in line and "0.000" in line                                                    # precision and F1 print n/a, recall prints a real 0.000
    boot_row = rep["overall"]["types"]["no_helmet"]
    assert boot_row["precision"] == 1.0 and boot_row["ci95_bootstrap_by_video"]["n_resamples_dropped"]["precision"] > 0
    assert "dropped, not counted as 0" in ev.format_report(rep)
