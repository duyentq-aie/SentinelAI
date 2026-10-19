import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import eval_ood as eo  # noqa: E402
import make_ood_set as mo  # noqa: E402

CLASSES = ["helmet", "motorcycle", "no-helmet"]
ROI = [0, 100, 400, 400]                                   # lower part of a 400x400 frame
A, B, C = [10, 150, 60, 200], [100, 150, 150, 200], [200, 300, 260, 360]          # three boxes inside the ROI
OUT = [10, 10, 60, 60]                                     # centre outside the ROI


def gt(*items):
    return {"img1": list(items)}


CEIL = 1.0                                                 # COCO-style AP: a perfect detector scores 1 (Ultralytics' own recipe would give 0.995)


def test_perfect_predictions_reach_the_ap_ceiling_and_full_counts():
    g = gt(("helmet", A), ("helmet", B), ("no-helmet", C))
    p = {"img1": [("helmet", 0.9, A), ("helmet", 0.8, B), ("no-helmet", 0.7, C)]}
    r = eo.evaluate(g, p, ROI, CLASSES)
    assert (r["helmet"]["n_gt"], r["no-helmet"]["n_gt"], r["motorcycle"]["n_gt"]) == (2, 1, 0)
    assert r["helmet"]["ap50"] == CEIL and r["helmet"]["ap50_95"] == CEIL and r["helmet"]["precision"] == 1.0 and r["helmet"]["recall"] == 1.0
    assert r["motorcycle"]["ap50"] is None and r["all"]["ap50"] == CEIL and r["all"]["classes_with_gt"] == 2        # classes without GT are not averaged in


def test_ap_and_pr_match_a_hand_computation_with_one_false_positive_and_one_miss():
    g = gt(("helmet", A), ("helmet", B))
    wrong = [300, 150, 350, 200]
    p = {"img1": [("helmet", 0.9, A), ("helmet", 0.8, wrong), ("helmet", 0.7, B)]}
    r = eo.evaluate(g, p, ROI, CLASSES)["helmet"]
    # ranked: TP(.9), FP(.8), TP(.7): (R, P) = (.5, 1), (.5, .5), (1, .667) -> envelope: 51 thresholds at 1.0, 50 at .667 -> AP = (51 + 50*.667)/101 = .835 (~ .5*1 + .5*.667)
    assert r["ap50"] == pytest.approx(0.835, abs=0.002)
    assert r["precision"] == pytest.approx(2 / 3, abs=1e-3) and r["recall"] == 1.0           # all three predictions are above conf 0.25
    miss = eo.evaluate(g, {"img1": [("helmet", 0.9, A)]}, ROI, CLASSES)["helmet"]
    assert miss["recall"] == 0.5 and miss["precision"] == 1.0 and miss["ap50"] == pytest.approx(51 / 101, abs=1e-3)        # recall never exceeds .5: the rest of the curve is 0
    none = eo.evaluate(g, {"img1": []}, ROI, CLASSES)["helmet"]
    assert none["ap50"] == 0.0 and none["recall"] == 0.0 and none["precision"] is None        # undefined precision stays n/a, never 0


def test_ap50_95_is_lower_for_a_loose_box_and_confidence_threshold_changes_pr_only():
    g = gt(("helmet", A))
    loose = [10, 150, 60, 215]                                                                # IoU with A ~ 0.77
    r = eo.evaluate(g, {"img1": [("helmet", 0.9, loose)]}, ROI, CLASSES)["helmet"]
    assert r["ap50"] == CEIL and 0.3 < r["ap50_95"] < 0.9
    low = eo.evaluate(g, {"img1": [("helmet", 0.1, A)]}, ROI, CLASSES)["helmet"]             # below conf 0.25: no P/R, but AP counts it
    assert low["ap50"] == CEIL and low["precision"] is None and low["recall"] == 0.0


def test_only_boxes_with_their_centre_in_the_roi_are_counted_on_both_sides():
    g = gt(("helmet", A), ("helmet", OUT))
    p = {"img1": [("helmet", 0.9, A), ("helmet", 0.9, [300, 10, 350, 60])]}                   # a prediction outside the ROI: not a false positive
    r = eo.evaluate(g, p, ROI, CLASSES)["helmet"]
    assert (r["n_gt"], r["n_pred"], r["ap50"], r["precision"]) == (1, 1, CEIL, 1.0)


def test_label_reading_handles_roboflow_names_denormalises_and_ignores_unknown_classes(tmp_path):
    lab = tmp_path / "export" / "train" / "labels"
    lab.mkdir(parents=True)
    (tmp_path / "export" / "data.yaml").write_text("nc: 4\nnames: ['helmet', 'motorcycle', 'no-helmet', 'plate']\n", encoding="utf-8")
    (lab / "ood_f0026_t00.87s_png.rf.0123abcd.txt").write_text("0 0.5 0.5 0.25 0.5\n3 0.5 0.5 0.1 0.1\n2 0.25 0.75 0.1 0.2\n", encoding="utf-8")
    gt_, warn = eo.read_labels(tmp_path / "export", {"ood_f0026_t00.87s": (400, 200), "ood_f0078": (400, 200)}, CLASSES)
    assert gt_["ood_f0026_t00.87s"][0] == ("helmet", pytest.approx([150, 50, 250, 150]))      # (0.5 +- 0.125) * 400, (0.5 +- 0.25) * 200
    assert [k for k, _ in gt_["ood_f0026_t00.87s"]] == ["helmet", "no-helmet"]                # the plate box is dropped
    assert gt_["ood_f0078"] == [] and "1 image(s) without a label file" in warn[0]


def test_frame_selection_is_even_and_inside_the_clip():
    idx = mo.frame_indices(787, 15)
    assert len(idx) == 15 and idx[0] > 0 and idx[-1] < 786 and all(b - a in (52, 53) for a, b in zip(idx, idx[1:]))


def test_merge_nms_keeps_the_most_confident_box_per_class_and_never_merges_across_classes():
    d = [(0, 0.5, [0, 0, 100, 100], "old"), (0, 0.9, [5, 5, 100, 100], "new"), (2, 0.4, [0, 0, 100, 100], "old"), (0, 0.6, [300, 300, 360, 360], "old")]
    kept = mo.merge_nms(d, 0.5)
    assert [(k[0], k[3]) for k in kept] == [(0, "new"), (0, "old"), (2, "old")] and len(kept) == 3
    assert mo.in_roi([0, 200, 100, 300], [0, 240, 1280, 720]) is True and mo.in_roi([0, 0, 100, 100], [0, 240, 1280, 720]) is False


# ---- 4-class support
def test_a_three_class_model_is_scored_on_the_shared_classes_and_a_four_class_model_on_all_four():
    cfg = ["helmet", "motorcycle", "no-helmet", "plate"]
    assert eo.common_classes({0: "helmet", 1: "motorcycle", 2: "no-helmet"}, cfg) == ["helmet", "motorcycle", "no-helmet"]
    assert eo.common_classes({0: "helmet", 1: "motorcycle", 2: "no-helmet", 3: "plate"}, cfg) == cfg
    assert eo.common_classes({0: "0"}, cfg) == []                                                    # a model with unknown names scores nothing, silently nothing wrong


def test_plate_boxes_are_scored_like_any_other_class_and_never_count_for_models_without_the_class():
    cfg = ["helmet", "motorcycle", "no-helmet", "plate"]
    g = {"img1": [("helmet", A), ("plate", C)]}
    four = eo.evaluate(g, {"img1": [("helmet", 0.9, A), ("plate", 0.8, C)]}, ROI, cfg)
    assert four["plate"]["n_gt"] == 1 and four["plate"]["ap50"] == CEIL and four["all"]["classes_with_gt"] == 2
    three = eo.evaluate(g, {"img1": [("helmet", 0.9, A)]}, ROI, eo.common_classes(["helmet", "motorcycle", "no-helmet"], cfg))
    assert "plate" not in three and three["all"]["classes_with_gt"] == 1 and three["all"]["ap50"] == CEIL        # the missing plate class does not drag the 3-class model down


def test_labels_with_plate_are_read_when_the_class_is_wanted_and_dropped_when_not(tmp_path):
    lab = tmp_path / "labels"
    lab.mkdir()
    (tmp_path / "data.yaml").write_text("names: ['helmet', 'motorcycle', 'no-helmet', 'plate']\n", encoding="utf-8")
    (lab / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n3 0.3 0.3 0.1 0.1\n", encoding="utf-8")
    four, _ = eo.read_labels(tmp_path, {"a": (100, 100)}, ["helmet", "motorcycle", "no-helmet", "plate"])
    three, _ = eo.read_labels(tmp_path, {"a": (100, 100)}, ["helmet", "motorcycle", "no-helmet"])
    assert [k for k, _ in four["a"]] == ["helmet", "plate"] and [k for k, _ in three["a"]] == ["helmet"]


def test_four_class_prelabels_take_plates_only_from_the_four_class_model_and_keep_roi_boxes_only():
    roi = [0, 240, 1280, 720]
    dets = [(0, 0.9, [10, 300, 60, 380], "old"), (0, 0.8, [12, 302, 62, 382], "exp016"),         # same helmet from two models: NMS keeps the 0.9 one
            (3, 0.7, [100, 500, 140, 520], "exp016"), (3, 0.9, [300, 500, 340, 520], "old"),          # a (hypothetical) plate from another model is ignored
            (3, 0.6, [100, 100, 140, 120], "exp016"),                                                  # plate outside the ROI
            (2, 0.5, [500, 400, 560, 470], "exp015")]
    kept = mo.merge_4cls(dets, "exp016", roi)
    assert sorted((d[0], d[3]) for d in kept) == [(0, "old"), (2, "exp015"), (3, "exp016")]
    assert [d[1] for d in kept if d[0] == 3] == [0.7]
