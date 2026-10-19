"""EXP-023, what-if (nothing in the pipeline changes): feed the stop-line rule with the COCO model's motorcycle tracks (conf 0.35, same tracker yaml) instead of the helmet model's,
on test_video with the approved camera geometry and the ROI light timeline of EXP-022. Output: every stop-line crossing (reported or not) and how far each track got. For diagnosis only:
there are no labels, so nothing here says the crossings are right.
Usage: python experiments/EXP-023-track-death/coco_source_check.py"""
import csv
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from sentinel.characterize import read_frames  # noqa: E402
from sentinel.config import RedLightRule, load_camera_config, load_pipeline_config  # noqa: E402
from sentinel.pipeline import Detectors  # noqa: E402
from sentinel.rules.legacy import Vehicle  # noqa: E402
from sentinel.rules.stopline import StopLineCrossing  # noqa: E402

OUT = Path(__file__).parent
light = [r["roi_state"] for r in csv.DictReader(open(REPO / "experiments" / "EXP-022-light-roi" / "light_timeline.csv", encoding="utf-8"))]
cfg = load_pipeline_config(root=REPO)
cam = load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
det = Detectors(cfg)
rule = StopLineCrossing(cam, RedLightRule(), log=lambda m: None)
best = {}
for i, frame in enumerate(read_frames(str(REPO / "test_video.mp4"))):
    f = cv2.resize(frame, tuple(cfg.frame_size))
    vs = [Vehicle(f"M_{t.id}", t.box, "motorcycle") for t in det.track_coco(f) if t.cls == "motorcycle"]
    rule.update(light[i], vs)
    for v in vs:
        s = rule.signed_distance(((v.coords[0] + v.coords[2]) / 2, v.coords[3]))
        best[v.id] = max(best.get(v.id, -1e9), s)
cross = [dict(c, time_s=round(c["frame"] / 30, 2)) for c in rule.crossings]
res = {"tracks": len(best), "crossings": len(cross), "reported": sum(c["violation"] for c in cross), "unassigned": rule.unassigned_crossings,
       "by_light": {k: sum(1 for c in cross if c["light"] == k) for k in sorted({c["light"] for c in cross})},
       "by_lane": {str(k): sum(1 for c in cross if c["lane"] == k) for k in sorted({c["lane"] for c in cross}, key=str)},
       "tracks_beyond_line_at_some_frame": sum(1 for s in best.values() if s >= 4), "crossing_list": cross}
(OUT / "coco_source_crossings.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "crossing_list"}, indent=1))
for c in cross:
    print(c["time_s"], c["id"], c["lane"], c["light"], c["violation"], "|", c["why"])
