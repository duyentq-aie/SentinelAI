"""Click to draw the stop line, travel direction, traffic-light ROI and red-light zone of a camera -> configs/cameras/<id>.yaml.

  python tools/camera_config.py --video test_video.mp4 --camera-id test_video            # new file
  python tools/camera_config.py --video test_video.mp4 --from configs/cameras/test_video.yaml --out configs/cameras/test_video.yaml

Coordinates are in the RESIZED frame (1280x720), the same space the pipeline works in. Keys:
  1 stop line (2 clicks)   2 direction arrow (2 clicks: tail -> head = legal travel direction across the line)
  3 traffic-light ROI (2 opposite corners)   4 red-light zone (polygon: click every corner)   5 wrong-lane zone (optional polygon)
  u undo last point   c clear this shape   s save   q quit (asks nothing; press s first)
  [ ] previous / next frame (to pick a frame where the line and the light are clearly visible)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from sentinel.camera_draft import MODES, CameraDraft  # noqa: E402

COLORS = {"stop_line": (0, 0, 255), "direction": (255, 200, 0), "light_roi": (0, 255, 255), "violation_zone": (0, 255, 0), "wrong_lane_zone": (0, 165, 255)}
KEYS = {ord("1"): "stop_line", ord("2"): "direction", ord("3"): "light_roi", ord("4"): "violation_zone", ord("5"): "wrong_lane_zone"}


def draw_overlay(img, draft):
    """Draw every shape of the draft on a copy of img (pure function, also used by the tests)."""
    out = img.copy()
    for m in MODES:
        pts = [tuple(p) for p in draft.points[m]]
        c = COLORS[m]
        for p in pts:
            cv2.circle(out, p, 4, c, -1)
        if m in ("violation_zone", "wrong_lane_zone") and len(pts) >= 2:
            cv2.polylines(out, [np.array(pts, np.int32)], len(pts) >= 3, c, 2)
        elif m == "light_roi" and len(pts) == 2:
            cv2.rectangle(out, pts[0], pts[1], c, 2)
        elif m == "stop_line" and len(pts) == 2:
            cv2.line(out, pts[0], pts[1], c, 3)
        elif m == "direction" and len(pts) == 2:
            cv2.arrowedLine(out, pts[0], pts[1], c, 3, tipLength=0.15)
    cv2.putText(out, f"mode: {draft.mode}  (1 stop line, 2 direction, 3 light ROI, 4 zone, 5 wrong lane | u undo, c clear, s save, q quit)",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--camera-id")
    ap.add_argument("--out")
    ap.add_argument("--from", dest="from_file", help="start from an existing camera yaml")
    ap.add_argument("--frame", type=int, default=0)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {a.video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    draft = CameraDraft.from_file(a.from_file) if a.from_file else CameraDraft(a.camera_id or Path(a.video).stem, Path(a.video).name)
    out = Path(a.out) if a.out else REPO / "configs" / "cameras" / f"{draft.camera_id}.yaml"
    idx = max(0, min(a.frame, n - 1))

    def grab(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        return cv2.resize(f, (draft.w, draft.h)) if ok else None

    frame = grab(idx)
    win = "camera_config"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda ev, x, y, fl, _: draft.click(x, y) if ev == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        cv2.imshow(win, draw_overlay(frame, draft))
        k = cv2.waitKey(30) & 0xFF
        if k in KEYS:
            draft.set_mode(KEYS[k])
        elif k == ord("u"):
            draft.undo()
        elif k == ord("c"):
            draft.clear()
        elif k in (ord("["), ord("]")):
            idx = max(0, min(n - 1, idx + (30 if k == ord("]") else -30)))
            frame = grab(idx) if grab(idx) is not None else frame
        elif k == ord("s"):
            try:
                draft.save(out)
                print(f"saved {out}")
            except ValueError as e:
                print(f"NOT saved: {e}")
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
