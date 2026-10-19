"""Draws the PROPOSED stop line, direction of entry and lanes for test_video on a video frame (1280x720) -> lanes_proposal.jpg (gitignored: people are visible).
The coordinates below are a PROPOSAL read off the road markings; they are NOT in configs/cameras/test_video.yaml until the author approves them.
Usage: python experiments/EXP-022-light-roi/draw_lanes_proposal.py [frame_index]"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
STOP_LINE = [[400, 302], [1040, 277]]            # ~12 px below the near edge of the zebra crossing (queue fronts stay below it)
DIRECTION = [[650, 430], [665, 200]]            # entry direction: up the image, towards the zebra crossing
LANES = [
    ("left-turn", [[400, 420], [400, 235], [560, 235], [560, 420]], "roi"),
    ("through", [[560, 420], [560, 235], [873, 235], [738, 420]], "roi"),
    ("right-turn", [[873, 235], [1048, 235], [1048, 420], [738, 420]], "exempt"),
]
COL = {"roi": (255, 200, 0), "exempt": (255, 0, 255)}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(str(REPO / "test_video.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, f = cap.read()
    f = cv2.resize(f, (1280, 720))
    over = f.copy()
    for name, poly, sig in LANES:
        cv2.fillPoly(over, [np.array(poly, np.int32)], COL[sig])
    f = cv2.addWeighted(over, 0.25, f, 0.75, 0)
    for name, poly, sig in LANES:
        cv2.polylines(f, [np.array(poly, np.int32)], True, COL[sig], 2)
        x, y = poly[1][0] + 6, poly[1][1] + 16
        cv2.putText(f, f"{name} ({sig})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
        cv2.putText(f, f"{name} ({sig})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL[sig], 1)
    cv2.line(f, tuple(STOP_LINE[0]), tuple(STOP_LINE[1]), (0, 0, 255), 3)
    cv2.arrowedLine(f, tuple(DIRECTION[0]), tuple(DIRECTION[1]), (0, 255, 0), 3, tipLength=0.12)
    for txt, org in (("stop_line", (410, 322)), ("direction (entry)", (600, 460))):
        cv2.putText(f, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
        cv2.putText(f, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if txt == "stop_line" else (0, 160, 0), 1)
    cv2.putText(f, f"PROPOSAL (frame {n}) - not in the config", (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    out = Path(__file__).parent / "lanes_proposal.jpg"
    cv2.imwrite(str(out), f, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(out)


if __name__ == "__main__":
    main()
