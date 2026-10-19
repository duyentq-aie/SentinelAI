"""Label violation events in a video by hand -> eval_sets/events/<video_id>.json (schema: docs/EVIDENCE.md, rules: docs/LABELING_GUIDE.md).

  python tools/label_events.py --video clips/ngatu_A_night_01.mp4 --split dev [--camera-config configs/cameras/ngatu_A.yaml]

Keys (video window):
  space play/pause     j / l  step -1 / +1 frame      J / L  step -30 / +30 frames
  s mark event START   e mark event END               left-drag draw the vehicle box, then  k  = key frame (current frame) + that box
  t cycle type (no_helmet, overload, red_light)       h  next event is 'hard'      Enter  commit event (asks plate text in the terminal)
  a next event is AMBIGUOUS (cannot tell whether the violation happened: blurred / occluded / too far; kept for matching, not scored)
  u undo last event    [  open / close an ANNOTATED RANGE (frames you labelled completely: false positives are counted only inside them)
  w save               q  save and quit
Existing events are shown as boxes while the current frame is inside their [start, end].
"""
import argparse
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from sentinel import labeling  # noqa: E402


def hud(img, session, idx, n, fps, playing, hard):
    d = session.draft
    lines = [f"frame {idx}/{n - 1}  t={idx / fps:7.2f}s  {'PLAY' if playing else 'pause'}   type: {session.type}   next hard: {hard}   next ambiguous: {session.ambiguous}",
             f"draft: start={d.get('frame_start')} end={d.get('frame_end')} key={d.get('key_frame')} box={d.get('key_box')}",
             f"events: {len(session.labels['events'])}   ranges: {session.labels['annotated_ranges']}"]
    for i, t in enumerate(lines):
        cv2.putText(img, t, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    for e in session.labels["events"]:
        if e["frame_start"] <= idx <= e["frame_end"]:
            x1, y1, x2, y2 = e["key_box"]
            col = (0, 165, 255) if e.get("ambiguous") else (0, 200, 0)           # orange = ambiguous
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, f"{e['id']} {e['type']}{' AMBIGUOUS' if e.get('ambiguous') else ''}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out")
    ap.add_argument("--split", choices=labeling.SPLITS, default="dev")
    ap.add_argument("--camera-config")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {a.video}")
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_id = Path(a.video).stem
    out = Path(a.out) if a.out else REPO / "eval_sets" / "events" / f"{video_id}.json"
    if out.exists():
        labels = labeling.load(out)
        print(f"loaded {len(labels['events'])} events from {out}")
    else:
        labels = labeling.new_labels(video_id, fps, w, h, a.split, labeling.video_sha256(a.video), a.camera_config)
    session = labeling.LabelSession(labels)

    state = {"idx": 0, "playing": False, "hard": False, "drag": None, "box": None}

    def mouse(ev, x, y, fl, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["drag"], state["box"] = (x, y), None
        elif ev == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state["box"] = (*state["drag"], x, y)
        elif ev == cv2.EVENT_LBUTTONUP and state["drag"]:
            state["box"], state["drag"] = (*state["drag"], x, y), None

    def read(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        return cap.read()

    win = "label_events"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse)
    ok, frame = read(0)
    while True:
        if state["playing"]:
            state["idx"] = min(n - 1, state["idx"] + 1)
            ok, frame = read(state["idx"])
        if ok:
            img = hud(frame.copy(), session, state["idx"], n, fps, state["playing"], state["hard"])
            if state["box"]:
                x1, y1, x2, y2 = state["box"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.imshow(win, img)
        k = cv2.waitKey(int(1000 / fps) if state["playing"] else 30) & 0xFF
        step = {ord("j"): -1, ord("l"): 1, ord("J"): -30, ord("L"): 30}.get(k)
        if step is not None:
            state["idx"] = max(0, min(n - 1, state["idx"] + step))
            ok, frame = read(state["idx"])
        elif k == ord(" "):
            state["playing"] = not state["playing"]
        elif k == ord("s"):
            session.mark_start(state["idx"])
        elif k == ord("e"):
            session.mark_end(state["idx"])
        elif k == ord("k") and state["box"]:
            session.set_key(state["idx"], state["box"])
        elif k == ord("t"):
            session.cycle_type()
        elif k == ord("h"):
            state["hard"] = not state["hard"]
        elif k == ord("a"):
            print("next event ambiguous:", session.toggle_ambiguous())
        elif k == ord("["):
            r = session.toggle_range(state["idx"])
            print("annotated range", r if r else "opened at frame %d" % state["idx"])
        elif k == ord("u"):
            print("undone", session.undo())
        elif k in (13, 10):
            if not session.ready():
                print("set start (s), end (e) and key frame + box (drag, then k) first")
                continue
            plate = input("plate text (blank = unreadable): ")
            try:
                print("added", session.commit(plate, "hard" if state["hard"] else "normal", input("notes: ")))
                state["hard"], state["box"] = False, None
            except ValueError as e:
                print("NOT added:", e)
        elif k == ord("w"):
            labeling.save(out, labels)
            print("saved", out)
        elif k == ord("q"):
            labeling.save(out, labels)
            print("saved", out)
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
