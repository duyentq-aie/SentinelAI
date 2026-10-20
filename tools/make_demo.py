"""Make the demo assets of the README from test_video.mp4, ANONYMISED (tools/anonymize.py): an annotated GIF and a blurred copy of the evidence folder for the /review screenshot.

  python tools/make_demo.py --out outputs/demo [--start 9 --seconds 13 --width 960 --max-mb 8]

The current DEFAULT pipeline runs over the whole video (so the trackers are warm); the frames of [start, start + seconds) become the GIF. The detections used for the blur come from the CLEAN frame
(lower confidence than the pipeline: heads 0.15, plates on vehicle crops 0.15, upper half of every COCO person, lower-centre of every vehicle box) and are applied to the annotated frame.
Banner = "PENDING REVIEW": events are candidates until a person confirms them (on this clip every reported event was a false alarm, EXP-020).
Writes <out>/demo.gif, <out>/frames_checked/ (the frames that are listed for the visual check), <out>/review_evidence/ (events.jsonl + blurred frame.jpg / vehicle.jpg, NO plate crops).
Nothing here is committed except what you copy into docs/assets/ after looking at it.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from anonymize import Anonymizer  # noqa: E402


def banner(img, open_events, t):
    h, w = img.shape[:2]
    cv2.rectangle(img, (w - 270, h - 34 if not open_events else h - 80), (w, h - 46 if open_events else h), (0, 0, 0), -1)
    cv2.putText(img, f"SentinelAI demo  t={t:5.2f}s", (w - 262, (h - 56) if open_events else (h - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if open_events:
        cv2.rectangle(img, (0, h - 46), (w, h), (0, 120, 220), -1)
        text = "PENDING REVIEW: " + "  |  ".join(f"{typ.upper()} {tid}" for tid, typ in open_events)
        cv2.putText(img, text[:105], (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def to_gif(frames, path, fps, max_bytes):
    """frames: list of BGR images. One shared palette (compresses far better than a palette per frame); reduces colours / frame rate until the file fits."""
    pick = [frames[i] for i in np.linspace(0, len(frames) - 1, min(10, len(frames))).astype(int)]
    montage = Image.fromarray(cv2.cvtColor(np.vstack(pick), cv2.COLOR_BGR2RGB))
    for step, colors in ((1, 128), (1, 96), (1, 64), (1, 48), (2, 96), (2, 64)):
        pal = montage.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        sel = frames[::step]
        pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(palette=pal, dither=Image.Dither.NONE) for f in sel]
        pil[0].save(path, save_all=True, append_images=pil[1:], duration=int(1000 * step / fps), loop=0, optimize=True, disposal=1)
        if Path(path).stat().st_size <= max_bytes:
            return {"frames": len(sel), "colors": colors, "fps": round(fps / step, 2), "bytes": Path(path).stat().st_size}
    raise SystemExit(f"{path} does not fit in {max_bytes} bytes even at the smallest setting")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(REPO / "test_video.mp4"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, default=9.0)
    ap.add_argument("--seconds", type=float, default=13.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--gif-fps", type=float, default=7.5)
    ap.add_argument("--max-mb", type=float, default=8.0)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty: nothing is overwritten (choose another --out)")
    out.mkdir(parents=True, exist_ok=True)

    from sentinel.characterize import read_frames
    from sentinel.config import load_camera_config, load_pipeline_config
    from sentinel.pipeline import Detectors, Pipeline, SharedState

    cfg = load_pipeline_config(root=REPO)
    cfg.evidence.dir = str(out / "evidence")
    pipe = Pipeline(Detectors(cfg), SharedState(), cfg, load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml"))
    anon = Anonymizer.from_config(persons=True, cfg=cfg)
    fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS) or 30.0
    f0, f1 = int(a.start * fps), int((a.start + a.seconds) * fps)
    step = max(1, int(round(fps / a.gif_fps)))
    frames, kept = [], []
    for i, frame in enumerate(read_frames(a.video)):
        r = pipe.process(frame)
        if f0 <= i < f1 and (i - f0) % step == 0:
            disp = r.frame
            disp[60:105, 40:230] = r.raw[60:105, 40:230]                                   # the pipeline's absolute 'FPS: n' text is not shown
            anon.apply(disp, plate_fallback=True, regions=anon.regions(r.raw))
            banner(disp, [(e.track_id, e.type) for e in pipe.recorder._open.values()], i / fps)
            h = int(round(disp.shape[0] * a.width / disp.shape[1]))
            frames.append(cv2.resize(disp, (a.width, h), interpolation=cv2.INTER_AREA))
            kept.append(i)
    pipe.flush()
    info = to_gif(frames, out / "demo.gif", fps / step, int(a.max_mb * 1024 * 1024))
    print("gif:", info, "frames", kept[0], "..", kept[-1])
    (out / "demo_frames.json").write_text(json.dumps({"video_frames": kept, **info}), encoding="utf-8")

    # blurred evidence for the /review screenshot: whole frame blurred, the vehicle crop cut from the blurred frame, no plate crop
    ev_src, ev_dst = out / "evidence", out / "review_evidence"
    ev_dst.mkdir()
    last = {}
    for line in (ev_src / "events.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        last[rec["id"]] = rec
    keep = []
    for eid, rec in last.items():
        fr = cv2.imread(str(ev_src / eid / "frame.jpg"))
        if fr is None:
            continue
        anon.apply(fr, plate_fallback=True)
        (ev_dst / eid).mkdir()
        cv2.imwrite(str(ev_dst / eid / "frame.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        x1, y1, x2, y2 = rec["key_box"]
        crop = fr[max(0, y1):y2, max(0, x1):x2]
        if crop.size:
            cv2.imwrite(str(ev_dst / eid / "vehicle.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rec = dict(rec, plate_pred=None, plate={"found": False, "attempts": 0}, evidence={"frame": f"{eid}/frame.jpg", "vehicle": f"{eid}/vehicle.jpg", "plate": None})
        keep.append(rec)
    (ev_dst / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
    print(f"{len(keep)} events -> {ev_dst}")


if __name__ == "__main__":
    main()
