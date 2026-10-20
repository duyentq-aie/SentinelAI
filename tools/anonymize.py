"""Anonymise a video before it is shown or published: strong Gaussian blur on every head box (helmet / no-helmet detections) and every licence-plate box; optionally on the upper
half of every COCO `person` box (pedestrians, passengers).

  python tools/anonymize.py --video in.mp4 --out out_blurred.mp4 [--persons] [--plate-fallback]

Detectors (low confidence on purpose: a missed face is worse than an extra blur):
  heads     the helmet model (default config weights), classes helmet / no-helmet, conf 0.10 at inference size 1280 on the whole frame plus conf 0.25 on a 2x2 grid of overlapping tiles, box padded by 25%
  plates    the plate model run on every vehicle crop (helmet-model motorcycles + COCO car/truck/bus), conf 0.15, box padded by 20%
  persons   COCO yolo11s.pt `person`, conf 0.12, upper 40% of the box (--persons)
  fallback  --plate-fallback: also blur the lower-centre part (lower 65%, central 70%) of every vehicle box up to 260 px high (where a rear plate sits) even when the plate detector found nothing
The tool blurs what the detectors find: it is NOT a guarantee. Look at the result (docs/assets were checked frame by frame) before publishing anything.
The blur functions work on any image (`anonymize_frame`), the models are only needed by `Anonymizer.from_config()`.
"""
import argparse
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

HEAD_CLASSES = ("helmet", "no-helmet")
VEHICLES = ("car", "truck", "bus", "motorcycle")
HEAD_PAD, PLATE_PAD = 0.25, 0.20
STRENGTH = 0.4                   # Gaussian sigma = STRENGTH x the shorter side of the box (>= MIN_SIGMA): the content of a box is no longer recognisable
MIN_SIGMA = 6.0
PERSON_FRAC = 0.4                # upper share of a COCO person box that is blurred (head, neck, shoulders)
FALLBACK_MAX_H = 260             # --plate-fallback only for vehicle boxes up to this height (px): on a big near vehicle the plate detector finds the plate and a blanket zone would hide the scene


def clip_box(box, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(max(0, min(w, x1))), int(max(0, min(h, y1))), int(max(0, min(w, x2))), int(max(0, min(h, y2)))
    return (x1, y1, x2, y2) if x2 - x1 >= 2 and y2 - y1 >= 2 else None


def pad_box(box, frac):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 - frac * w, y1 - frac * h, x2 + frac * w, y2 + frac * h)


def person_upper(box, frac=PERSON_FRAC):
    """The upper `frac` of a person box (head and shoulders)."""
    x1, y1, x2, y2 = box
    return (x1, y1, x2, y1 + frac * (y2 - y1))


def plate_zone(vehicle_box):
    """Where a rear plate can sit on a vehicle box (a truck's plate is at about 40% of the height, a car's / motorcycle's lower): the lower 65% of its height, the central 70% of its width."""
    x1, y1, x2, y2 = vehicle_box
    w, h = x2 - x1, y2 - y1
    return (x1 + 0.15 * w, y1 + 0.35 * h, x2 - 0.15 * w, y2)


def blur_boxes(img, boxes, strength=STRENGTH):
    """Blur every box of `img` IN PLACE (strong Gaussian blur, sigma proportional to the box). Boxes are clipped to the image; nothing outside a box changes. -> img"""
    for b in boxes:
        c = clip_box(b, img.shape)
        if c is None:
            continue
        x1, y1, x2, y2 = c
        sigma = max(MIN_SIGMA, strength * min(x2 - x1, y2 - y1))
        k = int(sigma * 6) | 1
        img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
    return img


def anonymize_frame(img, heads=(), plates=(), persons=(), vehicles=(), plate_fallback=False):
    """Blur the boxes of the given categories (already in image pixels). persons: full person boxes (their upper 40% is blurred); vehicles: used only with plate_fallback. -> img (in place)"""
    boxes = [pad_box(b, HEAD_PAD) for b in heads] + [pad_box(b, PLATE_PAD) for b in plates] + [person_upper(b) for b in persons]
    if plate_fallback:
        boxes += [plate_zone(b) for b in vehicles if b[3] - b[1] <= FALLBACK_MAX_H]
    return blur_boxes(img, boxes)


class Anonymizer:
    """regions(img) -> {"heads", "plates", "persons", "vehicles"}: lists of boxes. The three detectors are injectable callables (tests use fakes)."""

    def __init__(self, heads, plates, persons=None, vehicles=None):
        self.detect_heads, self.detect_plates, self.detect_persons, self.detect_vehicles = heads, plates, persons, vehicles

    @classmethod
    def from_config(cls, persons=False, cfg=None, device=None):
        from ultralytics import YOLO

        from sentinel.config import load_pipeline_config
        cfg = cfg or load_pipeline_config(root=REPO)
        kw = {} if device is None else {"device": device}
        helmet, plate = YOLO(str(cfg.path(cfg.models.helmet.weights))), YOLO(str(cfg.path(cfg.models.plate.weights)))
        coco = YOLO(str(cfg.path(cfg.models.coco.weights)))

        def run_once(model, img, conf, imgsz=1280):
            r = model.predict(img, conf=conf, imgsz=imgsz, verbose=False, **kw)[0]            # 1280: small, far heads are missed at the default 640
            return [(model.names[int(k)], [float(v) for v in b]) for b, k in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.cpu().tolist())]

        def run(model, img, conf, imgsz=1280, tiles=True):
            """The whole frame plus a 2x2 grid of overlapping tiles (each tile is looked at twice as large): far, small heads are found far more often."""
            out = run_once(model, img, conf, imgsz)
            if tiles and img.shape[0] >= 400:
                H, W = img.shape[:2]
                tw, th = int(W * 0.58), int(H * 0.58)
                for oy in (0, H - th):
                    for ox in (0, W - tw):
                        out += [(n, [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]) for n, b in run_once(model, img[oy:oy + th, ox:ox + tw], conf, imgsz)]
            return out

        def heads(img):
            """whole frame at conf 0.10 + a 2x2 grid of tiles at conf 0.25 (tiles see a head twice as large; their low-confidence hits are mostly background, so the bar is higher)"""
            return [b for n, b in run(helmet, img, 0.10, tiles=False) + run(helmet, img, 0.25) if n in HEAD_CLASSES]

        def vehicles(img):
            return [b for n, b in run(helmet, img, 0.15, tiles=False) if n == "motorcycle"] + [b for n, b in run(coco, img, 0.15, tiles=False) if n in VEHICLES]

        def plates(img):
            out = []
            for b in vehicles(img):
                x1, y1, x2, y2 = (max(0, int(b[0])), max(0, int(b[1])), min(img.shape[1], int(b[2])), min(img.shape[0], int(b[3])))
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                for _, pb in run(plate, img[y1:y2, x1:x2], 0.10, tiles=False):
                    out.append([x1 + pb[0], y1 + pb[1], x1 + pb[2], y1 + pb[3]])
            return out

        persons_fn = (lambda img: [b for n, b in run(coco, img, 0.12, tiles=False) if n == "person"]) if persons else None
        return cls(heads, plates, persons_fn, vehicles)

    def regions(self, img):
        return {"heads": list(self.detect_heads(img)), "plates": list(self.detect_plates(img)),
                "persons": list(self.detect_persons(img)) if self.detect_persons else [], "vehicles": list(self.detect_vehicles(img)) if self.detect_vehicles else []}

    def apply(self, img, plate_fallback=False, regions=None):
        """Blur `img` in place (regions are computed on `img` itself unless given, e.g. from the clean frame when `img` is an annotated copy of it)."""
        r = regions if regions is not None else self.regions(img)
        return anonymize_frame(img, r["heads"], r["plates"], r["persons"], r["vehicles"], plate_fallback)


def process_video(src, dst, anonymizer, plate_fallback=False, size=None):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = size or (w, h)
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, size)
        vw.write(anonymizer.apply(frame, plate_fallback))
        n += 1
    vw.release()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--persons", action="store_true", help="also blur the upper half of every COCO person box")
    ap.add_argument("--plate-fallback", action="store_true", help="also blur the lower-centre of every vehicle box")
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f"{a.out} exists: nothing is overwritten (choose another --out)")
    n = process_video(a.video, a.out, Anonymizer.from_config(persons=a.persons), a.plate_fallback, size=(1280, 720))
    print(f"{n} frames -> {a.out}")


if __name__ == "__main__":
    main()
