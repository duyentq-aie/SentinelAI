"""Anonymise a video before it is shown or published: strong Gaussian blur on every head box (helmet / no-helmet detections) and every licence-plate box; optionally on the upper
half of every COCO `person` box (pedestrians, passengers).

  python tools/anonymize.py --video in.mp4 --out out_blurred.mp4 [--persons] [--plate-fallback]

Detectors (low confidence on purpose: a missed face is worse than an extra blur):
  heads     the helmet model (default config weights), classes helmet / no-helmet, conf 0.10 at inference size 1280 on the whole frame plus conf 0.25 on a 2x2 grid of overlapping tiles, box padded by 25%
  plates    the plate model run on every vehicle crop (helmet-model motorcycles + COCO car/truck/bus), conf 0.15, box padded by 20%
  persons   COCO yolo11s.pt `person`, conf 0.12, upper 40% of the box (--persons)
  fallback  --plate-fallback: also blur the lower-centre part (lower 65%, central 70%) of every vehicle box up to 260 px high (where a rear plate sits) even when the plate detector found nothing
Mode "helmet_demo" (--mode helmet_demo, used for the README demo): the VIEWER must still see whether a rider wears a helmet, so
  plates   every plate box is PIXELATED (blocks of >= 8 px, about 3 blocks across the shorter side): unreadable; a vehicle without a detected plate gets an ESTIMATED plate zone pixelated
  heads    only the lower 45% of every head box (the face) is blurred; the upper part (helmet / hair) is left untouched
  persons  a pedestrian (a COCO person not on a motorcycle) gets the upper 40% of the box blurred; a rider is NOT blurred above the head box: when no head box was found for
           the rider an estimated head box (top 22% of the person box) is used and only its lower 45% is blurred
  no tiled scan of far regions. Annotations (boxes, ids, banner) are drawn AFTER the blur (tools/make_demo.py) so they stay sharp.
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


FACE_FRAC = 0.45                 # helmet_demo: the lower share of a head box that is blurred (the face); the upper 55% (helmet / hair) stays
RIDER_OVERLAP = 0.30             # helmet_demo: a person is a rider when at least this share of its box lies inside a motorcycle box
EST_HEAD_FRAC = 0.22             # helmet_demo: estimated head of a rider without a head box = the top 22% of the person box, central 45% of its width
MIN_BLOCK = 8                    # helmet_demo: smallest pixelation block (px)


def box_of(v):
    """A vehicle is a box (x1, y1, x2, y2) or (kind, box)."""
    return v[1] if len(v) == 2 and isinstance(v[0], str) else v


def kind_of(v):
    return v[0] if len(v) == 2 and isinstance(v[0], str) else "motorcycle"


def face_zone(head_box):
    """The lower FACE_FRAC of a head box (+5% on the sides and below): where the face is. The upper part (helmet) is not in it."""
    x1, y1, x2, y2 = head_box
    w, h = x2 - x1, y2 - y1
    return (x1 - 0.05 * w, y1 + (1 - FACE_FRAC) * h, x2 + 0.05 * w, y2 + 0.05 * h)


def estimated_plate_zone(vehicle):
    """Where a rear plate probably is on a vehicle without a detected plate: motorcycle lower 40% (central 54%), car lower 45% (central 60%), truck / bus lower 70% (central 60%)."""
    kind, (x1, y1, x2, y2) = kind_of(vehicle), box_of(vehicle)
    w, h = x2 - x1, y2 - y1
    top, half = {"motorcycle": (0.60, 0.27), "car": (0.55, 0.30)}.get(kind, (0.30, 0.30))
    cx = (x1 + x2) / 2
    return (cx - half * w, y1 + top * h, cx + half * w, y2)


def pixelate_boxes(img, boxes, min_block=MIN_BLOCK):
    """Pixelate every box IN PLACE: blocks of max(min_block, shorter side / 3) pixels, so a plate is at most about 3 blocks high. Nothing outside a box changes. -> img"""
    for b in boxes:
        c = clip_box(b, img.shape)
        if c is None:
            continue
        x1, y1, x2, y2 = c
        w, h = x2 - x1, y2 - y1
        blk = max(min_block, int(round(min(w, h) / 3)))
        small = cv2.resize(img[y1:y2, x1:x2], (max(1, w // blk), max(1, h // blk)), interpolation=cv2.INTER_AREA)       # floor: every block is at least `blk` px
        img[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return img


def _inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _overlap_share(a, b):
    """Share of box `a` that lies inside box `b`."""
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


def helmet_demo_boxes(heads=(), plates=(), persons=(), vehicles=()):
    """-> (blur_boxes, pixelate_boxes) for the helmet_demo mode (see the module docstring)."""
    blur, pix = [], []
    heads = list(heads)
    motos = [box_of(v) for v in vehicles if kind_of(v) == "motorcycle"]
    for hd in heads:
        blur.append(face_zone(hd))
    for p in persons:
        if any(_overlap_share(p, m) >= RIDER_OVERLAP for m in motos):                                   # a rider: the helmet stays visible
            if not any(_inside(((h[0] + h[2]) / 2, (h[1] + h[3]) / 2), p) for h in heads):              # no head box found for it: estimate one
                w, hgt = p[2] - p[0], p[3] - p[1]
                blur.append(face_zone((p[0] + 0.275 * w, p[1], p[2] - 0.275 * w, p[1] + EST_HEAD_FRAC * hgt)))
        else:                                                                                           # a pedestrian: no helmet to show, the face area is blurred
            blur.append(person_upper(p))
    plates = list(plates)
    for pb in plates:
        pix.append(pad_box(pb, PLATE_PAD))
    for v in vehicles:
        b = box_of(v)
        if not any(_inside(((q[0] + q[2]) / 2, (q[1] + q[3]) / 2), b) for q in plates):                 # no plate found on this vehicle: pixelate where it probably is
            pix.append(estimated_plate_zone(v))
    return blur, pix


def anonymize_helmet_demo(img, heads=(), plates=(), persons=(), vehicles=()):
    """The helmet_demo mode on `img` IN PLACE: Gaussian blur of the faces, pixelation of the plates (done last). -> img"""
    blur, pix = helmet_demo_boxes(heads, plates, persons, vehicles)
    blur_boxes(img, blur)
    return pixelate_boxes(img, pix)


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
        boxes += [plate_zone(box_of(v)) for v in vehicles if box_of(v)[3] - box_of(v)[1] <= FALLBACK_MAX_H]
    return blur_boxes(img, boxes)


class Anonymizer:
    """regions(img) -> {"heads", "plates", "persons", "vehicles"}: lists of boxes. The three detectors are injectable callables (tests use fakes)."""

    def __init__(self, heads, plates, persons=None, vehicles=None, mode="full"):
        self.detect_heads, self.detect_plates, self.detect_persons, self.detect_vehicles = heads, plates, persons, vehicles
        self.mode = mode                                    # "full" (everything blurred) | "helmet_demo" (helmets stay visible, plates pixelated)

    @classmethod
    def from_config(cls, persons=False, cfg=None, device=None, mode="full"):
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
            """full: whole frame at conf 0.10 + a 2x2 grid of tiles at conf 0.25 (tiles see a head twice as large; their low-confidence hits are mostly background, so the bar is higher).
            helmet_demo: whole frame only (no tiled scan of far regions)."""
            if mode == "helmet_demo":
                return [b for n, b in run(helmet, img, 0.10, tiles=False) if n in HEAD_CLASSES]
            return [b for n, b in run(helmet, img, 0.10, tiles=False) + run(helmet, img, 0.25) if n in HEAD_CLASSES]

        def vehicles(img):
            """[(kind, box)]: helmet-model motorcycles and COCO motorcycle / car / truck / bus"""
            return [("motorcycle", b) for n, b in run(helmet, img, 0.15, tiles=False) if n == "motorcycle"] + [(n, b) for n, b in run(coco, img, 0.15, tiles=False) if n in VEHICLES]

        def plates(img):
            out = []
            for _kind, b in vehicles(img):
                x1, y1, x2, y2 = (max(0, int(b[0])), max(0, int(b[1])), min(img.shape[1], int(b[2])), min(img.shape[0], int(b[3])))
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                for _, pb in run(plate, img[y1:y2, x1:x2], 0.10, tiles=False):
                    out.append([x1 + pb[0], y1 + pb[1], x1 + pb[2], y1 + pb[3]])
            return out

        persons_fn = (lambda img: [b for n, b in run(coco, img, 0.12, tiles=False) if n == "person"]) if persons else None
        return cls(heads, plates, persons_fn, vehicles, mode)

    def regions(self, img):
        return {"heads": list(self.detect_heads(img)), "plates": list(self.detect_plates(img)),
                "persons": list(self.detect_persons(img)) if self.detect_persons else [], "vehicles": list(self.detect_vehicles(img)) if self.detect_vehicles else []}

    def apply(self, img, plate_fallback=False, regions=None):
        """Blur `img` in place (regions are computed on `img` itself unless given, e.g. from the clean frame when `img` is an annotated copy of it)."""
        r = regions if regions is not None else self.regions(img)
        if self.mode == "helmet_demo":
            return anonymize_helmet_demo(img, r["heads"], r["plates"], r["persons"], r["vehicles"])
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
    ap.add_argument("--mode", choices=("full", "helmet_demo"), default="full", help="helmet_demo: pixelate plates, blur only the faces (the helmet stays visible), pedestrians blurred, riders not above the head box")
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f"{a.out} exists: nothing is overwritten (choose another --out)")
    n = process_video(a.video, a.out, Anonymizer.from_config(persons=a.persons or a.mode == "helmet_demo", mode=a.mode), a.plate_fallback, size=(1280, 720))
    print(f"{n} frames -> {a.out}")


if __name__ == "__main__":
    main()
