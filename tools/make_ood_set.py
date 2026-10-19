"""Build the out-of-domain detector evaluation set from test_video.mp4 (EXP-018): frames, ROI preview, pre-labels from BOTH helmet models, upload zip.

  python tools/make_ood_set.py frames  --video data/raw/field/test_video.mp4 --out data/ood_test_video [--n 15]
  python tools/make_ood_set.py prelabel --out data/ood_test_video --config configs/eval/ood_test_video.yaml [--conf 0.15]
  python tools/make_ood_set.py preview --out data/ood_test_video --config configs/eval/ood_test_video.yaml
  python tools/make_ood_set.py prelabel4 --out data/ood_test_video --config configs/eval/ood_test_video.yaml [--conf 0.15]

frames   : n frames evenly spaced over the whole clip (centre of n equal segments, so neither the first nor the last frame) -> <out>/images/*.png (lossless)
prelabel : runs helmet_best.pt AND the EXP-015 best.pt at conf 0.15, merges them with class-wise NMS (IoU 0.5, highest confidence wins), keeps only boxes whose centre
           lies in the evaluation ROI, writes YOLO txt (3 classes) -> <out>/prelabels/ and a zip <out>/ood_prelabels.zip (images/ labels/ data.yaml) for Roboflow.
           The pre-labels are SUGGESTIONS only: every box has to be checked and corrected by hand (see experiments/EXP-018-ood-eval/README.md).
prelabel4: 4-class suggestions (helmet, motorcycle, no-helmet from the THREE helmet models merged by class-wise NMS; plate ONLY from the 4-class model, conf 0.15), ROI only
           -> <out>/prelabels_4cls/ and <out>/ood_prelabels_4cls.zip (the 3-class zip and folder are left untouched).
preview  : draws the ROI on the first frame -> <out>/roi_preview.jpg
Everything under data/ is gitignored (frames show people and plates).
"""
import argparse
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
CLASSES = ["helmet", "motorcycle", "no-helmet"]
CLASSES4 = ["helmet", "motorcycle", "no-helmet", "plate"]


def frame_indices(n_frames, n):
    """Centres of n equal segments: 15 frames of a 787-frame clip -> ~52 frames apart (1.75 s at 30 fps)."""
    return [min(n_frames - 1, int((i + 0.5) * n_frames / n)) for i in range(n)]


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def in_roi(box, roi):
    """Centre of an [x1, y1, x2, y2] box inside roi = [x1, y1, x2, y2] (same pixel space)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]


def iou(a, b):
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def merge_nms(dets, iou_thr=0.5):
    """dets: [(cls, conf, [x1, y1, x2, y2], source)] from several models -> class-wise greedy NMS, highest confidence kept."""
    keep = []
    for d in sorted(dets, key=lambda d: -d[1]):
        if all(k[0] != d[0] or iou(k[2], d[2]) < iou_thr for k in keep):
            keep.append(d)
    return keep


def yolo_line(cls, box, w, h):
    x1, y1, x2, y2 = box
    return f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def cmd_frames(a):
    out = Path(a.out)
    img_dir = out / "images"
    if img_dir.exists() and any(img_dir.iterdir()):
        sys.exit(f"{img_dir} is not empty: nothing is overwritten or deleted")
    img_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(a.video))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    for idx in frame_indices(n, a.n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            sys.exit(f"cannot read frame {idx}")
        cv2.imwrite(str(img_dir / f"ood_f{idx:04d}_t{idx / fps:05.2f}s.png"), fr)
    print(f"{a.n} frames -> {img_dir}")


def cmd_preview(a):
    cfg = load_config(a.config)
    roi = cfg["roi"]
    img = cv2.imread(str(sorted((Path(a.out) / "images").glob("*.png"))[0]))
    shade = img.copy()
    cv2.rectangle(shade, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.rectangle(shade, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 255), -1)
    vis = np.where(shade == 0, (img * 0.35).astype(np.uint8), img)                 # outside the ROI darkened
    cv2.rectangle(vis, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 255), 2)
    cv2.putText(vis, f"evaluation ROI {roi} (box centre must lie inside)", (10, 24), 0, 0.6, (0, 255, 255), 2)
    p = Path(a.out) / "roi_preview.jpg"
    cv2.imwrite(str(p), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("->", p)


def predict_both(images, models, conf, imgsz=640):
    """-> {image name: [(class index in CLASSES4, conf, box, model label)]}; class ids come from each model's own names, so 3- and 4-class models mix safely."""
    from ultralytics import YOLO
    res = {}
    for name, path in models.items():
        m = YOLO(str(REPO / path))
        for im in images:
            r = m.predict(str(im), imgsz=imgsz, conf=conf, iou=0.7, verbose=False)[0]
            for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                cname = m.names[int(c)]
                if cname in CLASSES4:
                    res.setdefault(im.name, []).append((CLASSES4.index(cname), float(s), b, name))
    return res


def merge_4cls(dets, plate_source, roi, nms_iou=0.5):
    """dets: [(class index in CLASSES4, conf, box, model label)] of one image. Helmet / motorcycle / no-helmet: every model, class-wise NMS (highest confidence kept).
    Plate (index 3): ONLY detections of `plate_source`. Only boxes whose centre lies in `roi` are kept."""
    heads = [d for d in dets if d[0] != 3]
    plates = [d for d in dets if d[0] == 3 and d[3] == plate_source]
    return [d for d in merge_nms(heads, nms_iou) + merge_nms(plates, nms_iou) if in_roi(d[2], roi)]


def cmd_prelabel(a):
    cfg = load_config(a.config)
    out = Path(a.out)
    images = sorted((out / "images").glob("*.png"))
    lab_dir = out / "prelabels"
    if lab_dir.exists() and any(lab_dir.iterdir()):
        sys.exit(f"{lab_dir} is not empty: nothing is overwritten or deleted")
    lab_dir.mkdir(parents=True, exist_ok=True)
    dets = predict_both(images, {k: v for k, v in cfg["models"].items() if k in cfg.get("prelabel_models", cfg["models"])}, a.conf)
    stats = {"images": len(images), "boxes_per_class": {c: 0 for c in CLASSES}, "by_source_before_merge": {}}
    for im in images:
        h, w = cv2.imread(str(im)).shape[:2]
        for d in dets.get(im.name, []):
            stats["by_source_before_merge"][d[3]] = stats["by_source_before_merge"].get(d[3], 0) + 1
        merged = [d for d in merge_nms([d for d in dets.get(im.name, []) if d[0] < 3], cfg["nms_iou"]) if in_roi(d[2], cfg["roi"])]
        lines = [yolo_line(d[0], d[2], w, h) for d in sorted(merged, key=lambda d: (d[2][1], d[2][0]))]
        for d in merged:
            stats["boxes_per_class"][CLASSES[d[0]]] += 1
        (lab_dir / f"{im.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (out / "data.yaml").write_text("nc: 3\nnames: " + str(CLASSES).replace("'", '"') + "\n", encoding="utf-8")
    zp = out / "ood_prelabels.zip"
    if zp.exists():
        sys.exit(f"{zp} exists: nothing is overwritten or deleted")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for im in images:
            z.write(im, f"images/{im.name}")
            z.write(lab_dir / f"{im.stem}.txt", f"labels/{im.stem}.txt")
        z.write(out / "data.yaml", "data.yaml")
    print(stats, "->", zp)


def cmd_prelabel4(a):
    cfg = load_config(a.config)
    out = Path(a.out)
    images = sorted((out / "images").glob("*.png"))
    lab_dir = out / "prelabels_4cls"
    zp = out / "ood_prelabels_4cls.zip"
    if (lab_dir.exists() and any(lab_dir.iterdir())) or zp.exists():
        sys.exit(f"{lab_dir} / {zp} already exist: nothing is overwritten or deleted")
    lab_dir.mkdir(parents=True, exist_ok=True)
    models = {k: v for k, v in cfg["models"].items() if k in cfg["prelabel4_models"]}
    dets = predict_both(images, {k: (v["weights"] if isinstance(v, dict) else v) for k, v in models.items()}, a.conf)
    stats = {"images": len(images), "boxes_per_class": {c: 0 for c in CLASSES4}, "by_source_before_merge": {}}
    for im in images:
        h, w = cv2.imread(str(im)).shape[:2]
        for d in dets.get(im.name, []):
            key = f"{d[3]}:{CLASSES4[d[0]]}"
            stats["by_source_before_merge"][key] = stats["by_source_before_merge"].get(key, 0) + 1
        merged = merge_4cls(dets.get(im.name, []), cfg["prelabel4_plate_model"], cfg["roi"], cfg["nms_iou"])
        lines = [yolo_line(d[0], d[2], w, h) for d in sorted(merged, key=lambda d: (d[2][1], d[2][0]))]
        for d in merged:
            stats["boxes_per_class"][CLASSES4[d[0]]] += 1
        (lab_dir / f"{im.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (out / "data_4cls.yaml").write_text("nc: 4\nnames: " + str(CLASSES4).replace("'", '"') + "\n", encoding="utf-8")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for im in images:
            z.write(im, f"images/{im.name}")
            z.write(lab_dir / f"{im.stem}.txt", f"labels/{im.stem}.txt")
        z.write(out / "data_4cls.yaml", "data.yaml")
    print(stats, "->", zp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames")
    f.add_argument("--video", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--n", type=int, default=15)
    for name in ("prelabel", "prelabel4", "preview"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--config", required=True)
        if name in ("prelabel", "prelabel4"):
            p.add_argument("--conf", type=float, default=0.15)
    a = ap.parse_args()
    {"frames": cmd_frames, "prelabel": cmd_prelabel, "prelabel4": cmd_prelabel4, "preview": cmd_preview}[a.cmd](a)


if __name__ == "__main__":
    main()
