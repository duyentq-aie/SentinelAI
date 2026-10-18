"""How small can a plate be before the OCR chain stops working? (synthetic plate, NOT a measure of accuracy on real footage)

  python tools/ocr_resolution_probe.py [--text "29B1" "12345"] [--out experiments/EXP-008-ocr/resolution_probe.json]

Renders a two-line motorcycle plate with OpenCV, shrinks it to several heights (the real plate boxes from the 720p test video are ~26x22 px),
and degrades it (blur, JPEG) the way real footage is,
runs the full PlateOCR chain (preprocess -> PaddleOCR text recognition -> normalise -> vote) and reports what comes out.
Needs paddleocr; the first run downloads the recognition model.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from sentinel.ocr.engine import OcrUnavailable, PaddleRecognizer, PlateOCR  # noqa: E402


def render(top, bottom, w=400, h=300):
    img = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(img, (4, 4), (w - 5, h - 5), (30, 30, 30), 3)
    font = cv2.FONT_HERSHEY_DUPLEX
    for text, y, scale in ((top, int(h * 0.38), 2.4), (bottom, int(h * 0.82), 2.8)):
        (tw, th), _ = cv2.getTextSize(text, font, scale, 5)
        cv2.putText(img, text, ((w - tw) // 2, y), font, scale, (20, 20, 20), 5, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", nargs=2, default=["29B1", "12345"], metavar=("TOP", "BOTTOM"))
    ap.add_argument("--heights", type=int, nargs="+", default=[100, 50, 38, 25])
    ap.add_argument("--out")
    a = ap.parse_args()
    ocr = PlateOCR(PaddleRecognizer(), reads=1)
    truth = "".join(a.text)
    rows = []
    base = render(*a.text)

    def jpeg(img, q):
        return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)

    degradations = {"clean": lambda im: im,
                    "blur 0.8": lambda im: cv2.GaussianBlur(im, (0, 0), 0.8),
                    "blur 1.2": lambda im: cv2.GaussianBlur(im, (0, 0), 1.2),
                    "jpeg 25": lambda im: jpeg(im, 25),
                    "blur 0.8 + jpeg 25": lambda im: jpeg(cv2.GaussianBlur(im, (0, 0), 0.8), 25)}
    for h in a.heights:
        small0 = cv2.resize(base, (round(base.shape[1] * h / base.shape[0]), h), interpolation=cv2.INTER_AREA)
        for dname, deg in degradations.items():
            small = deg(small0)
            try:
                pred = ocr.read_track([small])
            except OcrUnavailable as e:
                sys.exit(f"OCR unavailable: {e}")
            rows.append({"height_px": h, "width_px": small.shape[1], "degradation": dname, "truth": truth, "text": pred["text"], "raw": pred["raw"][0],
                         "format_ok": pred["format_ok"], "exact": pred["text"] == truth, "latency_ms": pred["latency_ms"]})
            print(f"{h:4d}x{small.shape[1]:<4d} {dname:20s} raw={pred['raw'][0]!r:22} text={pred['text']!r:14} exact={pred['text'] == truth}  {pred['latency_ms']} ms")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"synthetic_plate": list(a.text), "note": "synthetic rendering, one font; not an accuracy measurement", "rows": rows},
                                          indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
