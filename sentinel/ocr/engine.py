"""Plate reader: PaddleOCR text recognition (recognition only, no text detection: the plate detector already cropped the plate)."""
import time

from sentinel.ocr import normalize, preprocess, quality, vote


class OcrUnavailable(RuntimeError):
    """PaddleOCR or its model could not be loaded (not installed, no model files, no network for the first download)."""


class PaddleRecognizer:
    """Callable: BGR image of ONE text line -> (text, score). Loads lazily so that importing the pipeline stays cheap."""

    def __init__(self, model_name="PP-OCRv5_mobile_rec"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from paddleocr import TextRecognition
                self._model = TextRecognition(model_name=self.model_name)
            except Exception as e:                                           # import error, missing model, offline, ...
                raise OcrUnavailable(f"cannot load PaddleOCR text recognition '{self.model_name}': {e}") from e
        return self._model

    def __call__(self, line_img):
        res = self._load().predict(input=line_img, batch_size=1)
        r = res[0] if isinstance(res, (list, tuple)) else next(iter(res))
        return str(r["rec_text"]), float(r["rec_score"])


class PlateOCR:
    def __init__(self, recognizer, reads=3, model_name="PP-OCRv5_mobile_rec"):
        self.recognizer, self.reads, self.model_name = recognizer, reads, model_name

    def read_one(self, crop):
        """One crop -> {"raw", "text", "format_ok", "repairs", "conf"}; two-line plates are read line by line and concatenated."""
        parts = [self.recognizer(line) for line in preprocess.prepare(crop)]
        raw = "".join(t for t, _ in parts)
        conf = sum(s for _, s in parts) / len(parts) if parts else 0.0
        return {**normalize.normalize(raw), "raw": raw, "conf": round(conf, 4)}

    def read_track(self, crops):
        """crops: plate crops of ONE vehicle (any order). Reads the `reads` best by quality and votes. -> plate_pred dict (always)."""
        t0 = time.perf_counter()
        order = quality.best(crops, self.reads)
        results = [self.read_one(crops[i]) for i in order]
        voted = vote.vote([(r["text"], r["conf"]) for r in results if r["format_ok"]])
        pred = {"text": voted["text"] if voted else None, "format_ok": bool(voted),
                "score": round(voted["score"] * (sum(r["conf"] for r in results if r["format_ok"]) / max(1, sum(r["format_ok"] for r in results))), 4)
                if voted else None,
                "raw": [r["raw"] for r in results], "reads": len(results), "valid_reads": voted["n_reads"] if voted else 0,
                "unanimous": voted["unanimous"] if voted else None,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "model": self.model_name}
        return pred
