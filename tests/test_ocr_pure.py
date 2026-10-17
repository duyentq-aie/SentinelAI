"""OCR building blocks without PaddleOCR: quality, preprocessing, normalisation, voting, PlateOCR with a fake recogniser."""
import cv2
import numpy as np
import pytest

from sentinel.ocr import normalize, preprocess, quality, vote
from sentinel.ocr.engine import PlateOCR


def textured(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


# ---- quality
def test_bigger_and_sharper_crops_score_higher():
    sharp, blurred = textured(40, 80), cv2.GaussianBlur(textured(40, 80), (15, 15), 5)
    assert quality.quality(sharp) > quality.quality(blurred)
    assert quality.quality(textured(60, 120)) > quality.quality(textured(20, 40))
    assert quality.best([blurred, sharp, textured(10, 20)], k=1) == [1]
    assert quality.best([blurred, sharp], k=5) == [1, 0]


def test_quality_is_bounded():
    assert quality.quality(textured(800, 1600)) <= 0.5 * 4 + 0.5 * 3 + 1e-9


# ---- preprocessing
def test_upscale_only_small_crops_and_caps_the_factor():
    H, S = preprocess.MIN_HEIGHT, preprocess.MAX_SCALE
    assert preprocess.upscale(textured(H + 10, 200)).shape[:2] == (H + 10, 200)           # already tall enough: untouched
    assert preprocess.upscale(textured(H // 2, 64)).shape[:2] == (H, 128)                 # x2
    h = int(H / S) // 2                                                                   # needs more than the cap
    assert preprocess.upscale(textured(h, 20)).shape[:2] == (round(h * S), round(20 * S))  # capped at MAX_SCALE x


def test_two_line_detection_and_split_at_the_gap():
    plate = np.full((80, 100, 3), 235, np.uint8)
    cv2.putText(plate, "29B1", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)           # top line
    cv2.putText(plate, "12345", (5, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)          # bottom line, white gap between
    assert preprocess.is_two_line(plate) and not preprocess.is_two_line(np.zeros((30, 120, 3), np.uint8))
    top, bottom = preprocess.split_two_lines(plate)
    assert top.shape[0] + bottom.shape[0] == 80 and 30 <= top.shape[0] <= 52
    assert len(preprocess.prepare(plate)) == 2 and len(preprocess.prepare(np.full((30, 120, 3), 200, np.uint8))) == 1


def test_enhance_keeps_shape_and_dtype():
    out = preprocess.enhance(textured(50, 90))
    assert out.shape == (50, 90, 3) and out.dtype == np.uint8


# ---- normalisation
@pytest.mark.parametrize("raw,text", [("29-B1 123.45", "29B112345"), ("30a-123.45", "30A12345"), (" 51F-1234 ", "51F1234"),
                                      ("29MD1 123.45", "29MD112345"), ("59X1-678.90", "59X167890")])
def test_valid_plates_are_cleaned(raw, text):
    n = normalize.normalize(raw)
    assert n["text"] == text and n["format_ok"] and n["repairs"] == 0


@pytest.mark.parametrize("raw,text", [("2gB112345", "26B112345"),      # g -> 6 where a digit is required
                                      ("O9B112345", "09B112345"),      # O -> 0
                                      ("298I12345", "29B112345"),      # 8 -> B (letter slot) and I -> 1 (digit slot)
                                      ("29B1I2345", "29B112345"),
                                      ("5IF12345", "51F12345"),
                                      ("29B112S45", "29B112545")])
def test_look_alike_characters_are_repaired_by_position(raw, text):
    n = normalize.normalize(raw)
    assert n["text"] == text and n["format_ok"] and n["repairs"] >= 1


@pytest.mark.parametrize("raw", ["", "ABCDEFGH", "1234", "29B1234567890", "HELLO WORLD", "29 B"])
def test_junk_is_not_a_plate(raw):
    n = normalize.normalize(raw)
    assert n["text"] is None and n["format_ok"] is False


def test_repair_picks_the_cheapest_template():
    text, cost = normalize.repair("29BI12345")           # I -> 1 (one substitution) in the motorcycle template
    assert text == "29B112345" and cost == 1


# ---- voting
def test_vote_is_character_wise_and_weighted():
    v = vote.vote([("29B112345", 0.9), ("29B112845", 0.4), ("29B112345", 0.8), (None, 0.99)])
    assert v["text"] == "29B112345" and v["n_reads"] == 3 and not v["unanimous"] and 0.9 < v["score"] <= 1
    assert vote.vote([("30A12345", 0.7)])["unanimous"] is True
    assert vote.vote([(None, 0.9), ("", 0.5)]) is None


def test_vote_keeps_the_best_supported_length():
    v = vote.vote([("29B112345", 0.5), ("29B11234", 0.4), ("29B11235", 0.4)])
    assert len(v["text"]) == 8                           # two 8-char reads outweigh one 9-char read
    assert v["text"] == "29B11234" or v["text"] == "29B11235"


# ---- PlateOCR with a fake recogniser
class FakeRec:
    def __init__(self, lines):
        self.lines, self.calls = list(lines), 0

    def __call__(self, img):
        self.calls += 1
        return self.lines[(self.calls - 1) % len(self.lines)]


def test_plate_ocr_two_line_plate_is_read_per_line_and_voted():
    crops = [textured(80, 100, s) for s in range(4)]
    rec = FakeRec([("29-B1", 0.9), ("123.45", 0.8)])                           # always the same plate: top line then bottom line
    pred = PlateOCR(rec, reads=3).read_track(crops)
    assert pred["text"] == "29B112345" and pred["format_ok"] and pred["reads"] == 3 and pred["unanimous"]
    assert pred["raw"] == ["29-B1123.45"] * 3 and rec.calls == 6 and pred["latency_ms"] >= 0 and 0 < pred["score"] <= 1


def test_plate_ocr_unreadable_gives_a_prediction_with_text_none():
    pred = PlateOCR(FakeRec([("???", 0.2)]), reads=2).read_track([textured(30, 90), textured(32, 96, 1)])
    assert pred["text"] is None and pred["format_ok"] is False and pred["score"] is None and pred["valid_reads"] == 0 and pred["reads"] == 2


def test_plate_ocr_corrects_a_noisy_read_by_voting():
    reads = iter([("29B112345", .9), ("29B1I2345", .9), ("29B112345", .9)])
    pred = PlateOCR(lambda img: next(reads), reads=3).read_track([textured(30, 120, s) for s in range(3)])
    assert pred["text"] == "29B112345"
