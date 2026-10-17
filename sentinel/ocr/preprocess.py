"""OpenCV preprocessing of a plate crop before recognition: upscale, contrast (CLAHE), split two-line motorcycle plates."""
import cv2
import numpy as np

TWO_LINE_RATIO = 0.55       # height / width above this = two-line plate (motorcycle plates ~0.7-0.8, car plates ~0.2-0.3)
MIN_HEIGHT = 128            # plate crops from a 720p frame are only ~25x23 px: upscale to this height (at most MAX_SCALE x).
MAX_SCALE = 6.0             # (with a 64 px target the recogniser answered with CJK glyphs; larger targets give Latin characters)


def is_two_line(img):
    h, w = img.shape[:2]
    return w > 0 and h / w > TWO_LINE_RATIO


def upscale(img, min_height=MIN_HEIGHT, max_scale=MAX_SCALE):
    h, w = img.shape[:2]
    if h >= min_height:
        return img
    s = min(max_scale, min_height / max(h, 1))
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_CUBIC)


def enhance(img, clip=2.0, tile=4):
    """CLAHE on the lightness channel (keeps 3 channels: PaddleOCR expects BGR)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def split_two_lines(img):
    """Cut a two-line plate at the darkest-ink gap near the middle: row with the fewest 'ink' pixels between 35% and 65% of the height."""
    h = img.shape[0]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:                                   # make ink the minority class regardless of plate colours
        bw = 255 - bw
    ink = (bw > 0).sum(axis=1).astype(float)
    ink = np.convolve(ink, np.ones(3) / 3, mode="same")   # smooth over 3 rows
    lo, hi = int(h * 0.35), max(int(h * 0.65), int(h * 0.35) + 1)
    cut = lo + int(np.argmin(ink[lo:hi]))
    return img[:cut], img[cut:]


def prepare(img):
    """crop -> list of line images ready for the recogniser (1 for car plates, 2 for motorcycle plates)."""
    img = enhance(upscale(img))
    if is_two_line(img):
        top, bottom = split_two_lines(img)
        return [x for x in (top, bottom) if x.shape[0] >= 8]
    return [img]
