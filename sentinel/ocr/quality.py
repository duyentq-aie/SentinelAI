"""Which plate crop is the best one to read? Larger and sharper wins."""
import cv2
import numpy as np

REF_AREA = 80 * 40          # px: a plate crop of this size scores 1.0 on the size term
REF_SHARP = 150.0           # Laplacian variance that scores 1.0 on the sharpness term


def sharpness(img):
    """Variance of the Laplacian of the grey image (blur -> low)."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def quality(img):
    """0.5 * size + 0.5 * sharpness, both clipped (size at 4x, sharpness at 3x the reference) so that neither can dominate.
    Heuristic, not tuned on labelled plates."""
    h, w = img.shape[:2]
    size = min(h * w / REF_AREA, 4.0)
    sharp = min(sharpness(img) / REF_SHARP, 3.0)
    return 0.5 * size + 0.5 * sharp


def best(crops, k=1):
    """Indices of the k best crops (highest quality first)."""
    order = sorted(range(len(crops)), key=lambda i: quality(crops[i]), reverse=True)
    return order[:k]
