import numpy as np
import pytest

from sentinel.viz import draw_sleek_box


def both(img, *args, **kw):
    a, b = img.copy(), img.copy()
    draw_sleek_box(a, *args, blend="full", **kw)
    draw_sleek_box(b, *args, blend="roi", **kw)
    return a, b


def assert_same_picture(a, b, max_diff_pixels=12):
    """Same image up to the +-1 rounding noise that the whole-frame blend adds to a few pixels outside the tag."""
    d = np.abs(a.astype(int) - b.astype(int))
    assert d.max() <= 1, d.max()
    assert np.count_nonzero(d.any(axis=2)) <= max_diff_pixels


def test_roi_blend_equals_the_full_frame_blend_up_to_rounding_noise_on_random_boxes():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    for _ in range(200):
        x1, y1 = int(rng.integers(-50, 1300)), int(rng.integers(-50, 760))
        x2, y2 = x1 + int(rng.integers(5, 300)), y1 + int(rng.integers(5, 300))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        label = "ID:M_%d %s" % (int(rng.integers(0, 999)), rng.choice(["NO-HELMET", "OVERLOAD & NO-HELMET", "PLATE", "CAR"]))
        a, b = both(img, x1, y1, x2, y2, label, color, int(rng.integers(1, 4)))
        assert_same_picture(a, b)


@pytest.mark.parametrize("box", [(0, 0, 50, 50), (1270, 710, 1400, 800), (-20, -20, 40, 40), (600, 3, 700, 90), (5, 715, 100, 719)])
def test_roi_blend_matches_at_the_frame_edges(box):
    img = np.random.default_rng(1).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    a, b = both(img, *box, "ID:C_12 WRONG-WAY", (0, 0, 255), 2)
    assert_same_picture(a, b)


def test_tag_pixels_are_exactly_equal_in_both_modes():
    img = np.random.default_rng(2).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    a, b = both(img, 100, 100, 200, 180, "ID:C_9 CAR", (200, 100, 50), 1)
    assert np.array_equal(a[80:181, 100:210], b[80:181, 100:210])


def test_roi_blend_touches_only_the_tag_and_the_box():
    img = np.full((200, 300, 3), 77, np.uint8)
    out = img.copy()
    draw_sleek_box(out, 100, 100, 160, 150, "TAG", (0, 200, 0), 1, blend="roi")
    changed = np.argwhere((out != img).any(axis=2))
    assert changed[:, 0].min() >= 100 - 25 and changed[:, 0].max() <= 150 and changed[:, 1].min() >= 100 and changed[:, 1].max() <= 160


def test_default_blend_is_roi_and_does_not_copy_the_frame(monkeypatch):
    img = np.zeros((200, 300, 3), np.uint8)
    copies = []
    class Spy(np.ndarray):
        def copy(self, *a, **k):
            copies.append(1)
            return super().copy(*a, **k)
    draw_sleek_box(img.view(Spy), 10, 40, 80, 90, "X", (1, 2, 3))
    assert copies == []
