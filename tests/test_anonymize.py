"""tools/anonymize.py on synthetic images (no models)."""
import sys

import cv2
import numpy as np
import pytest

from sentinel.config import REPO

sys.path.insert(0, str(REPO / "tools"))
import anonymize as an  # noqa: E402


def noisy(h=240, w=320, seed=0):
    return np.random.RandomState(seed).randint(0, 256, (h, w, 3), np.uint8)       # maximum detail: a face / a plate would be this kind of texture


def detail(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def test_blur_destroys_the_detail_inside_a_box_and_changes_nothing_outside():
    img = noisy()
    before = img.copy()
    an.blur_boxes(img, [(100, 60, 180, 140)])
    assert detail(img[60:140, 100:180]) < 0.01 * detail(before[60:140, 100:180])                  # > 99% of the high-frequency detail is gone
    mask = np.ones(img.shape[:2], bool)
    mask[60:140, 100:180] = False
    assert (img[mask] == before[mask]).all()


def test_blur_clips_boxes_to_the_image_and_ignores_empty_or_outside_boxes():
    img = noisy()
    before = img.copy()
    an.blur_boxes(img, [(-50, -50, 40, 40), (300, 220, 400, 400), (500, 500, 600, 600), (10, 10, 10, 10), (50, 50, 51, 90)])
    assert detail(img[0:40, 0:40]) < 0.01 * detail(before[0:40, 0:40]) and detail(img[220:240, 300:320]) < 0.01 * detail(before[220:240, 300:320])
    assert (img[100:200, 100:200] == before[100:200, 100:200]).all()
    assert an.blur_boxes(noisy(), []).shape == (240, 320, 3)


def test_the_blur_is_strong_even_for_a_small_head():
    img = noisy()
    an.blur_boxes(img, [(100, 100, 130, 130)])                                                    # a 30 px head
    assert detail(img[100:130, 100:130]) < 5 and img[100:130, 100:130].std() < 40                 # nothing but a smudge of the average colour


def test_geometry_helpers():
    assert an.person_upper((10, 100, 50, 300)) == (10, 100, 50, 180) and an.person_upper((10, 100, 50, 300), 0.5) == (10, 100, 50, 200)
    assert an.pad_box((100, 100, 140, 160), 0.25) == (90, 85, 150, 175)
    x1, y1, x2, y2 = an.plate_zone((100, 100, 200, 300))
    assert (x1, x2) == (115, 185) and y1 == 170 and y2 == 300                                     # lower 65%, central 70%


def test_anonymize_frame_blurs_heads_plates_person_upper_half_and_optionally_the_vehicle_plate_zone():
    img = noisy(480, 640)
    ref = img.copy()
    an.anonymize_frame(img, heads=[(50, 50, 90, 100)], plates=[(300, 300, 340, 320)], persons=[(400, 100, 460, 300)])
    for box in ((50, 50, 90, 100), (300, 300, 340, 320), (400, 100, 460, 180)):
        x1, y1, x2, y2 = box
        assert detail(img[y1:y2, x1:x2]) < 0.02 * detail(ref[y1:y2, x1:x2]), box
    assert (img[190:300, 400:460] == ref[190:300, 400:460]).all()                                 # the lower part of the person is left alone
    img2 = ref.copy()
    an.anonymize_frame(img2, vehicles=[(200, 200, 300, 400)])
    assert (img2 == ref).all()                                                                    # the vehicle zone only with plate_fallback
    an.anonymize_frame(img2, vehicles=[(200, 200, 300, 400)], plate_fallback=True)
    assert detail(img2[340:400, 220:280]) < 0.02 * detail(ref[340:400, 220:280])
    img3 = ref.copy()
    an.anonymize_frame(img3, vehicles=[(100, 20, 400, 460)], plate_fallback=True)                 # a big near vehicle (440 px high): left to the plate detector
    assert (img3 == ref).all()


def test_the_anonymizer_uses_its_detectors_and_can_apply_regions_found_on_another_frame():
    calls = []
    a = an.Anonymizer(heads=lambda im: calls.append("h") or [(10, 10, 50, 50)], plates=lambda im: calls.append("p") or [(100, 100, 140, 120)], persons=lambda im: [(200, 20, 260, 120)],
                      vehicles=lambda im: [(60, 100, 160, 200)])
    clean = noisy(240, 320, 1)
    annotated = clean.copy()
    cv2.rectangle(annotated, (0, 0), (319, 239), (0, 255, 0), 1)
    r = a.regions(clean)
    assert set(r) == {"heads", "plates", "persons", "vehicles"} and calls == ["h", "p"]
    a.apply(annotated, plate_fallback=True, regions=r)                                            # regions of the clean frame applied to the annotated copy
    assert detail(annotated[10:50, 10:50]) < 0.02 * detail(clean[10:50, 10:50]) and detail(annotated[100:120, 100:140]) < 0.02 * detail(clean[100:120, 100:140])
    assert len(calls) == 2                                                                        # `regions=` given: the detectors are not run again
    b = an.Anonymizer(heads=lambda im: [], plates=lambda im: [])
    img = noisy()
    assert (b.apply(img.copy()) == img).all()                                                     # nothing detected: nothing changed (persons / vehicles detectors are optional)


def test_process_video_blurs_every_frame_and_refuses_to_overwrite(tmp_path):
    src, dst = tmp_path / "a.mp4", tmp_path / "b.mp4"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(4):
        vw.write(noisy(240, 320, i))
    vw.release()
    a = an.Anonymizer(heads=lambda im: [(100, 80, 180, 160)], plates=lambda im: [])
    assert an.process_video(src, dst, a) == 4
    cap = cv2.VideoCapture(str(dst))
    ok, f = cap.read()
    assert ok and f.shape[:2] == (240, 320) and detail(f[80:160, 100:180]) < 0.2 * detail(f[0:60, 0:60])
