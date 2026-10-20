"""tools/anonymize.py on synthetic images (no models)."""
import sys
from types import SimpleNamespace

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


# ---------------------------------------------------------------- helmet_demo mode
def changed(a, b, box):
    x1, y1, x2, y2 = box
    return bool((a[y1:y2, x1:x2] != b[y1:y2, x1:x2]).any())


def test_helmet_demo_keeps_the_helmet_part_of_a_head_box_and_blurs_the_face_part():
    img = noisy(480, 640)
    ref = img.copy()
    an.anonymize_helmet_demo(img, heads=[(100, 100, 160, 180)])                          # an 80 px high head box: the upper 55% is rows 100..144
    assert not changed(img, ref, (100, 100, 160, 143))                                    # the hat region: not one pixel changed
    assert detail(img[150:182, 100:160]) < 0.01 * detail(ref[150:182, 100:160])           # the face region: detail gone
    assert (img[:98] == ref[:98]).all() and (img[:, :94] == ref[:, :94]).all()            # nothing far away changed
    z = an.face_zone((100, 100, 160, 180))
    assert z[1] == 100 + 0.55 * 80 and z[0] < 100 and z[2] > 160


def test_helmet_demo_pixelates_every_plate_box_in_blocks_of_at_least_8_px():
    img = noisy(480, 640)
    ref = img.copy()
    an.anonymize_helmet_demo(img, plates=[(300, 300, 340, 320)])                          # a 40 x 20 px plate (the padded box is 56 x 28)
    assert changed(img, ref, (300, 300, 340, 320))
    x1, y1, x2, y2 = an.clip_box(an.pad_box((300, 300, 340, 320), an.PLATE_PAD), img.shape)
    crop = img[y1:y2, x1:x2]
    assert detail(crop) < 0.05 * detail(ref[y1:y2, x1:x2])
    blk = max(8, round(min(x2 - x1, y2 - y1) / 3))
    assert blk >= 8
    for row in (crop[3], crop[crop.shape[0] // 2], crop[-3]):                             # along any row the colour is constant over runs of >= 8 px (the block width)
        runs, run = [], 1
        for k in range(1, len(row)):
            if (row[k] == row[k - 1]).all():
                run += 1
            else:
                runs.append(run)
                run = 1
        runs.append(run)
        assert min(runs) >= 8 and len(runs) <= crop.shape[1] // blk + 1
    big = noisy(480, 640)
    an.anonymize_helmet_demo(big, plates=[(100, 100, 220, 160)])                          # a big, near plate: blocks grow with it (at most about 3 across the short side)
    x1, y1, x2, y2 = an.clip_box(an.pad_box((100, 100, 220, 160), an.PLATE_PAD), big.shape)
    blk = round(min(x2 - x1, y2 - y1) / 3)
    assert blk > 8 and (big[y1:y1 + blk, x1:x1 + blk].reshape(-1, 3) == big[y1, x1]).all()


def test_a_vehicle_without_a_detected_plate_gets_an_estimated_zone_pixelated_and_one_with_a_plate_does_not():
    img = noisy(480, 640)
    ref = img.copy()
    moto, car = ("motorcycle", (100, 100, 180, 300)), ("car", (300, 100, 500, 300))
    an.anonymize_helmet_demo(img, plates=[(380, 260, 420, 280)], vehicles=[moto, car])
    assert changed(img, ref, (115, 240, 165, 296))                                         # estimated plate zone of the motorcycle: lower 40%, central part
    assert not changed(img, ref, (100, 100, 180, 215))                                     # the motorcycle's upper part (rider's head) is left alone
    assert changed(img, ref, (385, 262, 415, 278))                                         # the detected plate of the car
    assert not changed(img, ref, (300, 100, 500, 200)) and not changed(img, ref, (300, 220, 340, 296))      # the car: only the plate box, no blanket zone
    assert an.estimated_plate_zone(("truck", (0, 0, 100, 100)))[1] == pytest.approx(30) and an.estimated_plate_zone(("car", (0, 0, 100, 100)))[1] == pytest.approx(55)
    assert an.estimated_plate_zone((0, 0, 100, 100))[1] == pytest.approx(60)                                # a plain box counts as a motorcycle


def test_riders_keep_their_helmet_pedestrians_get_their_upper_body_blurred():
    img = noisy(600, 800)
    ref = img.copy()
    moto = ("motorcycle", (100, 200, 220, 500))
    rider = (110, 120, 210, 380)                                                           # sits on the motorcycle (> 30% of its box inside)
    ped = (500, 150, 560, 400)                                                             # on foot
    an.anonymize_helmet_demo(img, heads=[(135, 122, 185, 170)], persons=[rider, ped], vehicles=[moto, ("motorcycle", (600, 450, 700, 560))])
    assert not changed(img, ref, (135, 122, 185, 148))                                     # the rider's helmet region: untouched
    assert changed(img, ref, (135, 155, 185, 172))                                         # his face region (lower part of the head box)
    assert detail(img[150:240, 500:560]) < 0.01 * detail(ref[150:240, 500:560])            # the pedestrian: upper 40% of the person box (rows 150..250)
    assert not changed(img, ref, (500, 300, 560, 400))                                     # ... and only that


def test_a_rider_without_a_head_box_gets_an_estimated_head_with_only_its_lower_part_blurred():
    img = noisy(600, 800)
    ref = img.copy()
    rider, moto = (110, 120, 210, 380), ("motorcycle", (100, 200, 220, 500))
    an.anonymize_helmet_demo(img, persons=[rider], vehicles=[moto])
    top = 120 + an.EST_HEAD_FRAC * 260                                                      # the estimated head box spans rows 120..177
    assert not changed(img, ref, (140, 120, 180, int(120 + 0.55 * an.EST_HEAD_FRAC * 260) - 1))
    assert changed(img, ref, (140, int(top) - 8, 180, int(top)))


def test_helmet_demo_never_changes_pixels_outside_the_boxes_it_names():
    img = noisy(480, 640)
    ref = img.copy()
    an.anonymize_helmet_demo(img, heads=[(50, 50, 90, 100)], plates=[(300, 300, 340, 320)], persons=[(400, 100, 460, 300)], vehicles=[("car", (250, 200, 450, 330))])
    blur, pix = an.helmet_demo_boxes([(50, 50, 90, 100)], [(300, 300, 340, 320)], [(400, 100, 460, 300)], [("car", (250, 200, 450, 330))])
    mask = np.zeros(img.shape[:2], bool)
    for b in blur + pix:
        x1, y1, x2, y2 = an.clip_box(b, img.shape)
        mask[y1:y2, x1:x2] = True
    assert (img[~mask] == ref[~mask]).all() and (img[mask] != ref[mask]).any()


def test_the_helmet_demo_anonymizer_applies_the_mode_and_normal_mode_is_unchanged():
    regs = dict(heads=lambda im: [(50, 50, 90, 100)], plates=lambda im: [])
    img = noisy(480, 640)
    ref = img.copy()
    an.Anonymizer(**regs, mode="helmet_demo").apply(img)
    assert not changed(img, ref, (50, 50, 90, 76)) and changed(img, ref, (50, 80, 90, 100))
    img2 = ref.copy()
    an.Anonymizer(**regs).apply(img2)                                                       # default mode: the whole padded head box is blurred
    assert changed(img2, ref, (50, 50, 90, 90))


def test_annotations_drawn_after_the_blur_stay_sharp():
    sys.path.insert(0, str(REPO / "tools"))
    import make_demo as md
    img = noisy(480, 640)
    clean = img.copy()
    blurred = img.copy()
    an.anonymize_helmet_demo(blurred, plates=[(300, 300, 340, 320)])

    class FakePipe:
        light_source = "coco"
        camera = type("C", (), {"lanes": None})()

        def stop_line_active(self):
            return False

        def _draw(self, frame, p, result):
            cv2.rectangle(frame, (290, 290), (350, 330), (0, 255, 0), 2)               # a box drawn around the blurred plate

    out = md.render_annotated(FakePipe(), SimpleNamespace(raw=clean, perception=None, rules=None), blurred)
    assert (out[290, 290:351] == (0, 255, 0)).all()                                        # the box line is pure and sharp
    assert (out[305:315, 305:315] == blurred[305:315, 305:315]).all()                       # what is inside is the BLURRED frame, not the clean one
    assert not (out[305:315, 305:315] == clean[305:315, 305:315]).all()
