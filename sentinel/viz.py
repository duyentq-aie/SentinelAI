import cv2
import numpy as np


def draw_sleek_box(img, x1, y1, x2, y2, label, color, thickness=1, blend="roi"):
    """Box + semi-transparent label tag, drawn in place.

    blend="roi"  : blend only the label rectangle, no full-frame copy. Identical to "full" inside the tag and on the box.
    blend="full" : the original implementation (copies and blends the WHOLE frame); kept as the A/B reference. Its fixed-point
                   addWeighted also nudges a handful of pixels elsewhere in the frame by +-1 level, which "roi" does not.
    """
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, text_thickness)

    rx1, ry1, rx2, ry2 = x1, y1 - text_height - 6, x1 + text_width + 4, y1       # filled rectangle, endpoints inclusive
    if blend == "full":
        overlay = img.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), color, -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    else:
        h, w = img.shape[:2]
        cx1, cy1, cx2, cy2 = max(rx1, 0), max(ry1, 0), min(rx2, w - 1), min(ry2, h - 1)
        if cx2 >= cx1 and cy2 >= cy1:
            roi = img[cy1:cy2 + 1, cx1:cx2 + 1]
            solid = np.empty_like(roi)
            solid[:] = color
            cv2.addWeighted(solid, 0.7, roi, 0.3, 0, dst=roi)

    text_color = (0, 0, 0) if sum(color) > 300 else (255, 255, 255)
    cv2.putText(img, label, (x1 + 2, y1 - 4), font, font_scale, text_color, text_thickness)
    return img
