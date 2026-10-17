import cv2


def draw_sleek_box(img, x1, y1, x2, y2, label, color, thickness=1):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, text_thickness)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1 - text_height - 6), (x1 + text_width + 4, y1), color, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    text_color = (0, 0, 0) if sum(color) > 300 else (255, 255, 255)
    cv2.putText(img, label, (x1 + 2, y1 - 4), font, font_scale, text_color, text_thickness)
    return img
