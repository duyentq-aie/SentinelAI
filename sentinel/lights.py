import cv2
import numpy as np


def detect_traffic_light_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
    lower_green, upper_green = np.array([40, 50, 50]), np.array([90, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    if cv2.countNonZero(mask_red) > cv2.countNonZero(mask_green) and cv2.countNonZero(mask_red) > 5:
        return "RED"
    elif cv2.countNonZero(mask_green) > cv2.countNonZero(mask_red) and cv2.countNonZero(mask_green) > 5:
        return "GREEN"
    return "UNKNOWN"
