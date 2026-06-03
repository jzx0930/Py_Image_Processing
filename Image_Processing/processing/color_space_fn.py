import cv2
import numpy as np

COLOR_MODES = ["BGR", "HSV", "Lab", "YCrCb", "HLS", "LUV", "XYZ", "YUV", "GRAY", "RGB"]

_CONVERSIONS = {
    "HSV":   cv2.COLOR_BGR2HSV,
    "Lab":   cv2.COLOR_BGR2Lab,
    "YCrCb": cv2.COLOR_BGR2YCrCb,
    "HLS":   cv2.COLOR_BGR2HLS,
    "LUV":   cv2.COLOR_BGR2LUV,
    "XYZ":   cv2.COLOR_BGR2XYZ,
    "YUV":   cv2.COLOR_BGR2YUV,
    "GRAY":  cv2.COLOR_BGR2GRAY,
    "RGB":   cv2.COLOR_BGR2RGB,
}


def split_channels(image: np.ndarray, color_mode_index: int) -> list[np.ndarray]:
    mode = COLOR_MODES[color_mode_index]
    if mode == "BGR":
        converted = image
    elif mode == "GRAY":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return [gray]
    else:
        converted = cv2.cvtColor(image, _CONVERSIONS[mode])
    channels = cv2.split(converted)
    return list(channels)
