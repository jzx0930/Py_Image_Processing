import cv2
import numpy as np


def affine(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = image.shape[:2]
    src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    dst_pts = np.float32([[dx, dy], [w - 1 - dx, dy], [dx, h - 1 - dy]])
    M = cv2.getAffineTransform(src_pts, dst_pts)
    return cv2.warpAffine(image, M, (w, h))
