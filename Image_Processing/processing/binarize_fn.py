import cv2
import numpy as np


def binarize_custom(gray_image: np.ndarray, threshold: int) -> np.ndarray:
    _, result = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return result


def binarize_otsu(gray_image: np.ndarray) -> np.ndarray:
    _, result = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result


def binarize_adaptive(gray_image: np.ndarray, block_size_index: int, c_offset: int) -> np.ndarray:
    block_size = 2 * block_size_index + 3
    c_value = c_offset - 50
    result = cv2.adaptiveThreshold(
        gray_image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c_value
    )
    return result
