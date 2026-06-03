import cv2
import numpy as np


def canny(gray_image: np.ndarray, low: int, high: int) -> np.ndarray:
    return cv2.Canny(gray_image, low, high)


def sobel(gray_image: np.ndarray, ksize_index: int) -> np.ndarray:
    ksize = 2 * ksize_index + 3
    sx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)
    sy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(sx**2 + sy**2)
    return np.uint8(np.clip(magnitude / magnitude.max() * 255, 0, 255)) if magnitude.max() > 0 else np.zeros_like(gray_image)


def laplacian(gray_image: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(gray_image, cv2.CV_64F)
    return np.uint8(np.clip(np.abs(lap), 0, 255))


def scharr(gray_image: np.ndarray) -> np.ndarray:
    sx = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
    sy = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(sx**2 + sy**2)
    return np.uint8(np.clip(magnitude / magnitude.max() * 255, 0, 255)) if magnitude.max() > 0 else np.zeros_like(gray_image)
