import cv2
import numpy as np


def gamma(gray_image: np.ndarray, gamma_val: float) -> np.ndarray:
    gamma_val = max(gamma_val, 0.01)
    table = np.array([((i / 255.0) ** (1.0 / gamma_val)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray_image, table)


def log_transform(gray_image: np.ndarray, k: float) -> np.ndarray:
    result = k * np.log1p(gray_image.astype(np.float64))
    result = np.clip(result / result.max() * 255, 0, 255) if result.max() > 0 else result
    return result.astype(np.uint8)


def linear(gray_image: np.ndarray, a: float, b: float) -> np.ndarray:
    result = a * gray_image.astype(np.float64) + b
    return np.clip(result, 0, 255).astype(np.uint8)


def histogram_eq(gray_image: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray_image)


def clahe(gray_image: np.ndarray, clip_limit: float) -> np.ndarray:
    c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return c.apply(gray_image)
