import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy 數值運算函式庫，並取別名 np


def canny(gray_image: np.ndarray, low: int, high: int) -> np.ndarray:  # 定義 Canny 邊緣偵測，low/high 為雙門檻
    return cv2.Canny(gray_image, low, high)  # 直接呼叫 OpenCV 的 Canny 並回傳邊緣影像


def sobel(gray_image: np.ndarray, ksize_index: int) -> np.ndarray:  # 定義 Sobel 邊緣偵測，ksize_index 決定卷積核大小
    ksize = 2 * ksize_index + 3  # 把索引換算成奇數核大小 (3,5,7,...)
    sx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)  # 計算 x 方向的梯度
    sy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)  # 計算 y 方向的梯度
    magnitude = np.sqrt(sx**2 + sy**2)  # 結合 x、y 梯度求梯度強度
    # 若最大值大於 0 就正規化到 0~255 並轉 uint8，否則回傳全黑影像（避免除以零）
    return np.uint8(np.clip(magnitude / magnitude.max() * 255, 0, 255)) if magnitude.max() > 0 else np.zeros_like(gray_image)


def laplacian(gray_image: np.ndarray) -> np.ndarray:  # 定義 Laplacian 邊緣偵測
    lap = cv2.Laplacian(gray_image, cv2.CV_64F)  # 計算二階導數（拉普拉斯）結果，使用 64 位元浮點數
    return np.uint8(np.clip(np.abs(lap), 0, 255))  # 取絕對值並夾在 0~255 後轉 uint8 回傳


def scharr(gray_image: np.ndarray) -> np.ndarray:  # 定義 Scharr 邊緣偵測（比 Sobel 更精準的梯度核）
    sx = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)  # 計算 x 方向的梯度
    sy = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)  # 計算 y 方向的梯度
    magnitude = np.sqrt(sx**2 + sy**2)  # 結合 x、y 梯度求梯度強度
    # 若最大值大於 0 就正規化到 0~255 並轉 uint8，否則回傳全黑影像（避免除以零）
    return np.uint8(np.clip(magnitude / magnitude.max() * 255, 0, 255)) if magnitude.max() > 0 else np.zeros_like(gray_image)
