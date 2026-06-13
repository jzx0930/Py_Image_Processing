import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy 數值運算函式庫，並取別名 np


def gamma(gray_image: np.ndarray, gamma_val: float) -> np.ndarray:  # 定義 Gamma 校正函式，gamma_val 為 Gamma 值
    gamma_val = max(gamma_val, 0.01)  # 將 Gamma 值下限設為 0.01，避免除以零或指數爆炸
    # 預先計算 0~255 每個灰階的 Gamma 映射結果，建立查找表 (LUT)
    table = np.array([((i / 255.0) ** (1.0 / gamma_val)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray_image, table)  # 用查找表把每個像素映射成校正後的值並回傳


def log_transform(gray_image: np.ndarray, k: float) -> np.ndarray:  # 定義對數轉換函式，k 為縮放係數，可壓縮高動態範圍
    result = k * np.log1p(gray_image.astype(np.float64))  # 對像素值取 log(1+x) 後乘上係數 k，以浮點數運算
    # 若結果最大值大於 0 就正規化到 0~255，否則維持原結果（避免除以零）
    result = np.clip(result / result.max() * 255, 0, 255) if result.max() > 0 else result
    return result.astype(np.uint8)  # 轉成 uint8 後回傳


def linear(gray_image: np.ndarray, a: float, b: float) -> np.ndarray:  # 定義線性轉換函式，a 為斜率（對比）、b 為偏移（亮度）
    result = a * gray_image.astype(np.float64) + b  # 以 a*x + b 做線性映射，先轉浮點數避免溢位
    return np.clip(result, 0, 255).astype(np.uint8)  # 夾在 0~255 後轉 uint8 回傳


def histogram_eq(gray_image: np.ndarray) -> np.ndarray:  # 定義直方圖等化函式，用於增強整體對比
    return cv2.equalizeHist(gray_image)  # 直接呼叫 OpenCV 的直方圖等化並回傳


def clahe(gray_image: np.ndarray, clip_limit: float) -> np.ndarray:  # 定義限制對比自適應直方圖等化 (CLAHE)，clip_limit 控制對比上限
    c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))  # 建立 CLAHE 物件，影像分成 8x8 區塊分別等化
    return c.apply(gray_image)  # 對影像套用 CLAHE 並回傳結果
