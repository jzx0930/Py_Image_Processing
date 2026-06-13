import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy，用來做數值與陣列運算


def sharpen_unsharp(image, amount=10, radius=3):  # 遮罩銳化(Unsharp Mask)：用原圖減模糊圖來強化邊緣
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image  # 彩色就轉灰階，已是灰階則直接用
    blurred = cv2.GaussianBlur(gray, (radius * 2 + 1, radius * 2 + 1), 0)  # 高斯模糊取得低頻部分（核大小須為奇數）
    sharp = cv2.addWeighted(gray, 1 + amount / 10, blurred, -amount / 10, 0)  # 原圖加權減去模糊圖以放大細節
    return sharp  # 回傳銳化後影像


def sharpen_laplacian(image, scale=2):  # 拉普拉斯銳化：用二階導數偵測邊緣再加回原圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image  # 彩色就轉灰階，已是灰階則直接用
    lap = cv2.Laplacian(gray, cv2.CV_64F)  # 計算拉普拉斯（邊緣響應），用浮點型避免溢位
    sharp = np.clip(gray.astype(np.float64) - scale * lap, 0, 255).astype(np.uint8)  # 原圖減邊緣強化細節，裁切到0~255並轉回8位元
    return sharp  # 回傳銳化後影像
