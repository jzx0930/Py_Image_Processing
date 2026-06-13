import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy（本檔提供陣列支援）


def _ensure_gray(image):  # 輔助函式：確保影像為灰階
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image  # 彩色就轉灰階，已是灰階則直接用


def morph_dilate(image, ksize=3, iterations=1):  # 膨脹：擴大白色前景區域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize * 2 + 1, ksize * 2 + 1))  # 建立矩形結構元素（核大小為奇數）
    return cv2.dilate(_ensure_gray(image), kernel, iterations=iterations)  # 執行膨脹，iterations為重複次數


def morph_erode(image, ksize=3, iterations=1):  # 侵蝕：縮小白色前景區域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize * 2 + 1, ksize * 2 + 1))  # 建立矩形結構元素（核大小為奇數）
    return cv2.erode(_ensure_gray(image), kernel, iterations=iterations)  # 執行侵蝕，iterations為重複次數


def morph_open(image, ksize=3):  # 開運算：先侵蝕後膨脹，去除小雜點
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize * 2 + 1, ksize * 2 + 1))  # 建立矩形結構元素（核大小為奇數）
    return cv2.morphologyEx(_ensure_gray(image), cv2.MORPH_OPEN, kernel)  # 執行開運算


def morph_close(image, ksize=3):  # 閉運算：先膨脹後侵蝕，填補小孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize * 2 + 1, ksize * 2 + 1))  # 建立矩形結構元素（核大小為奇數）
    return cv2.morphologyEx(_ensure_gray(image), cv2.MORPH_CLOSE, kernel)  # 執行閉運算
