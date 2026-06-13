import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy 數值運算函式庫，並取別名 np


def filter_gaussian(image, ksize=3, sigma_x10=10):  # 定義高斯模糊函式，ksize 控制核大小、sigma_x10 為標準差的十倍
    k = ksize * 2 + 1  # 把參數換算成奇數核邊長 (3,5,7,...)，確保有中心點
    return cv2.GaussianBlur(image, (k, k), sigma_x10 / 10)  # 套用高斯模糊，sigma 還原為實際小數值後回傳


def filter_box(image, ksize=3):  # 定義均值（盒狀）模糊函式，ksize 控制核大小
    k = ksize * 2 + 1  # 把參數換算成奇數核邊長
    return cv2.blur(image, (k, k))  # 以 k x k 鄰域取平均做模糊並回傳


def filter_median(image, ksize=3):  # 定義中值濾波函式，常用於去除椒鹽雜訊
    k = ksize * 2 + 1  # 把參數換算成奇數核邊長
    return cv2.medianBlur(image, k)  # 以鄰域中位數取代中心像素並回傳


def filter_bilateral(image, d=9, sigma_color=75, sigma_space=75):  # 定義雙邊濾波函式，可在保留邊緣的同時平滑
    # 雙邊濾波需彩色輸入，若是單通道灰階就先轉成 BGR，否則直接使用原圖
    src = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    return cv2.bilateralFilter(src, d, sigma_color, sigma_space)  # 依鄰域直徑與顏色/空間標準差做雙邊濾波並回傳
