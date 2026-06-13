import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy 數值運算函式庫，並取別名 np


def rayleigh_transform(image, alpha=60):  # 定義雷利轉換函式，alpha 為分佈尺度參數，預設 60
    """雷利(Rayleigh)直方圖規定化：依累積分佈將灰階重新映射成雷利分佈以增強對比。"""
    # 若輸入是三通道彩色影像就轉成灰階，否則直接使用原灰階影像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # 計算灰階影像 0~255 共 256 級的直方圖，並攤平成一維陣列
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    cdf = hist.cumsum()  # 對直方圖逐項累加，得到累積分佈函數 (CDF)
    total = cdf[-1]  # 取累積分佈最後一項，即像素總數
    if total == 0:  # 若像素總數為 0（空影像）
        return gray  # 直接回傳原灰階影像，避免除以零
    # 將 CDF 正規化到 0~1 之間，並夾住上限略小於 1 以避免後面 log(0) 發生
    cdf = np.clip(cdf / total, 0, 1 - 1e-6)
    # 套用雷利分佈反函數公式，計算每個灰階對應的新值 z
    z = np.sqrt(2.0 * (alpha ** 2) * np.log(1.0 / (1.0 - cdf)))
    # 將計算結果夾在 0~255 後轉成 uint8，作為查找表 (LUT)
    lut = np.clip(z, 0, 255).astype(np.uint8)
    return lut[gray]  # 用查找表把每個原灰階值映射成新值並回傳
