import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy 數值運算函式庫，並取別名 np


def inpaint_auto(image, thresh=240, radius=3, method=0):  # 定義自動修復函式，thresh 亮度門檻、radius 修補半徑、method 演算法
    """自動修復：把過亮（高於門檻）的區域視為瑕疵/刮痕，用周圍像素填補。

    method: 0=Telea，1=Navier-Stokes。
    """
    # 若輸入是三通道就直接用，否則把灰階轉成 BGR，確保後續為彩色格式
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # 轉成灰階以便依亮度判斷瑕疵區域
    # 把亮度高於門檻的像素設為 255，產生標示瑕疵的遮罩
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    # 對遮罩做膨脹，稍微擴大瑕疵範圍，避免邊緣殘留
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    # 依 method 選擇修補演算法：0 用 Telea，否則用 Navier-Stokes
    flag = cv2.INPAINT_TELEA if method == 0 else cv2.INPAINT_NS
    return cv2.inpaint(bgr, mask, radius, flag)  # 依遮罩與半徑進行修補並回傳結果
