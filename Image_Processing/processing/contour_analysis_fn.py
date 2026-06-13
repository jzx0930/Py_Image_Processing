import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy（本檔提供陣列支援）


def analyze_contours(image, min_area=100):  # 輪廓特徵分析函式
    """輪廓特徵分析：找出物件輪廓，畫出並標註面積與重心，左上角顯示物件總數。"""  # 函式說明（保留原本 docstring）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image  # 彩色就轉灰階，已是灰階則直接用
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 以固定門檻127二值化，分出前景與背景
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 只找最外層輪廓並簡化端點
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 轉回三通道彩色，方便畫上彩色標註
    count = 0  # 符合面積門檻的物件計數器
    for c in contours:  # 逐一走訪每個輪廓
        area = cv2.contourArea(c)  # 計算輪廓面積
        if area < min_area:  # 若面積太小（視為雜訊）
            continue  # 跳過此輪廓
        count += 1  # 物件數加一
        cv2.drawContours(out, [c], -1, (0, 255, 0), 2)  # 以綠色畫出輪廓線
        M = cv2.moments(c)  # 計算輪廓的影像矩，用於求重心
        if M["m00"] != 0:  # 面積(m00)不為零才能計算重心，避免除以零
            cx = int(M["m10"] / M["m00"])  # 計算重心 x 座標
            cy = int(M["m01"] / M["m00"])  # 計算重心 y 座標
            cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)  # 以紅色實心點標示重心
            cv2.putText(out, f"A={int(area)}", (cx - 20, cy - 8),  # 在重心附近標註面積數值
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)  # 字型、大小0.4、黃色、粗細1
    cv2.putText(out, f"Objects: {count}", (10, 25),  # 在左上角顯示物件總數
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 字型、大小0.7、紅色、粗細2
    return out  # 回傳標註後的影像
