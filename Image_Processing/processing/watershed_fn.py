import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy 數值運算函式庫，並取別名 np


def watershed_segment(image, fg_ratio=50):  # 定義分水嶺分割函式，fg_ratio 為前景取樣百分比
    """分水嶺分割：自動標記前景/背景，分離相黏物件，以紅色邊界標示各區塊。

    fg_ratio: 前景取樣閾值（距離變換最大值的百分比），越大分離越嚴格。
    """
    # 若輸入是三通道就直接用，否則把灰階轉成 BGR，確保為彩色格式
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # 轉成灰階以利後續門檻處理
    # 用 Otsu 自動門檻做二值化，自動選出最佳分割閾值
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)  # 建立 3x3 的形態學運算核
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  # 開運算去除小雜訊
    sure_bg = cv2.dilate(opening, kernel, iterations=3)  # 膨脹擴大區域，得到確定的背景區
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # 距離變換，計算每點到最近背景的距離
    # 依 fg_ratio 取距離變換最大值的百分比為門檻，提取確定的前景區
    _, sure_fg = cv2.threshold(dist, (fg_ratio / 100.0) * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)  # 把前景圖轉成 uint8 格式
    unknown = cv2.subtract(sure_bg, sure_fg)  # 背景減前景，得到不確定（待分割）區域
    _, markers = cv2.connectedComponents(sure_fg)  # 對確定前景做連通元件標記，每個物件給不同編號
    markers = markers + 1  # 全部標記加 1，讓背景標記從 1 開始（保留 0 給未知區）
    markers[unknown == 255] = 0  # 把未知區域的標記設為 0，交由分水嶺決定歸屬
    markers = cv2.watershed(bgr, markers)  # 執行分水嶺演算法，邊界處會被標成 -1
    out = bgr.copy()  # 複製原影像作為輸出畫布
    out[markers == -1] = [0, 0, 255]  # 把分水嶺邊界（-1）的像素塗成紅色
    n = int(markers.max()) - 1  # 計算分割出的區域數（最大標記減去背景）
    # 在影像左上角標註偵測到的區域數量
    cv2.putText(out, f"Regions: {n}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out  # 回傳已標記分割邊界與區域數的影像
