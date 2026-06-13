import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy 數值運算函式庫，並取別名 np

# 外接形狀選項（顯示順序即索引）
SHAPE_MODES = ["最小外接矩形", "正外接矩形", "外接圓形", "凸包"]  # 定義可選的外接形狀清單，索引對應使用者選擇

# 框線顏色選項：(顯示名, BGR)
BOX_COLORS = [  # 定義框線顏色清單，每個元素是 (顯示名稱, BGR 顏色)
    ("紅", (0, 0, 255)),  # 紅色，BGR 表示法
    ("綠", (0, 255, 0)),  # 綠色
    ("藍", (255, 0, 0)),  # 藍色
    ("黃", (0, 255, 255)),  # 黃色
    ("青", (255, 255, 0)),  # 青色
    ("洋紅", (255, 0, 255)),  # 洋紅色
    ("白", (255, 255, 255)),  # 白色
]


# 定義抽取白色物件函式：shape_index 選形狀，color_index 選顏色，min_area 過濾面積，thickness 線寬
def extract_white_objects(image: np.ndarray, shape_index: int = 0,
                          color_index: int = 0, min_area: int = 50,
                          thickness: int = 2) -> np.ndarray:
    """找出二值影像中灰階值為 255 的相連白色區塊，並以指定外接形狀框選。

    回傳 BGR 影像（黑白底 + 彩色外框），左上角標註偵測到的物件總數。
    """
    # 若輸入是彩色影像就轉成灰階，否則直接使用原灰階影像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # 以門檻 127 做二值化，大於門檻設為 255（白），其餘設為 0（黑）
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 找出二值影像中最外層輪廓，只保留必要的端點以節省記憶體
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)  # 把二值影像轉回三通道，方便畫上彩色外框
    color = BOX_COLORS[color_index][1]  # 依使用者選的索引取出對應的 BGR 顏色
    count = 0  # 初始化物件計數器
    for cnt in contours:  # 逐一處理每個找到的輪廓
        if cv2.contourArea(cnt) < min_area:        # 最小面積過濾雜訊
            continue  # 面積太小視為雜訊，跳過此輪廓
        count += 1  # 通過面積過濾，物件數加一
        if shape_index == 0:        # 最小外接矩形（可旋轉）
            # 計算可旋轉的最小外接矩形四個頂點，並轉成整數座標
            box = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.intp)
            cv2.drawContours(out, [box], 0, color, thickness)  # 把矩形四頂點連線畫在影像上
        elif shape_index == 1:      # 正外接矩形（軸對齊）
            x, y, w, h = cv2.boundingRect(cnt)  # 取得軸對齊外接矩形的左上座標與寬高
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)  # 畫出矩形框
        elif shape_index == 2:      # 外接圓形
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)  # 取得最小外接圓的圓心與半徑
            cv2.circle(out, (int(cx), int(cy)), int(radius), color, thickness)  # 畫出圓形框
        else:                       # 凸包
            cv2.drawContours(out, [cv2.convexHull(cnt)], 0, color, thickness)  # 計算凸包並畫出

    # 在影像左上角標註偵測到的物件總數
    cv2.putText(out, f"Objects: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return out  # 回傳已畫好外框與計數的影像
