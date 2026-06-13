import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy，用來處理影像陣列

# 支援的色彩空間名稱清單，索引對應使用者選單的選項
COLOR_MODES = ["BGR", "HSV", "Lab", "YCrCb", "HLS", "LUV", "XYZ", "YUV", "GRAY", "RGB"]  # 各色彩模式名稱

# 各色彩模式名稱對應到 OpenCV 的色彩轉換常數（皆從 BGR 出發）
_CONVERSIONS = {  # 色彩轉換對照表
    "HSV":   cv2.COLOR_BGR2HSV,  # BGR 轉 HSV（色相、飽和、明度）
    "Lab":   cv2.COLOR_BGR2Lab,  # BGR 轉 Lab
    "YCrCb": cv2.COLOR_BGR2YCrCb,  # BGR 轉 YCrCb
    "HLS":   cv2.COLOR_BGR2HLS,  # BGR 轉 HLS
    "LUV":   cv2.COLOR_BGR2LUV,  # BGR 轉 LUV
    "XYZ":   cv2.COLOR_BGR2XYZ,  # BGR 轉 XYZ
    "YUV":   cv2.COLOR_BGR2YUV,  # BGR 轉 YUV
    "GRAY":  cv2.COLOR_BGR2GRAY,  # BGR 轉灰階
    "RGB":   cv2.COLOR_BGR2RGB,  # BGR 轉 RGB（調換通道順序）
}


def split_channels(image: np.ndarray, color_mode_index: int) -> list[np.ndarray]:  # 依選定色彩空間拆分通道
    mode = COLOR_MODES[color_mode_index]  # 由索引取得對應的色彩模式名稱
    if mode == "BGR":  # 若為原始 BGR
        converted = image  # 不需轉換，直接使用原影像
    elif mode == "GRAY":  # 若為灰階
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 轉成灰階
        return [gray]  # 灰階只有單一通道，直接回傳
    else:  # 其他色彩空間
        converted = cv2.cvtColor(image, _CONVERSIONS[mode])  # 依對照表轉換到目標色彩空間
    channels = cv2.split(converted)  # 將影像拆分成各個獨立通道
    return list(channels)  # 轉成串列回傳
