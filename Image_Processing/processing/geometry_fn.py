import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy，用來建立變換矩陣


def geo_scale(image, scale_x_pct=100, scale_y_pct=100):  # 影像縮放：依百分比放大或縮小
    h, w = image.shape[:2]  # 取得原影像的高度與寬度
    return cv2.resize(image, (max(1, w * scale_x_pct // 100), max(1, h * scale_y_pct // 100)))  # 依X/Y百分比計算新寬高並縮放（至少1像素）


def geo_rotate(image, angle=0):  # 影像旋轉函式
    """以影像中心旋轉，保留完整影像（填黑邊）。"""  # 函式說明（保留原本 docstring）
    h, w = image.shape[:2]  # 取得原影像的高度與寬度
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)  # 以影像中心為軸建立旋轉矩陣（縮放比例1.0）
    cos, sin = abs(M[0, 0]), abs(M[0, 1])  # 取出旋轉矩陣中的cos與sin絕對值，用來算新邊界
    nw = int(h * sin + w * cos)  # 計算旋轉後可容納整張圖的新寬度
    nh = int(h * cos + w * sin)  # 計算旋轉後可容納整張圖的新高度
    M[0, 2] += (nw - w) / 2  # 調整X平移量，使影像置中於新畫布
    M[1, 2] += (nh - h) / 2  # 調整Y平移量，使影像置中於新畫布
    return cv2.warpAffine(image, M, (nw, nh))  # 套用仿射變換完成旋轉，輸出新尺寸畫布


def geo_translate(image, tx=0, ty=0):  # 影像平移函式
    M = np.float32([[1, 0, tx], [0, 1, ty]])  # 建立平移矩陣（tx為水平位移、ty為垂直位移）
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))  # 套用仿射變換完成平移，輸出維持原尺寸


def geo_flip(image, flip_code=1):  # 影像翻轉函式
    """flip_code: 0=垂直, 1=水平, 2=兩軸。"""  # 函式說明（保留原本 docstring）
    code = flip_code if flip_code < 2 else -1  # OpenCV兩軸翻轉用-1，故將2轉換為-1
    return cv2.flip(image, code)  # 依翻轉代碼翻轉影像並回傳
