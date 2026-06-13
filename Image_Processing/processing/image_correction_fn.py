import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy 數值運算函式庫，並取別名 np


def affine(image: np.ndarray, dx: int, dy: int) -> np.ndarray:  # 定義仿射變換函式，dx/dy 控制邊角內縮量
    h, w = image.shape[:2]  # 取得影像的高與寬（忽略通道數）
    # 設定來源三個基準點：左上、右上、左下三角，用以決定仿射映射
    src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    # 設定目標三個對應點，依 dx/dy 將邊角向內位移，產生變形效果
    dst_pts = np.float32([[dx, dy], [w - 1 - dx, dy], [dx, h - 1 - dy]])
    M = cv2.getAffineTransform(src_pts, dst_pts)  # 由三組對應點計算 2x3 仿射變換矩陣
    return cv2.warpAffine(image, M, (w, h))  # 套用仿射矩陣對影像做變形並回傳，輸出尺寸維持原寬高
