import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy，用來處理影像的多維陣列


def _maybe_invert(image: np.ndarray, inv: int) -> np.ndarray:  # 定義輔助函式：依旗標決定是否反相
    return cv2.bitwise_not(image) if inv else image  # inv 為真就把黑白顛倒，否則原樣回傳


def binarize_custom(gray_image: np.ndarray, threshold: int, inv: int = 0) -> np.ndarray:  # 自訂門檻二值化
    _, result = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)  # 大於門檻設為255，其餘設為0
    return _maybe_invert(result, inv)  # 視需要反相後回傳結果


def binarize_otsu(gray_image: np.ndarray, inv: int = 0) -> np.ndarray:  # 大津法(Otsu)自動門檻二值化
    _, result = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 由影像自動計算最佳門檻
    return _maybe_invert(result, inv)  # 視需要反相後回傳結果


def binarize_adaptive(gray_image: np.ndarray, block_size_index: int, c_offset: int, inv: int = 0) -> np.ndarray:  # 自適應門檻二值化
    block_size = 2 * block_size_index + 3  # 由索引換算成奇數的鄰域區塊大小（必須為奇數）
    c_value = c_offset - 50  # 將偏移值平移到以0為中心的常數C（從平均值中扣除）
    result = cv2.adaptiveThreshold(  # 對每個區域分別計算門檻，適合光照不均的影像
        gray_image, 255,  # 輸入灰階影像，最大值設為255
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 以高斯加權平均計算區域門檻
        cv2.THRESH_BINARY,  # 採用一般二值化（大於門檻為白）
        block_size, c_value  # 區塊大小與常數C
    )
    return _maybe_invert(result, inv)  # 視需要反相後回傳結果


def binarize_zone(gray_image: np.ndarray, rows: int = 3, cols: int = 3, inv: int = 0) -> np.ndarray:  # 區域二值化
    """區域二值化：將影像切成 rows×cols 格子，每格各自用 Otsu 計算最佳門檻後拼回。"""  # 函式說明
    h, w = gray_image.shape[:2]  # 取得影像高度與寬度
    result = np.zeros_like(gray_image)  # 建立與輸入同大小的全零輸出陣列
    for r in range(rows):  # 逐列走訪
        y0 = h * r // rows  # 計算此列的起始Y座標
        y1 = h * (r + 1) // rows  # 計算此列的結束Y座標
        for c in range(cols):  # 逐欄走訪
            x0 = w * c // cols  # 計算此欄的起始X座標
            x1 = w * (c + 1) // cols  # 計算此欄的結束X座標
            tile = gray_image[y0:y1, x0:x1]  # 切出此格子的影像區塊
            if tile.size == 0:  # 跳過空白區塊（影像極小時可能發生）
                continue
            _, bin_tile = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 對此區塊用 Otsu 自動門檻二值化
            result[y0:y1, x0:x1] = bin_tile  # 將二值化結果填入輸出陣列對應位置
    return _maybe_invert(result, inv)  # 視需要反相後回傳完整結果
