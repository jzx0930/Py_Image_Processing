import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy，用來處理座標點與矩陣運算


def _order_points(pts):  # 輔助函式：將四個角點排成固定順序（左上、右上、右下、左下）
    rect = np.zeros((4, 2), dtype="float32")  # 建立4x2的陣列存放排序後的角點
    s = pts.sum(axis=1)  # 計算每個點的 x+y 總和
    rect[0] = pts[np.argmin(s)]          # 左上  # x+y最小者為左上角
    rect[2] = pts[np.argmax(s)]          # 右下  # x+y最大者為右下角
    diff = np.diff(pts, axis=1)  # 計算每個點的 y-x 差值
    rect[1] = pts[np.argmin(diff)]       # 右上  # y-x最小者為右上角
    rect[3] = pts[np.argmax(diff)]       # 左下  # y-x最大者為左下角
    return rect  # 回傳排序好的四個角點


def perspective_correct(image, canny_low=50, canny_high=150):  # 透視校正函式
    """自動偵測影像中最大的四邊形輪廓，做透視校正轉成俯視矩形（類似掃描文件）。"""  # 函式說明（保留原本 docstring）
    orig = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 確保有三通道彩色版本以利輸出
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)  # 轉成灰階，供邊緣偵測使用
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊去雜訊，避免偵測到細碎邊緣
    edged = cv2.Canny(blur, canny_low, canny_high)  # Canny 邊緣偵測找出物件輪廓
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)  # 膨脹邊緣使斷裂處相連，便於找封閉輪廓
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 找出所有輪廓並簡化端點
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 依面積由大到小排序輪廓
    quad = None  # 預設尚未找到四邊形
    for c in contours:  # 由大到小逐一檢查輪廓
        peri = cv2.arcLength(c, True)  # 計算輪廓周長（封閉曲線）
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 以周長2%為容差將輪廓近似為多邊形
        if len(approx) == 4 and cv2.contourArea(approx) > 0.1 * gray.size:  # 若為四邊形且面積夠大（大於全圖10%）
            quad = approx.reshape(4, 2).astype("float32")  # 整形成4個(x, y)浮點座標
            break  # 找到目標即停止搜尋
    if quad is None:  # 若整張圖都找不到合適四邊形
        out = orig.copy()  # 複製原圖以便加文字
        cv2.putText(out, "No quadrilateral found", (10, 30),  # 在影像上提示找不到四邊形
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 字型、大小0.7、紅色、粗細2
        return out  # 回傳提示影像
    rect = _order_points(quad)  # 將四個角點排成固定順序
    (tl, tr, br, bl) = rect  # 解包成左上、右上、右下、左下四點
    maxW = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)), 1)  # 取上下兩邊較長者作為輸出寬度（至少1）
    maxH = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)), 1)  # 取左右兩邊較長者作為輸出高度（至少1）
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")  # 定義校正後的目標矩形四角
    M = cv2.getPerspectiveTransform(rect, dst)  # 計算由原四邊形到目標矩形的透視變換矩陣
    return cv2.warpPerspective(orig, M, (maxW, maxH))  # 套用透視變換得到俯視矩形影像並回傳
