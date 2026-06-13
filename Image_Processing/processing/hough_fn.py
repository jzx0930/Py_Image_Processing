import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy，用來處理角度與陣列運算


def hough_lines(image, threshold=100, min_length=50, max_gap=10):  # 霍夫直線偵測函式
    """霍夫直線偵測：先 Canny 找邊緣再偵測直線，以紅線標示。"""  # 函式說明（保留原本 docstring）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image  # 彩色就轉灰階，已是灰階則直接用
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 轉回三通道彩色，方便畫上彩色線條
    edges = cv2.Canny(gray, 50, 150)  # Canny 邊緣偵測，找出直線候選的邊緣點
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,  # 機率霍夫轉換偵測線段（距離解析度1像素、角度1度）
                            minLineLength=min_length, maxLineGap=max_gap)  # 設定最短線段長度與允許的線段間隙
    n = 0  # 偵測到的直線數量計數器
    if lines is not None:  # 若有偵測到直線才進入處理
        n = len(lines)  # 取得直線數量
        for l in lines:  # 逐條走訪偵測到的線段
            x1, y1, x2, y2 = l[0]  # 取出線段兩端點座標
            cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 以紅色(BGR)畫出線段，線寬2
    cv2.putText(out, f"Lines: {n}", (10, 25),  # 在左上角標示偵測到的直線數量
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # 字型、大小0.7、綠色、粗細2
    return out  # 回傳標註後的影像


def hough_circles(image, min_dist=20, param2=40, min_radius=0, max_radius=0):  # 霍夫圓偵測函式
    """霍夫圓偵測：偵測影像中的圓形並標示圓心與圓周。"""  # 函式說明（保留原本 docstring）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image  # 彩色就轉灰階，已是灰階則直接用
    gray_blur = cv2.medianBlur(gray, 5)  # 中值模糊去雜訊，降低誤判圓的機率
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 轉回三通道彩色，方便畫上彩色圓
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, max(min_dist, 1),  # 霍夫梯度法偵測圓（圓心間最小距離至少1）
                               param1=100, param2=max(param2, 1),  # param1為Canny高門檻，param2為圓心累積門檻
                               minRadius=min_radius, maxRadius=max_radius)  # 限制偵測的最小與最大半徑
    n = 0  # 偵測到的圓數量計數器
    if circles is not None:  # 若有偵測到圓才進入處理
        circles = np.uint16(np.around(circles))  # 將圓心與半徑四捨五入並轉為整數
        n = circles.shape[1]  # 取得偵測到的圓數量
        for c in circles[0, :]:  # 逐個走訪偵測到的圓
            cv2.circle(out, (c[0], c[1]), c[2], (0, 255, 0), 2)  # 以綠色畫出圓周
            cv2.circle(out, (c[0], c[1]), 2, (0, 0, 255), 3)  # 以紅色小圓點標示圓心
    cv2.putText(out, f"Circles: {n}", (10, 25),  # 在左上角標示偵測到的圓數量
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # 字型、大小0.7、綠色、粗細2
    return out  # 回傳標註後的影像
