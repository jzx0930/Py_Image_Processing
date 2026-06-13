import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy，用來處理比對結果陣列


def template_match_center(image, region_pct=20, threshold=80, x_pct=50, y_pct=50):  # 模板比對函式
    """模板比對：在指定位置（x_pct, y_pct）選取模板區域（紅框），在整張影像中找出相似區塊（綠框）。

    region_pct: 模板大小（佔影像百分比）。threshold: 相似度門檻(%)。
    x_pct/y_pct: 模板中心的 X/Y 位置（佔影像百分比，50 = 中央）。
    """  # 函式說明（保留原本 docstring）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image  # 彩色就轉灰階，已是灰階則直接用
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 轉回三通道彩色，方便畫上彩色框
    h, w = gray.shape  # 取得影像的高與寬
    tw = max(8, w * region_pct // 100)  # 依百分比計算模板寬度（最少8像素）
    th = max(8, h * region_pct // 100)  # 依百分比計算模板高度（最少8像素）
    cx = int(w * x_pct / 100)  # 依百分比計算模板中心的X座標
    cy = int(h * y_pct / 100)  # 依百分比計算模板中心的Y座標
    x0 = max(0, min(cx - tw // 2, w - tw))  # 由中心算出模板左上X，並夾在影像範圍內
    y0 = max(0, min(cy - th // 2, h - th))  # 由中心算出模板左上Y，並夾在影像範圍內
    template = gray[y0:y0 + th, x0:x0 + tw]  # 從灰階影像切出模板區塊
    if template.shape[0] < 4 or template.shape[1] < 4 or template.shape[0] >= h or template.shape[1] >= w:  # 模板過小或過大都無法比對
        return out  # 直接回傳未標註影像
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)  # 以正規化相關係數法做模板比對，得到相似度地圖
    loc = np.where(res >= threshold / 100.0)  # 找出相似度大於等於門檻的所有位置
    count = 0  # 符合門檻的匹配數量計數器
    for pt in zip(*loc[::-1]):  # 將座標反轉成(x, y)並逐點走訪
        cv2.rectangle(out, pt, (pt[0] + tw, pt[1] + th), (0, 255, 0), 1)  # 以綠色框標出匹配到的區塊
        count += 1  # 匹配數加一
    cv2.rectangle(out, (x0, y0), (x0 + tw, y0 + th), (0, 0, 255), 2)  # 以紅色框標出原始模板位置
    cv2.putText(out, f"Matches: {count}", (10, 25),  # 在左上角顯示匹配總數
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # 字型、大小0.7、藍色、粗細2
    return out  # 回傳標註後的影像
