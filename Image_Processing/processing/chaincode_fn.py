import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy 數值運算函式庫，並取別名 np


def chain_code_8(image, min_area=100):  # 定義八方鍊碼函式，min_area 為過濾小物件的最小面積
    """八方鍊碼 (Freeman 8-direction chain code)。

    取面積最大的物件輪廓，在影像上畫出綠色邊界、紅點起點，左上角標註鍊碼長度與前 40 碼。
    建議先「二値化」再套用本方法。
    """
    # 若輸入是彩色影像就轉成灰階，否則直接使用原灰階影像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # 以門檻 127 做二值化，得到黑白影像
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 找出最外層輪廓，CHAIN_APPROX_NONE 保留所有邊界點以利逐點計算鍊碼
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 把灰階轉回三通道，方便畫上彩色標記
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]  # 過濾掉面積小於門檻的輪廓
    if not contours:  # 若過濾後沒有任何輪廓
        cv2.putText(out, "No object", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # 標註「No object」
        return out  # 直接回傳影像
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)  # 取面積最大的輪廓並整形成 (N,2) 的座標陣列
    # 定義 8 個方向位移對應的鍊碼編號（Freeman 編碼），鍵為 (dx, dy)
    dir_map = {(1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
               (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7}
    codes = []  # 用來收集鍊碼序列的清單
    for i in range(1, len(cnt)):  # 從第二個點開始，逐點與前一點比較方向
        dx = int(np.sign(cnt[i][0] - cnt[i - 1][0]))  # 計算 x 方向變化的符號 (-1/0/1)
        dy = int(np.sign(cnt[i][1] - cnt[i - 1][1]))  # 計算 y 方向變化的符號 (-1/0/1)
        if (dx, dy) in dir_map:  # 若此位移屬於 8 個合法方向之一
            codes.append(dir_map[(dx, dy)])  # 將對應的鍊碼編號加入序列
    cv2.drawContours(out, [cnt.reshape(-1, 1, 2)], -1, (0, 255, 0), 2)  # 用綠色畫出物件輪廓邊界
    cv2.circle(out, (int(cnt[0][0]), int(cnt[0][1])), 5, (0, 0, 255), -1)  # 在輪廓起點畫紅色實心圓
    cv2.putText(out, f"Chain len: {len(codes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 標註鍊碼總長度
    seq = "".join(str(c) for c in codes[:40]) + ("..." if len(codes) > 40 else "")  # 取前 40 碼組成字串，超過則加省略號
    cv2.putText(out, seq, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)  # 把鍊碼序列字串顯示在影像上
    return out  # 回傳已標記輪廓與鍊碼資訊的影像
