import cv2  # 匯入 OpenCV 函式庫，用於影像處理
import os  # 匯入 os 模組，用於檔案與路徑操作
import numpy as np  # 匯入 NumPy，用於數值與矩陣運算
from modules.utils import interactive_preview

def process(image_path):  # 定義主處理函式，傳入影像路徑作為參數
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰階模式讀取影像
    if image is None:  # 檢查影像是否成功讀取
        print("錯誤：無法讀取影像")  # 顯示錯誤訊息
        return None  # 回傳 None，表示讀取失敗

    # 顯示邊緣偵測選單，讓使用者選擇處理方式
    print("\n📐 選擇邊緣偵測方法：")
    print("1. Canny（可調整門檻值）")  # Canny 邊緣偵測（支援互動式參數調整）
    print("2. Sobel")  # Sobel 邊緣偵測
    print("3. Laplacian")  # Laplacian 邊緣偵測
    print("4. Scharr")  # Scharr 邊緣偵測
    method = input("請輸入選項編號：")  # 取得使用者輸入的選項編號

    suffix = ""  # 初始化檔名後綴字串（用於儲存結果）
    result = None  # 初始化處理結果影像（尚未處理）

    if method == '1':  # 如果使用者選擇 Canny 邊緣偵測
        def update(val=0):  # 定義拖曳桿更新函式，val 是 trackbar 的回傳值（預設為 0）
            if cv2.getWindowProperty("Canny Edge", cv2.WND_PROP_VISIBLE) < 1:  # 檢查視窗是否仍存在
                return  # 若視窗已關閉則跳出，不執行更新
            try:
                low = cv2.getTrackbarPos("Low Threshold", "Canny Edge")  # 取得低門檻值
                high = cv2.getTrackbarPos("High Threshold", "Canny Edge")  # 取得高門檻值
                edges = cv2.Canny(image, low, high)  # 執行 Canny 邊緣偵測
                cv2.imshow("Canny Edge", edges)  # 顯示邊緣偵測結果影像
            except cv2.error:  # 若視窗已關閉或 trackbar 不存在
                pass  # 忽略錯誤，避免程式中斷

        window_width = max(image.shape[1], 400) + 100  # 計算視窗寬度（至少 400，加上額外空間）
        window_height = image.shape[0] + 60  # 計算視窗高度（加上額外空間）
        cv2.namedWindow("Canny Edge", cv2.WINDOW_NORMAL)  # 建立可調整大小的視窗
        cv2.resizeWindow("Canny Edge", window_width, window_height)  # 設定視窗大小
        cv2.createTrackbar("Low Threshold", "Canny Edge", 100, 255, update)  # 建立低門檻值拖曳桿，初始值 100
        cv2.createTrackbar("High Threshold", "Canny Edge", 200, 255, update)  # 建立高門檻值拖曳桿，初始值 200
        update()  # 初次呼叫 update()，顯示初始邊緣偵測結果

        print("🖱️ 使用拖曳桿調整門檻值，按任意鍵儲存結果")  # 提示使用者操作方式

        while True:  # 進入等待使用者操作的迴圈
            key = cv2.waitKey(100)  # 每 100 毫秒檢查一次鍵盤輸入

            if cv2.getWindowProperty("Canny Edge", cv2.WND_PROP_VISIBLE) < 1:  # 若視窗已被關閉
                print("⚠️ 已取消 Canny 邊緣偵測流程，請重新選擇處理模式")  # 顯示提示訊息
                break  # 跳出迴圈，結束 Canny 模式

            if key != -1:  # 若使用者按下任意鍵
                try:
                    low = cv2.getTrackbarPos("Low Threshold", "Canny Edge")  # 再次取得低門檻值
                    high = cv2.getTrackbarPos("High Threshold", "Canny Edge")  # 再次取得高門檻值
                    result = cv2.Canny(image, low, high)  # 執行 Canny 邊緣偵測，儲存結果
                    suffix = f"canny(L{low},H{high})"  # 設定檔名後綴，記錄參數值
                except cv2.error:  # 若無法取得 trackbar（視窗可能已關閉）
                    print("⚠️ 無法取得 trackbar，可能視窗已關閉")  # 顯示錯誤訊息
                break  # 跳出迴圈，準備儲存結果

        cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗，釋放資源

    elif method == '2':  # 如果使用者選擇 Sobel 邊緣偵測
            def update(val=0):  # 拖曳桿更新函式
                if cv2.getWindowProperty("Sobel Edge", cv2.WND_PROP_VISIBLE) < 1:
                    return
                try:
                    index = cv2.getTrackbarPos("Kernel Size Index", "Sobel Edge")
                    ksize = 2 * index + 3  # 將 index 0~14 映射為 ksize 3~31
                    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
                    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
                    magnitude = cv2.magnitude(sobelx, sobely)
                    preview = np.uint8(np.clip(magnitude, 0, 255))
                    cv2.imshow("Sobel Edge", preview)
                    cv2.setWindowTitle("Sobel Edge", f"Sobel Edge - ksize={ksize}")  # 顯示目前 ksize
                except cv2.error:
                    pass

            window_width = max(image.shape[1], 400) + 100
            window_height = image.shape[0] + 60
            cv2.namedWindow("Sobel Edge", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Sobel Edge", window_width, window_height)
            cv2.createTrackbar("Kernel Size Index", "Sobel Edge", 0, 14, update)  # 對應 ksize 3~31
            update()

            print("🖱️ 使用拖曳桿選擇 Sobel 核大小（奇數 3~31），按任意鍵儲存結果")

            while True:
                key = cv2.waitKey(100)
                if cv2.getWindowProperty("Sobel Edge", cv2.WND_PROP_VISIBLE) < 1:
                    print("⚠️ 已取消 Sobel 邊緣偵測流程，請重新選擇處理模式")
                    break
                if key != -1:
                    try:
                        index = cv2.getTrackbarPos("Kernel Size Index", "Sobel Edge")
                        ksize = 2 * index + 3
                        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
                        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
                        magnitude = cv2.magnitude(sobelx, sobely)
                        result = np.uint8(np.clip(magnitude, 0, 255))
                        suffix = f"sobel(K{ksize})"
                    except cv2.error:
                        print("⚠️ 無法取得 trackbar，可能視窗已關閉")
                    break

            cv2.destroyAllWindows()

    elif method == '3':  # 如果使用者選擇 Laplacian 邊緣偵測
        result = cv2.Laplacian(image, cv2.CV_64F)  # 執行 Laplacian 邊緣偵測
        result = np.uint8(np.clip(result, 0, 255))  # 將結果限制在 0~255 並轉為 8-bit 格式
        suffix = "laplacian"  # 設定儲存檔案的後綴名稱

    elif method == '4':  # 如果使用者選擇 Scharr 邊緣偵測
        scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)  # 計算 X 方向的 Scharr 梯度
        scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)  # 計算 Y 方向的 Scharr 梯度
        result = cv2.magnitude(scharrx, scharry)  # 計算梯度大小（合併 X 和 Y）
        result = np.uint8(np.clip(result, 0, 255))  # 將結果限制在 0~255 並轉為 8-bit 格式
        suffix = "scharr"  # 設定儲存檔案的後綴名稱

    else:  # 如果使用者輸入的選項無效
        print("❌ 無效選項。")  # 顯示錯誤訊息
        return None  # 回傳 None，表示處理失敗

    if result is None:  # 檢查是否有成功產生結果影像
        return None  # 若沒有結果則回傳 None


    # 顯示結果（僅限非互動式方法）
    if method in ['3', '4']:  # Laplacian 和 Scharr 沒有互動式視窗
        print("⭕若確定則按下(Enter)儲存結果 ❌若按下(Esc)則返回流程大廳")
        if not interactive_preview(f"Edge Detection - {suffix.capitalize()}", result):
            print("⚠️ 已取消流程，未儲存結果")
            return None


    # 儲存結果影像
    base_name = os.path.splitext(os.path.basename(image_path))[0]  # 取得原始檔案名稱（不含副檔名）
    save_dir = os.path.join(os.path.dirname(image_path), "Edge_Detected")  # 建立儲存資料夾路徑
    os.makedirs(save_dir, exist_ok=True)  # 建立資料夾（若已存在則略過）
    save_path = os.path.join(save_dir, f"{base_name}_{suffix}.jpg")  # 組合儲存檔案的完整路徑
    cv2.imwrite(save_path, result)  # 將處理後影像儲存到指定路徑
    print(f"✅ 邊緣偵測影像已儲存至：{save_path}")  # 顯示儲存成功訊息

    return save_path  # 回傳儲存後的影像路徑，供後續流程使用
