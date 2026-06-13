# import cv2
# import numpy as np
# import os

# def process(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         print("❌ 無法讀取影像")
#         return None

#     print("\n📌 Gamma 強度轉換")
#     print("🖱️ 請拖曳滑桿調整 Gamma 值（0.01 ~ 3.00）")
#     print("⭕若確定則按下(Enter)儲存結果 ❌若按下(Esc)或關閉視窗則返回流程大廳")

#     gamma_value = [1.0]

#     def update(val):
#         gamma = val / 100.0
#         gamma = max(gamma, 0.01)
#         gamma_value[0] = gamma
#         adjusted = np.power(image / 255.0, gamma) * 255
#         adjusted = np.uint8(adjusted)
#         cv2.imshow("Gamma Preview", adjusted)
#         cv2.setWindowTitle("Gamma Preview", f"Gamma Preview - γ={gamma:.2f}")

#     cv2.namedWindow("Gamma Preview", cv2.WINDOW_NORMAL)
#     cv2.createTrackbar("Gamma x100", "Gamma Preview", 100, 300, update)
#     update(100)

#     while True:
#         key = cv2.waitKey(100)
#         if key == 27 or cv2.getWindowProperty("Gamma Preview", cv2.WND_PROP_VISIBLE) < 1:
#             print("⚠️ 使用者取消流程，未儲存結果")
#             cv2.destroyAllWindows()
#             return None
#         if key != -1:
#             break

#     cv2.destroyAllWindows()

#     gamma = gamma_value[0]
#     adjusted = np.power(image / 255.0, gamma) * 255
#     adjusted = np.uint8(adjusted)

#     base = os.path.splitext(os.path.basename(image_path))[0]
#     save_dir = os.path.join(os.path.dirname(image_path), "Intensity_Transform")
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"{base}_gamma{gamma:.2f}.jpg")
#     cv2.imwrite(save_path, adjusted)

#     print(f"✅ Gamma 調整影像已儲存至：{save_path}")
#     return save_path
import cv2  # 匯入 OpenCV 函式庫，用於影像處理
import numpy as np  # 匯入 NumPy，用於數值與矩陣運算
import os  # 匯入 os 模組，用於檔案與路徑操作

def process(image_path):  # 定義影像強度轉換的主處理函式，傳入影像路徑
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰階模式讀取影像
    if image is None:  # 檢查是否成功讀取
        print("❌ 無法讀取影像")  # 顯示錯誤訊息
        return None  # 回傳 None 表示失敗

    print("\n📌 請選擇影像強度轉換模式：")  # 顯示功能標題
    print("1️⃣ Gamma 轉換")  # 選項 1：Gamma 轉換
    print("2️⃣ 對數轉換")  # 選項 2：對數轉換
    print("3️⃣ 線性轉換")  # 選項 3：線性轉換
    print("4️⃣ 對比拉伸")  # 選項 4：對比拉伸
    print("5️⃣ 灰階直方圖等化")  # 選項 5：直方圖等化
    print("6️⃣ CLAHE 自適應等化")  # 選項 6：CLAHE 自適應等化

    mode_map = {  # 建立選項編號對應到模式名稱的字典
        1: 'gamma',  # 1 對應 Gamma
        2: 'log',  # 2 對應對數
        3: 'linear',  # 3 對應線性
        4: 'stretch',  # 4 對應對比拉伸
        5: 'equalize',  # 5 對應直方圖等化
        6: 'clahe'  # 6 對應 CLAHE
    }

    try:  # 嘗試讀取並解析使用者輸入
        mode_num = int(input("🔢 請輸入模式編號（1~6）："))  # 取得使用者輸入並轉為整數
        mode = mode_map.get(mode_num)  # 依編號取出對應的模式名稱
        if mode is None:  # 若編號不在字典中
            print("❌ 無效的選擇")  # 顯示錯誤訊息
            return None  # 回傳 None 表示失敗
    except:  # 若輸入無法轉成整數等錯誤
        print("❌ 輸入錯誤")  # 顯示錯誤訊息
        return None  # 回傳 None 表示失敗

    print(f"\n📌 強度轉換模式：{mode}")  # 顯示目前選到的模式
    print("⭕若確定則按下(Enter)儲存結果 ❌若按下(Esc)或關閉視窗則返回流程大廳")  # 操作提示

    adjusted = image.copy()  # 先複製原影像作為調整結果的初始值

    def show_preview(title, img):  # 定義顯示預覽影像的小工具函式
        cv2.imshow(title, img)  # 顯示影像
        cv2.setWindowTitle(title, title)  # 設定視窗標題

    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)  # 建立可調整大小的 Preview 視窗

    # 模式處理邏輯
    if mode == 'gamma':  # 若選擇 Gamma 轉換模式
        print("🖱️ 請拖曳滑桿調整 Gamma 值（0.01 ~ 3.00）")  # 提示使用者可調整 gamma 值

        gamma_value = [1.0]  # 儲存目前的 gamma 值（使用 list 以便在內部函式中修改）

        def update(val):
            gamma = max(val / 100.0, 0.01)  # 將滑桿值轉換為 gamma，避免為 0
            gamma_value[0] = gamma  # 更新目前的 gamma 值
            out = np.power(image / 255.0, gamma) * 255  # 套用 gamma 轉換公式
            out = np.uint8(out)  # 轉換為 uint8 格式以供顯示
            show_preview("Preview", out)  # 顯示預覽影像

        cv2.createTrackbar("Gamma x100", "Preview", 100, 300, update)  # 建立滑桿，範圍 100~300 對應 gamma 0.01~3.00
        update(100)  # 初始化預覽畫面，使用 gamma=1.0
   

    elif mode == 'log':  # 若選擇對數轉換模式
        print("🖱️ 調整對數轉換倍率 k（0.1 ~ 5.0）")  # 操作提示
        k_value = [1.0]  # 以 list 儲存目前的倍率 k（方便內部函式修改）

        # 計算標準化係數 c（根據原始影像最大值）
        c = 255 / np.log(1 + np.max(image))  # 讓轉換後的最大值落在 255 附近

        def show_preview(title, img):  # 定義對數模式專用的預覽顯示函式
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # 建立可調整大小的視窗
            cv2.resizeWindow(title, 640, 480)  # 設定視窗大小

            # ✅ 將影像縮放至視窗大小（或你想要的比例）
            preview_resized = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)  # 縮放影像供顯示
            cv2.imshow(title, preview_resized)  # 顯示縮放後的影像
            cv2.setWindowTitle(title, title)  # 設定視窗標題

        def show_histogram(img, title="Histogram"):  # 定義顯示直方圖的函式
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # 計算灰階值 0~255 的分佈
            hist_img = np.full((150, 256), 255, dtype=np.uint8)  # 建立白色底圖作為直方圖畫布

            cv2.normalize(hist, hist, 0, 150, cv2.NORM_MINMAX)  # 將直方圖高度正規化到 0~150
            for x in range(256):  # 逐一處理每個灰階值
                y = int(hist[x])  # 取得該灰階值對應的高度
                cv2.line(hist_img, (x, 150), (x, 150 - y), 0, 1)  # 由底部往上畫出柱狀線

            cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # 建立可調整大小的直方圖視窗
            cv2.resizeWindow(title, 640, 480)  # 設定視窗大小
            cv2.imshow(title, hist_img)  # 顯示直方圖
            cv2.setWindowTitle(title, title)  # 設定視窗標題

        def update(val):  # 滑桿變動時呼叫的更新函式
            k = max(val / 10.0, 0.1)  # 將滑桿值轉換為 k（0.1 ~ 5.0）
            k_value[0] = k  # 更新目前的 k 值
            out = k * c * np.log1p(image.astype(np.float32))  # 套用對數轉換公式
            out = np.uint8(np.clip(out, 0, 255))  # 限制在 0~255 並轉為 8-bit

            preview = out.copy()  # 複製一份作為預覽
            cv2.putText(preview, f"k: {k:.1f}", (10, 30),  # 在預覽圖上標註目前 k 值
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  # 設定字型、大小、顏色與粗細
            show_preview("Preview", preview)  # 顯示預覽影像
            cv2.setWindowTitle("Preview", f"Log Transform - k={k:.1f}")  # 更新視窗標題顯示 k 值
            show_histogram(out)  # 顯示轉換後影像的直方圖
            cv2.waitKey(1)  # 短暫等待以更新畫面

        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)  # 建立可調整大小的 Preview 視窗
        cv2.resizeWindow("Preview", 640, 480)  # 設定視窗大小
        cv2.createTrackbar("k x10", "Preview", 10, 50, update)  # 建立 k 滑桿（10~50 對應 1.0~5.0）
        update(10)  # 初始化預覽畫面，k=1.0


    elif mode == 'linear':  # 若選擇線性轉換模式
        print("🖱️ 調整對比 a（50~300%）與亮度 b（-100~100）")  # 提示使用者可調整對比與亮度
        a_value = [1.0]  # 儲存目前的對比係數 a
        b_value = [0]    # 儲存目前的亮度偏移 b

        def update(_):
            a = cv2.getTrackbarPos("a%", "Preview") / 100.0  # 取得滑桿值並轉換為 a（0.5~3.0）
            b = cv2.getTrackbarPos("b", "Preview") - 100   # 取得滑桿值並轉換為 b（-100~100）
            a_value[0], b_value[0] = a, b  # 更新目前參數
            out = np.clip(image * a + b, 0, 255)  # 套用線性轉換公式
            show_preview("Preview", np.uint8(out))  # 顯示預覽影像
            cv2.setWindowTitle("Preview", f"Linear Transform - contrast={a:.2f}, brightness={b}")  # ✅ 即時更新視窗標題

            cv2.createTrackbar("a%", "Preview", 100, 300, update)  # 建立對比滑桿（100~300 對應 a=1.0~3.0）
            cv2.createTrackbar("b", "Preview", 100, 200, update)  # 建立亮度滑桿（100~200 對應 b=-100~100）
        update(0)  # 初始化預覽畫面

    elif mode == 'stretch':  # 若選擇對比拉伸模式
        print("📈 自動對比拉伸")  # 顯示提示訊息
        min_val, max_val = np.min(image), np.max(image)  # 取得影像的最小與最大灰階值
        out = (image - min_val) * 255.0 / (max_val - min_val)  # 將灰階範圍線性拉伸到 0~255
        adjusted = np.uint8(out)  # 轉換為 8-bit 格式
        show_preview("Preview", adjusted)  # 顯示預覽影像

    elif mode == 'equalize':  # 若選擇直方圖等化模式
        print("📊 灰階直方圖等化")  # 顯示提示訊息
        adjusted = cv2.equalizeHist(image)  # 套用直方圖等化，增強整體對比
        show_preview("Preview", adjusted)  # 顯示預覽影像

    elif mode == 'clahe':  # 若選擇 CLAHE 自適應等化模式
        print("🖱️ 調整 CLAHE clipLimit（1.0 ~ 4.0）")  # 操作提示
        clip_value = [2.0]  # 以 list 儲存目前的 clipLimit 值

        def update(val):  # 滑桿變動時呼叫的更新函式
            clip = max(val / 10.0, 0.1)  # 將滑桿值轉換為 clipLimit（0.1 ~ 4.0）
            clip_value[0] = clip  # 更新目前的 clipLimit 值
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))  # 建立 CLAHE 物件，設定區塊大小
            out = clahe.apply(image)  # 套用 CLAHE 等化

            preview = out.copy()  # 複製一份作為預覽
            cv2.putText(preview, f"clipLimit: {clip:.1f}", (10, 30),  # 在預覽圖上標註 clipLimit
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  # 設定字型、大小、顏色與粗細
            show_preview("Preview", preview)  # 顯示預覽影像
            cv2.setWindowTitle("Preview", f"CLAHE - clipLimit={clip:.1f}")  # 更新視窗標題

            cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)  # 建立可調整大小的視窗
            cv2.resizeWindow("Preview", 640, 480)  # 或你想要的大小
            cv2.createTrackbar("clipLimit x10", "Preview", 20, 40, update)  # 對應 clipLimit 2.0 ~ 4.0
        update(20)  # 初始化預覽畫面，clipLimit=2.0

    # 等待使用者操作
    while True:  # 進入等待使用者確認或取消的迴圈
        key = cv2.waitKey(100)  # 每 100 毫秒檢查一次鍵盤
        if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:  # 若按 ESC 或關閉視窗
            print("⚠️ 使用者取消流程，未儲存結果")  # 顯示取消訊息
            cv2.destroyAllWindows()  # 關閉所有視窗
            return None  # 回傳 None 表示取消
        if key != -1:  # 若按下任意其他鍵
            break  # 跳出迴圈，確認儲存

    cv2.destroyAllWindows()  # 關閉所有視窗

    # 儲存結果
    if mode == 'gamma':  # 若是 Gamma 模式，依最後參數重新計算結果
        gamma = gamma_value[0]  # 取得最後的 gamma 值
        adjusted = np.power(image / 255.0, gamma) * 255  # 套用 gamma 轉換公式
        adjusted = np.uint8(adjusted)  # 轉換為 8-bit 格式
        suffix = f"gamma{gamma:.2f}"  # 設定檔名後綴
    elif mode == 'log':  # 若是對數模式
        k = k_value[0]  # 取得最後的 k 值
        out = k * c * np.log1p(image.astype(np.float32))  # 套用對數轉換公式
        adjusted = np.uint8(np.clip(out, 0, 255))  # 限制在 0~255 並轉為 8-bit
        suffix = f"log(k{k:.1f},c{c:.2f})"  # 設定檔名後綴
    elif mode == 'linear':  # 若是線性模式
        a, b = a_value[0], b_value[0]  # 取得最後的對比 a 與亮度 b
        adjusted = np.clip(image * a + b, 0, 255).astype(np.uint8)  # 套用線性轉換並限制範圍
        suffix = f"linear(a{a:.2f},b{b})"  # 設定檔名後綴
    elif mode == 'stretch':  # 若是對比拉伸模式
        suffix = "stretch"  # 設定檔名後綴（結果已於前面算好）
    elif mode == 'equalize':  # 若是直方圖等化模式
        suffix = "equalized"  # 設定檔名後綴（結果已於前面算好）
    elif mode == 'clahe':  # 若是 CLAHE 模式
        clip = clip_value[0]  # 取得最後的 clipLimit
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))  # 建立 CLAHE 物件
        adjusted = clahe.apply(image)  # 套用 CLAHE 等化
        suffix = f"clahe{clip:.1f}"  # 設定檔名後綴

    base = os.path.splitext(os.path.basename(image_path))[0]  # 取得原始檔名（不含副檔名）
    save_dir = os.path.join(os.path.dirname(image_path), "Intensity_Transform")  # 組合儲存資料夾路徑
    os.makedirs(save_dir, exist_ok=True)  # 建立資料夾（若已存在則略過）
    save_path = os.path.join(save_dir, f"{base}_{suffix}.jpg")  # 組合儲存檔案路徑
    cv2.imwrite(save_path, adjusted)  # 儲存處理後的影像

    print(f"✅ {mode} 模式影像已儲存至：{save_path}")  # 顯示儲存成功訊息
    return save_path  # 回傳儲存後的影像路徑
