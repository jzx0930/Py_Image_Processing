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
import cv2
import numpy as np
import os

def process(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("❌ 無法讀取影像")
        return None

    print("\n📌 請選擇影像強度轉換模式：")
    print("1️⃣ Gamma 轉換")
    print("2️⃣ 對數轉換")
    print("3️⃣ 線性轉換")
    print("4️⃣ 對比拉伸")
    print("5️⃣ 灰階直方圖等化")
    print("6️⃣ CLAHE 自適應等化")

    mode_map = {
        1: 'gamma',
        2: 'log',
        3: 'linear',
        4: 'stretch',
        5: 'equalize',
        6: 'clahe'
    }

    try:
        mode_num = int(input("🔢 請輸入模式編號（1~6）："))
        mode = mode_map.get(mode_num)
        if mode is None:
            print("❌ 無效的選擇")
            return None
    except:
        print("❌ 輸入錯誤")
        return None

    print(f"\n📌 強度轉換模式：{mode}")
    print("⭕若確定則按下(Enter)儲存結果 ❌若按下(Esc)或關閉視窗則返回流程大廳")

    adjusted = image.copy()

    def show_preview(title, img):
        cv2.imshow(title, img)
        cv2.setWindowTitle(title, title)

    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)

    # 模式處理邏輯
    if mode == 'gamma':
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
   

    elif mode == 'log':
        print("🖱️ 調整對數轉換倍率 k（0.1 ~ 5.0）")
        k_value = [1.0]

        # 計算標準化係數 c（根據原始影像最大值）
        c = 255 / np.log(1 + np.max(image))

        def show_preview(title, img):
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 640, 480)

            # ✅ 將影像縮放至視窗大小（或你想要的比例）
            preview_resized = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
            cv2.imshow(title, preview_resized)
            cv2.setWindowTitle(title, title)

        def show_histogram(img, title="Histogram"):
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_img = np.full((150, 256), 255, dtype=np.uint8)

            cv2.normalize(hist, hist, 0, 150, cv2.NORM_MINMAX)
            for x in range(256):
                y = int(hist[x])
                cv2.line(hist_img, (x, 150), (x, 150 - y), 0, 1)

            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 640, 480)
            cv2.imshow(title, hist_img)
            cv2.setWindowTitle(title, title)

        def update(val):
            k = max(val / 10.0, 0.1)  # 將滑桿值轉換為 k（0.1 ~ 5.0）
            k_value[0] = k
            out = k * c * np.log1p(image.astype(np.float32))
            out = np.uint8(np.clip(out, 0, 255))

            preview = out.copy()
            cv2.putText(preview, f"k: {k:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            show_preview("Preview", preview)
            cv2.setWindowTitle("Preview", f"Log Transform - k={k:.1f}")
            show_histogram(out)
            cv2.waitKey(1)

        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preview", 640, 480)
        cv2.createTrackbar("k x10", "Preview", 10, 50, update)
        update(10)


    elif mode == 'linear':
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

    elif mode == 'stretch':
        print("📈 自動對比拉伸")
        min_val, max_val = np.min(image), np.max(image)
        out = (image - min_val) * 255.0 / (max_val - min_val)
        adjusted = np.uint8(out)
        show_preview("Preview", adjusted)

    elif mode == 'equalize':
        print("📊 灰階直方圖等化")
        adjusted = cv2.equalizeHist(image)
        show_preview("Preview", adjusted)

    elif mode == 'clahe':
        print("🖱️ 調整 CLAHE clipLimit（1.0 ~ 4.0）")
        clip_value = [2.0]

        def update(val):
            clip = max(val / 10.0, 0.1)  # 將滑桿值轉換為 clipLimit（0.1 ~ 4.0）
            clip_value[0] = clip
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            out = clahe.apply(image)

            preview = out.copy()
            cv2.putText(preview, f"clipLimit: {clip:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            show_preview("Preview", preview)
            cv2.setWindowTitle("Preview", f"CLAHE - clipLimit={clip:.1f}")

            cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Preview", 640, 480)  # 或你想要的大小
            cv2.createTrackbar("clipLimit x10", "Preview", 20, 40, update)  # 對應 clipLimit 2.0 ~ 4.0
        update(20)

    # 等待使用者操作
    while True:
        key = cv2.waitKey(100)
        if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
            print("⚠️ 使用者取消流程，未儲存結果")
            cv2.destroyAllWindows()
            return None
        if key != -1:
            break

    cv2.destroyAllWindows()

    # 儲存結果
    if mode == 'gamma':
        gamma = gamma_value[0]
        adjusted = np.power(image / 255.0, gamma) * 255
        adjusted = np.uint8(adjusted)
        suffix = f"gamma{gamma:.2f}"
    elif mode == 'log':
        k = k_value[0]
        out = k * c * np.log1p(image.astype(np.float32))
        adjusted = np.uint8(np.clip(out, 0, 255))
        suffix = f"log(k{k:.1f},c{c:.2f})"
    elif mode == 'linear':
        a, b = a_value[0], b_value[0]
        adjusted = np.clip(image * a + b, 0, 255).astype(np.uint8)
        suffix = f"linear(a{a:.2f},b{b})"
    elif mode == 'stretch':
        suffix = "stretch"
    elif mode == 'equalize':
        suffix = "equalized"
    elif mode == 'clahe':
        clip = clip_value[0]
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        adjusted = clahe.apply(image)
        suffix = f"clahe{clip:.1f}"

    base = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.join(os.path.dirname(image_path), "Intensity_Transform")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{base}_{suffix}.jpg")
    cv2.imwrite(save_path, adjusted)

    print(f"✅ {mode} 模式影像已儲存至：{save_path}")
    return save_path
