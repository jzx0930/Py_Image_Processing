import cv2  # 匯入 OpenCV 函式庫
import os  # 匯入 os 模組，用於路徑與檔案操作
from modules.utils import interactive_preview


def process(image_path):  # 定義二值化處理函式
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰階模式讀取影像
    if image is None:  # 檢查是否成功讀取
        print("錯誤：無法讀取影像")
        return None  # 回傳 None，表示失敗

    # 顯示二值化選單
    print("📌二值化處理")
    print("選擇二值化模式：(輸入後按下Enter)")
    print("1. 自訂門檻值")
    print("2. Otsu 自動門檻")
    print("3. 自適應二值化")
    mode = input("請輸入選項編號：")  # 取得使用者輸入

    binary = None  # 初始化結果影像
    suffix = ""  # 初始化檔名後綴

    if mode == '1':  # 自訂門檻值
        try:
            thresh_val = int(input("請輸入門檻值（0~255）："))  # 取得使用者輸入的門檻值
            _, binary = cv2.threshold(image, thresh_val, 255, cv2.THRESH_BINARY)  # 執行二值化
            suffix = f"threshold({thresh_val})"  # 設定檔名後綴
        except ValueError:
            print("❌ 非法輸入，請輸入數字")
            return None

    elif mode == '2':  # Otsu 自動門檻
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 執行 Otsu 二值化
        suffix = "otsu"

    elif mode == '3':  # 自適應二值化（互動式）
        def update(val=0):
            if cv2.getWindowProperty("Adaptive Threshold", cv2.WND_PROP_VISIBLE) < 1:
                return
            try:
                index = cv2.getTrackbarPos("BlockSize Index", "Adaptive Threshold")
                blockSize = 2 * index + 3  # index 0~249 → blockSize 3~501
                c = cv2.getTrackbarPos("C", "Adaptive Threshold") - 50  # C 可為負數

                binary = cv2.adaptiveThreshold(image, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,
                                            blockSize=blockSize,
                                            C=c)
                preview = binary.copy()
                cv2.putText(preview, f"blockSize={blockSize}, C={c}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                cv2.imshow("Adaptive Threshold", preview)
                cv2.setWindowTitle("Adaptive Threshold", f"Adaptive Threshold - blockSize={blockSize}, C={c}")
            except cv2.error:
                pass

        window_width = max(image.shape[1], 400) + 100
        window_height = image.shape[0] + 60
        cv2.namedWindow("Adaptive Threshold", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Adaptive Threshold", window_width, window_height)
        cv2.createTrackbar("BlockSize Index", "Adaptive Threshold", 4, 249, update)  # 預設 blockSize=11
        cv2.createTrackbar("C", "Adaptive Threshold", 52, 100, update)  # 預設 C=2（滑桿值 52）
        update()

        print("🖱️ 使用拖曳桿調整 blockSize（奇數 3~31）與 C（偏移值 -50~50），⭕若確定則按下(Enter)儲存結果 ❌若按下(Esc)則返回流程大廳")

        while True:
            key = cv2.waitKey(100)
            if cv2.getWindowProperty("Adaptive Threshold", cv2.WND_PROP_VISIBLE) < 1:
                print("⚠️ 已取消自適應二值化流程，請重新選擇處理模式")
                break
            if key != -1:
                try:
                    index = cv2.getTrackbarPos("BlockSize Index", "Adaptive Threshold")
                    blockSize = 2 * index + 3
                    c = cv2.getTrackbarPos("C", "Adaptive Threshold") - 50
                    binary = cv2.adaptiveThreshold(image, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY,
                                                blockSize=blockSize,
                                                C=c)
                    suffix = f"adaptive(K{blockSize},C{c})"
                except cv2.error:
                    print("⚠️ 無法取得 trackbar，可能視窗已關閉")
                break

        cv2.destroyAllWindows()


    else:  # 無效選項
        print("❌ 無效選項。")
        return None

    # 模式 1 和 2 需要確認預覽
    if mode in ['1', '2']:
        print("⭕若確定則按下(Enter)儲存結果 ❌若按下(Esc)則返回流程大廳")
        if not interactive_preview("Binarized Image", binary):
            print("⚠️ 已取消二值化流程，未儲存結果")
            return None


    # 儲存結果影像
    base_name = os.path.splitext(os.path.basename(image_path))[0]  # 取得原始檔名（不含副檔名）
    save_dir = os.path.join(os.path.dirname(image_path), "Binarized")  # 建立儲存資料夾路徑
    os.makedirs(save_dir, exist_ok=True)  # 建立資料夾（若已存在則略過）
    save_path = os.path.join(save_dir, f"{base_name}_{suffix}.jpg")  # 組合儲存檔案路徑
    cv2.imwrite(save_path, binary)  # 儲存影像
    print(f"✅ 二值化影像已儲存至：{save_path}")  # 顯示儲存成功訊息

    return save_path  # 回傳儲存後的影像路徑
