import cv2  # 匯入 OpenCV 函式庫，用於影像處理
import numpy as np  # 匯入 NumPy，用於數值與矩陣運算
import os  # 匯入 os 模組，用於檔案與路徑操作

def process(image_path):  # 定義影像校正（仿射變換）的主處理函式，傳入影像路徑
    image = cv2.imread(image_path)  # 以彩色模式讀取影像
    if image is None:  # 檢查是否成功讀取
        print("❌ 無法讀取影像")  # 顯示錯誤訊息
        return None  # 回傳 None 表示失敗

    h, w = image.shape[:2]  # 取得影像的高度與寬度

    def update(val):  # 滑桿變動時呼叫的更新函式，val 為滑桿值
        dx = cv2.getTrackbarPos("X Offset", "Affine Preview")  # 取得 X 方向偏移量
        dy = cv2.getTrackbarPos("Y Offset", "Affine Preview")  # 取得 Y 方向偏移量

        src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])  # 設定變換前的三個基準點（左上、右上、左下）
        dst = np.float32([[dx, dy], [w - dx, dy], [dx, h - dy]])  # 設定變換後的對應三點（依偏移量位移）
        matrix = cv2.getAffineTransform(src, dst)  # 計算仿射變換矩陣
        warped = cv2.warpAffine(image, matrix, (w, h))  # 套用仿射變換得到結果影像
        cv2.imshow("Affine Preview", warped)  # 顯示預覽影像

    cv2.namedWindow("Affine Preview")  # 建立名為 Affine Preview 的視窗
    cv2.createTrackbar("X Offset", "Affine Preview", 30, 100, update)  # 建立 X 偏移滑桿，初始值 30
    cv2.createTrackbar("Y Offset", "Affine Preview", 30, 100, update)  # 建立 Y 偏移滑桿，初始值 30
    update(0)  # 初始化預覽畫面

    key = cv2.waitKey(0)  # 等待使用者按鍵
    cv2.destroyAllWindows()  # 關閉所有視窗

    if key == 27:  # 若按下 ESC
        print("⚠️ 使用者取消操作")  # 顯示取消訊息
        return None  # 回傳 None 表示取消

    dx = cv2.getTrackbarPos("X Offset", "Affine Preview")  # 再次取得最終 X 偏移量
    dy = cv2.getTrackbarPos("Y Offset", "Affine Preview")  # 再次取得最終 Y 偏移量

    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])  # 設定變換前的三個基準點
    dst = np.float32([[dx, dy], [w - dx, dy], [dx, h - dy]])  # 設定變換後的對應三點
    matrix = cv2.getAffineTransform(src, dst)  # 計算仿射變換矩陣
    warped = cv2.warpAffine(image, matrix, (w, h))  # 套用仿射變換得到最終影像

    base = os.path.splitext(os.path.basename(image_path))[0]  # 取得原始檔名（不含副檔名）
    save_dir = os.path.join(os.path.dirname(image_path), "Image_Correction")  # 組合儲存資料夾路徑
    os.makedirs(save_dir, exist_ok=True)  # 建立資料夾（若已存在則略過）
    save_path = os.path.join(save_dir, f"{base}_affine_dx{dx}_dy{dy}.jpg")  # 組合儲存檔案路徑（含偏移參數）
    cv2.imwrite(save_path, warped)  # 儲存校正後的影像

    print(f"✅ 仿射校正影像已儲存至：{save_path}")  # 顯示儲存成功訊息
    return save_path  # 回傳儲存後的影像路徑

