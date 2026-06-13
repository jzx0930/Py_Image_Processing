import os  # 用於檔案與路徑操作
import cv2  # OpenCV：影像處理函式庫
import tkinter as tk  # Tkinter：建立 GUI 元件（這裡用來開啟檔案對話框）
from tkinter import filedialog  # 檔案選擇對話框模組
from modules.workflow_manager import WorkflowManager  # ✅ 自訂的流程控制器模組，用來管理影像處理流程

# 🧭 切換工作目錄到 .py 檔案所在位置，確保相對路徑正確
os.chdir(os.path.dirname(os.path.abspath(__file__)))#是因為{}launch.json中有 "cwd": "${fileDirname}"  // 設定工作目錄為目前檔案所在資料夾

# 🔧 建立 tkinter 主視窗（不顯示），只用來開啟檔案對話框
root = tk.Tk()  # 建立 Tkinter 主視窗
root.withdraw()  # 隱藏主視窗，只顯示檔案選擇對話框

# 📂 開啟檔案選擇對話框，讓使用者選擇影像檔案
file_path = filedialog.askopenfilename(
    title='選擇影像檔案',  # 對話框標題
    filetypes=[('Image Files', '*.png;*.jpg;*.jpeg;*.bmp')]  # 限制可選檔案類型
)

# 🧾 檢查是否選擇了檔案，若未選擇則結束程式
if not file_path:
    print("未選擇任何檔案")  # 顯示提示訊息
    exit()  # 結束程式

# 📥 讀取影像（以彩色模式）
image = cv2.imread(file_path)  # 使用 OpenCV 讀取影像
if image is None:  # 檢查是否成功讀取
    print("錯誤：無法讀取影像")  # 顯示錯誤訊息
    exit()  # 結束程式

# 📁 建立儲存資料夾（若不存在），用來儲存原始影像
save_dir = "Original Image"  # 資料夾名稱
os.makedirs(save_dir, exist_ok=True)  # 建立資料夾（若已存在則略過）

# 💾 儲存原始影像到指定資料夾
base_name = os.path.splitext(os.path.basename(file_path))[0]  # 取得檔案名稱（不含副檔名）
original_path = os.path.join(save_dir, f"{base_name}_original.jpg")  # 組合儲存路徑


cv2.imwrite(original_path, image)  # 儲存影像
print(f"✅ 原始影像已儲存至：{original_path}")  # 顯示儲存成功訊息

# 🖼️ 顯示原始影像在視窗中
cv2.imshow("Original Image", image)  # 顯示影像
cv2.waitKey(0)  # 等待使用者按鍵
cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗

# 🧭 建立流程管理器，並載入原始影像路徑
manager = WorkflowManager(original_path)  # 初始化流程控制器，傳入影像路徑

# 🔁 主控流程開始，進入使用者互動迴圈
    # 顯示選單，讓使用者選擇要執行的影像處理流程
while True:
        print("\n📌(流程大廳)📌\n請選擇要執行的影像處理流程：\n-----------------------------")
        print("0. 結束程式")            # 結束整個流程
        print("1. 色彩空間分離")        # 將影像轉換為 HSV、Lab、YCrCb 等
        print("2. 二值化處理")          # 對影像進行門檻處理
        print("3. 邊緣偵測")            # 執行 Canny、Sobel 等
        print("4. 影像強度轉換")        # 對比度、Gamma、Log 等
        print("5. 影像校正")            # 幾何校正、透視變換等
        print("00. 手動指定最後有效影像路徑")            

        print("-----------------------------")  # 印出分隔線，讓選單更清楚
        choice = input("請輸入選項編號：")  # 取得使用者輸入的選項編號

        if choice == '0':  # 若輸入 0
            print("🚪 程式結束。")  # 顯示結束訊息
            break  # 跳出主迴圈，結束程式

        if choice == '00':  # 若輸入 00（手動指定影像）
            success = manager.manual_select_image()  # 呼叫手動選圖功能，回傳是否成功
        else:  # 其他選項
            success = manager.run_step(choice)  # 依選項執行對應的影像處理步驟

        success = manager.run_step(choice)  # 再次執行該步驟並更新成功狀態

        if not success:  # 若處理未成功（取消或失敗）
            continue  # 回到迴圈開頭，重新顯示選單

        print(f"✅ 已完成處理，目前影像路徑：{manager.current_image_path}")  # 顯示處理完成與目前影像路徑
