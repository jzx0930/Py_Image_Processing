# modules/image_selector.py  # 本檔案：影像選擇器，負責讓使用者用對話框挑選影像

import os  # 匯入 os 模組，用於路徑與檔案存在性檢查
import cv2  # 匯入 OpenCV 函式庫，用於讀取與顯示影像
import tkinter as tk  # 匯入 Tkinter，用於建立檔案對話框的視窗
from tkinter import filedialog  # 匯入檔案選擇對話框模組

class ImageSelector:  # 定義影像選擇器類別
    def __init__(self, initial_path=None):  # 初始化方法，可傳入預設的起始路徑
        self.initial_path = initial_path or os.getcwd()  # 預設開啟目前工作目錄

    def select_image(self):  # 開啟對話框讓使用者選擇影像並回傳結果
        print("📂 請選擇影像檔案...")  # 顯示提示訊息

        # 建立隱藏的 tkinter 視窗
        root = tk.Tk()  # 建立 Tkinter 主視窗
        root.withdraw()  # 隱藏主視窗，只顯示檔案對話框

        initial_folder = os.path.dirname(self.initial_path)  # 取得起始路徑所在的資料夾

        file_path = filedialog.askopenfilename(  # 開啟檔案選擇對話框，回傳所選路徑
            title="選擇影像檔案",  # 對話框標題
            initialdir=initial_folder,  # 預設開啟的資料夾
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]  # 限制可選的影像格式
        )

        if not file_path or not os.path.exists(file_path):  # 若未選檔或檔案不存在
            print("❌ 未選擇檔案或檔案無效。")  # 顯示錯誤訊息
            return None, None  # 回傳兩個 None 表示失敗

        image = cv2.imread(file_path)  # 讀取所選的影像
        if image is None:  # 若影像無法讀取
            print("❌ 無法讀取影像。")  # 顯示錯誤訊息
            return None, None  # 回傳兩個 None 表示失敗

        # 預覽影像
        preview = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)  # 縮放影像供預覽
        cv2.imshow("🔍 預覽新影像", preview)  # 顯示預覽視窗
        cv2.waitKey(0)  # 等待使用者按任意鍵
        cv2.destroyWindow("🔍 預覽新影像")  # 關閉預覽視窗

        return file_path, image  # 回傳所選的路徑與影像
