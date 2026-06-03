# modules/image_selector.py

import os
import cv2
import tkinter as tk
from tkinter import filedialog

class ImageSelector:
    def __init__(self, initial_path=None):
        self.initial_path = initial_path or os.getcwd()  # 預設開啟目前工作目錄

    def select_image(self):
        print("📂 請選擇影像檔案...")

        # 建立隱藏的 tkinter 視窗
        root = tk.Tk()
        root.withdraw()

        initial_folder = os.path.dirname(self.initial_path)

        file_path = filedialog.askopenfilename(
            title="選擇影像檔案",
            initialdir=initial_folder,
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )

        if not file_path or not os.path.exists(file_path):
            print("❌ 未選擇檔案或檔案無效。")
            return None, None

        image = cv2.imread(file_path)
        if image is None:
            print("❌ 無法讀取影像。")
            return None, None

        # 預覽影像
        preview = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        cv2.imshow("🔍 預覽新影像", preview)
        cv2.waitKey(0)
        cv2.destroyWindow("🔍 預覽新影像")

        return file_path, image
