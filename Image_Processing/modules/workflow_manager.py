from modules.image_selector import ImageSelector

class WorkflowManager:  # 建立流程管理器類別，用來控制影像處理流程與狀態
    def __init__(self, original_image_path):  # 初始化方法，傳入原始影像路徑
        self.original_image_path = original_image_path  # 儲存原始影像路徑（不變）
        self.last_valid_image_path = original_image_path  # 儲存最後成功處理的影像路徑（可回退）
        self.current_image_path = original_image_path  # 儲存目前正在處理的影像路徑（可更新）

    def run_step(self, method):  # 執行一個處理步驟，根據使用者選擇的 method
        result_path = self.dispatch_process(method, self.current_image_path)  # 呼叫對應的處理函式，傳入目前影像

        if result_path is None:  # 若處理失敗或使用者取消
            print("⚠️ 使用者取消流程，回到主選單")  # 顯示提示訊息
            self.current_image_path = self.last_valid_image_path  # 回復到上一張有效影像
            return False  # 回傳 False，表示流程未成功

        self.current_image_path = result_path  # 更新目前影像路徑為處理結果
        self.last_valid_image_path = result_path  # 同時更新最後有效影像路徑
        return True  # 回傳 True，表示流程成功

    def dispatch_process(self, method, image_path):
        if method == '1':
            from color_space import process
            return process(image_path)
        elif method == '2':
            from binarize import process
            return process(image_path)
        elif method == '3':
            from edge_detect import process
            return process(image_path)
        elif method == '4':
            from intensity_transform import process
            return process(image_path)
        elif method == '5':
            from image_correction import process
            return process(image_path)
        else:
            print("❌ 無效選項")
            return None

    def manual_select_image(self):
        selector = ImageSelector(self.last_valid_image_path)
        file_path, image = selector.select_image()

        if file_path and image is not None:
            self.last_valid_image_path = file_path
            self.current_image_path = file_path
            print(f"✅ 已更新影像路徑為：{file_path}")
            return True
        else:
            print("⚠️ 未更新影像路徑。")
            return False



    # import tkinter as tk
    # from tkinter import filedialog


    # def dispatch_process(self, method, image_path):
    #     if method == '1':
    #         from color_space import process
    #         return process(image_path)
    #     elif method == '2':
    #         from binarize import process
    #         return process(image_path)
    #     elif method == '3':
    #         from edge_detect import process
    #         return process(image_path)
    #     elif method == '4':
    #         from intensity_transform import process
    #         return process(image_path)
    #     elif method == '5':
    #         from image_correction import process
    #         return process(image_path)
    #     elif method == '00':
    #         print("📂 請選擇影像檔案作為新的處理目標...")

    #         import tkinter as tk
    #         from tkinter import filedialog
    #         import os

    #         # 建立隱藏的 tkinter 視窗
    #         root = tk.Tk()
    #         root.withdraw()

    #         # # 跳出檔案選擇視窗
    #         # file_path = filedialog.askopenfilename(
    #         #     title="選擇影像檔案",
    #         #     filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
    #         # )

    #         initial_folder = image_path

    #         file_path = filedialog.askopenfilename(
    #             title="選擇影像檔案",
    #             initialdir=initial_folder,  # ✅ 指定預設開啟資料夾
    #             filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
    #         )

    #         if file_path and os.path.exists(file_path):
    #             self.current_image_path = file_path
    #             self.image = cv2.imread(file_path)
    #             print(f"✅ 已更新影像路徑為：{file_path}")

    #             # 預覽影像
    #             preview = cv2.resize(self.image, (640, 480), interpolation=cv2.INTER_AREA)
    #             import cv2  # OpenCV：影像處理函式庫
    #             cv2.imshow("🔍 預覽新影像", preview)
    #             cv2.waitKey(0)
    #             cv2.destroyWindow("🔍 預覽新影像")

    #             return self.image
    #         else:
    #             print("❌ 未選擇檔案或檔案無效。")
    #             return None
    #     else:
    #         print("❌ 無效選項")
    #         return None


