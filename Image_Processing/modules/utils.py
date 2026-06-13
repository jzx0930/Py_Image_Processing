import cv2  # 匯入 OpenCV 函式庫，用於顯示影像與處理視窗

def interactive_preview(window_name, image):  # 定義互動式預覽函式，傳入視窗名稱與影像
    """
    顯示影像並等待使用者操作。
    若使用者關閉視窗，則回傳 False。
    若使用者按下任意鍵，則回傳 True。
    """
    cv2.imshow(window_name, image)  # 在指定名稱的視窗中顯示影像
    key = cv2.waitKey(0)  # 等待使用者按鍵（會一直停在這裡直到有按鍵）
    visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)  # 取得視窗是否仍可見的狀態
    cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗
    return visible >= 1  # 視窗仍存在（未被關閉）回傳 True，否則回傳 False
