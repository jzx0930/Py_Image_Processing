import cv2

def interactive_preview(window_name, image):
    """
    顯示影像並等待使用者操作。
    若使用者關閉視窗，則回傳 False。
    若使用者按下任意鍵，則回傳 True。
    """
    cv2.imshow(window_name, image)
    key = cv2.waitKey(0)
    visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
    cv2.destroyAllWindows()
    return visible >= 1
