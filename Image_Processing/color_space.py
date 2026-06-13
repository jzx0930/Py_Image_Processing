

import cv2  # 匯入 OpenCV 函式庫，用於影像處理
import os  # 匯入 os 模組，用於檔案與路徑操作

def process(image_path):  # 定義色彩空間分離的主處理函式，傳入影像路徑
    image = cv2.imread(image_path)  # 以彩色模式讀取影像
    if image is None:  # 檢查是否成功讀取
        print("❌ 無法讀取影像")  # 顯示錯誤訊息
        return None  # 回傳 None 表示失敗

    color_spaces = [  # 定義所有可選的色彩空間（名稱、轉換代碼、各通道標籤）
        ("BGR", None, ["B", "G", "R"]),  # BGR 為原始格式，不需轉換
        ("HSV", cv2.COLOR_BGR2HSV, ["H", "S", "V"]),  # 色相、飽和度、明度
        ("Lab", cv2.COLOR_BGR2Lab, ["L", "a", "b"]),  # 亮度與兩個色彩分量
        ("YCrCb", cv2.COLOR_BGR2YCrCb, ["Y", "Cr", "Cb"]),  # 亮度與色差分量
        ("HLS", cv2.COLOR_BGR2HLS, ["H", "L", "S"]),  # 色相、亮度、飽和度
        ("LUV", cv2.COLOR_BGR2LUV, ["L", "U", "V"]),  # LUV 色彩空間
        ("XYZ", cv2.COLOR_BGR2XYZ, ["X", "Y", "Z"]),  # XYZ 色彩空間
        ("YUV", cv2.COLOR_BGR2YUV, ["Y", "U", "V"]),  # YUV 色彩空間
        ("GRAY", cv2.COLOR_BGR2GRAY, ["Gray"]),  # 灰階（只有單一通道）
        ("RGB", cv2.COLOR_BGR2RGB, ["R", "G", "B"])  # RGB 色彩空間
    ]

    selected_index = [0]  # 以 list 儲存目前選到的色彩空間索引（方便內部函式修改）

    def update(val):  # 滑桿變動時呼叫的更新函式，val 為滑桿值
        selected_index[0] = val  # 記錄目前選到的索引
        name, code, _ = color_spaces[val]  # 取出名稱與轉換代碼（忽略通道標籤）
        converted = image.copy() if code is None else cv2.cvtColor(image, code)  # 依代碼轉換色彩空間，BGR 則直接複製
        preview = converted.copy()  # 複製一份作為預覽影像
        cv2.putText(preview, f"Color Mode: {name}", (10, 30),  # 在預覽圖上標註目前色彩模式名稱
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # 設定字型、大小、顏色與粗細
        cv2.imshow("Preview", preview)  # 顯示預覽影像

    cv2.namedWindow("Preview")  # 建立名為 Preview 的視窗
    cv2.createTrackbar("Color Mode", "Preview", 0, len(color_spaces) - 1, update)  # 建立滑桿，範圍涵蓋所有色彩空間

    print("📌色彩空間分離")  # 顯示功能標題
    print("拖曳滑桿選擇要分離的色彩空間")  # 操作提示
    print("停留於待分離之色彩空間，請按下任意鍵繼續（或按 ESC 取消返回流程大廳)")  # 操作提示
    update(0)  # 初始化預覽，顯示第一個色彩空間

    while True:  # 進入等待使用者選擇色彩空間的迴圈
        key = cv2.waitKey(100)  # 每 100 毫秒檢查一次鍵盤
        if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:  # 若按 ESC 或視窗被關閉
            print("⚠️ 使用者取消或關閉視窗")  # 顯示提示訊息
            cv2.destroyAllWindows()  # 關閉所有視窗
            return None  # 回傳 None 表示取消
        if key != -1:  # 若使用者按下任意其他鍵
            break  # 跳出迴圈，確認選擇

    cv2.destroyAllWindows()  # 關閉所有視窗

    idx = selected_index[0]  # 取得最後選到的索引
    name, code, channels = color_spaces[idx]  # 取出名稱、轉換代碼與通道標籤
    converted = image.copy() if code is None else cv2.cvtColor(image, code)  # 依代碼轉換影像色彩空間

    base = os.path.splitext(os.path.basename(image_path))[0]  # 取得原始檔名（不含副檔名）
    save_dir = os.path.join(os.path.dirname(image_path), "Color_Space_Channel")  # 組合儲存資料夾路徑
    os.makedirs(save_dir, exist_ok=True)  # 建立資料夾（若已存在則略過）

    if name == "GRAY":  # 若選的是灰階（只有單一通道）
        window_name = f"{name} - {channels[0]}"  # 組合視窗名稱
        cv2.imshow(window_name, converted)  # 顯示灰階影像

        while True:  # 等待使用者確認或取消
            key = cv2.waitKey(100)  # 每 100 毫秒檢查一次鍵盤
            if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:  # 若按 ESC 或關閉視窗
                print("⚠️ 使用者取消流程，未儲存結果")  # 顯示提示訊息
                cv2.destroyAllWindows()  # 關閉所有視窗
                return None  # 回傳 None 表示取消
            if key == ord('1') or key == ord('a'):  # 若按下 1 或 a 確認儲存
                break  # 跳出迴圈

        cv2.destroyAllWindows()  # 關閉所有視窗
        save_path = os.path.join(save_dir, f"{base}_{name}_{channels[0]}.jpg")  # 組合儲存路徑
        cv2.imwrite(save_path, converted)  # 儲存灰階影像
        print(f"✅ 已儲存灰階通道：{save_path}")  # 顯示儲存成功訊息
        return save_path  # 回傳儲存後的路徑

    else:  # 其他多通道的色彩空間
        ch = cv2.split(converted)  # 將影像拆分成各個通道
        window_names = []  # 建立空清單，記錄各通道視窗名稱

        for i in range(len(channels)):  # 逐一顯示每個通道
            window_name = f"{name} - {channels[i]}"  # 組合該通道的視窗名稱
            cv2.imshow(window_name, ch[i])  # 顯示該通道影像
            window_names.append(window_name)  # 把視窗名稱加入清單

        print("⌨️ 請依指定按鍵選擇要儲存的通道：(按下後即儲存)")  # 操作提示
        for i, ch_name in enumerate(channels):  # 逐一列出每個通道對應的按鍵
            print(f"  按 {i+1} → 儲存 {ch_name} 通道")  # 顯示按鍵與通道對應說明
        print("  按 0 → 儲存所有通道")  # 提示按 0 儲存全部
        print("  按 ESC → 取消")  # 提示按 ESC 取消

        selected_index = None  # 重設選擇索引（尚未選擇單一通道）
        save_all = False  # 是否儲存所有通道的旗標

        while True:  # 等待使用者按鍵選擇
            key = cv2.waitKey(100)  # 每 100 毫秒檢查一次鍵盤

            if key == 27:  # 若按下 ESC
                print("⚠️ 使用者取消流程，未儲存結果")  # 顯示提示訊息
                cv2.destroyAllWindows()  # 關閉所有視窗
                return None  # 回傳 None 表示取消

            for win in window_names:  # 逐一檢查每個通道視窗
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:  # 若有任何視窗被關閉
                    print(f"⚠️ 視窗「{win}」已被關閉，取消流程")  # 顯示提示訊息
                    cv2.destroyAllWindows()  # 關閉所有視窗
                    return None  # 回傳 None 表示取消

            if ord('1') <= key <= ord(str(len(channels))):  # 若按下的是有效通道編號
                selected_index = key - ord('1')  # 換算成通道索引
                break  # 跳出迴圈
            elif key == ord('0'):  # 若按下 0
                save_all = True  # 設定為儲存所有通道
                break  # 跳出迴圈

        cv2.destroyAllWindows()  # 關閉所有視窗
        saved_paths = []  # 建立空清單，記錄已儲存的路徑

        if save_all:  # 若選擇儲存所有通道
            for i in range(len(channels)):  # 逐一儲存每個通道
                save_path = os.path.join(save_dir, f"{base}_{name}({channels[i]}).jpg")  # 組合該通道的儲存路徑
                cv2.imwrite(save_path, ch[i])  # 儲存該通道影像
                saved_paths.append(save_path)  # 把路徑加入清單
            print(f"✅ 已儲存所有通道：\n" + "\n".join(saved_paths))  # 顯示所有已儲存路徑
            return saved_paths[0]  # 回傳第一個儲存路徑
        else:  # 只儲存單一選定通道
            save_path = os.path.join(save_dir, f"{base}_{name}({channels[selected_index]}).jpg")  # 組合儲存路徑
            cv2.imwrite(save_path, ch[selected_index])  # 儲存選定通道影像
            print(f"✅ 已儲存通道：{save_path}")  # 顯示儲存成功訊息
            return save_path  # 回傳儲存後的路徑




