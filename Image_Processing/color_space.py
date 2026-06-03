

import cv2
import os

def process(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 無法讀取影像")
        return None

    color_spaces = [
        ("BGR", None, ["B", "G", "R"]),
        ("HSV", cv2.COLOR_BGR2HSV, ["H", "S", "V"]),
        ("Lab", cv2.COLOR_BGR2Lab, ["L", "a", "b"]),
        ("YCrCb", cv2.COLOR_BGR2YCrCb, ["Y", "Cr", "Cb"]),
        ("HLS", cv2.COLOR_BGR2HLS, ["H", "L", "S"]),
        ("LUV", cv2.COLOR_BGR2LUV, ["L", "U", "V"]),
        ("XYZ", cv2.COLOR_BGR2XYZ, ["X", "Y", "Z"]),
        ("YUV", cv2.COLOR_BGR2YUV, ["Y", "U", "V"]),
        ("GRAY", cv2.COLOR_BGR2GRAY, ["Gray"]),
        ("RGB", cv2.COLOR_BGR2RGB, ["R", "G", "B"])
    ]

    selected_index = [0]

    def update(val):
        selected_index[0] = val
        name, code, _ = color_spaces[val]
        converted = image.copy() if code is None else cv2.cvtColor(image, code)
        preview = converted.copy()
        cv2.putText(preview, f"Color Mode: {name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Preview", preview)

    cv2.namedWindow("Preview")
    cv2.createTrackbar("Color Mode", "Preview", 0, len(color_spaces) - 1, update)
    
    print("📌色彩空間分離")
    print("拖曳滑桿選擇要分離的色彩空間")
    print("停留於待分離之色彩空間，請按下任意鍵繼續（或按 ESC 取消返回流程大廳)")
    update(0)

    while True:
        key = cv2.waitKey(100)
        if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
            print("⚠️ 使用者取消或關閉視窗")
            cv2.destroyAllWindows()
            return None
        if key != -1:
            break

    cv2.destroyAllWindows()

    idx = selected_index[0]
    name, code, channels = color_spaces[idx]
    converted = image.copy() if code is None else cv2.cvtColor(image, code)

    base = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.join(os.path.dirname(image_path), "Color_Space_Channel")
    os.makedirs(save_dir, exist_ok=True)

    if name == "GRAY":
        window_name = f"{name} - {channels[0]}"
        cv2.imshow(window_name, converted)

        while True:
            key = cv2.waitKey(100)
            if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("⚠️ 使用者取消流程，未儲存結果")
                cv2.destroyAllWindows()
                return None
            if key == ord('1') or key == ord('a'):
                break

        cv2.destroyAllWindows()
        save_path = os.path.join(save_dir, f"{base}_{name}_{channels[0]}.jpg")
        cv2.imwrite(save_path, converted)
        print(f"✅ 已儲存灰階通道：{save_path}")
        return save_path

    else:
        ch = cv2.split(converted)
        window_names = []

        for i in range(len(channels)):
            window_name = f"{name} - {channels[i]}"
            cv2.imshow(window_name, ch[i])
            window_names.append(window_name)

        print("⌨️ 請依指定按鍵選擇要儲存的通道：(按下後即儲存)")
        for i, ch_name in enumerate(channels):
            print(f"  按 {i+1} → 儲存 {ch_name} 通道")
        print("  按 0 → 儲存所有通道")
        print("  按 ESC → 取消")

        selected_index = None
        save_all = False

        while True:
            key = cv2.waitKey(100)

            if key == 27:
                print("⚠️ 使用者取消流程，未儲存結果")
                cv2.destroyAllWindows()
                return None

            for win in window_names:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    print(f"⚠️ 視窗「{win}」已被關閉，取消流程")
                    cv2.destroyAllWindows()
                    return None

            if ord('1') <= key <= ord(str(len(channels))):
                selected_index = key - ord('1')
                break
            elif key == ord('0'):
                save_all = True
                break

        cv2.destroyAllWindows()
        saved_paths = []

        if save_all:
            for i in range(len(channels)):
                save_path = os.path.join(save_dir, f"{base}_{name}({channels[i]}).jpg")
                cv2.imwrite(save_path, ch[i])
                saved_paths.append(save_path)
            print(f"✅ 已儲存所有通道：\n" + "\n".join(saved_paths))
            return saved_paths[0]
        else:
            save_path = os.path.join(save_dir, f"{base}_{name}({channels[selected_index]}).jpg")
            cv2.imwrite(save_path, ch[selected_index])
            print(f"✅ 已儲存通道：{save_path}")
            return save_path




