import cv2
import numpy as np
import os

def process(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 無法讀取影像")
        return None

    h, w = image.shape[:2]

    def update(val):
        dx = cv2.getTrackbarPos("X Offset", "Affine Preview")
        dy = cv2.getTrackbarPos("Y Offset", "Affine Preview")

        src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
        dst = np.float32([[dx, dy], [w - dx, dy], [dx, h - dy]])
        matrix = cv2.getAffineTransform(src, dst)
        warped = cv2.warpAffine(image, matrix, (w, h))
        cv2.imshow("Affine Preview", warped)

    cv2.namedWindow("Affine Preview")
    cv2.createTrackbar("X Offset", "Affine Preview", 30, 100, update)
    cv2.createTrackbar("Y Offset", "Affine Preview", 30, 100, update)
    update(0)

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == 27:
        print("⚠️ 使用者取消操作")
        return None

    dx = cv2.getTrackbarPos("X Offset", "Affine Preview")
    dy = cv2.getTrackbarPos("Y Offset", "Affine Preview")

    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    dst = np.float32([[dx, dy], [w - dx, dy], [dx, h - dy]])
    matrix = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(image, matrix, (w, h))

    base = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.join(os.path.dirname(image_path), "Image_Correction")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{base}_affine_dx{dx}_dy{dy}.jpg")
    cv2.imwrite(save_path, warped)

    print(f"✅ 仿射校正影像已儲存至：{save_path}")
    return save_path

