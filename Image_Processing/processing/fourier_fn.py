import cv2  # 匯入 OpenCV 影像處理函式庫
import numpy as np  # 匯入 NumPy，用來做傅立葉與陣列運算

# 暫存傅立葉正轉換的相位與原始形狀，供 fft_inverse 使用
_FFT_CACHE: dict = {}  # 模組層級的快取字典，跨函式保存相位與形狀


def fourier_spectrum(image):  # 計算並回傳頻譜圖
    """傅立葉變換：計算並顯示頻譜 (magnitude spectrum)，中央為低頻、外圍為高頻。"""  # 函式說明（保留原本 docstring）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image  # 彩色就轉灰階，已是灰階則直接用
    f = np.fft.fft2(gray)  # 對影像做二維快速傅立葉轉換到頻域
    fshift = np.fft.fftshift(f)  # 將低頻搬移到頻譜中央，方便觀察
    mag = 20 * np.log(np.abs(fshift) + 1)  # 取振幅並做對數壓縮（+1避免log(0)）以利顯示
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 正規化到0~255範圍
    return mag.astype(np.uint8)  # 轉成8位元影像回傳


def fourier_filter(image, filter_type=0, cutoff=30):  # 頻域濾波後反轉換
    """頻域濾波後反轉換 (IFFT)。

    filter_type: 0=低通，1=高通。
    cutoff: 截止半徑（像素）。
    """  # 函式說明（保留原本 docstring）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image  # 彩色就轉灰階，已是灰階則直接用
    rows, cols = gray.shape  # 取得影像的高與寬
    crow, ccol = rows // 2, cols // 2  # 計算頻譜中心座標（低頻所在）
    f = np.fft.fft2(gray)  # 二維傅立葉轉換到頻域
    fshift = np.fft.fftshift(f)  # 將低頻搬移到中央
    y, x = np.ogrid[:rows, :cols]  # 建立座標格點，用來計算每點到中心的距離
    dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)  # 計算每個像素到頻譜中心的距離
    mask = (dist <= cutoff).astype(np.float32) if filter_type == 0 else (dist > cutoff).astype(np.float32)  # 低通保留近中心、高通保留遠中心
    f_ishift = np.fft.ifftshift(fshift * mask)  # 套用遮罩後將頻譜搬回原位
    img_back = np.abs(np.fft.ifft2(f_ishift))  # 反傅立葉轉換回空間域並取振幅
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)  # 正規化到0~255範圍
    return img_back.astype(np.uint8)  # 轉成8位元影像回傳


def fft_forward(image):  # 傅立葉正轉換並暫存相位
    """傅立葉正轉換：輸出頻譜圖供後續處理，同時暫存相位資訊。"""  # 函式說明（保留原本 docstring）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image  # 彩色就轉灰階，已是灰階則直接用
    f = np.fft.fft2(gray)  # 二維傅立葉轉換到頻域
    fshift = np.fft.fftshift(f)  # 將低頻搬移到中央
    _FFT_CACHE['phase'] = np.angle(fshift)  # 暫存相位資訊，供反轉換重建用
    _FFT_CACHE['shape'] = gray.shape  # 暫存原始影像形狀
    mag_log = np.log(np.abs(fshift) + 1)  # 取振幅做對數壓縮（+1避免log(0)）
    mag_norm = cv2.normalize(mag_log, None, 0, 255, cv2.NORM_MINMAX)  # 正規化到0~255範圍
    return mag_norm.astype(np.uint8)  # 轉成8位元頻譜圖回傳


def fft_inverse(image):  # 傅立葉反轉換重建影像
    """傅立葉反轉換：用 fft_forward 暫存的相位 + 當前影像的振幅重建空間域影像。"""  # 函式說明（保留原本 docstring）
    if 'phase' not in _FFT_CACHE:  # 若尚未做過正轉換（沒有相位）
        return image  # 直接回傳原影像
    phase = _FFT_CACHE['phase']  # 取出暫存的相位
    orig_shape = _FFT_CACHE['shape']  # 取出暫存的原始形狀
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image  # 彩色就轉灰階，已是灰階則直接用
    if gray.shape != orig_shape:  # 若當前影像尺寸與原始不同
        gray = cv2.resize(gray, (orig_shape[1], orig_shape[0]))  # 縮放回原始尺寸以對齊相位
    fshift_new = gray.astype(np.float64) * np.exp(1j * phase)  # 以當前影像當振幅、暫存相位組成複數頻譜
    f_ishift = np.fft.ifftshift(fshift_new)  # 將頻譜從中央搬回原位
    img_back = np.real(np.fft.ifft2(f_ishift))  # 反傅立葉轉換回空間域並取實部
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)  # 正規化到0~255範圍
    return img_back.astype(np.uint8)  # 轉成8位元影像回傳
