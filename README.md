# 影像處理工具 / Image Processing Tool

Python 影像處理專案，提供多種常見的影像處理功能。
支援**線上瀏覽器版**驗證功能、**本機桌面版**完整執行，以及**一鍵安裝工具**。

---

## 線上驗證（免安裝）

可直接在瀏覽器開啟以下連結，上傳圖片即可測試所有功能，不需要安裝任何軟體：

**https://pyimageprocessing-qb3njrgpzsvgiz9eihvvkb.streamlit.app/**

---

## 功能列表

| 分類 | 方法 | 可調整參數 |
|------|------|-----------|
| 色彩空間分離 | 色彩空間分離 | 色彩空間（BGR/HSV/Lab/YCrCb/HLS/LUV/XYZ/YUV/GRAY/RGB）、通道 |
| 二值化 | 自訂門檻 / Otsu 自動 / 自適應 | 門檻值、BlockSize、C 值 |
| 邊緣偵測 | Canny / Sobel / Laplacian / Scharr | 低門檻、高門檻、Kernel Size |
| 強度轉換 | Gamma / 對數 / 線性 / 直方圖等化 / CLAHE | 各方法對應參數 |
| 影像校正 | 仿射變換 | X 偏移、Y 偏移 |

---

## 鏈式處理（Pipeline）

支援將多個方法**依序串接**，讓每一步的輸出成為下一步的輸入，最終獲得想要的結果。

### 兩種操作模式（可切換）

| 模式 | 說明 |
|------|------|
| **逐步即時** | 選方法、調參數 → 按「套用此步驟」→ 結果立即更新，可繼續疊加下一步 |
| **批次清單** | 先把多個步驟加入清單 → 確認後按「執行全部」一次跑完 |

### 輔助功能

- **步驟歷史**：顯示已套用的每個步驟與參數
- **復原上一步**：取消最後一個步驟
- **重設為原圖**：清除所有步驟，從原圖重新開始
- **儲存結果**：輸出最終影像到 `Pipeline_Result/` 資料夾

---

## 執行版本比較

| | 線上版（Streamlit） | 桌面版（PyQt6） |
|---|---|---|
| 執行方式 | 瀏覽器，免安裝 | 本機安裝執行 |
| 即時預覽 | 有 | 有 |
| 滑桿互動 | 有 | 有 |
| 鏈式處理 | 有 | 有 |
| 結果下載 | 有（下載按鈕） | 有（自動儲存到資料夾） |
| 適合用途 | 快速驗證功能 | 正式使用 |

---

## 本機執行（PyQt6 桌面版）

### 前置需求

請先安裝 **Python 3.10 以上**：https://www.python.org/downloads/

> 安裝時請務必勾選 **"Add Python to PATH"**

### 下載專案

點右上角 `Code` → `Download ZIP`，解壓縮後進入資料夾。

### 方法一：一鍵執行（推薦）

解壓縮後直接雙擊 `setup.bat`，腳本會自動：
- 檢查所有必要套件是否已安裝
- **已安裝**：跳過安裝，直接啟動程式
- **有缺套件**：自動安裝後再啟動程式

### 方法二：手動安裝

```bash
pip install PyQt6 opencv-python numpy
cd Image_Processing
python ui_main.py
```

---

## 建議工作流程

```
1. 開啟線上版 → 上傳圖片，用鏈式處理驗證想要的效果
         ↓
2. 確認後，下載專案 ZIP 並解壓縮
         ↓
3. 雙擊 setup.bat → 自動檢查套件並啟動桌面版
```

---

## 如何新增影像處理方法

本專案採用「方法註冊表」架構，**新增方法只需兩步，UI 自動更新**：

**步驟 1**：在 `Image_Processing/processing/` 對應的 `_fn.py` 新增純函數：
```python
def my_method(image: np.ndarray, param_a: int) -> np.ndarray:
    # 影像處理邏輯
    return result
```

**步驟 2**：在 `Image_Processing/processing/registry.py` 的 `METHODS` 清單加入一筆：
```python
Method(
    id="my_method",
    category="我的分類",       # 現有分類或新分類皆可
    name="我的方法名稱",
    params=[
        ParamSpec("param_a", "參數A顯示名", "int", default=50, min=0, max=100, step=1),
    ],
    fn=lambda img, p: my_method(img, p["param_a"]),
    output_folder="My_Output",
),
```

完成後桌面版與網頁版都會自動出現此方法，可以調整參數並加入鏈式處理流程。

---

## 專案結構

```
250319_Py_Image_Processing/
├── setup.bat                   # 一鍵安裝並啟動桌面版
├── requirements.txt            # 套件清單
└── Image_Processing/
    ├── ui_main.py              # PyQt6 桌面版主程式
    ├── streamlit_app.py        # Streamlit 線上/本機網頁版主程式
    ├── main.py                 # 原始命令列版主程式
    ├── binarize.py
    ├── color_space.py
    ├── edge_detect.py
    ├── image_correction.py
    ├── intensity_transform.py
    ├── modules/
    │   ├── image_selector.py
    │   ├── workflow_manager.py
    │   └── utils.py
    └── processing/             # 核心模組（桌面版與網頁版共用）
        ├── registry.py         # 方法註冊表（擴充新方法在此）
        ├── pipeline.py         # 鏈式處理引擎
        ├── binarize_fn.py
        ├── color_space_fn.py
        ├── edge_detect_fn.py
        ├── intensity_transform_fn.py
        └── image_correction_fn.py
```
