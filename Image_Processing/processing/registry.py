from dataclasses import dataclass, field  # 匯入 dataclass 裝飾器與 field（用於定義資料類別與其欄位）
from typing import Callable, Optional  # 匯入型別註記：Callable（可呼叫物件）、Optional（可為 None 的型別）
import numpy as np  # 匯入 NumPy，用於陣列與數值運算（影像以 ndarray 表示）
import cv2  # 匯入 OpenCV，用於影像處理（色彩轉換等）

from processing.binarize_fn import binarize_custom, binarize_otsu, binarize_adaptive, binarize_zone  # 匯入二值化函式：自訂門檻、Otsu 自動、自適應、區域
from processing.color_space_fn import split_channels, COLOR_MODES  # 匯入色彩空間通道分離函式與可選色彩空間清單
from processing.edge_detect_fn import canny, sobel, laplacian, scharr  # 匯入邊緣偵測函式：Canny、Sobel、Laplacian、Scharr
from processing.intensity_transform_fn import gamma, log_transform, linear, histogram_eq, clahe  # 匯入強度轉換函式：Gamma、對數、線性、直方圖等化、CLAHE
from processing.image_correction_fn import affine  # 匯入影像校正的仿射變換函式
from processing.object_extract_fn import extract_white_objects, SHAPE_MODES, BOX_COLORS  # 匯入白色物件萃取函式與外接形狀、框線顏色清單
from processing.sharpen_fn import sharpen_unsharp, sharpen_laplacian  # 匯入影像銳化函式：Unsharp Mask、Laplacian 銳化
from processing.geometry_fn import geo_scale, geo_rotate, geo_translate, geo_flip  # 匯入幾何轉換函式：縮放、旋轉、平移、翻轉
from processing.morphology_fn import morph_dilate, morph_erode, morph_open, morph_close  # 匯入形態學運算函式：膨脹、侵蝕、開運算、閉運算
from processing.filter_fn import filter_gaussian, filter_box, filter_median, filter_bilateral  # 匯入濾波函式：高斯、均值、中值、雙邊
from processing.rayleigh_fn import rayleigh_transform  # 匯入雷利（Rayleigh）轉換函式
from processing.fourier_fn import fourier_spectrum, fourier_filter, fft_forward, fft_inverse  # 匯入傅立葉相關函式：頻譜、頻域濾波、FFT 正轉換、FFT 反轉換
from processing.chaincode_fn import chain_code_8  # 匯入八方向鍊碼計算函式
from processing.perspective_fn import perspective_correct  # 匯入透視校正函式
from processing.inpaint_fn import inpaint_auto  # 匯入自動影像修復函式
from processing.contour_analysis_fn import analyze_contours  # 匯入輪廓特徵分析函式
from processing.hough_fn import hough_lines, hough_circles  # 匯入霍夫偵測函式：直線、圓
from processing.watershed_fn import watershed_segment  # 匯入分水嶺分割函式
from processing.template_match_fn import template_match_center  # 匯入可移動模板比對函式


@dataclass  # 以 dataclass 自動生成 __init__ 等方法
class ParamSpec:  # 參數規格類別：描述每個處理方法可調整的單一參數
    key: str  # 參數的鍵名（程式內部存取用的識別字）
    label: str  # 參數在介面上顯示的標籤文字
    kind: str                          # "int" | "float" | "choice" | "buttons"  # 參數型別：整數、浮點、下拉選擇、按鈕群
    default: object = 0  # 參數的預設值
    min: Optional[float] = None  # 參數允許的最小值（可為 None 表示不限）
    max: Optional[float] = None  # 參數允許的最大值（可為 None 表示不限）
    step: Optional[float] = None  # 參數調整的步進值（可為 None）
    choices: Optional[list] = None     # list of (value, display_str)  # 選擇型參數的選項清單，元素為 (值, 顯示字串)
    help: str = ""                     # 參數說明文字  # 參數的說明文字，供介面提示使用


@dataclass  # 以 dataclass 自動生成 __init__ 等方法
class Method:  # 處理方法類別：描述一個完整的影像處理功能項目
    id: str  # 方法的唯一識別字
    category: str  # 方法所屬的分類名稱
    name: str  # 方法在介面上顯示的名稱
    params: list  # 此方法的參數清單（ParamSpec 的串列）
    fn: Callable  # 實際執行處理的可呼叫函式（接收影像與參數字典）
    output_folder: str  # 處理結果輸出的資料夾名稱
    description: str = ""  # 方法的詳細說明文字


def _to_gray(image: np.ndarray) -> np.ndarray:  # 工具函式：將影像轉為灰階
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image  # 若為三通道彩色則轉灰階，否則原樣回傳


METHODS: list[Method] = [  # 所有處理方法的註冊清單（核心註冊表）
    # ── 色彩空間分離 ──
    Method(  # 定義「色彩空間分離」方法
        id="color_split",  # 方法識別字
        category="色彩空間分離",  # 所屬分類
        name="色彩空間分離",  # 顯示名稱
        description="將影像轉換到指定色彩空間（HSV、Lab、YCrCb 等）並取出單一通道，常用於依顏色特性分離前景或做後續分析。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("mode_idx", "色彩空間", "buttons", default=0,  # 參數：色彩空間選擇（按鈕群）
                      choices=[(i, m) for i, m in enumerate(COLOR_MODES)],  # 選項由 COLOR_MODES 動態產生（索引, 名稱）
                      help="選擇要轉換到的色彩空間，不同空間能凸顯不同顏色特性（如 HSV 適合依色相分離）。"),  # 參數說明
            ParamSpec("ch_idx", "通道", "buttons", default=0,  # 參數：要取出的通道索引（按鈕群）
                      choices=[(0, "B(0)"), (1, "G(1)"), (2, "R(2)")],  # 通道選項：0/1/2
                      help="取出該色彩空間的第幾個通道顯示，例如 HSV 的通道 0 是色相 H。"),  # 參數說明
        ],
        fn=lambda img, p: split_channels(img, p["mode_idx"])[  # 呼叫 split_channels 取得通道清單，再索引出指定通道
            min(p["ch_idx"], len(split_channels(img, p["mode_idx"])) - 1)  # 以 min 夾住索引避免超出通道數上限
        ],
        output_folder="Color_Space_Channel",  # 輸出資料夾
    ),

    # ── 二值化 ──
    Method(  # 定義「自訂門檻」二值化方法
        id="binarize_custom",  # 方法識別字
        category="二值化",  # 所屬分類
        name="自訂門檻",  # 顯示名稱
        description="以手動指定的門檻值將影像分成黑白兩色，高於門檻為白、低於為黑，適合對比明顯的影像。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("threshold", "門檻值", "int", default=127, min=0, max=255, step=1,  # 參數：二值化門檻值（整數）
                      help="亮度高於此值的像素變白、低於變黑。值越大，被判為白的區域越少。"),  # 參數說明
            ParamSpec("inv", "反轉", "buttons", default=0, choices=[(0, "正常"), (1, "反轉")],  # 參數：是否反轉黑白
                      help="反轉黑白：原本白的變黑、黑的變白。用於前景比背景暗時。"),  # 參數說明
        ],
        fn=lambda img, p: binarize_custom(_to_gray(img), p["threshold"], p["inv"]),  # 先轉灰階再呼叫 binarize_custom
        output_folder="Binarized",  # 輸出資料夾
    ),
    Method(  # 定義「Otsu 自動」二值化方法
        id="binarize_otsu",  # 方法識別字
        category="二值化",  # 所屬分類
        name="Otsu 自動",  # 顯示名稱
        description="自動計算最佳門檻值進行二值化，免手動調整，適合前景與背景分明的影像。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("inv", "反轉", "buttons", default=0, choices=[(0, "正常"), (1, "反轉")],  # 參數：是否反轉黑白
                      help="反轉黑白：原本白的變黑、黑的變白。用於前景比背景暗時。"),  # 參數說明
        ],
        fn=lambda img, p: binarize_otsu(_to_gray(img), p["inv"]),  # 先轉灰階再呼叫 binarize_otsu
        output_folder="Binarized",  # 輸出資料夾
    ),
    Method(  # 定義「自適應」二值化方法
        id="binarize_adaptive",  # 方法識別字
        category="二值化",  # 所屬分類
        name="自適應",  # 顯示名稱
        description="依每個區域的局部亮度各自決定門檻，適合照明不均勻的影像。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("block_size_idx", "BlockSize", "int", default=4, min=0, max=249, step=1,  # 參數：局部鄰域大小索引（整數）
                      help="計算局部門檻時參考的鄰域大小。值越大，參考範圍越廣、結果越平滑。"),  # 參數說明
            ParamSpec("c_offset", "C 值", "int", default=52, min=0, max=100, step=1,  # 參數：門檻的常數偏移（整數）
                      help="從局部平均減去的常數（會換算為 -50~+50）。值越大門檻越低、白色越多。"),  # 參數說明
            ParamSpec("inv", "反轉", "buttons", default=0, choices=[(0, "正常"), (1, "反轉")],  # 參數：是否反轉黑白
                      help="反轉黑白：原本白的變黑、黑的變白。用於前景比背景暗時。"),  # 參數說明
        ],
        fn=lambda img, p: binarize_adaptive(_to_gray(img), p["block_size_idx"], p["c_offset"], p["inv"]),  # 先轉灰階再呼叫 binarize_adaptive
        output_folder="Binarized",  # 輸出資料夾
    ),
    Method(  # 定義「區域二值化」方法
        id="binarize_zone",  # 方法識別字
        category="二值化",  # 所屬分類
        name="區域二值化",  # 顯示名稱
        description="將影像切成 列×欄 的格子，每格各自用 Otsu 自動門檻二值化後拼回，適合影像不同區域亮度差異大的場景。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("rows", "列數", "int", default=3, min=1, max=8, step=1,  # 參數：垂直切割數
                      help="將影像上下切成幾列，每列獨立計算門檻。"),  # 參數說明
            ParamSpec("cols", "欄數", "int", default=3, min=1, max=8, step=1,  # 參數：水平切割數
                      help="將影像左右切成幾欄，每欄獨立計算門檻。"),  # 參數說明
            ParamSpec("inv", "反轉", "buttons", default=0, choices=[(0, "正常"), (1, "反轉")],  # 參數：是否反轉黑白
                      help="反轉黑白：原本白的變黑、黑的變白。"),  # 參數說明
        ],
        fn=lambda img, p: binarize_zone(_to_gray(img), p["rows"], p["cols"], p["inv"]),  # 先轉灰階再呼叫 binarize_zone
        output_folder="Binarized",  # 輸出資料夾
    ),

    # ── 物件萃取 ──
    Method(  # 定義「白色物件框選」方法
        id="object_extract_white",  # 方法識別字
        category="物件萃取",  # 所屬分類
        name="白色物件框選",  # 顯示名稱
        description="在二值影像中找出白色相連區塊，計算數量並用外接形狀框選標示，可用於物件計數與定位。建議先二值化再使用。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("shape_idx", "外接形狀", "buttons", default=0,  # 參數：外接框形狀選擇
                      choices=[(i, s) for i, s in enumerate(SHAPE_MODES)],  # 選項由 SHAPE_MODES 動態產生
                      help="框選物件時使用的外接形狀（矩形、圓形等）。"),  # 參數說明
            ParamSpec("color_idx", "框線顏色", "buttons", default=1,  # 參數：框線顏色選擇
                      choices=[(i, name) for i, (name, _bgr) in enumerate(BOX_COLORS)],  # 選項由 BOX_COLORS 動態產生（取名稱）
                      help="框線顯示的顏色。"),  # 參數說明
            ParamSpec("thick_idx", "線寬", "buttons", default=1,  # 參數：框線線寬選擇
                      choices=[(0, "1"), (1, "2"), (2, "3")],  # 線寬選項：對應 1/2/3 像素
                      help="框線的粗細（像素）。"),  # 參數說明
            ParamSpec("min_area", "最小面積", "int", default=50, min=0, max=2000, step=1,  # 參數：最小面積過濾門檻（整數）
                      help="面積小於此值的區塊會被忽略，可過濾雜訊小白點。"),  # 參數說明
        ],
        fn=lambda img, p: extract_white_objects(  # 呼叫 extract_white_objects 萃取白色物件
            img, p["shape_idx"], p["color_idx"], p["min_area"], p["thick_idx"] + 1),  # 傳入影像、形狀、顏色、最小面積與線寬（索引加 1 換算實際像素）
        output_folder="Object_Extract",  # 輸出資料夾
    ),

    # ── 形態學運算 ──
    Method(  # 定義「膨脹」形態學方法
        id="morph_dilate",  # 方法識別字
        category="形態學運算",  # 所屬分類
        name="膨脹",  # 顯示名稱
        description="讓白色區域向外擴張，可連接斷裂處、填補細縫、加粗線條。建議用於二值影像。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("ksize", "核心大小", "int", default=3, min=0, max=20, step=1,  # 參數：結構元素大小（整數）
                      help="形態學結構元素大小（換算為 2k+1 的奇數方塊）。值越大，作用範圍越廣。"),  # 參數說明
            ParamSpec("iterations", "次數", "int", default=1, min=1, max=10, step=1,  # 參數：重複運算次數（整數）
                      help="重複套用運算的次數，次數越多效果越強。"),  # 參數說明
        ],
        fn=lambda img, p: morph_dilate(img, p["ksize"], p["iterations"]),  # 呼叫 morph_dilate 進行膨脹
        output_folder="Morphology",  # 輸出資料夾
    ),
    Method(  # 定義「侵蝕」形態學方法
        id="morph_erode",  # 方法識別字
        category="形態學運算",  # 所屬分類
        name="侵蝕",  # 顯示名稱
        description="讓白色區域向內收縮，可去除小雜點、分離相黏的物件、使線條變細。建議用於二值影像。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("ksize", "核心大小", "int", default=3, min=0, max=20, step=1,  # 參數：結構元素大小（整數）
                      help="形態學結構元素大小（換算為 2k+1 的奇數方塊）。值越大，作用範圍越廣。"),  # 參數說明
            ParamSpec("iterations", "次數", "int", default=1, min=1, max=10, step=1,  # 參數：重複運算次數（整數）
                      help="重複套用運算的次數，次數越多效果越強。"),  # 參數說明
        ],
        fn=lambda img, p: morph_erode(img, p["ksize"], p["iterations"]),  # 呼叫 morph_erode 進行侵蝕
        output_folder="Morphology",  # 輸出資料夾
    ),
    Method(  # 定義「開運算」形態學方法
        id="morph_open",  # 方法識別字
        category="形態學運算",  # 所屬分類
        name="開運算（去除小白點）",  # 顯示名稱
        description="先侵蝕再膨脹，去除細小白點雜訊但保持主體大小不變。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("ksize", "核心大小", "int", default=3, min=0, max=20, step=1,  # 參數：結構元素大小（整數）
                      help="形態學結構元素大小（換算為 2k+1 的奇數方塊）。值越大，作用範圍越廣。"),  # 參數說明
        ],
        fn=lambda img, p: morph_open(img, p["ksize"]),  # 呼叫 morph_open 進行開運算
        output_folder="Morphology",  # 輸出資料夾
    ),
    Method(  # 定義「閉運算」形態學方法
        id="morph_close",  # 方法識別字
        category="形態學運算",  # 所屬分類
        name="閉運算（填補孔洞）",  # 顯示名稱
        description="先膨脹再侵蝕，填補物件內部的孔洞與小縫隙，使區塊更完整。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("ksize", "核心大小", "int", default=3, min=0, max=20, step=1,  # 參數：結構元素大小（整數）
                      help="形態學結構元素大小（換算為 2k+1 的奇數方塊）。值越大，作用範圍越廣。"),  # 參數說明
        ],
        fn=lambda img, p: morph_close(img, p["ksize"]),  # 呼叫 morph_close 進行閉運算
        output_folder="Morphology",  # 輸出資料夾
    ),

    # ── 影像銳化 ──
    Method(  # 定義「Unsharp Mask」銳化方法
        id="sharpen_unsharp",  # 方法識別字
        category="影像銳化",  # 所屬分類
        name="Unsharp Mask",  # 顯示名稱
        description="透過原圖減去模糊版本來強化邊緣細節，使影像看起來更清晰銳利。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("amount", "強度", "int", default=10, min=1, max=30, step=1,  # 參數：銳化強度（整數）
                      help="銳化強度，值越大邊緣對比越明顯，過大會出現雜訊與光暈。"),  # 參數說明
            ParamSpec("radius", "半徑", "int", default=3, min=1, max=15, step=1,  # 參數：模糊半徑（整數）
                      help="模糊半徑，決定要強化的細節尺度。半徑越大強化越粗的邊緣。"),  # 參數說明
        ],
        fn=lambda img, p: sharpen_unsharp(img, p["amount"], p["radius"]),  # 呼叫 sharpen_unsharp 進行銳化
        output_folder="Sharpened",  # 輸出資料夾
    ),
    Method(  # 定義「Laplacian 銳化」方法
        id="sharpen_laplacian",  # 方法識別字
        category="影像銳化",  # 所屬分類
        name="Laplacian 銳化",  # 顯示名稱
        description="用 Laplacian 高通核心強化邊緣，提升輪廓對比，讓細節更突出。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("scale", "強度", "int", default=2, min=1, max=5, step=1,  # 參數：高通強化倍數（整數）
                      help="Laplacian 高通強化倍數，值越大邊緣越強。"),  # 參數說明
        ],
        fn=lambda img, p: sharpen_laplacian(img, p["scale"]),  # 呼叫 sharpen_laplacian 進行銳化
        output_folder="Sharpened",  # 輸出資料夾
    ),

    # ── 邊緣偵測 ──
    Method(  # 定義「Canny」邊緣偵測方法
        id="edge_canny",  # 方法識別字
        category="邊緣偵測",  # 所屬分類
        name="Canny",  # 顯示名稱
        description="經典的多階段邊緣偵測，可得到乾淨細緻的邊緣線，最適合用來擷取物件輪廓。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("low", "低門檻", "int", default=100, min=0, max=255, step=1,  # 參數：Canny 弱邊緣門檻（整數）
                      help="Canny 弱邊緣門檻。低於此值不算邊緣；值越低偵測到的邊越多。"),  # 參數說明
            ParamSpec("high", "高門檻", "int", default=200, min=0, max=255, step=1,  # 參數：Canny 強邊緣門檻（整數）
                      help="Canny 強邊緣門檻。高於此值一定是邊緣；建議約為低門檻的 2~3 倍。"),  # 參數說明
        ],
        fn=lambda img, p: canny(_to_gray(img), p["low"], p["high"]),  # 先轉灰階再呼叫 canny 偵測邊緣
        output_folder="Edge_Detected",  # 輸出資料夾
    ),
    Method(  # 定義「Sobel」邊緣偵測方法
        id="edge_sobel",  # 方法識別字
        category="邊緣偵測",  # 所屬分類
        name="Sobel",  # 顯示名稱
        description="計算亮度梯度找出邊緣，對水平與垂直方向的明暗變化特別敏感。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("ksize_idx", "Kernel Size", "int", default=0, min=0, max=14, step=1,  # 參數：Sobel 核心大小索引（整數）
                      help="Sobel 運算子的核心大小（換算為奇數）。值越大邊緣越粗、抗雜訊越強。"),  # 參數說明
        ],
        fn=lambda img, p: sobel(_to_gray(img), p["ksize_idx"]),  # 先轉灰階再呼叫 sobel 偵測邊緣
        output_folder="Edge_Detected",  # 輸出資料夾
    ),
    Method(  # 定義「Laplacian」邊緣偵測方法
        id="edge_laplacian",  # 方法識別字
        category="邊緣偵測",  # 所屬分類
        name="Laplacian",  # 顯示名稱
        description="用二階微分偵測邊緣，能找出各方向的細節，但對雜訊也較敏感。",  # 方法說明
        params=[],  # 無可調參數
        fn=lambda img, p: laplacian(_to_gray(img)),  # 先轉灰階再呼叫 laplacian 偵測邊緣
        output_folder="Edge_Detected",  # 輸出資料夾
    ),
    Method(  # 定義「Scharr」邊緣偵測方法
        id="edge_scharr",  # 方法識別字
        category="邊緣偵測",  # 所屬分類
        name="Scharr",  # 顯示名稱
        description="Sobel 的改良版，對細微邊緣的反應更準確，適合需要精細梯度的場合。",  # 方法說明
        params=[],  # 無可調參數
        fn=lambda img, p: scharr(_to_gray(img)),  # 先轉灰階再呼叫 scharr 偵測邊緣
        output_folder="Edge_Detected",  # 輸出資料夾
    ),

    # ── 強度轉換 ──
    Method(  # 定義「Gamma」強度轉換方法
        id="intensity_gamma",  # 方法識別字
        category="強度轉換",  # 所屬分類
        name="Gamma",  # 顯示名稱
        description="以指數曲線調整明暗：小於 1 提亮暗部、大於 1 壓暗亮部，用於校正影像亮度。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("gamma_x100", "Gamma x100", "int", default=100, min=1, max=300, step=1,  # 參數：Gamma 值乘以 100（整數）
                      help="Gamma 值（會除以 100）。小於 1 提亮暗部，大於 1 壓暗。100 為不變。"),  # 參數說明
        ],
        fn=lambda img, p: gamma(_to_gray(img), p["gamma_x100"] / 100),  # 先轉灰階再呼叫 gamma（參數除以 100 還原實際值）
        output_folder="Intensity_Transform",  # 輸出資料夾
    ),
    Method(  # 定義「對數轉換」強度轉換方法
        id="intensity_log",  # 方法識別字
        category="強度轉換",  # 所屬分類
        name="對數轉換",  # 顯示名稱
        description="壓縮高亮度、拉伸暗部細節，適合處理動態範圍很大的影像。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("k_x10", "k x10", "int", default=10, min=1, max=50, step=1,  # 參數：縮放係數 k 乘以 10（整數）
                      help="對數轉換的縮放係數（會除以 10）。值越大整體越亮、暗部細節越明顯。"),  # 參數說明
        ],
        fn=lambda img, p: log_transform(_to_gray(img), p["k_x10"] / 10),  # 先轉灰階再呼叫 log_transform（參數除以 10 還原實際值）
        output_folder="Intensity_Transform",  # 輸出資料夾
    ),
    Method(  # 定義「線性轉換」強度轉換方法
        id="intensity_linear",  # 方法識別字
        category="強度轉換",  # 所屬分類
        name="線性轉換",  # 顯示名稱
        description="以線性公式同時調整對比（斜率 a）與亮度（位移 b）。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("a_x100", "a x100", "int", default=100, min=50, max=300, step=1,  # 參數：對比增益 a 乘以 100（整數）
                      help="對比增益 a（會除以 100）。大於 1 增加對比，小於 1 降低對比。"),  # 參數說明
            ParamSpec("b_offset", "b 值", "int", default=100, min=0, max=200, step=1,  # 參數：亮度偏移 b（整數）
                      help="亮度偏移 b（會換算為 -100~+100）。值越大整體越亮。"),  # 參數說明
        ],
        fn=lambda img, p: linear(_to_gray(img), p["a_x100"] / 100, p["b_offset"] - 100),  # 先轉灰階再呼叫 linear（a 除以 100、b 減 100 還原）
        output_folder="Intensity_Transform",  # 輸出資料夾
    ),
    Method(  # 定義「直方圖等化」強度轉換方法
        id="intensity_hist_eq",  # 方法識別字
        category="強度轉換",  # 所屬分類
        name="直方圖等化",  # 顯示名稱
        description="重新分布灰階使整體對比最大化，讓原本不明顯的細節變清楚。",  # 方法說明
        params=[],  # 無可調參數
        fn=lambda img, p: histogram_eq(_to_gray(img)),  # 先轉灰階再呼叫 histogram_eq 進行等化
        output_folder="Intensity_Transform",  # 輸出資料夾
    ),
    Method(  # 定義「CLAHE」強度轉換方法
        id="intensity_clahe",  # 方法識別字
        category="強度轉換",  # 所屬分類
        name="CLAHE",  # 顯示名稱
        description="限制對比的區域型直方圖等化，增強局部細節同時避免過度放大雜訊。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("clip_x10", "clipLimit x10", "int", default=20, min=1, max=40, step=1,  # 參數：對比限制乘以 10（整數）
                      help="CLAHE 對比限制（會除以 10）。值越大局部對比增強越強，但雜訊也越明顯。"),  # 參數說明
        ],
        fn=lambda img, p: clahe(_to_gray(img), p["clip_x10"] / 10),  # 先轉灰階再呼叫 clahe（參數除以 10 還原實際值）
        output_folder="Intensity_Transform",  # 輸出資料夾
    ),
    Method(  # 定義「雷利轉換」強度轉換方法
        id="intensity_rayleigh",  # 方法識別字
        category="強度轉換",  # 所屬分類
        name="雷利轉換",  # 顯示名稱
        description="依雷利（Rayleigh）分佈重新映射灰階以增強對比，常用於超音波、雷達等影像。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("alpha", "alpha", "int", default=60, min=10, max=150, step=1,  # 參數：冪次指數 alpha（整數）
                      help="冪次轉換的指數參數，控制亮度曲線的彎曲程度。"),  # 參數說明
        ],
        fn=lambda img, p: rayleigh_transform(img, p["alpha"]),  # 呼叫 rayleigh_transform 進行雷利轉換
        output_folder="Intensity_Transform",  # 輸出資料夾
    ),

    # ── 影像校正 ──
    Method(  # 定義「仿射變換」影像校正方法
        id="correction_affine",  # 方法識別字
        category="影像校正",  # 所屬分類
        name="仿射變換",  # 顯示名稱
        description="以仿射矩陣校正影像的偏移與傾斜，用於幾何位置的對齊校準。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("dx", "X 偏移", "int", default=30, min=0, max=100, step=1,  # 參數：水平偏移校正量（整數）
                      help="水平方向的偏移校正量（像素）。"),  # 參數說明
            ParamSpec("dy", "Y 偏移", "int", default=30, min=0, max=100, step=1,  # 參數：垂直偏移校正量（整數）
                      help="垂直方向的偏移校正量（像素）。"),  # 參數說明
        ],
        fn=lambda img, p: affine(img, p["dx"], p["dy"]),  # 呼叫 affine 進行仿射變換
        output_folder="Image_Correction",  # 輸出資料夾
    ),

    # ── 幾何轉換 ──
    Method(  # 定義「縮放」幾何轉換方法
        id="geo_scale",  # 方法識別字
        category="幾何轉換",  # 所屬分類
        name="縮放",  # 顯示名稱
        description="依百分比放大或縮小影像尺寸，X、Y 可分別設定。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("scale_x_pct", "X 縮放%", "int", default=100, min=10, max=300, step=1,  # 參數：水平縮放百分比（整數）
                      help="水平縮放比例（百分比）。100 為原寬，200 為放大兩倍。"),  # 參數說明
            ParamSpec("scale_y_pct", "Y 縮放%", "int", default=100, min=10, max=300, step=1,  # 參數：垂直縮放百分比（整數）
                      help="垂直縮放比例（百分比）。100 為原高，200 為放大兩倍。"),  # 參數說明
        ],
        fn=lambda img, p: geo_scale(img, p["scale_x_pct"], p["scale_y_pct"]),  # 呼叫 geo_scale 進行縮放
        output_folder="Geometry",  # 輸出資料夾
    ),
    Method(  # 定義「旋轉」幾何轉換方法
        id="geo_rotate",  # 方法識別字
        category="幾何轉換",  # 所屬分類
        name="旋轉",  # 顯示名稱
        description="以影像中心旋轉指定角度，並自動擴張畫布以避免內容被裁切。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("angle", "角度", "int", default=0, min=-180, max=180, step=1,  # 參數：旋轉角度（整數）
                      help="以影像中心旋轉的角度（度）。正值逆時針，負值順時針。"),  # 參數說明
        ],
        fn=lambda img, p: geo_rotate(img, p["angle"]),  # 呼叫 geo_rotate 進行旋轉
        output_folder="Geometry",  # 輸出資料夾
    ),
    Method(  # 定義「平移」幾何轉換方法
        id="geo_translate",  # 方法識別字
        category="幾何轉換",  # 所屬分類
        name="平移",  # 顯示名稱
        description="將影像沿 X / Y 方向移動指定的像素距離（可為負值）。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("tx", "X 平移", "int", default=0, min=-300, max=300, step=1,  # 參數：水平平移量（整數）
                      help="水平平移量（像素）。正值往右，負值往左。"),  # 參數說明
            ParamSpec("ty", "Y 平移", "int", default=0, min=-300, max=300, step=1,  # 參數：垂直平移量（整數）
                      help="垂直平移量（像素）。正值往下，負值往上。"),  # 參數說明
        ],
        fn=lambda img, p: geo_translate(img, p["tx"], p["ty"]),  # 呼叫 geo_translate 進行平移
        output_folder="Geometry",  # 輸出資料夾
    ),
    Method(  # 定義「翻轉」幾何轉換方法
        id="geo_flip",  # 方法識別字
        category="幾何轉換",  # 所屬分類
        name="翻轉",  # 顯示名稱
        description="將影像做水平、垂直或兩軸的鏡像翻轉。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("flip_code", "方向", "buttons", default=1,  # 參數：翻轉方向選擇
                      choices=[(0, "垂直"), (1, "水平"), (2, "兩軸")],  # 方向選項：垂直/水平/兩軸
                      help="翻轉方向：垂直（上下）、水平（左右）或兩軸同時翻轉。"),  # 參數說明
        ],
        fn=lambda img, p: geo_flip(img, p["flip_code"]),  # 呼叫 geo_flip 進行翻轉
        output_folder="Geometry",  # 輸出資料夾
    ),

    # ── 各種濾波 ──
    Method(  # 定義「高斯模糊」濾波方法
        id="filter_gaussian",  # 方法識別字
        category="各種濾波",  # 所屬分類
        name="高斯模糊",  # 顯示名稱
        description="以高斯權重平滑影像，自然地降噪與柔化，是最常用的模糊方式。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("ksize", "核心大小", "int", default=3, min=0, max=15, step=1,  # 參數：濾波視窗大小（整數）
                      help="濾波視窗大小（換算為 2k+1 的奇數）。值越大模糊越強。"),  # 參數說明
            ParamSpec("sigma_x10", "Sigma x10", "int", default=10, min=1, max=50, step=1,  # 參數：高斯標準差乘以 10（整數）
                      help="高斯分布的標準差（會除以 10）。值越大模糊範圍越廣、越柔和。"),  # 參數說明
        ],
        fn=lambda img, p: filter_gaussian(img, p["ksize"], p["sigma_x10"]),  # 呼叫 filter_gaussian 進行高斯模糊
        output_folder="Filtered",  # 輸出資料夾
    ),
    Method(  # 定義「均值模糊」濾波方法
        id="filter_box",  # 方法識別字
        category="各種濾波",  # 所屬分類
        name="均值模糊",  # 顯示名稱
        description="以鄰域平均值平滑影像，運算快速但邊緣會比較糊。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("ksize", "核心大小", "int", default=3, min=0, max=15, step=1,  # 參數：濾波視窗大小（整數）
                      help="濾波視窗大小（換算為 2k+1 的奇數）。值越大模糊越強。"),  # 參數說明
        ],
        fn=lambda img, p: filter_box(img, p["ksize"]),  # 呼叫 filter_box 進行均值模糊
        output_folder="Filtered",  # 輸出資料夾
    ),
    Method(  # 定義「中值濾波」方法
        id="filter_median",  # 方法識別字
        category="各種濾波",  # 所屬分類
        name="中值濾波",  # 顯示名稱
        description="取鄰域的中位數，特別擅長去除椒鹽雜訊，同時保留邊緣。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("ksize", "核心大小", "int", default=3, min=0, max=15, step=1,  # 參數：濾波視窗大小（整數）
                      help="濾波視窗大小（換算為 2k+1 的奇數）。值越大模糊越強。"),  # 參數說明
        ],
        fn=lambda img, p: filter_median(img, p["ksize"]),  # 呼叫 filter_median 進行中值濾波
        output_folder="Filtered",  # 輸出資料夾
    ),
    Method(  # 定義「雙邊濾波」方法
        id="filter_bilateral",  # 方法識別字
        category="各種濾波",  # 所屬分類
        name="雙邊濾波",  # 顯示名稱
        description="在平滑的同時保留邊緣的濾波，常用於影像美化與保邊降噪。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("d", "直徑 d", "int", default=9, min=1, max=30, step=1,  # 參數：鄰域直徑 d（整數）
                      help="每個像素參考的鄰域直徑。值越大運算越慢、平滑範圍越廣。"),  # 參數說明
            ParamSpec("sigma_color", "顏色 Sigma", "int", default=75, min=1, max=150, step=1,  # 參數：顏色相似度標準差（整數）
                      help="顏色相似度容忍度。值越大，色差較大的像素也會被一起平滑。"),  # 參數說明
            ParamSpec("sigma_space", "空間 Sigma", "int", default=75, min=1, max=150, step=1,  # 參數：空間距離標準差（整數）
                      help="空間距離容忍度。值越大，較遠的像素也會影響平滑。"),  # 參數說明
        ],
        fn=lambda img, p: filter_bilateral(img, p["d"], p["sigma_color"], p["sigma_space"]),  # 呼叫 filter_bilateral 進行雙邊濾波
        output_folder="Filtered",  # 輸出資料夾
    ),

    # ── 頻率域處理 ──
    Method(  # 定義「傅立葉變換（頻譜）」方法
        id="fourier_spectrum",  # 方法識別字
        category="頻率域處理",  # 所屬分類
        name="傅立葉變換（頻譜）",  # 顯示名稱
        description="將影像轉到頻率域並顯示頻譜，中央為低頻、外圍為高頻，用於分析週期性紋理。",  # 方法說明
        params=[],  # 無可調參數
        fn=lambda img, p: fourier_spectrum(img),  # 呼叫 fourier_spectrum 計算並顯示頻譜
        output_folder="Fourier",  # 輸出資料夾
    ),
    Method(  # 定義「傅立葉反轉換（頻域濾波）」方法
        id="fourier_filter",  # 方法識別字
        category="頻率域處理",  # 所屬分類
        name="傅立葉反轉換（頻域濾波）",  # 顯示名稱
        description="在頻率域套用低通或高通遮罩後轉回影像：低通使其模糊、高通凸顯邊緣。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("filter_type", "濾波類型", "buttons", default=0,  # 參數：濾波類型選擇
                      choices=[(0, "低通(模糊)"), (1, "高通(邊緣)")],  # 類型選項：低通/高通
                      help="低通保留低頻使影像模糊；高通保留高頻凸顯邊緣。"),  # 參數說明
            ParamSpec("cutoff", "截止半徑", "int", default=30, min=1, max=150, step=1,  # 參數：頻域遮罩截止半徑（整數）
                      help="頻率域遮罩的半徑。低通時越大越清晰；高通時越大邊緣越少。"),  # 參數說明
        ],
        fn=lambda img, p: fourier_filter(img, p["filter_type"], p["cutoff"]),  # 呼叫 fourier_filter 進行頻域濾波
        output_folder="Fourier",  # 輸出資料夾
    ),
    Method(  # 定義「FFT 正轉換（暫存相位）」方法
        id="fft_forward",  # 方法識別字
        category="頻率域處理",  # 所屬分類
        name="FFT 正轉換（暫存相位）",  # 顯示名稱
        description="做傅立葉正轉換輸出頻譜圖並暫存相位，可接著對頻譜套用其他處理，最後再用 FFT 反轉換還原。",  # 方法說明
        params=[],  # 無可調參數
        fn=lambda img, p: fft_forward(img),  # 呼叫 fft_forward 進行 FFT 正轉換並暫存相位
        output_folder="Fourier",  # 輸出資料夾
    ),
    Method(  # 定義「FFT 反轉換（還原空間域）」方法
        id="fft_inverse",  # 方法識別字
        category="頻率域處理",  # 所屬分類
        name="FFT 反轉換（還原空間域）",  # 顯示名稱
        description="用暫存的相位與處理後的頻譜還原回空間域影像，需搭配前面的 FFT 正轉換使用。",  # 方法說明
        params=[],  # 無可調參數
        fn=lambda img, p: fft_inverse(img),  # 呼叫 fft_inverse 進行 FFT 反轉換還原影像
        output_folder="Fourier",  # 輸出資料夾
    ),

    # ── 輪廓編碼 ──
    Method(  # 定義「八方鍊碼」輪廓編碼方法
        id="chain_code_8",  # 方法識別字
        category="輪廓編碼",  # 所屬分類
        name="八方鍊碼",  # 顯示名稱
        description="取最大物件輪廓並沿邊界計算 0~7 的方向編碼，在影像上畫出邊界與鍊碼，用於形狀描述。建議先二值化。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("min_area", "最小面積", "int", default=100, min=0, max=2000, step=1,  # 參數：最小面積過濾門檻（整數）
                      help="面積小於此值的輪廓會被忽略，用於過濾雜訊。"),  # 參數說明
        ],
        fn=lambda img, p: chain_code_8(img, p["min_area"]),  # 呼叫 chain_code_8 計算八方向鍊碼
        output_folder="Chain_Code",  # 輸出資料夾
    ),

    # ── 透視校正 ──
    Method(  # 定義「透視變換校正」方法
        id="perspective_correct",  # 方法識別字
        category="透視校正",  # 所屬分類
        name="透視變換校正",  # 顯示名稱
        description="自動偵測影像中最大的四邊形（例如文件、牌子）並校正成俯視矩形，類似手機掃描文件的效果。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("canny_low", "低門檻", "int", default=50, min=0, max=255, step=1,  # 參數：邊緣偵測弱門檻（整數）
                      help="邊緣偵測的弱門檻，用於找出文件邊框。值越低偵測到的邊越多。"),  # 參數說明
            ParamSpec("canny_high", "高門檻", "int", default=150, min=0, max=255, step=1,  # 參數：邊緣偵測強門檻（整數）
                      help="邊緣偵測的強門檻，建議約為低門檻的 2~3 倍。"),  # 參數說明
        ],
        fn=lambda img, p: perspective_correct(img, p["canny_low"], p["canny_high"]),  # 呼叫 perspective_correct 進行透視校正
        output_folder="Perspective",  # 輸出資料夾
    ),

    # ── 影像修復 ──
    Method(  # 定義「自動修復（亮部瑕疵）」方法
        id="inpaint_auto",  # 方法識別字
        category="影像修復",  # 所屬分類
        name="自動修復（亮部瑕疵）",  # 顯示名稱
        description="自動將過亮的刮痕/瑕疵區域當作需修復的遮罩，用周圍像素自然填補，去除白點、文字或銳點。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("thresh", "亮度門檻", "int", default=240, min=100, max=255, step=1,  # 參數：待修復區的亮度門檻（整數）
                      help="亮度高於此值的像素視為待修復區（如反光、刮痕）。值越低修復範圍越大。"),  # 參數說明
            ParamSpec("radius", "修復半徑", "int", default=3, min=1, max=15, step=1,  # 參數：修復參考半徑（整數）
                      help="修復時參考周圍像素的半徑。值越大越平滑但可能模糊。"),  # 參數說明
            ParamSpec("method", "演算法", "buttons", default=0,  # 參數：修復演算法選擇
                      choices=[(0, "Telea"), (1, "NS")],  # 演算法選項：Telea/NS
                      help="修復演算法：Telea 快速擴散；NS 基於流體的方法，邊界較自然。"),  # 參數說明
        ],
        fn=lambda img, p: inpaint_auto(img, p["thresh"], p["radius"], p["method"]),  # 呼叫 inpaint_auto 進行自動修復
        output_folder="Inpaint",  # 輸出資料夾
    ),

    # ── 輪廓分析 ──
    Method(  # 定義「輪廓特徵分析」方法
        id="contour_analyze",  # 方法識別字
        category="輪廓分析",  # 所屬分類
        name="輪廓特徵分析",  # 顯示名稱
        description="找出所有物件輪廓，畫出輪廓並標註每個物件的面積與重心，左上角顯示物件總數。建議先二值化。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("min_area", "最小面積", "int", default=100, min=0, max=2000, step=1,  # 參數：最小面積過濾門檻（整數）
                      help="面積小於此值的輪廓會被忽略，用於過濾雜訊。"),  # 參數說明
        ],
        fn=lambda img, p: analyze_contours(img, p["min_area"]),  # 呼叫 analyze_contours 進行輪廓特徵分析
        output_folder="Contour_Analysis",  # 輸出資料夾
    ),

    # ── 霍夫偵測 ──
    Method(  # 定義「霍夫直線」偵測方法
        id="hough_lines",  # 方法識別字
        category="霍夫偵測",  # 所屬分類
        name="霍夫直線",  # 顯示名稱
        description="偵測影像中的直線並以紅線標示，常用於偵測道路、表格、建築邊界等線條。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("threshold", "累積門檻", "int", default=100, min=10, max=300, step=1,  # 參數：直線判定累積票數門檻（整數）
                      help="判定為直線所需的累積票數。值越大，只保留越明顯的長直線。"),  # 參數說明
            ParamSpec("min_length", "最短線長", "int", default=50, min=0, max=300, step=1,  # 參數：線段最短長度（整數）
                      help="線段最短長度（像素），短於此值的線會被捨棄。"),  # 參數說明
            ParamSpec("max_gap", "最大間隙", "int", default=10, min=0, max=100, step=1,  # 參數：同線最大斷裂間隙（整數）
                      help="同一直線上允許的最大斷裂間隙，越大越容易把斷線連起來。"),  # 參數說明
        ],
        fn=lambda img, p: hough_lines(img, p["threshold"], p["min_length"], p["max_gap"]),  # 呼叫 hough_lines 偵測直線
        output_folder="Hough",  # 輸出資料夾
    ),
    Method(  # 定義「霍夫圓」偵測方法
        id="hough_circles",  # 方法識別字
        category="霍夫偵測",  # 所屬分類
        name="霍夫圓",  # 顯示名稱
        description="偵測影像中的圓形並標示圓心與圓周，常用於偵測硬幣、球體、孔洞等圓形物件。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("min_dist", "圓心最小間距", "int", default=20, min=1, max=200, step=1,  # 參數：兩圓心最小距離（整數）
                      help="兩圓心之間的最小距離。值太小會重複偵測，太大會漏掉相鄰的圓。"),  # 參數說明
            ParamSpec("param2", "偵測門檻", "int", default=40, min=10, max=150, step=1,  # 參數：圓偵測累積門檻（整數）
                      help="圓偵測的累積門檻。值越小偵測到的圓越多（含假圓），越大越嚴格。"),  # 參數說明
            ParamSpec("min_radius", "最小半徑", "int", default=0, min=0, max=200, step=1,  # 參數：偵測最小圓半徑（整數）
                      help="要偵測的最小圓半徑（像素）。0 表示不限制。"),  # 參數說明
            ParamSpec("max_radius", "最大半徑", "int", default=0, min=0, max=300, step=1,  # 參數：偵測最大圓半徑（整數）
                      help="要偵測的最大圓半徑（像素）。0 表示不限制。"),  # 參數說明
        ],
        fn=lambda img, p: hough_circles(img, p["min_dist"], p["param2"], p["min_radius"], p["max_radius"]),  # 呼叫 hough_circles 偵測圓
        output_folder="Hough",  # 輸出資料夾
    ),

    # ── 影像分割 ──
    Method(  # 定義「分水嶺分割」方法
        id="watershed",  # 方法識別字
        category="影像分割",  # 所屬分類
        name="分水嶺分割",  # 顯示名稱
        description="自動標記前景與背景，分離相黏在一起的物件（例如重疊的細胞、硬幣），以紅色邊界標示各區塊。建議用於前景明顯的影像。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("fg_ratio", "前景門檻%", "int", default=50, min=10, max=90, step=1,  # 參數：確定前景的門檻百分比（整數）
                      help="判定確定前景的嚴格程度（百分比）。值越大前景區越小、物件分得越開。"),  # 參數說明
        ],
        fn=lambda img, p: watershed_segment(img, p["fg_ratio"]),  # 呼叫 watershed_segment 進行分水嶺分割
        output_folder="Watershed",  # 輸出資料夾
    ),

    # ── 模板比對 ──
    Method(  # 定義「可移動模板比對」方法
        id="template_match",  # 方法識別字
        category="模板比對",  # 所屬分類
        name="可移動模板比對",  # 顯示名稱
        description="拖動 X/Y 位置滑桿移動紅框，選取影像中任意區域作為模板，在整張影像中找出相似區塊（綠框）。",  # 方法說明
        params=[  # 參數清單
            ParamSpec("region_pct", "模板大小%", "int", default=20, min=5, max=50, step=1,  # 參數：模板區域佔比百分比（整數）
                      help="模板區域佔影像的大小（百分比）。需略大於要尋找的圖案。"),  # 參數說明
            ParamSpec("threshold", "相似度門檻%", "int", default=80, min=50, max=99, step=1,  # 參數：比對相似度門檻百分比（整數）
                      help="比對相似度門檻。值越高只保留越相似的區塊，越低找到越多。"),  # 參數說明
            ParamSpec("x_pct", "模板中心 X%", "int", default=50, min=0, max=100, step=1,  # 參數：模板中心水平位置百分比（整數）
                      help="紅框（模板）中心的水平位置（百分比），可在預覽圖上點擊移動。"),  # 參數說明
            ParamSpec("y_pct", "模板中心 Y%", "int", default=50, min=0, max=100, step=1,  # 參數：模板中心垂直位置百分比（整數）
                      help="紅框（模板）中心的垂直位置（百分比），可在預覽圖上點擊移動。"),  # 參數說明
        ],
        fn=lambda img, p: template_match_center(img, p["region_pct"], p["threshold"], p["x_pct"], p["y_pct"]),  # 呼叫 template_match_center 進行模板比對
        output_folder="Template_Match",  # 輸出資料夾
    ),
]


def get_method(method_id: str) -> Method:  # 依識別字取得對應的 Method 物件
    for m in METHODS:  # 走訪所有註冊的方法
        if m.id == method_id:  # 若識別字相符
            return m  # 回傳該方法
    raise KeyError(f"Unknown method: {method_id}")  # 找不到時拋出 KeyError


def get_categories() -> list[str]:  # 取得所有不重複的分類名稱（依出現順序）
    seen = []  # 已收錄的分類清單
    for m in METHODS:  # 走訪所有方法
        if m.category not in seen:  # 若分類尚未收錄
            seen.append(m.category)  # 加入清單
    return seen  # 回傳分類清單


def get_methods_by_category(category: str) -> list[Method]:  # 取得指定分類下的所有方法
    return [m for m in METHODS if m.category == category]  # 篩選分類相符的方法並回傳
