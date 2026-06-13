import numpy as np  # 匯入 NumPy 數值運算函式庫，並取別名 np
from processing.registry import get_method  # 從 registry 模組匯入 get_method，用來依名稱取得處理方法


def apply_step(image: np.ndarray, step: dict) -> np.ndarray:  # 定義套用單一處理步驟的函式，step 為描述方法與參數的字典
    method = get_method(step["method"])  # 依步驟中的方法名稱取得對應的方法物件
    return method.fn(image, step["params"])  # 以該方法的函式處理影像並帶入參數，回傳結果


def run_pipeline(original: np.ndarray, steps: list[dict]) -> np.ndarray:  # 定義執行整條流水線的函式，steps 為步驟清單
    result = original.copy()  # 複製原始影像作為起始結果，避免修改到原圖
    for step in steps:  # 依序處理每一個步驟
        try:  # 嘗試套用此步驟（容錯處理）
            result = apply_step(result, step)  # 套用步驟並把輸出當作下一步的輸入
        except Exception:  # 若此步驟發生任何例外
            pass  # 忽略錯誤並保留前一步的結果，繼續下一步
    return result  # 回傳所有步驟處理後的最終影像


def make_step_label(step: dict) -> str:  # 定義產生步驟顯示文字的函式，回傳字串
    method = get_method(step["method"])  # 依方法名稱取得方法物件，以便取用分類與名稱
    if not step["params"]:  # 若此步驟沒有任何參數
        return f"{method.category} / {method.name}"  # 只回傳「分類 / 名稱」
    param_str = "  ".join(f"{k}={v}" for k, v in step["params"].items())  # 把所有參數組成「key=value」並以兩個空白連接
    return f"{method.category} / {method.name}  ({param_str})"  # 回傳含參數說明的完整標籤
