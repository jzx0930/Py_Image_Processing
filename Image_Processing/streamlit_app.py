import sys  # 匯入 sys 模組，用於操作 Python 執行環境（例如模組搜尋路徑）
import os  # 匯入 os 模組，用於檔案與路徑操作
import io  # 匯入 io 模組，用於在記憶體中讀寫位元組資料（例如圖片下載）
sys.path.insert(0, os.path.dirname(__file__))  # 把目前檔案所在資料夾加入模組搜尋路徑最前面，確保能匯入本地模組

import cv2  # 匯入 OpenCV 函式庫，用於影像處理
import numpy as np  # 匯入 NumPy，用於數值與陣列運算
import streamlit as st  # 匯入 Streamlit，用於建立網頁互動介面
from PIL import Image  # 從 Pillow 匯入 Image 類別，用於圖片格式轉換

from processing.registry import get_categories, get_methods_by_category, get_method, METHODS  # 匯入方法註冊表相關函式（取得分類、方法等）
from processing.pipeline import apply_step, run_pipeline, make_step_label  # 匯入流程處理相關函式（套用步驟、執行整條流程、產生步驟標籤）


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:  # 定義函式：把 PIL 圖片轉換為 OpenCV 使用的 BGR 陣列
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)  # 先轉成 RGB 陣列再轉成 BGR，回傳 NumPy 陣列


def bgr_to_pil(img: np.ndarray) -> Image.Image:  # 定義函式：把 OpenCV 的 BGR 陣列轉換為 PIL 圖片
    if len(img.shape) == 2:  # 若影像只有兩個維度，代表是灰階圖
        return Image.fromarray(img)  # 灰階圖直接轉換為 PIL 圖片
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 彩色圖先把 BGR 轉成 RGB，再轉為 PIL 圖片


st.set_page_config(page_title="影像處理工具", layout="wide")  # 設定網頁標題與版面為寬版
st.title("影像處理工具")  # 在頁面上顯示主標題

st.markdown("""
<style>
.col-header {
    height: 3.2rem;
    display: flex;
    align-items: flex-end;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
    line-height: 1.3;
}
.method-desc {
    background: #1b3a4b;
    border-left: 3px solid #4dabf7;
    padding: 0.5rem 0.7rem;
    border-radius: 4px;
    font-size: 0.9rem;
    line-height: 1.5;
    margin: 0.3rem 0 0.6rem 0;
}
[data-testid="stSidebarContent"] .stRadio > div { gap: 0.3rem; }
</style>
""", unsafe_allow_html=True)  # 注入自訂 CSS 樣式（標題列與方法說明框的外觀），允許使用原始 HTML

# ── session_state 初始化
for key, val in [("original", None), ("steps", []), ("inspect_idx", None)]:  # 逐一檢查需要的狀態鍵與其預設值
    if key not in st.session_state:  # 若該狀態鍵尚未存在於 session_state
        st.session_state[key] = val  # 就設定為預設值（原圖、步驟清單、檢視索引）

# ── 載入圖片
uploaded = st.file_uploader("載入圖片", type=["png", "jpg", "jpeg", "bmp"])  # 顯示上傳元件，限制可上傳的圖片格式
if uploaded is not None:  # 若使用者有上傳檔案
    if st.session_state.original is None or uploaded.file_id != st.session_state.get("last_file_id"):  # 若還沒有原圖，或上傳的是新檔案
        st.session_state.original = pil_to_bgr(Image.open(uploaded))  # 讀取圖片並轉成 BGR 存為原圖
        st.session_state.steps = []  # 清空已套用的步驟清單
        st.session_state.inspect_idx = None  # 清除目前檢視的步驟索引
        st.session_state.last_file_id = uploaded.file_id  # 記錄這次的檔案 ID，用來判斷下次是否為新檔案

if st.session_state.original is None:  # 若目前沒有任何原圖
    st.info("請上傳一張圖片開始使用")  # 顯示提示訊息
    st.stop()  # 停止後續程式執行，等待使用者上傳

original = st.session_state.original  # 取出原圖，方便後續使用
steps = st.session_state.steps  # 取出步驟清單，方便後續使用

# ── 側邊欄：模式 + 分類 + 方法 + 參數
st.sidebar.markdown("## 處理模式")  # 在側邊欄顯示「處理模式」標題
mode = st.sidebar.radio("模式", ["逐步即時", "批次清單"], horizontal=True)  # 提供兩種處理模式的單選按鈕

st.sidebar.markdown("---")  # 在側邊欄畫一條分隔線
st.sidebar.markdown("## 分類")  # 顯示「分類」標題
selected_cat = st.sidebar.radio("選擇分類", get_categories(), label_visibility="collapsed")  # 列出所有分類供選擇，隱藏標籤文字

st.sidebar.markdown("## 方法")  # 顯示「方法」標題
methods_in_cat = get_methods_by_category(selected_cat)  # 取得所選分類底下的所有方法
selected_name = st.sidebar.radio(
    "選擇方法", [m.name for m in methods_in_cat], label_visibility="collapsed"
)  # 列出該分類的方法名稱供選擇，隱藏標籤文字
selected_method = next(m for m in methods_in_cat if m.name == selected_name)  # 找出名稱符合的方法物件

if selected_method.description:  # 若該方法有說明文字
    st.sidebar.markdown(
        f'<div class="method-desc">📖 {selected_method.description}</div>',
        unsafe_allow_html=True,
    )  # 在側邊欄以自訂樣式顯示方法說明

st.sidebar.markdown("---")  # 在側邊欄畫一條分隔線
st.sidebar.markdown("## 參數調整")  # 顯示「參數調整」標題

# 模板比對：在預覽圖點擊移動紅框，位置存於 session_state
is_template = selected_method.id == "template_match"  # 判斷目前方法是否為模板比對
if is_template:  # 若是模板比對方法
    for k in ("tpl_x_pct", "tpl_y_pct"):  # 逐一檢查紅框的 X、Y 百分比位置狀態鍵
        if k not in st.session_state:  # 若該狀態鍵尚未存在
            st.session_state[k] = 50  # 預設位置設為 50（置中）

param_values = {}  # 建立空字典，用來蒐集各參數的目前數值
for p in selected_method.params:  # 逐一處理該方法的每個參數
    p_help = p.help or None  # 取得參數的說明文字，若沒有則為 None
    if p.kind in ("choice", "buttons"):  # 若參數類型是選項或按鈕
        displays = [d for _, d in p.choices]  # 取出每個選項的顯示文字
        vals = [v for v, _ in p.choices]  # 取出每個選項對應的實際值
        idx = st.sidebar.radio(
            p.label, range(len(displays)),
            format_func=lambda i, d=displays: d[i],
            index=int(p.default),
            horizontal=(p.kind == "buttons"),
            help=p_help,
        )  # 顯示單選按鈕，回傳選到的索引（按鈕類型則水平排列）
        param_values[p.key] = vals[idx]  # 把選到的實際值存入參數字典
    elif is_template and p.key in ("x_pct", "y_pct"):  # 若是模板比對且參數為 X 或 Y 位置百分比
        skey = "tpl_x_pct" if p.key == "x_pct" else "tpl_y_pct"  # 決定對應的 session_state 鍵名
        val = st.sidebar.slider(
            p.label, int(p.min), int(p.max), int(st.session_state[skey]),
            help=p_help,
        )  # 顯示滑桿，初始值取自 session_state
        st.session_state[skey] = val  # 把滑桿值寫回 session_state
        param_values[p.key] = val  # 把滑桿值存入參數字典
    else:  # 其他一般數值參數
        param_values[p.key] = st.sidebar.slider(
            p.label, int(p.min), int(p.max), int(p.default),
            help=p_help,
        )  # 顯示滑桿並把選到的值存入參數字典

if is_template:  # 若是模板比對方法
    st.sidebar.caption("💡 也可直接在中間「即時預覽」圖上點一下，紅框會移到該位置。")  # 在側邊欄顯示操作提示

current_step = {"method": selected_method.id, "params": param_values}  # 組合目前要套用的步驟（方法 ID 與參數）

# ── 計算工作圖與即時預覽
working = run_pipeline(original, steps)  # 把已套用的步驟全部執行一次，得到目前的工作影像
try:  # 嘗試計算套用目前步驟後的預覽
    whatif = apply_step(working, current_step)  # 在工作影像上套用目前步驟，得到預覽影像
except Exception:  # 若套用過程出錯
    whatif = working  # 預覽就退回為原本的工作影像

# ── 主區域：當前影像 | 效果預覽
col_current, col_preview = st.columns(2)  # 把主畫面切成左右兩欄
with col_current:  # 在左欄中
    st.markdown('<div class="col-header">當前影像</div>', unsafe_allow_html=True)  # 顯示「當前影像」標題
    st.image(bgr_to_pil(working), use_container_width=True)  # 顯示工作影像，寬度填滿欄位
with col_preview:  # 在右欄中
    st.markdown(
        f'<div class="col-header">效果預覽：{selected_cat} / {selected_name}</div>',
        unsafe_allow_html=True,
    )  # 顯示「效果預覽」標題，包含目前分類與方法名稱
    if is_template:  # 若是模板比對方法
        try:  # 嘗試使用可點擊取得座標的圖片元件
            from streamlit_image_coordinates import streamlit_image_coordinates  # 匯入第三方點擊座標元件
            coords = streamlit_image_coordinates(
                bgr_to_pil(whatif), use_column_width="always", key="tpl_click"
            )  # 顯示可點擊的預覽圖，回傳點擊座標
            if coords is not None and coords.get("width"):  # 若有點擊且取得到圖片寬度
                new_x = int(coords["x"] / coords["width"] * 100)  # 把點擊的 X 座標換算成百分比
                new_y = int(coords["y"] / coords["height"] * 100)  # 把點擊的 Y 座標換算成百分比
                if (new_x, new_y) != (st.session_state.tpl_x_pct, st.session_state.tpl_y_pct):  # 若新位置與目前位置不同
                    st.session_state.tpl_x_pct = max(0, min(100, new_x))  # 更新 X 位置並限制在 0~100
                    st.session_state.tpl_y_pct = max(0, min(100, new_y))  # 更新 Y 位置並限制在 0~100
                    st.rerun()  # 重新執行頁面以反映新位置
        except ModuleNotFoundError:  # 若沒有安裝點擊座標元件
            st.image(bgr_to_pil(whatif), use_container_width=True)  # 改用一般方式顯示預覽圖
            st.caption("（未安裝 streamlit-image-coordinates，請用左側 X/Y 滑桿移動紅框）")  # 顯示替代操作提示
    else:  # 非模板比對的一般方法
        st.image(bgr_to_pil(whatif), use_container_width=True)  # 直接顯示預覽圖
    if selected_method.description:  # 若該方法有說明文字
        st.caption(f"📖 {selected_method.description}")  # 在預覽圖下方顯示方法說明

# ── 操作按鈕
st.markdown("---")  # 畫一條分隔線
c1, c2, c3, c4 = st.columns(4)  # 把按鈕區切成四欄

with c1:  # 第一欄
    btn_label = "套用此步驟" if mode == "逐步即時" else "加入清單"  # 依模式決定按鈕文字
    if st.button(btn_label, use_container_width=True):  # 若按下該按鈕
        try:  # 嘗試套用步驟
            apply_step(working, current_step)  # 先試套用一次，確認不會出錯
            st.session_state.steps.append(current_step)  # 把步驟加入步驟清單
            st.session_state.inspect_idx = None  # 清除目前檢視的步驟索引
            st.rerun()  # 重新執行頁面以更新畫面
        except Exception as e:  # 若套用失敗
            st.error(f"無法套用：{e}")  # 顯示錯誤訊息

with c2:  # 第二欄
    if mode == "批次清單":  # 只有在批次清單模式才顯示
        st.button("執行全部", use_container_width=True, disabled=len(steps) == 0)  # 顯示「執行全部」按鈕，無步驟時停用

with c3:  # 第三欄
    if st.button("復原上一步", use_container_width=True, disabled=len(steps) == 0):  # 若按下且有步驟可復原
        st.session_state.steps.pop()  # 移除最後一個步驟
        st.session_state.inspect_idx = None  # 清除目前檢視的步驟索引
        st.rerun()  # 重新執行頁面以更新畫面

with c4:  # 第四欄
    if st.button("重設為原圖", use_container_width=True, disabled=len(steps) == 0):  # 若按下且有步驟可重設
        st.session_state.steps = []  # 清空所有步驟
        st.session_state.inspect_idx = None  # 清除目前檢視的步驟索引
        st.rerun()  # 重新執行頁面以更新畫面

# ── 步驟歷史 + 點選預覽
if steps:  # 若有任何已套用的步驟
    st.markdown("### 步驟歷史（點 ▶ 可預覽該步驟結果）")  # 顯示步驟歷史標題
    for i, s in enumerate(steps):  # 逐一列出每個步驟（i 為索引，s 為步驟內容）
        col_lbl, col_btn = st.columns([6, 1])  # 切成左右兩欄（標籤與按鈕，比例 6:1）
        with col_lbl:  # 左欄顯示步驟標籤
            st.markdown(f"`{i+1}.` {make_step_label(s)}")  # 顯示步驟編號與描述文字
        with col_btn:  # 右欄顯示檢視按鈕
            is_selected = st.session_state.inspect_idx == i  # 判斷目前檢視的是否為這個步驟
            btn_txt = "■" if is_selected else "▶"  # 已選顯示停止符號，未選顯示播放符號
            if st.button(btn_txt, key=f"inspect_{i}", use_container_width=True):  # 若按下該檢視按鈕
                st.session_state.inspect_idx = None if is_selected else i  # 已選則取消，否則設為檢視此步驟
                st.rerun()  # 重新執行頁面以更新畫面

    # 步驟點選預覽區
    idx = st.session_state.inspect_idx  # 取得目前檢視的步驟索引
    if idx is not None and idx < len(steps):  # 若有選取且索引有效
        st.markdown(f"#### 🔍 步驟 {idx+1}：{make_step_label(steps[idx])}")  # 顯示所檢視步驟的標題
        col_a, col_b = st.columns(2)  # 切成左右兩欄（套用前、套用後）
        with col_a:  # 左欄顯示套用前
            label_before = "原圖" if idx == 0 else f"步驟 {idx} 套用後"  # 決定左欄標題文字
            img_before = original if idx == 0 else run_pipeline(original, steps[:idx])  # 計算套用此步驟之前的影像
            st.caption(f"當前影像（{label_before}）")  # 顯示左欄說明文字
            st.image(bgr_to_pil(img_before), use_container_width=True)  # 顯示套用前的影像
        with col_b:  # 右欄顯示套用後
            st.caption(f"套用步驟 {idx+1} 後")  # 顯示右欄說明文字
            st.image(bgr_to_pil(run_pipeline(original, steps[:idx + 1])), use_container_width=True)  # 顯示套用此步驟後的影像

    st.markdown("---")  # 畫一條分隔線
    buf = io.BytesIO()  # 建立記憶體位元組緩衝區，用來暫存圖片
    bgr_to_pil(working).save(buf, format="PNG")  # 把最終工作影像以 PNG 格式存入緩衝區
    st.download_button(
        label="下載最終結果",
        data=buf.getvalue(),
        file_name="pipeline_result.png",
        mime="image/png",
        use_container_width=True,
    )  # 顯示下載按鈕，讓使用者下載最終結果圖片
