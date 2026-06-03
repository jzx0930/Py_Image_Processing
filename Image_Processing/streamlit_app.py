import sys
import os
import io
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from processing.registry import get_categories, get_methods_by_category, get_method
from processing.pipeline import apply_step, run_pipeline, make_step_label


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(img: np.ndarray) -> Image.Image:
    if len(img.shape) == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


st.set_page_config(page_title="影像處理工具", layout="wide")
st.title("影像處理工具")

# 注入 CSS：標題區固定高度，避免文字斷行導致圖片排列高低不一
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
/* sidebar radio 選項按鍵排列式樣式 */
[data-testid="stSidebarContent"] .stRadio > div {
    gap: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ── session_state 初始化
if "original" not in st.session_state:
    st.session_state.original = None
if "steps" not in st.session_state:
    st.session_state.steps = []

# ── 載入圖片
uploaded = st.file_uploader("載入圖片", type=["png", "jpg", "jpeg", "bmp"])
if uploaded is not None:
    if st.session_state.original is None or uploaded.file_id != st.session_state.get("last_file_id"):
        st.session_state.original = pil_to_bgr(Image.open(uploaded))
        st.session_state.steps = []
        st.session_state.last_file_id = uploaded.file_id

if st.session_state.original is None:
    st.info("請上傳一張圖片開始使用")
    st.stop()

original = st.session_state.original
steps = st.session_state.steps

# ── 側邊欄控制項
st.sidebar.markdown("## 處理模式")
mode = st.sidebar.radio("模式", ["逐步即時", "批次清單"], horizontal=True)

st.sidebar.markdown("---")
st.sidebar.markdown("## 分類")
selected_cat = st.sidebar.radio("選擇分類", get_categories(), label_visibility="collapsed")

st.sidebar.markdown("## 方法")
methods_in_cat = get_methods_by_category(selected_cat)
selected_name = st.sidebar.radio(
    "選擇方法", [m.name for m in methods_in_cat], label_visibility="collapsed"
)
selected_method = next(m for m in methods_in_cat if m.name == selected_name)

st.sidebar.markdown("---")
st.sidebar.markdown("## 參數調整")
param_values = {}
for p in selected_method.params:
    if p.kind in ("choice", "buttons"):
        displays = [d for _, d in p.choices]
        vals = [v for v, _ in p.choices]
        idx = st.sidebar.radio(
            p.label, range(len(displays)),
            format_func=lambda i, d=displays: d[i],
            index=int(p.default),
            horizontal=(p.kind == "buttons"),
        )
        param_values[p.key] = vals[idx]
    else:
        param_values[p.key] = st.sidebar.slider(
            p.label, int(p.min), int(p.max), int(p.default)
        )

current_step = {"method": selected_method.id, "params": param_values}

# ── 計算工作圖與即時預覽
working = run_pipeline(original, steps)
try:
    whatif = apply_step(working, current_step)
except Exception:
    whatif = working

# ── 主區域：原圖 | 即時預覽（中間）| 已套用結果
col_orig, col_preview, col_result = st.columns(3)
with col_orig:
    st.markdown('<div class="col-header">原圖</div>', unsafe_allow_html=True)
    st.image(bgr_to_pil(original), use_container_width=True)
with col_preview:
    st.markdown(
        f'<div class="col-header">即時預覽：{selected_cat} / {selected_name}</div>',
        unsafe_allow_html=True,
    )
    st.image(bgr_to_pil(whatif), use_container_width=True)
with col_result:
    st.markdown(
        f'<div class="col-header">已套用結果（{len(steps)} 步）</div>',
        unsafe_allow_html=True,
    )
    st.image(bgr_to_pil(working), use_container_width=True)

# ── 操作按鈕
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)

with c1:
    btn_label = "套用此步驟" if mode == "逐步即時" else "加入清單"
    if st.button(btn_label, use_container_width=True):
        try:
            apply_step(working, current_step)
            st.session_state.steps.append(current_step)
            st.rerun()
        except Exception as e:
            st.error(f"無法套用：{e}")

with c2:
    if mode == "批次清單":
        st.button("執行全部", use_container_width=True, disabled=len(steps) == 0)

with c3:
    if st.button("復原上一步", use_container_width=True, disabled=len(steps) == 0):
        st.session_state.steps.pop()
        st.rerun()

with c4:
    if st.button("重設為原圖", use_container_width=True, disabled=len(steps) == 0):
        st.session_state.steps = []
        st.rerun()

# ── 步驟歷史
if steps:
    st.markdown("### 步驟歷史")
    for i, s in enumerate(steps):
        st.markdown(f"`{i+1}.` {make_step_label(s)}")

    st.markdown("---")
    buf = io.BytesIO()
    bgr_to_pil(working).save(buf, format="PNG")
    st.download_button(
        label="下載最終結果",
        data=buf.getvalue(),
        file_name="pipeline_result.png",
        mime="image/png",
        use_container_width=True,
    )
