from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import cv2

from processing.binarize_fn import binarize_custom, binarize_otsu, binarize_adaptive
from processing.color_space_fn import split_channels, COLOR_MODES
from processing.edge_detect_fn import canny, sobel, laplacian, scharr
from processing.intensity_transform_fn import gamma, log_transform, linear, histogram_eq, clahe
from processing.image_correction_fn import affine


@dataclass
class ParamSpec:
    key: str
    label: str
    kind: str                          # "int" | "float" | "choice" | "buttons"
    default: object = 0
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[list] = None     # list of (value, display_str)


@dataclass
class Method:
    id: str
    category: str
    name: str
    params: list
    fn: Callable
    output_folder: str


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image


METHODS: list[Method] = [
    # ── 色彩空間分離 ─────────────────────────────────────────
    Method(
        id="color_split",
        category="色彩空間分離",
        name="色彩空間分離",
        params=[
            ParamSpec("mode_idx", "色彩空間", "buttons", default=0,
                      choices=[(i, m) for i, m in enumerate(COLOR_MODES)]),
            ParamSpec("ch_idx", "通道", "buttons", default=0,
                      choices=[(0, "B(0)"), (1, "G(1)"), (2, "R(2)")]),
        ],
        fn=lambda img, p: split_channels(img, p["mode_idx"])[
            min(p["ch_idx"], len(split_channels(img, p["mode_idx"])) - 1)
        ],
        output_folder="Color_Space_Channel",
    ),

    # ── 二值化 ─────────────────────────────────────────────
    Method(
        id="binarize_custom",
        category="二值化",
        name="自訂門檻",
        params=[
            ParamSpec("threshold", "門檻值", "int", default=127, min=0, max=255, step=1),
        ],
        fn=lambda img, p: binarize_custom(_to_gray(img), p["threshold"]),
        output_folder="Binarized",
    ),
    Method(
        id="binarize_otsu",
        category="二值化",
        name="Otsu 自動",
        params=[],
        fn=lambda img, p: binarize_otsu(_to_gray(img)),
        output_folder="Binarized",
    ),
    Method(
        id="binarize_adaptive",
        category="二值化",
        name="自適應",
        params=[
            ParamSpec("block_size_idx", "BlockSize", "int", default=4, min=0, max=249, step=1),
            ParamSpec("c_offset", "C 值", "int", default=52, min=0, max=100, step=1),
        ],
        fn=lambda img, p: binarize_adaptive(_to_gray(img), p["block_size_idx"], p["c_offset"]),
        output_folder="Binarized",
    ),

    # ── 邊緣偵測 ───────────────────────────────────────────
    Method(
        id="edge_canny",
        category="邊緣偵測",
        name="Canny",
        params=[
            ParamSpec("low", "低門檻", "int", default=100, min=0, max=255, step=1),
            ParamSpec("high", "高門檻", "int", default=200, min=0, max=255, step=1),
        ],
        fn=lambda img, p: canny(_to_gray(img), p["low"], p["high"]),
        output_folder="Edge_Detected",
    ),
    Method(
        id="edge_sobel",
        category="邊緣偵測",
        name="Sobel",
        params=[
            ParamSpec("ksize_idx", "Kernel Size", "int", default=0, min=0, max=14, step=1),
        ],
        fn=lambda img, p: sobel(_to_gray(img), p["ksize_idx"]),
        output_folder="Edge_Detected",
    ),
    Method(
        id="edge_laplacian",
        category="邊緣偵測",
        name="Laplacian",
        params=[],
        fn=lambda img, p: laplacian(_to_gray(img)),
        output_folder="Edge_Detected",
    ),
    Method(
        id="edge_scharr",
        category="邊緣偵測",
        name="Scharr",
        params=[],
        fn=lambda img, p: scharr(_to_gray(img)),
        output_folder="Edge_Detected",
    ),

    # ── 強度轉換 ───────────────────────────────────────────
    Method(
        id="intensity_gamma",
        category="強度轉換",
        name="Gamma",
        params=[
            ParamSpec("gamma_x100", "Gamma x100", "int", default=100, min=1, max=300, step=1),
        ],
        fn=lambda img, p: gamma(_to_gray(img), p["gamma_x100"] / 100),
        output_folder="Intensity_Transform",
    ),
    Method(
        id="intensity_log",
        category="強度轉換",
        name="對數轉換",
        params=[
            ParamSpec("k_x10", "k x10", "int", default=10, min=1, max=50, step=1),
        ],
        fn=lambda img, p: log_transform(_to_gray(img), p["k_x10"] / 10),
        output_folder="Intensity_Transform",
    ),
    Method(
        id="intensity_linear",
        category="強度轉換",
        name="線性轉換",
        params=[
            ParamSpec("a_x100", "a x100", "int", default=100, min=50, max=300, step=1),
            ParamSpec("b_offset", "b 值", "int", default=100, min=0, max=200, step=1),
        ],
        fn=lambda img, p: linear(_to_gray(img), p["a_x100"] / 100, p["b_offset"] - 100),
        output_folder="Intensity_Transform",
    ),
    Method(
        id="intensity_hist_eq",
        category="強度轉換",
        name="直方圖等化",
        params=[],
        fn=lambda img, p: histogram_eq(_to_gray(img)),
        output_folder="Intensity_Transform",
    ),
    Method(
        id="intensity_clahe",
        category="強度轉換",
        name="CLAHE",
        params=[
            ParamSpec("clip_x10", "clipLimit x10", "int", default=20, min=1, max=40, step=1),
        ],
        fn=lambda img, p: clahe(_to_gray(img), p["clip_x10"] / 10),
        output_folder="Intensity_Transform",
    ),

    # ── 影像校正 ───────────────────────────────────────────
    Method(
        id="correction_affine",
        category="影像校正",
        name="仿射變換",
        params=[
            ParamSpec("dx", "X 偏移", "int", default=30, min=0, max=100, step=1),
            ParamSpec("dy", "Y 偏移", "int", default=30, min=0, max=100, step=1),
        ],
        fn=lambda img, p: affine(img, p["dx"], p["dy"]),
        output_folder="Image_Correction",
    ),
]


def get_method(method_id: str) -> Method:
    for m in METHODS:
        if m.id == method_id:
            return m
    raise KeyError(f"Unknown method: {method_id}")


def get_categories() -> list[str]:
    seen = []
    for m in METHODS:
        if m.category not in seen:
            seen.append(m.category)
    return seen


def get_methods_by_category(category: str) -> list[Method]:
    return [m for m in METHODS if m.category == category]
