import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QListWidget, QSlider, QComboBox,
    QFileDialog, QGroupBox, QSizePolicy, QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage

from processing.registry import get_categories, get_methods_by_category, get_method
from processing.pipeline import apply_step, run_pipeline, make_step_label


def ndarray_to_qpixmap(image: np.ndarray) -> QPixmap:
    if image is None:
        return QPixmap()
    img = np.ascontiguousarray(image)
    if len(img.shape) == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        h, w, ch = img.shape
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class MethodParamPanel(QWidget):
    changed = pyqtSignal()

    def __init__(self, method):
        super().__init__()
        self._method = method
        self._sliders = {}
        self._combos = {}
        self._button_groups = {}      # key -> (QButtonGroup, {value: QRadioButton})
        self._val_labels = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        if not method.params:
            layout.addWidget(QLabel("此方法無需調整參數"))
        else:
            for p in method.params:
                if p.kind == "buttons":
                    # 按鈕選項：名稱在上，下方以多欄按鈕呈現
                    layout.addWidget(QLabel(p.label))
                    grid = QGridLayout()
                    group = QButtonGroup(self)
                    group.setExclusive(True)
                    btn_map = {}
                    for i, (val, display) in enumerate(p.choices):
                        rb = QRadioButton(display)
                        group.addButton(rb)
                        grid.addWidget(rb, i // 3, i % 3)
                        btn_map[val] = rb
                        if val == p.default:
                            rb.setChecked(True)
                    group.buttonToggled.connect(
                        lambda btn, checked: self.changed.emit() if checked else None
                    )
                    layout.addLayout(grid)
                    self._button_groups[p.key] = (group, btn_map)
                    continue

                row = QHBoxLayout()
                name_lbl = QLabel(p.label)
                name_lbl.setFixedWidth(110)
                row.addWidget(name_lbl)

                if p.kind == "choice":
                    w = QComboBox()
                    for val, display in p.choices:
                        w.addItem(display, val)
                    w.setCurrentIndex(int(p.default))
                    w.currentIndexChanged.connect(lambda _: self.changed.emit())
                    row.addWidget(w)
                    self._combos[p.key] = w
                else:
                    slider = QSlider(Qt.Orientation.Horizontal)
                    slider.setRange(int(p.min), int(p.max))
                    slider.setValue(int(p.default))
                    val_lbl = QLabel(str(int(p.default)))
                    val_lbl.setFixedWidth(50)

                    def _make_cb(lbl):
                        def cb(v):
                            lbl.setText(str(v))
                            self.changed.emit()
                        return cb

                    slider.valueChanged.connect(_make_cb(val_lbl))
                    row.addWidget(slider)
                    row.addWidget(val_lbl)
                    self._sliders[p.key] = slider
                    self._val_labels[p.key] = val_lbl

                layout.addLayout(row)

        layout.addStretch()

    def get_params(self) -> dict:
        params = {}
        for p in self._method.params:
            if p.kind == "choice":
                params[p.key] = self._combos[p.key].currentData()
            elif p.kind == "buttons":
                _group, btn_map = self._button_groups[p.key]
                for val, rb in btn_map.items():
                    if rb.isChecked():
                        params[p.key] = val
                        break
            else:
                params[p.key] = self._sliders[p.key].value()
        return params

    def set_params(self, params: dict):
        """把指定參數值寫回控制項（不觸發 changed 訊號）。"""
        for p in self._method.params:
            if p.key not in params:
                continue
            if p.kind == "choice":
                w = self._combos[p.key]
                w.blockSignals(True)
                for i in range(w.count()):
                    if w.itemData(i) == params[p.key]:
                        w.setCurrentIndex(i)
                        break
                w.blockSignals(False)
            elif p.kind == "buttons":
                group, btn_map = self._button_groups[p.key]
                group.blockSignals(True)
                for val, rb in btn_map.items():
                    rb.setChecked(val == params[p.key])
                group.blockSignals(False)
            else:
                s = self._sliders[p.key]
                s.blockSignals(True)
                s.setValue(int(params[p.key]))
                s.blockSignals(False)
                self._val_labels[p.key].setText(str(int(params[p.key])))

    def make_step(self) -> dict:
        return {"method": self._method.id, "params": self.get_params()}

    def apply(self, image: np.ndarray) -> np.ndarray:
        return apply_step(image, self.make_step())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("影像處理工具")
        self.resize(1200, 760)

        self._original: np.ndarray | None = None
        self._image_path: str = ""
        self._steps: list[dict] = []
        self._panels: dict[str, MethodParamPanel] = {}
        self._current_mid: str | None = None
        self._pending = False      # 是否有「尚未套用」的編輯（控制是否疊加即時預覽）
        self._loading = False      # 程式化載入步驟時抑制訊號副作用

        self._build_ui()
        self._build_category_radios()

    # ── UI 建構 ──────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 頂部工具列
        toolbar = QHBoxLayout()
        self._load_btn = QPushButton("載入圖片")
        self._save_btn = QPushButton("儲存結果")
        self._status_lbl = QLabel("尚未載入圖片")
        self._save_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._load_image)
        self._save_btn.clicked.connect(self._save_result)
        toolbar.addWidget(self._load_btn)
        toolbar.addWidget(self._save_btn)
        toolbar.addWidget(self._status_lbl)
        toolbar.addStretch()
        root.addLayout(toolbar)

        body = QHBoxLayout()
        root.addLayout(body)

        # 左側：預覽
        self._preview = QLabel("請先載入圖片")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(560, 480)
        self._preview.setStyleSheet("background:#1e1e1e; color:#888; border:1px solid #444;")
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        body.addWidget(self._preview, stretch=3)

        # 右側
        right = QVBoxLayout()
        body.addLayout(right, stretch=1)

        # 模式切換
        mode_group = QGroupBox("處理模式")
        mode_layout = QHBoxLayout(mode_group)
        self._mode_step = QRadioButton("逐步即時")
        self._mode_batch = QRadioButton("批次清單")
        self._mode_step.setChecked(True)
        mode_layout.addWidget(self._mode_step)
        mode_layout.addWidget(self._mode_batch)
        self._mode_step.toggled.connect(self._on_mode_changed)
        right.addWidget(mode_group)

        # 分類（按鍵）
        cat_group = QGroupBox("分類")
        self._cat_layout = QVBoxLayout(cat_group)
        self._cat_group_btns = QButtonGroup(self)
        right.addWidget(cat_group)

        # 方法（按鍵，依分類動態產生）
        method_group = QGroupBox("方法")
        self._method_layout = QVBoxLayout(method_group)
        self._method_group_btns = QButtonGroup(self)
        right.addWidget(method_group)

        # 參數
        param_group = QGroupBox("參數調整")
        self._param_layout = QVBoxLayout(param_group)
        right.addWidget(param_group)

        # 操作按鈕
        self._apply_btn = QPushButton("套用此步驟")
        self._run_all_btn = QPushButton("執行全部")
        self._undo_btn = QPushButton("復原上一步")
        self._reset_btn = QPushButton("重設為原圖")
        for b in (self._apply_btn, self._run_all_btn, self._undo_btn, self._reset_btn):
            b.setEnabled(False)
            right.addWidget(b)
        self._apply_btn.clicked.connect(self._apply_step)
        self._run_all_btn.clicked.connect(self._run_all)
        self._undo_btn.clicked.connect(self._undo)
        self._reset_btn.clicked.connect(self._reset)
        self._on_mode_changed()

        # 步驟歷史
        history_group = QGroupBox("步驟歷史")
        hist_layout = QVBoxLayout(history_group)
        self._history_list = QListWidget()
        self._history_list.setMaximumHeight(120)
        hist_layout.addWidget(self._history_list)
        right.addWidget(history_group)

        right.addStretch()

    def _build_category_radios(self):
        for cat in get_categories():
            rb = QRadioButton(cat)
            self._cat_group_btns.addButton(rb)
            self._cat_layout.addWidget(rb)
            rb.toggled.connect(lambda checked, c=cat: self._on_category_changed(c) if checked else None)
        self._cat_group_btns.buttons()[0].setChecked(True)

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _on_category_changed(self, cat):
        # 重建方法按鍵
        self._method_group_btns = QButtonGroup(self)
        self._clear_layout(self._method_layout)
        for m in get_methods_by_category(cat):
            rb = QRadioButton(m.name)
            rb.setProperty("method_id", m.id)
            self._method_group_btns.addButton(rb)
            self._method_layout.addWidget(rb)
            rb.toggled.connect(lambda checked, mid=m.id: self._on_method_chosen(mid) if checked else None)
        if not self._loading:
            self._method_group_btns.buttons()[0].setChecked(True)

    def _on_method_chosen(self, mid):
        self._show_panel(mid)
        if not self._loading:
            self._pending = True
            self._apply_btn.setEnabled(self._original is not None)
            self._refresh_preview()

    def _show_panel(self, mid):
        self._current_mid = mid
        while self._param_layout.count():
            item = self._param_layout.takeAt(0)
            if item.widget():
                item.widget().hide()
        if mid not in self._panels:
            self._panels[mid] = MethodParamPanel(get_method(mid))
            self._panels[mid].changed.connect(self._on_param_changed)
        self._param_layout.addWidget(self._panels[mid])
        self._panels[mid].show()

    def _current_panel(self) -> MethodParamPanel | None:
        return self._panels.get(self._current_mid) if self._current_mid else None

    # ── 事件 ─────────────────────────────────────────────────
    def _on_param_changed(self):
        if self._loading:
            return
        self._pending = True
        self._refresh_preview()

    def _on_mode_changed(self):
        batch = self._mode_batch.isChecked()
        self._apply_btn.setText("加入清單" if batch else "套用此步驟")
        self._run_all_btn.setVisible(batch)
        self._refresh_preview()

    # ── 預覽 ─────────────────────────────────────────────────
    def _show_image(self, image: np.ndarray):
        pix = ndarray_to_qpixmap(image)
        scaled = pix.scaled(
            self._preview.width(), self._preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self._preview.setPixmap(scaled)

    def _refresh_preview(self):
        if self._original is None:
            return
        working = run_pipeline(self._original, self._steps)
        # 有未套用的編輯時，無論哪種模式都疊加當前面板做即時預覽
        if self._pending and self._current_panel() is not None:
            try:
                self._show_image(apply_step(working, self._current_panel().make_step()))
                return
            except Exception as e:
                self._status_lbl.setText(f"預覽錯誤：{e}")
        self._show_image(working)

    # ── 載入步驟到編輯區（復原時用） ─────────────────────────
    def _load_step_into_editor(self, step: dict):
        self._loading = True
        m = get_method(step["method"])
        for b in self._cat_group_btns.buttons():
            if b.text() == m.category:
                if b.isChecked():
                    self._on_category_changed(m.category)  # 同分類不會自動觸發，手動重建
                else:
                    b.setChecked(True)
                break
        for b in self._method_group_btns.buttons():
            if b.property("method_id") == m.id:
                b.setChecked(True)
                break
        self._show_panel(m.id)
        self._panels[m.id].set_params(step["params"])
        self._loading = False

    # ── 主要操作 ─────────────────────────────────────────────
    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇圖片", "", "圖片檔案 (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self._status_lbl.setText("無法讀取圖片")
            return
        self._original = img
        self._image_path = path
        self._steps = []
        self._history_list.clear()
        self._pending = True
        orig_folder = os.path.join(os.path.dirname(path), "Original Image")
        os.makedirs(orig_folder, exist_ok=True)
        cv2.imwrite(os.path.join(orig_folder, os.path.basename(path)), img)
        self._status_lbl.setText(f"已載入：{os.path.basename(path)}")
        self._save_btn.setEnabled(True)
        self._apply_btn.setEnabled(self._current_mid is not None)
        self._reset_btn.setEnabled(True)
        self._refresh_preview()

    def _apply_step(self):
        panel = self._current_panel()
        if panel is None or self._original is None:
            return
        step = panel.make_step()
        working = run_pipeline(self._original, self._steps)
        try:
            apply_step(working, step)
        except Exception as e:
            self._status_lbl.setText(f"無法套用此步驟：{e}")
            return
        self._steps.append(step)
        self._history_list.addItem(f"{len(self._steps)}. {make_step_label(step)}")
        self._undo_btn.setEnabled(True)
        self._pending = False
        if self._mode_batch.isChecked():
            self._run_all_btn.setEnabled(True)
            self._status_lbl.setText(f"已加入清單：{make_step_label(step)}")
        else:
            self._refresh_preview()
            self._status_lbl.setText(f"已套用：{make_step_label(step)}")

    def _run_all(self):
        if self._original is None or not self._steps:
            return
        self._pending = False
        self._show_image(run_pipeline(self._original, self._steps))
        self._status_lbl.setText(f"已執行全部 {len(self._steps)} 個步驟")

    def _undo(self):
        if not self._steps:
            return
        removed = self._steps.pop()
        self._history_list.takeItem(self._history_list.count() - 1)
        # 把被復原的步驟還原回編輯區（方法分類 / 方法 / 參數），並只顯示已套用的結果
        self._load_step_into_editor(removed)
        self._pending = False
        if not self._steps:
            self._undo_btn.setEnabled(False)
            self._run_all_btn.setEnabled(False)
        self._refresh_preview()
        self._status_lbl.setText(f"已復原：{make_step_label(removed)}（已還原此步驟的設定）")

    def _reset(self):
        self._steps = []
        self._history_list.clear()
        self._undo_btn.setEnabled(False)
        self._run_all_btn.setEnabled(False)
        self._pending = False
        self._refresh_preview()
        self._status_lbl.setText("已重設為原圖")

    def _save_result(self):
        if self._original is None:
            return
        working = run_pipeline(self._original, self._steps)
        folder = os.path.join(os.path.dirname(self._image_path), "Pipeline_Result")
        os.makedirs(folder, exist_ok=True)
        base = os.path.splitext(os.path.basename(self._image_path))[0]
        out = os.path.join(folder, f"{base}_result.jpg")
        cv2.imwrite(out, working)
        self._status_lbl.setText(f"已儲存：Pipeline_Result/{os.path.basename(out)}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
