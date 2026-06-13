import sys  # 匯入 sys 模組，用於取得命令列參數與結束程式
import os  # 匯入 os 模組，用於檔案路徑處理與建立資料夾
import cv2  # 匯入 OpenCV，用於讀取、寫入與轉換影像
import numpy as np  # 匯入 NumPy 並取別名 np，影像以 ndarray 形式處理
from PyQt6.QtWidgets import (  # 從 PyQt6 匯入各種 GUI 元件類別
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,  # 應用程式、主視窗、基底元件與各種版面配置
    QLabel, QPushButton, QListWidget, QSlider, QComboBox,  # 標籤、按鈕、清單、滑桿、下拉選單
    QFileDialog, QGroupBox, QSizePolicy, QRadioButton, QButtonGroup,  # 檔案對話框、群組框、大小策略、單選鈕、單選鈕群組
)
from PyQt6.QtCore import Qt, pyqtSignal  # 匯入 Qt 常數列舉與自訂訊號工具 pyqtSignal
from PyQt6.QtGui import QPixmap, QImage  # 匯入點陣圖 QPixmap 與影像 QImage，用於顯示影像

from processing.registry import get_categories, get_methods_by_category, get_method, METHODS  # 從方法登錄表匯入查詢分類／方法的函式
from processing.pipeline import apply_step, run_pipeline, make_step_label  # 從流程模組匯入套用單步、執行整條流程、產生步驟標籤的函式


def ndarray_to_qpixmap(image: np.ndarray) -> QPixmap:  # 定義將 NumPy 影像轉成 QPixmap 的函式，方便在 GUI 顯示
    if image is None:  # 若傳入影像為 None（尚未載入）
        return QPixmap()  # 回傳空的 QPixmap，避免後續錯誤
    img = np.ascontiguousarray(image)  # 將影像轉成記憶體連續陣列，QImage 才能正確讀取資料
    if len(img.shape) == 2:  # 若影像只有兩個維度，代表是灰階圖
        h, w = img.shape  # 取出高度 h 與寬度 w
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)  # 以灰階格式建立 QImage，每列位元組數為 w
    else:  # 否則為彩色影像（三維）
        h, w, ch = img.shape  # 取出高度、寬度與通道數 ch
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # OpenCV 為 BGR，轉成 RGB 並保持記憶體連續
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)  # 以 RGB888 格式建立 QImage，每列位元組數為 w*ch
    return QPixmap.fromImage(qimg.copy())  # 複製 QImage（避免引用已釋放的 numpy 緩衝）後轉為 QPixmap 回傳


class PreviewLabel(QLabel):  # 定義預覽用標籤類別，繼承自 QLabel
    """預覽用 QLabel，支援滑鼠按下／拖曳回傳點位（佔影像百分比）。"""  # 類別說明：支援滑鼠取點並以百分比回報
    pct_picked = pyqtSignal(float, float)  # 自訂訊號：被點擊／拖曳時送出 (x%, y%) 兩個浮點數

    def __init__(self, *args, **kwargs):  # 建構子，接受任意參數轉交父類別
        super().__init__(*args, **kwargs)  # 呼叫 QLabel 的建構子完成初始化
        self._dragging = False  # 初始化拖曳狀態旗標為 False（尚未按下）

    def _emit_pct(self, pos):  # 內部方法：依滑鼠座標計算並發出百分比訊號
        pm = self.pixmap()  # 取得目前顯示的點陣圖
        if pm is None or pm.isNull():  # 若沒有圖片或圖片為空
            return  # 直接返回，不做計算
        pw, ph = pm.width(), pm.height()  # 取得實際顯示圖片的寬與高
        off_x = (self.width() - pw) / 2.0  # 計算圖片在標籤中水平置中的左邊空白偏移
        off_y = (self.height() - ph) / 2.0  # 計算圖片在標籤中垂直置中的上方空白偏移
        x = pos.x() - off_x  # 將滑鼠 x 座標換算成相對於圖片左上角的座標
        y = pos.y() - off_y  # 將滑鼠 y 座標換算成相對於圖片左上角的座標
        if x < 0 or y < 0 or x > pw or y > ph or pw == 0 or ph == 0:  # 若點落在圖片範圍外或圖片尺寸為零
            return  # 忽略此次點擊
        self.pct_picked.emit(x / pw * 100.0, y / ph * 100.0)  # 將座標轉成百分比並發出訊號

    def mousePressEvent(self, event):  # 覆寫滑鼠按下事件
        self._dragging = True  # 標記進入拖曳狀態
        self._emit_pct(event.position())  # 立即依按下位置發出百分比訊號

    def mouseMoveEvent(self, event):  # 覆寫滑鼠移動事件
        if self._dragging:  # 只有在按住拖曳時才處理
            self._emit_pct(event.position())  # 依目前位置即時發出百分比訊號

    def mouseReleaseEvent(self, event):  # 覆寫滑鼠放開事件
        self._dragging = False  # 結束拖曳狀態


class MethodParamPanel(QWidget):  # 定義方法參數面板類別，繼承自 QWidget
    changed = pyqtSignal()  # 自訂訊號：任一參數變更時發出，用於即時更新預覽

    def __init__(self, method):  # 建構子，傳入要呈現參數的方法物件
        super().__init__()  # 呼叫父類別建構子
        self._method = method  # 保存方法物件供後續取得參數定義
        self._sliders = {}  # 以參數 key 對應滑桿元件的字典
        self._combos = {}  # 以參數 key 對應下拉選單元件的字典
        self._button_groups = {}  # 以參數 key 對應單選鈕群組的字典
        self._val_labels = {}  # 以參數 key 對應滑桿數值顯示標籤的字典
        layout = QVBoxLayout(self)  # 建立垂直版面配置作為本面板主版面
        layout.setContentsMargins(4, 4, 4, 4)  # 設定版面四周邊距為 4 像素

        if not method.params:  # 若此方法沒有任何可調參數
            layout.addWidget(QLabel("此方法無需調整參數"))  # 顯示提示文字告知使用者
        else:  # 否則逐一為每個參數建立對應的控制元件
            for p in method.params:  # 遍歷方法的每一個參數定義
                if p.kind == "buttons":  # 若參數類型為按鈕（單選鈕群組）
                    lbl = QLabel(p.label)  # 建立顯示參數名稱的標籤
                    if p.help:  # 若該參數有說明文字
                        lbl.setToolTip(p.help)  # 設定滑鼠停留提示
                    layout.addWidget(lbl)  # 將名稱標籤加入版面
                    grid = QGridLayout()  # 建立格狀版面以排列多個單選鈕
                    group = QButtonGroup(self)  # 建立單選鈕群組以管理互斥選擇
                    group.setExclusive(True)  # 設定群組為互斥（一次只能選一個）
                    btn_map = {}  # 建立 值->單選鈕 的對應字典
                    for i, (val, display) in enumerate(p.choices):  # 遍歷每個選項（含索引、值與顯示文字）
                        rb = QRadioButton(display)  # 以顯示文字建立單選鈕
                        if p.help:  # 若有說明文字
                            rb.setToolTip(p.help)  # 為單選鈕設定提示
                        group.addButton(rb)  # 將單選鈕加入群組
                        grid.addWidget(rb, i // 3, i % 3)  # 每列放三個，依索引計算列與欄位置
                        btn_map[val] = rb  # 記錄此選項值對應的單選鈕
                        if val == p.default:  # 若此選項為預設值
                            rb.setChecked(True)  # 將其設為選取狀態
                    group.buttonToggled.connect(  # 連接群組的切換事件
                        lambda btn, checked: self.changed.emit() if checked else None  # 當有鈕被選取時發出 changed 訊號
                    )
                    layout.addLayout(grid)  # 將單選鈕格狀版面加入主版面
                    self._button_groups[p.key] = (group, btn_map)  # 保存群組與對應字典供取值使用
                    self._add_help_label(layout, p.help)  # 在下方加入說明文字標籤
                    continue  # 此參數已處理完，跳到下一個參數

                row = QHBoxLayout()  # 建立水平版面，放名稱與控制元件於同一列
                name_lbl = QLabel(p.label)  # 建立參數名稱標籤
                name_lbl.setFixedWidth(110)  # 固定名稱標籤寬度，使排版對齊
                if p.help:  # 若有說明文字
                    name_lbl.setToolTip(p.help)  # 設定提示
                row.addWidget(name_lbl)  # 將名稱標籤加入此列

                if p.kind == "choice":  # 若參數類型為下拉選擇
                    w = QComboBox()  # 建立下拉選單
                    for val, display in p.choices:  # 遍歷每個選項
                        w.addItem(display, val)  # 加入項目，顯示文字為 display、資料為 val
                    w.setCurrentIndex(int(p.default))  # 設定預設選取項（以索引）
                    if p.help:  # 若有說明文字
                        w.setToolTip(p.help)  # 設定提示
                    w.currentIndexChanged.connect(lambda _: self.changed.emit())  # 選項改變時發出 changed 訊號
                    row.addWidget(w)  # 將下拉選單加入此列
                    self._combos[p.key] = w  # 記錄此下拉選單供取值
                else:  # 否則為數值型參數，使用滑桿
                    slider = QSlider(Qt.Orientation.Horizontal)  # 建立水平滑桿
                    slider.setRange(int(p.min), int(p.max))  # 設定滑桿可選範圍的最小與最大值
                    slider.setValue(int(p.default))  # 設定滑桿初始值為預設值
                    if p.help:  # 若有說明文字
                        slider.setToolTip(p.help)  # 設定提示
                    val_lbl = QLabel(str(int(p.default)))  # 建立顯示目前數值的標籤
                    val_lbl.setFixedWidth(50)  # 固定數值標籤寬度避免跳動

                    def _make_cb(lbl):  # 工廠函式：產生綁定特定標籤的回呼，避免閉包共用變數問題
                        def cb(v):  # 實際的回呼，接收滑桿目前數值 v
                            lbl.setText(str(v))  # 更新數值標籤文字
                            self.changed.emit()  # 發出 changed 訊號以更新預覽
                        return cb  # 回傳回呼函式

                    slider.valueChanged.connect(_make_cb(val_lbl))  # 將滑桿數值變化連接到對應回呼
                    row.addWidget(slider)  # 將滑桿加入此列
                    row.addWidget(val_lbl)  # 將數值標籤加入此列
                    self._sliders[p.key] = slider  # 記錄此滑桿供取值
                    self._val_labels[p.key] = val_lbl  # 記錄此數值標籤供同步更新

                layout.addLayout(row)  # 將此列（名稱＋控制元件）加入主版面
                self._add_help_label(layout, p.help)  # 在下方加入說明文字標籤

        layout.addStretch()  # 在底部加入彈性空間，使元件靠上對齊

    @staticmethod  # 靜態方法裝飾器，不需 self
    def _add_help_label(layout, help_text: str):  # 在指定版面加入灰色說明文字標籤
        if not help_text:  # 若沒有說明文字
            return  # 不加任何東西
        hint = QLabel(help_text)  # 建立說明文字標籤
        hint.setWordWrap(True)  # 啟用自動換行避免文字溢出
        hint.setStyleSheet("color:#8aa0b0; font-size:11px; margin:0 0 6px 2px;")  # 設定灰藍色小字樣式與邊距
        layout.addWidget(hint)  # 將說明標籤加入版面

    def get_params(self) -> dict:  # 從各控制元件蒐集目前參數值並回傳字典
        params = {}  # 初始化參數字典
        for p in self._method.params:  # 遍歷每個參數定義
            if p.kind == "choice":  # 若為下拉選擇
                params[p.key] = self._combos[p.key].currentData()  # 取得目前選項所帶的資料值
            elif p.kind == "buttons":  # 若為單選鈕群組
                _group, btn_map = self._button_groups[p.key]  # 取出群組與值對應字典
                for val, rb in btn_map.items():  # 遍歷每個選項
                    if rb.isChecked():  # 找出被選取的單選鈕
                        params[p.key] = val  # 記錄其對應的值
                        break  # 找到即跳出迴圈
            else:  # 否則為滑桿數值
                params[p.key] = self._sliders[p.key].value()  # 取得滑桿目前數值
        return params  # 回傳蒐集到的參數字典

    def set_params(self, params: dict):  # 依傳入字典將各控制元件設定為指定值
        for p in self._method.params:  # 遍歷每個參數定義
            if p.key not in params:  # 若傳入字典沒有此參數
                continue  # 跳過不處理
            if p.kind == "choice":  # 若為下拉選擇
                w = self._combos[p.key]  # 取得對應下拉選單
                w.blockSignals(True)  # 暫時封鎖訊號，避免設定時觸發 changed
                for i in range(w.count()):  # 遍歷所有選項
                    if w.itemData(i) == params[p.key]:  # 找出資料值符合目標的選項
                        w.setCurrentIndex(i)  # 設為目前選項
                        break  # 找到即跳出
                w.blockSignals(False)  # 解除訊號封鎖
            elif p.kind == "buttons":  # 若為單選鈕群組
                group, btn_map = self._button_groups[p.key]  # 取出群組與對應字典
                group.blockSignals(True)  # 封鎖群組訊號
                for val, rb in btn_map.items():  # 遍歷每個選項
                    rb.setChecked(val == params[p.key])  # 將符合目標值的單選鈕設為選取
                group.blockSignals(False)  # 解除訊號封鎖
            else:  # 否則為滑桿
                s = self._sliders[p.key]  # 取得對應滑桿
                s.blockSignals(True)  # 封鎖滑桿訊號
                s.setValue(int(params[p.key]))  # 設定滑桿數值
                s.blockSignals(False)  # 解除訊號封鎖
                self._val_labels[p.key].setText(str(int(params[p.key])))  # 同步更新數值顯示標籤

    def make_step(self) -> dict:  # 將目前面板狀態包裝成一個處理步驟字典
        return {"method": self._method.id, "params": self.get_params()}  # 回傳含方法 id 與參數的步驟

    def apply(self, image: np.ndarray) -> np.ndarray:  # 將此步驟直接套用到輸入影像
        return apply_step(image, self.make_step())  # 呼叫流程模組套用步驟並回傳結果影像


class MainWindow(QMainWindow):  # 定義主視窗類別，繼承自 QMainWindow
    def __init__(self):  # 主視窗建構子
        super().__init__()  # 呼叫父類別建構子
        self.setWindowTitle("影像處理工具")  # 設定視窗標題
        self.resize(1200, 800)  # 設定視窗初始大小

        self._original: np.ndarray | None = None  # 原始影像，尚未載入時為 None
        self._image_path: str = ""  # 目前載入影像的檔案路徑
        self._steps: list[dict] = []  # 已套用的處理步驟清單
        self._panels: dict[str, MethodParamPanel] = {}  # 已建立的方法參數面板快取（以方法 id 為鍵）
        self._current_mid: str | None = None  # 目前選取的方法 id
        self._pending = False  # 是否有尚未套用的待處理變更（用於即時預覽）
        self._loading = False  # 是否正在程式化載入設定（避免觸發多餘事件）
        self._inspect_idx: int | None = None  # 目前檢視的歷史步驟索引，None 表示未檢視

        self._build_ui()  # 建構整體使用者介面
        self._build_category_radios()  # 建構分類單選鈕

    # ── UI 建構 ──  # 以下為介面建構相關方法
    def _build_ui(self):  # 建構主介面版面與元件
        central = QWidget()  # 建立中央容器元件
        self.setCentralWidget(central)  # 設定為主視窗的中央元件
        root = QVBoxLayout(central)  # 建立最外層垂直版面

        toolbar = QHBoxLayout()  # 建立頂部工具列水平版面
        self._load_btn = QPushButton("載入圖片")  # 建立載入圖片按鈕
        self._save_btn = QPushButton("儲存結果")  # 建立儲存結果按鈕
        self._status_lbl = QLabel("尚未載入圖片")  # 建立狀態提示標籤
        self._save_btn.setEnabled(False)  # 初始停用儲存按鈕（尚無結果可存）
        self._load_btn.clicked.connect(self._load_image)  # 載入按鈕點擊時呼叫載入影像
        self._save_btn.clicked.connect(self._save_result)  # 儲存按鈕點擊時呼叫儲存結果
        toolbar.addWidget(self._load_btn)  # 將載入按鈕加入工具列
        toolbar.addWidget(self._save_btn)  # 將儲存按鈕加入工具列
        toolbar.addWidget(self._status_lbl)  # 將狀態標籤加入工具列
        toolbar.addStretch()  # 加入彈性空間使前面元件靠左
        root.addLayout(toolbar)  # 將工具列加入最外層版面

        body = QHBoxLayout()  # 建立主體水平版面（左預覽、右控制）
        root.addLayout(body)  # 將主體版面加入最外層版面

        self._preview = PreviewLabel("請先載入圖片")  # 建立可取點的預覽標籤並顯示提示文字
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)  # 設定內容置中對齊
        self._preview.setMinimumSize(560, 480)  # 設定預覽區最小尺寸
        self._preview.setStyleSheet("background:#1e1e1e; color:#888; border:1px solid #444;")  # 設定深色背景與邊框樣式
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # 設定為可隨視窗擴展
        self._preview.pct_picked.connect(self._on_preview_pct)  # 預覽圖被點擊/拖曳時，呼叫處理函式更新模板位置
        body.addWidget(self._preview, stretch=3)  # 將預覽加入主體並給予較大伸縮比例

        right = QVBoxLayout()  # 建立右側控制面板的垂直版面
        body.addLayout(right, stretch=1)  # 將右側版面加入主體並給予較小伸縮比例

        # 模式切換  # 建立處理模式選擇區
        mode_group = QGroupBox("處理模式")  # 建立模式群組框
        mode_layout = QHBoxLayout(mode_group)  # 建立群組框內的水平版面
        self._mode_step = QRadioButton("逐步即時")  # 建立逐步即時模式單選鈕
        self._mode_batch = QRadioButton("批次清單")  # 建立批次清單模式單選鈕
        self._mode_step.setChecked(True)  # 預設選取逐步即時模式
        mode_layout.addWidget(self._mode_step)  # 將逐步單選鈕加入版面
        mode_layout.addWidget(self._mode_batch)  # 將批次單選鈕加入版面
        self._mode_step.toggled.connect(self._on_mode_changed)  # 模式切換時呼叫處理函式更新介面
        right.addWidget(mode_group)  # 將模式群組加入右側版面

        # 分類  # 建立方法分類選擇區
        cat_group = QGroupBox("分類")  # 建立分類群組框
        self._cat_layout = QVBoxLayout(cat_group)  # 建立群組框內垂直版面，供動態加入分類鈕
        self._cat_group_btns = QButtonGroup(self)  # 建立分類單選鈕群組（互斥）
        right.addWidget(cat_group)  # 將分類群組加入右側版面

        # 方法  # 建立方法選擇區
        method_group = QGroupBox("方法")  # 建立方法群組框
        self._method_layout = QVBoxLayout(method_group)  # 建立群組框內垂直版面，供動態加入方法鈕
        self._method_group_btns = QButtonGroup(self)  # 建立方法單選鈕群組（互斥）
        right.addWidget(method_group)  # 將方法群組加入右側版面

        # 參數（含方法說明）  # 建立參數調整區
        param_group = QGroupBox("參數調整")  # 建立參數群組框
        param_outer = QVBoxLayout(param_group)  # 建立群組框內的外層垂直版面
        self._method_desc_lbl = QLabel()  # 建立顯示方法說明的標籤
        self._method_desc_lbl.setWordWrap(True)  # 啟用自動換行
        self._method_desc_lbl.setStyleSheet(  # 設定說明標籤的樣式
            "color:#cfe8ff; background:#1b3a4b; border-left:3px solid #4dabf7;"  # 淺藍字、深藍底與左側強調邊
            "padding:6px 8px; border-radius:4px;"  # 內距與圓角
        )
        self._method_desc_lbl.setVisible(False)  # 初始隱藏（尚未選方法時不顯示）
        param_outer.addWidget(self._method_desc_lbl)  # 將說明標籤加入外層版面
        self._param_layout = QVBoxLayout()  # 建立放置實際參數面板的版面
        param_outer.addLayout(self._param_layout)  # 將參數面板版面加入外層版面
        right.addWidget(param_group)  # 將參數群組加入右側版面

        # 操作按鈕  # 建立各項操作按鈕
        self._apply_btn = QPushButton("套用此步驟")  # 建立套用步驟按鈕
        self._run_all_btn = QPushButton("執行全部")  # 建立執行全部步驟按鈕
        self._undo_btn = QPushButton("復原上一步")  # 建立復原按鈕
        self._reset_btn = QPushButton("重設為原圖")  # 建立重設按鈕
        for b in (self._apply_btn, self._run_all_btn, self._undo_btn, self._reset_btn):  # 遍歷四個操作按鈕
            b.setEnabled(False)  # 初始全部停用（尚未載入影像）
            right.addWidget(b)  # 依序加入右側版面
        self._apply_btn.clicked.connect(self._apply_step)  # 套用按鈕點擊時呼叫套用步驟
        self._run_all_btn.clicked.connect(self._run_all)  # 執行全部按鈕點擊時執行整條流程
        self._undo_btn.clicked.connect(self._undo)  # 復原按鈕點擊時復原上一步
        self._reset_btn.clicked.connect(self._reset)  # 重設按鈕點擊時重設為原圖
        self._on_mode_changed()  # 依目前模式初始化按鈕文字與顯示狀態

        # 步驟歷史  # 建立步驟歷史清單區
        history_group = QGroupBox("步驟歷史（點選可預覽該步結果）")  # 建立歷史群組框
        hist_layout = QVBoxLayout(history_group)  # 建立群組框內垂直版面
        self._history_list = QListWidget()  # 建立顯示步驟歷史的清單元件
        self._history_list.setMaximumHeight(130)  # 限制清單最大高度
        self._history_list.itemClicked.connect(self._on_history_clicked)  # 點選清單項目時呼叫檢視該步結果
        hist_layout.addWidget(self._history_list)  # 將清單加入版面
        right.addWidget(history_group)  # 將歷史群組加入右側版面

        right.addStretch()  # 在右側底部加入彈性空間使元件靠上

    def _build_category_radios(self):  # 動態建立分類單選鈕
        for cat in get_categories():  # 遍歷所有方法分類名稱
            rb = QRadioButton(cat)  # 為每個分類建立單選鈕
            self._cat_group_btns.addButton(rb)  # 加入分類群組以互斥管理
            self._cat_layout.addWidget(rb)  # 加入分類版面顯示
            rb.toggled.connect(lambda checked, c=cat: self._on_category_changed(c) if checked else None)  # 被選取時依分類更新方法清單
        self._cat_group_btns.buttons()[0].setChecked(True)  # 預設選取第一個分類

    def _clear_layout(self, layout):  # 清空指定版面中的所有元件
        while layout.count():  # 當版面仍有元件時持續處理
            item = layout.takeAt(0)  # 取出並移除第一個項目
            if item.widget():  # 若該項目為實體元件
                item.widget().deleteLater()  # 安排稍後刪除該元件以釋放資源

    def _on_category_changed(self, cat):  # 當選擇的分類改變時更新方法清單
        self._method_group_btns = QButtonGroup(self)  # 重建方法單選鈕群組（清掉舊群組關係）
        self._clear_layout(self._method_layout)  # 清空舊的方法單選鈕
        for m in get_methods_by_category(cat):  # 遍歷該分類下的所有方法
            rb = QRadioButton(m.name)  # 以方法名稱建立單選鈕
            rb.setProperty("method_id", m.id)  # 將方法 id 存入單選鈕屬性供日後查找
            self._method_group_btns.addButton(rb)  # 加入方法群組
            self._method_layout.addWidget(rb)  # 加入方法版面顯示
            rb.toggled.connect(lambda checked, mid=m.id: self._on_method_chosen(mid) if checked else None)  # 被選取時切換到該方法
        if not self._loading:  # 若非程式化載入狀態
            self._method_group_btns.buttons()[0].setChecked(True)  # 預設選取第一個方法

    def _on_method_chosen(self, mid):  # 當選擇的方法改變時的處理
        self._show_panel(mid)  # 顯示對應方法的參數面板
        if not self._loading:  # 若非程式化載入
            self._pending = True  # 標記有待處理變更
            self._apply_btn.setEnabled(self._original is not None)  # 有載入影像時才啟用套用按鈕
            self._refresh_preview()  # 重新整理預覽

    def _show_panel(self, mid):  # 顯示指定方法的參數面板與說明
        self._current_mid = mid  # 記錄目前選取的方法 id
        method = get_method(mid)  # 依 id 取得方法物件
        desc = getattr(method, "description", "")  # 取得方法說明文字（無則為空字串）
        self._method_desc_lbl.setText(desc)  # 設定說明標籤文字
        self._method_desc_lbl.setVisible(bool(desc))  # 有說明才顯示說明標籤
        while self._param_layout.count():  # 當參數版面仍有元件時
            item = self._param_layout.takeAt(0)  # 取出第一個項目
            if item.widget():  # 若為實體元件
                item.widget().hide()  # 隱藏舊面板（保留快取不刪除）
        if mid not in self._panels:  # 若此方法的面板尚未建立
            self._panels[mid] = MethodParamPanel(method)  # 建立新的參數面板並快取
            self._panels[mid].changed.connect(self._on_param_changed)  # 參數變更時連接到更新預覽
        self._param_layout.addWidget(self._panels[mid])  # 將對應面板加入版面
        self._panels[mid].show()  # 顯示該面板

    def _current_panel(self) -> MethodParamPanel | None:  # 取得目前選取方法的參數面板
        return self._panels.get(self._current_mid) if self._current_mid else None  # 若有選取方法則回傳其面板，否則 None

    # ── 事件 ──  # 以下為各種事件處理方法
    def _on_preview_pct(self, x_pct, y_pct):  # 預覽圖被取點時的處理（接收百分比座標）
        """在預覽圖上拖曳：模板比對時即時移動紅框位置。"""  # 說明：僅在模板比對方法下生效
        if self._current_mid != "template_match":  # 若目前方法不是模板比對
            return  # 忽略此次取點
        panel = self._current_panel()  # 取得目前參數面板
        if panel is None or self._original is None:  # 若無面板或無影像
            return  # 不處理
        panel.set_params({"x_pct": int(round(x_pct)), "y_pct": int(round(y_pct))})  # 將取點百分比寫回 x/y 參數
        self._pending = True  # 標記有待處理變更
        self._refresh_preview()  # 重新整理預覽以顯示新位置

    def _on_param_changed(self):  # 參數面板任一控制元件變更時的處理
        if self._loading:  # 若正在程式化載入
            return  # 不觸發更新避免重複
        self._pending = True  # 標記有待處理變更
        self._refresh_preview()  # 重新整理預覽

    def _on_mode_changed(self):  # 處理模式切換時的處理
        batch = self._mode_batch.isChecked()  # 判斷是否為批次清單模式
        self._apply_btn.setText("加入清單" if batch else "套用此步驟")  # 依模式調整套用按鈕文字
        self._run_all_btn.setVisible(batch)  # 僅批次模式顯示「執行全部」按鈕
        self._refresh_preview()  # 重新整理預覽

    def _on_history_clicked(self, item):  # 點選步驟歷史項目時的處理
        row = self._history_list.currentRow()  # 取得目前選取的列索引
        if row < 0 or row >= len(self._steps):  # 若索引超出有效範圍
            return  # 不處理
        if row == self._inspect_idx:  # 若再次點擊已檢視的步驟（取消檢視）
            self._inspect_idx = None  # 清除檢視索引
            self._history_list.clearSelection()  # 清除清單選取
            self._refresh_preview()  # 回到最新預覽
            self._status_lbl.setText("已取消步驟檢視")  # 更新狀態文字
            return  # 結束處理
        self._inspect_idx = row  # 記錄目前檢視的步驟索引
        img = run_pipeline(self._original, self._steps[:row + 1])  # 執行到該步為止的流程取得結果影像
        self._show_image(img)  # 顯示該步結果
        self._status_lbl.setText(f"[步驟 {row + 1}] {make_step_label(self._steps[row])}")  # 更新狀態顯示步驟資訊

    # ── 預覽 ──  # 以下為預覽顯示相關方法
    def _show_image(self, image: np.ndarray):  # 將影像縮放後顯示於預覽標籤
        pix = ndarray_to_qpixmap(image)  # 將 ndarray 轉成 QPixmap
        scaled = pix.scaled(  # 依預覽區大小縮放圖片
            self._preview.width(), self._preview.height(),  # 目標寬高為預覽區尺寸
            Qt.AspectRatioMode.KeepAspectRatio,  # 保持長寬比
            Qt.TransformationMode.SmoothTransformation  # 使用平滑縮放
        )
        self._preview.setPixmap(scaled)  # 將縮放後的圖片設定到預覽標籤

    def _refresh_preview(self):  # 重新計算並更新預覽影像
        if self._original is None:  # 若尚未載入影像
            return  # 不處理
        if self._inspect_idx is not None:  # 若正在檢視某個歷史步驟
            return  # 維持檢視畫面不更新
        working = run_pipeline(self._original, self._steps)  # 執行已套用步驟得到目前工作影像
        if self._pending and self._current_panel() is not None:  # 若有待處理變更且有選取方法
            try:  # 嘗試套用目前面板的待處理步驟以即時預覽
                self._show_image(apply_step(working, self._current_panel().make_step()))  # 顯示套用後結果
                return  # 成功則結束
            except Exception as e:  # 若套用發生例外
                self._status_lbl.setText(f"預覽錯誤：{e}")  # 在狀態列顯示錯誤訊息
        self._show_image(working)  # 顯示未含待處理變更的工作影像

    def _clear_inspect(self):  # 清除歷史步驟檢視狀態
        self._inspect_idx = None  # 重置檢視索引
        self._history_list.clearSelection()  # 清除清單選取

    # ── 載入步驟到編輯區 ──  # 將既有步驟還原到分類/方法/參數控制項
    def _load_step_into_editor(self, step: dict):  # 把一個步驟載入到編輯區（用於復原）
        self._loading = True  # 標記進入程式化載入狀態避免觸發事件
        m = get_method(step["method"])  # 依步驟中的方法 id 取得方法物件
        for b in self._cat_group_btns.buttons():  # 遍歷所有分類單選鈕
            if b.text() == m.category:  # 找到該方法所屬分類的單選鈕
                if b.isChecked():  # 若該分類已是選取狀態
                    self._on_category_changed(m.category)  # 手動重建方法清單（toggled 不會觸發）
                else:  # 否則
                    b.setChecked(True)  # 選取該分類（會自動觸發更新方法清單）
                break  # 找到即跳出
        for b in self._method_group_btns.buttons():  # 遍歷所有方法單選鈕
            if b.property("method_id") == m.id:  # 找到對應方法 id 的單選鈕
                b.setChecked(True)  # 選取該方法
                break  # 找到即跳出
        self._show_panel(m.id)  # 顯示該方法的參數面板
        self._panels[m.id].set_params(step["params"])  # 將步驟參數還原到面板控制元件
        self._loading = False  # 結束程式化載入狀態

    # ── 主要操作 ──  # 以下為載入、套用、執行、復原、重設、儲存等主要操作
    def _load_image(self):  # 載入圖片
        path, _ = QFileDialog.getOpenFileName(  # 開啟檔案對話框讓使用者選擇圖片
            self, "選擇圖片", "", "圖片檔案 (*.png *.jpg *.jpeg *.bmp)"  # 標題與檔案類型過濾
        )
        if not path:  # 若使用者取消未選擇
            return  # 不處理
        img = cv2.imread(path)  # 以 OpenCV 讀取影像
        if img is None:  # 若讀取失敗
            self._status_lbl.setText("無法讀取圖片")  # 顯示錯誤訊息
            return  # 結束
        self._original = img  # 保存原始影像
        self._image_path = path  # 記錄影像路徑
        self._steps = []  # 清空步驟清單
        self._history_list.clear()  # 清空歷史清單顯示
        self._pending = True  # 標記有待處理變更（顯示目前方法預覽）
        self._clear_inspect()  # 清除檢視狀態
        orig_folder = os.path.join(os.path.dirname(path), "Original Image")  # 在來源資料夾下組出「Original Image」路徑
        os.makedirs(orig_folder, exist_ok=True)  # 建立該資料夾（已存在則略過）
        cv2.imwrite(os.path.join(orig_folder, os.path.basename(path)), img)  # 將原圖另存一份到該資料夾備份
        self._status_lbl.setText(f"已載入：{os.path.basename(path)}")  # 更新狀態顯示已載入檔名
        self._save_btn.setEnabled(True)  # 啟用儲存按鈕
        self._apply_btn.setEnabled(self._current_mid is not None)  # 有選方法時啟用套用按鈕
        self._reset_btn.setEnabled(True)  # 啟用重設按鈕
        self._refresh_preview()  # 重新整理預覽顯示載入的影像

    def _apply_step(self):  # 套用目前步驟（或加入批次清單）
        panel = self._current_panel()  # 取得目前參數面板
        if panel is None or self._original is None:  # 若無面板或無影像
            return  # 不處理
        step = panel.make_step()  # 從面板產生步驟字典
        working = run_pipeline(self._original, self._steps)  # 執行既有步驟得到目前工作影像
        try:  # 嘗試套用新步驟以驗證可行性
            apply_step(working, step)  # 套用步驟（此處僅測試是否會出錯）
        except Exception as e:  # 若套用失敗
            self._status_lbl.setText(f"無法套用此步驟：{e}")  # 顯示錯誤訊息
            return  # 結束不加入
        self._steps.append(step)  # 將步驟加入清單
        self._history_list.addItem(f"{len(self._steps)}. {make_step_label(step)}")  # 在歷史清單新增此步驟標籤
        self._undo_btn.setEnabled(True)  # 啟用復原按鈕
        self._pending = False  # 清除待處理旗標（已正式套用）
        self._clear_inspect()  # 清除檢視狀態
        if self._mode_batch.isChecked():  # 若為批次清單模式
            self._run_all_btn.setEnabled(True)  # 啟用「執行全部」按鈕
            self._status_lbl.setText(f"已加入清單：{make_step_label(step)}")  # 更新狀態為已加入清單
        else:  # 否則為逐步即時模式
            self._refresh_preview()  # 重新整理預覽
            self._status_lbl.setText(f"已套用：{make_step_label(step)}")  # 更新狀態為已套用

    def _run_all(self):  # 執行清單中的全部步驟
        if self._original is None or not self._steps:  # 若無影像或無步驟
            return  # 不處理
        self._pending = False  # 清除待處理旗標
        self._clear_inspect()  # 清除檢視狀態
        self._show_image(run_pipeline(self._original, self._steps))  # 執行整條流程並顯示結果
        self._status_lbl.setText(f"已執行全部 {len(self._steps)} 個步驟")  # 更新狀態顯示步驟數

    def _undo(self):  # 復原上一個步驟
        if not self._steps:  # 若沒有任何步驟
            return  # 不處理
        removed = self._steps.pop()  # 移除並取得最後一個步驟
        self._history_list.takeItem(self._history_list.count() - 1)  # 從歷史清單移除對應項目
        self._load_step_into_editor(removed)  # 將被移除步驟的設定還原到編輯區
        self._pending = False  # 清除待處理旗標
        self._clear_inspect()  # 清除檢視狀態
        if not self._steps:  # 若已無剩餘步驟
            self._undo_btn.setEnabled(False)  # 停用復原按鈕
            self._run_all_btn.setEnabled(False)  # 停用執行全部按鈕
        self._refresh_preview()  # 重新整理預覽
        self._status_lbl.setText(f"已復原：{make_step_label(removed)}（已還原此步驟的設定）")  # 更新狀態顯示已復原

    def _reset(self):  # 重設為原圖（清空所有步驟）
        self._steps = []  # 清空步驟清單
        self._history_list.clear()  # 清空歷史清單顯示
        self._undo_btn.setEnabled(False)  # 停用復原按鈕
        self._run_all_btn.setEnabled(False)  # 停用執行全部按鈕
        self._pending = False  # 清除待處理旗標
        self._clear_inspect()  # 清除檢視狀態
        self._refresh_preview()  # 重新整理預覽顯示原圖
        self._status_lbl.setText("已重設為原圖")  # 更新狀態顯示已重設

    def _save_result(self):  # 儲存處理結果
        if self._original is None:  # 若無影像
            return  # 不處理
        working = run_pipeline(self._original, self._steps)  # 執行整條流程得到最終影像
        folder = os.path.join(os.path.dirname(self._image_path), "Pipeline_Result")  # 在來源資料夾下組出結果資料夾路徑
        os.makedirs(folder, exist_ok=True)  # 建立該資料夾（已存在則略過）
        base = os.path.splitext(os.path.basename(self._image_path))[0]  # 取得原檔名（不含副檔名）
        out = os.path.join(folder, f"{base}_result.jpg")  # 組出輸出檔案完整路徑
        cv2.imwrite(out, working)  # 將結果影像寫入檔案
        self._status_lbl.setText(f"已儲存：Pipeline_Result/{os.path.basename(out)}")  # 更新狀態顯示已儲存路徑

    def resizeEvent(self, event):  # 覆寫視窗縮放事件
        super().resizeEvent(event)  # 呼叫父類別預設處理
        self._refresh_preview()  # 視窗大小改變後重新整理預覽以重新縮放圖片


def main():  # 程式進入點函式
    app = QApplication(sys.argv)  # 建立 Qt 應用程式物件並傳入命令列參數
    app.setStyle("Fusion")  # 設定使用 Fusion 樣式風格
    win = MainWindow()  # 建立主視窗實例
    win.show()  # 顯示主視窗
    sys.exit(app.exec())  # 進入事件迴圈，結束時以其回傳碼結束程式


if __name__ == "__main__":  # 若此檔案被直接執行（而非被匯入）
    main()  # 呼叫主函式啟動程式
