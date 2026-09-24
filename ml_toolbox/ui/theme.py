# -*- coding: utf-8 -*-
"""全局视觉主题 + 通用小组件（UI 重设计的地基）。

设计语言：浅灰底 + 白色圆角卡片 + 蓝色主色（#2d6cdf，与运行按钮一致）。
层次映射：QGroupBox=一个配置段（卡片），标题即段名；
ModeSwitch=顶层状态切换（胶囊分段控件），对应兄弟框架的 Perspective 语义。

字号：所有 font-size 字面量已按 core.contracts.FONT_SCALE(=2) 放大；
matplotlib / pyqtgraph 的默认字号在 apply_to 里统一缩放。
QSS 挂在窗口根上（apply_to），对话框/子控件自动继承。
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton

from ..core.contracts import FONT_SCALE   # 单一来源（绘图层零 Qt 也用它）


def mpl_scale():
    """matplotlib 默认字号放大（显式 fontsize= 的调用点已按 FONT_SCALE 写死）。"""
    import matplotlib
    base = 10 * FONT_SCALE
    for key in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
                "ytick.labelsize", "legend.fontsize", "figure.titlesize"):
        matplotlib.rcParams[key] = base


def pg_scale():
    """pyqtgraph 全局字号缩放（轴刻度、标题、图例）。"""
    import pyqtgraph as pg
    # pyqtgraph 0.13 无全局 fontScale 配置项：默认字体取自 QApplication 字体
    # （入口已按 FONT_SCALE 放大）；需显式指定处（热图 tickFont、散点图例
    # labelTextSize）已按 2 倍写死。
    _ = pg


QSS = """
QWidget { color:#1f2329; font-size:26px; }
QMainWindow, QDialog { background:#eef1f5; }
QLabel { background:transparent; }

/* ---- 顶栏 ---- */
#header { background:#ffffff; border-bottom:1px solid #dfe3e8; }
#brand { font-size:30px; font-weight:bold; color:#1f2329; background:transparent; }
#brandSub { font-size:22px; color:#8a93a0; background:transparent; }

/* ---- 分段模式开关（胶囊） ---- */
#segbar { background:#e6e9ee; border-radius:24px; }
QPushButton#seg { background:transparent; border:none; padding:10px 32px;
                  border-radius:20px; color:#5f6b7a; font-weight:600;
                  font-size:26px; }
QPushButton#seg:hover { color:#2d6cdf; }
QPushButton#seg:checked { background:#2d6cdf; color:#ffffff; }

/* ---- 卡片分组 ---- */
QGroupBox { background:#ffffff; border:1px solid #e1e5ea; border-radius:8px;
            margin-top:22px; font-weight:bold; font-size:26px; }
QGroupBox::title { subcontrol-origin: margin; left:12px; padding:2px 10px;
                   color:#2d6cdf; background:#e8f0fe; border-radius:4px; }
QFrame#card { background:#ffffff; border:1px solid #e1e5ea; border-radius:8px; }
QFrame#card QTreeWidget, QFrame#card QTableWidget, QFrame#card QListWidget {
    border:none; }
QStackedWidget#centerStack { background:#ffffff; border:1px solid #e1e5ea;
                             border-radius:8px; }
QLabel#cardTitle { font-size:26px; font-weight:bold; color:#1f2329; }
QLabel#muted { color:#8a93a0; font-size:22px; }
QLabel#faint { color:#a0a6ae; font-size:20px; }
QLabel#accent { color:#2d6cdf; font-size:20px; }   /* 主色强调（画廊指标等） */
QLabel#capTitle { font-weight:bold; font-size:22px; }
QLabel#errText { color:#c0392b; font-size:22px; }  /* 校验/错误红字 */
QLabel#srcStatus { font-size:22px; }   /* 数据源状态行：字号归主题，
                                          颜色=连接语义（绿/红/灰）由调用点内联 */
/* 阶段导航分隔线头（—— 看数据 ——）：主题色/字号只在此定义 */
QLabel#railHead { font-size:24px; font-weight:bold; color:#8a93a0;
                  background:transparent; }
/* 检视页标题 + 页内小按钮（导出图/导出数据） */
QLabel#pageHead { font-size:30px; font-weight:bold; background:transparent; }
QPushButton#mini { font-size:22px; }
/* 占位提示（模式 7：不让用户面对空白） */
QLabel#placeholder { color:#8a93a0; font-size:24px; padding:24px; }

/* ---- 滚动容器：内容透明，白底由外层卡片提供 ---- */
QScrollArea { border:none; background:transparent; }
QScrollArea > QWidget > QWidget { background:transparent; }

/* ---- 控件 ---- */
QPushButton { background:#ffffff; border:1px solid #c9d1da; border-radius:5px;
              padding:8px 18px; }
QPushButton:hover { border-color:#2d6cdf; color:#2d6cdf; }
QPushButton:pressed { background:#e8f0fe; }
QPushButton:disabled { color:#a0a6ae; background:#f2f3f5; border-color:#e1e5ea; }
QPushButton#primary { background:#2d6cdf; color:#ffffff; border:none;
                      font-weight:bold; padding:14px 28px; border-radius:6px;
                      font-size:26px; }
QPushButton#primary:hover { background:#1e54b8; }
QPushButton#primary:disabled { background:#aab6c8; color:#eef1f5; }
QPushButton#ghost { background:transparent; border:none; color:#5f6b7a;
                    padding:6px 12px; }
QPushButton#ghost:hover { color:#2d6cdf; background:#e8f0fe; border-radius:4px; }
QPushButton#ghost:checked { color:#2d6cdf; background:#e8f0fe; border-radius:4px;
                            font-weight:bold; }
/* 阶段导航入口：ghost 基础上左对齐 */
QPushButton#railBtn { background:transparent; border:none; color:#5f6b7a;
                      padding:7px 10px; text-align:left; font-size:26px; }
QPushButton#railBtn:hover { color:#2d6cdf; background:#e8f0fe; border-radius:4px; }
QPushButton#railBtn:checked { color:#2d6cdf; background:#e8f0fe;
                              border-radius:4px; font-weight:bold; }

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background:#ffffff; border:1px solid #c9d1da; border-radius:5px;
    padding:6px 10px; selection-background-color:#2d6cdf; }
QLineEdit:focus, QComboBox:focus { border-color:#2d6cdf; }
QLineEdit:disabled, QComboBox:disabled { background:#f5f6f8; color:#a0a6ae; }
QComboBox::drop-down { border:none; width:32px; }
QComboBox QAbstractItemView { background:#ffffff; border:1px solid #dfe3e8;
    selection-background-color:#e8f0fe; selection-color:#1f2329; }

QCheckBox { spacing:8px; background:transparent; }
QCheckBox::indicator { width:24px; height:24px; border:1px solid #b6bfca;
    border-radius:4px; background:#ffffff; }
QCheckBox::indicator:checked { background:#2d6cdf; border-color:#2d6cdf;
    image:none; }

/* ---- 页签 ---- */
QTabWidget::pane { background:#ffffff; border:1px solid #e1e5ea;
                   border-radius:8px; top:-1px; }
QTabBar::tab { background:transparent; padding:10px 22px;
               border-bottom:2px solid transparent; color:#5f6b7a; }
QTabBar::tab:selected { color:#2d6cdf; border-bottom:2px solid #2d6cdf;
                        font-weight:600; }
QTabBar::tab:hover { color:#2d6cdf; }
QTabWidget#barePane::pane { border:none; background:transparent; }

/* ---- 表格 / 树 ---- */
QTreeWidget, QTableWidget, QListWidget { background:#ffffff;
    border:1px solid #e1e5ea; border-radius:8px; }
QTreeWidget::item, QTableWidget::item, QListWidget::item { min-height:44px; }
QTreeWidget::item:hover, QTableWidget::item:hover { background:#f0f5ff; }
/* 选中行：必须显式定义，否则默认调色板画深蓝底 + 全局深色字 = 文字不可读 */
QTreeWidget::item:selected, QTableWidget::item:selected,
QListWidget::item:selected { background:#e8f0fe; color:#2d6cdf; }
QHeaderView::section { background:#f4f6f9; border:none;
    border-right:1px solid #e6e9ee; border-bottom:1px solid #e6e9ee;
    padding:8px 12px; font-weight:600; color:#42505f; }

QStatusBar { background:#ffffff; border-top:1px solid #e1e5ea; color:#5f6b7a; }
QSplitter::handle { background:#e1e5ea; }
QSplitter::handle:horizontal { width:6px; }
QSplitter::handle:vertical { height:6px; }
QToolTip { background:#ffffff; border:1px solid #c9d1da; color:#1f2329;
           padding:6px 10px; }
QMenu { background:#ffffff; border:1px solid #dfe3e8; }
QMenu::item:selected { background:#e8f0fe; }
QPlainTextEdit { background:#ffffff; border:1px solid #e1e5ea; border-radius:6px; }
"""


def apply_to(widget: QWidget):
    """把主题挂到窗口根（MainWindow / 独立 OptWorkbench 都要调）。

    同时完成 matplotlib / pyqtgraph 的全局字号缩放。
    """
    widget.setStyleSheet(QSS)
    mpl_scale()
    pg_scale()


class ModeSwitch(QWidget):
    """胶囊分段开关：兄弟框架的顶层状态切换（ML 试验台 ⇄ 优化调参）。"""

    mode_changed = pyqtSignal(int)

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self.setObjectName("segbar")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        self._btns: list[QPushButton] = []
        for i, lb in enumerate(labels):
            b = QPushButton(lb)
            b.setObjectName("seg")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=i: self._on_click(k))
            lay.addWidget(b)
            self._btns.append(b)
        self._btns[0].setChecked(True)

    def _on_click(self, i: int):
        self.set_mode(i)
        self.mode_changed.emit(i)

    def set_mode(self, i: int):
        for k, b in enumerate(self._btns):
            b.setChecked(k == i)
