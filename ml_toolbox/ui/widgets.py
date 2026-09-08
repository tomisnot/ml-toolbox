# -*- coding: utf-8 -*-
"""matplotlib 画布组件 + 通用小部件。"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
# matplotlib 独立字体管理：不加 YaHei 则中文轴标签渲染成方框（pitfalls #1 变体）
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton


class MplCanvas(FigureCanvas):
    """同步绘制安全的画布（截图 grab 前 draw() 已是同步语义）。"""

    def __init__(self, width=6, height=4, dpi=100, parent=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi,
                          constrained_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.mpl_ax = None

    def draw_result(self, plot_fn, result):
        """调用方法声明的 plot(ax, result)；异常显示文本而非崩溃。"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        try:
            plot_fn(ax, result)
        except Exception as e:
            ax.clear()
            ax.text(0.5, 0.5, f"绘图失败: {e}", ha="center", va="center",
                    color="firebrick", fontsize=9)
        self.draw_idle()


class Placeholder(QWidget):
    """语义化占位（模式 7：不让用户面对空白猜原因）。"""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setAlignment(Qt.AlignCenter)
        lab.setStyleSheet("color:#888; font-size:12px; padding:24px;")
        lay.addWidget(lab)


class PGScatter(QWidget):
    """pyqtgraph 交互式散点（模式 5：滚轮缩放/拖拽平移 GPU blit，大点量不卡）。

    从 artifacts 取 embedding + labels，按标签离散着色。
    方法侧只需声明 PageSpec(kind="pg")，零 Qt 依赖不变（P1）。
    """

    def __init__(self, result, parent=None):
        super().__init__(parent)
        import pyqtgraph as pg
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        glw = pg.GraphicsLayoutWidget()
        glw.setBackground("w")
        lay.addWidget(glw)
        emb = np.asarray(result.artifacts.get("embedding"), float)
        if emb.ndim != 2 or emb.shape[1] < 2:
            lay.addWidget(QLabel("无 2D 嵌入数据"))
            return
        labels = result.artifacts.get("labels")
        p = glw.addPlot(title=f"{result.method_name} · 嵌入（滚轮缩放/左键平移）")
        p.showGrid(x=True, y=True, alpha=0.2)
        x, y = emb[:, 0], emb[:, 1]
        if labels is not None:
            labels = np.asarray(labels)
            uniq = np.unique(labels)
            palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                       "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            for k, v in enumerate(uniq):
                m = labels == v
                si = pg.ScatterPlotItem(x=x[m], y=y[m], size=6,
                                        brush=pg.mkBrush(palette[k % 10]),
                                        pen=None, name=str(v))
                p.addItem(si)
            p.addLegend(labelTextSize="8pt")
        else:
            p.addItem(pg.ScatterPlotItem(x=x, y=y, size=6,
                                         brush=pg.mkBrush("#1f77b4"), pen=None))
        btn = QPushButton("⟲ 还原视图")
        btn.setFixedWidth(110)
        btn.clicked.connect(lambda: p.getViewBox().autoRange())
        lay.addWidget(btn, 0, Qt.AlignLeft)
