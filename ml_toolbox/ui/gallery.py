# -*- coding: utf-8 -*-
"""对比画廊：遍历结果的核心图并排缩略（G1 遍历的视觉回报）。

一次遍历后，每个方法用它自己的第一张 mpl 页渲染成缩略图，网格排布。
横向对比"谁拟合得好"不再只盯指标数字——一眼看出形状差异。
点缩略图放大（复用方法检视器）。
"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QScrollArea, QGridLayout,
                            QLabel, QFrame, QSizePolicy)


class _Thumb(QFrame):
    clicked = pyqtSignal(int)

    def __init__(self, idx, record, pages, width=3.0, height=2.4, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.setCursor(Qt.PointingHandCursor)
        self.setFrameShape(QFrame.StyledPanel)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(2)

        cap = QLabel(f"{record.result.method_name}")
        cap.setStyleSheet("font-weight:bold; font-size:11px;")
        pm = record.result.primary_metric
        val = record.result.metrics.get(pm)
        sub = QLabel(f"{pm}={val:.3f}" if isinstance(val, (int, float)) else pm)
        sub.setStyleSheet("color:#2d6cdf; font-size:10px;")
        lay.addWidget(cap)
        lay.addWidget(sub)

        fig = Figure(figsize=(width, height), dpi=90, constrained_layout=True)
        ax = fig.add_subplot(111)
        plot_fn = self._pick_plot(pages)
        if plot_fn is None:
            ax.text(0.5, 0.5, "无图", ha="center", va="center", color="#999")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            try:
                plot_fn(ax, record.result)
            except Exception as e:
                ax.clear(); ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.5, 0.5, f"绘错\n{str(e)[:30]}", ha="center",
                        va="center", color="firebrick", fontsize=7)
        cv = FigureCanvas(fig)
        cv.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(cv, 1)

    def _pick_plot(self, pages):
        for ps in pages:
            if ps.kind == "mpl" and ps.plot is not None:
                return ps.plot
        return None

    def mousePressEvent(self, ev):
        self.clicked.emit(self.idx)
        super().mousePressEvent(ev)


class CompareGallery(QWidget):
    """网格缩略；records 顺序即点击回调的 idx。"""

    thumb_clicked = pyqtSignal(int)

    def __init__(self, parent=None, columns=4):
        super().__init__(parent)
        self._columns = columns
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._host = QWidget()
        self._grid = QGridLayout(self._host)
        self._grid.setSpacing(8)
        self._grid.setAlignment(Qt.AlignTop)
        self._scroll.setWidget(self._host)
        outer.addWidget(self._scroll)

    def show_records(self, records, pages_for):
        # 清空
        while self._grid.count():
            it = self._grid.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        ok = [(i, r) for i, r in enumerate(records) if r.result.ok]

        # 按主指标排序展示（与对比视图一致；点击仍回原始 idx）
        def _key(ir):
            i, r = ir
            v = r.result.metrics.get(r.result.primary_metric)
            return v if isinstance(v, (int, float)) and not np.isnan(v) \
                else float("inf")
        asc = ok[0][1].result.primary_metric in ("rmse", "mae", "mape") \
            if ok else True
        ok.sort(key=_key, reverse=not asc)
        for n, (idx, r) in enumerate(ok):
            pages = pages_for(r)
            th = _Thumb(idx, r, pages)
            th.clicked.connect(self.thumb_clicked.emit)
            self._grid.addWidget(th, n // self._columns, n % self._columns)
        # 占位撑开最后一行
        self._grid.setRowStretch(self._grid.rowCount(), 1)
