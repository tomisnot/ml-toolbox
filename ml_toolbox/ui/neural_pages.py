# -*- coding: utf-8 -*-
"""神经网络自定义检视页（经 inspector.PAGE_BUILDERS 注册）。

两个新 kind：
  nn_weights —— 权重热图页：层下拉 + 视图切换（初始/最终/更新幅度）
                + 复用 AdaptiveMatrixHeatmap（滚轮缩放）
  nn_replay  —— 表示演化回放页（方案 C 核心体验）：
                epoch 滑块 + ▶播放 + 嵌入散点（PGScatter 风格）
                + 同步 loss 游标 + 类纯度读数
                —— 回答"中间发生了什么"：看类簇如何逐 epoch 展开

builder 约定：build(result, spec) -> QWidget | None；None = 工件不足，
由 InspectPage 显示 spec.hint 语义化占位。
"""
from __future__ import annotations

import numpy as np

import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                            QComboBox, QPushButton, QSlider)

from .inspector import register_page_builder
from .heatmap import AdaptiveMatrixHeatmap
from .widgets import MplCanvas


# ================================================================ 权重热图页
class WeightsPage(QWidget):
    def __init__(self, result, parent=None):
        super().__init__(parent)
        self._result = result
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        top = QHBoxLayout()
        top.addWidget(QLabel("层"))
        self._layer = QComboBox()
        self._layer.addItems(list(result.artifacts["nn_weights"]["last"].keys()))
        top.addWidget(self._layer, 1)
        top.addWidget(QLabel("视图"))
        self._view = QComboBox()
        self._view.addItems(["最终权重", "初始权重", "更新幅度 ΔW"])
        top.addWidget(self._view)
        self._info = QLabel("")
        self._info.setStyleSheet("color:#666; font-size:11px;")
        top.addWidget(self._info)
        lay.addLayout(top)
        self._heat = AdaptiveMatrixHeatmap()
        lay.addWidget(self._heat, 1)
        self._layer.currentIndexChanged.connect(self._refresh)
        self._view.currentIndexChanged.connect(self._refresh)
        self._refresh()

    def _refresh(self):
        name = self._layer.currentText()
        view = self._view.currentIndex()
        key = ["last", "first", "update"][view]
        W = self._result.artifacts["nn_weights"][key].get(name)
        if W is None:
            self._info.setText("该视图无快照（如首 epoch 权重）")
            return
        W = np.asarray(W, float)
        rows, cols = (W.shape[0], W.shape[1]) if W.ndim == 2 else (1, W.size)
        self._heat.set_data(W, [f"out{i}" for i in range(rows)],
                            [f"in{j}" for j in range(cols)])
        self._info.setText(f"{name}  {rows}×{cols}  "
                           f"|W|∈[{np.abs(W).min():.3f},{np.abs(W).max():.3f}]")


def build_weights_page(result, spec):
    w = result.artifacts.get("nn_weights")
    if not w or not w.get("last"):
        return None
    return WeightsPage(result)


# ================================================================ 表示演化回放页
class ReplayPage(QWidget):
    """epoch 滑块 + 播放 + 嵌入散点 + loss 游标 + 类纯度读数。"""

    def __init__(self, result, parent=None):
        super().__init__(parent)
        self._result = result
        art = result.artifacts
        self._lat = np.asarray(art["nn_latent"], float)          # (T, n, 2)
        self._epochs = np.asarray(art.get("nn_latent_epochs",
                                          np.arange(self._lat.shape[0])))
        self._labels = art.get("labels")
        hist = art.get("nn_history") or {}
        self._loss = np.asarray(hist.get("loss", []), float)
        self._val = np.asarray(hist.get("val_loss", []), float)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        # 控制条
        ctrl = QHBoxLayout()
        self._play = QPushButton("▶ 播放")
        self._play.setFixedWidth(80)
        self._play.clicked.connect(self._toggle_play)
        ctrl.addWidget(self._play)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, len(self._epochs) - 1)
        self._slider.valueChanged.connect(self._show_frame)
        ctrl.addWidget(self._slider, 1)
        self._readout = QLabel("")
        self._readout.setStyleSheet("font-weight:bold; min-width:150px;")
        ctrl.addWidget(self._readout)
        lay.addLayout(ctrl)

        # 主体：左散点（pyqtgraph）右 loss 游标（mpl）
        grid = QHBoxLayout()
        import pyqtgraph as pg
        self._gl = pg.GraphicsLayoutWidget()
        self._gl.setBackground("w")
        self._plot = self._gl.addPlot(title="隐层表示（最后一层输出 → PCA 2D）")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._items = []
        grid.addWidget(self._gl, 3)
        self._loss_canvas = MplCanvas(width=4, height=3)
        self._draw_loss_cursor(-1)
        grid.addWidget(self._loss_canvas, 2)
        lay.addLayout(grid, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(160)
        self._timer.timeout.connect(self._next_frame)

        self._show_frame(0)

    # ------------------------------------------------ 渲染
    def _show_frame(self, i):
        i = int(i)
        if i < 0 or i >= len(self._epochs):
            return
        ep = int(self._epochs[i])
        pts = self._lat[i]
        for it in self._items:
            self._plot.removeItem(it)
        self._items = []
        if self._labels is not None:
            lab = np.asarray(self._labels)
            uniq = np.unique(lab)
            palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                       "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            for k, v in enumerate(uniq):
                m = lab == v
                it = pg.ScatterPlotItem(x=pts[m, 0], y=pts[m, 1], size=6,
                                        brush=pg.mkBrush(palette[k % 10]),
                                        pen=None)
                self._plot.addItem(it)
                self._items.append(it)
        else:
            it = pg.ScatterPlotItem(x=pts[:, 0], y=pts[:, 1], size=6,
                                    brush=pg.mkBrush("#1f77b4"), pen=None)
            self._plot.addItem(it)
            self._items.append(it)
        loss_txt = (f" · loss={self._loss[ep]:.4f}"
                    if len(self._loss) > ep else "")
        self._plot.setTitle(f"epoch {ep} / {int(self._epochs[-1])}{loss_txt}"
                            " · 隐层表示（最后一层输出 → PCA 2D）")
        purity = self._purity(pts)
        extra = f" · 类纯度 {purity:.0%}" if purity == purity else ""
        self._readout.setText(f"epoch {ep}{extra}")
        self._draw_loss_cursor(ep)

    def _purity(self, pts) -> float:
        """k-近邻类纯度：当前帧下，每个点的最近邻同类比例（分群进度读数）。"""
        if self._labels is None:
            return float("nan")
        lab = np.asarray(self._labels)
        if lab.dtype == object or len(np.unique(lab)) < 2 or len(pts) < 10:
            return float("nan")
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        nn = d.argmin(axis=1)
        return float(np.mean(lab[nn] == lab))

    def _draw_loss_cursor(self, ep):
        self._loss_canvas.fig.clear()
        ax = self._loss_canvas.fig.add_subplot(111)
        if len(self._loss):
            ax.plot(self._loss, "b-", lw=1.2, label="train")
            if np.isfinite(self._val).any():
                ax.plot(self._val, "r--", lw=1.2, label="val")
            if 0 <= ep < len(self._loss):
                ax.axvline(ep, color="green", lw=1, alpha=0.7)
                ax.plot([ep], [self._loss[ep]], "go", ms=5)
        ax.set_xlabel("epoch"); ax.legend(fontsize=7)
        ax.set_title("loss 游标", fontsize=9)
        self._loss_canvas.draw_idle()

    # ------------------------------------------------ 播放
    def _toggle_play(self):
        if self._timer.isActive():
            self._timer.stop()
            self._play.setText("▶ 播放")
        else:
            if self._slider.value() >= self._slider.maximum():
                self._slider.setValue(0)
            self._timer.start()
            self._play.setText("⏸ 暂停")

    def _next_frame(self):
        nxt = self._slider.value() + 1
        if nxt > self._slider.maximum():
            self._timer.stop()
            self._play.setText("▶ 播放")
            return
        self._slider.setValue(nxt)


def build_replay_page(result, spec):
    lat = result.artifacts.get("nn_latent")
    if lat is None or np.asarray(lat).ndim != 3:
        return None
    return ReplayPage(result)


# ================================================================ 注册
register_page_builder("nn_weights", build_weights_page)
register_page_builder("nn_replay", build_replay_page)
