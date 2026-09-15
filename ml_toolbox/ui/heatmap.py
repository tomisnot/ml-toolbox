# -*- coding: utf-8 -*-
"""自适应矩阵热图（pyqtgraph，移植自 UI design 模式 5 的 ML 版）。

与上游 AdaptiveHeatmap 的差异：ML 场景是"特征相关矩阵"这类方阵，
不需要扇区/流形分组，但保留其核心价值——
- 滚轮缩放 + 左键平移（GPU blit，上百维不卡）；
- 分级刻度：视野大抽稀短名，视野小逐维全名（自动 ≤max_ticks 防重叠）；
- 长名折行（pyqtgraph 隐形坑：文本超轴宽被静默丢弃，pitfalls #13）。
"""
from __future__ import annotations

import numpy as np

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
import pyqtgraph as pg


def _wrap(s, width=8):
    s = str(s)
    if len(s) <= width:
        return s
    return "\n".join(s[j:j + width] for j in range(0, len(s), width))


class _TickAxis(pg.AxisItem):
    """自定义轴：显式绘制刻度文字（深色），扩容防 contains 静默丢弃。"""

    def boundingRect(self):
        br = super().boundingRect()
        return br.adjusted(-30, -30, 30, 30)

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        p.setRenderHint(p.Antialiasing, False)
        p.setRenderHint(p.TextAntialiasing, True)
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        for pen, p1, p2 in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)
        p.setPen(pg.mkPen("#2c3137"))
        if self.style.get("tickFont") is not None:
            p.setFont(self.style["tickFont"])
        for rect, flags, text in textSpecs:
            p.save()
            c = rect.center()
            p.translate(c.x(), c.y())
            p.drawText(QRectF(-rect.width() / 2, -rect.height() / 2,
                              rect.width(), rect.height()),
                       Qt.AlignCenter, text)
            p.restore()


class AdaptiveMatrixHeatmap(QWidget):
    """方阵热图：set_data(A, names)。A[a,b] -> a=纵轴向上, b=横轴。"""

    def __init__(self, parent=None, max_ticks=12, detail_span=30):
        super().__init__(parent)
        self._dim = 0
        self._names = None
        self._names_c = None
        self._rows = self._cols = 0
        self._max_ticks = max_ticks
        self._detail_span = detail_span
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        self.glw.ci.setContentsMargins(130, 10, 40, 130)
        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.vb = self.plot.getViewBox()
        self.vb.setMouseMode(self.vb.PanMode)
        self.vb.setMenuEnabled(False)
        self.img = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.img)
        self.cmap = pg.colormap.get("viridis")
        self.img.setLookupTable(self.cmap.getLookupTable(0.0, 1.0, 256))
        self.ax_b = _TickAxis(orientation="bottom")
        self.ax_l = _TickAxis(orientation="left")
        self.plot.setAxisItems({"bottom": self.ax_b, "left": self.ax_l})
        self.ax_b.setHeight(110)
        self.ax_l.setWidth(110)
        from PyQt5.QtGui import QFont
        f = QFont("Microsoft YaHei", 14)
        for ax in (self.ax_b, self.ax_l):
            ax.setStyle(tickFont=f)
            ax.setStyle(hideOverlappingLabels=False, textFillLimits=[(0, 1.0)])
        self.vb.sigXRangeChanged.connect(lambda *_: self._update_ticks())
        self.vb.sigYRangeChanged.connect(lambda *_: self._update_ticks())
        lay.addWidget(self.glw, 1)
        row = QHBoxLayout()
        b = QPushButton("⟲ 还原视图")
        b.clicked.connect(self.reset_view)
        row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

    def set_data(self, A, names, names_c=None):
        """A: 矩阵；names: 行标签（纵轴）；names_c: 列标签（横轴，缺省同行）。
        支持矩形矩阵（如权重 W[out, in]）。"""
        A = np.asarray(A, float)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        self._rows, self._cols = A.shape
        self._dim = max(self._rows, self._cols)
        self._names = list(names)
        self._names_c = list(names_c) if names_c is not None else list(names)
        # 实测本版本 pyqtgraph：row-major 图像 row0 在底部（y 向上），
        # 直接 setImage(A) 即 A[a,b] -> a=纵轴、b=横轴（勿再 .T，会抵消成转置）
        self.img.setImage(A, autoLevels=False)
        lo, hi = float(np.nanmin(A)), float(np.nanmax(A))
        self.img.setLevels([lo, hi if hi > lo else lo + 1])
        self.img.setRect(QRectF(0, 0, self._cols, self._rows))
        # 方阵锁比例（相关矩阵语义）；长条矩阵放开，否则格子被压成细线
        self.plot.setAspectLocked(self._rows == self._cols)
        self.reset_view()
        self._update_ticks()

    def reset_view(self):
        if getattr(self, "_rows", 0):
            self.vb.setRange(xRange=(0, self._cols), yRange=(0, self._rows),
                             padding=0)

    def _ticks(self, lo, hi, names=None, n=None):
        names = names if names is not None else self._names
        n = n if n is not None else getattr(self, "_rows", 0)
        if not n or not names:
            return []
        lo_i = max(int(np.floor(lo)), 0)
        hi_i = min(int(np.ceil(hi)), n - 1)
        if hi_i < lo_i:
            return []
        span = hi_i - lo_i + 1
        step = max(1, int(np.ceil(span / self._max_ticks)))
        # 视野小 -> 全名折行；视野大 -> 抽稀短名
        detail = span <= self._detail_span
        out = []
        for t in range(lo_i, hi_i + 1, step):
            nm = names[t] if t < len(names) else str(t)
            out.append((t + 0.5, _wrap(nm) if detail else str(nm)[:8]))
        return [out]

    def _update_ticks(self):
        (x0, x1), (y0, y1) = self.vb.viewRange()
        self.ax_b.setTicks(self._ticks(x0, x1, self._names_c, self._cols))
        self.ax_l.setTicks(self._ticks(y0, y1, self._names, self._rows))
