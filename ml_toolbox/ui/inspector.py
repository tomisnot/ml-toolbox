# -*- coding: utf-8 -*-
"""检视页基类（G2 核心）：UI 不预设图长什么样，只装配方法声明的 PageSpec。

自由度设计：
- 方法通过 inspect_pages() 返回 PageSpec 列表，每页 kind 可为
  mpl（plot 函数）/ table（DataFrame 表格）/ text（诊断文本）；
- 方法没声明的页由 runner.auto_pages 兜底；
- 数据缺失时显示语义化占位提示（模式 7），绝不空白；
- 每页自带导出按钮（模式 9：npz/png）。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTableWidget, QTableWidgetItem, QPlainTextEdit,
                             QFileDialog, QLabel)

from ..core.runner import RunRecord
from .widgets import MplCanvas, Placeholder


# ------------------------------------------------ 页类型注册表（G2 自由度）
# 新页类型 = UI 层注册一个工厂 builder(result, spec) -> QWidget | None。
# 方法层只声明 kind 字符串并把数据装进 artifacts，渲染归 UI 层。
PAGE_BUILDERS: dict = {}


def register_page_builder(kind: str, builder):
    PAGE_BUILDERS[kind] = builder


class InspectPage(QWidget):
    """单页检视容器：标题 + 内容 + 导出。kind 决定内容形态。"""

    def __init__(self, pagespec, parent=None):
        super().__init__(parent)
        self.spec = pagespec
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        head = QHBoxLayout()
        title = QLabel(pagespec.title)
        title.setStyleSheet("font-size:15px; font-weight:bold;")
        head.addWidget(title)
        head.addStretch(1)
        btn_png = QPushButton("导出图")
        btn_png.clicked.connect(self._export_png)
        btn_npz = QPushButton("导出数据")
        btn_npz.clicked.connect(self._export_npz)
        for b in (btn_png, btn_npz):
            b.setStyleSheet("font-size:11px;")
            head.addWidget(b)
        lay.addLayout(head)

        self._body = QHBoxLayout()
        lay.addLayout(self._body, 1)
        self.canvas: MplCanvas | None = None
        self._placeholder = None
        self._record: RunRecord | None = None

    # ------------------------------------------------ 装配
    def show_result(self, record: RunRecord):
        self._record = record
        self._clear_body()
        res = record.result
        if not res.ok:
            self._set_body(QLabel(f"运行失败：\n{(res.error or '')[:600]}"))
            return
        kind = self.spec.kind
        if kind == "pg":
            # pyqtgraph 交互式图（模式 5）：大数据量缩放平移不卡
            from .widgets import PGScatter
            if res.artifacts.get("embedding") is not None:
                self._set_body(PGScatter(res))
            else:
                self._set_body(Placeholder(
                    self.spec.hint or "该方法未产出 pg 页所需工件（embedding）"))
            return
        if kind == "mpl":
            if self.spec.plot is None:
                self._set_body(Placeholder(
                    self.spec.hint or "该方法未声明此图；在 inspect_pages() 里加一个 "
                    "PageSpec(kind='mpl', plot=fn) 即可接入。"))
                return
            self.canvas = MplCanvas(width=7, height=5)
            self.canvas.draw_result(self.spec.plot, res)
            self._set_body(self.canvas)
        elif kind == "table":
            df = self._table_source(res)
            if isinstance(df, pd.DataFrame) and not df.empty:
                self._set_body(self._make_table(df))
            else:
                self._set_body(Placeholder(
                    self.spec.hint or f"artifacts['{self.spec.key}'] 无表格数据"))
        elif kind == "text":
            box = QPlainTextEdit()
            box.setReadOnly(True)
            box.setPlainText(str(res.diag or {}) if res.diag
                             else (self.spec.hint or "开启诊断开关后显示中间产物"))
            self._set_body(box)
        elif kind in PAGE_BUILDERS:
            # 注册页类型（如 nn_weights / nn_replay）：builder 返回 None = 数据不足
            w = PAGE_BUILDERS[kind](res, self.spec)
            self._set_body(w if w is not None else Placeholder(
                self.spec.hint or f"kind={kind!r} 缺少所需工件"))
        else:
            self._set_body(Placeholder(f"未知页类型 kind={kind!r}"))

    def _table_source(self, res) -> pd.DataFrame | None:
        """table 页的数据源：共享页映射到 metrics/params，其余取 artifacts。"""
        if self.spec.key == "metrics":
            m = {k: v for k, v in res.metrics.items()
                 if isinstance(v, (int, float))}
            return pd.DataFrame({"值": [m[k] for k in m]},
                                index=list(m)) if m else None
        if self.spec.key == "params":
            p = res.params
            return pd.DataFrame({"值": [p[k] for k in p]},
                                index=list(p)) if p else None
        df = res.artifacts.get(self.spec.key)
        return df if isinstance(df, pd.DataFrame) else None

    def _make_table(self, df: pd.DataFrame) -> QTableWidget:
        tw = QTableWidget(len(df), df.shape[1])
        tw.setHorizontalHeaderLabels([str(c) for c in df.columns])
        tw.setVerticalHeaderLabels([str(i)[:24] for i in df.index])
        for i in range(len(df)):
            for j in range(df.shape[1]):
                v = df.iat[i, j]
                txt = f"{v:.4g}" if isinstance(v, (int, float, np.floating)) \
                    else str(v)
                tw.setItem(i, j, QTableWidgetItem(txt))
        tw.resizeColumnsToContents()
        return tw

    def _clear_body(self):
        while self._body.count():
            it = self._body.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        self.canvas = None

    def _set_body(self, widget):
        self._body.addWidget(widget, 1)

    # ------------------------------------------------ 导出（模式 9）
    def _export_png(self):
        if self._record is None:
            return
        base = f"{self._record.method}_{self.spec.key}"
        if self.canvas is not None:
            p, _ = QFileDialog.getSaveFileName(self, "导出图片",
                                               f"{base}.png", "PNG (*.png)")
            if p:
                self.canvas.fig.savefig(p, dpi=150, bbox_inches="tight")
        elif self.spec.kind == "pg":
            # pyqtgraph 画布导出
            p, _ = QFileDialog.getSaveFileName(self, "导出图片",
                                               f"{base}.png", "PNG (*.png)")
            if p:
                import pyqtgraph as pg
                from pyqtgraph.exporters import ImageExporter
                for w in self._body_children():
                    if isinstance(w, pg.GraphicsLayoutWidget):
                        ImageExporter(w.ci).export(p)
                        return

    def _body_children(self):
        import pyqtgraph as pg
        out = []
        for i in range(self._body.count()):
            w = self._body.itemAt(i).widget()
            if w:
                out.append(w)
                out.extend(w.findChildren(pg.GraphicsLayoutWidget))
        return out

    def _export_npz(self):
        if self._record is None:
            return
        p, _ = QFileDialog.getSaveFileName(
            self, "导出数据", f"{self._record.method}_{self.spec.key}.npz",
            "NumPy (*.npz)")
        if not p:
            return
        art = {k: v for k, v in self._record.result.artifacts.items()
               if isinstance(v, np.ndarray)}
        np.savez(p, **art)


class MethodInspector(QWidget):
    """方法检视器：按方法的 PageSpec 列表动态装配分页（基类高自由度所在）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        from PyQt5.QtWidgets import QTabWidget
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        lay.addWidget(self.tabs)
        self._pages: list[InspectPage] = []

    def show_record(self, record: RunRecord, pages: list):
        # 页数变化才重建（避免每次切换都重建画布）
        if [p.spec.key for p in self._pages] != [s.key for s in pages]:
            self._destroy_pages()
            self._pages = [InspectPage(s) for s in pages]
            for pgw, s in zip(self._pages, pages):
                self.tabs.addTab(pgw, s.title)
        for pgw in self._pages:
            pgw.show_result(record)
        self.tabs.setCurrentIndex(0)

    def _destroy_pages(self):
        while self.tabs.count():
            w = self.tabs.widget(0)
            self.tabs.removeTab(0)
            if w:
                w.setParent(None)
                w.deleteLater()
        self._pages = []

    def clear(self):
        self._destroy_pages()
