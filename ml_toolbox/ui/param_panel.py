# -*- coding: utf-8 -*-
"""参数面板（模式 3）：旋钮放在语义位置 + 就地重跑。

- 每个方法的 param_schema 自动生成控件；
- 空 = 默认（占位符写"默认"），覆写仅本次运行生效、不写回全局；
- "⟳ 应用并重跑" 发信号给主窗口触发新一轮运行。
"""
from __future__ import annotations

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QLineEdit,
                             QComboBox, QCheckBox, QPushButton, QLabel,
                             QScrollArea, QFrame)

from ..core.contracts import ParamSpec, RunConfig


class ParamPanel(QWidget):
    """一个方法的旋钮面板。set_method() 重建控件；collect() 读覆写。"""

    rerun_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self._title = QLabel("（未选择方法）")
        self._title.setStyleSheet("font-size:14px; font-weight:bold;")
        lay.addWidget(self._title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._host = QFrame()
        self._form = QFormLayout(self._host)
        self._form.setLabelAlignment(Qt.AlignRight)
        self._form.setSpacing(8)
        scroll.setWidget(self._host)
        lay.addWidget(scroll, 1)

        self._widgets: dict[str, tuple[ParamSpec, QWidget]] = {}

        self._btn = QPushButton("⟳ 应用并重跑")
        self._btn.setStyleSheet(
            "QPushButton{background:#2d6cdf;color:white;font-weight:bold;"
            "padding:8px;border-radius:4px;}"
            "QPushButton:hover{background:#1e54b8;}")
        self._btn.clicked.connect(lambda: self.rerun_requested.emit())
        lay.addWidget(self._btn)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size:11px; color:#c0392b;")
        lay.addWidget(self._status)

    # ------------------------------------------------ 构建
    def set_method(self, method, cfg: RunConfig):
        while self._form.rowCount():
            self._form.removeRow(0)
        self._widgets.clear()
        self._title.setText(f"{method.display_name} · 参数")
        for p in method.param_schema:
            w = self._make_widget(p, cfg.overrides.get(p.key))
            self._form.addRow(self._label(p), w)
            self._widgets[p.key] = (p, w)
        self._status.setText("")

    def _label(self, p: ParamSpec) -> QLabel:
        lab = QLabel(p.label or p.key)
        lab.setToolTip(p.hint or p.key)
        return lab

    def _make_widget(self, p: ParamSpec, current):
        if p.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(current) if current is not None else bool(p.default))
            w.stateChanged.connect(lambda *_: self._status.setText(""))
            return w
        if p.kind == "select":
            w = QComboBox()
            opts = [str(c) for c in (p.choices or [])]
            w.addItems(opts)
            if current is not None and str(current) in opts:
                w.setCurrentText(str(current))
            w.currentIndexChanged.connect(lambda *_: self._status.setText(""))
            return w
        w = QLineEdit("" if current is None else str(current))
        w.setPlaceholderText(f"默认: {p.default}")
        tip = p.hint or ""
        if p.min is not None and p.max is not None:
            tip = (tip + " " if tip else "") + f"范围 [{p.min}, {p.max}]"
        w.setToolTip(tip)
        w.textChanged.connect(lambda *_: self._status.setText(""))
        return w

    # ------------------------------------------------ 读值
    def collect(self) -> tuple[dict, list[str]]:
        """返回 (覆写字典, 错误列表)。非法值不写入覆写，错误就地显示。"""
        out, errs = {}, []
        for key, (p, w) in self._widgets.items():
            if p.kind == "bool":
                if w.isChecked() != bool(p.default):
                    out[key] = w.isChecked()
                continue
            raw = w.currentText() if p.kind == "select" else w.text()
            if p.kind == "select":
                if raw != str(p.default):
                    out[key] = raw
                continue
            if raw.strip() == "":
                continue
            try:
                out[key] = p.clean(raw)
            except ValueError as e:
                errs.append(str(e))
        if errs:
            self._status.setText("参数错误：" + "；".join(errs))
        return out, errs
