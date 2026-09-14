# -*- coding: utf-8 -*-
"""参数面板（模式 3）：旋钮放在语义位置 + 就地重跑。

- 每个方法的 param_schema 自动生成控件；
- 空 = 默认（占位符写"默认"），覆写仅本次运行生效、不写回全局；
- "⟳ 应用并重跑" 发信号给主窗口触发新一轮运行。

控件构建与三态回收委托 ParamForm（C8：与优化侧超参面板共享一份实现）。
"""
from __future__ import annotations

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QPushButton,
                            QLabel, QScrollArea, QFrame)

from ..core.contracts import RunConfig
from .param_form import ParamForm


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

        self._pf = ParamForm(self._form, status_cb=self._status.setText)

    # ------------------------------------------------ 构建
    @property
    def _widgets(self):
        return self._pf.widgets          # 兼容既有测试/调用点的读接口

    def set_method(self, method, cfg: RunConfig):
        self._title.setText(f"{method.display_name} · 参数")
        self._pf.build(method.param_schema, cfg.overrides)

    # ------------------------------------------------ 读值
    def collect(self) -> tuple[dict, list]:
        """返回 (覆写字典, 错误列表)。非法值不写入覆写，错误就地显示。"""
        return self._pf.collect()
