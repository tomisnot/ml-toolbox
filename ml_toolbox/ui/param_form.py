# -*- coding: utf-8 -*-
"""ParamSpec -> 控件 + 三态回收的共享逻辑（M9/C8：消除两份参数面板实现）。

ML 侧 ParamPanel 与优化侧超参面板此前各写一份"建控件 + 读覆写"，且优化侧
的"面板归属"（O8：超参只作用于当前选中的那个优化器）是第二份实现上的补丁。
本模块把两件事收敛为一：

    form = ParamForm(qformlayout, status_cb)
    form.build(schema, overrides, owner="gp_bo")   # 建控件
    ov, errs = form.collect()                       # 三态回收（空=默认不写入）
    form.owner                                      # 当前归属（opt 侧用）

三态语义（唯一实现）：
- bool：勾选 != 默认 才写入；
- select：当前值 != 默认 才写入；
- number/int：空 -> 不写（用默认）；非法 -> 进 errs、不写；合法 -> 写入。
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFormLayout, QLineEdit, QComboBox, QCheckBox, QLabel

from ..core.contracts import ParamSpec


class ParamForm:
    def __init__(self, form: QFormLayout,
                 status_cb: Optional[Callable[[str], None]] = None):
        self._form = form
        self._status_cb = status_cb or (lambda msg: None)
        self.widgets: dict[str, tuple[ParamSpec, object]] = {}
        self.owner: str = ""          # 面板归属的优化器名（ML 侧忽略）

    # ------------------------------------------------ 构建
    def build(self, schema, overrides: Optional[dict] = None,
              owner: str = "", show_default: bool = False,
              field_width: Optional[int] = None):
        """建控件。show_default=True 时数值框预填默认值（优化器超参面板观感），
        False 时空框 + "默认:…" 占位（ML 参数面板的三态观感）。
        field_width：数值框固定像素宽（优化侧窄栏用 90，ML 侧 None=自适应）。"""
        overrides = overrides or {}
        self.clear()
        self.owner = owner
        for p in schema:
            w = self._make_widget(p, overrides.get(p.key), show_default,
                                  field_width)
            self._form.addRow(self._label(p), w)
            self.widgets[p.key] = (p, w)
        self._status_cb("")

    def clear(self):
        while self._form.rowCount():
            self._form.removeRow(0)
        self.widgets.clear()

    @staticmethod
    def _label(p: ParamSpec):
        lab = QLabel(p.label or p.key)
        lab.setToolTip(p.hint or p.key)
        return lab

    def _make_widget(self, p: ParamSpec, current, show_default: bool = False,
                     field_width: Optional[int] = None):
        if p.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(current) if current is not None else bool(p.default))
            w.stateChanged.connect(lambda *_: self._status_cb(""))
            return w
        if p.kind == "select":
            w = QComboBox()
            opts = [str(c) for c in (p.choices or [])]
            w.addItems(opts)
            if current is not None and str(current) in opts:
                w.setCurrentText(str(current))
            w.currentIndexChanged.connect(lambda *_: self._status_cb(""))
            return w
        if show_default and current is None:
            w = QLineEdit(str(p.default))
        else:
            w = QLineEdit("" if current is None else str(current))
            w.setPlaceholderText(f"默认: {p.default}")
        if field_width:
            w.setFixedWidth(field_width)
        tip = p.hint or ""
        if p.min is not None and p.max is not None:
            tip = (tip + " " if tip else "") + f"范围 [{p.min}, {p.max}]"
        w.setToolTip(tip)
        w.textChanged.connect(lambda *_: self._status_cb(""))
        return w

    # ------------------------------------------------ 回收
    def collect(self) -> tuple[dict, list]:
        """返回 (覆写字典, 错误列表)。非法值不写入覆写，错误就地显示。"""
        out, errs = {}, []
        for key, (p, w) in self.widgets.items():
            if p.kind == "bool":
                if w.isChecked() != bool(p.default):
                    out[key] = w.isChecked()
                continue
            if p.kind == "select":
                raw = w.currentText()
                if raw != str(p.default):
                    out[key] = raw
                continue
            raw = w.text()
            if raw.strip() == "":
                continue
            try:
                out[key] = p.clean(raw)
            except ValueError as e:
                errs.append(str(e))
        if errs:
            self._status_cb("参数错误：" + "；".join(errs))
        return out, errs

    def values(self) -> dict:
        """当前控件值（配置导出用，不做"改没改"过滤）。"""
        out = {}
        for key, (p, w) in self.widgets.items():
            if p.kind == "bool":
                out[key] = w.isChecked()
            elif p.kind == "select":
                out[key] = w.currentText()
            else:
                try:
                    out[key] = p.clean(w.text())
                except ValueError:
                    out[key] = p.default
        return out

    def set_values(self, vals: dict):
        """按 key 回填控件（配置导入用）；未知 key 忽略。"""
        for k, v in (vals or {}).items():
            pair = self.widgets.get(k)
            if pair is None:
                continue
            p, w = pair
            if p.kind == "bool":
                w.setChecked(bool(v))
            elif p.kind == "select":
                i = w.findText(str(v))
                if i >= 0:
                    w.setCurrentIndex(i)
            else:
                w.setText("" if v is None else str(v))
