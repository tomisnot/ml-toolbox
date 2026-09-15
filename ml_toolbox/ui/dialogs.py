# -*- coding: utf-8 -*-
"""主窗口的两个对话框（C7-7a 自 main_window.py 抽出，纯搬运不改行为）。

TargetDialog：列角色（目标/时间列）；PipelineDialog：预处理管道编辑。
main_window 经 from .dialogs import ... 使用（test_ui 兼容旧导入路径）。
"""
from __future__ import annotations

import json

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QDialogButtonBox, QLineEdit,
                             QCheckBox, QPushButton, QTableWidget,
                             QTableWidgetItem, QHeaderView, QMessageBox)

from ..core.pipeline import Pipeline, BUILTIN_STEPS

class TargetDialog(QDialog):
    """选择目标列 / 时间列。"""

    def __init__(self, parent, columns):
        super().__init__(parent)
        self.setWindowTitle("列角色")
        lay = QVBoxLayout(self)
        form = QHBoxLayout()
        form.addWidget(QLabel("目标列（y）："))
        self._t = QComboBox()
        self._t.addItems(["（无 / 无监督）"] + list(columns))
        form.addWidget(self._t, 1)
        lay.addLayout(form)
        form2 = QHBoxLayout()
        form2.addWidget(QLabel("时间列："))
        self._c = QComboBox()
        self._c.addItems(["（无）"] + list(columns))
        form2.addWidget(self._c, 1)
        lay.addLayout(form2)
        from ..core.contracts import REGRESSION_CARDINALITY
        lay.addWidget(QLabel(f"提示：数值且类别数 >{REGRESSION_CARDINALITY} "
                             "视为回归，否则分类。"))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def result_(self):
        return ("" if self._t.currentIndex() == 0 else self._t.currentText(),
                "" if self._c.currentIndex() == 0 else self._c.currentText())


class PipelineDialog(QDialog):
    """管道步骤编辑：启用/禁用、参数覆写、顺序（上移/下移）。"""

    def __init__(self, parent, pipeline: Pipeline):
        super().__init__(parent)
        self.setWindowTitle("预处理管道")
        self.resize(560, 460)
        self._pipe = Pipeline(steps=[type(s)(enabled=s.enabled, **s.params)
                                     for s in pipeline.steps],
                              seed=pipeline.seed, test_size=pipeline.test_size,
                              stratify=pipeline.stratify,
                              time_split=pipeline.time_split)
        lay = QVBoxLayout(self)
        self._list = QTableWidget(len(self._pipe.steps), 4)
        self._list.setHorizontalHeaderLabels(["启用", "步骤", "参数(JSON)", ""])
        self._list.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._refresh_list()
        lay.addWidget(self._list, 1)

        row = QHBoxLayout()
        for nm, key in (("缺失值", "missing"), ("编码", "encode"),
                        ("缩放", "scale"), ("特征筛选", "feature_select"),
                        ("划分", "split")):
            b = QPushButton(f"+ {nm}")
            b.clicked.connect(lambda _=False, k=key: self._add_step(k))
            row.addWidget(b)
        lay.addLayout(row)

        form = QHBoxLayout()
        form.addWidget(QLabel("test_size"))
        self._ts = QLineEdit(str(self._pipe.test_size))
        self._ts.setFixedWidth(120)
        form.addWidget(self._ts)
        self._time = QCheckBox("按时间顺序切分（时序）")
        self._time.setChecked(self._pipe.time_split)
        form.addWidget(self._time)
        lay.addLayout(form)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _refresh_list(self):
        self._list.setRowCount(len(self._pipe.steps))
        for i, st in enumerate(self._pipe.steps):
            chk = QCheckBox()
            chk.setChecked(st.enabled)
            chk.stateChanged.connect(lambda _s, idx=i: self._toggle(idx))
            self._list.setCellWidget(i, 0, chk)
            self._list.setItem(i, 1, QTableWidgetItem(
                f"{st.title} ({st.key})"))
            import json
            self._list.setItem(i, 2, QTableWidgetItem(
                json.dumps(st.params, ensure_ascii=False)))
            mv = QPushButton("删")
            mv.clicked.connect(lambda _=False, idx=i: self._del_step(idx))
            self._list.setCellWidget(i, 3, mv)

    def _toggle(self, idx):
        w = self._list.cellWidget(idx, 0)
        self._pipe.steps[idx].enabled = w.isChecked()

    def _add_step(self, key):
        self._pipe.steps.append(BUILTIN_STEPS[key]())
        self._refresh_list()

    def _del_step(self, idx):
        del self._pipe.steps[idx]
        self._refresh_list()

    def _accept(self):
        import json
        for i, st in enumerate(self._pipe.steps):
            raw = self._list.item(i, 2).text().strip()
            if raw:
                try:
                    st.params = json.loads(raw)
                except Exception:
                    QMessageBox.warning(self, "参数 JSON 非法",
                                        f"步骤 {st.key}: {raw}")
                    return
        try:
            self._pipe.test_size = float(self._ts.text())
        except ValueError:
            pass
        self._pipe.time_split = self._time.isChecked()
        self.accept()

    def result_pipeline(self) -> Pipeline:
        return self._pipe
