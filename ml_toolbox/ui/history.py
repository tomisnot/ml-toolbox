# -*- coding: utf-8 -*-
"""实验历史面板：从 runs/ 重建结果并回看（G3 实验记录可恢复）。

关键设计：estimator 不落盘（pickle 跨版本不可靠），只存 artifacts.npz；
所有核心图只消费 artifacts（契约 §3），因此历史运行无需重训即可完整检视。
"""
from __future__ import annotations

import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
                            QTableWidgetItem, QPushButton, QLabel,
                            QHeaderView, QDialogButtonBox)

from ..core import persistence
from ..core.contracts import RunConfig
from ..core.runner import RunRecord


def load_run_record(run_id: str) -> RunRecord | None:
    """record.json + artifacts.npz -> 可检视的 RunRecord（无 estimator）。

    ⚠ **实现已下沉到 `core.persistence.load_run_record`**（2026-10-03）：那是纯数据访问、
    零 Qt；而边界层 `ml_mecha/` 不许 import 本层（宪章纪律，判据守着），
    面板又正需要它 ⇒ 只能下沉，否则要么违规、要么两处各抄一份（两处可以各自漂移）。
    本函数**保留为转发**，GUI 既有调用点一行不改。
    """
    from ..core.persistence import load_run_record as _core_loader
    return _core_loader(run_id)


class HistoryDialog(QDialog):
    """历史运行列表（ML + 优化统一）；选中 -> load_requested(run_id, kind)。"""

    load_requested = pyqtSignal(str, str)        # run_id, kind("ML"|"优化")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("实验历史（runs/）")
        self.resize(800, 460)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("每次运行都自动存档；双击载入回看。ML 运行基于存档工件"
                             "重绘核心图（无需重训）；优化运行载入评估历史到调参工作区。"))
        self._table = QTableWidget()
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._table.cellDoubleClicked.connect(
            lambda r, c: self._emit(r))
        lay.addWidget(self._table, 1)
        row = QHBoxLayout()
        self._btn_load = QPushButton("载入所选")
        self._btn_load.clicked.connect(lambda: self._emit(self._table.currentRow()))
        self._btn_refresh = QPushButton("刷新")
        self._btn_refresh.clicked.connect(self._reload)
        close = QPushButton("关闭")
        close.clicked.connect(self.reject)
        row.addWidget(self._btn_load)
        row.addWidget(self._btn_refresh)
        row.addStretch(1)
        row.addWidget(close)
        lay.addLayout(row)
        self._ids: list[tuple] = []              # (run_id, kind)
        self._reload()

    def _reload(self):
        from ..core import runs
        recs = runs.list_all_records(persistence.RUNS_DIR,
                                     _opt_runs_dir())
        cols = ["类型", "run_id", "方法/优化器", "族/目标", "数据集/objective",
                "primary", "值", "elapsed_s", "saved_at", "状态"]
        self._table.clear()
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setRowCount(len(recs))
        self._ids = [(r.get("run_id", ""), r.get("_kind", "ML")) for r in recs]
        for i, r in enumerate(recs):
            val = r.get("value", "")
            vals = [r.get("_kind", ""), r.get("run_id", ""),
                    r.get("name", ""), r.get("family", ""),
                    r.get("dataset", ""), r.get("primary", ""),
                    f"{val:.4g}" if isinstance(val, (int, float)) else str(val),
                    str(r.get("elapsed", "")), r.get("saved_at", ""),
                    "失败" if r.get("error") else "OK"]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j == 9 and v == "失败":
                    it.setForeground(Qt.red)
                self._table.setItem(i, j, it)
        self._table.resizeColumnsToContents()

    def _emit(self, row):
        if 0 <= row < len(self._ids):
            rid, kind = self._ids[row]
            self.load_requested.emit(rid, kind)
            self.accept()


def _opt_runs_dir() -> str:
    from ..opt import persistence as opt_persistence
    return opt_persistence.RUNS_DIR
