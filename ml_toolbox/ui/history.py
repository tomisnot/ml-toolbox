# -*- coding: utf-8 -*-
"""实验历史面板：从 runs/ 重建结果并回看（G3 实验记录可恢复）。

关键设计：estimator 不落盘（pickle 跨版本不可靠），只存 artifacts.npz；
所有核心图只消费 artifacts（契约 §3），因此历史运行无需重训即可完整检视。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
                            QTableWidgetItem, QPushButton, QLabel,
                            QHeaderView, QDialogButtonBox)

from ..core import persistence
from ..core.contracts import MLResult, RunConfig
from ..core.runner import RunRecord


def load_run_record(run_id: str) -> RunRecord | None:
    """record.json + artifacts.npz -> 可检视的 RunRecord（无 estimator）。"""
    try:
        meta = persistence.load_record(run_id)
    except Exception:
        return None
    res = MLResult(method_name=meta.get("method", "?"),
                   task=meta.get("task", "?"),
                   target_kind=meta.get("target_kind"),
                   metrics={k: v for k, v in meta.get("metrics", {}).items()
                            if isinstance(v, (int, float))},
                   primary_metric=meta.get("primary_metric", ""),
                   params=meta.get("params", {}),
                   elapsed=meta.get("elapsed", 0.0),
                   error=meta.get("error"))
    d = persistence.RUNS_DIR
    ap = os.path.join(d, run_id, "artifacts.npz")
    if os.path.exists(ap):
        with np.load(ap, allow_pickle=False) as z:
            flat = {k: z[k] for k in z.files}
        # 嵌套 dict 工件（save_record 用 "__" 展平）：递归重组
        res.artifacts = {}
        for k, v in flat.items():
            if "__" not in k:
                res.artifacts[k] = v
                continue
            *path, leaf = k.split("__")
            node = res.artifacts
            for p in path:
                node = node.setdefault(p, {})
            node[leaf] = v
    # 通用还原：<key>_values + <key>_names(+<key>_cols) -> DataFrame(<key>)
    for k in list(res.artifacts):
        if k.endswith("_values") and k[:-7] + "_names" in res.artifacts:
            base = k[:-7]
            vals = np.asarray(res.artifacts.pop(k), float)
            names = res.artifacts.pop(base + "_names")
            cols = res.artifacts.pop(base + "_cols", None)
            if cols is None:
                cols = np.array(["importance"])
                vals = vals.ravel()
            res.artifacts[base] = pd.DataFrame(
                vals.reshape(len(names), len(cols)),
                index=[str(s) for s in names], columns=[str(c) for c in cols])
    return RunRecord(run_id=run_id, method=res.method_name,
                     family=meta.get("family", "?"), task=res.task,
                     target_kind=res.target_kind,
                     dataset=meta.get("dataset", ""),
                     pipeline_id=meta.get("pipeline_id", ""),
                     config=meta.get("config", {}), result=res)


class HistoryDialog(QDialog):
    """历史运行列表；选中 -> load_requested(run_id)。"""

    load_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("实验历史（runs/）")
        self.resize(760, 460)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("每次遍历运行都自动存档；双击载入回看"
                             "（核心图基于存档工件重绘，无需重训）。"))
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
        self._ids: list[str] = []
        self._reload()

    def _reload(self):
        recs = persistence.list_records()
        cols = ["run_id", "method", "family", "dataset", "primary",
                "值", "elapsed_s", "saved_at", "状态"]
        self._table.clear()
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setRowCount(len(recs))
        self._ids = [r.get("run_id", "") for r in recs]
        for i, r in enumerate(recs):
            pm = r.get("primary_metric", "")
            val = r.get("metrics", {}).get(pm, "")
            vals = [r.get("run_id", ""), r.get("method", ""),
                    r.get("family", ""), r.get("dataset", ""), pm,
                    f"{val:.4g}" if isinstance(val, (int, float)) else str(val),
                    str(r.get("elapsed", "")), r.get("saved_at", ""),
                    "失败" if r.get("error") else "OK"]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j == 8 and v == "失败":
                    it.setForeground(Qt.red)
                self._table.setItem(i, j, it)
        self._table.resizeColumnsToContents()

    def _emit(self, row):
        if 0 <= row < len(self._ids):
            self.load_requested.emit(self._ids[row])
            self.accept()
