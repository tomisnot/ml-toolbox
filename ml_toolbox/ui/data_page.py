# -*- coding: utf-8 -*-
"""数据检视页：特征相关热图（自适应缩放）+ 特征统计摘要。

G3 统一数据管理的可视化落点：所有方法共享的这份 X，长什么样一眼看清。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter
from PyQt5.QtCore import Qt

from .heatmap import AdaptiveMatrixHeatmap


class DataPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.summary = QLabel("（未加载数据）")
        self.summary.setStyleSheet("font-size:12px; color:#555;")
        lay.addWidget(self.summary)
        self.heat = AdaptiveMatrixHeatmap()
        lay.addWidget(self.heat, 1)

    def show_spec(self, spec):
        X = spec.X
        num = X.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            self.summary.setText(f"特征 {X.shape[1]} 列（数值列 <2，无相关热图）")
            return
        corr = num.corr().to_numpy(float)
        names = list(num.columns)
        self.heat.set_data(np.nan_to_num(corr, nan=0.0), names)
        # 最强相关对摘要
        n = len(names)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((abs(corr[i, j]), names[i], names[j], corr[i, j]))
        pairs.sort(reverse=True)
        top = "; ".join(f"{a}~{b} r={r:+.2f}" for _, a, b, r in pairs[:3])
        self.summary.setText(
            f"特征 {X.shape[1]} 列 · 数值 {n} 列 · 样本 {spec.meta.get('n_samples')} · "
            f"最强相关: {top}")
