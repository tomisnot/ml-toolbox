# -*- coding: utf-8 -*-
"""DataFrame -> QTableWidget 的统一渲染（C7-7b：消三份复制粘贴）。

原状：main_window._fill_compare_table、opt_page._refresh_history/_refresh_compare、
inspector._make_table 各写一遍"清表 + 表头 + 逐格格式化（float .4g / NaN 空）
+ 特定列红字"。差异只在：是否显示行索引、哪些列标红。

render_dataframe(tw, df, red_when=None, show_index=False) 收敛全部差异：
- red_when(col, text) -> bool：决定该格红字（error 非空 / status==failed）；
- show_index：inspector 的表格页显示 DataFrame 行索引，其余不显示。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem


def _cell_text(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or (
            isinstance(v, np.floating) and np.isnan(v)):
        return ""
    if isinstance(v, (int, float, np.integer, np.floating)) \
            and not isinstance(v, bool):
        return f"{float(v):.4g}"
    return str(v)


def render_dataframe(tw: QTableWidget, df: pd.DataFrame,
                     red_when=None, show_index: bool = False) -> QTableWidget:
    """把 df 渲染进 tw（就地填充）。red_when(col, text)->bool 决定红字。"""
    tw.clear()
    cols = list(df.columns)
    tw.setColumnCount(len(cols))
    tw.setHorizontalHeaderLabels([str(c) for c in cols])
    tw.setRowCount(len(df))
    if show_index:
        tw.setVerticalHeaderLabels([str(i)[:24] for i in df.index])
    for i in range(len(df)):
        for j, c in enumerate(cols):
            txt = _cell_text(df.iat[i, j])
            it = QTableWidgetItem(txt)
            if red_when is not None and red_when(c, txt):
                it.setForeground(Qt.red)
            tw.setItem(i, j, it)
    tw.resizeColumnsToContents()
    return tw
