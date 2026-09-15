# -*- coding: utf-8 -*-
"""处理链页（模式 2/4）：每步一张全宽大卡片，右侧 19px 大号关键数字。"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QScrollArea, QFrame, QSizePolicy)

_STEP_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759",
                "#b07aa1", "#76b7b2", "#edc948"]


class StepCard(QFrame):
    """单张步骤卡片：左色条 + 徽标标题 + kv 明细 + 右侧大数字。"""

    def __init__(self, idx: int, step_desc: dict, summary: dict,
                 skipped: bool = False, is_profile: bool = False, parent=None):
        super().__init__(parent)
        color = _STEP_COLORS[idx % len(_STEP_COLORS)]
        self.setObjectName("stepCard")
        dim = "opacity:0.45;" if skipped else ""
        self.setStyleSheet(f"""
            QFrame#stepCard {{
                background:#fafbfc; border:1px solid #e3e6ea;
                border-radius:6px; {dim}
            }}""")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 10, 14, 10)
        lay.setSpacing(12)

        bar = QFrame()
        bar.setFixedWidth(5)
        bar.setStyleSheet(f"background:{color}; border-radius:2px;")
        lay.addWidget(bar)

        left = QVBoxLayout()
        prefix = "" if is_profile else f"步骤 {idx + 1} · "
        title = QLabel(prefix + step_desc.get("title", "?")
                       + ("（已跳过）" if skipped else ""))
        title.setStyleSheet(f"font-weight:bold; font-size:26px; color:{color};")
        left.addWidget(title)
        kv = summary or {}
        detail = QLabel("   ".join(
            f"{k}: {v}" for k, v in kv.items() if k not in ("big_num", "diag")))
        detail.setWordWrap(True)
        detail.setObjectName("muted")
        left.addWidget(detail)
        params = step_desc.get("params") or {}
        if params:
            pl = QLabel("参数: " + ", ".join(f"{k}={v}" for k, v in params.items()))
            pl.setObjectName("faint")
            left.addWidget(pl)
        lay.addLayout(left, 1)

        big = QLabel(str(kv.get("big_num", "")))
        big.setStyleSheet(f"font-size:38px; font-weight:bold; color:{color};")
        big.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(big)


class ChainPage(QWidget):
    """处理链页：数据画像卡 + 每步一卡；回答"软件对我做了什么"。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._host = QWidget()
        self._lay = QVBoxLayout(self._host)
        self._lay.setSpacing(8)
        self._lay.addStretch(1)
        self._scroll.setWidget(self._host)
        outer.addWidget(self._scroll)

    def show_spec(self, spec, dataset_profile: dict):
        # 清空旧卡片（保留尾部 stretch）
        while self._lay.count() > 1:
            it = self._lay.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        prof = StepCard(0, {"title": "数据画像"}, {
            "big_num": f"n = {spec.meta.get('n_samples', '?')}",
            "数据集": spec.meta.get("dataset", "?"),
            "来源": spec.meta.get("source", "?"),
            "样本数": dataset_profile.get("n_rows"),
            "特征数": spec.meta.get("n_features"),
            "目标": dataset_profile.get("target") or "（无）",
            "任务": dataset_profile.get("target_kind") or "无监督",
        }, is_profile=True)
        self._lay.insertWidget(0, prof)
        chain = spec.meta.get("chain", [])
        for i, entry in enumerate(chain):
            card = StepCard(i, entry.get("step", {}),
                            entry.get("summary", {}),
                            skipped=entry.get("skipped", False))
            card.setMinimumHeight(64)
            self._lay.insertWidget(i + 1, card)

    def clear(self):
        while self._lay.count() > 1:
            it = self._lay.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
