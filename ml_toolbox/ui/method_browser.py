# -*- coding: utf-8 -*-
"""方法浏览器（G1 入口）：按 family 分组列出注册方法，勾选进入遍历候选集。"""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QFrame, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
                             QLineEdit, QLabel, QHBoxLayout, QPushButton,
                             QComboBox)

from ..core import registry
from ..core.contracts import MLMethod, PURPOSE_LABELS


class MethodBrowser(QFrame):
    """左栏：搜索 + 分组树。勾选方法 -> selection_changed(names)。"""

    selection_changed = pyqtSignal(list)
    method_activated = pyqtSignal(str)     # 单击某方法 -> 检视/调参

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(6)

        head = QLabel("方法库")
        head.setObjectName("cardTitle")
        lay.addWidget(head)

        self._search = QLineEdit()
        self._search.setPlaceholderText("搜索方法 / 标签…")
        self._search.textChanged.connect(self._refilter)
        lay.addWidget(self._search)

        prow = QHBoxLayout()
        prow.addWidget(QLabel("用途"))
        self._purpose = QComboBox()
        self._purpose.addItem("全部")
        self._purpose.addItems(list(PURPOSE_LABELS.values()))
        # 注意：勿直连 _refilter——currentTextChanged 会把用途名当搜索词传入
        self._purpose.currentTextChanged.connect(
            lambda *_: self._refilter(self._search.text()))
        prow.addWidget(self._purpose, 1)
        lay.addLayout(prow)

        row = QHBoxLayout()
        b_all = QPushButton("全选")
        b_none = QPushButton("清空")
        for b in (b_all, b_none):
            b.clicked.connect(lambda _=False, x=b: self._bulk(x is b_all))
            row.addWidget(b)
        lay.addLayout(row)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        lay.addWidget(self._tree, 1)

        self._count = QLabel("")
        self._count.setObjectName("muted")
        lay.addWidget(self._count)

        self._build()
        self._tree.itemChanged.connect(self._on_change)
        self._tree.itemClicked.connect(self._on_click)

    def _build(self):
        self._tree.blockSignals(True)
        self._tree.clear()
        fams = registry.families()
        for fam, names in fams.items():
            top = QTreeWidgetItem([f"{fam}  ({len(names)})"])
            top.setFlags(top.flags() | Qt.ItemIsUserCheckable)
            top.setCheckState(0, Qt.PartiallyChecked)
            top.setData(0, Qt.UserRole, "__family__")
            for nm in names:
                m = registry.get(nm)
                it = QTreeWidgetItem([m.display_name])
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(0, Qt.Unchecked)
                it.setData(0, Qt.UserRole, nm)
                it.setToolTip(0, f"{nm}\ntask={m.task}  "
                                 f"用途={m.purposes_cn()}  "
                                 f"tags={','.join(m.tags)}")
                top.addChild(it)
            self._tree.addTopLevelItem(top)
            top.setExpanded(True)
        self._tree.blockSignals(False)
        self._refilter("")

    def _refilter(self, text: str):
        text = (text or "").lower()
        ptxt = getattr(self, "_purpose", None)
        want = None if (ptxt is None or ptxt.currentText() == "全部") \
            else ptxt.currentText()
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            any_vis = False
            for j in range(top.childCount()):
                ch = top.child(j)
                m = registry.get(ch.data(0, Qt.UserRole))
                hay = " ".join([m.name, m.display_name, *m.tags]).lower()
                vis = ((not text) or text in hay) and \
                      (want is None or want in m.purposes_cn())
                ch.setHidden(not vis)
                any_vis = any_vis or vis
            top.setHidden(not any_vis)

    def _bulk(self, checked: bool):
        self._tree.blockSignals(True)
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            if top.isHidden():
                continue
            for j in range(top.childCount()):
                ch = top.child(j)
                if not ch.isHidden():
                    ch.setCheckState(0, state)
        self._tree.blockSignals(False)
        self._emit()

    def _on_change(self, item, _col):
        if item.data(0, Qt.UserRole) == "__family__":
            self._tree.blockSignals(True)
            st = item.checkState(0)
            for j in range(item.childCount()):
                if st != Qt.PartiallyChecked:
                    item.child(j).setCheckState(0, st)
            self._tree.blockSignals(False)
        self._emit()

    def _on_click(self, item, _col):
        nm = item.data(0, Qt.UserRole)
        if nm and nm != "__family__":
            self.method_activated.emit(nm)

    def _emit(self):
        names = self.selected()
        self._count.setText(f"已选 {len(names)} 个方法")
        self.selection_changed.emit(names)

    def selected(self) -> list[str]:
        out = []
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            for j in range(top.childCount()):
                ch = top.child(j)
                if ch.checkState(0) == Qt.Checked:
                    out.append(ch.data(0, Qt.UserRole))
        return out
