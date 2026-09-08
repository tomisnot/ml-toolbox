# -*- coding: utf-8 -*-
"""ML Toolbox 桌面入口：python app.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ⚠ 导入顺序铁律：lightgbm（经 methods 顶层导入）必须先于任何 PyQt5 模块，
# 否则其 OpenMP 与 Qt 冲突 -> fit 时 access violation（见 methods/ensemble.py）
from ml_toolbox.core import registry
registry.load_builtin()

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFontDatabase, QFont

from ml_toolbox.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    for fn in ("msyh.ttc", "msyh.ttf", "simhei.ttf"):
        p = os.path.join("C:\\", "Windows", "Fonts", fn)
        if os.path.exists(p):
            fid = QFontDatabase.addApplicationFont(p)
            fams = QFontDatabase.applicationFontFamilies(fid)
            if fams:
                app.setFont(QFont(fams[0], 10))
                break
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
