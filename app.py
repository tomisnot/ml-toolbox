# -*- coding: utf-8 -*-
"""ML Toolbox 桌面入口：python app.py"""
import faulthandler
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 崩溃现场：native 段错误/fail-fast 时把 Python 栈打到 stderr（日志兜底）
try:
    faulthandler.enable()
except Exception:
    pass


def _setup_logging():
    """logs/ml_toolbox.log：GUI 全链路留痕（实战排错刚需，见 pitfalls）。"""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(d, exist_ok=True)
    log_path = os.path.join(d, "ml_toolbox.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler()])
    # native 崩溃栈也写进日志文件（pythonw 启动时 stderr 会丢）
    try:
        _fh = open(log_path, "a", encoding="utf-8")
        faulthandler.enable(file=_fh, all_threads=True)
    except Exception:
        pass


_setup_logging()
_LOG = logging.getLogger("ml_toolbox.app")

# ⚠ 导入顺序铁律：lightgbm（经 methods 顶层导入）必须先于任何 PyQt5 模块，
# 否则其 OpenMP 与 Qt 冲突 -> fit 时 access violation（见 methods/ensemble.py）
from ml_toolbox.core import registry
registry.load_builtin()

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFontDatabase, QFont

from ml_toolbox.ui.main_window import MainWindow


def main():
    _LOG.info("=== GUI 启动 pid=%s ===", os.getpid())
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
    code = app.exec_()
    _LOG.info("=== GUI 正常退出 code=%s ===", code)
    sys.exit(code)


if __name__ == "__main__":
    main()
