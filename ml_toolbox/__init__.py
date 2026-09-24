# -*- coding: utf-8 -*-
"""ML Toolbox —— 可复用的机器学习工具箱。

核心逻辑与界面分离（原则 P1）：本包不含任何 Qt 依赖，
GUI 位于 ml_toolbox.ui，通过契约消费本包。
"""
__version__ = "0.2.0"

# Stable headless facade.  Importing the package remains Qt-free; GUI code is
# still isolated under ml_toolbox.ui.
from .api import API_VERSION, RunRequest, RunSnapshot, RunState, Session, TerminalReason

__all__ = [
    "__version__", "API_VERSION", "Session", "RunRequest", "RunSnapshot",
    "RunState", "TerminalReason",
]
