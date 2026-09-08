# -*- coding: utf-8 -*-
"""优化器引擎族。导入本包即完成注册（与 methods/ 同构）。

阶段 0：baseline（随机搜索 / 网格）——参照物，任何优化器都该赢它
        （对标 ML 侧的 dummy）。
阶段 1：bo（自研 GP-BO，复用 GPR 代理）。
阶段 3：evo / schedule / local。
"""
from . import baseline  # noqa: F401
try:
    from . import bo     # noqa: F401
except Exception:
    pass
