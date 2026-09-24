# -*- coding: utf-8 -*-
"""优化器引擎族。导入本包即完成注册（与 methods/ 同构）。

阶段 0：baseline（随机搜索 / 网格）——参照物，任何优化器都该赢它
        （对标 ML 侧的 dummy）。
阶段 1：bo（自研 GP-BO，复用 GPR 代理）。
阶段 3：evo（自研 CMA-ES）/ local（自研 Nelder-Mead）/
        optuna 适配（tpe / nsga_ii / asha，未装 optuna 自动跳过）。
量子启发：quantum（自研横场量子退火 SQA，书第四章）。
"""
from . import baseline  # noqa: F401
from . import anneal    # noqa: F401
try:
    from . import bo     # noqa: F401
except Exception:
    pass
try:
    from . import evo    # noqa: F401
except Exception:
    pass
try:
    from . import local  # noqa: F401
except Exception:
    pass
try:
    from . import optuna_bridge  # noqa: F401
except Exception:
    pass
