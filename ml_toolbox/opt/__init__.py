# -*- coding: utf-8 -*-
"""ml_toolbox.opt —— 序贯黑盒优化 / 自动调参子框架（方案：内部记录（未随仓发布，存档在仓外））。

与 ML 工具箱的关系 = 兄弟框架 + 共享内核 + 三接缝：
- 共享：core.ParamSpec（参数空间）、registry 模式、PageSpec 检视契约、
  persistence 留痕思想、ui/inspector 装配器；
- 独立：Optimizer.ask/tell 增量契约不与 MLMethod.fit 批式契约混血；
- 互操作（阶段4）：GPR 当 BO 代理 / AutoTuner 调 ML 超参 / 评估历史回流 ML。

零 Qt（P8）：本包可完全脱离界面使用::

    from ml_toolbox.opt import registry
    from ml_toolbox.opt.contracts import ParamSpace, Budget, make_objective
    from ml_toolbox.core.contracts import ParamSpec
    from ml_toolbox.opt.runner import optimize

    registry.load_builtin()
    space = ParamSpace([ParamSpec("x", "x", "number", 0.0, min=-5, max=5)])
    obj = make_objective(lambda p: (p["x"] - 2.3) ** 2, space, name="quad")
    rec = optimize(obj, registry.get("random_search"), Budget(n_evals=30))
    print(rec.best)
"""
from .contracts import (ParamSpace, Objective, CallableObjective, Optimizer,  # noqa: F401
                        OptRecord, Budget, make_objective)   # noqa: F401
