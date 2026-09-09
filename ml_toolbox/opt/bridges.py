# -*- coding: utf-8 -*-
"""三接缝桥（docs/优化定位.md §2）：ML 工具箱 <-> 优化框架 的互操作实体。

接缝1（ML→调参）：GPR 当 GP-BO 代理 —— 已在 engines/bo.py 直接复用
    sklearn GaussianProcessRegressor（purpose=surrogate 的兑现），无需额外代码。
接缝2（调参→ML）：**AutoTuner** —— 把"某方法在某数据集上的 CV 主指标"
    包装成 Objective，优化器调 RunConfig.overrides。数模"自动选最优超参"入口。
接缝3（调参历史→ML）：OptRecord.to_dataset() —— 已在 contracts 实现，
    评估历史回流 ML 做响应面/敏感性。本模块提供 response_surface() 便捷封装。

依赖方向：opt.bridges -> core（单向，不反向污染 core）。
"""
from __future__ import annotations

import numpy as np

from .contracts import Objective, ParamSpace
from ..core import registry as ml_registry
from ..core.contracts import RunConfig


# 越小越好的主指标（直接最小化）；其余指标取负
_LOWER_BETTER = {"rmse", "mae", "mape", "silhouette_deficit"}


def method_param_space(method_name: str,
                       exclude: tuple = ()) -> ParamSpace:
    """方法的 param_schema -> 可寻优 ParamSpace（无界/文本参数自动固定）。"""
    ml_registry.load_builtin()          # 幂等：库函数不要求调用方先注册
    m = ml_registry.get(method_name)
    return ParamSpace([p for p in m.param_schema if p.key not in exclude])


class AutoTunerObjective(Objective):
    """objective(params) = 方法在数据上的 CV 主指标（统一最小化方向）。

    数据来源三选一（接缝2 修正：不绑定 ML 工作区的当前数据集）：
      AutoTunerObjective("logistic", X, y, space)              # 裸数组（向后兼容）
      AutoTunerObjective("logistic", source, space=space)     # DataSource（文件/内部）
      source 也可以是 DataSpec（自动包成 SpecSource 快照）。
    """

    def __init__(self, method_name: str, X=None, y=None,
                 space: ParamSpace = None, metric: str = "",
                 cv_folds: int = 4, seed: int = 42, name: str = ""):
        # 位置参数宽容：AutoTunerObjective(name, source, space=...)
        if space is None and isinstance(y, ParamSpace):
            space, y = y, None
        if (y is None and not hasattr(X, "fetch") and not hasattr(X, "X")
                and not hasattr(X, "frame")):
            raise ValueError("需要 (X, y) 或 DataSource/DataSpec/Dataset")
        self._src = self._norm_source(X, y)
        ml_registry.load_builtin()       # 幂等：显式传 space 时也要能取到方法
        self._m = ml_registry.get(method_name)
        if space is None:
            space = ParamSpace([p for p in self._m.param_schema])
        self._cv = cv_folds
        self._seed = seed
        # 主指标：显式给定优先，否则按方法 target_kind 推（回归 rmse / 分类 f1）
        self._metric = metric or self._m._new_result(
            target_kind=getattr(self._m, "target_kind", None)).primary_metric \
            or "f1"
        super().__init__(name=name or f"autotune_{method_name}",
                         space=space, minimize=True)

    @staticmethod
    def _norm_source(X, y):
        from .sources import DataSource, SpecSource
        if isinstance(X, DataSource):
            return X
        if y is None and hasattr(X, "X"):        # DataSpec
            return SpecSource(X)
        if y is None and hasattr(X, "frame"):    # Dataset
            from ..core.pipeline import Pipeline
            return SpecSource(Pipeline.default().run(X))
        import pandas as pd
        from .sources import ArraySource
        return ArraySource(pd.DataFrame(X), None if y is None else pd.Series(y))

    def evaluate(self, params):
        X, y = self._src.fetch()               # 每次评估取最新数据（文件源=热更新）
        cfg = RunConfig(overrides=dict(params), seed=self._seed,
                        extras={"cv_folds": self._cv} if self._cv >= 2 else {})
        try:
            if self._cv >= 2:
                cv = self._m.cross_validate(X, y, cfg, self._cv)
                key = f"cv_{self._metric}_mean"
                if key not in cv:                      # 主指标名不匹配：取任一 cv_*_mean
                    key = next((k for k in cv if k.startswith("cv_")
                                and k.endswith("_mean")), None)
                score = cv.get(key, float("nan"))
            else:
                res = self._m.fit(X, y, cfg)
                pred = self._m.predict(X, res)
                res.artifacts["y_true"] = np.asarray(y)
                res.artifacts["y_pred"] = np.asarray(pred)
                m = self._m.evaluate(res, X, y)
                score = m.get(self._metric, float("nan"))
        except Exception:
            return float("inf")
        if not np.isfinite(score):
            return float("inf")
        # 统一最小化：越大越好的指标取负
        return score if self._metric in _LOWER_BETTER else -score

    def to_maximize(self, score_min):
        """把最小化分数还原成原始指标值（展示用）。"""
        return (-score_min if self._metric in _LOWER_BETTER else score_min) \
            if self._metric not in _LOWER_BETTER else score_min


def autotune(objective_or_name, X=None, y=None, optimizer=None,
             budget=None, cv_folds=4, metric="", seed=42, cfg=None):
    """一步式自动调参：返回 (OptRecord, best_params, best_metric_value)。

    数据侧三种入口都支持：
      autotune("logistic", X, y, ...)                  # 裸数组
      autotune("logistic", source, ...)                # DataSource/DataSpec/Dataset
    """
    from .runner import optimize
    from . import registry as opt_registry
    if isinstance(objective_or_name, str):
        name = objective_or_name
        obj = AutoTunerObjective(name, X, y, metric=metric,
                                 cv_folds=cv_folds, seed=seed)
    else:
        obj = objective_or_name
    opt = optimizer or opt_registry.get("gp_bo")
    rec = optimize(obj, opt, budget, cfg=cfg, seed=seed)
    best_val = None
    if rec.best:
        bv = rec.best["score"]
        lower = getattr(obj, "_metric", "score") in _LOWER_BETTER
        best_val = bv if lower else -bv
    return rec, (rec.best or {}), best_val


# ================================================================ 接缝3
def response_surface(opt_record, method_name: str = "ridge"):
    """评估历史 -> 拟合响应面（回归器），返回 (method, result, R²)。

    数模敏感性分析：哪些参数真正影响结果。历史经 to_dataset 回流 ML 工具箱，
    用任意监督方法拟合"参数->分数"映射。R² 高说明该参数域确实有结构可学。
    """
    from ..core.dataset import Dataset
    from ..core import runner
    from ..core.contracts import RunConfig
    ds = opt_record.to_dataset()
    if ds is None:
        return None
    spec = _quick_spec(ds)
    m = ml_registry.get(method_name)
    rec = runner.run_one(m, spec, RunConfig())
    r2 = rec.result.metrics.get("r2", float("nan"))
    return m, rec.result, r2


def _quick_spec(ds: Dataset):
    """Dataset -> DataSpec（响应面：分数是连续目标，强制回归 + 禁分层）。"""
    from ..core.pipeline import Pipeline
    spec = Pipeline(steps=Pipeline.default().steps, stratify=False).run(ds)
    spec.target_kind = "regression"      # 分数列会被 _infer_kind 误判为分类
    spec.n_classes = 0
    return spec
