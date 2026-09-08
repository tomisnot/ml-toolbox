# -*- coding: utf-8 -*-
"""基线族：数模里"任何模型都必须赢它"的参照物。

遍历结果若无方法显著优于 Dummy，说明特征/任务本身不可学——
这是判断"要不要继续建模"的第一道关。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.contracts import (MLMethod, RunConfig, PageSpec, ParamSpec,
                              TASK_SUPERVISED, regression_metrics,
                              classification_metrics)
from ..core.registry import register


@register
class DummyBaseline(MLMethod):
    name = "dummy"
    display_name = "基线（常数/先验）"
    family = "baseline"
    task = TASK_SUPERVISED
    target_kind = None
    tags = ("baseline", "must-beat", "reference")
    param_schema = [
        ParamSpec("strategy", "策略", "select", "prior",
                  choices=["prior", "uniform", "most_frequent", "median"]),
    ]

    def fit(self, X, y, cfg, diag=False):
        from sklearn.dummy import DummyRegressor, DummyClassifier
        from ..core.pipeline import _infer_kind
        p = self.params(cfg)
        kind = _infer_kind(y)
        strat = p["strategy"]
        if kind == "regression":
            est = DummyRegressor(strategy="mean" if strat == "prior" else strat)
            yv = np.asarray(y, float)
        else:
            est = DummyClassifier(strategy="prior" if strat == "prior" else strat)
            yv = np.asarray(y)
        est.fit(X.to_numpy(float), yv)
        res = self._new_result(target_kind=kind, params=p)
        res.est = est
        if diag:
            res.diag = {"note": "基线：任何真实模型都应显著优于它"}
        return res

    def predict(self, X, result):
        return result.est.predict(X.to_numpy(float))
