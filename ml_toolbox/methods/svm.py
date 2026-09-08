# -*- coding: utf-8 -*-
"""SVM / 核方法族。"""
from __future__ import annotations

import numpy as np

from ..core.contracts import ParamSpec
from ..core.registry import register
from .base import SklearnSupervised


@register
class SVC(SklearnSupervised):
    name = "svc"
    display_name = "支持向量机"
    family = "svm"
    target_kind = None
    tags = ("kernel", "small-sample")
    param_schema = [
        ParamSpec("kernel", "核函数", "select", "rbf",
                  choices=["rbf", "linear", "poly", "sigmoid"]),
        ParamSpec("C", "惩罚系数 C", "number", 1.0, min=0.01),
        ParamSpec("gamma", "核宽度 γ", "select", "scale",
                  choices=["scale", "auto"], hint="数值列需已缩放"),
        ParamSpec("class_weight", "类别加权", "select", "none",
                  choices=["none", "balanced"]),
    ]

    def make(self, p, kind):
        from sklearn.svm import SVC as _SVC, SVR
        kw = dict(kernel=p["kernel"], C=float(p["C"]), gamma=p["gamma"])
        if kind == "classification":
            kw.update(probability=True, random_state=42)
            if p["class_weight"] == "balanced":
                kw["class_weight"] = "balanced"
            return _SVC(**kw)
        return SVR(**kw)


@register
class LinearSVC(SklearnSupervised):
    name = "linear_svc"
    display_name = "线性 SVM"
    family = "svm"
    target_kind = "classification"
    tags = ("kernel", "fast", "interpretable")
    param_schema = [
        ParamSpec("C", "惩罚系数 C", "number", 1.0, min=0.01),
        ParamSpec("class_weight", "类别加权", "select", "none",
                  choices=["none", "balanced"]),
    ]

    def make(self, p, kind):
        from sklearn.svm import LinearSVC
        kw = dict(C=float(p["C"]), max_iter=5000, random_state=42)
        if p["class_weight"] == "balanced":
            kw["class_weight"] = "balanced"
        return LinearSVC(**kw)

    def fit_extra_artifacts(self, X, result):
        """LinearSVC 无 predict_proba：用决策函数值做 min-max 近似概率。"""
        est = getattr(result, "est", None)
        if est is None:
            return {}
        try:
            d = est.decision_function(X.to_numpy(float))
            if d.ndim == 1:
                p = 1 / (1 + __import__("numpy").exp(-d))
                return {"y_prob": __import__("numpy").column_stack([1 - p, p])}
            return {"y_prob": _softmax(d)}
        except Exception:
            return {}


def _softmax(d):
    import numpy as np
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@register
class NuSVC(SklearnSupervised):
    name = "nusvc"
    display_name = "ν-SVM"
    family = "svm"
    target_kind = "classification"
    tags = ("kernel",)
    param_schema = [
        ParamSpec("nu", "ν 上界比例", "number", 0.5, min=0.05, max=0.95),
        ParamSpec("kernel", "核函数", "select", "rbf",
                  choices=["rbf", "linear", "poly"]),
    ]

    def make(self, p, kind):
        from sklearn.svm import NuSVC
        return NuSVC(nu=float(p["nu"]), kernel=p["kernel"],
                     probability=True, random_state=42)
