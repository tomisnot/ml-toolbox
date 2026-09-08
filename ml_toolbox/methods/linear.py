# -*- coding: utf-8 -*-
"""线性模型族：回归 + 分类。"""
from __future__ import annotations

from ..core.contracts import ParamSpec, PageSpec
from ..core.registry import register
from .base import SklearnSupervised
from . import plots


@register
class LinearRegression(SklearnSupervised):
    name = "linear_regression"
    display_name = "普通最小二乘"
    family = "linear"
    target_kind = "regression"
    tags = ("baseline", "interpretable")
    param_schema = [
        ParamSpec("fit_intercept", "截距", "bool", True),
        ParamSpec("positive", "系数非负", "bool", False,
                  hint="所有系数 >= 0（物理量常为非负）"),
    ]

    def make(self, p, kind):
        from sklearn.linear_model import LinearRegression as _LR
        return _LR(fit_intercept=bool(p["fit_intercept"]),
                   positive=bool(p["positive"]))


@register
class Ridge(SklearnSupervised):
    name = "ridge"
    display_name = "Ridge 回归"
    family = "linear"
    target_kind = "regression"
    tags = ("regularized", "collinearity")
    param_schema = [
        ParamSpec("alpha", "L2 强度 α", "number", 1.0, min=0),
    ]

    def make(self, p, kind):
        from sklearn.linear_model import Ridge
        return Ridge(alpha=float(p["alpha"]))


@register
class Lasso(SklearnSupervised):
    name = "lasso"
    display_name = "Lasso 回归"
    family = "linear"
    target_kind = "regression"
    tags = ("sparse", "feature-selection")
    param_schema = [
        ParamSpec("alpha", "L1 强度 α", "number", 1.0, min=0),
        ParamSpec("max_iter", "最大迭代", "int", 5000, min=100),
    ]

    def make(self, p, kind):
        from sklearn.linear_model import Lasso
        return Lasso(alpha=float(p["alpha"]), max_iter=int(p["max_iter"]))


@register
class ElasticNet(SklearnSupervised):
    name = "elasticnet"
    display_name = "ElasticNet"
    family = "linear"
    target_kind = "regression"
    tags = ("sparse", "regularized")
    param_schema = [
        ParamSpec("alpha", "总强度 α", "number", 1.0, min=0),
        ParamSpec("l1_ratio", "L1 比例", "number", 0.5, min=0, max=1),
        ParamSpec("max_iter", "最大迭代", "int", 5000, min=100),
    ]

    def make(self, p, kind):
        from sklearn.linear_model import ElasticNet
        return ElasticNet(alpha=float(p["alpha"]), l1_ratio=float(p["l1_ratio"]),
                          max_iter=int(p["max_iter"]))


@register
class Huber(SklearnSupervised):
    name = "huber"
    display_name = "Huber 稳健回归"
    family = "linear"
    target_kind = "regression"
    tags = ("robust", "outlier")
    param_schema = [
        ParamSpec("epsilon", "δ 阈值", "number", 1.35, min=1.0),
        ParamSpec("alpha", "正则强度", "number", 0.0001, min=0),
    ]

    def make(self, p, kind):
        from sklearn.linear_model import HuberRegressor
        return HuberRegressor(epsilon=float(p["epsilon"]),
                              alpha=float(p["alpha"]), max_iter=500)


@register
class TheilSen(SklearnSupervised):
    name = "theilsen"
    display_name = "Theil-Sen 稳健回归"
    family = "linear"
    target_kind = "regression"
    tags = ("robust", "outlier")
    param_schema = [
        ParamSpec("max_iter", "子采样轮数", "int", 300, min=10),
    ]

    def make(self, p, kind):
        from sklearn.linear_model import TheilSenRegressor
        return TheilSenRegressor(max_iter=int(p["max_iter"]), random_state=42)


@register
class LogisticRegression(SklearnSupervised):
    name = "logistic"
    display_name = "逻辑回归"
    family = "linear"
    target_kind = "classification"
    tags = ("baseline", "interpretable", "probability")
    param_schema = [
        ParamSpec("C", "正则强度 C", "number", 1.0, min=1e-4),
        ParamSpec("penalty", "惩罚", "select", "l2", choices=["l1", "l2", "elasticnet", "none"]),
        ParamSpec("class_weight", "类别加权", "select", "none",
                  choices=["none", "balanced"], hint="不平衡样本设 balanced"),
    ]

    def make(self, p, kind):
        from sklearn.linear_model import LogisticRegression
        kw = dict(C=float(p["C"]), max_iter=2000, random_state=42)
        pen = p["penalty"]
        if pen == "none":
            kw["penalty"] = None
        elif pen == "elasticnet":
            kw.update(penalty="elasticnet", l1_ratio=0.5, solver="saga")
        else:
            kw["penalty"] = pen
            if pen == "l1":
                kw["solver"] = "liblinear"
        if p["class_weight"] == "balanced":
            kw["class_weight"] = "balanced"
        return LogisticRegression(**kw)


@register
class LDA(SklearnSupervised):
    name = "lda"
    display_name = "线性判别分析"
    family = "linear"
    target_kind = "classification"
    tags = ("generative", "dimension")
    param_schema = [
        ParamSpec("solver", "求解器", "select", "svd",
                  choices=["svd", "lsqr", "eigen"]),
    ]

    def make(self, p, kind):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis(solver=p["solver"])


@register
class QDA(SklearnSupervised):
    name = "qda"
    display_name = "二次判别分析"
    family = "linear"
    target_kind = "classification"
    tags = ("generative",)
    param_schema = [
        ParamSpec("reg_factor", "协方差正则", "number", 0.05, min=0, max=1,
                  hint="小样本时 >0 防奇异（sklearn 要求正数）"),
    ]

    def make(self, p, kind):
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        return QuadraticDiscriminantAnalysis(reg_param=max(float(p["reg_factor"]), 1e-4))


@register
class KernelRidge(SklearnSupervised):
    name = "kernel_ridge"
    display_name = "核岭回归"
    family = "linear"
    target_kind = "regression"
    tags = ("kernel", "smooth", "small-sample")
    param_schema = [
        ParamSpec("alpha", "正则 α", "number", 1.0, min=1e-6),
        ParamSpec("kernel", "核函数", "select", "rbf",
                  choices=["linear", "poly", "rbf", "cosine"]),
        ParamSpec("gamma", "核宽度 γ", "select", "scale",
                  choices=["scale", "auto"]),
    ]

    def make(self, p, kind):
        from sklearn.kernel_ridge import KernelRidge as _KR
        g = p["gamma"]
        gamma = None if g == "scale" else ("auto" if g == "auto" else float(g))
        return _KR(alpha=float(p["alpha"]), kernel=p["kernel"], gamma=gamma)


@register
class GaussianProcess(SklearnSupervised):
    name = "gpr"
    display_name = "高斯过程回归"
    family = "linear"
    target_kind = "regression"
    tags = ("bayesian", "uncertainty", "surrogate", "small-sample")
    param_schema = [
        ParamSpec("length_scale", "核长度尺度", "number", 1.0, min=0.01,
                  hint="RBF 核；>0 越小越灵活"),
        ParamSpec("noise", "噪声水平", "number", 0.1, min=1e-6,
                  hint="白噪声 alpha"),
    ]

    def make(self, p, kind):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        kern = RBF(length_scale=float(p["length_scale"])) \
            + WhiteKernel(noise_level=float(p["noise"]))
        return GaussianProcessRegressor(kernel=kern, normalize_y=True,
                                        random_state=42)
