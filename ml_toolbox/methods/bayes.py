# -*- coding: utf-8 -*-
"""贝叶斯 / 生成式族。"""
from __future__ import annotations

import pandas as pd

from ..core.contracts import ParamSpec
from ..core.registry import register
from .base import SklearnSupervised


@register
class GaussianNB(SklearnSupervised):
    name = "gaussian_nb"
    display_name = "高斯朴素贝叶斯"
    family = "bayes"
    target_kind = "classification"
    tags = ("generative", "baseline", "fast")
    param_schema = [
        ParamSpec("var_smoothing", "平滑", "number", 1e-9, min=0),
    ]

    def make(self, p, kind):
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(var_smoothing=float(p["var_smoothing"]))


@register
class BernoulliNB(SklearnSupervised):
    name = "bernoulli_nb"
    display_name = "伯努利朴素贝叶斯"
    family = "bayes"
    target_kind = "classification"
    tags = ("generative", "binary-features")
    param_schema = [
        ParamSpec("alpha", "拉普拉斯 α", "number", 1.0, min=0),
        ParamSpec("binarize", "二值化阈值", "number", 0.0,
                  hint="特征 > 阈值记为 1；留空风格用 -inf 关闭"),
    ]

    def make(self, p, kind):
        from sklearn.naive_bayes import BernoulliNB
        return BernoulliNB(alpha=float(p["alpha"]),
                           binarize=float(p["binarize"]))


@register
class MultinomialNB(SklearnSupervised):
    name = "multinomial_nb"
    display_name = "多项式朴素贝叶斯"
    family = "bayes"
    target_kind = "classification"
    tags = ("generative", "count-features")
    param_schema = [
        ParamSpec("alpha", "拉普拉斯 α", "number", 1.0, min=0),
    ]

    def make(self, p, kind):
        from sklearn.naive_bayes import MultinomialNB
        return MultinomialNB(alpha=float(p["alpha"]))

    def fit(self, X, y, cfg, diag=False):
        # 多项式 NB 要求非负特征：管道缩放后可能为负，内部统一 MinMax 平移
        # （CV 复用实例，fit 时必须重拟 scaler）
        from sklearn.preprocessing import MinMaxScaler
        self._scaler = MinMaxScaler().fit(X.to_numpy(float))
        return super().fit(self._apply(X), y, cfg, diag)

    def predict(self, X, result):
        return super().predict(self._apply(X), result)

    def _apply(self, X):
        return pd.DataFrame(self._scaler.transform(X.to_numpy(float)),
                            columns=X.columns, index=X.index)


@register
class BayesianRidge(SklearnSupervised):
    name = "bayesian_ridge"
    display_name = "贝叶斯岭回归"
    family = "bayes"
    target_kind = "regression"
    tags = ("bayesian", "uncertainty", "regularized")
    param_schema = [
        ParamSpec("max_iter", "迭代次数", "int", 300, min=10),
        ParamSpec("alpha_1", "先验 α1", "number", 1e-6, min=0),
        ParamSpec("lambda_1", "先验 λ1", "number", 1e-6, min=0),
    ]

    def make(self, p, kind):
        from sklearn.linear_model import BayesianRidge
        return BayesianRidge(max_iter=int(p["max_iter"]),
                             alpha_1=float(p["alpha_1"]),
                             lambda_1=float(p["lambda_1"]))
