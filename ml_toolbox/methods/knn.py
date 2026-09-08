# -*- coding: utf-8 -*-
"""KNN / 近邻族。"""
from __future__ import annotations

from ..core.contracts import ParamSpec
from ..core.registry import register
from .base import SklearnSupervised


@register
class KNN(SklearnSupervised):
    name = "knn"
    display_name = "K 近邻"
    family = "knn"
    target_kind = None
    tags = ("lazy", "nonparametric")
    param_schema = [
        ParamSpec("n_neighbors", "K 值", "int", 5, min=1, max=100),
        ParamSpec("weights", "加权", "select", "uniform",
                  choices=["uniform", "distance"]),
        ParamSpec("metric", "距离", "select", "euclidean",
                  choices=["euclidean", "manhattan", "chebyshev", "minkowski"]),
        ParamSpec("leaf_size", "叶大小", "int", 30, min=1),
    ]

    def make(self, p, kind):
        from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
        kw = dict(n_neighbors=int(p["n_neighbors"]), weights=p["weights"],
                  metric=p["metric"], leaf_size=int(p["leaf_size"]))
        return (KNeighborsClassifier(**kw) if kind == "classification"
                else KNeighborsRegressor(**kw))
