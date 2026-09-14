# -*- coding: utf-8 -*-
"""异常检测族。核心图 = 分数分布 + 阈值线。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.contracts import (MLMethod, RunConfig, PageSpec, ParamSpec,
                              TASK_ANOMALY)
from ..core.parallel import nj
from ..core.registry import register
from . import plots


def _binary_truth(y):
    """把各种异常标签表示归一为 0/1 Series；无法识别返回 None。"""
    s = pd.Series(y)
    if s.dtype == bool:
        return s.astype(int)
    vals = set(pd.unique(s.dropna()))
    if vals <= {0, 1} or vals <= {0.0, 1.0}:
        return s.astype(int)
    lowered = s.astype(str).str.lower()
    if set(pd.unique(lowered)) & {"true", "anomaly", "异常", "outlier", "1"}:
        return lowered.isin(["1", "true", "anomaly", "异常", "outlier"]).astype(int)
    return None


class AnomalyMethod(MLMethod):
    task = TASK_ANOMALY
    family = "anomaly"

    def detector(self, p: dict):
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y, cfg: RunConfig, diag: bool = False):
        p = self.params(cfg)
        est = self.detector(p)
        Xv = X.to_numpy(float)
        labels = np.asarray(est.fit_predict(Xv)).ravel()   # sklearn: 1=正常, -1=异常
        # 统一分数方向：越大越异常
        if hasattr(est, "decision_scores_"):            # 越低越异常 -> 取负
            scores = -np.asarray(est.decision_scores_, float)
        elif hasattr(est, "negative_outlier_factor_"):  # 越高越正常 -> 取负
            scores = -np.asarray(est.negative_outlier_factor_, float)
        else:
            scores = -np.asarray(est.score_samples(Xv), float)
        labels = (labels == -1).astype(int)             # 1=异常
        res = self._new_result(target_kind=None, params=p)
        thr = float(np.quantile(scores, 1 - float(p.get("contamination", 0.1)))) \
            if "contamination" in p else None
        res.artifacts = {"scores": scores, "labels": labels,
                         "threshold": thr}
        res.metrics["n_anomalies"] = float(int(labels.sum()))
        res.metrics["contamination"] = float(labels.mean())
        # 有真实标签时算 AUC（标签兼容 0/1、bool、文本）
        if y is not None:
            truth = _binary_truth(y)
            if truth is not None and truth.nunique() > 1:
                from sklearn.metrics import roc_auc_score
                res.metrics["auc"] = float(roc_auc_score(truth, scores))
                res.primary_metric = "auc"
        if diag:
            res.diag = {"n_samples": int(len(Xv)),
                        "score_stats": {"min": float(scores.min()),
                                        "max": float(scores.max()),
                                        "mean": float(scores.mean())}}
        return res

    def inspect_pages(self, cfg):
        return [PageSpec("score", "异常分数", "mpl", plots.plot_anomaly_score)]


@register
class IsolationForest(AnomalyMethod):
    name = "iforest"
    display_name = "孤立森林"
    tags = ("tree", "baseline", "fast")
    param_schema = [
        ParamSpec("n_estimators", "树数量", "int", 200, min=10),
        ParamSpec("contamination", "异常比例", "number", 0.1, min=0.01, max=0.5),
        ParamSpec("max_samples", "子采样比例", "number", 0.8, min=0.1, max=1),
    ]

    def detector(self, p):
        from sklearn.ensemble import IsolationForest
        return IsolationForest(n_estimators=int(p["n_estimators"]),
                               contamination=float(p["contamination"]),
                               max_samples=float(p["max_samples"]),
                               random_state=42, n_jobs=nj())


@register
class LOF(AnomalyMethod):
    name = "lof"
    display_name = "局部离群因子"
    tags = ("density", "local")
    param_schema = [
        ParamSpec("n_neighbors", "邻域大小", "int", 20, min=5),
        ParamSpec("contamination", "异常比例", "number", 0.1, min=0.01, max=0.5),
    ]

    def detector(self, p):
        from sklearn.neighbors import LocalOutlierFactor
        return LocalOutlierFactor(n_neighbors=int(p["n_neighbors"]),
                                  contamination=float(p["contamination"]),
                                  novelty=False)


@register
class OneClassSVM(AnomalyMethod):
    name = "ocsvm"
    display_name = "单类 SVM"
    tags = ("kernel", "global")
    param_schema = [
        ParamSpec("nu", "异常比例上界 ν", "number", 0.1, min=0.01, max=0.5),
        ParamSpec("gamma", "核宽度 γ", "select", "scale",
                  choices=["scale", "auto"]),
    ]

    def detector(self, p):
        from sklearn.svm import OneClassSVM
        return OneClassSVM(nu=float(p["nu"]), gamma=p["gamma"])

    def fit(self, X, y, cfg, diag=False):
        # OneClassSVM 无 decision_scores_：用 score_samples
        res = super().fit(X, y, cfg, diag)
        return res


@register
class Mahalanobis(AnomalyMethod):
    name = "mahalanobis"
    display_name = "马氏距离检测"
    tags = ("statistical", "interpretable")
    param_schema = [
        ParamSpec("quantile", "分位数阈值", "number", 0.9, min=0.5, max=0.999,
                  hint="超过该分位数判异常"),
    ]

    def detector(self, p):
        raise NotImplementedError

    def fit(self, X, y, cfg, diag=False):
        p = self.params(cfg)
        Xv = X.to_numpy(float)
        mu = np.nanmean(Xv, axis=0)
        cov = np.cov(Xv.T) + np.eye(Xv.shape[1]) * 1e-6
        try:
            inv = np.linalg.inv(cov)
            d = np.sqrt(np.einsum("ij,jk,ik->i", Xv - mu, inv, Xv - mu))
        except np.linalg.LinAlgError:
            d = np.sqrt(np.einsum("ij,ij->i", Xv - mu, Xv - mu))
        thr = float(np.quantile(d, float(p["quantile"])))
        labels = (d > thr).astype(int)
        res = self._new_result(target_kind=None, params=p)
        res.artifacts = {"scores": d, "labels": labels, "threshold": thr}
        res.metrics["n_anomalies"] = float(labels.sum())
        res.metrics["contamination"] = float(labels.mean())
        if y is not None:
            truth = _binary_truth(y)
            if truth is not None and truth.nunique() > 1:
                from sklearn.metrics import roc_auc_score
                res.metrics["auc"] = float(roc_auc_score(truth, d))
                res.primary_metric = "auc"
        return res
