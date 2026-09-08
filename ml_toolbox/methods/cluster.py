# -*- coding: utf-8 -*-
"""聚类族：划分 / 层次 / 密度 / 谱。核心图 = 2D 嵌入散点 + 轮廓系数。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.contracts import (MLMethod, RunConfig, PageSpec, ParamSpec,
                              TASK_CLUSTER)
from ..core.registry import register
from . import plots


class ClusterMethod(MLMethod):
    task = TASK_CLUSTER
    family = "cluster"
    primary = "silhouette"

    def clusterer(self, p: dict):
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y, cfg: RunConfig, diag: bool = False):
        p = self.params(cfg)
        est = self.clusterer(p)
        Xv = X.to_numpy(float)
        labels = np.asarray(est.fit_predict(Xv) if hasattr(est, "fit_predict")
                            else est.fit(Xv).labels_)
        labels = np.asarray(labels).ravel()
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        res = self._new_result(target_kind=None, params=p)
        emb = _embed2d(Xv)
        res.artifacts = {"labels": labels, "embedding": emb,
                         "X_scaled": Xv, "n_clusters": n_clusters}
        if n_clusters >= 2:
            from sklearn.metrics import silhouette_score
            mask = labels >= 0
            if mask.sum() > n_clusters:
                res.metrics["silhouette"] = float(silhouette_score(
                    Xv[mask], labels[mask])) if len(set(labels[mask])) > 1 \
                    else float("nan")
        res.metrics["n_clusters"] = float(n_clusters)
        if diag:
            res.diag = {"n_samples": int(len(Xv)), "n_features": int(Xv.shape[1]),
                        "cluster_sizes": pd.Series(labels).value_counts()
                        .to_dict()}
        return res

    def inspect_pages(self, cfg):
        return [PageSpec("emb", "聚类散点", "mpl", plots.plot_scatter_emb),
                PageSpec("emb_pg", "交互缩放", "pg",
                         hint="pyqtgraph 滚轮缩放/拖拽平移，大点量不卡"),
                PageSpec("sil", "轮廓系数", "mpl", plots.plot_silhouette,
                         hint="需要缩放后的数值特征（开启数值缩放步骤效果最佳）")]


def _embed2d(Xv):
    from sklearn.decomposition import PCA
    try:
        return PCA(n_components=2, random_state=42).fit_transform(Xv)
    except Exception:
        return np.column_stack([Xv[:, 0], Xv[:, 1] if Xv.shape[1] > 1 else Xv[:, 0]])


@register
class KMeans(ClusterMethod):
    name = "kmeans"
    display_name = "K-Means"
    tags = ("partition", "baseline")
    param_schema = [
        ParamSpec("n_clusters", "簇数 K", "int", 3, min=2, max=50),
        ParamSpec("n_init", "重启次数", "int", 10, min=1),
    ]

    def clusterer(self, p):
        from sklearn.cluster import KMeans as _K
        return _K(n_clusters=int(p["n_clusters"]), n_init=int(p["n_init"]),
                  random_state=42)


@register
class GMM(ClusterMethod):
    name = "gmm"
    display_name = "高斯混合模型"
    tags = ("probabilistic", "soft-assignment")
    param_schema = [
        ParamSpec("n_components", "成分数", "int", 3, min=2, max=50),
        ParamSpec("covariance", "协方差类型", "select", "full",
                  choices=["full", "tied", "diag", "spherical"]),
    ]

    def clusterer(self, p):
        from sklearn.mixture import GaussianMixture
        return GaussianMixture(n_components=int(p["n_components"]),
                               covariance_type=p["covariance"],
                               random_state=42)

    def fit(self, X, y, cfg, diag=False):
        p = self.params(cfg)
        est = self.clusterer(p)
        Xv = X.to_numpy(float)
        labels = est.fit_predict(Xv)
        res = self._new_result(target_kind=None, params=p)
        emb = _embed2d(Xv)
        res.artifacts = {"labels": np.asarray(labels).ravel(),
                         "embedding": emb, "X_scaled": Xv,
                         "probs": est.predict_proba(Xv),
                         "n_clusters": len(set(labels))}
        if len(set(labels)) >= 2:
            from sklearn.metrics import silhouette_score
            res.metrics["silhouette"] = float(silhouette_score(Xv, labels))
        res.metrics["n_clusters"] = float(len(set(labels)))
        if diag:
            res.diag = {"n_samples": int(len(Xv)),
                        "bic": float(est.bic(Xv)),
                        "converged": bool(est.converged_)}
        return res


@register
class DBSCAN(ClusterMethod):
    name = "dbscan"
    display_name = "DBSCAN 密度聚类"
    tags = ("density", "noise", "auto-k")
    param_schema = [
        ParamSpec("eps", "邻域半径 ε", "number", 0.5, min=0.05,
                  hint="缩放后数据常用 0.3~1.0"),
        ParamSpec("min_samples", "核心点最小邻居", "int", 5, min=2),
    ]

    def clusterer(self, p):
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=float(p["eps"]), min_samples=int(p["min_samples"]))


@register
class HDBSCAN(ClusterMethod):
    name = "hdbscan"
    display_name = "HDBSCAN"
    tags = ("density", "hierarchical", "auto-k")
    param_schema = [
        ParamSpec("min_cluster_size", "最小簇规模", "int", 15, min=2),
    ]

    def clusterer(self, p):
        from sklearn.cluster import HDBSCAN as _H
        return _H(min_cluster_size=int(p["min_cluster_size"]))


@register
class Agglomerative(ClusterMethod):
    name = "agglomerative"
    display_name = "层次聚类"
    tags = ("hierarchical")
    param_schema = [
        ParamSpec("n_clusters", "簇数", "int", 3, min=2, max=50),
        ParamSpec("linkage", "连接方式", "select", "ward",
                  choices=["ward", "complete", "average", "single"]),
    ]

    def clusterer(self, p):
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(n_clusters=int(p["n_clusters"]),
                                       linkage=p["linkage"])


@register
class Spectral(ClusterMethod):
    name = "spectral"
    display_name = "谱聚类"
    tags = ("graph", "nonconvex")
    param_schema = [
        ParamSpec("n_clusters", "簇数", "int", 3, min=2, max=30),
        ParamSpec("affinity", "相似度", "select", "rbf",
                  choices=["rbf", "nearest_neighbors"]),
    ]

    def clusterer(self, p):
        from sklearn.cluster import SpectralClustering
        return SpectralClustering(n_clusters=int(p["n_clusters"]),
                                  affinity=p["affinity"], random_state=42,
                                  assign_labels="kmeans")
