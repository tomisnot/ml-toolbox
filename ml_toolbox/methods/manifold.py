# -*- coding: utf-8 -*-
"""降维 / 流形族：线性 + 非线性。核心图 = 2D 嵌入散点（可选按 y 着色）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.contracts import (MLMethod, RunConfig, PageSpec, ParamSpec,
                              TASK_MANIFOLD)
from ..core.registry import register
from . import plots


class ManifoldMethod(MLMethod):
    task = TASK_MANIFOLD
    family = "manifold"

    def reducer(self, p: dict):
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y, cfg: RunConfig, diag: bool = False):
        p = self.params(cfg)
        est = self.reducer(p)
        Xv = X.to_numpy(float)
        emb = np.asarray(est.fit_transform(Xv), float)
        if emb.ndim == 1:
            emb = emb.reshape(-1, 1)
        res = self._new_result(target_kind=None, params=p)
        res.artifacts = {"embedding": emb}
        if hasattr(est, "explained_variance_ratio_"):
            evr = np.asarray(est.explained_variance_ratio_, float)
            res.artifacts["explained_variance_ratio"] = evr
            res.metrics["explained_variance"] = float(evr[:2].sum())
        else:
            res.metrics["explained_variance"] = float("nan")
        if y is not None:
            res.artifacts["labels"] = pd.factorize(np.asarray(y))[0]
        if diag:
            res.diag = {"n_in": int(Xv.shape[1]), "n_out": int(emb.shape[1])}
        return res

    def inspect_pages(self, cfg):
        return [PageSpec("emb", "降维散点", "mpl", plots.plot_scatter_emb),
                PageSpec("emb_pg", "交互缩放", "pg",
                         hint="pyqtgraph 滚轮缩放/拖拽平移，大点量不卡"),
                PageSpec("evr", "方差解释", "mpl", plots.plot_explained_variance,
                         hint="仅线性方法（PCA 族）产出方差解释；"
                              "t-SNE/UMAP 无此概念")]


@register
class PCA(ManifoldMethod):
    name = "pca"
    display_name = "主成分分析"
    tags = ("linear", "baseline")
    param_schema = [
        ParamSpec("n_components", "保留成分数", "int", 2, min=1,
                  hint="-1 = 按方差比例自动"),
        ParamSpec("var_threshold", "方差阈值(自动时)", "number", 0.95,
                  min=0.5, max=1.0),
        ParamSpec("whiten", "白化", "bool", False),
    ]

    def reducer(self, p):
        from sklearn.decomposition import PCA as _PCA
        nc = int(p["n_components"])
        return _PCA(n_components=nc if nc > 0 else float(p["var_threshold"]),
                    whiten=bool(p["whiten"]), random_state=42)


@register
class KernelPCA(ManifoldMethod):
    name = "kernel_pca"
    display_name = "核 PCA"
    tags = ("nonlinear", "kernel")
    param_schema = [
        ParamSpec("n_components", "成分数", "int", 2, min=1),
        ParamSpec("kernel", "核函数", "select", "rbf",
                  choices=["rbf", "poly", "sigmoid", "cosine"]),
        ParamSpec("gamma", "核宽度 γ", "select", "scale",
                  choices=["scale", "auto"]),
    ]

    def reducer(self, p):
        from sklearn.decomposition import KernelPCA
        g = p["gamma"]
        gamma = None if g == "scale" else ("auto" if g == "auto" else float(g))
        return KernelPCA(n_components=int(p["n_components"]),
                         kernel=p["kernel"], gamma=gamma,
                         random_state=42)


@register
class LDAProj(ManifoldMethod):
    name = "lda_proj"
    display_name = "LDA 投影（监督降维）"
    tags = ("linear", "supervised")
    param_schema = [
        ParamSpec("n_components", "成分数", "int", 2, min=1, max=10),
    ]

    def reducer(self, p):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis(n_components=int(p["n_components"]))

    def fit(self, X, y, cfg, diag=False):
        if y is None:
            res = self._new_result(target_kind=None, params=self.params(cfg))
            res.error = "LDA 投影需要标签列（在数据集里指定目标列）"
            return res
        p = self.params(cfg)
        est = self.reducer(p)
        Xv = X.to_numpy(float)
        emb = est.fit_transform(Xv, pd.factorize(np.asarray(y))[0])
        res = self._new_result(target_kind=None, params=p)
        res.artifacts = {"embedding": np.asarray(emb, float),
                         "labels": pd.factorize(np.asarray(y))[0]}
        evr = getattr(est, "explained_variance_ratio_", None)
        if evr is not None:
            res.artifacts["explained_variance_ratio"] = np.asarray(evr, float)
            res.metrics["explained_variance"] = float(np.asarray(evr)[:2].sum())
        return res


@register
class TSNE(ManifoldMethod):
    name = "tsne"
    display_name = "t-SNE"
    tags = ("nonlinear", "visualization", "slow-large")
    param_schema = [
        ParamSpec("perplexity", "邻域困惑度", "number", 30, min=5, max=50,
                  hint="须小于样本数"),
        ParamSpec("learning_rate", "学习率", "select", "auto",
                  choices=["auto"]),
        ParamSpec("init", "初始化", "select", "pca", choices=["pca", "random"]),
    ]

    def reducer(self, p):
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=float(p["perplexity"]),
                    init=p["init"], random_state=42, max_iter=1000)


@register
class UMAP(ManifoldMethod):
    name = "umap"
    display_name = "UMAP"
    tags = ("nonlinear", "visualization", "fast")
    param_schema = [
        ParamSpec("n_neighbors", "邻域大小", "int", 15, min=2, max=100,
                  hint="小=保局部结构，大=保全局结构"),
        ParamSpec("min_dist", "点间最小距离", "number", 0.1, min=0, max=1),
    ]

    def reducer(self, p):
        import umap
        return umap.UMAP(n_components=2, n_neighbors=int(p["n_neighbors"]),
                         min_dist=float(p["min_dist"]), random_state=42)


@register
class Isomap(ManifoldMethod):
    name = "isomap"
    display_name = "Isomap"
    tags = ("nonlinear", "geodesic")
    param_schema = [
        ParamSpec("n_neighbors", "邻域大小", "int", 10, min=2),
        ParamSpec("n_components", "输出维数", "int", 2, min=1),
    ]

    def reducer(self, p):
        from sklearn.manifold import Isomap
        return Isomap(n_neighbors=int(p["n_neighbors"]),
                      n_components=int(p["n_components"]))
