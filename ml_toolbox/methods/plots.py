# -*- coding: utf-8 -*-
"""方法族共享的 matplotlib 绘图函数（纯函数，零 Qt 依赖 —— P1）。

约定签名：plot(ax, result) -> None，从 result.artifacts 取数。
UI 层负责把 ax 放进画布；导出层可直接 savefig。
"""
from __future__ import annotations

import numpy as np

from ..core import plot_mpl
from . import specs as _specs


# ---------------------------------------------------------------- 监督·回归
def plot_fit_1d(ax, result):
    """一维回归：散点 + 拟合曲线 + 理想 y=x。

    ⚠ **已迁移**：数由 `specs.spec_fit_1d` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    plot_mpl.draw(_specs.spec_fit_1d(result), ax)


def plot_residual(ax, result):
    a = result.artifacts
    yt, yp = np.asarray(a["y_true"], float), np.asarray(a["y_pred"], float)
    resid = yt - yp
    ax.scatter(yp, resid, s=14, alpha=0.55, color="tab:blue")
    ax.axhline(0, color="k", lw=0.8)
    sd = float(np.nanstd(resid))
    if sd > 0:
        ax.axhspan(-sd, sd, color="tab:orange", alpha=0.12, label="±1σ")
    ax.set_xlabel("预测值"); ax.set_ylabel("残差")
    ax.legend(fontsize=16); ax.grid(alpha=0.25)


def plot_importance(ax, result):
    """特征重要性：横向条形（最多 20 条）。

    ⚠ **已迁移**：数由 `specs.spec_importance` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    plot_mpl.draw(_specs.spec_importance(result), ax)


def plot_confusion(ax, result):
    """混淆矩阵（每格标数字）。

    ⚠ **已迁移**：数由 `specs.spec_confusion` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    plot_mpl.draw(_specs.spec_confusion(result), ax)


def plot_roc(ax, result):
    """ROC 曲线（二分类一条 / 多分类每类一条）。

    ⚠ **已迁移**：数由 `specs.spec_roc` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    plot_mpl.draw(_specs.spec_roc(result), ax)


def plot_pr(ax, result):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    a = result.artifacts
    prob = a.get("y_prob")
    yt = np.asarray(a["y_true"])
    if prob is None:
        return
    prob = np.asarray(prob, float)
    classes = np.unique(yt)
    if len(classes) == 2:
        p = prob[:, 1] if prob.ndim == 2 else prob
        prec, rec, _ = precision_recall_curve(yt, p, pos_label=classes[1])
        ax.plot(rec, prec, lw=1.4,
                label=f"AP={average_precision_score(yt, p):.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=14); ax.grid(alpha=0.25)


# ---------------------------------------------------------------- 无监督
def _discrete_colors(labels):
    """整数标签 -> 离散 RGBA 颜色数组。
    注意：tab10 是 ListedColormap，用分数位置调用会插值出灰调——
    必须直接取 cmap.colors[i % 10] 的精确调色板色。"""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    uniq = np.unique(labels)
    if len(uniq) <= 20:
        cmap = cm.get_cmap("tab10" if len(uniq) <= 10 else "tab20")
        palette = list(cmap.colors)
        idx = {v: i for i, v in enumerate(uniq)}
        return mcolors.to_rgba_array(
            [palette[idx[v] % len(palette)] for v in labels])
    return labels   # 连续值走默认着色


def plot_scatter_emb(ax, result):
    """2D 嵌入散点（聚类/降维共用），按标签着色。"""
    a = result.artifacts
    emb = np.asarray(a.get("embedding", a.get("scores")), float)
    if emb.ndim != 2 or emb.shape[1] < 2:
        return
    labels = a.get("labels")
    if labels is not None:
        labels = np.asarray(labels)
        ax.scatter(emb[:, 0], emb[:, 1], c=_discrete_colors(labels),
                   s=12, alpha=0.8, edgecolors="none")
    else:
        ax.scatter(emb[:, 0], emb[:, 1], s=12, alpha=0.6, color="tab:blue")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.grid(alpha=0.25)


def plot_silhouette(ax, result):
    from sklearn.metrics import silhouette_samples, silhouette_score
    a = result.artifacts
    X = a.get("X_scaled")
    labels = np.asarray(a.get("labels", []))
    if X is None or len(labels) < 2 or len(np.unique(labels)) < 2:
        return
    s_vals = silhouette_samples(np.asarray(X, float), labels)
    y_lower = 10
    for i in np.unique(labels):
        vals = np.sort(s_vals[labels == i])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * len(vals), str(i), fontsize=14)
        y_lower = y_upper + 10
    ax.axvline(silhouette_score(np.asarray(X, float), labels),
               color="red", ls="--", lw=1, label="均值")
    ax.set_xlabel("silhouette value"); ax.set_ylabel("cluster")
    ax.legend(fontsize=14)


def plot_explained_variance(ax, result):
    evr = result.artifacts.get("explained_variance_ratio")
    if evr is None:
        return
    evr = np.asarray(evr, float)
    ax.bar(range(1, len(evr) + 1), evr, color="tab:purple", alpha=0.75)
    ax.plot(range(1, len(evr) + 1), np.cumsum(evr), "o-", color="k", lw=1,
            ms=3, label="累计")
    ax.set_xlabel("component"); ax.set_ylabel("variance ratio")
    ax.legend(fontsize=14); ax.grid(alpha=0.25, axis="y")


def plot_anomaly_score(ax, result):
    a = result.artifacts
    s = np.asarray(a["scores"], float)
    labels = np.asarray(a.get("labels", np.zeros_like(s)))
    ax.hist(s, bins=40, color="lightgray", label="全体")
    if (labels == 1).any():
        ax.hist(s[labels == 1], bins=40, color="tab:red", alpha=0.8,
                label="判为异常")
    thr = a.get("threshold")
    if thr is not None:
        ax.axvline(float(thr), color="k", ls="--", lw=1, label=f"阈值={float(thr):.3f}")
    ax.set_xlabel("anomaly score"); ax.legend(fontsize=14); ax.grid(alpha=0.25)


# ---------------------------------------------------------------- 时序
def plot_forecast(ax, result):
    """时序预测：实际 / 预测 + 95% 区间带。

    ⚠ **已迁移**：数由 `specs.spec_forecast` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    plot_mpl.draw(_specs.spec_forecast(result), ax)


def plot_ts_residual(ax, result):
    a = result.artifacts
    y, f = np.asarray(a["y_true"], float), np.asarray(a["forecast"], float)
    resid = y - f
    ax.plot(np.arange(len(resid)), resid, lw=1, color="tab:blue")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("t"); ax.set_ylabel("residual")
    ax.grid(alpha=0.25)


# ---------------------------------------------------------------- 兜底页（C3）
def auto_pages(method, spec) -> list:
    """方法未声明检视页时，按 task 给出兜底页（UI 永远有东西可显示）。

    原住 core/runner.py，因依赖本模块绘图函数而构成 core→methods 反向依赖
    （架构审视 M4）。现下沉到 methods 侧并经 registry 槽反向注册：
    core.runner.auto_pages 退化为查表薄壳，依赖方向恢复为 methods→core。
    """
    from ..core.contracts import (PageSpec, TASK_SUPERVISED, TASK_CLUSTER,
                                  TASK_MANIFOLD, TASK_ANOMALY, TASK_TIMESERIES)
    t = method.task
    if t == TASK_SUPERVISED:
        if spec.target_kind == "regression":
            return [PageSpec("fit", "拟合效果", "mpl", plot_fit_1d),
                    PageSpec("resid", "残差诊断", "mpl", plot_residual),
                    PageSpec("imp", "特征重要性", "mpl", plot_importance,
                             hint="该方法无系数/重要性输出")]
        return [PageSpec("cm", "混淆矩阵", "mpl", plot_confusion),
                PageSpec("roc", "ROC / PR", "mpl", plot_roc,
                         hint="该方法不输出概率（如 LinearSVC），无法画 ROC"),
                PageSpec("pr", "PR 曲线", "mpl", plot_pr,
                         hint="需要概率输出"),
                PageSpec("imp", "特征重要性", "mpl", plot_importance,
                         hint="该方法无系数/重要性输出")]
    if t == TASK_CLUSTER:
        return [PageSpec("emb", "聚类散点", "mpl", plot_scatter_emb),
                PageSpec("sil", "轮廓系数", "mpl", plot_silhouette)]
    if t == TASK_MANIFOLD:
        return [PageSpec("emb", "降维散点", "mpl", plot_scatter_emb),
                PageSpec("evr", "方差解释", "mpl", plot_explained_variance,
                         hint="仅 PCA/LDA 等线性方法有方差解释概念")]
    if t == TASK_ANOMALY:
        return [PageSpec("score", "异常分数", "mpl", plot_anomaly_score)]
    if t == TASK_TIMESERIES:
        return [PageSpec("fc", "预测曲线", "mpl", plot_forecast),
                PageSpec("res", "残差序列", "mpl", plot_ts_residual)]
    return []


def _register():
    """methods 包加载时把兜底页提供者注册进 core.registry 的槽（C3）。"""
    from ..core import registry
    registry.register_page_provider(auto_pages)


_register()
