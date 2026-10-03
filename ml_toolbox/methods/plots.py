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
    """残差图：预测值 vs 残差 + 零线 + ±1σ 带。

    ⚠ **已迁移**：数由 `specs.spec_residual` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    plot_mpl.draw(_specs.spec_residual(result), ax)


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
    """PR 曲线（二分类）。

    ⚠ **已迁移**：数由 `specs.spec_pr` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    plot_mpl.draw(_specs.spec_pr(result), ax)


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
    """2D 嵌入散点，按标签逐点着色。

    ⚠ **已迁移**：数由 `specs.spec_scatter_emb` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    plot_mpl.draw(_specs.spec_scatter_emb(result), ax)


def plot_silhouette(ax, result):
    """轮廓系数：每簇一条面积带 + 均值竖线。

    ⚠ **已迁移**：数由 `specs.spec_silhouette` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    plot_mpl.draw(_specs.spec_silhouette(result), ax)


def plot_explained_variance(ax, result):
    """方差解释：条形 + 累计曲线。

    ⚠ **已迁移**：数由 `specs.spec_explained_variance` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    plot_mpl.draw(_specs.spec_explained_variance(result), ax)


def plot_anomaly_score(ax, result):
    """异常分数直方图 + 阈值竖线。

    ⚠ **已迁移**：数由 `specs.spec_anomaly_score` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    plot_mpl.draw(_specs.spec_anomaly_score(result), ax)


def plot_forecast(ax, result):
    """时序预测：实际 / 预测 + 95% 区间带。

    ⚠ **已迁移**：数由 `specs.spec_forecast` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    plot_mpl.draw(_specs.spec_forecast(result), ax)


def plot_ts_residual(ax, result):
    """时序残差折线 + 零线。

    ⚠ **已迁移**：数由 `specs.spec_ts_residual` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    plot_mpl.draw(_specs.spec_ts_residual(result), ax)


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
