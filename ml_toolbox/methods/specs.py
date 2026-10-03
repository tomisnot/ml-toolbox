# -*- coding: utf-8 -*-
"""方法层共享的**绘图规格函数**：``spec_xxx(result) -> PlotSpec``（纯函数，零渲染依赖）。

## 与 `plots.py` 的分工（用户裁决后的分层）

```
方法层  ──spec_xxx(result)──▶  PlotSpec  ──┬─▶ plot_mpl.draw(spec, ax)   GUI
                                            └─▶ panel_svg.svg(spec)      web
```

- 原先 ``plot_xxx(ax, result)`` 里**方法自己调 matplotlib** ⇒ 数据与图分不开、换不了渲染器；
- 现在**数只在这里取**，画法归渲染器。⚠ `plots.py` 里的旧函数保留为**薄适配器**
  （取 spec → 交给 matplotlib 渲染器），GUI 的调用点因此不用改。

## 迁移进度（如实标注，避免"看起来全迁完了"）

已迁移：`fit_1d` · `importance` · `confusion` · `roc` · `forecast`。
其余（`residual` · `pr` · `scatter_emb` · `silhouette` · `explained_variance` ·
`anomaly_score` · `ts_residual` · `nn_dynamics` · `feature_attribution` · opt 那 5 个）
**仍在 `plots.py` 里直接画**，逐批迁移；未迁移的页在 web 端仍走 PNG。
"""

from __future__ import annotations

import numpy as np

from ..core.plot_spec import PlotSpec, Series

#: **页 key → spec 函数**：面板据此把该页渲染成 SVG（未登记的在 web 端仍走 PNG）。
#: ⚠ 这张表就是"迁移进度"的单一真源：迁好一个 `spec_xxx` 就在这里加一行，
#: 两个渲染器与面板**都不用改**（面板只查表）。
SPEC_BY_KEY = {
    "fit": "spec_fit_1d",
    "imp": "spec_importance",
    "cm": "spec_confusion",
    "roc": "spec_roc",
    "fc": "spec_forecast",
}


def spec_for_key(key: str):
    """按页 key 取 spec 函数；未迁移 ⇒ ``None``（调用方回退到 PNG 路径）。"""
    name = SPEC_BY_KEY.get(str(key or ""))
    return globals().get(name) if name else None

#: 与渲染器同一支红/蓝（两处配色一致由判据核对）。
RED = "#cf222e"
BLUE = "#0969da"


def _l(seq):
    """numpy → list（PlotSpec 只认可序列化的 list；``None`` 保留为断线）。"""
    return [None if v is None or (isinstance(v, float) and v != v) else float(v)
            for v in np.asarray(seq, dtype=object).ravel().tolist()] \
        if seq is not None else []


# ---------------------------------------------------------------- 监督·回归
def spec_fit_1d(result) -> PlotSpec:
    """一维回归：散点 + 拟合曲线（按 x 排序连线）+ 理想 y=x。

    ⚠ 三条内容合成**一个** ``PlotSpec``：散点用 ``connect=False``，
    拟合线用 ``marker="o"``，理想线用 ``no_legend``。原 GUI 是三次 ``ax`` 调用，
    契约里靠这三个开关表达同一件事（两个渲染器都按开关画）。
    """
    a = result.artifacts
    yt = np.asarray(a["y_true"], float)
    yp = np.asarray(a["y_pred"], float)
    order = np.argsort(yt)
    lo, hi = float(yt.min()), float(yt.max())
    return PlotSpec(
        kind="line",
        x=_l(yt),
        series=[
            Series(name="样本", y=_l(yp), color=BLUE, connect=False, marker="o",
                   alpha=0.55),
            Series(name="拟合", x=_l(yt[order]), y=_l(yp[order]), color=RED,
                   width=1.2),
            Series(name="", x=[lo, hi], y=[lo, hi], color="#57606a",
                   dash=True, width=0.8, no_legend=True),
        ],
    ).axes(xlabel="真实值", ylabel="预测值")


def spec_importance(result) -> PlotSpec:
    """特征重要性：横向条形（最多 20 条）。"""
    fi = result.artifacts.get("feature_importance")
    if fi is None or getattr(fi, "empty", True):
        return PlotSpec(kind="barh")
    s = fi["importance"].sort_values()
    top = s.tail(20)
    return PlotSpec(kind="barh",
                    rows=[str(i)[:28] for i in top.index],
                    values=_l(top.values),
                    title="").axes(xlabel="importance").no_legend()


# ---------------------------------------------------------------- 分类
def spec_confusion(result) -> PlotSpec:
    """混淆矩阵（每格标数字，**深底用白字**）。

    ⚠ 格子里数字的颜色**由 spec 决定**（浅色/深色底）：实测深蓝底上的黑字几乎看不清。
    渲染器只管照着画 —— 这正是"画法归渲染器、但可读性判断归数据侧"的分工。
    """
    from sklearn.metrics import confusion_matrix
    a = result.artifacts
    yt = np.asarray(a["y_true"])
    yp = np.asarray(a["y_pred"])
    labels = np.unique(np.concatenate([yt, yp]))
    cm = confusion_matrix(yt, yp, labels=labels)
    hi = float(cm.max()) if cm.size else 1.0
    thresh = hi / 2.0
    texts = [{"x": j, "y": i, "text": str(int(cm[i, j])),
              "color": "#ffffff" if float(cm[i, j]) > thresh else "#1f2328"}
             for i in range(cm.shape[0]) for j in range(cm.shape[1])]
    return PlotSpec(kind="heatmap",
                    rows=[str(x)[:10] for x in labels],
                    values=[[int(v) for v in row] for row in cm],
                    texts=texts, cmap="Blues").axes(xlabel="预测", ylabel="真实")


def spec_roc(result) -> PlotSpec:
    """ROC 曲线（二分类一条；多分类每类一条）+ y=x 对角参考线。"""
    from sklearn.metrics import roc_curve, auc
    a = result.artifacts
    prob = a.get("y_prob")
    if prob is None:
        return PlotSpec(kind="line")
    prob = np.asarray(prob, float)
    yt = np.asarray(a["y_true"])
    classes = np.unique(yt)
    series = []
    if len(classes) == 2:
        p = prob[:, 1] if prob.ndim == 2 else prob
        fpr, tpr, _ = roc_curve(yt, p, pos_label=classes[1])
        series.append(Series(name=f"AUC={auc(fpr, tpr):.3f}",
                             x=_l(fpr), y=_l(tpr), color=BLUE))
    elif prob.ndim == 2 and prob.shape[1] == len(classes):
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve((yt == c).astype(int), prob[:, i])
            series.append(Series(name=f"{str(c)[:10]} {auc(fpr, tpr):.2f}",
                                 x=_l(fpr), y=_l(tpr)))
    else:
        return PlotSpec(kind="line")
    series.append(Series(name="", x=[0.0, 1.0], y=[0.0, 1.0], color="#57606a",
                         dash=True, width=0.8, no_legend=True))
    return PlotSpec(kind="line", series=series).axes(xlabel="FPR", ylabel="TPR")


# ---------------------------------------------------------------- 时序
def spec_forecast(result) -> PlotSpec:
    """时序预测：实际 / 预测两条线 + 可选的 95% 区间带。"""
    a = result.artifacts
    t = _l(a["t_true"])
    y = _l(a["y_true"])
    f = _l(a["forecast"])
    series = [Series(name="历史/实际", x=t, y=y, color=BLUE, marker="o", width=1.0),
              Series(name="拟合/预测", x=t, y=f, color=RED, marker="s",
                     dash=True, width=1.0)]
    ci = a.get("ci")
    if ci is not None:
        c = np.asarray(ci, float)
        fn = np.asarray(f, float)
        s = series[1]
        s.fill_lo = _l(fn - c)
        s.fill_hi = _l(fn + c)
        s.name = "拟合/预测（±95%）"
    return PlotSpec(kind="line", series=series).axes(xlabel="t", ylabel="y")
