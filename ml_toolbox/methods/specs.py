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

- **ML 侧 14 个页 key 全在这张表里**（见下 `SPEC_BY_KEY`）；
- 优化侧另有一张同款表（`opt/specs.SPEC_BY_KEY`，5 个 key）；
- **还没迁的只有量子启发方法的 3 个页**（`qk` 量子核矩阵 / `qdist` 保真度距离 /
  `mc` 记忆容量，住在 `methods/quantum.py`，仍在函数里直接调 `ax`）
  ⇒ 这 3 页在 web 端走服务端 PNG；
- ⚠ **`surrogate`（代理切片）不在上面任何一张表里，而且不该在里面**：
  它由优化器实例在 `bo.py::inspect_pages` 现造，图要那个实例的 GP 后验，
  不是"只吃存档"的自足页 ⇒ 面板显式把它列为**只能走 PNG**（见
  `ml_mecha/panel_web.py::_PNG_ONLY_PAGES` 与被 `test_plot_spec.py` 钉住的那条判据）。
"""

from __future__ import annotations

import numpy as np

from ..core.plot_spec import PlotSpec, Series

#: **页 key → spec 函数**：面板据此把该页渲染成 SVG（未登记的在 web 端仍走 PNG）。
#: ⚠ 这张表就是"迁移进度"的单一真源：迁好一个 `spec_xxx` 就在这里加一行，
#: 两个渲染器与面板**都不用改**（面板只查表）。
SPEC_BY_KEY = {
    "fit": "spec_fit_1d", "resid": "spec_residual", "imp": "spec_importance",
    "cm": "spec_confusion", "roc": "spec_roc", "pr": "spec_pr",
    "emb": "spec_scatter_emb", "sil": "spec_silhouette",
    "evr": "spec_explained_variance", "score": "spec_anomaly_score",
    "fc": "spec_forecast", "res": "spec_ts_residual",
    "nn_dyn": "spec_nn_dynamics", "nn_attr": "spec_feature_attribution",
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
        return _note("这一页需要概率输出（`y_prob`）——该 run 没产生它。")
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
        return _note("ROC 需要二分类概率（这一档的概率矩阵与该页不匹配）。")
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

# ---------------------------------------------------------------- 调色板（与渲染器一致）
RED, BLUE, GREEN = "#cf222e", "#0969da", "#1a7f37"
PURPLE, ORANGE, GRAY = "#8250df", "#bc4c00", "#57606a"

#: tab10 的精确色值（**不经过 matplotlib 的 colormap 调用**——那是 GUI 侧的事；
#: 这里只需要"按标签分配离散色"这一件事，两个渲染器都能照用）。
TAB10 = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")


def _note(msg: str) -> PlotSpec:
    """**没有可画的东西时，在图上写明原因**（而不是给一张空图）。

    ⚠ 这条纪律与"面板不空白"同源：空图会让人以为"跑出来就是这样"，
    而实际是"这一页不适用于这一档"。
    """
    return PlotSpec(kind="barh", rows=[msg], values=[0.0], rowcolors=["#8b949e"],
                    legend=False, grid=False).axes(title="这一页没有可画的数据")


def _discrete(n: int) -> list:
    """给 n 个类别分色（超过 10 个就循环）。"""
    return [TAB10[i % len(TAB10)] for i in range(max(0, int(n)))]


def _labels_colors(labels):
    """整数标签 → 逐点颜色数组（按出现顺序分配离散色）。"""
    uniq = list(dict.fromkeys(np.asarray(labels).ravel().tolist()))
    lut = {v: TAB10[i % len(TAB10)] for i, v in enumerate(uniq)}
    return [lut[v] for v in np.asarray(labels).ravel().tolist()]


# ---------------------------------------------------------------- 监督·残差 / 分类
def spec_residual(result) -> PlotSpec:
    """残差图：预测值 vs 残差 + 零线 + ±1σ 带。"""
    a = result.artifacts
    yt = np.asarray(a["y_true"], float)
    yp = np.asarray(a["y_pred"], float)
    resid = yt - yp
    sd = float(np.nanstd(resid))
    series = [Series(name="样本", x=_l(yp), y=_l(resid), color=BLUE,
                     connect=False, marker="o")]
    if sd > 0:
        # 带用一条"空线 + fill_lo/hi"表达（渲染器把它画成区域，且不进图例时另给名字）
        series.append(Series(name="±1σ", x=_l(yp), y=_l(resid),
                             fill_lo=[-sd] * len(resid), fill_hi=[sd] * len(resid),
                             color=ORANGE, connect=False, no_legend=True))
    return PlotSpec(kind="line", x=_l(yp), series=series,
                    hlines=[{"y": 0.0, "color": "#1f2328"}],
                    ).axes(xlabel="预测值", ylabel="残差")


def spec_pr(result) -> PlotSpec:
    """PR 曲线（二分类）。"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    a = result.artifacts
    prob = a.get("y_prob")
    if prob is None:
        return _note("这一页需要概率输出（`y_prob`）——该 run 没产生它。")
    prob = np.asarray(prob, float)
    yt = np.asarray(a["y_true"])
    classes = np.unique(yt)
    if len(classes) != 2:
        # ⚠ 原实现是静默返回（空图）。**读不到要说原因**：多分类的 PR 曲线要每类一条，
        #   这里没做 ⇒ 明写"这一页只对二分类"，而不是给一张空图。
        return _note(f"PR 曲线这一页目前只对**二分类**（这一档有 {len(classes)} 类）。")
    p = prob[:, 1] if prob.ndim == 2 else prob
    prec, rec, _ = precision_recall_curve(yt, p, pos_label=classes[1])
    ap = average_precision_score(yt, p)
    return PlotSpec(kind="line", series=[
        Series(name=f"AP={ap:.3f}", x=_l(rec), y=_l(prec), color=BLUE),
    ]).axes(xlabel="Recall", ylabel="Precision")


# ---------------------------------------------------------------- 聚类 / 降维
def spec_scatter_emb(result) -> PlotSpec:
    """2D 嵌入散点（聚类/降维共用），**按标签逐点着色**。"""
    a = result.artifacts
    src = a.get("embedding")
    if src is None:
        src = a.get("scores")
    emb = np.asarray(src, float)
    if emb.ndim != 2 or emb.shape[1] < 2:
        return PlotSpec(kind="scatter")
    labels = a.get("labels")
    if labels is not None:
        colors = _labels_colors(labels)
    else:
        colors = BLUE
    return PlotSpec(kind="scatter", x=_l(emb[:, 0]), series=[
        Series(y=_l(emb[:, 1]), color=colors, no_legend=True),
    ]).axes(xlabel="dim 1", ylabel="dim 2").no_legend()


def spec_silhouette(result) -> PlotSpec:
    """轮廓系数：每个簇一条**面积**条带 + 均值竖线。"""
    from sklearn.metrics import silhouette_samples, silhouette_score
    a = result.artifacts
    X = a.get("X_scaled")
    labels = np.asarray(a.get("labels", []))
    if X is None or len(labels) < 2 or len(np.unique(labels)) < 2:
        return PlotSpec(kind="line")
    X = np.asarray(X, float)
    s_vals = silhouette_samples(X, labels)
    series = []
    y_lower = 10
    colors = _discrete(len(np.unique(labels)))
    for k, i in enumerate(np.unique(labels)):
        vals = np.sort(s_vals[labels == i])
        y_upper = y_lower + len(vals)
        ys = np.arange(y_lower, y_upper)
        series.append(Series(name=f"{i}", x=_l(vals), y=_l(ys), color=colors[k],
                             area=True, fill_base=0.0, connect=True, width=0.8))
        y_lower = y_upper + 10
    mean = float(silhouette_score(X, labels))
    return PlotSpec(kind="line", series=series,
                    vlines=[{"x": mean, "color": RED, "label": "均值"}],
                    ).axes(xlabel="silhouette value", ylabel="cluster")


def spec_explained_variance(result) -> PlotSpec:
    """方差解释：逐主成分条形 + 累计曲线。"""
    evr = result.artifacts.get("explained_variance_ratio")
    if evr is None:
        return PlotSpec(kind="bar")
    evr = np.asarray(evr, float)
    xs = list(range(1, len(evr) + 1))
    return PlotSpec(kind="bar", x=xs, series=[
        Series(name="", y=_l(evr), color=PURPLE, as_bar=True, no_legend=True),
        Series(name="累计", y=_l(np.cumsum(evr)), color="#1f2328", marker="o",
               width=1.0),
    ]).axes(xlabel="component", ylabel="variance ratio")


# ---------------------------------------------------------------- 异常 / 时序
def spec_anomaly_score(result) -> PlotSpec:
    """异常分数直方图：全体 + 判为异常的一组 + 阈值竖线。"""
    a = result.artifacts
    s = np.asarray(a["scores"], float)
    labels = np.asarray(a.get("labels", np.zeros_like(s)))
    series = []
    if (labels == 1).any():
        series.append(Series(name="判为异常", y=_l(s[labels == 1]), color=RED,
                             alpha=0.8))
    thr = a.get("threshold")
    vlines = [{"x": float(thr), "color": "#1f2328", "label": f"阈值={float(thr):.3f}"}]         if thr is not None else []
    return PlotSpec(kind="hist", x=_l(s), series=series, vlines=vlines,
                    ).axes(xlabel="anomaly score")


def spec_ts_residual(result) -> PlotSpec:
    """时序残差：实际 − 预测 随 t 的折线 + 零线。"""
    a = result.artifacts
    y = np.asarray(a["y_true"], float)
    f = np.asarray(a["forecast"], float)
    n = min(len(y), len(f))
    resid = (y[:n] - f[:n])
    return PlotSpec(kind="line", x=list(range(n)), series=[
        Series(y=_l(resid), color=BLUE, width=1.0, no_legend=True),
    ], hlines=[{"y": 0.0, "color": "#1f2328"}],
    ).axes(xlabel="t", ylabel="residual").no_legend()


# ---------------------------------------------------------------- 神经网络
def spec_nn_dynamics(result) -> PlotSpec:
    """训练动态：loss/val_loss 主轴，lr/grad_norm 走**副轴（log）**。

    ⚠ 原 GUI 用 `ax.twinx()` + `set_yscale("log")`；契约里表达为
    `Series.axis2=True` + `Spec.yscale2="log"`（副轴按自身数据缩放）。
    """
    h = result.artifacts.get("nn_history")
    if h is None:
        return PlotSpec(kind="line")
    ep = _l(h["epoch"])
    series = [Series(name="train loss", x=ep, y=_l(h["loss"]), color=BLUE)]
    if np.isfinite(np.asarray(h["val_loss"], float)).any():
        series.append(Series(name="val loss", x=ep, y=_l(h["val_loss"]),
                             color=RED, dash=True))
    series.append(Series(name="lr", x=ep, y=_l(h["lr"]), color=GREEN,
                         dash=True, width=1.0, axis2=True))
    series.append(Series(name="grad norm", x=ep, y=_l(h["grad_norm"]),
                         color=PURPLE, dash=True, width=1.0, axis2=True))
    return PlotSpec(kind="line", series=series, yscale2="log",
                    ylabel2="lr / grad norm（副轴 · log）",
                    ).axes(xlabel="epoch", ylabel="loss", title="训练动态")


def spec_feature_attribution(result) -> PlotSpec:
    """积分梯度归因：Top-20 横向条形，**正负逐条着色**。"""
    attr = result.artifacts.get("nn_attr")
    if attr is None:
        return PlotSpec(kind="barh")
    attr = np.atleast_2d(np.asarray(attr, float))
    mean = attr.mean(axis=0)
    names = result.artifacts.get("nn_feat_names")
    idx = np.argsort(np.abs(mean))[-20:]
    rows = [str(names[i])[:22] if names is not None else f"f{i}" for i in idx]
    colors = [RED if mean[i] >= 0 else BLUE for i in idx]
    return PlotSpec(kind="barh", rows=rows, values=_l(mean[idx]),
                    rowcolors=colors,
                    hlines=[], vlines=[{"x": 0.0, "color": "#1f2328"}],
                    ).axes(xlabel="平均积分梯度归因（红=推高预测，蓝=压低）",
                           title="特征归因（Top 20，%d 个探针样本平均）" % len(attr)
                           ).no_legend()
