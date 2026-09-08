# -*- coding: utf-8 -*-
"""方法族共享的 matplotlib 绘图函数（纯函数，零 Qt 依赖 —— P1）。

约定签名：plot(ax, result) -> None，从 result.artifacts 取数。
UI 层负责把 ax 放进画布；导出层可直接 savefig。
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- 监督·回归
def plot_fit_1d(ax, result):
    """一维回归：散点 + 拟合曲线（按 x 排序连线）。"""
    a = result.artifacts
    yt, yp = np.asarray(a["y_true"], float), np.asarray(a["y_pred"], float)
    order = np.argsort(yt)
    ax.scatter(yt, yp, s=14, alpha=0.55, label="样本")
    ax.plot(yt[order], yp[order], lw=1.2, color="tab:red", label="拟合")
    lo, hi = yt.min(), yt.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="理想 y=x")
    ax.set_xlabel("真实值"); ax.set_ylabel("预测值")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)


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
    ax.legend(fontsize=8); ax.grid(alpha=0.25)


def plot_importance(ax, result):
    fi = result.artifacts.get("feature_importance")
    if fi is None or fi.empty:
        return
    s = fi["importance"].sort_values()
    top = s.tail(20)
    ax.barh(range(len(top)), top.values, color="tab:green", alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([str(i)[:28] for i in top.index], fontsize=7)
    ax.set_xlabel("importance"); ax.grid(alpha=0.25, axis="x")


# ---------------------------------------------------------------- 监督·分类
def plot_confusion(ax, result):
    from sklearn.metrics import confusion_matrix
    a = result.artifacts
    yt, yp = np.asarray(a["y_true"]), np.asarray(a["y_pred"])
    labels = np.unique(np.concatenate([yt, yp]))
    cm = confusion_matrix(yt, yp, labels=labels)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([str(l)[:10] for l in labels], fontsize=7)
    ax.set_yticklabels([str(l)[:10] for l in labels], fontsize=7)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("预测"); ax.set_ylabel("真实")


def plot_roc(ax, result):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    a = result.artifacts
    prob = a.get("y_prob")
    yt = np.asarray(a["y_true"])
    if prob is None:
        return
    prob = np.asarray(prob, float)
    classes = np.unique(yt)
    if len(classes) == 2:
        p = prob[:, 1] if prob.ndim == 2 else prob
        fpr, tpr, _ = roc_curve(yt, p, pos_label=classes[1])
        ax.plot(fpr, tpr, lw=1.4, label=f"AUC={auc(fpr, tpr):.3f}")
    elif prob.ndim == 2 and prob.shape[1] == len(classes):
        for i, c in enumerate(classes):
            yb = (yt == c).astype(int)
            fpr, tpr, _ = roc_curve(yb, prob[:, i])
            ax.plot(fpr, tpr, lw=1.1, label=f"{str(c)[:10]} {auc(fpr, tpr):.2f}")
    else:
        return
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)


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
    ax.legend(fontsize=7); ax.grid(alpha=0.25)


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
        ax.text(-0.05, y_lower + 0.5 * len(vals), str(i), fontsize=7)
        y_lower = y_upper + 10
    ax.axvline(silhouette_score(np.asarray(X, float), labels),
               color="red", ls="--", lw=1, label="均值")
    ax.set_xlabel("silhouette value"); ax.set_ylabel("cluster")
    ax.legend(fontsize=7)


def plot_explained_variance(ax, result):
    evr = result.artifacts.get("explained_variance_ratio")
    if evr is None:
        return
    evr = np.asarray(evr, float)
    ax.bar(range(1, len(evr) + 1), evr, color="tab:purple", alpha=0.75)
    ax.plot(range(1, len(evr) + 1), np.cumsum(evr), "o-", color="k", lw=1,
            ms=3, label="累计")
    ax.set_xlabel("component"); ax.set_ylabel("variance ratio")
    ax.legend(fontsize=7); ax.grid(alpha=0.25, axis="y")


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
    ax.set_xlabel("anomaly score"); ax.legend(fontsize=7); ax.grid(alpha=0.25)


# ---------------------------------------------------------------- 时序
def plot_forecast(ax, result):
    a = result.artifacts
    t = np.asarray(a["t_true"], float)
    y = np.asarray(a["y_true"], float)
    f = np.asarray(a["forecast"], float)
    ax.plot(t, y, "o-", ms=3, lw=1, color="tab:blue", label="历史/实际")
    ax.plot(t, f, "s--", ms=3, lw=1, color="tab:red", label="拟合/预测")
    ci = a.get("ci")
    if ci is not None:
        ci = np.asarray(ci, float)
        ax.fill_between(t, f - ci, f + ci, color="tab:red", alpha=0.12,
                        label="95% 区间")
    ax.set_xlabel("t"); ax.set_ylabel("y")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)


def plot_ts_residual(ax, result):
    a = result.artifacts
    y, f = np.asarray(a["y_true"], float), np.asarray(a["forecast"], float)
    resid = y - f
    ax.plot(np.arange(len(resid)), resid, lw=1, color="tab:blue")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("t"); ax.set_ylabel("residual")
    ax.grid(alpha=0.25)
