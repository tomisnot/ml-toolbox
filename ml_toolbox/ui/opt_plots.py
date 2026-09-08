# -*- coding: utf-8 -*-
"""优化检视绘图函数（纯 matplotlib，零 Qt）——对标 methods/plots.py 的角色。

消费 OptRecord.history（DataFrame：参数列 + score/status/ts/best_so_far），
对应优化工作区的检视页（docs/优化定位.md §7）：
    plot_convergence   —— best-so-far 阶梯 + 全部评估散点（可叠多条轨迹对比）
    plot_parallel      —— 参数×分数平行坐标（哪个参数敏感一眼可见）
    plot_scatter2d     —— 前两个连续参数的探索地图（颜色=分数，形状=状态）
    plot_surrogate_1d  —— GP 后验切片（需 optimizer 实例，仅 gp_bo 可用）
"""
from __future__ import annotations

import numpy as np

_GREEN = "#2ca02c"
_BLUE = "#1f77b4"


def _ok(record):
    return record.history[record.history["status"] == "ok"]


def plot_convergence(ax, record, extra=None):
    """收敛曲线：best-so-far 阶梯 + 评估散点。extra=[(label, history)] 叠加对比。"""
    h = record.history
    if h.empty:
        ax.text(0.5, 0.5, "尚无评估", ha="center", va="center", color="#888")
        return
    ok = _ok(record)
    ax.plot(np.arange(1, len(h) + 1), h["best_so_far"], "-o", ms=3,
            color=_GREEN, lw=1.6, label=f"best-so-far（{record.optimizer}）")
    if len(ok):
        ax.plot(np.arange(1, len(h) + 1), h["score"], ".", ms=4,
                color=_BLUE, alpha=0.5, label="单次评估")
    for label, hh in (extra or []):
        if len(hh):
            ax.plot(np.arange(1, len(hh) + 1), hh["best_so_far"], "--",
                    lw=1.2, alpha=0.85, label=label)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("评估次数")
    ax.set_ylabel("分数（log 邻域）")
    ax.set_title("收敛曲线（越小越好）")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)


def plot_parallel(ax, record):
    """平行坐标：每维一根竖轴，每条线=一次评估，颜色=分数（绿=好）。"""
    ok = _ok(record)
    keys = [d["key"] for d in record.space_desc]
    keys = [k for k in keys if k in ok.columns]
    if ok.empty or not keys:
        ax.text(0.5, 0.5, "无参数数据", ha="center", va="center", color="#888")
        return
    M = ok[keys].to_numpy(float)
    lo, hi = M.min(0), M.max(0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    Z = (M - lo) / span
    s = ok["score"].to_numpy(float)
    smin, smax = np.nanmin(s), np.nanmax(s)
    sn = (s - smin) / max(smax - smin, 1e-12)
    cmap = plt_cmap()
    xs = np.arange(len(keys))
    for i in range(len(Z)):
        ax.plot(xs, Z[i], color=cmap(1 - sn[i]), alpha=0.35, lw=0.8)
    best_i = int(np.argmin(s))
    ax.plot(xs, Z[best_i], color=_GREEN, lw=2.0, label="当前最优")
    for j, k in enumerate(keys):
        ax.axvline(j, color="#ccc", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(keys, fontsize=8, rotation=20)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["min", "", "max"], fontsize=8)
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_title("参数×分数平行坐标（绿=好，红=差）")
    ax.legend(fontsize=8)


def plt_cmap():
    import matplotlib.pyplot as plt
    return plt.get_cmap("RdYlGn")


def plot_scatter2d(ax, record):
    """探索地图：前两个连续参数为轴，颜色=分数，×=失败点。"""
    cont = [d["key"] for d in record.space_desc
            if d["kind"] in ("uniform", "log-uniform", "randint")]
    h = record.history
    if len(cont) < 2 or h.empty:
        ax.text(0.5, 0.5, "需要 ≥2 个连续参数（当前为 1D/类别空间，看平行坐标）",
                ha="center", va="center", color="#888", fontsize=9)
        return
    a, b = cont[0], cont[1]
    ok = _ok(record)
    fail = h[h["status"] == "failed"]
    sc = ax.scatter(ok[a], ok[b], c=ok["score"], cmap="viridis_r",
                    s=28, edgecolors="k", linewidths=0.3)
    if len(fail):
        ax.scatter(fail[a], fail[b], marker="x", s=40, color="firebrick",
                   label=f"失败 {len(fail)}")
    if len(ok):
        bx = ok.loc[ok["score"].idxmin()]
        ax.scatter([bx[a]], [bx[b]], marker="*", s=220, color=_GREEN,
                   edgecolors="k", zorder=5, label="最优")
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title("探索地图（前两个连续参数）")
    ax.legend(fontsize=8)
    try:
        import matplotlib.pyplot as plt
        plt.colorbar(sc, ax=ax, shrink=0.8).set_label("score", fontsize=8)
    except Exception:
        pass


def plot_surrogate_1d(ax, bundle):
    """GP 后验切片：bundle = (xs, mu, sd, obs_x, obs_y, label)。"""
    xs, mu, sd, obs_x, obs_y, label = bundle
    ax.plot(xs, mu, color=_BLUE, lw=1.6, label="GP 后验均值")
    ax.fill_between(xs, mu - 2 * sd, mu + 2 * sd, color=_BLUE, alpha=0.18,
                    label="±2σ")
    if len(obs_x):
        ax.plot(obs_x, obs_y, "k.", ms=7, label="已观测")
    ax.set_xlabel(label)
    ax.set_ylabel("score")
    ax.set_title("代理模型切片（固定其余参数于当前最优）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
