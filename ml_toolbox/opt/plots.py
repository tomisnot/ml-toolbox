# -*- coding: utf-8 -*-
"""优化检视绘图函数 + 默认页声明（opt 层，纯 matplotlib，零 Qt）。

C4：原住 ui/opt_plots.py，绘图知识与优化框架同层归置（对标 methods/plots.py
在 ML 侧的角色）。优化器经 inspect_pages(record) 声明 PageSpec 列表，
UI 只装配不预设——G2 纪律推广到优化侧，终结"UI 硬编码五 canvas + 逐页特判"。

消费 OptRecord.history（DataFrame：参数列 + score/status/ts/best_so_far）。
代理切片页需要 optimizer 实例（surrogate_1d），不属"record 自足页"，
由 UI 按 supports_surrogate 能力标志追加（见 default_pages 说明）。
"""
from __future__ import annotations

import numpy as np

from ..core.contracts import PageSpec

_GREEN = "#2ca02c"
_BLUE = "#1f77b4"


def _ok(record):
    return record.history[record.history["status"] == "ok"]


def plot_convergence(ax, record, extra=None):
    """收敛曲线：best-so-far 阶梯 + 评估散点。extra=[(label, history)] 叠加对比。

    extra 缺省时读 record._overlay（UI 直播时注入的已完成轨迹——呈现层
    上下文放 record 上，绘图函数保持自足，声明式装配不传第二参数）。
    """
    if extra is None:
        extra = getattr(record, "_overlay", None)
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
    ax.legend(fontsize=14)
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
    scol = "score" if "score" in ok.columns else "f0"   # 多目标用 f0 着色
    s = ok[scol].to_numpy(float)
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
    ax.set_xticklabels(keys, fontsize=16, rotation=20)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["min", "", "max"], fontsize=16)
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_title("参数×分数平行坐标（绿=好，红=差）")
    ax.legend(fontsize=16)


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
                ha="center", va="center", color="#888", fontsize=18)
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
    ax.legend(fontsize=16)
    try:
        import matplotlib.pyplot as plt
        plt.colorbar(sc, ax=ax, shrink=0.8).set_label("score", fontsize=16)
    except Exception:
        pass


def plot_pareto(ax, record):
    """Pareto 前沿：全部可行点（灰）+ 非支配前沿（绿）+ 连线。"""
    h = record.history
    if h is None or h.empty or "f0" not in h.columns:
        ax.text(0.5, 0.5, "多目标运行才有 Pareto 前沿（选 NSGA-II + ZDT）",
                ha="center", va="center", color="#888", fontsize=18)
        return
    ok = h[h["status"] == "ok"]
    ax.scatter(ok["f0"], ok["f1"], s=14, color="#bbb", alpha=0.7,
               label="全部评估")
    pf = record.pareto
    if pf is not None and len(pf):
        p = pf.sort_values("f0")
        ax.scatter(p["f0"], p["f1"], s=42, color=_GREEN, edgecolors="k",
                   linewidths=0.5, zorder=5, label=f"Pareto 前沿（{len(pf)}）")
        ax.plot(p["f0"], p["f1"], "-", color=_GREEN, lw=1.2, alpha=0.8)
    ax.set_xlabel("目标 f0")
    ax.set_ylabel("目标 f1")
    ax.set_title("Pareto 前沿（均最小化，绿=非支配）")
    ax.legend(fontsize=16)
    ax.grid(alpha=0.25)


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
    ax.legend(fontsize=16)
    ax.grid(alpha=0.25)


# ---------------------------------------------------------------- 默认页声明
def default_pages(record) -> list:
    """所有优化运行的通用页（多目标自动切 Pareto 组合）。

    代理切片页不在此列：它需要 optimizer 实例的 GP 状态，属"实例自足页"，
    由 GP-BO.inspect_pages 追加（见 engines/bo.py）。曲面体检页对任何有
    ≥10 个 ok 点的运行都有意义，故通用声明。
    """
    from .health import surface_health
    common = [PageSpec("parallel", "平行坐标", "mpl", plot_parallel),
              PageSpec("health", "曲面体检", "table",
                       data=surface_health)]
    if getattr(record, "multi", False):
        return [PageSpec("pareto", "Pareto 前沿", "mpl", plot_pareto),
                common[0], common[1]]
    return [PageSpec("conv", "收敛曲线", "mpl", plot_convergence),
            common[0],
            PageSpec("scatter", "探索地图", "mpl", plot_scatter2d),
            common[1]]
