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

from ..core.contracts import PageSpec
from ..core import plot_mpl
from . import specs as _specs

_GREEN = "#2ca02c"
_BLUE = "#1f77b4"


def _ok(record):
    return record.history[record.history["status"] == "ok"]


def plot_convergence(ax, record, extra=None):
    """收敛曲线：best-so-far 阶梯 + 单次评估散点（+可选叠加对比）。

    ⚠ **已迁移**：数由 `opt/specs.spec_convergence` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 `default_pages` 与 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    spec = _specs.spec_convergence(record, extra=extra)
    plot_mpl.draw(spec, ax)
    _finish_opt(spec, ax)


def plot_parallel(ax, record):
    """平行坐标：每维一根竖轴，每条线=一次评估。

    ⚠ **已迁移**：数由 `opt/specs.spec_parallel` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 `default_pages` 与 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    spec = _specs.spec_parallel(record)
    plot_mpl.draw(spec, ax)
    _finish_opt(spec, ax)


def plt_cmap():
    import matplotlib.pyplot as plt
    return plt.get_cmap("RdYlGn")


def plot_scatter2d(ax, record):
    """探索地图：前两个连续参数为轴，颜色=分数。

    ⚠ **已迁移**：数由 `opt/specs.spec_scatter2d` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 `default_pages` 与 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    spec = _specs.spec_scatter2d(record)
    plot_mpl.draw(spec, ax)
    _finish_opt(spec, ax)


def plot_pareto(ax, record):
    """Pareto 前沿：全部评估（灰）+ 非支配前沿（绿）+ 连线。

    ⚠ **已迁移**：数由 `opt/specs.spec_pareto` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 `default_pages` 与 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    spec = _specs.spec_pareto(record)
    plot_mpl.draw(spec, ax)
    _finish_opt(spec, ax)


def plot_surrogate_1d(ax, bundle):
    """GP 后验切片：均值线 + ±2σ 带 + 已观测点（入参是 `bundle`，不是 record）。

    ⚠ **已迁移**：数由 `opt/specs.spec_surrogate_1d` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 `default_pages` 与 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    spec = _specs.spec_surrogate_1d(bundle)
    plot_mpl.draw(spec, ax)
    _finish_opt(spec, ax)


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


def _finish_opt(spec, ax):
    """优化侧共同的收尾：**colorbar** 是 matplotlib 专属（SVG 侧用色阶自带）。

    ⚠ 为什么放在这里而不是渲染器里：`scatter2d` 的色标是"这一页要的"，
    不是所有散点图都要 ⇒ 归页的声明，不归渲染器的通用行为。
    """
    try:
        if getattr(spec, "cmap", "") and spec.kind == "scatter":
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
            vals = [v for s in spec.series for v in (s.y or ()) if v is not None]
            if vals:
                sm = plt.cm.ScalarMappable(cmap="viridis_r",
                                           norm=Normalize(min(vals), max(vals)))
                ax.figure.colorbar(sm, ax=ax, shrink=0.8).set_label(
                    getattr(spec, "colorbar_label", "score"), fontsize=9)
    except Exception:
        pass
