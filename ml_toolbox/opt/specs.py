# -*- coding: utf-8 -*-
"""优化侧的绘图规格函数：``spec_xxx(record, extra=None) -> PlotSpec``（纯函数、零渲染依赖）。

对标 `methods/specs.py`：**数在这里取，画法归渲染器**。
`opt/plots.py` 里的旧函数保留为薄适配器（GUI 的 `default_pages` 指向它们，调用点不改）。

## 与 ML 侧的差异（如实记）

- 入参是 ``OptRecord``（不是 ``result``）：优化侧的"结果"就是 `history` DataFrame；
- **平行坐标**用 ``kind="parallel"``：x 是**等距分类轴**（参数名），不做数值换算；
- **收敛曲线**用 ``yscale="symlog"``：分数会跨很多量级，线性轴看不出来。
"""

from __future__ import annotations

import numpy as np

from ..core.plot_spec import PlotSpec, Series

GREEN = "#1a7f37"
BLUE = "#0969da"
GRAY = "#8c8c8c"

#: 页 key → spec 函数（与 `methods/specs.SPEC_BY_KEY` 同款，面板查同一张表）。
SPEC_BY_KEY = {
    "conv": "spec_convergence",
    "parallel": "spec_parallel",
    "scatter": "spec_scatter2d",
    "pareto": "spec_pareto",
    "surrogate": "spec_surrogate_1d",
}


def spec_for_key(key: str):
    """按页 key 取 spec 函数；未迁移 ⇒ ``None``。"""
    name = SPEC_BY_KEY.get(str(key or ""))
    return globals().get(name) if name else None


def _l(seq):
    out = []
    for v in list(seq):
        try:
            f = float(v)
            out.append(None if f != f else f)
        except (TypeError, ValueError):
            out.append(None)
    return out


def _ok(record):
    h = record.history
    return h[h["status"] == "ok"]


def spec_convergence(record, extra=None) -> PlotSpec:
    """收敛曲线：best-so-far 阶梯 + 单次评估散点（+可选叠加对比）。"""
    if extra is None:
        extra = getattr(record, "_overlay", None)
    h = record.history
    if h is None or getattr(h, "empty", True):
        return PlotSpec(kind="line")
    ok = _ok(record)
    xs = list(range(1, len(h) + 1))
    series = [Series(name=f"best-so-far（{record.optimizer}）",
                     x=xs, y=_l(h["best_so_far"]), color=GREEN, marker="o",
                     width=1.6)]
    if len(ok):
        series.append(Series(name="单次评估", x=xs, y=_l(h["score"]),
                             color=BLUE, connect=False, marker="o", alpha=0.5))
    for label, hh in (extra or []):
        if hh is not None and len(hh):
            series.append(Series(name=str(label),
                                 x=list(range(1, len(hh) + 1)),
                                 y=_l(hh["best_so_far"]), dash=True, width=1.2,
                                 alpha=0.85))
    return PlotSpec(kind="line", series=series, yscale="symlog", linthresh=1e-3,
                    ).axes(xlabel="评估次数", ylabel="分数（log 邻域）",
                           title="收敛曲线（越小越好）")


def spec_parallel(record) -> PlotSpec:
    """平行坐标：每维一根竖轴，每条线=一次评估（颜色=分数，绿=好）。"""
    ok = _ok(record)
    keys = [d["key"] for d in record.space_desc]
    keys = [k for k in keys if k in ok.columns]
    if ok.empty or not keys:
        return PlotSpec(kind="parallel")
    # ⚠ 参数里**可能有字符串**（类别型，如 'b'）⇒ 不能直接 `to_numpy(float)`
    #   （实测：`ValueError: could not convert string to float: 'b'`，原实现同样会崩）。
    #   处理：数值列按数值归一化；非数值列按**取值序号**归一化（类别轴的标准做法）。
    cols, used = [], []
    for k in keys:
        col = ok[k]
        if col.dtype.kind in "ifu":
            cols.append(col.to_numpy(float))
            used.append(k)
            continue
        vals = list(dict.fromkeys(col.astype(str).tolist()))
        lut = {v: i for i, v in enumerate(vals)}
        if len(vals) < 2:
            continue
        cols.append(np.asarray([lut[v] for v in col.astype(str)], float))
        used.append(k)
    if not cols:
        return PlotSpec(kind="parallel")
    M = np.column_stack(cols)
    keys = used
    lo, hi = M.min(0), M.max(0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    Z = (M - lo) / span
    scol = "score" if "score" in ok.columns else "f0"
    s = ok[scol].to_numpy(float)
    smin, smax = np.nanmin(s), np.nanmax(s)
    sn = (s - smin) / max(smax - smin, 1e-12)
    # 逐条颜色：好=绿、差=红（用色阶算出具体色值，两个渲染器照画）
    colors = [_rdylgn(1.0 - float(v)) for v in sn]
    series = [Series(y=_l(Z[i]), color=colors[i], width=0.8, alpha=0.35,
                     no_legend=True) for i in range(len(Z))]
    best_i = int(np.argmin(s))
    series.append(Series(name="当前最优", y=_l(Z[best_i]), color=GREEN, width=2.0))
    return PlotSpec(kind="parallel", series=series, xticklabels=keys,
                    title="参数×分数平行坐标（绿=好，红=差）")


def spec_scatter2d(record) -> PlotSpec:
    """探索地图：前两个连续参数为轴，颜色=分数，×=失败点，★=最优。"""
    cont = [d["key"] for d in record.space_desc
            if d["kind"] in ("uniform", "log-uniform", "randint")]
    h = record.history
    if len(cont) < 2 or h is None or h.empty:
        return PlotSpec(kind="scatter")
    a, b = cont[0], cont[1]
    ok = _ok(record)
    fail = h[h["status"] == "failed"]
    series = []
    if len(ok):
        series.append(Series(name="", x=_l(ok[a]), y=_l(ok[b]),
                             color=[_viridis_r(v) for v in
                                    _norm(_l(ok["score"]))],
                             no_legend=True, connect=False, marker="o"))
    if len(fail):
        series.append(Series(name=f"失败 {len(fail)}", x=_l(fail[a]),
                             y=_l(fail[b]), color="#b62324", connect=False,
                             marker="x"))
    if len(ok):
        bx = ok.loc[ok["score"].idxmin()]
        series.append(Series(name="最优", x=[float(bx[a])], y=[float(bx[b])],
                             color=GREEN, connect=False, marker="*"))
    return PlotSpec(kind="scatter", series=series, cmap="viridis_r",
                    colorbar_label="score",
                    title="探索地图（前两个连续参数）").axes(xlabel=a, ylabel=b)


def spec_pareto(record) -> PlotSpec:
    """Pareto 前沿：全部可行点（灰）+ 非支配前沿（绿）+ 连线。"""
    h = record.history
    if h is None or h.empty or "f0" not in h.columns:
        return PlotSpec(kind="scatter")
    ok = h[h["status"] == "ok"]
    series = [Series(name="全部评估", x=_l(ok["f0"]), y=_l(ok["f1"]),
                     color="#bbbbbb", connect=False, marker="o", alpha=0.7)]
    pf = record.pareto
    if pf is not None and len(pf):
        p = pf.sort_values("f0")
        series.append(Series(name=f"Pareto 前沿（{len(pf)}）", x=_l(p["f0"]),
                             y=_l(p["f1"]), color=GREEN, marker="o"))
    return PlotSpec(kind="line", series=series,
                    ).axes(xlabel="目标 f0", ylabel="目标 f1",
                           title="Pareto 前沿（均最小化，绿=非支配）")


def spec_surrogate_1d(bundle) -> PlotSpec:
    """GP 后验切片：``bundle = (xs, mu, sd, obs_x, obs_y, label)``。"""
    xs, mu, sd, obs_x, obs_y, label = bundle
    mu = np.asarray(mu, float)
    sd = np.asarray(sd, float)
    series = [Series(name="GP 后验均值", x=_l(xs), y=_l(mu), color=BLUE, width=1.6,
                     fill_lo=_l(mu - 2 * sd), fill_hi=_l(mu + 2 * sd))]
    if len(obs_x):
        series.append(Series(name="已观测", x=_l(obs_x), y=_l(obs_y),
                             color="#1f2328", connect=False, marker="o"))
    return PlotSpec(kind="line", series=series,
                    ).axes(xlabel=str(label), ylabel="score",
                           title="代理模型切片（固定其余参数于当前最优）")


# ---------------------------------------------------------------- 色阶（两处共用同一实现）
def _rdylgn(t: float) -> str:
    """RdYlGn 的近似（红→黄→绿）。**自己算色值**，免得渲染器各引一套 colormap。"""
    t = max(0.0, min(1.0, float(t)))
    if t < 0.5:
        u = t / 0.5
        return _mix((166, 35, 38), (255, 235, 120), u)
    u = (t - 0.5) / 0.5
    return _mix((255, 235, 120), (26, 127, 55), u)


def _viridis_r(t: float) -> str:
    """viridis 的近似（紫→蓝→绿→黄）；``t=1`` 是低分（viridis_r）。"""
    stops = ((68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98),
             (253, 231, 37))
    t = max(0.0, min(1.0, float(t)))
    k = t * (len(stops) - 1)
    i = min(int(k), len(stops) - 2)
    return _mix(stops[i], stops[i + 1], k - i)


def _mix(c1, c2, u: float) -> str:
    return "#%02x%02x%02x" % tuple(
        int(round(a + (b - a) * u)) for a, b in zip(c1, c2))


def _norm(vals):
    """线性归一化到 [0,1]（全等时给 0.5，避免除零）。"""
    nums = [v for v in vals if v is not None]
    if not nums:
        return [0.5 for _ in vals]
    lo, hi = min(nums), max(nums)
    if hi - lo < 1e-12:
        return [0.5 for _ in vals]
    return [0.5 if v is None else (v - lo) / (hi - lo) for v in vals]
