# -*- coding: utf-8 -*-
"""把 :class:`PlotSpec` 画成 **SVG**（web 侧渲染器）。

## 为什么是 SVG 而不是引前端图表库

- **零依赖**：面板是个链接页，不该为画图引一个几百 KB 的库；
- **矢量**：与页面同分辨率、缩放不糊（PNG 的"格格不入"有一半来自栅格化）；
- **纯函数**：`spec → 字符串` ⇒ 可以被 `python` 判据直接覆盖
  （"每条分支都非空、坐标算得对、未知 kind 响亮"），而引库就只能靠肉眼看。

## 与 `plot_mpl.py` 的关系

**同一份 `PlotSpec`、两套渲染器、同一套配色与版式**（配色写在各自模块的 `PALETTE`，
判据里核对两边一致）。⇒ 改版式要改两处，这是"两个渲染器"的固有代价；
换来的是"方法层只谈数据"。
"""

from __future__ import annotations

from html import escape as _esc

#: 与 `plot_mpl.py` 同一套（判据核对两边一致）。
PALETTE = ("#0969da", "#cf222e", "#1a7f37", "#9a6700", "#8250df",
           "#0550ae", "#bc4c00", "#6639ba")

_GRID = "#d8dee4"
_AXIS = "#57606a"
_INK = "#1f2328"

W, H = 680, 400
L, R, T, B = 58, 16, 26, 44          # 边距（留左边给 y 标签、下边给 x 标签）
PW, PH = W - L - R, H - T - B


def _f(v, digits: int = 4) -> str:
    """数值 → 短字符串（去掉多余 0）。非有限 ⇒ ""。"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    if x != x or x in (float("inf"), float("-inf")):
        return ""
    s = f"{x:.{digits}g}"
    return s


def _fmt_tick(v) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    if x != x:
        return ""
    if x == 0:
        return "0"
    a = abs(x)
    if a >= 10000 or a < 0.01:
        return f"{x:.1e}"
    return f"{x:.3g}"


# ---------------------------------------------------------------- 刻度与范围
def _nice_ticks(lo: float, hi: float, n: int = 5) -> list:
    """等距刻度（够用即可；不做"漂亮数字"那一套，免得两套渲染器各算一套）。"""
    if not (hi > lo):
        hi = lo + 1.0
    step = (hi - lo) / max(1, n)
    return [lo + step * i for i in range(n + 1)]


def _slog(v: float, thr: float) -> float:
    """symlog 变换（与 matplotlib 的 symlog 同形）：近零线性、远零对数。"""
    import math
    a = abs(v)
    if a <= thr:
        return v / thr
    return math.copysign(1.0 + math.log10(a / thr), v)


def _extent(values, *, frac: float = 0.06):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    if hi == lo:
        pad = abs(lo) * 0.1 or 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * frac
    return lo - pad, hi + pad


def _collect_x(spec) -> list:
    xs = []
    if spec.x:
        xs += [v for v in spec.x if v is not None]
    for s in spec.series or ():
        if s.x is not None:
            xs += [v for v in s.x if v is not None]
    if xs:
        return xs
    # 没有 x ⇒ 用序号（取最长那条的长度）
    longest = max((len(s.y or ()) for s in (spec.series or ())), default=0)
    return list(range(longest))


def _collect_y(spec) -> list:
    ys = []
    for s in spec.series or ():
        ys += [v for v in (s.y or ()) if v is not None]
        for band in (s.fill_lo, s.fill_hi):
            if band is not None:
                ys += [v for v in band if v is not None]
    return ys


# ---------------------------------------------------------------- 主入口
def svg(spec) -> str:
    """``PlotSpec`` → SVG 字符串。**未知 kind 抛错**（空白页是那种假绿）。"""
    kind = getattr(spec, "kind", "")
    fn = {"line": _line, "scatter": _scatter, "barh": _barh, "bar": _bar,
          "hist": _hist, "heatmap": _heatmap, "parallel": _parallel}.get(kind)
    if fn is None:
        raise NotImplementedError(f"PlotSpec 渲染器不认识 kind={kind!r}")
    body, legend_items = fn(spec)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="100%" height="auto" role="img" '
        f'font-family="-apple-system,Segoe UI,Microsoft YaHei,sans-serif" '
        f'font-size="11" fill="{_INK}">',
    ]
    if spec.title:
        parts.append(f'<text x="{L}" y="16" font-size="12.5" '
                     f'font-weight="600">{_esc(spec.title)}</text>')
    parts.append(body)
    # 轴标签
    if spec.xlabel:
        parts.append(f'<text x="{L + PW / 2:.0f}" y="{H - 10}" '
                     f'text-anchor="middle" fill="{_AXIS}">{_esc(spec.xlabel)}</text>')
    if spec.ylabel:
        parts.append(f'<text x="14" y="{T + PH / 2:.0f}" text-anchor="middle" '
                     f'fill="{_AXIS}" transform="rotate(-90 14 {T + PH / 2:.0f})">'
                     f'{_esc(spec.ylabel)}</text>')
    # 色标（`scatter2d` 的分数色条）：用一小条离散色阶 + 两端数值说明
    if getattr(spec, "cmap", "") and getattr(spec, "colorbar_label", ""):
        vals = [v for s in (spec.series or ()) for v in (s.y or ()) if v is not None]
        if vals:
            parts.append(_colorbar(float(min(vals)), float(max(vals)),
                                   str(spec.colorbar_label)))
    if legend_items and spec.legend:
        x = L + 8
        y = T + 10
        for name, color in legend_items:
            parts.append(f'<rect x="{x}" y="{y - 7}" width="9" height="9" rx="2" '
                         f'fill="{color}"/>'
                         f'<text x="{x + 13}" y="{y + 1}">{_esc(name)}</text>')
            x += 13 + int(len(name) * 6.6) + 14
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- 各图种
def _frame(spec, xlo, xhi, ylo, yhi, xticks=None, xlabels=None):
    """网格 + 两条轴线 + 刻度文字（**不去上/右边框**：SVG 里本来就没有）。"""
    out = []
    if spec.grid:
        for t in _nice_ticks(ylo, yhi):
            y = _py(t, ylo, yhi)
            out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + PW}" y2="{y:.1f}" '
                       f'stroke="{_GRID}" stroke-width="1"/>')
        ticks = xticks if xticks is not None else _nice_ticks(xlo, xhi)
        for i, t in enumerate(ticks):
            x = _px(t, xlo, xhi) if xticks is None else \
                L + (PW * (i + 0.5) / max(1, len(ticks)))
            out.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T + PH}" '
                       f'stroke="{_GRID}" stroke-width="1"/>')
    if spec.yscale == "log":
        out.append(f'<text x="{L - 6}" y="{T + 10}" text-anchor="end" '
                   f'fill="{_AXIS}">log</text>')
    # 轴刻度数字
    for t in _nice_ticks(ylo, yhi):
        out.append(f'<text x="{L - 6}" y="{_py(t, ylo, yhi) + 3.5:.1f}" '
                   f'text-anchor="end" fill="{_AXIS}">{_fmt_tick(t)}</text>')
    ticks = xticks if xticks is not None else _nice_ticks(xlo, xhi)
    for i, t in enumerate(ticks):
        if xticks is None:
            x = _px(t, xlo, xhi)
            lab = _fmt_tick(t)
        else:
            x = L + (PW * (i + 0.5) / max(1, len(ticks)))
            lab = str(xlabels[i]) if xlabels and i < len(xlabels) else _fmt_tick(t)
        out.append(f'<text x="{x:.1f}" y="{T + PH + 14}" text-anchor="middle" '
                   f'fill="{_AXIS}">{_esc(lab)}</text>')
    # 轴线
    out.append(f'<line x1="{L}" y1="{T + PH}" x2="{L + PW}" y2="{T + PH}" '
               f'stroke="{_AXIS}" stroke-width="1"/>')
    out.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T + PH}" '
               f'stroke="{_AXIS}" stroke-width="1"/>')
    # 横线标记（残差零线、归因零线、异常阈值线…）
    for h in getattr(spec, "hlines", ()) or ():
        try:
            y = _py(float(h.get("y")), ylo, yhi)
        except (TypeError, ValueError):
            continue
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + PW}" y2="{y:.1f}" '
                   f'stroke="{_esc(h.get("color") or _AXIS)}" stroke-width="1" '
                   f'stroke-dasharray="4 3"/>')
    # 副轴提示（SVG 只有一条 y 轴；副轴数据按自己的极值缩放，这里标注一下）
    if getattr(spec, "ylabel2", ""):
        out.append(f'<text x="{L + PW}" y="{T + 10}" text-anchor="end" '
                   f'fill="{_AXIS}" font-size="10">{_esc(spec.ylabel2)}'
                   f'（副轴，按自身范围缩放）</text>')
    # 竖线标记
    for v in spec.vlines or ():
        try:
            x = _px(float(v.get("x")), xlo, xhi)
        except (TypeError, ValueError):
            continue
        out.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T + PH}" '
                   f'stroke="{_esc(v.get("color") or _AXIS)}" stroke-width="1" '
                   f'stroke-dasharray="4 3"/>')
    return "".join(out)


def _px(v, lo, hi) -> float:
    return L + PW * (float(v) - lo) / (hi - lo) if hi > lo else L + PW / 2


def _py(v, lo, hi) -> float:
    return T + PH - PH * (float(v) - lo) / (hi - lo) if hi > lo else T + PH / 2


def _series_xy(spec, s):
    """取一条 series 的 (x, y)：``s.x`` 优先，回落到 ``spec.x``，再回落到序号。"""
    x = s.x if s.x is not None else spec.x
    y = list(s.y or ())
    if not x:
        x = list(range(len(y)))
    return list(x), y


def _color_at(s, i, idx):
    """取第 idx 个点的颜色：``s.color`` 可以是**单色**或**逐点色数组**。"""
    c = getattr(s, "color", "")
    if isinstance(c, (list, tuple)):
        return str(c[idx]) if idx < len(c) else PALETTE[i % len(PALETTE)]
    return str(c) if c else PALETTE[i % len(PALETTE)]


def _polyline(pts, color, *, dash=False, width=1.4, marker="", alpha=0.9) -> str:
    if len(pts) < 2 and not marker:
        return ""
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    style = (f'fill="none" stroke="{color}" stroke-width="{width}" '
             f'stroke-opacity="{float(alpha):.2f}" stroke-linejoin="round"')
    if dash:
        style += ' stroke-dasharray="5 3"'
    out = [f'<polyline points="{d}" {style}/>']
    if marker:
        for x, y in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{color}"/>')
    return "".join(out)


def _legend_of(spec):
    """图例项（跳过 ``no_legend`` 的 series）。"""
    out = []
    for i, s in enumerate(spec.series or ()):
        if getattr(s, "no_legend", False) or not s.name:
            continue
        c = getattr(s, "color", "")
        out.append((s.name, c if isinstance(c, str) and c else PALETTE[i % len(PALETTE)]))
    for h in getattr(spec, "hlines", ()) or ():
        if h.get("label"):
            out.append((str(h["label"]), h.get("color") or _AXIS))
    for v in getattr(spec, "vlines", ()) or ():
        if v.get("label"):
            out.append((str(v["label"]), v.get("color") or _AXIS))
    return out


def _line(spec):
    """折线/散点（同一支：`connect=False` 就是散点）。

    ⚠ 这里处理四件事，两个渲染器必须一致：
    - `area=True` ⇒ 到 `fill_base` 之间填色（轮廓图、残差带）；
    - `color` 可以是**逐点色数组**（按标签着色）；
    - `axis2=True` 的 series 按**它自己那组数据**的极值缩放（SVG 里就一条 y 轴，
      所以画在右半边并标注，见 `_frame` 的 `ylabel2`）；
    - `yscale="symlog"` 时先做 symlog 变换再缩放。
    """
    sym = getattr(spec, "yscale", "") == "symlog"
    thr = float(getattr(spec, "linthresh", 1e-3) or 1e-3)
    tr = (lambda v: _slog(float(v), thr)) if sym else (lambda v: float(v))

    prim = [s for s in (spec.series or ()) if not getattr(s, "axis2", False)]
    sec = [s for s in (spec.series or ()) if getattr(s, "axis2", False)]
    xs_all = _collect_x(spec)
    ys_prim = [tr(v) for v in
               [y for s in prim for y in (s.y or ()) if y is not None]]
    if spec.xlim:
        xe = (float(spec.xlim[0]), float(spec.xlim[1]))
    else:
        xe = _extent(xs_all)
    if spec.ylim:
        ye = (tr(spec.ylim[0]), tr(spec.ylim[1]))
    else:
        ye = _extent(ys_prim)
    if xe is None or ye is None:
        return _empty("没有可画的点"), []
    xlo, xhi = xe
    ylo, yhi = ye
    ysec = [tr(v) for s in sec for v in (s.y or ()) if v is not None]
    yl2 = getattr(spec, "ylim2", None)
    se = ((tr(yl2[0]), tr(yl2[1])) if yl2 else (_extent(ysec) or (0.0, 1.0)))
    out = [_frame(spec, xlo, xhi, ylo, yhi)]
    legend = _legend_of(spec)
    for i, s in enumerate(spec.series or ()):
        second = getattr(s, "axis2", False)
        axis = se if second else (ylo, yhi)
        x, y = _series_xy(spec, s)
        if not x:
            x = list(range(len(y)))
        n = min(len(x), len(y))
        pts = [(_px(x[j], xlo, xhi), _py(tr(y[j]), axis[0], axis[1]))
               for j in range(n) if y[j] is not None]
        color = _color_at(s, i, 0)
        # 面积：到基线之间填色
        if getattr(s, "area", False) and pts:
            base = _py(tr(float(getattr(s, "fill_base", 0.0))), axis[0], axis[1])
            up = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
            out.append(f'<polygon points="{up} {pts[-1][0]:.1f},{base:.1f} '
                       f'{pts[0][0]:.1f},{base:.1f}" fill="{color}" '
                       f'fill-opacity="{float(s.alpha) * 0.35:.2f}"/>')
        if s.fill_lo is not None and s.fill_hi is not None:
            lo, hi = list(s.fill_lo)[:n], list(s.fill_hi)[:n]
            up = " ".join(f"{_px(x[j], xlo, xhi):.1f},{_py(tr(hi[j]), axis[0], axis[1]):.1f}"
                          for j in range(n) if hi[j] is not None)
            dn = " ".join(f"{_px(x[j], xlo, xhi):.1f},{_py(tr(lo[j]), axis[0], axis[1]):.1f}"
                          for j in range(n - 1, -1, -1) if lo[j] is not None)
            if up and dn:
                out.append(f'<polygon points="{up} {dn}" fill="{color}" '
                           f'fill-opacity="0.12"/>')
        if not getattr(s, "connect", True):
            for j, (px, py) in enumerate(pts):
                out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" '
                           f'fill="{_color_at(s, i, j)}" '
                           f'fill-opacity="{float(s.alpha):.2f}"/>')
            continue
        dash = ' stroke-dasharray="5 3"' if s.dash else ""
        d = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="{color}" '
                   f'stroke-width="{float(s.width)}" '
                   f'stroke-opacity="{float(s.alpha):.2f}"{dash} '
                   f'stroke-linejoin="round"/>')
        if s.marker:
            for px, py in pts:
                out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" '
                           f'fill="{color}"/>')
    return "".join(out), legend


def _scatter(spec):
    xall, yall = _collect_x(spec), _collect_y(spec)
    xe, ye = _extent(xall), _extent(yall)
    if xe is None or ye is None:
        return _empty("没有可画的点"), []
    xlo, xhi = xe
    ylo, yhi = ye
    out = [_frame(spec, xlo, xhi, ylo, yhi)]
    legend = []
    for i, s in enumerate(spec.series or ()):
        color = s.color or PALETTE[i % len(PALETTE)]
        xs = list(s.x if s.x is not None else (spec.x or range(len(s.y or ()))))
        ys = list(s.y or ())
        n = min(len(xs), len(ys))
        for j in range(n):
            out.append(f'<circle cx="{_px(xs[j], xlo, xhi):.1f}" '
                       f'cy="{_py(ys[j], ylo, yhi):.1f}" r="3" fill="{color}" '
                       f'fill-opacity="0.55"/>')
        if s.name:
            legend.append((s.name, color))
    return "".join(out), legend


def _barh(spec):
    """横向条形。

    ⚠ 这一支**不能用通用 `_frame`**（实测踩过三个坑）：
    1. 通用框架会画 y 轴**数字刻度**，而横向条形的 y 是**类别** ⇒ 数字会压在类别名上；
    2. 条形高度按"n 等分"算会在 n 小的时候**溢出**绘图区；
    3. y 范围用 `(-0.6, n-0.4)` 会在顶部留一大块空白 ⇒ 改成 `n-1`。
    ⇒ 这一支自己画网格与刻度（x 轴数字 + y 轴类别名），只在需要时借公共的坐标换算。
    """
    vals = [v for v in (spec.values or ())]
    rows = [str(r) for r in (spec.rows or ())]
    n = len(vals)
    if not n:
        return _empty("没有可画的条目"), []
    nums = [float(v) for v in vals if v is not None] or [0.0]
    vmax = max(nums + [0.0]) or 1.0
    xlo, xhi = 0.0, vmax * 1.06
    ylo, yhi = -0.5, n - 0.5                      # ⚠ 不留多余空白
    out = []
    # 网格（竖线，按 x 刻度）
    xticks = _nice_ticks(xlo, xhi)
    if spec.grid:
        for t in xticks:
            x = _px(t, xlo, xhi)
            out.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T + PH}" '
                       f'stroke="{_GRID}" stroke-width="1"/>')
    band = PH / n
    bar_h = min(band * 0.66, 26)
    for i, v in enumerate(vals):
        if v is None:
            continue
        y = _py(i, ylo, yhi)
        w = _px(float(v), xlo, xhi) - L
        rc = list(getattr(spec, "rowcolors", ()) or ())
        col = rc[i] if i < len(rc) and rc[i] else PALETTE[2]
        out.append(f'<rect x="{L}" y="{y - bar_h / 2:.1f}" width="{max(w, 0):.1f}" '
                   f'height="{bar_h:.1f}" rx="2" fill="{col}" '
                   f'fill-opacity="0.85"/>')
        lab = rows[i] if i < len(rows) else str(i)
        out.append(f'<text x="{L - 6}" y="{y + 3.5:.1f}" text-anchor="end" '
                   f'fill="{_AXIS}">{_esc(lab[:22])}</text>')
    # x 轴刻度数字（**只画 x**，不再画 y 数字）
    for t in xticks:
        out.append(f'<text x="{_px(t, xlo, xhi):.1f}" y="{T + PH + 14}" '
                   f'text-anchor="middle" fill="{_AXIS}">{_fmt_tick(t)}</text>')
    out.append(f'<line x1="{L}" y1="{T + PH}" x2="{L + PW}" y2="{T + PH}" '
               f'stroke="{_AXIS}" stroke-width="1"/>')
    out.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T + PH}" '
               f'stroke="{_AXIS}" stroke-width="1"/>')
    return "".join(out), []


def _bar(spec):
    """竖条形（`as_bar` 的 series 画条，其余按线画：方差+累计线就是这种组合）。"""
    xall = _collect_x(spec)
    yall = [v for s in (spec.series or ()) for v in (s.y or ()) if v is not None]
    xe = _extent(xall) or (0.0, 1.0)
    ye = _extent(yall) or (0.0, 1.0)
    ylo, yhi = min(0.0, ye[0]), ye[1]
    xlo, xhi = xe
    out = [_frame(spec, xlo, xhi, ylo, yhi)]
    legend = _legend_of(spec)
    if xhi > xlo:
        bw = PW / (len(xall) or 1) * 0.72
    else:
        bw = 8.0
    for i, s in enumerate(spec.series or ()):
        x, y = _series_xy(spec, s)
        if not x:
            x = list(range(1, len(y) + 1))
        n = min(len(x), len(y))
        color = _color_at(s, i, 0)
        if getattr(s, "as_bar", False):
            for j in range(n):
                if y[j] is None:
                    continue
                y0 = _py(max(0.0, 0.0), ylo, yhi)
                y1 = _py(float(y[j]), ylo, yhi)
                cx = _px(x[j], xlo, xhi)
                out.append(f'<rect x="{cx - bw / 2:.1f}" y="{min(y0, y1):.1f}" '
                           f'width="{bw:.1f}" height="{abs(y1 - y0):.1f}" rx="2" '
                           f'fill="{color}" fill-opacity="{float(s.alpha):.2f}"/>')
        else:
            pts = " ".join(f"{_px(x[j], xlo, xhi):.1f},{_py(float(y[j]), ylo, yhi):.1f}"
                           for j in range(n) if y[j] is not None)
            out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                       f'stroke-width="{float(s.width)}"/>')
            for j in range(n):
                if y[j] is None:
                    continue
                out.append(f'<circle cx="{_px(x[j], xlo, xhi):.1f}" '
                           f'cy="{_py(float(y[j]), ylo, yhi):.1f}" r="2.6" '
                           f'fill="{color}"/>')
    return "".join(out), legend


def _hist(spec):
    """直方图：`spec.x` 是"全体"一组，各 series 各一组（异常分数图两组叠加）。"""
    groups = []
    if spec.x:
        groups.append(("", [float(v) for v in spec.x if v is not None],
                       PALETTE[0], 0.35, True))
    for i, s in enumerate(spec.series or ()):
        vals = [float(v) for v in (s.y or ()) if v is not None]
        if vals:
            groups.append((s.name, vals, _color_at(s, i, 0), float(s.alpha),
                           bool(getattr(s, "no_legend", False))))
    if not groups:
        return _empty("没有可画的数据"), []
    allv = [v for _n, vals, _c, _a, _q in groups for v in vals]
    lo, hi = min(allv), max(allv)
    if hi <= lo:
        hi = lo + 1.0
    nb = max(1, int(getattr(spec, "bins", 40) or 40))
    step = (hi - lo) / nb
    counts = []
    for _n, vals, _c, _a, _q in groups:
        c = [0] * nb
        for v in vals:
            k = min(nb - 1, max(0, int((v - lo) / step)))
            c[k] += 1
        counts.append(c)
    ymax = max([1] + [max(c) for c in counts])
    out = [_frame(spec, lo, hi, 0.0, ymax * 1.06)]
    bw = PW / nb
    for gi, (_n, _vals, color, alpha, _q) in enumerate(groups):
        for k in range(nb):
            hgt = PH * counts[gi][k] / (ymax * 1.06)
            if hgt <= 0:
                continue
            out.append(f'<rect x="{L + k * bw:.1f}" y="{T + PH - hgt:.1f}" '
                       f'width="{max(bw - 0.6, 0.6):.1f}" height="{hgt:.1f}" '
                       f'fill="{color}" fill-opacity="{alpha:.2f}"/>')
    legend = [(n, c) for n, _v, c, _a, q in groups if n and not q]
    return "".join(out), legend


def _parallel(spec):
    """平行坐标：x 是**等距分类轴**（参数名），每条 series 一次评估的轨迹。"""
    n_axes = max([len(s.y or ()) for s in (spec.series or ())] + [0])
    if not n_axes:
        return _empty("没有可画的参数"), []
    xlo, xhi = -0.5, n_axes - 0.5
    ylo, yhi = 0.0, 1.0
    out = []
    if spec.grid:
        for j in range(n_axes):
            x = _px(j, xlo, xhi)
            out.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T + PH}" '
                       f'stroke="{_GRID}" stroke-width="1"/>')
    labels = [str(v) for v in (getattr(spec, "xticklabels", ()) or ())]
    for j in range(n_axes):
        lab = labels[j] if j < len(labels) else str(j)
        out.append(f'<text x="{_px(j, xlo, xhi):.1f}" y="{T + PH + 14}" '
                   f'text-anchor="middle" fill="{_AXIS}">{_esc(lab[:12])}</text>')
    for name, y in (("min", 0.0), ("max", 1.0)):
        out.append(f'<text x="{L - 6}" y="{_py(y, ylo, yhi) + 3.5:.1f}" '
                   f'text-anchor="end" fill="{_AXIS}">{name}</text>')
    for i, s in enumerate(spec.series or ()):
        y = list(s.y or ())
        pts = " ".join(f"{_px(j, xlo, xhi):.1f},{_py(float(y[j]), ylo, yhi):.1f}"
                       for j in range(len(y)) if y[j] is not None)
        out.append(f'<polyline points="{pts}" fill="none" '
                   f'stroke="{_color_at(s, i, 0)}" stroke-width="{float(s.width)}" '
                   f'stroke-opacity="{float(s.alpha):.2f}"/>')
    return "".join(out), _legend_of(spec)


def _heatmap(spec):
    vals = list(spec.values or ())
    if not vals:
        return _empty("没有可画的值"), []
    nr = len(vals)
    nc = len(vals[0]) if isinstance(vals[0], (list, tuple)) else 0
    if not nc:
        return _empty("热力图需要二维数据"), []
    flat = [v for row in vals for v in row if v is not None]
    hi = max(flat) if flat else 1.0
    lo = min(flat) if flat else 0.0
    rows = [str(r) for r in (spec.rows or ())]
    cw, ch = PW / nc, PH / nr
    out = []
    for i in range(nr):
        for j in range(nc):
            v = vals[i][j]
            t = 0.0 if hi == lo else (float(v) - lo) / (hi - lo)
            # 蓝色阶：#ddf4ff（浅）→ #0550ae（深），与页面调色一致
            col = _blend((221, 244, 255), (5, 80, 174), t)
            out.append(f'<rect x="{L + j * cw:.1f}" y="{T + i * ch:.1f}" '
                       f'width="{cw:.1f}" height="{ch:.1f}" fill="{col}" '
                       f'stroke="#fff" stroke-width="1"/>')
            label = f"{v:g}" if isinstance(v, (int, float)) else str(v)
            # 格子里数字的颜色**优先用 spec 给的**（深底白字）——实测深蓝上的黑字看不清
            tcol = _INK
            for tx in spec.texts or ():
                if tx.get("x") == j and tx.get("y") == i and tx.get("color"):
                    tcol = str(tx["color"])
                    break
            else:
                tcol = "#fff" if t > 0.55 else _INK
            out.append(f'<text x="{L + j * cw + cw / 2:.1f}" '
                       f'y="{T + i * ch + ch / 2 + 4:.1f}" text-anchor="middle" '
                       f'fill="{tcol}" font-size="10">'
                       f'{_esc(label)}</text>')
    for i in range(nr):
        lab = rows[i] if i < len(rows) else str(i)
        out.append(f'<text x="{L - 6}" y="{T + i * ch + ch / 2 + 4:.1f}" '
                   f'text-anchor="end" fill="{_AXIS}">{_esc(lab[:12])}</text>')
    return "".join(out), []


def _blend(c1, c2, t) -> str:
    t = max(0.0, min(1.0, float(t)))
    return "#%02x%02x%02x" % tuple(
        int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def _empty(why: str) -> str:
    """没有可画的东西 ⇒ **说出来**（不是空白页）。"""
    return (f'<text x="{L}" y="{T + 24}" fill="{_AXIS}">{_esc(why)}'
            f'（这一页没有可画的数据）</text>')


def _colorbar(lo: float, hi: float, label: str, steps: int = 24) -> str:
    """画一条竖直色标（SVG 里没有 matplotlib 的 colorbar 可借，自己画一条）。"""
    x = L + PW + 6
    y0, hgt = T, PH * 0.7
    cell = hgt / steps
    out = []
    for i in range(steps):
        t = i / max(1, steps - 1)
        out.append(f'<rect x="{x:.1f}" y="{y0 + i * cell:.1f}" width="9" '
                   f'height="{cell + 0.6:.1f}" fill="{_rdylgn_inv(t)}"/>')
    out.append(f'<text x="{x + 13:.1f}" y="{y0 + 8:.1f}" fill="{_AXIS}" '
               f'font-size="10">{_fmt_tick(lo)}</text>')
    out.append(f'<text x="{x + 13:.1f}" y="{y0 + hgt:.1f}" fill="{_AXIS}" '
               f'font-size="10">{_fmt_tick(hi)}</text>')
    out.append(f'<text x="{x:.1f}" y="{y0 + hgt + 14:.1f}" fill="{_AXIS}" '
               f'font-size="10">{_esc(label)}</text>')
    return "".join(out)


def _rdylgn_inv(t: float) -> str:
    """viridis 近似的**反向**（色标上端=高分）。与 `opt/specs._viridis_r` 同一支。"""
    stops = ((68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98),
             (253, 231, 37))
    t = max(0.0, min(1.0, float(t)))
    k = t * (len(stops) - 1)
    i = min(int(k), len(stops) - 2)
    return _blend(stops[i], stops[i + 1], k - i)
