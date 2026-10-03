# -*- coding: utf-8 -*-
"""把 :class:`PlotSpec` 画成 matplotlib 图（**GUI 侧渲染器**）。

## 这层为什么存在（用户裁决的直接结果）

原先 ``plot(ax, result)`` 让**方法自己调 matplotlib** ⇒ 数据与图分不开、换不了渲染器。
现在方法只吐 ``PlotSpec``，**画法集中在这里**（GUI）与 ``panel_svg.py``（web）。
⇒ 同一份数据两处渲染，且**改版式只需改渲染器**（不必碰方法）。

## 风格：与 web 面板对齐（用户裁决"接受统一新风格"）

- 字体与页面同一族（Segoe UI / 微软雅黑）；中文不再变方框；
- **去掉上/右边框**、网格更淡、图例无边框；背景透明。
"""

from __future__ import annotations

#: 与 `ml_mecha/panel_view.py` 的页面风格对齐的配色（改这里 = 两处一起变）。
PALETTE = ("#0969da", "#cf222e", "#1a7f37", "#9a6700", "#8250df",
           "#0550ae", "#bc4c00", "#6639ba")

_GRID = "#d8dee4"
_AXIS = "#57606a"
_INK = "#1f2328"
_CJK_DONE = False


def apply_page_style() -> None:
    """把页面风格写进 matplotlib 的 rcParams（**进程级、幂等**）。"""
    try:
        from matplotlib import rcParams
        _ensure_cjk()
        rcParams["font.size"] = 11
        rcParams["axes.edgecolor"] = _AXIS
        rcParams["axes.labelcolor"] = _INK
        rcParams["axes.titlecolor"] = _INK
        rcParams["text.color"] = _INK
        rcParams["xtick.color"] = _AXIS
        rcParams["ytick.color"] = _AXIS
        rcParams["axes.spines.top"] = False        # 去上/右边框（网页式版面）
        rcParams["axes.spines.right"] = False
        rcParams["legend.frameon"] = False
        rcParams["figure.facecolor"] = "none"
        rcParams["axes.facecolor"] = "none"
        rcParams["savefig.transparent"] = True
    except Exception:
        pass


def _ensure_cjk() -> None:
    """让中文能显示（缺字体时轴标签变方框 ⇒ 图能看但读不懂）。"""
    global _CJK_DONE
    if _CJK_DONE:
        return
    _CJK_DONE = True
    try:
        from matplotlib import font_manager, rcParams
        have = {f.name for f in font_manager.fontManager.ttflist}
        for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                     "Source Han Sans SC", "PingFang SC", "WenQuanYi Zen Hei"):
            if name in have:
                rcParams["font.family"] = [name, "DejaVu Sans"]
                rcParams["axes.unicode_minus"] = False
                return
    except Exception:
        pass


def draw(spec, ax) -> None:
    """把 ``spec`` 画到 ``ax`` 上。**未知 kind 抛错**（不画空白——空白是那种假绿）。"""
    apply_page_style()
    kind = getattr(spec, "kind", "")
    fn = {"line": _line, "scatter": _scatter, "barh": _barh, "bar": _bar,
          "hist": _hist, "heatmap": _heatmap, "parallel": _parallel}.get(kind)
    if fn is None:
        raise NotImplementedError(f"PlotSpec 渲染器不认识 kind={kind!r}")
    fn(spec, ax)
    _decorate(spec, ax)


# ---------------------------------------------------------------- 取数
def _series_xy(spec, s):
    x = s.x if s.x is not None else spec.x
    return list(x or ()), list(s.y or ())


def _color_of(s, i):
    """单色或**逐点色数组**；空 ⇒ 按调色板。"""
    c = getattr(s, "color", "")
    if isinstance(c, (list, tuple)):
        return list(c)
    return c or PALETTE[i % len(PALETTE)]


def _ax2(spec, ax):
    """副轴（按需创建一次；`nn_dyn` 的 lr/grad_norm 要 log 副轴）。"""
    a2 = getattr(ax, "_spec_ax2", None)
    if a2 is None:
        a2 = ax.twinx()
        a2.spines["top"].set_visible(False)
        if getattr(spec, "ylabel2", ""):
            a2.set_ylabel(spec.ylabel2)
        if getattr(spec, "yscale2", ""):
            try:
                a2.set_yscale(spec.yscale2)
            except Exception:
                pass
        ax._spec_ax2 = a2
    return a2


# ---------------------------------------------------------------- 各图种
def _line(spec, ax) -> None:
    for i, s in enumerate(spec.series or ()):
        target = _ax2(spec, ax) if getattr(s, "axis2", False) else ax
        x, y = _series_xy(spec, s)
        color = _color_of(s, i)
        if not x:
            x = list(range(len(y)))
        label = None if s.no_legend else (s.name or None)
        if s.fill_lo is not None and s.fill_hi is not None:
            n = min(len(x), len(s.fill_lo), len(s.fill_hi))
            target.fill_between(x[:n], list(s.fill_lo)[:n], list(s.fill_hi)[:n],
                                color=color if isinstance(color, str) else PALETTE[0],
                                alpha=0.12)
        if getattr(s, "area", False):
            n = len(y)
            base = [float(getattr(s, "fill_base", 0.0))] * n
            target.fill_between(x[:n], list(y)[:n], base,
                                color=color if isinstance(color, str) else PALETTE[0],
                                alpha=s.alpha * 0.35)
        if getattr(s, "as_bar", False):
            target.bar(x[:len(y)], y, color=color, alpha=s.alpha, label=label,
                       width=0.72)
            continue
        if not s.connect:
            target.scatter(x[: len(y)], y, s=16, alpha=s.alpha, color=color,
                           label=label)
            continue
        kw = {"lw": s.width, "color": color, "alpha": s.alpha, "label": label}
        if s.dash:
            kw["ls"] = "--"
        if s.marker:
            kw["marker"] = s.marker
            kw["ms"] = 3.2
        target.plot(x[: len(y)], y, **kw)


def _scatter(spec, ax) -> None:
    for i, s in enumerate(spec.series or ()):
        x, y = _series_xy(spec, s)
        if not x:
            x = list(range(len(y)))
        ax.scatter(x[: len(y)], y, s=14, alpha=0.55, color=_color_of(s, i),
                   label=None if s.no_legend else (s.name or None))


def _bar(spec, ax) -> None:
    """竖条形：`as_bar=True` 的 series 画条，其余仍按线画（累计曲线等）。"""
    for i, s in enumerate(spec.series or ()):
        x, y = _series_xy(spec, s)
        color = _color_of(s, i)
        label = None if s.no_legend else (s.name or None)
        if not x:
            x = list(range(1, len(y) + 1))
        if getattr(s, "as_bar", False):
            ax.bar(x[: len(y)], y, color=color, alpha=s.alpha, label=label,
                   width=0.72)
        else:
            ax.plot(x[: len(y)], y, lw=s.width, color=color, alpha=s.alpha,
                    marker=s.marker or None, ms=3.2, label=label)


def _hist(spec, ax) -> None:
    """直方图：每个 series 一组样本；`Spec.x` 是另一组（异常分数图用它画"全体"）。"""
    if spec.x:
        ax.hist(list(spec.x), bins=int(getattr(spec, "bins", 40) or 40),
                color=PALETTE[0], alpha=0.35, label=None)
    for i, s in enumerate(spec.series or ()):
        vals = [v for v in (s.y or ()) if v is not None]
        if not vals:
            continue
        ax.hist(vals, bins=int(getattr(spec, "bins", 40) or 40),
                color=_color_of(s, i), alpha=s.alpha,
                label=None if s.no_legend else (s.name or None))


def _barh(spec, ax) -> None:
    """横向条形：特征重要性 / 归因（支持**逐条颜色**：正负着色）。"""
    vals = [v for v in (spec.values or ())]
    rows = [str(r) for r in (spec.rows or ())]
    n = len(vals)
    if not n:
        return
    colors = list(spec.rowcolors or ())
    ax.barh(list(range(n)), [0.0 if v is None else v for v in vals],
            color=[colors[i] if i < len(colors) else PALETTE[2] for i in range(n)],
            alpha=0.85)
    ax.set_yticks(list(range(n)))
    ax.set_yticklabels(rows[:n] or [str(i) for i in range(n)])
    ax.set_ylim(-0.6, n - 0.4)


def _heatmap(spec, ax) -> None:
    vals = list(spec.values or ())
    if not vals:
        return
    im = ax.imshow(vals, cmap=spec.cmap or "Blues", aspect="auto")
    rows = [str(r) for r in (spec.rows or ())]
    k = len(vals[0]) if vals and isinstance(vals[0], (list, tuple)) else 0
    ax.set_xticks(range(k))
    ax.set_yticks(range(len(vals)))
    if rows:
        ax.set_yticklabels(rows)
    try:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    except Exception:
        pass


def _parallel(spec, ax) -> None:
    """平行坐标：x 是**等距分类轴**（参数名），每条 series 是一次评估的轨迹。

    ⚠ 这一支**不做数值 x 换算**（分类轴），与 `line` 不同 ⇒ 两个渲染器里都是独立分支。
    """
    n_axes = max([len(s.y or ()) for s in (spec.series or ())] + [0])
    if not n_axes:
        return
    x = list(range(n_axes))
    if spec.grid:
        for j in x:
            ax.axvline(j, color=_GRID, lw=0.8)
    for i, s in enumerate(spec.series or ()):
        y = list(s.y or ())
        ax.plot(x[: len(y)], y, lw=s.width, alpha=s.alpha, color=_color_of(s, i),
                label=None if s.no_legend else (s.name or None))
    labels = [str(t) for t in (spec.xticklabels or ())]
    ax.set_xticks(x)
    if labels:
        ax.set_xticklabels(labels[:n_axes], rotation=20, fontsize=9)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["min", "", "max"])
    ax.set_xlim(-0.5, n_axes - 0.5)


# ---------------------------------------------------------------- 版式
def _decorate(spec, ax) -> None:
    if spec.xlabel:
        ax.set_xlabel(spec.xlabel)
    if spec.ylabel:
        ax.set_ylabel(spec.ylabel)
    if spec.title:
        ax.set_title(spec.title)
    if spec.yscale in ("log", "symlog"):
        try:
            if spec.yscale == "symlog":
                ax.set_yscale("symlog", linthresh=float(spec.linthresh))
            else:
                ax.set_yscale("log")
        except Exception:
            pass
    if spec.xlim:
        ax.set_xlim(*spec.xlim)
    if spec.ylim:
        ax.set_ylim(*spec.ylim)
    for h in spec.hlines or ():
        ax.axhline(h.get("y"), color=h.get("color") or _AXIS, lw=0.8, alpha=0.8,
                   label=h.get("label") or None)
    for v in spec.vlines or ():
        ax.axvline(v.get("x"), color=v.get("color") or _AXIS, lw=0.8, alpha=0.7,
                   label=v.get("label") or None)
    for t in spec.texts or ():
        ax.text(t.get("x"), t.get("y"), str(t.get("text") or ""),
                ha="center", va="center", fontsize=9,
                color=t.get("color") or _INK)
    if spec.grid and spec.kind not in ("heatmap", "parallel"):
        ax.grid(alpha=0.25, color=_GRID, lw=0.6)
        ax.set_axisbelow(True)
    has_label = any(getattr(s, "name", "") and not getattr(s, "no_legend", False)
                    for s in (spec.series or ()))
    has_label = has_label or any(h.get("label") for h in (spec.hlines or ()))
    has_label = has_label or any(v.get("label") for v in (spec.vlines or ()))
    if spec.legend and has_label and spec.kind != "heatmap":
        ax.legend(fontsize=9)
