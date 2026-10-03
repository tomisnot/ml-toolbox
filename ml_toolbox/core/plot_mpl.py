# -*- coding: utf-8 -*-
"""把 :class:`PlotSpec` 画成 matplotlib 图（**GUI 侧渲染器**）。

## 这层为什么存在（用户裁决的直接结果）

原先 ``plot(ax, result)`` 让**方法自己调 matplotlib** ⇒ 数据与图分不开、换不了渲染器。
现在方法只吐 ``PlotSpec``，**画法集中在这里**（GUI）与 ``panel_svg.py``（web）。
⇒ 同一份数据两处渲染，且**改版式只需改渲染器**（不必碰方法）。

## 风格：与 web 面板对齐（用户裁决"接受统一新风格"）

- 字体与页面同一族（Segoe UI / 微软雅黑）；中文不再变方框；
- **去掉上/右边框**、网格更淡、图例无边框；
- 背景透明（由调用方决定白底还是页面色）。
"""

from __future__ import annotations

#: 与 `ml_mecha/panel_view.py` 的页面风格对齐的配色（改这里 = 两处一起变）。
PALETTE = ("#0969da", "#cf222e", "#1a7f37", "#9a6700", "#8250df",
           "#0550ae", "#bc4c00", "#6639ba")

_GRID = "#d8dee4"
_AXIS = "#57606a"
_INK = "#1f2328"


def apply_page_style() -> None:
    """把页面风格写进 matplotlib 的 rcParams（**进程级、幂等**）。"""
    try:
        import matplotlib
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
        del matplotlib
    except Exception:
        pass


_CJK_DONE = False


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
    fn = {"line": _line, "scatter": _scatter, "barh": _barh,
          "heatmap": _heatmap}.get(kind)
    if fn is None:
        raise NotImplementedError(f"PlotSpec 渲染器不认识 kind={kind!r}")
    fn(spec, ax)
    _decorate(spec, ax)


# ---------------------------------------------------------------- 各图种
def _series_xy(spec, s):
    x = s.x if s.x is not None else spec.x
    return list(x or ()), list(s.y or ())


def _line(spec, ax) -> None:
    for i, s in enumerate(spec.series or ()):
        x, y = _series_xy(spec, s)
        color = s.color or PALETTE[i % len(PALETTE)]
        if s.fill_lo is not None and s.fill_hi is not None:
            _fill(spec, ax, x, s.fill_lo, s.fill_hi, color)
        if not x:                                    # 只有 y ⇒ 用序号当 x
            x = list(range(len(y)))
        label = None if s.no_legend else (s.name or None)
        if not s.connect:
            # 纯散点（`connect=False`）：线与点分开表达，避免"点之间被连起来"
            ax.scatter(x[: len(y)], y, s=12, alpha=s.alpha, color=color,
                       label=label)
            continue
        kw = {"lw": s.width, "color": color, "alpha": s.alpha, "label": label}
        if s.dash:
            kw["ls"] = "--"
        if s.marker:
            kw["marker"] = s.marker
            kw["ms"] = 3.2
        ax.plot(x[: len(y)], y, **kw)


def _scatter(spec, ax) -> None:
    for i, s in enumerate(spec.series or ()):
        x, y = _series_xy(spec, s)
        if not x:
            x = list(range(len(y)))
        ax.scatter(x[: len(y)], y, s=14, alpha=0.55,
                   color=s.color or PALETTE[i % len(PALETTE)],
                   label=s.name or None)


def _barh(spec, ax) -> None:
    vals = list(spec.values or ())
    rows = [str(r) for r in (spec.rows or ())]
    n = len(vals)
    pos = list(range(n))
    ax.barh(pos, vals, color=PALETTE[2], alpha=0.8)   # 与页面同一支绿
    ax.set_yticks(pos)
    ax.set_yticklabels(rows[:n] or [str(i) for i in pos])
    ax.set_ylim(-0.6, n - 0.4)


def _heatmap(spec, ax) -> None:
    vals = list(spec.values or ())
    if not vals:
        return
    cmap = {"Blues": "Blues", "": "Blues"}.get(spec.cmap or "", spec.cmap or "Blues")
    im = ax.imshow(vals, cmap=cmap, aspect="auto")
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


def _fill(spec, ax, x, lo, hi, color) -> None:
    if not x:
        x = list(range(len(lo)))
    n = min(len(x), len(lo), len(hi))
    ax.fill_between(x[:n], list(lo)[:n], list(hi)[:n], color=color, alpha=0.12)


# ---------------------------------------------------------------- 版式
def _decorate(spec, ax) -> None:
    if spec.xlabel:
        ax.set_xlabel(spec.xlabel)
    if spec.ylabel:
        ax.set_ylabel(spec.ylabel)
    if spec.title:
        ax.set_title(spec.title)
    if spec.yscale:
        try:
            ax.set_yscale(spec.yscale)
        except Exception:
            pass
    if spec.xlim:
        ax.set_xlim(*spec.xlim)
    if spec.ylim:
        ax.set_ylim(*spec.ylim)
    for v in spec.vlines or ():
        ax.axvline(v.get("x"), color=v.get("color") or _AXIS, lw=0.8, alpha=0.7,
                   label=v.get("label") or None)
    for t in spec.texts or ():
        ax.text(t.get("x"), t.get("y"), str(t.get("text") or ""),
                ha="center", va="center", fontsize=9,
                color=t.get("color") or _INK)
    if spec.grid and spec.kind != "heatmap":
        ax.grid(alpha=0.25, color=_GRID, lw=0.6)
        ax.set_axisbelow(True)
    has_label = any(getattr(s, "name", "") and not getattr(s, "no_legend", False)
                    for s in (spec.series or ())) or \
        any(v.get("label") for v in (spec.vlines or ()))
    if spec.legend and has_label and spec.kind != "heatmap":
        ax.legend(fontsize=9)
