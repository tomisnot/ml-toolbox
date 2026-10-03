# -*- coding: utf-8 -*-
"""**中立绘图契约**（`PlotSpec`）：方法只吐"画什么"，渲染器各自决定"怎么画"。

## 为什么要有它（用户裁决的由来）

原先的接口是 ``plot(ax, result)``——**方法自己调 matplotlib**。
后果有两个，都是实测出来的：

1. **数据与图分不开**：方法里的 ``ax.plot(...)`` 把数据直接喂进 matplotlib 的画家，
   ⇒ 想换渲染器（web）就只能**栅格化**（PNG），于是"png 原图与页面风格格格不入"；
2. **同一份数据无法两处渲染**：GUI 用 Qt 画布、web 用网页，只能二选一或各画一套（会漂移）。

⇒ 用户裁决：**方法吐数据，GUI 与 web 各画各的**；并**接受两边统一新风格**
（即 GUI 的图也改由本契约的渲染器画，不再由方法直接调 matplotlib）。

## 契约形状（**只描述画什么**，不含任何 matplotlib / Qt 概念）

```python
PlotSpec(kind="line", x=[...], series=[Series(name, y, color, dash, marker)], ...)
```

- ``kind`` 决定必填字段；渲染器遇到不认识的 ``kind`` **必须响亮**（不能画空白）；
- ``x``/``y`` 一律是**可直接 JSON 序列化**的（渲染器可能在另一个进程/语言里）；
- ``rows`` 是类别标签（heatmap 的轴、barh 的条目名）；
- 轴/图例/网格这些"版式"由契约携带，**渲染器不再自己猜**（否则两处画出来不一致）。

## ⚠ 纪律

- **本模块零第三方依赖**（不 import numpy/matplotlib/pandas）⇒ 方法层与渲染层都能轻量引用；
- 数值一律 ``float``/``None``（``None`` = "这里没有点"，渲染器要断线，**不是 0**）；
- 新增 ``kind`` 时**必须同时**在两个渲染器里实现，否则会在判据里响
  （见 ``tests/test_plot_spec.py`` 的"每个 kind 都能渲染"）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

#: 已实现的图种（两个渲染器都必须支持；测试按这个表逐个验）。
KINDS = ("line", "scatter", "barh", "bar", "hist", "heatmap", "parallel")


@dataclass
class Series:
    """一条线/一组点/一组条。``name`` 是图例名（无图例可省）。"""

    name: str = ""
    y: list = field(default_factory=list)
    x: Optional[list] = None          # 省略则用 PlotSpec.x
    #: 颜色。**可以是单个色值，也可以是逐点色值数组**（散点按标签着色、条形按正负着色）。
    color: Any = ""
    dash: bool = False                # 虚线
    marker: str = ""                  # "o" / "s" / ""（不画点）
    width: float = 1.4
    alpha: float = 0.9
    #: 是否把点**连成线**。``connect=False`` 且 ``marker`` 非空 = 纯散点
    #: （GUI 原先"散点 + 拟合线"是两次调用；契约里合成一条 series，两个渲染器都按这两个开关画）
    connect: bool = True
    #: 图例里**不列**这条（比如 ROC 的 y=x 对角参考线：它在图上但不必占图例位）
    no_legend: bool = False
    fill_lo: Optional[list] = None    # 有值 ⇒ 画带（置信区间）：y±ci 的两条边界
    fill_hi: Optional[list] = None
    #: 用**条**而不是线来画这条 series（`bar`/`hist` 图种用；`bar` 里可再叠折线）
    as_bar: bool = False
    #: 面积填充：到 ``fill_base``（默认 0）之间的区域着色（轮廓图、残差带）
    area: bool = False
    fill_base: float = 0.0
    #: 画在**副轴**上（`nn_dyn` 的 lr / grad_norm 要 log 副轴）
    axis2: bool = False


@dataclass
class PlotSpec:
    """一张图的中立描述。渲染器（GUI / web）据此画出**同一张图**。"""

    kind: str
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    x: list = field(default_factory=list)
    series: list = field(default_factory=list)      # [Series]（line/scatter）
    # barh / heatmap
    rows: list = field(default_factory=list)        # 类别名
    values: list = field(default_factory=list)      # barh：数值；heatmap：二维
    # 通用版式
    legend: bool = True
    grid: bool = True
    xlim: Optional[list] = None
    ylim: Optional[list] = None
    yscale: str = ""                                # "log" / "symlog" / ""
    #: log 轴近零的线性邻域（`symlog` 用；收敛曲线的分数会跨很多量级）
    linthresh: float = 1e-3
    #: 分类 x 轴的刻度文字（平行坐标的参数名）；给了就按"等距分类"画，不做数值换算
    xticklabels: list = field(default_factory=list)
    #: 直方图分箱数（`hist` 图种）
    bins: int = 40
    #: 柱宽（`bar` 图种；`None` ⇒ 由渲染器按点数估）
    bar_width: Optional[float] = None
    #: 副轴的 y 范围（不给 ⇒ 按副轴数据自身极值）
    ylim2: Optional[list] = None
    #: 逐条颜色（barh / parallel 的每条）
    rowcolors: list = field(default_factory=list)
    #: 副轴标签（有 `Series.axis2` 时才用）
    ylabel2: str = ""
    yscale2: str = ""                               # 副轴的标度（`nn_dyn` 要 log）
    #: 横线标记 ``[{y, color, label}]``（残差的零线、归因的零线）
    hlines: list = field(default_factory=list)
    #: 竖线标记 ``[{x, color, label}]``（ROC 的对角线之外还用它画阈值线）
    vlines: list = field(default_factory=list)
    #: 文字标注 ``[{x, y, text}]``（热力图每格标数字、柱状图标值）
    texts: list = field(default_factory=list)
    #: 热力图色标名（渲染器各自映射；空 = 默认蓝）
    cmap: str = ""
    #: CSV 色标标签（`scatter2d` 的分数色条）
    colorbar_label: str = ""

    # ------------------------------------------------------------ 便捷构造
    def render_with(self, renderer):
        """交给一个渲染器（GUI 的 matplotlib 适配器 / web 的 SVG）画出来。"""
        return renderer(self)

    def to_dict(self) -> dict:
        """转成**可 JSON 序列化**的形状（渲染器可能在另一个进程里）。"""
        return {
            "kind": self.kind, "title": self.title,
            "xlabel": self.xlabel, "ylabel": self.ylabel,
            "x": _nums(self.x), "rows": [str(r) for r in self.rows],
            "values": _values(self.values),
            "series": [{
                "name": s.name, "y": _nums(s.y),
                "x": _nums(s.x) if s.x is not None else None,
                # ⚠ 颜色**两种形态**：单色（str）或逐点色数组（list）——都要能序列化
                "color": ([str(c) for c in s.color]
                          if isinstance(s.color, (list, tuple)) else str(s.color or "")),
                "dash": bool(s.dash), "marker": s.marker,
                "width": float(s.width), "alpha": float(s.alpha),
                "connect": bool(s.connect),
                "no_legend": bool(s.no_legend),
                "as_bar": bool(s.as_bar), "area": bool(s.area),
                "fill_base": float(s.fill_base), "axis2": bool(s.axis2),
                "fill_lo": _nums(s.fill_lo) if s.fill_lo is not None else None,
                "fill_hi": _nums(s.fill_hi) if s.fill_hi is not None else None,
            } for s in self.series],
            "legend": bool(self.legend), "grid": bool(self.grid),
            "xlim": _nums(self.xlim) if self.xlim is not None else None,
            "ylim": _nums(self.ylim) if self.ylim is not None else None,
            "yscale": self.yscale, "linthresh": float(self.linthresh),
            "xticklabels": [str(v) for v in self.xticklabels],
            "rowcolors": [str(c) for c in self.rowcolors],
            "ylabel2": self.ylabel2, "yscale2": self.yscale2,
            "hlines": [{"y": _num(v.get("y")), "color": str(v.get("color") or ""),
                        "label": str(v.get("label") or "")} for v in self.hlines],
            "vlines": [{"x": _num(v.get("x")), "color": str(v.get("color") or ""),
                        "label": str(v.get("label") or "")} for v in self.vlines],
            "texts": [{"x": _num(t.get("x")), "y": _num(t.get("y")),
                       "text": str(t.get("text") or ""),
                       "color": str(t.get("color") or "")} for t in self.texts],
            "cmap": self.cmap, "bins": int(self.bins),
            "colorbar_label": str(self.colorbar_label),
            "bar_width": (None if self.bar_width is None else float(self.bar_width)),
            "ylim2": _nums(self.ylim2) if self.ylim2 is not None else None,
        }

    # ------------------------------------------------------------ 常用版式（链式，读起来顺）
    def axes(self, xlabel: str = "", ylabel: str = "", title: str = "") -> "PlotSpec":
        self.xlabel, self.ylabel, self.title = xlabel, ylabel, title
        return self

    def no_legend(self) -> "PlotSpec":
        self.legend = False
        return self

    def no_grid(self) -> "PlotSpec":
        self.grid = False
        return self


def _num(v) -> Optional[float]:
    """单值：非有限/不可转 ⇒ ``None``（**不猜成 0**）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _nums(seq) -> list:
    return [_num(v) for v in (seq or ())]


def _values(obj) -> Any:
    """``values`` 允许一维（barh）或二维（heatmap）。"""
    if obj is None:
        return None
    first = next(iter(obj), None)
    if first is None:
        return []
    if isinstance(first, (list, tuple)):
        return [_nums(row) for row in obj]
    return _nums(obj)
