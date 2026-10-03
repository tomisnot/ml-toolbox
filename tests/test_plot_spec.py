# -*- coding: utf-8 -*-
"""中立绘图契约（`PlotSpec`）的判据：**契约本身 + 两个渲染器**。

## 判什么

1. **契约可序列化**（渲染器可能在另一个进程/语言里）且非有限数不猜（回 ``None``）；
2. ⭐ **两个渲染器对同一份 spec 都出图**，且**每个已实现 kind 都能画**
   （新增 kind 只实现一边 ⇒ 这里红）；
3. **未知 kind 响亮**（不画空白——空白页是本仓反复在防的假绿）；
4. **两处配色一致**（GUI 与 web 必须是同一支色，否则"统一风格"是空话）；
5. **渲染器不吃注入**（spec 里的字符串进 SVG 前要转义）。

## 不判什么

**不判"好不好看"** —— 那只有人眼能判（我明确说过做不到）。
本条判的是"数据进得去、图出得来、失败会响"。

运行：``python tests/test_plot_spec.py``（自带 main，与仓内其它 ``test_*.py`` 同形）。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_mecha import panel_svg                                    # noqa: E402
from ml_toolbox.core import plot_mpl                              # noqa: E402
from ml_toolbox.core.plot_spec import KINDS, PlotSpec, Series     # noqa: E402


def _sample(kind: str) -> PlotSpec:
    """每个 kind 的最小样例（两个渲染器都要能画）。"""
    if kind == "line":
        return PlotSpec(kind="line", x=[0, 1, 2, 3],
                        series=[Series(name="a", y=[1, 3, 2, 5], marker="o"),
                                Series(name="b", y=[0, 1, 1, 2], dash=True),
                                Series(name="", x=[0, 3], y=[0, 5], no_legend=True)]
                        ).axes("x", "y", "标题")
    if kind == "scatter":
        return PlotSpec(kind="scatter", x=[0, 1, 2],
                        series=[Series(name="s", y=[1, 2, 3])]).axes("x", "y")
    if kind == "barh":
        return PlotSpec(kind="barh", rows=["甲", "乙", "丙"],
                        values=[0.2, 0.9, 0.5]).axes(xlabel="importance")
    if kind == "heatmap":
        return PlotSpec(kind="heatmap", rows=["a", "b"],
                        values=[[3, 0], [1, 2]],
                        texts=[{"x": 0, "y": 0, "text": "3", "color": "#fff"}],
                        cmap="Blues").axes("预测", "真实")
    raise AssertionError(f"样例里没有 kind={kind!r}（新增 kind 时补上）")


# ============================================================ 契约
def test_to_dict_is_serializable_and_never_guesses_numbers():
    """⭐ 可 JSON 序列化（渲染器可能在别的进程/语言）；非有限数回 ``None``（**不猜成 0**）。"""
    import json
    spec = PlotSpec(kind="line", x=[0, 1, float("nan")],
                    series=[Series(name="a", y=[1.0, float("inf"), None])])
    d = spec.to_dict()
    json.dumps(d, ensure_ascii=False)              # 不抛即合格
    assert d["x"][2] is None, "NaN 应转成 None（断线），不是 0"
    assert d["series"][0]["y"][1] is None, "inf 应转成 None"
    assert d["series"][0]["y"][2] is None, "None 应保留为 None"


def test_every_kind_draws_in_both_renderers():
    """⭐ 每个已实现 kind 在**两个渲染器**里都能画出来（只实现一边 ⇒ 这里红）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for kind in KINDS:
        spec = _sample(kind)
        svg = panel_svg.svg(spec)
        assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>"), \
            f"{kind}: SVG 形状不对"
        assert len(svg) > 200, f"{kind}: SVG 太短（{len(svg)} 字节）——多半没画出东西"
        fig, ax = plt.subplots(figsize=(3, 2))
        try:
            plot_mpl.draw(spec, ax)                # 不抛即合格
        finally:
            plt.close(fig)


def test_unknown_kind_fails_loudly_in_both_renderers():
    """⚠ 未知 kind **响亮**（不画空白）——空白页是那种假绿。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bad = PlotSpec(kind="不存在的图种", series=[Series(name="a", y=[1, 2])])
    try:
        panel_svg.svg(bad)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("SVG 渲染器对未知 kind 没有响亮失败")
    fig, ax = plt.subplots(figsize=(2, 2))
    try:
        try:
            plot_mpl.draw(bad, ax)
        except NotImplementedError:
            pass
        else:
            raise AssertionError("matplotlib 渲染器对未知 kind 没有响亮失败")
    finally:
        plt.close(fig)


def test_renderer_palettes_match():
    """两处配色必须一致（否则"统一风格"是空话）。"""
    assert tuple(plot_mpl.PALETTE) == tuple(panel_svg.PALETTE), \
        f"配色漂移了：\n  mpl={plot_mpl.PALETTE}\n  svg={panel_svg.PALETTE}"


def test_svg_escapes_text():
    """spec 里的字符串进 SVG 前要转义（渲染层不信任任何来源）。"""
    spec = PlotSpec(kind="barh", rows=['<script>alert(1)</script>'],
                    values=[1.0], title='<img src=x onerror=y>')
    svg = panel_svg.svg(spec)
    assert "<script>" not in svg, "类别名没转义"
    assert "onerror" not in svg or "&lt;" in svg, "标题没转义"


# ============================================================ 已迁移的页
def test_placeholder_when_no_data():
    """没有可画的点 ⇒ 图上**写出原因**（不是空白 SVG）。"""
    svg = panel_svg.svg(PlotSpec(kind="line"))
    assert "没有可画" in svg, svg[:200]


def test_migrated_pages_are_registered_and_produce_specs():
    """已迁移的页在 `specs.SPEC_BY_KEY` 里登记，且真能从存档产出 spec。"""
    from ml_mecha import panel_web as W
    from ml_toolbox.methods import specs
    assert specs.SPEC_BY_KEY, "迁移表是空的"
    for key, fn_name in specs.SPEC_BY_KEY.items():
        assert callable(getattr(specs, fn_name, None)), f"{key} 指向的函数不存在：{fn_name}"
        assert specs.spec_for_key(key) is not None, f"{key} 查不到 spec 函数"
    assert specs.spec_for_key("没有这一页") is None, "未知页应回 None（调用方回退 PNG）"

    # 真跑一遍：拿存档里的页产出 spec，两个渲染器都要过
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    checked = 0
    for r in W.list_runs("ml", limit=300):
        side, rec = W.load_any(r["run_id"])
        if rec is None:
            continue
        for k, _t, kd, _s in W.pages_for(side, rec):
            fn = specs.spec_for_key(k)
            if fn is None:
                continue
            try:
                spec = fn(rec.result)
            except Exception:
                continue                            # 该方法的工件不含这张图要的键
            d = spec.to_dict()
            assert d["kind"] in KINDS, f"{k}: 产出了未登记的 kind {d['kind']!r}"
            assert panel_svg.svg(spec).startswith("<svg")
            fig, ax = plt.subplots(figsize=(3, 2))
            try:
                plot_mpl.draw(spec, ax)
            finally:
                plt.close(fig)
            checked += 1
            if checked >= 5:
                break
        if checked >= 5:
            break
    assert checked >= 1, "一个已迁移的页都没真跑起来（前置条件不成立，不是通过）"


# ============================================================ 直跑入口
_TESTS = [
    test_to_dict_is_serializable_and_never_guesses_numbers,
    test_every_kind_draws_in_both_renderers,
    test_unknown_kind_fails_loudly_in_both_renderers,
    test_renderer_palettes_match,
    test_svg_escapes_text,
    test_placeholder_when_no_data,
    test_migrated_pages_are_registered_and_produce_specs,
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    failures = []
    for fn in _TESTS:
        try:
            fn()
            print(f"  \u2713 {fn.__name__}")
        except Exception as exc:                       # noqa: BLE001
            failures.append((fn.__name__, exc))
            print(f"  \u2717 {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(_TESTS) - len(failures)}/{len(_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
