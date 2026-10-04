# -*- coding: utf-8 -*-
"""ML 面板（web viewer）的判据：**渲染层** + **服务层**。

## 判什么

1. **渲染层（纯函数）**：每条分支都返回**非空且可读**的 HTML；"读不到"与"确实没有"
   必须**两套文案**（这是本仓反复在守的纪律：空白页与假绿是同一类事故）；
2. **服务层**：三个并列 view 都能出页；`/plot` 真回 PNG；错误路径**响亮**；
   端口文件的写/清符合 `mecha.portfile` 的规则（收尾只删自己写的那个）。

## 不判什么

- **不判"页面好不好看"**——那只有人眼能判（我明确说过做不到）；
- **不判 TensorBoard 界面**——那是它的责任。

运行：``python tests/test_panel_web.py``（自带 main，与仓内其它 ``test_*.py`` 同形）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_mecha import panel_view as V                            # noqa: E402
from ml_mecha import panel_web as W                             # noqa: E402


# ============================================================ 渲染层（纯函数）
def test_esc_and_numbers_never_guess():
    """转义与数值：危险字符必须转义；非有限数**回 '—'**（与 0 分得开）。"""
    assert V.esc('<a href="x">&') == "&lt;a href=&quot;x&quot;&gt;&amp;"
    assert V.esc(None) == ""
    assert V._num(0) == "0"
    assert V._num(None) == "—"
    assert V._num(float("nan")) == "—"
    assert V._num(float("inf")) == "—"


def test_every_branch_returns_nonempty_readable_html():
    """⭐ 每条分支都非空、且**说得出发生了什么**（空串 = 那种假绿）。"""
    cases = {
        "空表": V.runs_pane([], view="ml"),
        "有表": V.runs_pane([{"run_id": "r1", "method": "ridge", "primary": "rmse",
                            "score": 0.5, "elapsed": 0.01, "saved_at": "t"}], view="ml"),
        "无表数据": V.table_html(None),
        "无诊断": V.text_html(None),
        "占位": V.placeholder("没东西", "原因在此"),
        "读不到": V.unreadable("run x", "存档打不开"),
        "状态条": V.views_bar("ml"),
        "徽标": V.badge("写权 AI"),
    }
    for name, html in cases.items():
        assert html and html.strip(), f"{name} 返回了空串"
        assert "<" in html, f"{name} 不像 HTML"
    # "确实没有" 与 "读不到" 必须是**不同**的文案
    assert V.EMPTY_TEXT not in V.UNREADABLE_PREFIX
    assert "读不到" in V.unreadable("x", "y")
    assert "确实没有" in V.placeholder(V.EMPTY_TEXT)


def test_table_html_escapes_cell_content():
    """表里的内容也要转义（渲染层不信任任何来源）。"""
    import pandas as pd
    df = pd.DataFrame({"<b>列</b>": ["<script>alert(1)</script>"]})
    html = V.table_html(df)
    assert "<script>" not in html, "表格没有转义（注入）"
    assert "&lt;script&gt;" in html


def test_unknown_kind_gets_placeholder_not_blank():
    """未知/未实现的页种 ⇒ **占位 + 原因**，不是空白。"""
    class P:
        key, title, kind, plot, data = "x", "某页", "pg", None, None
    html = W.page_body("ml", None, "x", "pg", P(), "r1")
    assert html.strip(), "未知页种返回空"
    assert "pyqtgraph" in html or "暂无渲染器" in html


# ============================================================ 服务层
def test_runs_and_pages_for_all_three_views():
    """三个并列 view 都能列出东西，且 run 至少能列出一页。

    ⚠ 先按 application 层的做法预载 opt 层：优化侧的读档能力是它 import 时**注册进
    core 的槽**的，而边界层不许直接 import 它（见 `preload_modules` 的说明）。
    """
    W.preload_modules(["ml_toolbox.opt.persistence"])
    views = {v: W.list_runs(v) for v, _ in V.VIEWS}
    total = sum(len(r) for r in views.values())
    assert total > 0, f"三个 view 一条 run 都列不出：{ {k: len(v) for k, v in views.items()} }"
    # 至少有两个 view 能给出"页"（否则 viewer 等于空壳）
    listed = 0
    for rows in views.values():
        for r in rows[:20]:
            side, rec = W.load_any(r["run_id"])
            if rec is None:
                continue
            if W.pages_for(side, rec):
                listed += 1
                break
    assert listed >= 2, f"能列出页的 view 太少（{listed}）⇒ 注册表没载入？"


def test_opt_records_do_not_need_a_direct_opt_import():
    """⭐ 优化侧经 **core 的槽** 取，边界层**不 import opt 层**（结构性纪律）。

    为什么单列一条：`ml_mecha/` 不许 import `ml_toolbox.opt`/`ml_toolbox.ui`
    （判据 `test_ml_mecha.py::...does_not_import_el_modules` 用 `ast.walk` 静态守着，
    **静默 import 也算违规**），而面板确实需要列/读优化运行 ⇒ 只能是"opt 注册、core 代转"。
    本条从**行为**侧守着同一件事：不 import opt，槽里照样有东西（预载之后）。
    """
    W.preload_modules(["ml_toolbox.opt.persistence"])
    from ml_toolbox.core import persistence
    lst, load = persistence.opt_readers()
    assert callable(lst) and callable(load), "opt 层没有把读档能力注册进 core 的槽"
    rows = persistence.list_opt_records()
    assert rows, "槽里注册了但列不出优化运行"
    rec = persistence.load_opt_record(rows[0]["run_id"])
    assert rec is not None and hasattr(rec, "history"), "读回的优化记录形状不对"


def test_missing_opt_slot_degrades_honestly():
    """⚠ 槽**没注册**时诚实降级（空表/None），**不是**假装有数据、也不是崩。"""
    from ml_toolbox.core import persistence
    saved = dict(persistence._OPT_SLOT)
    try:
        persistence._OPT_SLOT.clear()
        assert persistence.list_opt_records() == []
        assert persistence.load_opt_record("whatever") is None
    finally:
        persistence._OPT_SLOT.update(saved)


def test_pages_for_does_not_swallow_errors():
    """⚠ 取页失败**不静默**：`last_error` 要留下原因（否则表现为"每个 run 都没页"）。"""
    class Bad:
        method = "这个方法不存在"
    ps = W.pages_for("ml", Bad())
    assert ps == []
    assert W.pages_for.last_error, "失败被静默吞掉了（应当留下 last_error）"


def test_render_plot_caches_and_returns_file():
    """渲染一页 → 得到非空 PNG 文件；再取一次**命中缓存**（同一路径）。

    ⚠ 判据必须挑**与该方法 target_kind 匹配、且档内自洽**的页：
    历史上有 47 条档自述"分类"而指标是回归（归档 bug，见 `test_regressions.py` 里那条
    不变量），按自述去挑会挑到混淆矩阵，那**注定画不出来**——那不是 viewer 的 bug。
    这里显式跳过自相矛盾的档，测的是"正常档能渲染 + 能缓存"。
    """
    rows = W.list_runs("ml", limit=80)
    picked = None
    for r in rows:
        side, rec = W.load_any(r["run_id"])
        if rec is None or W.target_kind_mismatch(rec):
            continue
        for k, _t, kd, s in W.pages_for(side, rec):
            if kd == "mpl":
                picked = (r["run_id"], k, s, rec, side)
                break
        if picked:
            break
    assert picked, "找不到任何一个有 mpl 页且档内自洽的 run（判据无法成立，不是通过）"
    rid, key, spec, rec2, side2 = picked

    def _render():
        try:
            return W.render_plot(side2, rec2, spec, rid, key)
        except Exception as exc:
            raise AssertionError(f"渲染 {rid}/{key} 失败：{type(exc).__name__}: {exc}")

    p1 = _render()
    assert p1.is_file() and p1.stat().st_size > 0, "没渲染出 PNG"
    assert p1.read_bytes()[:4] == b"\x89PNG", "不是 PNG"
    p2 = W.render_plot(side2, rec2, spec, rid, key)
    assert p2 == p1, "第二次应命中同一缓存路径"
    try:
        p1.unlink()
    except OSError:
        pass


def test_plot_cache_is_invalidated_when_the_drawing_code_changes():
    """⭐ PNG 缓存**随画法来源的 mtime 失效**——不是"文件在就永远复用"。

    为什么必须有：`runs/<id>/web/<key>.png` 原先只判「在且非空」⇒ **永续复用**。
    而画法全在渲染器与两张页 key 表里（方法层只吐 `PlotSpec`）⇒ 绘图代码改了、
    旧 run 还显示旧画法，**页面上没有任何东西会告诉你**。今天刚做完 19/19 迁移，
    一旦有页回退到 PNG 路径，看到的就是陈旧画法。

    ⚠ 判据**不许**真去改仓内源码的 mtime（那会污染工作区、还会让别的判据飘）
    ⇒ 用一个 temp 探针文件冒充"来源"，并**明确自证扰动落上了**（两次取 mtime
    必须不能都等于 t0）——否则就是哑守卫：假如 `_plot_sources` 被换成空表，
    "陈旧那一路"根本不会触发，而断言照样绿。
    """
    import tempfile
    import time as _t

    rows = W.list_runs("ml", limit=80)
    picked = None
    for r in rows:
        side, rec = W.load_any(r["run_id"])
        if rec is None or W.target_kind_mismatch(rec):
            continue
        for k, _t2, kd, s in W.pages_for(side, rec):
            if kd == "mpl":
                picked = (r["run_id"], k, s, rec, side)
                break
        if picked:
            break
    assert picked, "找不到任何有 mpl 页且档内自洽的 run（前提变了，本判据要重写）"
    rid, key, spec, rec2, side2 = picked

    with tempfile.TemporaryDirectory() as td:
        probe = Path(td) / "fake_drawing_source.py"
        probe.write_text("# 冒充渲染器源码\n", encoding="utf-8")
        saved = W._plot_sources
        W._plot_sources = lambda: (probe,)
        try:
            p1 = W.render_plot(side2, rec2, spec, rid, key)
            assert p1.is_file() and p1.stat().st_size > 0, "没渲染出 PNG"
            assert not W._is_stale(p1), "刚画出来的图不该被当成陈旧"

            # ① 来源比图**新** ⇒ 陈旧（要重画），且自证扰动真的落上了
            t0 = _t.time()
            p1.touch()
            _t.sleep(0.02)
            probe.touch()
            assert probe.stat().st_mtime > t0 or p1.stat().st_mtime > t0, \
                "扰动没落上（mtime 精度不够？）⇒ 这条判据此刻是哑的"
            assert W._is_stale(p1), "画法来源更新后，旧 PNG 没有被判为陈旧"
            # ② 真走一遍 render_plot：陈旧就重画（新 mtime），新鲜就复用（同 mtime）
            m_old = p1.stat().st_mtime
            p2 = W.render_plot(side2, rec2, spec, rid, key)
            assert p2 == p1, "缓存路径应不变（只换内容）"
            assert p2.stat().st_mtime > m_old, "陈旧缓存没有重画（画法改了还显示旧图）"
            # ③ 再取一次 ⇒ 现在新鲜了，**不许**再重画
            p3 = W.render_plot(side2, rec2, spec, rid, key)
            assert p3.stat().st_mtime == p2.stat().st_mtime, "新鲜缓存被无谓重画了"
        finally:
            W._plot_sources = saved
            try:
                p1.unlink()
            except (OSError, UnboundLocalError):
                pass


def test_surrogate_page_is_honestly_png_only():
    """⭐ 代理切片页**不再"假迁移"**：显式走 PNG，并在页面上说清为什么。

    事实（2026-10-03 实测）：`opt/specs.SPEC_BY_KEY` 曾把 `surrogate` 收进表，
    但 `spec_surrogate_1d(bundle)` 吃的是 **六元组** `(xs, mu, sd, obs_x, obs_y, label)`
    ——那个 bundle 要**活的优化器实例**（`bo.py:239` 的 `self.surrogate_1d(...)`）
    才算得出来，而面板按统一约定喂 `OptRecord` ⇒ 每次 `TypeError`
    ⇒ 页面表现成"SVG 渲染失败 + 回退 PNG"，而表上看起来"已迁移"。

    ⇒ 本条同时钉两件（结构 + 页面），并**不重复** `test_plot_spec.py` 的覆盖面：
    那边只有**页 key（字符串）**、没有声明对象；本仓的教训是
    "surrogate 在表里"≠"面板能喂对参"，所以必须拿**真的 `PageSpec`** 走一遍。
    """
    from ml_toolbox.opt.engines.bo import GPBO
    from ml_toolbox.opt.specs import SPEC_BY_KEY, spec_for_key

    class _Rec:
        """给闭包用的最小壳：`inspect_pages` 只用到它把页面拼出来（不读数据）。"""

    pages = GPBO().inspect_pages(_Rec())
    sup = next((p for p in pages if getattr(p, "key", "") == "surrogate"), None)
    assert sup is not None, "优化器不再声明代理切片页了（前提变了，本判据要重写）"
    assert callable(getattr(sup, "plot", None)), \
        "代理切片页没有 plot（闭包）⇒ PNG 那条路也断了"

    # ① 表里那条映射**是**这个函数名（漂了就说明有人改了没同步本判据）
    assert SPEC_BY_KEY.get("surrogate") == "spec_surrogate_1d", "表与函数名对不上了"
    # ② 面板要显式知道它只能走 PNG
    assert "surrogate" in W._PNG_ONLY_PAGES, \
        "面板没把 surrogate 列为只能走 PNG ⇒ 又会去画那张注定失败的 SVG"
    # ③ 结构侧：喂 record 确实不行（这就是"只能走 PNG"的根因）——异常类型别放宽
    try:
        spec_for_key("surrogate")(_Rec())
        raise AssertionError("spec_surrogate_1d 居然吃得下 record？那本判据要重写")
    except TypeError:
        pass
    # ④ 真声明走一遍：**只给 PNG**，并在页面上说得出"为什么没有 SVG 适配器"
    html = W.page_body("opt", _Rec(), sup.key, sup.kind, sup, "r1")
    assert "<svg" not in html, "这一页还是想画 SVG（那张图必然失败）"
    assert "/plot?run=r1&amp;page=surrogate" in html, \
        f"没给 PNG 兜底图：{html[:200]}"
    assert "SVG 适配器" in html, "没有说明为什么这一页只能走 PNG（静默降级）"


def test_plot_sources_cover_the_renderers_and_both_key_tables():
    """`_plot_sources` 要**真的**覆盖到渲染器与两张页 key 表（不是空表/瞎指）。

    为什么单列：上面那条判据会**替换** `_plot_sources`。若它替换的是空表，
    "陈旧那一路"永远不触发而断言照样绿 ⇒ 必须另有一条钉住**真身**指向什么。
    """
    srcs = W._plot_sources()
    assert srcs, "取不到画法来源（缓存将永不失效）"
    assert all(p.is_file() for p in srcs), \
        f"来源里有不存在的文件（那一路会静默失效）：{[str(p) for p in srcs if not p.is_file()]}"
    # ⚠ 按**路径**断言，不按文件名：两张表都叫 specs.py，用名字集合会自相吞掉
    #   （实测踩过——`{p.name}` 里两个 specs.py 只剩一个，"两张表都在"这条就形同虚设）。
    tails = {"/".join(p.parts[-3:]) for p in srcs}
    for want in ("ml_mecha/panel_web.py", "ml_mecha/panel_svg.py",
                 "core/plot_mpl.py", "methods/specs.py", "opt/specs.py"):
        assert any(t.endswith(want) for t in tails), \
            f"少了画法来源 {want}（已有的：{sorted(tails)}）"


def test_conflicting_archive_is_disclosed_not_hidden():
    """⭐ 档内自相矛盾（历史错标）⇒ **说出来**，而不是静默跳过或崩掉。

    为什么必须有：那批档（实测前 400 条里 47 条）**改不了**（已落盘），展示层必须
    扛得住；而"按自述的 target_kind 挑页"会挑到混淆矩阵、画不出来 ⇒ 若静默跳过，
    人看到的是"少了一页"，永远不知道根因。⇒ 必须把矛盾摆在页面上。

    ⚠ **2026-10-03 修（本条原先依赖仓库历史）**：旧版在 `runs/` 里**翻找**一条自相
    矛盾的历史档当前提，找不到就红。清掉探针 run 之后它立刻红了 —— 报错正是它自己
    写下的那句"前提变了？那本判据要重写"。判据的前提必须**由判据自己构造**，
    不能靠"仓库里恰好还剩那种档"（换台机器、清一次 runs 就没了）。
    ⇒ 现在直接**造出那条矛盾档**：`target_kind` 标成"分类"而指标全是回归指标
    （这正是旧 `run_one` 的行为：存的是**数据集**的 kind，不是方法实际用的那个）。
    """
    from types import SimpleNamespace

    # ① 探测器：自述"分类" + 全是回归指标 ⇒ 必须回一句可读原因
    rec = SimpleNamespace(
        method="ridge", target_kind="classification", run_id="stub-conflict",
        result=SimpleNamespace(metrics={"rmse": 1.0, "mae": 0.5, "r2": 0.9}),
        artifacts={})
    why = W.target_kind_mismatch(rec)
    assert why, "自相矛盾的档没被探测出来"
    assert "归档" in why or "target_kind" in why, why
    # 同一条档两次问必须同答（探测器要稳定）
    assert W.target_kind_mismatch(rec) == why

    # ② 依赖 target_kind 的页 ⇒ 页面把**矛盾本身**说出来
    #   （`page_body` 在这条分支上只用得到 `spec.title`，所以桩够用 ——
    #    这也顺带钉住"矛盾检查发生在画图之前"这个顺序）
    key = next(iter(W._KIND_DEPENDENT_PAGES))

    class _Spec:
        title = "混淆矩阵"

    html = W.page_body("ml", rec, key, "mpl", _Spec(), rec.run_id)
    assert "画不出来" in html, html[:200]
    assert "归档" in html or "自述" in html, "矛盾没有被说出来"

    # ③ 顺带（**明确不是前提**）：仓库里若真有历史错标档，也要能探出来
    found = []
    for r in W.list_runs("ml", limit=400):
        _side, legacy = W.load_any(r["run_id"])
        if legacy is not None and W.target_kind_mismatch(legacy):
            found.append(r["run_id"])
            break
    print(f"    （顺带扫描：仓库里现存的历史错标档 {found or '无'}）")


def test_plot_failure_returns_explanatory_png():
    """图渲染失败 ⇒ 回**一张写明原因的图**（不是 404、不是空白×）。"""
    class Boom:
        key, title, kind = "boom", "坏页", "mpl"
        data = None
        @staticmethod
        def plot(ax, res):
            raise RuntimeError("故意炸")
    png = W.PanelWeb(port=0).render_error_png("r1", "boom", RuntimeError("故意炸"))
    assert png[:4] == b"\x89PNG", "失败时没回图"
    assert len(png) > 1000, "图太小，可能没有把原因画上去"


def test_server_serves_views_and_port_file(tmp_dir=None):
    """起真服务：三个 view 都 200、`/plot` 真 PNG、收尾按规则清端口文件。"""
    import json
    import urllib.request
    pf = Path(__file__).resolve().parent.parent / ".mlweb-port"
    pf.unlink(missing_ok=True)
    svc = W.PanelWeb(port=0, port_file=pf)
    port = svc.start()
    try:
        assert pf.is_file() and pf.read_text().strip() == str(port), "端口文件没写对"
        base = f"http://127.0.0.1:{port}"

        def get(p, raw=False):
            with urllib.request.urlopen(base + p, timeout=60) as r:
                b = r.read()
                return r.status, (b if raw else b.decode("utf-8", "replace"))

        for view in ("ml", "opt", "nn"):
            st, html = get(f"/?view={view}")
            assert st == 200 and len(html) > 500, f"view={view} 页面异常（{st}, {len(html)}）"
        st, body = get("/runs?view=ml")
        assert st == 200 and isinstance(json.loads(body).get("runs"), list)
        # 错误路径必须**响亮**（500 + 可读原因），而不是空白 200
        st, html = get("/plot?run=no-such-run&page=x", raw=True)
        assert html[:4] == b"\x89PNG", "不存在的 run 应回说明图"
        st, html = get("/?run=no-such-run")
        assert st == 200 and ("读不到" in html or "找不到" in html), \
            "不存在的 run 没有给出可读原因"
    finally:
        svc.stop()
    assert not pf.exists(), "收尾没清掉自己的端口文件"


def test_resolve_panel_base_rejects_stale_port_file():
    """⭐ 端口文件残留（指向**死进程**）时**认不出来** ⇒ 回 ``None``，不给坏地址。

    为什么必须有：进程被强杀（`terminate()`／任务被会话收掉）**不跑 `finally`**
    ⇒ 端口文件会留一个**没人听**的端口号。只读文件就会把浏览器开到那儿，
    现象正是"面板打不开／端口好像不对"。⇒ 读出来之后必须**探一下** `/health`。
    """
    pf = Path(__file__).resolve().parent.parent / ".mlweb-resolve-probe"
    try:
        # ① 文件不存在 ⇒ None
        pf.unlink(missing_ok=True)
        assert W.resolve_panel_base(pf) is None
        # ② 内容非法 ⇒ None（端口文件不校验合法性，非法一律按"不知道"）
        pf.write_text("not-a-port", encoding="utf-8")
        assert W.resolve_panel_base(pf) is None
        # ③ 指向没人听的端口 ⇒ None（**关键那条**：陈旧残留）
        import socket
        with socket.socket() as s:                    # 抢一个端口再立刻放掉 ⇒ 必无人听
            s.bind(("127.0.0.1", 0))
            dead = s.getsockname()[1]
        pf.write_text(str(dead), encoding="utf-8")
        assert W.resolve_panel_base(pf) is None, "陈旧端口文件被当成有效地址了"
        # ④ 真起服务 ⇒ 认得出，且地址与实际端口一致
        svc = W.PanelWeb(port=0, port_file=pf)
        port = svc.start()
        try:
            got = W.resolve_panel_base(pf)
            assert got == f"http://127.0.0.1:{port}/", f"解析出的地址不对：{got}"
        finally:
            svc.stop()
        assert W.resolve_panel_base(pf) is None, "收尾后不该还能解析出地址"
    finally:
        pf.unlink(missing_ok=True)


def test_runs_cache_is_fast_and_never_stale():
    """⭐ 运行列表有缓存（否则每次页签点击重读 1473 个 json），且**不会静默变陈旧**。

    为什么必须有：`list_records()` 要读 `runs/` 下每个 `record.json`
    （实测 1473 个 ⇒ **293–336 ms**），而 web 面板每次点页签都是
    **无状态的整页重载** ⇒ 不缓存就是"每点一下等一秒"。
    但缓存最危险的失效方式是**悄悄陈旧**（新跑完的 run 看不见）
    ⇒ 本条两件都钉：快（命中缓存）＋准（新 run 出现后**立刻**能被看到）。
    """
    import time
    W.list_runs("ml")                                   # 预热
    t0 = time.perf_counter()
    for _ in range(3):
        W.list_runs("ml")
    warm = (time.perf_counter() - t0) / 3 * 1000
    assert warm < 60, f"缓存没生效：命中一次仍要 {warm:.0f} ms"

    root = Path(W._runs_root())
    assert root.is_dir(), "取不到 runs/ 目录，判据无法成立"
    # 准：真加一个带 record.json 的 run ⇒ 必须出现（这条才是"不陈旧"的正面证据）
    real = root / "cache-probe-1-42"
    try:
        real.mkdir(exist_ok=True)
        (real / "record.json").write_text(
            '{"run_id": "cache-probe-1-42", "method": "ridge", "family": "linear",'
            ' "task": "supervised", "target_kind": "regression",'
            ' "metrics": {"rmse": 0.5}, "primary_metric": "rmse"}', encoding="utf-8")
        got = [r["run_id"] for r in W.list_runs("ml", limit=100000)]
        assert "cache-probe-1-42" in got, "新落盘的 run 没出现在列表里（缓存陈旧了）"
    finally:
        try:
            (real / "record.json").unlink(missing_ok=True)
            real.rmdir()
        except OSError:
            pass


def test_layout_follows_the_four_agreed_constraints():
    """⭐ 布局硬要求（用户逐条提的）—— 钉住，免得以后又长回去。

    1. 左栏可折叠，**且折叠后真的让出宽度**（右边必须占满腾出的空间）；
    2. **不要固定高度**：列表在**自己那格里滚**，整页不被顶长；
    3. 内容区**不套卡片框**；
    4. **不要"详情"二字**（谁都看得出来，那行是白占高度）。

    ⚠ 判据必须**先剥掉 CSS 注释与 `<style>`**：实测踩过——注释里写了 `max-height`
    与"详情"，直接搜字符串会**误报**。断言的是真声明，不是注释文字。

    ⚠ 折叠**不能用 `<details>`**（实测过：它的内容盒被浏览器包进**匿名 flex item**，
    那个 item 的 `min-height:auto` 会按内容长 ⇒ `min-height:0` / `overflow:hidden` /
    `height:100%` / 再加一层包裹 **全都无效**；量到 `detailsH=661` 而
    `listClientH=2820`，于是**整页被顶长**）。⇒ 显式结构 `aside > .dh + .scroll`。
    """
    import re
    html = V.page(title="t", view="ml", header_right="", body="<p>BODY</p>",
                  pages=[("fit", "拟合")], active="fit", run_id="r1",
                  runs_html=V.runs_pane([{"run_id": "r1", "method": "ridge"}],
                                        active="r1"))
    style = re.search(r"<style>(.*?)</style>", html, re.S)
    assert style, "页面里应该有样式块"
    css = re.sub(r"/\*.*?\*/", "", style.group(1), flags=re.S)
    flat = css.replace(" ", "").replace("\n", "")
    # ⚠ 剥两层注释再看"结构"：CSS 注释与 HTML 注释里都解释了"为什么不用 details / max-height"
    #   ⇒ 不剥就误报（**同一类坑踩了两次**，所以这里显式剥两层，且断言全部针对剥完的文本）。
    htmlish = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    plain = re.sub(r"/\*.*?\*/", "", re.sub(r"<style>.*?</style>", "", htmlish,
                                           flags=re.S), flags=re.S)

    # 1. 可折叠 + 折叠**真的让出宽度**
    assert "data-toggle=sidebar" in plain, "缺折叠开关"
    assert re.search(r"\.collapsed\s+main>aside\{[^}]*width:0", css), \
        "折叠没让出宽度：需要 `.collapsed main>aside{...width:0...}`"
    assert "flex:000" in flat, "折叠要整条重写 flex（`flex-basis:0` 压不过 `flex:0 0 300px`）"
    assert "<details" not in plain, \
        "左栏又用回 <details> 了 ⇒ 高度夹不住、整页会被顶长（实测过）"
    assert 'class="dh"' in plain and 'class="scroll runs"' in plain, \
        "左栏应是显式两层：`.dh`（表头）+ `.scroll`（列表）"
    # 2. 只限宽度、没有高度上限；列表自己滚
    assert "max-width" in css, "应该只限宽度"
    assert "max-height" not in css, "不该再有高度上限（会在下方留白）"
    assert re.search(r"flex\s*:\s*1\s+1\s+auto", css), "主区应靠 flex 撑满"
    assert "min-height:0" in flat, "min-height:0 是 flex 里正确滚动的前提"
    assert ".scroll{flex:11auto;min-height:0;overflow:auto}" in flat, \
        "列表那格必须自己滚（overflow:auto）"
    # 3. 内容区不套卡片框
    assert 'class="detail"' in html and 'class="pane' not in html, "内容区不该套卡片框"
    # 4. 不见"详情"二字
    assert "详情" not in plain, "页面上不该出现“详情”这两个字"
    # 折叠状态要跨导航保留（点页签是整页重载）
    assert "localStorage" in plain, "折叠状态没持久化（每次点页签都会弹回）"


# ============================================================ 直跑入口
#: ⭐ **内省收集**（2026-10-03 改）：此前这里是**手写**的判据清单，而门禁用
#: **脚本**方式跑本文件 ⇒ 新判据忘了加进来就**永远不会进闸**。实锤：
#: `test_ml_delivery.py::test_tool_params_stay_within_command_declaration`
#: （"工具声明与实现漂移"那条）从未在门禁里跑过。
#: ⇒ 定义即入闸，漂移不可能。（前提：本文件的 `test_*` 都是无参函数。）
_TESTS = [fn for name, fn in sorted(globals().items())
          if name.startswith("test_") and callable(fn)]


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
