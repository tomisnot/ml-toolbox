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


def test_conflicting_archive_is_disclosed_not_hidden():
    """⭐ 档内自相矛盾（历史错标）⇒ **说出来**，而不是静默跳过或崩掉。

    为什么必须有：那批档（47/400）**改不了**（已落盘），展示层必须扛得住；
    而"按自述的 target_kind 挑页"会挑到混淆矩阵、画不出来 ⇒ 若静默跳过，
    人看到的是"少了一页"，永远不知道根因。⇒ 必须把矛盾摆在页面上。
    """
    conflict = None
    for r in W.list_runs("ml", limit=400):
        side, rec = W.load_any(r["run_id"])
        if rec is None:
            continue
        why = W.target_kind_mismatch(rec)
        if why:
            conflict = (r["run_id"], rec, side, why)
            break
    assert conflict, "找不到自相矛盾的档（前提变了？那本判据要重写）"
    rid, rec, side, why = conflict
    assert "target_kind" in why or "归档" in why, why

    # 该档的依赖页 ⇒ 页面说矛盾；且页面顶部也带警示
    pages = W.pages_for(side, rec)
    dep = next((p for p in pages if p[0] in W._KIND_DEPENDENT_PAGES), None)
    if dep is not None:
        html = W.page_body(side, rec, dep[0], dep[2], dep[3], rid)
        assert "归档" in html or "自述" in html, "矛盾没有被说出来"
    assert W.target_kind_mismatch(rec), "探测函数对同一条档应稳定"


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


# ============================================================ 直跑入口
_TESTS = [
    test_esc_and_numbers_never_guess,
    test_every_branch_returns_nonempty_readable_html,
    test_table_html_escapes_cell_content,
    test_unknown_kind_gets_placeholder_not_blank,
    test_runs_and_pages_for_all_three_views,
    test_opt_records_do_not_need_a_direct_opt_import,
    test_missing_opt_slot_degrades_honestly,
    test_pages_for_does_not_swallow_errors,
    test_render_plot_caches_and_returns_file,
    test_conflicting_archive_is_disclosed_not_hidden,
    test_plot_failure_returns_explanatory_png,
    test_server_serves_views_and_port_file,
    test_resolve_panel_base_rejects_stale_port_file,
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
