# -*- coding: utf-8 -*-
"""ML 面板（web viewer）的 **HTML 渲染层**：纯函数，不碰 Qt、不碰网络。

## 为什么单独一层（照资产 `panel/` 的分层纪律）

呈现是**纯字符串生成** ⇒ 可以用 `node`/`python` 直测"**每条分支都返回非空且可读的 HTML**"，
而那正是本仓反复在防的那类假绿（面板空白、读不到被画成空表）。
⇒ 渲染与取数分开：本文件只管"给定数据 → HTML"，取数在 `viewer.py`。

## 布局参考 GUI（去掉方法选择）

GUI 是「左选方法 | 中看页 | 右看参数」，而人要的是**看跑过什么、以及那次跑出来什么** ⇒
本面板把"左"换成**运行列表**，"中"是**该 run 的检视页（PageSpec）**：

```
顶栏：模式徽标 · 三个并列 view（机器学习 / 优化调参 / 神经网络）· 刷新
运行列表        | 详情：页签 = 该 run 的 PageSpec 页
方法 · 主指标   |   mpl→PNG · table→HTML · text→文本 · pg→占位(见下)
耗时 · 时间     |   NN 的「训练过程」页 = TensorBoard iframe
```

⚠ **不做的**：方法选择（GUI 才有）、参数编辑、任何写操作 —— 本面板**只读**。
"""

from __future__ import annotations

from html import escape as _esc

#: 三个并列 view（对应 ML 内部的族，不是启动模式）。
VIEWS = (("ml", "机器学习"), ("opt", "优化调参"), ("nn", "神经网络"))

#: "读不到"与"确实没有"必须**两套文案**（同共享面板 `emptyText`/`unreadableText` 的纪律）。
EMPTY_TEXT = "这里还没有内容（确实没有数据，不是读取失败）"
UNREADABLE_PREFIX = "读不到（不是空的）"


def esc(v) -> str:
    """进 HTML 的文本一律转义（渲染层不信任任何来源）。"""
    return _esc("" if v is None else str(v))


def _num(v, digits: int = 4) -> str:
    """数值渲染；**非有限数不猜**（回 '—'，与 0 分得开）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if f != f or f in (float("inf"), float("-inf")):
        return "—"
    return f"{f:.{digits}g}"


STYLE = """
*{box-sizing:border-box}
/* ⚠ 高度必须"从顶到底每一层都夹住"，链条里漏一层，整页就会被内容顶长（实测过：
   `docScrollH=2896 / aside 列表的 clientHeight=2820` ⇒ 列表根本没被限制，整页跟着变长）。
   做法：`body` 定 100vh + `overflow:hidden`，`main` 与 `section.detail` 都 `min-height:0`
   + `overflow:hidden`，**只有真正要滚的那格**给 `overflow:auto`。 */
html,body{height:100%}
body{margin:0;display:flex;flex-direction:column;height:100vh;overflow:hidden;
     font:13px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif;
     color:#1f2328;background:#f6f8fa}
header{flex:0 0 auto;background:#fff;border-bottom:1px solid #d0d7de;
       padding:8px 14px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.brand{font-weight:600}
.views a{padding:3px 10px;border:1px solid #d0d7de;border-radius:14px;
         text-decoration:none;color:#1f2328;background:#fff}
.views a.on{background:#0969da;color:#fff;border-color:#0969da}
.badge{padding:2px 8px;border-radius:10px;font-size:12px}
.badge.ok{background:#dafbe1;color:#1a7f37}
.badge.warn{background:#fff8c5;color:#7d4e00}
.badge.err{background:#ffebe9;color:#cf222e}
/* 主区：横向两栏、占满剩余高度（**不给固定高度**，否则下方留白） */
main{flex:1 1 auto;min-height:0;overflow:hidden;display:flex;gap:0}
/* 左栏：可折叠的运行列表（只限宽度、高度随主区；折叠见 `.collapsed main>aside`） */
aside{flex:0 0 300px;max-width:46vw;display:flex;flex-direction:column;
      min-height:0;border-right:1px solid #d0d7de;background:#fff}
/* ⚠ 为什么**不用 `<details>`**（我在这里绕了三圈，实测数据在下面）：
   `<details>` 的可伸缩内容会被包进**匿名 flex item**，而那个 item 的 `min-height:auto`
   会按内容长 ⇒ `min-height:0`、`overflow:hidden`、`height:100%`、显式包裹层**全都无效**：
       detailsH=661  但  dbodyH=2820 / listClientH=2820（列表按内容长，整页跟着变长）
   ⇒ 改用**显式结构**：`aside > .dh（表头）+ .scroll（列表）`，折叠靠 body 上的类。
   这样每一层都是显式盒子、高度都能夹住。零脚本仍然成立（折叠只用 state class）。 */
aside{flex:0 0 300px;max-width:46vw;display:flex;flex-direction:column;
      min-height:0;overflow:hidden;border-right:1px solid #d0d7de;background:#fff}
aside .dh{flex:0 0 auto;padding:6px 10px;color:#57606a;font-size:12px;
          border-bottom:1px solid #eaeef2;background:#fff}
/* 列表自己滚（**不加 max-height**：高度由父容器给，不用魔法数字） */
/* 列表自己滚（**不加 max-height**：高度由父容器给，不用魔法数字）。
   ⚠ 它必须自带宽高约束：`.dbody` 给了明确高度，这里 `flex:1` 就能吃到"壳 − summary"。 */
aside .scroll{flex:1 1 auto;min-height:0;overflow:auto}
/* ⚠ 折叠三处要点（早先三条都错了，用户实测"折叠了宽度没让出来"）：
   1. `aside` 是 `main` 的**子元素**、`.collapsed` 加在 `body` 上
      ⇒ 选择器必须是 `.collapsed main>aside`（写成 `.collapsed aside` 根本匹配不到）；
   2. `flex-basis:0` **压不过** `flex:0 0 300px` 这个简写 ⇒ 要整条 `flex` 重写；
   3. 再加 `width:0` 兜底：`flex-basis` 在某些浏览器/嵌套 flex 下不收缩到底。 */
.collapsed main>aside{flex:0 0 0;width:0;max-width:0;overflow:hidden;border-right:0}
.runs a{display:block;padding:7px 12px;border-bottom:1px solid #eaeef2;
        text-decoration:none;color:#1f2328}
.runs a:hover{background:#f6f8fa}
.runs a.on{background:#ddf4ff}
.runs .m{font-weight:600}
.runs .s{color:#57606a;font-size:12px}
/* 详情：**没有框、没有标题**（页签已经说明在哪个页），直接吃满剩余空间 */
section.detail{flex:1 1 auto;min-width:0;display:flex;flex-direction:column;
               min-height:0;overflow:hidden;background:#fff}
.tabs{display:flex;gap:6px;flex-wrap:wrap;padding:8px 12px;
      border-bottom:1px solid #d0d7de;flex:0 0 auto}
.tabs a{padding:3px 10px;border:1px solid #d0d7de;border-radius:6px;
        text-decoration:none;color:#1f2328}
.tabs a.on{background:#0969da;color:#fff;border-color:#0969da}
.body{flex:1 1 auto;min-height:0;overflow:auto;padding:12px}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #d0d7de;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#f6f8fa}
img.plot{max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff}
/* SVG 图：**不加边框**（矢量图直接坐在页面上，与页签/表格同层）
   宽度上限放宽到 1400：折叠左栏后内容区变宽，图要**跟着占满**（用户明确要求）。
   ⚠ 图表本身有内在比例（viewBox 680×400），太宽会变扁 ⇒ 1400 是折中。 */
.chart{max-width:1400px}
.chart svg{display:block;width:100%;height:auto}
pre{margin:0;padding:10px;background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;
    overflow:auto;white-space:pre-wrap}
.muted{color:#57606a}
.ph{padding:14px;border:1px dashed #d0d7de;border-radius:6px;color:#57606a}
iframe{width:100%;height:100%;min-height:60vh;border:0;background:#fff}
.note{padding:8px 12px;color:#57606a}
a.btn{display:inline-block;padding:5px 12px;border:1px solid #0969da;border-radius:6px;
      background:#0969da;color:#fff;text-decoration:none}
a.btn:hover{background:#0550ae}
a.toggle{padding:3px 10px;border:1px solid #d0d7de;border-radius:6px;
         text-decoration:none;color:#1f2328;background:#fff;cursor:pointer}
"""


def head(title: str) -> str:
    return ("<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{esc(title)}</title><style>{STYLE}</style></head><body>")


def views_bar(active: str, *, names=None) -> str:
    """三个并列 view 的切换条（`:active` 为当前）。"""
    items = names or VIEWS
    links = "".join(
        f'<a class="{"on" if key == active else ""}" href="?view={esc(key)}">'
        f"{esc(label)}</a>" for key, label in items)
    return f'<span class="views">{links}</span>'


def badge(text: str, kind: str = "ok") -> str:
    """状态徽标（kind ∈ ok|warn|err）。"""
    k = kind if kind in ("ok", "warn", "err") else "warn"
    return f'<span class="badge {k}">{esc(text)}</span>'


def runs_pane(rows, *, active: str = "", view: str = "ml") -> str:
    """左栏（`<aside>`）：可折叠的运行列表，只限宽度、高度由父容器给。

    ## ⚠ 为什么**没有**用 `<details>`（本文件最反直觉的一处）

    最初用 `<details open>` 做"原生可折叠、零脚本"，但**实测它和 flex 高度约束互相打架**：
    `<details>` 的可伸缩内容会被浏览器包进**匿名 flex item**，那个 item 的 `min-height:auto`
    会按内容长 ⇒ 下面这些**全都无效**（逐个试过）：
    `min-height:0` · `overflow:hidden` · 给内容层 `height:100%` · 再加一层显式包裹 `.dbody`。
    量到的是：`detailsH=661` 而 `dbodyH=2820 / listClientH=2820`
    —— 列表按内容长，**整页跟着被顶长**（用户反馈"网页顶得超级长"）。

    ⇒ 改用**显式结构**：`aside > .dh（表头）+ .scroll（列表）`，折叠靠 `body.collapsed`。
    每一层都是显式盒子，高度就夹得住；且**仍然零脚本**（没有交互逻辑，只是样式切换）。

    ⚠ **折叠必须真的让出宽度**：光改 `flex-basis` 不够（压不过 `flex:0 0 300px` 简写），
    选择器也得写 `.collapsed main>aside`（`aside` 是 `main` 的子元素、类加在 `body` 上）。
    """
    items = []
    for r in rows or ():
        rid = str(r.get("run_id") or "")
        sel = " on" if rid == active else ""
        score = r.get("score")
        metric = r.get("primary") or ""
        extra = f"{esc(metric)}={_num(score)}" if metric or score is not None else "—"
        items.append(
            f'<a class="{sel.strip()}" href="?view={esc(view)}&run={esc(rid)}">'
            f'<div class="m">{esc(r.get("method") or "?")}</div>'
            f'<div class="s">{esc(rid)}</div>'
            f'<div class="s">{extra} · {_num(r.get("elapsed"), 3)}s · '
            f'{esc(r.get("saved_at") or "")}</div></a>')
    rows_n = len(list(rows or ()))
    inner = "".join(items) or f'<div class="note">{esc(EMPTY_TEXT)}</div>'
    # ⚠ 显式两层：`.dh`（表头，固定高）+ `.scroll`（列表，`flex:1` 吃掉余量并自己滚）。
    #   **不是 `<details>`**——见函数 docstring 里那份实测数据（匿名 flex item 会按内容长）。
    return (f'<aside><div class="dh">运行列表（{rows_n}）</div>'
            f'<div class="scroll runs">{inner}</div></aside>')


def tabstrip(pages, active_key: str, *, view: str, run_id: str) -> str:
    """详情页签（= 该 run 的 ``PageSpec`` 页；NN 的「训练过程」是一个额外页）。"""
    links = []
    for key, title in pages or ():
        sel = " on" if key == active_key else ""
        links.append(f'<a class="{sel.strip()}" href="?view={esc(view)}'
                     f'&run={esc(run_id)}&page={esc(key)}">{esc(title)}</a>')
    return f'<nav class="tabs">{"".join(links)}</nav>' if links else ""


def table_html(df) -> str:
    """DataFrame → HTML 表（**不引 pandas 的 to_html**，免得它自己塞样式与转义差异）。"""
    if df is None:
        return f'<div class="ph">{esc(EMPTY_TEXT)}</div>'
    try:
        cols = [str(c) for c in df.columns]
        idx = list(df.index)
        head_cells = "".join(f"<th>{esc(c)}</th>" for c in cols)
        body = []
        for i, (_, row) in enumerate(df.iterrows()):
            cells = "".join(f"<td>{_cell(row[c])}</td>" for c in df.columns)
            body.append(f"<tr><th>{esc(idx[i] if i < len(idx) else i)}</th>{cells}</tr>")
        return (f"<table><thead><tr><th></th>{head_cells}</tr></thead>"
                f"<tbody>{''.join(body)}</tbody></table>")
    except Exception as exc:                      # 渲染失败要**看得见**
        return f'<div class="ph">{esc(UNREADABLE_PREFIX)}：{esc(type(exc).__name__)}: {esc(exc)}</div>'


def _cell(v) -> str:
    try:
        import numpy as _np
        if isinstance(v, _np.ndarray):
            return esc(f"⟨数组 {v.shape}⟩")
    except Exception:
        pass
    try:
        import pandas as _pd
        if isinstance(v, _pd.DataFrame):
            return esc(f"⟨表 {v.shape}⟩")
        if v is None or (not isinstance(v, (str, bytes)) and _pd.isna(v)):
            return '<span class="muted">∅</span>'
    except Exception:
        if v is None:
            return '<span class="muted">∅</span>'
    if isinstance(v, float):
        return _num(v)
    return esc(v)


def text_html(diag) -> str:
    """``diag``（诊断 dict/list/文本）→ 可读块。"""
    if diag in (None, "", {}, []):
        return f'<div class="ph">{esc(EMPTY_TEXT)}</div>'
    if isinstance(diag, str):
        return f"<pre>{esc(diag)}</pre>"
    try:
        import json
        return f"<pre>{esc(json.dumps(diag, ensure_ascii=False, indent=2, default=str))}</pre>"
    except Exception:
        return f"<pre>{esc(diag)}</pre>"


def plot_html(src: str, *, title: str = "", note: str = "") -> str:
    """matplotlib 图（服务端已渲染成 PNG，经路由取回）。"""
    cap = f'<div class="muted">{esc(title)}</div>' if title else ""
    tail = f'<div class="note">{esc(note)}</div>' if note else ""
    return f'{cap}<img class="plot" src="{esc(src)}" alt="{esc(title)}">{tail}'


def placeholder(text: str, why: str = "") -> str:
    """占位：**必须说清为什么这里没东西**（不是空白）。"""
    detail = f'<div class="muted">{esc(why)}</div>' if why else ""
    return f'<div class="ph">{esc(text)}{detail}</div>'


def unreadable(what: str, reason: str) -> str:
    """读不到 ⇒ **响亮**（不是空表）；文案与"确实没有"分开。"""
    return (f'<div class="ph">{esc(UNREADABLE_PREFIX)} {esc(what)}'
            f'<div class="muted">{esc(reason)}</div></div>')


def page(*, title: str, view: str, header_right: str, body: str,
         pages=(), active: str = "", run_id: str = "", runs_html: str = "",
         names=None) -> str:
    """整页装配（左栏列表 + 右侧内容）。

    ⚠ 三条按用户要求的定稿：
    - **不写"详情"二字**：页签已经说明在哪个页，标题行是白占高度；
    - **不给内容加框**：左栏与内容区都不套卡片边框（`section.detail` 只有分隔线）；
    - **折叠开关放顶栏**（不是左栏里的第二个折叠钮）；左栏用**显式结构**
      `aside > .dh + .scroll`（**不是 `<details>`** —— 它的匿名 flex item 会让
      整页被顶长，实测过程见 `runs_pane` 的 docstring 与 `STYLE` 里的注释）。
    """
    bar = (f'<header><span class="brand">ML 面板</span>'
           f"{views_bar(view, names=names)}"
           f'<a class="toggle" href="#" data-toggle="sidebar">折叠列表</a>'
           f"{header_right}</header>")
    tabs = tabstrip(pages, active, view=view, run_id=run_id)
    detail = f'<section class="detail">{tabs}<div class="body">{body}</div></section>'
    # 折叠状态**跨导航保留**（点页签是整页重载，不存就只能每次重新折）。
    # 用一小段内联脚本：设置 body.collapsed + 存 localStorage；零依赖、失败不影响页面。
    script = (
        "<script>(function(){var b=document.body;"
        "try{if(localStorage.getItem('mlpanel.collapsed')==='1')"
        "b.classList.add('collapsed');}catch(e){}"
        "document.addEventListener('click',function(ev){"
        "var t=ev.target.closest('[data-toggle=sidebar]');if(!t)return;"
        "ev.preventDefault();b.classList.toggle('collapsed');"
        "try{localStorage.setItem('mlpanel.collapsed',"
        "b.classList.contains('collapsed')?'1':'0');}catch(e){}});})();</script>")
    return (f"{head(title)}{bar}<main>{runs_html}{detail}</main>{script}"
            "</body></html>")
