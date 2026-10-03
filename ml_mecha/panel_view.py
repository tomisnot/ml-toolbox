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
body{margin:0;font:13px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif;
     color:#1f2328;background:#f6f8fa}
header{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid #d0d7de;
       padding:8px 14px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.brand{font-weight:600}
.views a{padding:3px 10px;border:1px solid #d0d7de;border-radius:14px;
         text-decoration:none;color:#1f2328;background:#fff}
.views a.on{background:#0969da;color:#fff;border-color:#0969da}
.badge{padding:2px 8px;border-radius:10px;font-size:12px}
.badge.ok{background:#dafbe1;color:#1a7f37}
.badge.warn{background:#fff8c5;color:#7d4e00}
.badge.err{background:#ffebe9;color:#cf222e}
main{display:grid;grid-template-columns:300px 1fr;gap:12px;padding:12px;align-items:start}
.pane{background:#fff;border:1px solid #d0d7de;border-radius:8px}
.pane h2{margin:0;padding:8px 12px;font-size:12px;letter-spacing:.08em;
         text-transform:uppercase;color:#0969da;border-bottom:1px solid #d0d7de}
.runs{max-height:78vh;overflow:auto}
.runs a{display:block;padding:7px 12px;border-bottom:1px solid #eaeef2;
        text-decoration:none;color:#1f2328}
.runs a:hover{background:#f6f8fa}
.runs a.on{background:#ddf4ff}
.runs .m{font-weight:600}
.runs .s{color:#57606a;font-size:12px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;padding:8px 12px;border-bottom:1px solid #d0d7de}
.tabs a{padding:3px 10px;border:1px solid #d0d7de;border-radius:6px;
        text-decoration:none;color:#1f2328}
.tabs a.on{background:#0969da;color:#fff;border-color:#0969da}
.body{padding:12px}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #d0d7de;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#f6f8fa}
img.plot{max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff}
pre{margin:0;padding:10px;background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;
    overflow:auto;white-space:pre-wrap}
.muted{color:#57606a}
.ph{padding:14px;border:1px dashed #d0d7de;border-radius:6px;color:#57606a}
iframe{width:100%;height:78vh;border:1px solid #d0d7de;border-radius:6px;background:#fff}
.note{padding:8px 12px;color:#57606a;border-top:1px solid #d0d7de}
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
    """左栏：运行列表。``rows`` = ``[{run_id, method, primary, score, elapsed, saved_at}]``。

    ⚠ **空表要说明为什么空**（"确实没跑过" ≠ "读不到"），这句在 `<div class="note">` 里。
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
    inner = "".join(items) or f'<div class="note">{esc(EMPTY_TEXT)}</div>'
    return f'<section class="pane runs"><h2>运行列表（{len(list(rows or ()))}）</h2>{inner}</section>'


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
    """整页装配（列表 + 详情）。"""
    bar = (f'<header><span class="brand">ML 面板</span>'
           f"{views_bar(view, names=names)}{header_right}</header>")
    tabs = tabstrip(pages, active, view=view, run_id=run_id)
    detail = f'<section class="pane"><h2>详情</h2>{tabs}<div class="body">{body}</div></section>'
    return f"{head(title)}{bar}<main>{runs_html}{detail}</main></body></html>"
