# -*- coding: utf-8 -*-
"""读回 TensorBoard 事件文件，给面板与 AI 提供**同一份**数值。

## 为什么需要它（而不是让调用方自己解析）

事件文件是 TensorBoard 的格式；我们的面板走 JSON 只读路由、AI 走工具调用。
⇒ 两边都需要"**按轴翻译**"这一步：把 ``Scalars``/``Histograms``/``Images`` 翻成
我们的 wire 形状（``rows`` 给表格、``series`` 给折线、``latest`` 给一眼看数）。
⇒ 翻译只写一遍，**一份数据两种消费者**（同 `panel-data` 与 `/summary` 的既有分工）。

## ⚠ 两条纪律

1. **读不到就说读不到**：``available`` 为假时回原因（``reason``），**绝不回空数组假装**"跑到第 0 个 epoch"。
   （与 ML 既有的"空表 ≠ 没跑过"同一条：`archived_runs` 的 docstring 明写）
2. **缓存 accumulator**：实测每次新建 ~13.5ms、复用 ~5ms ⇒ 按 ``run_id`` 缓存；
   而 ``Reload()`` **能读到增量**（实测 3 个点 → 追加后读到 7 个）⇒ **直播靠它**。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

#: ``run_id`` 白名单：只允许字母数字与 ``-_.``，且不许以 ``.`` 开头。
#: ⚠ 防路径穿越（``run_id`` 会拼进文件路径）；**不做"猜"，直接拒绝**。
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")

#: 读回上限：每个 tag 最多回这么多点（事件文件可以很长，别整个倒出去）。
MAX_POINTS = 500

#: `run_id` → (`EventAccumulator`, 建它时的「事件文件集」签名)。模块级缓存（单进程只读端点）。
#: ⚠ 签名用来**发现目录里多/少了事件文件**：`EventAccumulator` 只盯**它创建时那个文件**，
#: 目录里后来新增的文件它不会读（实测：模拟"重启后续写"的第二个文件时，复用缓存读不到）。
#: ⇒ 文件集变了就**重建** accumulator。这是"跨进程续写同一个 run"能看见的前提。
_CACHE: dict[str, tuple[Any, tuple]] = {}


def _files_signature(d: Path) -> tuple:
    """事件目录的「文件集」签名（名字 + mtime + 大小）。文件集不变则复用缓存。"""
    out = []
    for f in sorted(d.glob("events*")):
        try:
            st = f.stat()
            out.append((f.name, int(st.st_mtime_ns), int(st.st_size)))
        except OSError:
            continue
    return tuple(out)


def runs_root() -> Path:
    """事件与小器的根目录（= 落盘 ``RUNS_DIR``）。**
    取不到就抛**（读不到不该被伪装成"没有数据"）。"""
    from ml_toolbox.core.persistence import RUNS_DIR
    return Path(RUNS_DIR)


def tb_dir(run_id: str, root: Path | None = None) -> Path | None:
    """``runs/<run_id>/tb`` 目录；``run_id`` 非法或目录不存在 ⇒ ``None``。"""
    if not _RUN_ID_RE.match(str(run_id or "")):
        return None
    d = (root or runs_root()) / str(run_id) / "tb"
    return d if d.is_dir() else None


def latest_run_id(root: Path | None = None) -> str:
    """**最近写过训练事件**的 run_id（读磁盘，按 ``tb/`` 目录的改动时间取最新）。

    ⚠ 为什么需要它（实测抓到的**描述与行为不符**）：`describe_training` 的工具描述
    写着"留空则按最近落盘的 run 找"，但引擎里 `run_id` 是**直接透传**的
    —— 留空 ⇒ `read_run("")` ⇒ 一律 `available=false`。于是那句话是**假话**，
    而 AI 会照着它省略参数、然后得到一个"读不到"的结论（最坏那类：把
    "我调用方式不对"读成"这次训练没有观测数据"）。
    ⇒ 要么实现它、要么删掉那句话；这里选**实现**（"最近的训练"是常见需求）。

    ⚠ 只在**真有 tb/ 目录**的 run 里挑：挑一个没有事件的 run 然后报"读不到"，
    等于把"历史里没有事件"伪装成"最近那次没事件"。没有可选项就回 ``""``
    （调用方据此给**明确**理由，而不是一个含糊的失败）。
    """
    base = root or runs_root()
    best, best_m = "", -1.0
    try:
        children = list(base.iterdir())
    except OSError:
        return ""
    for child in children:
        try:
            if not child.is_dir():
                continue
            d = child / "tb"
            if not d.is_dir():
                continue
            # 目录里得有真事件文件（空 tb/ 不算"有观测数据"）
            has_events = any(f.name.startswith("events.out.tfevents")
                             for f in d.iterdir() if f.is_file())
            if not has_events:
                continue
            m = d.stat().st_mtime
        except OSError:
            continue
        if m > best_m:
            best, best_m = child.name, m
    return best


def _accumulator(run_id: str, root: Path | None = None):
    """取（或建）该 run 的 accumulator 并 ``Reload``（**读到最新**）。

    ⚠ 两步各自解决一件事：
    1. **同文件内的增量**（训练逐 epoch 追加）⇒ 复用同一个 accumulator，``Reload()`` 就读得到；
    2. **目录里新增文件**（SummaryWriter 重启新建文件）⇒ 靠 ``_files_signature`` 发现并**重建**
       —— 因为 accumulator 不会自动去读它创建后新出现的文件（实测）。
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    d = tb_dir(run_id, root)
    if d is None:
        return None
    key = str(d)
    sig = _files_signature(d)
    acc, old_sig = _CACHE.get(key, (None, None))
    if acc is None or old_sig != sig:
        acc = EventAccumulator(key)                 # 文件集变了 ⇒ 重建（别复用旧的视野）
        _CACHE[key] = (acc, sig)
    acc.Reload()                                    # ⭐ 同文件增量：新写的事件这里读得到
    return acc


def _tail(values, limit: int = MAX_POINTS):
    return list(values)[-limit:]


#: 文本（TEXT 轴）单条上限：逐层健康表这类 markdown 表通常几 KB，留足余量。
#: ⚠ 必须**有界**：它是模型可见面的一部分，不能因为某个 run 写了一篇长文就把上下文挤爆。
TEXT_MAX_CHARS = 4000

#: 分布轴（Distributions）在事件文件里的元素是 **``(百分位, 值)`` 元组**，
#: ⚠ **不是**带 ``percentile``/``value`` 属性的对象（我第一版按属性读 ⇒ 静默回 0 帧，实测抓到）。
#: 元组里的百分位是"相对 10000 的整数"（如 500 = 5%）⇒ 统一除 100。
_DIST_ATTR_PERCENTILES = (0.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 100.0)


def _histo_proto(ev: Any) -> Any:
    """从直方图事件里取 protobuf（⚠ 跨版本：``histogram_value`` 与 ``histo`` 两种拼法）。"""
    return getattr(ev, "histogram_value", None) or getattr(ev, "histo", None)


def _read_histogram_values(acc: Any, tags: list[str]) -> dict[str, Any]:
    """直方图的**数值**：每个 tag 回末次的 ``min/max/num/sum`` + 写入次数。

    ## 为什么只回这四个数（而不是桶）
    框架写的是"**min/max/num/sum 型**"直方图：实测 ``bucket`` 里混着裸 ``float``
    （占位），真正的桶对象拿不到 ⇒ 回桶只能是空数组（那是**假装给了数据**）。
    ⇒ 回**确实读得到**的四个数；前端 Histograms 面板画柱状图也靠这四个数兜底。
    ⚠ 逐次演化**读不到**（框架只在终态写 1 次）⇒ 回 ``writes`` 让调用方知道有几帧。
    """
    out: dict[str, Any] = {}
    for tag in tags:
        try:
            evs = acc.Histograms(tag)
        except Exception:
            continue
        if not evs:
            continue
        last = evs[-1]
        hp = _histo_proto(last)
        if hp is None:
            continue
        out[str(tag)] = {
            "writes": len(evs),
            "step": int(getattr(last, "step", 0)),
            "min": float(getattr(hp, "min", 0.0)),
            "max": float(getattr(hp, "max", 0.0)),
            "num": float(getattr(hp, "num", 0.0)),
            "sum": float(getattr(hp, "sum", 0.0)),
        }
    return out


def _read_distribution_values(acc: Any, tags: list[str]) -> dict[str, Any]:
    """分布（分位数）数值：每个 tag 回末次的分位点。

    ⚠⚠ **两个实测的坑**（第一版两个都踩了 ⇒ 静默回 0 帧）：
    1. 元素是 ``(percentile, value)`` **元组**（10000 制：500 = 5%），不是带属性的对象；
    2. protobuf 字段名在两版里是 ``compressed_histogram_values`` / ``compressed_histogram_value``。
    ⇒ 两种字段名都试、两种元素形状都收（元组走下标，对象走属性），**不白名单过滤**
    （照实回它写了几个分位，缺信息的是调用方自己判断）。
    """
    out: dict[str, Any] = {}
    for tag in tags:
        try:
            evs = acc.CompressedHistograms(tag)
        except Exception:
            continue
        if not evs:
            continue
        last = evs[-1]
        raw = (getattr(last, "compressed_histogram_values", None)
               or getattr(last, "compressed_histogram_value", None) or [])
        pts: dict[str, float] = {}
        for p in raw:
            try:
                if isinstance(p, (tuple, list)) and len(p) >= 2:
                    pct, val = float(p[0]) / 100.0, float(p[1])
                elif hasattr(p, "percentile") and hasattr(p, "value"):
                    pct, val = float(p.percentile) / 100.0, float(p.value)
                else:
                    continue
            except Exception:
                continue
            pts["%g" % pct] = val
        if pts:
            out[str(tag)] = {"writes": len(evs), "step": int(getattr(last, "step", 0)),
                             "percentiles": pts}
    return out


def _read_text_values(acc: Any, tags: list[str]) -> dict[str, Any]:
    """文本（TEXT 轴）**正文**：解码 ``TensorProto``；超长**截断并留痕**。

    ⚠ 纪律：**截断必须说出来**（``truncated`` + ``chars_total``）——
    静默截断会让调用方以为"正文就这么长"。
    """
    out: dict[str, Any] = {}
    try:
        from tensorboard.util import tensor_util
    except Exception:
        return out
    for tag in tags:
        try:
            evs = acc.Tensors(tag)
        except Exception:
            continue
        if not evs:
            continue
        last = evs[-1]
        try:
            arr = tensor_util.make_ndarray(last.tensor_proto)
            raw = arr.item() if getattr(arr, "size", 0) == 1 else arr
            if isinstance(raw, bytes):
                text = raw.decode("utf-8", "replace")
            else:
                text = str(raw)
        except Exception as exc:
            out[str(tag)] = {"writes": len(evs), "step": int(getattr(last, "step", 0)),
                             "error": f"{type(exc).__name__}: {exc}"}
            continue
        ent = {"writes": len(evs), "step": int(getattr(last, "step", 0)),
               "chars_total": len(text)}
        if len(text) > TEXT_MAX_CHARS:
            ent["text"] = text[:TEXT_MAX_CHARS]
            ent["truncated"] = True
        else:
            ent["text"] = text
            ent["truncated"] = False
        out[str(tag)] = ent
    return out


def read_run(run_id: str, *, root: Path | None = None,
             limit: int = MAX_POINTS,
             with_values: bool = False) -> dict:
    """读一个 run 的事件文件，翻成 wire 形状。

    @param with_values
        ⭐ **是否把"值"也带出来**（默认 ``False`` = 只回 tag 名，与从前逐位一致）。
        为什么要有这个开关（2026-10-11，实测缺口）：
        `histograms` / `text` 原来**只回 tag 名**、`distributions` **连名字都不回**
        ⇒ AI 想回答"是哪一层/哪个参数出问题了"时只能**绕开工具面自己去解析事件文件**
        （我为此写过一个 `EventAccumulator` 脚本）。而 `layers/health/text_summary`
        里那张"每层 参数数/梯度范数/**权重范数**"表，正是**唯一**能指出
        "表示长在哪一层"的证据 ⇒ 它是**观测面缺口**，不是体积问题。
        ⚠ 开关默认关：正文是**大负载**（几 KB/条），别让默认回执无谓膨胀。

    @returns ``{available, reason?, run_id, scalars:{tag:{steps,values,latest}},
               histograms:[tag…], distributions:[tag…], images:[tag…], text:[tag…],
               histogram_values?:{tag:{min,max,num,sum,writes,step}},
               distribution_values?:{tag:{percentiles,writes,step}},
               text_values?:{tag:{text,chars_total,truncated,writes,step}}}``
    """
    d = tb_dir(run_id, root)
    if d is None:
        return {"available": False, "run_id": str(run_id),
                "reason": "该 run 没有 TensorBoard 事件目录（runs/<run_id>/tb 不存在）"}
    try:
        acc = _accumulator(run_id, root)
        tags = acc.Tags() if acc is not None else {}
    except Exception as exc:
        return {"available": False, "run_id": str(run_id),
                "reason": f"读事件文件失败：{type(exc).__name__}: {exc}"}

    scalars: dict[str, Any] = {}
    duplicated: list[str] = []
    for tag in tags.get("scalars", []) or []:
        try:
            ev = acc.Scalars(tag)
        except Exception:
            continue
        pts = _tail(ev, limit)
        if not pts:
            continue
        # ⚠ **同一步出现多次**：同一个 `tb/` 目录里堆了**多个事件文件**时（训练重跑没清目录、
        # 或 SummaryWriter 因重启新建文件），accumulator 会把两份都读出来 —— TensorBoard 自己
        # 也会把它画成两条重叠曲线。
        # ⇒ 按步去重（**后写的赢**，与 SummaryWriter 的语义一致），并**留痕**：
        #    静默去重会把"目录脏了"这件事藏起来，而那正是要让人看见的。
        by_step: dict[int, float] = {}
        order: list[int] = []
        for p in pts:
            st = int(p.step)
            if st not in by_step:
                order.append(st)
            by_step[st] = float(p.value)
        if len(order) != len(pts):
            duplicated.append(tag)
        scalars[tag] = {
            "steps": order,
            "values": [by_step[s] for s in order],
            "latest": {"step": order[-1], "value": by_step[order[-1]]},
        }
    out = {
        "available": True,
        "run_id": str(run_id),
        "dir": d.name,
        "scalars": scalars,
        "histograms": list(tags.get("histograms", []) or []),
        #: ⭐ **新增**：`distributions` 原来**连名字都没回**（`Tags()` 里一直有它，
        #:   只是没人读）⇒ 前端 Distributions 面板的数据在 AI 侧是**完全不可见**的。
        "distributions": list(tags.get("distributions", []) or []),
        "images": list(tags.get("images", []) or []),
        "text": list(tags.get("tensors", []) or []),
    }
    if with_values and acc is not None:
        #: ⚠ 值**按需**才读：直方图/分位/正文都要碰 protobuf，且正文可达数 KB
        out["histogram_values"] = _read_histogram_values(acc, out["histograms"])
        out["distribution_values"] = _read_distribution_values(acc, out["distributions"])
        out["text_values"] = _read_text_values(acc, out["text"])
    if duplicated:
        # 留痕（不静默）：调用方/AI 能看出"这个 run 的目录里有重复写入"
        out["duplicate_steps"] = sorted(duplicated)
    return out


#: 一屏能懂的几条（顺序即重要性）：与 LightlyTrain 的"梯度范数是最有用信号"一致。
_LATEST_ORDER = ("loss", "val_loss", "grad_norm", "lr", "dead_relu")


def latest_of(payload: dict, *, grad_layers: str = "summary") -> dict:
    """从 :func:`read_run` 的结果抽"最新一轮"，按 ``_LATEST_ORDER`` 排；其余归 ``other``。

    @param grad_layers ``summary``（默认）或 ``full``。
        ⚠ **默认摘要**（缺口 B，实测）：他想"只看 4 个指标"，而 `latest`（**必填键**）
        里带着 30 个 `grad_by_layer` —— `pick` 删得掉别的键，却**删不掉必填键内部的大块**
        ⇒ "只看 4 个数"要付 30 个梯度的钱。
        ⇒ 默认给"前 3 + 后 3 + 总数"，要全量写 `grad_layers="full"`。
        （不改成"默认不给"：逐层梯度是排查"某层学不动"的主要证据，全砍掉会逼人再拉一次。）
    """
    if not payload.get("available"):
        return {"available": False, "reason": payload.get("reason", "")}
    scal = payload.get("scalars") or {}
    out: dict[str, Any] = {"available": True, "run_id": payload.get("run_id")}
    ep = None
    for tag in _LATEST_ORDER:
        got = scal.get(tag)
        if got:
            out[tag] = got["latest"]["value"]
            if ep is None:
                ep = got["latest"]["step"]
    if ep is not None:
        out["step"] = ep
    # 逐层梯度：归成一个子表（`grad/层名`）
    per_layer = {t.split("/", 1)[1]: v["latest"]["value"]
                 for t, v in scal.items()
                 if t.startswith("grad/") and v.get("latest")}
    if per_layer:
        if str(grad_layers).lower() in ("full", "all") or len(per_layer) <= 8:
            out["grad_by_layer"] = per_layer
        else:
            items = list(per_layer.items())
            out["grad_by_layer"] = dict(items[:3] + items[-3:])
            out["grad_by_layer_n"] = len(per_layer)
            out["grad_by_layer_note"] = (
                f"共 {len(per_layer)} 层，这里只给**前 3 + 后 3**（省上下文）；"
                '要全量：describe_training(…, grad_layers="full")')
    other = {t: v["latest"]["value"] for t, v in scal.items()
             if t not in _LATEST_ORDER and not t.startswith("grad/")
             and v.get("latest")}
    if other:
        out["other"] = other
    return out


def rows_of(payload: dict) -> list[dict]:
    """把 ``scalars`` 摊成**表格行**（共享面板附加页的 ``rows`` 契约）。

    一行 = 一个 tag：``{tag, step, value, points}``。
    ⚠ 与 `/runs`、`/summary` 同形状 ⇒ 面板不必为它新写渲染。
    """
    if not payload.get("available"):
        return []
    rows = []
    for tag, got in sorted((payload.get("scalars") or {}).items()):
        latest = got.get("latest") or {}
        rows.append({"tag": tag, "step": latest.get("step"),
                     "value": latest.get("value"),
                     "points": len(got.get("values") or [])})
    return rows
