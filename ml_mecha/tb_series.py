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

#: `run_id` → (`EventAccumulator`, 事件目录 mtime 快照)。模块级缓存（单进程只读端点）。
_CACHE: dict[str, tuple[Any, float]] = {}


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


def _accumulator(run_id: str, root: Path | None = None):
    """取（或建）该 run 的 accumulator 并 ``Reload``（**读到最新**）。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    d = tb_dir(run_id, root)
    if d is None:
        return None
    key = str(d)
    acc, _ = _CACHE.get(key, (None, 0.0))
    if acc is None:
        acc = EventAccumulator(key)
        _CACHE[key] = (acc, 0.0)
    acc.Reload()                                    # ⭐ 增量：新写的事件这里读得到
    return acc


def _tail(values, limit: int = MAX_POINTS):
    return list(values)[-limit:]


def read_run(run_id: str, *, root: Path | None = None,
             limit: int = MAX_POINTS) -> dict:
    """读一个 run 的事件文件，翻成 wire 形状。

    @returns ``{available, reason?, run_id, scalars:{tag:{steps,values,latest}},
               histograms:[tag…], images:[tag…], text:[tag…]}``
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
    for tag in tags.get("scalars", []) or []:
        try:
            ev = acc.Scalars(tag)
        except Exception:
            continue
        pts = _tail(ev, limit)
        if not pts:
            continue
        scalars[tag] = {
            "steps": [int(p.step) for p in pts],
            "values": [float(p.value) for p in pts],
            "latest": {"step": int(pts[-1].step), "value": float(pts[-1].value)},
        }
    return {
        "available": True,
        "run_id": str(run_id),
        "dir": d.name,
        "scalars": scalars,
        "histograms": list(tags.get("histograms", []) or []),
        "images": list(tags.get("images", []) or []),
        "text": list(tags.get("tensors", []) or []),
    }


#: 一屏能懂的几条（顺序即重要性）：与 LightlyTrain 的"梯度范数是最有用信号"一致。
_LATEST_ORDER = ("loss", "val_loss", "grad_norm", "lr", "dead_relu")


def latest_of(payload: dict) -> dict:
    """从 :func:`read_run` 的结果抽"最新一轮"，按 ``_LATEST_ORDER`` 排；其余归 ``other``。"""
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
        out["grad_by_layer"] = per_layer
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
