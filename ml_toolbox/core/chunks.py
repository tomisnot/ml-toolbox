# -*- coding: utf-8 -*-
"""**框架级分块**：大负载 + 小显存时的通用帮手（`ctx["chunks"]`）。

## 为什么这是框架的事（用户实测 §3.3）

使用者为了不 OOM，**在自己的网络代码里手写了分块**（按 R 切块、按块加权累加梯度）。
他的原话："这是通用需求（'大数据负载 + 小显存'），不该每个用户各写一遍。"

而"各写一遍"的真实代价不只是重复——**分块最容易写错的地方是累加口径**：
只取最后一块、或者把每块都当全量重复计数。⇒ 本模块把"切块 + 加权累加"
做成**正确到无法写错**的形态：

.. code-block:: python

    def metrics(self, model, ctx):
        # 全量对 R 行算一次代价太大 ⇒ 交给框架分块，它负责**加权求和**
        return ctx["chunks"].reduce(
            ctx["payload"], lambda blk: {"fid": _fidelity(model, blk)},
            mode="mean", budget_mb=256)

`mode="mean"` ⇒ 框架算 `Σ (n_block / n_total) * value`（这是"逐样本量的均值"的
正确口径）；`mode="sum"` ⇒ 直接 `Σ value`（"计数/求和型"量用这个）。
**不要**在返回里自己乘权重（那会算成两次）。

## 切块单位是**第 0 维**（= 行）

逐行负载的契约就是"第 0 维 = 观测表的行数"（见 `payload` 的说明）⇒ 切块只需切
每一张张量的第 0 维，并**一起切**（同一批行）——否则负载与观测表就错位了。

## 预算怎么变成"每块多少行"

按**实际字节数**估算而不是拍一个行数：`每行字节 = Σ(单行张量字节) + 观测表每行字节`，
再用 `budget_mb` 除出每块行数（至少 1 行）。⇒ 复数/大 L 的负载会自动切得更小，
而小负载不会被无谓地切碎。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

_MIB = 1024.0 * 1024.0

#: 默认预算（MiB）。可在 `prepare_dataset`/`run_custom` 的 `resource_guard` 里用
#: `chunk_budget_mb` 覆盖，或调用时逐次传 `budget_mb=`。
DEFAULT_BUDGET_MB = 256.0

#: 单块行数下限（预算再小也至少 1 行；0 行块会让"均值"变成 NaN）。
MIN_BLOCK_ROWS = 1


@dataclass(frozen=True)
class BlockPlan:
    """切块方案（**可读、可入回执**）：多少块、每块多少行、按什么预算算的。"""

    n_rows: int
    n_blocks: int
    rows_per_block: int
    budget_mb: float
    bytes_per_row: float

    def facts(self) -> dict[str, Any]:
        return {"n_rows": int(self.n_rows), "n_blocks": int(self.n_blocks),
                "rows_per_block": int(self.rows_per_block),
                "budget_mb": round(float(self.budget_mb), 3),
                "bytes_per_row": int(self.bytes_per_row),
                "semantics": "按第 0 维（行）切；同一批行一起切"}


def _row_bytes(obj: Any) -> int:
    """一张张量"每行"的字节数（取不到就算 0——估算宁小不大是**错的方向**，
    所以下面还会用 `n_rows` 兜底至少切一次）。"""
    try:
        shape = tuple(getattr(obj, "shape", ()) or ())
        if not shape:
            return 0
        n = int(shape[0]) or 1
        total = int(getattr(obj, "element_size", lambda: 8)()) * int(
            getattr(obj, "numel", lambda: 0)())
        return max(0, total // n)
    except Exception:                                          # noqa: BLE001
        return 0


class RowChunks:
    """`ctx["chunks"]` 的实现：**切块 + 加权累加**（见模块 docstring）。"""

    def __init__(self, *, budget_mb: float | None = None,
                 x_row_bytes: int = 0) -> None:
        self.budget_mb = float(budget_mb or DEFAULT_BUDGET_MB)
        #: 观测表每行的字节（调用方给；用于把预算折算成行数）
        self.x_row_bytes = int(x_row_bytes or 0)
        #: 上次实际用的方案（回执里报"已分 N 块"）
        self.last_plan: BlockPlan | None = None

    # ---------------------------------------------------------------- 计划
    def plan(self, rows: int, payload: Mapping[str, Any] | None = None,
             *, budget_mb: float | None = None) -> BlockPlan:
        """算出切块方案（**纯函数**，不碰张量内容）。"""
        n = max(0, int(rows))
        budget = float(budget_mb or self.budget_mb)
        per_row = int(self.x_row_bytes)
        for value in (payload or {}).values():
            per_row += _row_bytes(value)
        if per_row <= 0:
            per_row = 8          # 兜底：至少按 8 字节/行估，避免"除出无穷多行"
        rows_per_block = max(MIN_BLOCK_ROWS, int(budget * _MIB // per_row))
        n_blocks = max(1, (n + rows_per_block - 1) // rows_per_block) if n else 0
        return BlockPlan(n_rows=n, n_blocks=n_blocks,
                         rows_per_block=rows_per_block, budget_mb=budget,
                         bytes_per_row=per_row)

    def blocks(self, payload: Mapping[str, Any], *,
               rows: int | None = None,
               budget_mb: float | None = None) -> list[dict[str, Any]]:
        """切成若干块：每块是 `{名字: 该块的行切片}`（**同一批行**）。"""
        arrays = {k: v for k, v in (payload or {}).items() if hasattr(v, "__getitem__")}
        n = int(rows if rows is not None else
                (len(next(iter(arrays.values()))) if arrays else 0))
        plan = self.plan(n, arrays, budget_mb=budget_mb)
        self.last_plan = plan
        out: list[dict[str, Any]] = []
        for start in range(0, n, plan.rows_per_block):
            stop = min(n, start + plan.rows_per_block)
            out.append({k: v[start:stop] for k, v in arrays.items()})
        return out

    # ---------------------------------------------------------------- 累加
    def reduce(self, payload: Mapping[str, Any],
               fn: Callable[[Mapping[str, Any]], Mapping[str, float]],
               *, mode: str = "mean", rows: int | None = None,
               budget_mb: float | None = None) -> dict[str, float]:
        """分块跑 `fn` 并按 `mode` **正确累加**（用户不必自己写权重）。

        - ``mode="mean"``：``Σ (n_block / n_total) * value`` —— **逐样本量的均值**
          的正确口径（例如保真度、准确率：大块要占更大权重）；
        - ``mode="sum"``：``Σ value`` —— 计数/求和型量。

        ⚠ 两条纪律：
        1. `fn` **只算它拿到的那一块**，不许在里面再乘权重（框架负责加权，
           用户再乘一次就是"权重平方"，那是静默的数值错）；
        2. 返回值的键必须**每块一致**（第一块的键就是全集的键）；不一致 ⇒ 抛
           `ValueError`（宁可响亮失败，也不要"少了一个键"的静默结果）。
        """
        if mode not in ("mean", "sum"):
            raise ValueError(f"mode 只能是 'mean' 或 'sum'，收到 {mode!r}")
        arrays = {k: v for k, v in (payload or {}).items() if hasattr(v, "__getitem__")}
        n = int(rows if rows is not None else
                (len(next(iter(arrays.values()))) if arrays else 0))
        if n <= 0:
            return {}
        blocks = self.blocks(arrays, rows=n, budget_mb=budget_mb)
        acc: dict[str, float] = {}
        for blk in blocks:
            rows_in_block = len(next(iter(blk.values()))) if blk else 0
            if rows_in_block <= 0:
                continue
            got = dict(fn(blk) or {})
            if not acc:
                acc = {k: 0.0 for k in got}
            extra = set(got) - set(acc)
            missing = set(acc) - set(got)
            if extra or missing:
                raise ValueError(
                    "分块累加要求每块返回**同一组键**："
                    f"多出来 {sorted(extra)}，缺 {sorted(missing)}")
            weight = (rows_in_block / n) if mode == "mean" else 1.0
            for key, value in got.items():
                acc[key] = float(acc[key]) + weight * float(value)
        return acc

    # ---------------------------------------------------------------- 回执
    def facts(self) -> dict[str, Any]:
        """给回执用的**人话事实**（"已分 N 块"）；没跑过就回空。"""
        if self.last_plan is None:
            return {}
        return self.last_plan.facts()


__all__ = ["RowChunks", "BlockPlan", "DEFAULT_BUDGET_MB", "MIN_BLOCK_ROWS"]
