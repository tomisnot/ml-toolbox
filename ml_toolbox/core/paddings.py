# -*- coding: utf-8 -*-
"""**变长 → 定长**：按行 pad + 生成 mask（逐行负载的入场券）。

## 为什么要有它（而不是让每个调用方自己 pad）

逐行负载（`aux["payload"]`）要求"每个数组第 0 维 = 行数"，于是**变长的那一维**
必须被 pad 到同一个长度，并配一个 mask 说明**哪些位置是真的**。

⚠ 这件事**极其容易做错**，而且错了以后很难发现：
- pad 用 0 填充：若真实数据里 0 是合法值（比如位串的 0），**没有 mask 就分不出**
  "这是填充"还是"这是真数据" ⇒ 损失会被算到填充上（**静默错**）；
- mask 的约定（True=有效 还是 True=填充）不统一 ⇒ 两个人写出相反的东西。

⇒ 收成一个函数：**pad + mask 一次产出**，并明确 `mask=True 表示"有效"`。
这条约定与 `Problem.masks()`（布尔掩码，True=在集合里）**同一口径**，别反。
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def pad_rows(rows: Sequence[Any], *, pad_value: float = 0.0,
             dtype: Any = None, mask_name: str = "") -> dict:
    """把"每行一个不等长数组"pad 成 ``(R, K, …)``，并给出 ``(R, K)`` 的 mask。

    @param rows 每行一个数组（列表里可以有 ``None`` 或空数组 = 该行没有内容）
    @param pad_value 填充值（**默认 0**；⚠ 若你的真实数据里 0 有物理含义，
        必须靠返回的 `mask` 区分——这正是本函数一定要给 mask 的原因）
    @param mask_name 给了就把 mask 一并放进返回值（键名 = 它）
    @returns ``{"values": (R,K,…) 数组, "mask": (R,K) bool(True=有效)[, mask_name: …]}``

    ⚠ 所有行都为空时，K=0：返回 ``(R, 0)`` 与 ``(R, 0)`` 而不是报错
    —— "这一份数据没有内容"是**事实**，该由调用方判断，不由本函数假装有。
    """
    arrs = [np.asarray(r) for r in rows if r is not None and np.asarray(r).size]
    if not arrs:
        dt = dtype if dtype is not None else float
        out = np.zeros((len(rows), 0), dtype=dt)
        m = np.zeros((len(rows), 0), dtype=bool)
        res = {"values": out, "mask": m}
        if mask_name:
            res[mask_name] = m
        return res
    widths = np.asarray([a.shape[0] for a in arrs])
    k = int(widths.max())
    tail = arrs[0].shape[1:]
    dt = dtype if dtype is not None else np.result_type(*[a.dtype for a in arrs])
    values = np.full((len(rows), k, *tail), pad_value, dtype=dt)
    mask = np.zeros((len(rows), k), dtype=bool)
    for i, r in enumerate(rows):
        if r is None:
            continue
        a = np.asarray(r)
        if a.size == 0:
            continue
        if a.shape[1:] != tail:
            raise ValueError(
                f"pad_rows: 第 {i} 行的尾维 {a.shape[1:]} 与其它行 {tail} 不一致"
                "（逐行负载要求除变长那维外形状相同）")
        values[i, :a.shape[0]] = a
        mask[i, :a.shape[0]] = True
    out = {"values": values, "mask": mask}
    if mask_name:
        out[mask_name] = mask
    return out


def pack_hdf5(path: str, datasets: dict, *, n_rows: int) -> dict:
    """从 HDF5 读逐行负载：``{负载名: "数据集路径"}`` → ``{负载名: 数组}``。

    ## 为什么 HDF5 是这条需求的主路径（而不是内联 JSON）

    NQS 用户的实测数据：974 个配置、N 到 18 ⇒ **2^18 = 26 万个复数/配置**
    ≈ 4 MiB（complex128）⇒ 全量 2–4 GB。JSON 内联不现实，而 HDF5 **原生支持
    复数**（内联 JSON 不能表达复数，那正是"复数被静默变成实数"的现场之一）。

    ⚠ 读进来的数组**保持 dtype**（复数就是复数、位串就是整数），**不做 float 强转**。
    ⚠ 每个数据集**第 0 维必须等于行数**：这是"切一批行 = 切每个数组前几维"的前提。
    """
    try:
        import h5py
    except ImportError as exc:                                 # pragma: no cover
        raise RuntimeError(
            "读 HDF5 需要 h5py（本机没装）：`pip install h5py`；"
            "或把数组导成 npy/csv 后用 `arrays` 内联给（不适用于复数）") from exc
    out: dict[str, Any] = {}
    with h5py.File(str(path), "r") as f:
        for name, h5path in dict(datasets or {}).items():
            if str(h5path) not in f:
                avail = []
                f.visit(lambda p: avail.append(p))
                raise ValueError(
                    f"HDF5 里没有数据集 {h5path!r}（可用：{avail[:12]}"
                    f"{'…' if len(avail) > 12 else ''}）")
            arr = np.asarray(f[str(h5path)])
            if arr.shape and arr.shape[0] != int(n_rows):
                raise ValueError(
                    f"负载 {name!r} 的第 0 维是 {arr.shape[0]}，而观测表有 "
                    f"{n_rows} 行 —— 逐行负载要求**第 0 维 = 行数**"
                    "（否则'切一批行'就对不上）。检查数据集是否按行排好、"
                    "或是否需要先转置。")
            out[str(name)] = arr
    return out
