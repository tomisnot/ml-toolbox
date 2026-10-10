# -*- coding: utf-8 -*-
"""runs/ 目录的共享机械层（M3/C2：两份 persistence 的"目录+列举"重复收编）。

只收编**机械重复**（解析 runs 根、按标记文件列举元数据），不合并存储语义：
ML 存 record.json + artifacts.npz，优化存 opt_record.json + history.csv——
这个差异是有理由的（内部记录（未随仓发布，存档在仓外） §6）。

两个 runs 根各自独立（ML=ML_TOOLBOX_RUNS，opt=ML_TOOLBOX_OPT_RUNS），
本模块用 marker 文件名区分归属，list_all_records() 汇总供历史面板消费。
"""
from __future__ import annotations

import json
import os

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))          # ml_toolbox 的上一级 = 项目根


def resolve_dir(env: str, subdir: str = "runs") -> str:
    """runs 根目录：环境变量覆盖，否则项目根/subdir。"""
    return os.environ.get(env) or os.path.join(_PKG_ROOT, subdir)


#: ``list_meta(limit=…)`` 的**预筛倍数 / 下限**：先按目录 mtime 取最新
#: ``max(limit*倍数, 下限)`` 个候选，再读内容按 ``saved_at`` 排序截断。
#: ⚠ **为什么留余量**：目录 mtime 是"文件写完那一刻"，与记录里的 ``saved_at`` 同源但**不保证同序**
#: （同一秒内多次落盘、或有人拷/碰过文件）⇒ 余量让"有限额"的结果与"全量读完再截断"一致。
_LIST_META_MARGIN = 4
_LIST_META_FLOOR = 200


def _read_meta(path: str, rid: str) -> dict | None:
    """读一条元数据；**坏记录跳过**（旧行为：解析失败不拖垮整张列表）。"""
    try:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:                                  # noqa: BLE001
        return None
    meta.setdefault("run_id", rid)
    return meta


def _sort_key(meta: dict) -> tuple[str, str]:
    """排序键：``saved_at`` 新→旧，**同秒用 run_id 定序**。

    ⚠ **为什么必须带第二键**（2026-10-10 实测）：同一秒内落盘多条时，旧实现只按
    ``saved_at`` 排 ⇒ 并列者的先后取决于 ``os.listdir`` 的**任意次序**（"同一份数据两次读
    可能不同序"）。而"限额路径"按 mtime 预筛、并列次序又不同 ⇒ 两条路给出**同集合不同序**
    （判据里就红在这一处）。带上 ``run_id`` ⇒ 排序成为**全序且确定**，两条路逐条一致。
    """
    return (str(meta.get("saved_at", "")), str(meta.get("run_id", "")))


def list_meta(root: str, marker: str, limit: int | None = None) -> list[dict]:
    """列出 root 下含 marker 文件的子目录，按 saved_at 新→旧返回元数据 dict。

    ## ⭐ ``limit``：只读"够用的那些"（2026-10-10 实测抓到的真问题）

    真实仓里 ``runs/`` 有**数千个**子目录，而消费方（监控面板那一页）只要最新几十条；
    旧实现**把每一条都读一遍再切片** ⇒ 实测 **16.6 秒**（页面等于打不开）。
    ⇒ ``limit`` 给定时先按**目录 mtime** 预筛（``os.scandir`` + stat，实测 0.5 秒级），
    只读那些候选的内容。

    ⚠ **语义不变**：``limit=None`` 与旧行为逐字相同（全量读 + 排序）；
    有限额时返回的就是**全量结果的前 limit 条**（判据 ``tests/test_runs_limit.py`` 用合成存档钉住）。
    """
    if not os.path.isdir(root):
        return []
    if limit is None:
        out = [m for rid in os.listdir(root)
               if (m := _read_meta(os.path.join(root, rid, marker), rid)) is not None]
        out.sort(key=_sort_key, reverse=True)
        return out

    want = max(0, int(limit))
    if want == 0:
        return []
    keep = max(want * _LIST_META_MARGIN, _LIST_META_FLOOR)
    scored: list[tuple[float, str, str]] = []
    with os.scandir(root) as entries:
        for entry in entries:
            try:
                if not entry.is_dir():
                    continue
                path = os.path.join(entry.path, marker)
                if os.path.exists(path):
                    scored.append((entry.stat().st_mtime, entry.name, path))
            except OSError:
                continue                               # 竞态删除/不可读 ⇒ 跳过这一条
    scored.sort(reverse=True)
    out = [m for _mtime, rid, path in scored[:keep]
           if (m := _read_meta(path, rid)) is not None]
    out.sort(key=_sort_key, reverse=True)
    return out[:want]


def list_all_records(ml_root: str, opt_root: str) -> list[dict]:
    """ML + 优化两类运行记录合并 + 归一化（每条带 _kind 与统一展示列）。

    非 UI 逻辑放这里（分层纪律：UI 只渲染），可被 opt 侧单测直接覆盖。
    两类记录字段不同（ML: method/family/dataset/metrics[primary]；
    优化: optimizer/objective/best[score]），归一到同一套展示列。
    """
    rows = []
    for m in list_meta(ml_root, "record.json"):
        pm = m.get("primary_metric", "")
        val = m.get("metrics", {}).get(pm, "")
        rows.append({
            "_kind": "ML", "run_id": m.get("run_id", ""),
            "name": m.get("method", "?"), "family": m.get("family", "?"),
            "dataset": m.get("dataset", ""), "primary": pm, "value": val,
            "elapsed": m.get("elapsed", ""), "saved_at": m.get("saved_at", ""),
            "error": m.get("error"),
        })
    for m in list_meta(opt_root, "opt_record.json"):
        best = m.get("best") or {}
        val = best.get("score", "")
        rows.append({
            "_kind": "优化", "run_id": m.get("run_id", ""),
            "name": m.get("optimizer", "?"), "family": "opt",
            "dataset": m.get("objective", ""), "primary": "best", "value": val,
            "elapsed": m.get("elapsed", ""), "saved_at": m.get("saved_at", ""),
            "error": m.get("error"),
        })
    rows.sort(key=lambda m: m.get("saved_at", ""), reverse=True)
    return rows
