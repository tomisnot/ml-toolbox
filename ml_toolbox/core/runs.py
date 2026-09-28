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


def list_meta(root: str, marker: str) -> list[dict]:
    """列出 root 下含 marker 文件的子目录，按 saved_at 新→旧返回元数据 dict。"""
    if not os.path.isdir(root):
        return []
    out = []
    for rid in os.listdir(root):
        p = os.path.join(root, rid, marker)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    meta = json.load(f)
                meta.setdefault("run_id", rid)
                out.append(meta)
            except Exception:
                continue
    out.sort(key=lambda m: m.get("saved_at", ""), reverse=True)
    return out


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
