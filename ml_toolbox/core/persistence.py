# -*- coding: utf-8 -*-
"""实验记录持久化：每次运行 = 数据快照 + 管道指纹 + 方法 + 参数 + 指标。

存储布局（工具箱根目录下 runs/，包外——包应保持纯净可分发）：
    runs/<run_id>/record.json     元数据 + 指标 + 参数 + 配置
    runs/<run_id>/artifacts.npz   数值契约数据（数组）
    runs/<run_id>/figure.png      核心图快照（UI 渲染时补写）
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

from .runner import RunRecord

# 项目根 = ml_toolbox 的上一级；可用环境变量 ML_TOOLBOX_RUNS 覆盖
_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
RUNS_DIR = os.environ.get("ML_TOOLBOX_RUNS", os.path.join(_root, "runs"))


def save_record(rec: RunRecord) -> str:
    d = os.path.join(RUNS_DIR, rec.run_id)
    os.makedirs(d, exist_ok=True)
    meta = {
        "run_id": rec.run_id,
        "method": rec.method, "family": rec.family, "task": rec.task,
        "target_kind": rec.target_kind,
        "dataset": rec.dataset, "pipeline_id": rec.pipeline_id,
        "config": rec.config,
        "metrics": {k: (v if isinstance(v, (int, float, str, list)) else str(v))
                    for k, v in rec.result.metrics.items()},
        "primary_metric": rec.result.primary_metric,
        "params": {k: (v if isinstance(v, (int, float, str, bool, list)) else str(v))
                   for k, v in rec.result.params.items()},
        "elapsed": round(rec.result.elapsed, 3),
        "error": rec.result.error,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(d, "record.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    art = {k: v for k, v in rec.result.artifacts.items()
           if isinstance(v, np.ndarray)}
    # 嵌套 dict-of-arrays（如 nn_weights / nn_history）：展平成 a__b__c 键存盘，
    # load_run_record 按 "__" 递归重组。方法层 artifacts 契约因此天然支持嵌套结构。
    def _flatten(prefix, obj):
        for k, v in obj.items():
            key = f"{prefix}__{k}"
            if isinstance(v, dict):
                _flatten(key, v)
            elif isinstance(v, np.ndarray):
                art[key] = v
    for k, v in rec.result.artifacts.items():
        if isinstance(v, dict):
            _flatten(k, v)
    # DataFrame 工件（如 feature_importance / nn_layers）拆成 值+行名(+列名) 数组存盘
    import pandas as pd
    for k, v in rec.result.artifacts.items():
        if isinstance(v, pd.DataFrame):
            art[k + "_values"] = v.to_numpy(float)
            art[k + "_names"] = np.array([str(i) for i in v.index])
            if v.shape[1] > 1:
                art[k + "_cols"] = np.array([str(c) for c in v.columns])
    if art:
        np.savez_compressed(os.path.join(d, "artifacts.npz"), **art)
    return d


def load_record(run_id: str) -> dict:
    with open(os.path.join(RUNS_DIR, run_id, "record.json"), encoding="utf-8") as f:
        return json.load(f)


def list_records() -> list[dict]:
    if not os.path.isdir(RUNS_DIR):
        return []
    out = []
    for rid in sorted(os.listdir(RUNS_DIR), reverse=True):
        p = os.path.join(RUNS_DIR, rid, "record.json")
        if os.path.exists(p):
            try:
                out.append(load_record(rid))
            except Exception:
                continue
    return out
