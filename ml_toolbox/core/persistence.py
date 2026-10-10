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
from . import runs as _runs

# 项目根 = ml_toolbox 的上一级；可用环境变量 ML_TOOLBOX_RUNS 覆盖
RUNS_DIR = _runs.resolve_dir("ML_TOOLBOX_RUNS")


def save_record(rec: RunRecord) -> str:
    d = os.path.join(RUNS_DIR, rec.run_id)
    os.makedirs(d, exist_ok=True)
    meta = {
        "run_id": rec.run_id,
        "method": rec.method, "family": rec.family, "task": rec.task,
        "target_kind": rec.target_kind,
        "dataset": rec.dataset, "pipeline_id": rec.pipeline_id,
        # ⭐ **内容身份**（`ds-<fp12>`）：跨会话/跨重命名唯一能对齐"这次跑的是哪份
        #   数据"的键（`dataset` 是显示名，会重复也会改）。老档没有这个键 ⇒
        #   `load_record` 读出来是空，如实。
        "dataset_id": getattr(rec, "dataset_id", ""),
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


def list_records(limit: int | None = None) -> list[dict]:
    """列出 ML 运行元数据（新→旧）。机械层见 core.runs（M3/C2）。

    ⚠ ``limit`` 给定时走**有限额路径**（只读够用的那些，见 ``core.runs.list_meta``）：
    大存档（实测数千个 ``record.json``）全量读要**十几秒**，而调用方常常只要最新几条。
    ``limit=None`` 与旧行为逐字相同。
    """
    return _runs.list_meta(RUNS_DIR, "record.json", limit)


# ---------------------------------------------------------------- 优化侧的读档槽
#: 优化层注册进来的 ``(list_records, load_record)``。
#:
#: ## 为什么是"槽"而不是 core 直接 import opt
#:
#: 边界层 `ml_mecha/` **不许** import `ml_toolbox.opt`（宪章纪律，判据
#: `test_ml_mecha.py::test_static_ml_mecha_does_not_import_el_modules` 静态守着，
#: 且用 `ast.walk` 查**所有**节点——静默 import 也算违规）。而 web 面板需要列/读优化运行。
#: ⇒ 让 **opt 自己注册**（依赖方向仍是 opt → core），core 只存两个 callable、不反向依赖。
#: 同款手法见 `runner.auto_pages` 用的 `registry.page_provider()` 槽。
#:
#: ⚠ 没注册时**诚实降级**（回空表/None），不假装有数据。
_OPT_SLOT: dict = {}


def register_opt_readers(list_records_fn, load_record_fn) -> None:
    """由优化层在 import 时调用（幂等）。"""
    _OPT_SLOT["list"] = list_records_fn
    _OPT_SLOT["load"] = load_record_fn


def opt_readers():
    """取优化侧的 ``(list_records, load_record)``；未注册回 ``(None, None)``。"""
    return _OPT_SLOT.get("list"), _OPT_SLOT.get("load")


def list_opt_records() -> list[dict]:
    """优化运行元数据（新→旧）。未注册 ⇒ ``[]``（**不是**"没有数据"，见 `opt_readers`）。"""
    fn = _OPT_SLOT.get("list")
    return list(fn()) if fn else []


def load_opt_record(run_id: str):
    """读一条优化运行；未注册 ⇒ ``None``。"""
    fn = _OPT_SLOT.get("load")
    return fn(run_id) if fn else None


#: 优化侧的"未声明页时的兜底页"提供者（`rec -> [PageSpec]`）。
#: 与上面两个读档槽同源：边界层要用，但不能 import opt 层 ⇒ 由 opt 注册、core 代转。
_OPT_PAGES: dict = {}


def register_opt_pages(fn) -> None:
    """由优化层在 import 时调用（幂等）。"""
    _OPT_PAGES["pages"] = fn


def opt_pages(rec) -> list:
    """未声明检视页时的兜底页；未注册 ⇒ ``[]``（诚实降级，不假装有页）。"""
    fn = _OPT_PAGES.get("pages")
    return list(fn(rec)) if fn else []


#: 优化侧的"页 key → 绘图规格函数"查表（`key -> fn|None`）。
#: 与上面几个槽同源：**面板要用，但边界层不许 import opt 层** ⇒ 由 opt 注册、core 代转。
_OPT_SPEC_FN: dict = {}


def register_opt_spec_lookup(fn) -> None:
    """由优化层在 import 时调用（幂等）。"""
    _OPT_SPEC_FN["lookup"] = fn


def opt_spec_for_key(key: str):
    """优化侧的规格函数查表；未注册 ⇒ ``None``（调用方回退 PNG）。"""
    fn = _OPT_SPEC_FN.get("lookup")
    try:
        return fn(key) if fn else None
    except Exception:
        return None


def load_run_record(run_id: str):
    """``record.json`` + ``artifacts.npz`` → 可检视的 ``RunRecord``（**无 estimator**）。

    ## 为什么住在 core（而不是原先的 `ui/history.py`）

    这是**纯数据访问**：把落盘的两份东西还原成内存里可用的对象
    （含 `a__b__c` 嵌套重组、`<key>_values/_names/_cols` → DataFrame 的通用还原）。
    它**没有一行 Qt** ⇒ 不该住在 GUI 层。

    ⚠ 实测后果：边界层 `ml_mecha/` **不许** import `ml_toolbox.ui`（宪章纪律，判据
    `test_ml_mecha.py::test_static_ml_mecha_does_not_import_el_modules` 守着），
    而 web 面板正需要读存档 ⇒ 它若留在 `ui` 里，面板要么违规、要么自己抄一遍
    （**两处实现 = 两处可以各自漂移**）。⇒ 下沉到 core，两侧共用这一份；
    `ui/history.py` 保留**同名转发**，GUI 既有调用点一行不改。
    """
    import os as _os

    import numpy as _np
    import pandas as _pd

    from .contracts import MLResult
    from .runner import RunRecord

    try:
        meta = load_record(run_id)
    except Exception:
        return None
    res = MLResult(method_name=meta.get("method", "?"),
                   task=meta.get("task", "?"),
                   target_kind=meta.get("target_kind"),
                   metrics={k: v for k, v in meta.get("metrics", {}).items()
                            if isinstance(v, (int, float))},
                   primary_metric=meta.get("primary_metric", ""),
                   params=meta.get("params", {}),
                   elapsed=meta.get("elapsed", 0.0),
                   error=meta.get("error"))
    ap = _os.path.join(RUNS_DIR, run_id, "artifacts.npz")
    if _os.path.exists(ap):
        with _np.load(ap, allow_pickle=False) as z:
            flat = {k: z[k] for k in z.files}
        # 嵌套 dict 工件（save_record 用 "__" 展平）：递归重组
        res.artifacts = {}
        for k, v in flat.items():
            if "__" not in k:
                res.artifacts[k] = v
                continue
            *path, leaf = k.split("__")
            node = res.artifacts
            for p in path:
                node = node.setdefault(p, {})
            node[leaf] = v
    # 通用还原：<key>_values + <key>_names(+<key>_cols) -> DataFrame(<key>)
    for k in list(res.artifacts):
        if k.endswith("_values") and k[:-7] + "_names" in res.artifacts:
            base = k[:-7]
            vals = _np.asarray(res.artifacts.pop(k), float)
            names = res.artifacts.pop(base + "_names")
            cols = res.artifacts.pop(base + "_cols", None)
            if cols is None:
                cols = _np.array(["importance"])
                vals = vals.ravel()
            res.artifacts[base] = _pd.DataFrame(
                vals.reshape(len(names), len(cols)),
                index=[str(s) for s in names], columns=[str(c) for c in cols])
    return RunRecord(run_id=run_id, method=res.method_name,
                     family=meta.get("family", "?"), task=res.task,
                     target_kind=res.target_kind,
                     dataset=meta.get("dataset", ""),
                     pipeline_id=meta.get("pipeline_id", ""),
                     dataset_id=meta.get("dataset_id", ""),
                     config=meta.get("config", {}), result=res)
