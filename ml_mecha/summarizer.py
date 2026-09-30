# -*- coding: utf-8 -*-
"""ML 监控概括：人类可读摘要 + **可对账** ``Claim``（宪章 §8.3，硬纪律 5）。

两层信任（mecha ``monitor.py``）：

- **概括层**（本模块）：我们写给人和模型看的，可能出错，必须标明派生；
- **原始层**（``History``）：append-only，唯一权威；不一致时**原始赢**。

因此本模块的规则是机械的：

1. 凡是复述某个值的句子，都写成 ``Claim(value, target=<原始键>)``——
   ``target`` 是 History 快照里的**真实键名**（``current.method`` / 命令事件键），
   不是概括层自己造的名字；
2. 概括层**不猜**：要复述的键在原始史里不存在就不写这条 Claim，
   而不是编一个值出来（"没有数据"≠"值是 0"）；
3. 事件体只取紧凑标量（``run_id`` / ``method`` / ``metrics``），
   绝不把 ``y_pred`` / ``artifacts`` / 完整配置拖进概括。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mecha.history import Event, fold
from mecha.monitor import Claim

#: 域运行事件键（不是命令审计；命令审计是核心的 ``command.<name>``）。
DOMAIN_RUN_EVENT = "ml.run"
#: 兼容旧别名（本模块只在本文件内使用）。
COMMAND_EVENT_PREFIX = DOMAIN_RUN_EVENT

#: 状态键（概括层只复述这些；同名键在原始史里由 Gate 写入）。
STATE_KEY_DATASET = "current.dataset_id"
STATE_KEY_PIPELINE = "current.pipeline_id"
STATE_KEY_METHOD = "current.method"
STATE_KEY_SEED = "current.seed"
STATE_KEY_DEVICE = "current.device"
STATE_KEY_GUARD = "current.resource_guard"

#: 概括里最多列几条运行（更多条目走 run_id 引用，不撑大概括）。
MAX_LISTED_RUNS = 5


def ml_summarizer(events: list[Event]) -> Mapping[str, Any]:
    """把 ML 操作史概括成人话，并对关键值挂 ``Claim`` 供 Monitor 对账。

    返回的每个 Claim 的 ``target`` 都指向原始史里的键；Monitor 会逐条比对，
    不一致即把整份读数标成 ``disputed``（原始赢）。
    """
    raw = fold(events)
    # ⚠ 事件名在 `op`（新形状）：`ml.run` 是 history 事件，`target` 为空、不进快照。
    commands = [e for e in events if str(e.op).startswith(COMMAND_EVENT_PREFIX)]
    runs = [_compact_run(e) for e in commands if e.after]
    runs = [r for r in runs if r]

    methods = _unique(r["method"] for r in runs if r["method"])
    datasets = _unique(r["dataset_id"] for r in runs if r["dataset_id"])
    ok_runs = [r for r in runs if r["ok"]]
    failed = [r for r in runs if not r["ok"]]

    summary: dict[str, Any] = {
        "headline": _headline(raw, runs),
        "command_count": len(commands),
        "run_count": len(runs),
        "ok_count": len(ok_runs),
        "failed_count": len(failed),
        "methods_used": methods,
        "datasets_used": datasets,
        "recent_runs": [_public_run(r) for r in runs[-MAX_LISTED_RUNS:]],
        "notes": [],
    }
    if failed:
        summary["notes"].append(
            f"{len(failed)} 次运行未成功：{[r['method'] for r in failed]}")

    # ---- Claim：只在原始史里真有该键时复述（不猜、不补零）----
    if STATE_KEY_METHOD in raw and raw[STATE_KEY_METHOD]:
        summary["current_method"] = Claim(raw[STATE_KEY_METHOD],
                                          target=STATE_KEY_METHOD)
    if STATE_KEY_DATASET in raw and raw[STATE_KEY_DATASET]:
        summary["current_dataset"] = Claim(raw[STATE_KEY_DATASET],
                                           target=STATE_KEY_DATASET)
    if STATE_KEY_PIPELINE in raw and raw[STATE_KEY_PIPELINE]:
        summary["current_pipeline"] = Claim(raw[STATE_KEY_PIPELINE],
                                            target=STATE_KEY_PIPELINE)
    if STATE_KEY_SEED in raw:
        summary["current_seed"] = Claim(raw[STATE_KEY_SEED], target=STATE_KEY_SEED)
    if STATE_KEY_DEVICE in raw:
        summary["current_device"] = Claim(raw[STATE_KEY_DEVICE],
                                          target=STATE_KEY_DEVICE)
    if STATE_KEY_GUARD in raw:
        summary["current_resource_guard"] = Claim(raw[STATE_KEY_GUARD],
                                                  target=STATE_KEY_GUARD)
    # 最近一次成功运行的指标也是"复述"：把整条事件当成原始值对账
    if runs:
        last_key = None
        for e in reversed(commands):
            if e.after:
                last_key = e.op
                break
        if last_key is not None and last_key in raw:
            summary["latest_run"] = Claim(raw[last_key], target=last_key)
    if not runs:
        summary["notes"].append("本会话尚无 ML 运行记录")
    return summary


def _headline(raw: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]) -> str:
    """一句人话：数据 + 方法 + 主指标 + run_id（每个数字都能回原始史核）。

    优先讲**最近一次成功**的运行；全失败时如实讲失败（不挑好看的报）。
    """
    dataset = raw.get(STATE_KEY_DATASET) or "(未准备数据集)"
    method = raw.get(STATE_KEY_METHOD)
    if not runs:
        return f"ML 会话：当前数据集 {dataset}，尚无运行"
    ok_runs = [r for r in runs if r["ok"]]
    if not ok_runs:
        last = runs[-1]
        return (f"ML 会话：在 {dataset} 上运行 {last.get('method') or method} 失败，"
                f"run_id={last.get('run_id')}（共 {len(runs)} 次运行均未成功）")
    last = ok_runs[-1]
    metric = last.get("primary_metric") or ""
    score = (last.get("metrics") or {}).get(metric)
    tail = "" if len(ok_runs) == len(runs) else \
        f"（另有 {len(runs) - len(ok_runs)} 次失败运行）"
    return (f"ML 会话：在 {dataset} 上运行 {last.get('method') or method}，"
            f"{metric}={_fmt(score)}，run_id={last.get('run_id')}{tail}")


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4g}"
    return str(value)


def _compact_run(event: Event) -> dict[str, Any]:
    value = event.after
    if not isinstance(value, Mapping):
        return {}
    metrics = value.get("metrics")
    return {
        "run_id": str(value.get("run_id", "")),
        "method": str(value.get("method", "")),
        "dataset_id": str(value.get("dataset_id", "")),
        "pipeline_id": str(value.get("pipeline_id", "")),
        "ok": bool(value.get("ok", False)),
        "primary_metric": str(value.get("primary_metric", "")),
        "metrics": {str(k): v for k, v in dict(metrics).items()
                    if isinstance(v, (int, float))} if isinstance(metrics, Mapping) else {},
        "seq": event.seq,
        "actor": event.actor,
        "call_id": event.call_id,
    }


def _public_run(run: Mapping[str, Any]) -> dict[str, Any]:
    """概括里列出的运行条目：只留引用与标量指标。"""
    return {
        "run_id": run["run_id"],
        "method": run["method"],
        "dataset_id": run["dataset_id"],
        "ok": run["ok"],
        "primary_metric": run["primary_metric"],
        "score": run["metrics"].get(run["primary_metric"]),
        "actor": run["actor"],
        "call_id": run["call_id"],
    }


def _unique(values) -> list[str]:
    out: list[str] = []
    for v in values:
        if v and v not in out:
            out.append(str(v))
    return out


__all__ = ["ml_summarizer", "COMMAND_EVENT_PREFIX", "MAX_LISTED_RUNS"]
