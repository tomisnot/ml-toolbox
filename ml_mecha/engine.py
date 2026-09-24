# -*- coding: utf-8 -*-
"""MLEngine：ML 宿主引擎的 mecha ``Engine`` 实现（薄适配，宪章 §11）。

职责边界（P1：宿主是唯一领域真值）：

- **ML 真值留在 ML**：``DataSpec`` / ``RunRecord`` / metrics / artifacts 住在
  ``ml_toolbox`` 与 ML 运行存档里；mecha 只拿到 ``run_id`` 与紧凑摘要。
- **mecha 只拿操作语义**：History 事件是"谁、经哪个通道、执行了哪条 ML 命令、
  拿到哪个 run_id、资源估算如何"，不是第二份实验记录。
- **本模块不 import EL**：命令名、状态键、方法名、指标名全是 ML 词汇。

``run()`` 是受控命令分派（不是"一次物理演化"的复刻）：命令面在
:mod:`ml_mecha.commands` 声明，回执**必填键在 ``schema()['run_output_required``
里显式写死（消 mecha D-15 的隐式契约键）。

状态写入（``current.*``）经注入的 ``state_writer`` 落到 mecha ``Gate``：引擎
自己没有快照、没有第二条写路径，也就无法绕过写权门（硬纪律 1）。

## Known Limitations and Deferred Work

- 命令面只覆盖宪章 §11.2 的"只读 + 单次执行 + 批量对比"三段；
  ``tune_method`` / ``run_benchmark`` / ``predict_new`` / ``export_prediction``
  按"最小可运行优先"显式推迟（未声明即 fail loud，不静默降级）。
- 数据集/管道/记录都**只在内存**（会话级）。跨进程找回要等 mecha 的
  Artifact / Job Contract 补齐（见汇报里的接口缺口清单）。
- ``device`` 只做声明与校验，不真的切设备：ML 方法层尚未消费该字段，
  本层不替方法层做决定（诚实推迟，不假装已生效）。
"""
from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mecha.errors import MechaError
from mecha.gate import Channel
from mecha.surface import Engine, ExecutionContext

from ml_toolbox.api import RunRequest, Session, SessionError
from ml_toolbox.core import registry
from ml_toolbox.core.contracts import DataSpec, is_lower_better
from ml_toolbox.core.dataset import Dataset
from ml_toolbox.core.pipeline import Pipeline
from ml_toolbox.core.resources import (ResourceBudget, ResourceEstimate,
                                       estimate_resources)

from .commands import COMMANDS, command_map
from .validator import (DeviceSpec, MLValidator, clean_device, clean_export_root,
                        clean_overrides, clean_resource_guard, clean_seed)

#: 回执必填键（消 D-15：隐式契约键显式化；``Surface.run`` 逐条校验）。
RECEIPT_REQUIRED_KEYS: tuple[str, ...] = (
    "ok", "command", "run_ids", "primary_metric", "metrics", "warnings",
)

#: 一次命令**最多**允许多少行数据进入运行（大表是资源风险，不是"能跑就行"）。
DEFAULT_MAX_DATASET_ROWS = 200_000

#: 保守吞吐：每秒可完成多少"操作单元"（``resources`` 里的 ops）。用于把
#: 内存/操作量估算折算成 ``est_sec``。刻意取小（宁可高估）——估时是防呆闸，
#: 不是性能预测；真实耗时以 RunRecord.timings 为准。
DEFAULT_OPS_PER_SEC = 2.0e7

#: 保护性上限：单条命令估时不超过 24 小时（超过即"该走 Job 而不是同步"）。
MAX_EST_SEC = 86_400.0

#: 回执/History 里最多列几个列名（宽表不截断 = 把表头塞进事件）。
MAX_LISTED_COLUMNS = 64

#: 当前执行通道（供工具层把 actor/side 带进来；未设置则写状态 fail closed）。
_EXEC_CHANNEL: contextvars.ContextVar[Channel | None] = contextvars.ContextVar(
    "ml_mecha_exec_channel", default=None)


class ChannelRequired(MechaError):
    """写域状态时没有可用通道——fail closed，绝不替调用方假定写权。"""


@dataclass(frozen=True)
class DatasetEntry:
    """一次 ``prepare_dataset`` 的结果（内存态；大对象只在这里，不进 History）。"""

    dataset_id: str
    name: str
    target: str | None
    time_col: str | None
    source: str
    n_rows: int
    n_features: int
    target_kind: str | None
    pipeline_id: str
    diag: bool
    spec: DataSpec = field(repr=False)
    fitted: Any = field(repr=False, default=None)

    def facts(self) -> dict[str, Any]:
        """给校验器/只读工具用的**紧凑**事实面（无数组、无大对象）。

        列名按 ``MAX_LISTED_COLUMNS`` 截断：这个 dict 会随回执进 History，
        宽表（几千列）不截断就等于把表头整个塞进事件（宪章 §8 明令禁止）。
        """
        columns = [str(c) for c in self.spec.X.columns]
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "target": self.target,
            "time_col": self.time_col,
            "source": self.source,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "target_kind": self.target_kind,
            "target_kinds": kinds_of(self.target_kind),
            "y_available": self.spec.y is not None,
            "pipeline_id": self.pipeline_id,
            "columns": columns[:MAX_LISTED_COLUMNS],
            "columns_truncated": max(0, len(columns) - MAX_LISTED_COLUMNS),
        }


@dataclass(frozen=True)
class RunSummary:
    """一次 ML 运行的紧凑投影（回执、History 事件、Monitor 对账共用同一份）。"""

    run_id: str
    method: str
    dataset_id: str
    pipeline_id: str
    ok: bool
    primary_metric: str
    metrics: dict[str, float]
    elapsed_s: float
    warning: str = ""

    def compact(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "method": self.method,
            "dataset_id": self.dataset_id,
            "pipeline_id": self.pipeline_id,
            "ok": self.ok,
            "primary_metric": self.primary_metric,
            "metrics": dict(self.metrics),
            "elapsed_s": self.elapsed_s,
            "warning": self.warning,
        }


def use_channel(channel: Channel):
    """把当前操作者通道绑进执行上下文（工具层/宿主适配器调用）。

    返回 contextvars token 也可，但为薄起见只提供"设置 + 重置"两个函数：
    :func:`use_channel` / :func:`clear_channel`。AI 侧通道 ``Channel('ai','ai')``
    拿不到 human 的闸——side 在构造时钉死（mecha 硬纪律 4）。
    """
    return _EXEC_CHANNEL.set(channel)


def clear_channel(token: Any) -> None:
    _EXEC_CHANNEL.reset(token)


def active_channel() -> Channel | None:
    return _EXEC_CHANNEL.get()


def make_state_writer(gate: Any) -> Callable[[str, Mapping[str, Any], str], list[Any]]:
    """构造"把 ML 状态键经 Gate 落地"的写口。

    这是 adapter 里**唯一**碰 ``Gate`` 的地方：引擎只调这个回调，自己不持有
    快照。写权检查、校验、History 追加都在 Gate 内部四步里完成。

    通道取自执行上下文（:func:`use_channel`），**没有显式通道就 fail closed**：
    不提供"默认人类通道"这种便利——那会让 AI 路径悄悄以人类身份写。
    """

    def write(reason: str, updates: Mapping[str, Any], call_id: str = "") -> list[Any]:
        channel = active_channel()
        if channel is None:
            raise ChannelRequired(
                "写 ML 状态时没有绑定操作者通道",
                kind="missing_channel",
                hint="经 ml_mecha.engine.use_channel(Channel(...)) 绑定操作者通道；"
                     "工具投影已自动绑定，直接调引擎的宿主需自己绑",
                suggest="with use_channel(channel): engine.run(spec)",
            )
        return list(gate.set_batch(channel, dict(updates), reason, call_id))

    return write


class MLEngine(Engine):
    """ML 宿主的 Engine Provider：受控命令分派 + 紧凑回执 + 资源估算。"""

    def __init__(self, *, session: Session | None = None,
                 state_writer: Callable[[str, Mapping[str, Any], str], list[Any]] | None = None,
                 dataset_roots: Sequence[str] | None = None,
                 allowed_export_roots: Sequence[str] | None = None,
                 seed: int = 42,
                 device: str | DeviceSpec | None = None,
                 resource_guard: Mapping[str, Any] | None = None,
                 export_root: str = "",
                 default_pipeline: Pipeline | None = None,
                 max_batch_methods: int = 8,
                 max_dataset_rows: int = DEFAULT_MAX_DATASET_ROWS,
                 ops_per_sec: float = DEFAULT_OPS_PER_SEC) -> None:
        self.session = session or Session(seed=seed)
        self._state_writer = state_writer
        self._dataset_roots = tuple(str(r) for r in (dataset_roots or ()))
        self._allowed_export_roots = tuple(
            str(r) for r in (allowed_export_roots or ()))
        self._seed = clean_seed(seed)
        self._device = clean_device(device)
        self._resource_guard = clean_resource_guard(resource_guard)
        self._export_root = ""
        self._default_pipeline = default_pipeline or Pipeline.default()
        self._max_dataset_rows = int(max_dataset_rows)
        self._ops_per_sec = float(ops_per_sec)
        if self._ops_per_sec <= 0:
            raise ValueError("ops_per_sec 必须是正数")
        self._datasets: dict[str, DatasetEntry] = {}
        self._dataset_seq = 0
        self._lock = threading.RLock()
        self._surface_query: Callable[..., Any] | None = None
        self._validator = MLValidator(
            self, allowed_export_roots=self._allowed_export_roots,
            max_batch_methods=max_batch_methods,
            max_batch_rows=max_dataset_rows)
        # export_root 得等校验器就位后才能校验（白名单来自装配点）
        if export_root:
            self._export_root = clean_export_root(
                export_root, allowed_roots=self._allowed_export_roots)

    # ------------------------------------------------------------ 事实面
    @property
    def validator(self) -> MLValidator:
        return self._validator

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def device(self) -> DeviceSpec:
        return self._device

    @property
    def resource_guard(self) -> dict[str, Any]:
        return dict(self._resource_guard)

    @property
    def export_root(self) -> str:
        return self._export_root

    @property
    def allowed_export_roots(self) -> tuple[str, ...]:
        return self._allowed_export_roots

    def method_names(self) -> list[str]:
        registry.load_builtin()
        return list(registry.names())

    def method_descriptor(self, name: str) -> Mapping[str, Any] | None:
        return next((m for m in self.session.list_methods()
                     if m["name"] == name), None)

    def method_param_schema(self, name: str) -> list[Mapping[str, Any]]:
        """方法的参数声明（覆写校验用）；未注册方法返回空表。"""
        try:
            method = registry.get(name)
        except KeyError:
            return []
        return [{"key": p.key, "kind": p.kind, "default": p.default,
                 "min": p.min, "max": p.max,
                 "choices": list(p.choices) if p.choices else None,
                 "hint": p.hint}
                for p in getattr(method, "param_schema", [])]

    def dataset_ids(self) -> list[str]:
        with self._lock:
            return list(self._datasets)

    def dataset_entry(self, dataset_id: str) -> DatasetEntry | None:
        with self._lock:
            return self._datasets.get(dataset_id)

    def dataset_facts(self, dataset_id: str) -> Mapping[str, Any] | None:
        entry = self.dataset_entry(dataset_id)
        return entry.facts() if entry is not None else None

    def pipeline_ids(self) -> list[str]:
        with self._lock:
            return sorted({e.pipeline_id for e in self._datasets.values()})

    def pipeline_id_for(self, dataset_id: str) -> str:
        entry = self.dataset_entry(dataset_id)
        return entry.pipeline_id if entry is not None else ""

    def allowed_export_roots(self) -> tuple[str, ...]:  # type: ignore[override]
        return self._allowed_export_roots

    # ------------------------------------------------------------ 只读查询
    def query(self, op_name: str, **args: Any) -> Any:
        """只读查询入口（``Surface.query`` 形状；**不经写权门**）。

        绑定 Surface 后所有查询走 ``mecha.Surface.query``（机制层注册项），
        未绑定（裸引擎/单测）则回退本地分派。两条路都不经写权门。
        """
        if self._surface_query is not None:
            return self._surface_query(op_name, **args)
        return self._query_direct(op_name, **args)

    def _query_direct(self, op_name: str, **args: Any) -> Any:
        handlers: dict[str, Callable[..., Any]] = {
            "method_catalog": self._query_method_catalog,
            "method_detail": self._query_method_detail,
            "dataset_detail": self._query_dataset_detail,
            "run_detail": self._query_run_detail,
        }
        fn = handlers.get(op_name)
        if fn is None:
            raise MechaError(
                f"未注册的只读查询 {op_name!r}",
                kind="unknown_query",
                hint=f"已注册查询：{sorted(handlers)}",
            )
        return fn(**args)

    def _query_method_catalog(self, *, family: str = "",
                              dataset_id: str = "") -> dict[str, Any]:
        rows = self.session.list_methods()
        if family:
            rows = [r for r in rows if r["family"] == family]
        if dataset_id:
            self._validator.check_dataset_ref(dataset_id)
            runnable = set(self._validator.runnable_method_names(dataset_id))
            rows = [r for r in rows if r["name"] in runnable]
        return {"method_count": len(rows), "methods": rows}

    def _query_method_detail(self, *, name: str) -> dict[str, Any]:
        descriptor = self.method_descriptor(name)
        if descriptor is None:
            self._validator.check_method_ref(name)  # 抛可教学错误
        method_cls = registry.get(name)
        schema = [{"key": p.key, "label": p.label, "kind": p.kind,
                   "default": p.default, "min": p.min, "max": p.max,
                   "choices": list(p.choices) if p.choices else None,
                   "hint": p.hint}
                  for p in getattr(method_cls, "param_schema", [])]
        out = dict(descriptor or {})
        out["param_schema"] = schema
        out["purposes_cn"] = method_cls.purposes_cn()
        return out

    def _query_dataset_detail(self, *, dataset_id: str) -> dict[str, Any]:
        facts = self._validator.check_dataset_ref(dataset_id)
        entry = self.dataset_entry(dataset_id)
        assert entry is not None  # check_dataset_ref 已确认存在
        chain = entry.spec.meta.get("chain", [])
        return {
            **dict(facts),
            "pipeline_chain": [
                {"step": c.get("step", {}).get("key", "?"),
                 "enabled": c.get("step", {}).get("enabled", True),
                 "skipped": bool(c.get("skipped")),
                 "big_num": (c.get("summary") or {}).get("big_num", "")}
                for c in chain
            ],
            "schema_mismatch": bool(entry.spec.meta.get("schema_mismatch", False)),
            "fit_scope": entry.spec.meta.get("fit_scope", ""),
        }

    def _query_run_detail(self, *, run_id: str) -> dict[str, Any]:
        self._validator.check_run_ref(run_id)
        record = self.session.record(run_id)
        if record is None:
            raise MechaError(
                f"未知 run_id {run_id!r}",
                kind="unknown_run",
                hint="run_id 由 run_method 回执给出；本会话只保留内存态记录",
            )
        return {
            "run_id": record.run_id,
            "method": record.method,
            "family": record.family,
            "task": record.task,
            "ok": bool(record.result.ok),
            "error": record.result.error or "",
            "primary_metric": record.result.primary_metric,
            "metrics": _jsonable_metrics(record.result.metrics),
            "params": _jsonable_scalars(record.result.params),
            "timings": _jsonable_scalars(record.timings),
            "dataset": record.dataset,
            "pipeline_id": record.pipeline_id,
            "artifact_keys": sorted(str(k) for k in record.result.artifacts),
        }

    # ------------------------------------------------------------ Engine ABC
    def schema(self) -> Mapping[str, Any]:
        """一处声明：命令面 + 回执契约 + 状态键 + 资源维度。"""
        return {
            "engine": type(self).__name__,
            "project": "ml_toolbox",
            "run_output_required": list(RECEIPT_REQUIRED_KEYS),
            "commands": {c.name: c.to_dict() for c in COMMANDS},
            "state_keys": sorted(_state_keys(self._validator)),
            "queries": ["method_catalog", "method_detail", "dataset_detail",
                        "run_detail"],
            "limits": {
                "max_batch_methods": self._validator.max_batch_methods,
                "max_dataset_rows": self._max_dataset_rows,
            },
            "resource_dimensions": ["est_sec", "peak_mib", "ops",
                                    "primary_cost", "warnings", "violations"],
        }

    def estimate(self, spec: Mapping[str, Any]) -> Mapping[str, Any]:
        """契约先行：任何命令都返回 ``est_sec``（mecha 由 ops 管道校验）。"""
        if not isinstance(spec, Mapping):
            raise MechaError(
                f"estimate 需要映射，收到 {type(spec).__name__}",
                kind="bad_spec",
                hint="示例 {'command': 'run_method', 'method': 'logistic'}",
            )
        command = str(spec.get("command", ""))
        base = {
            "command": command,
            "est_sec": 0.0,
            "peak_mib": 0.0,
            "ops": 0.0,
            "primary_cost": "no_run_concept",
            "warnings": [],
            "violations": [],
        }
        if command in ("prepare_dataset",):
            if "source" not in spec:
                return {**base, "est_sec": 0.5, "primary_cost": "pipeline_prepare"}
            est = self._estimate_prepare(spec)
            return {**base, **est}
        if command not in command_map():
            raise MechaError(
                f"未知命令 {command!r}，无法估算",
                kind="unknown_command",
                hint=f"受控命令：{sorted(command_map())}",
            )
        if command == "run_method":
            return {**base, **self._estimate_run(spec)}
        names = self._requested_methods(spec)
        per = [self._estimate_run({**dict(spec), "method": n}) for n in names]
        total = sum(float(p["est_sec"]) for p in per) or float(len(names)) * 0.5
        warnings = sorted({w for p in per for w in p["warnings"]})
        violations = sorted({v for p in per for v in p["violations"]})
        return {
            **base,
            "est_sec": min(MAX_EST_SEC, round(total, 3)),
            "peak_mib": max((float(p["peak_mib"]) for p in per), default=0.0),
            "ops": sum(float(p["ops"]) for p in per),
            "primary_cost": "batch_of_methods",
            "warnings": warnings + [f"批量 {len(names)} 个方法，估时为各方法之和"],
            "violations": violations,
        }

    def summary(self) -> Mapping[str, Any]:
        """复用 ``Session.export_state()`` 的只读导出（不另造第二份状态）。"""
        state = dict(self.session.export_state())
        with self._lock:
            datasets = [e.facts() for e in self._datasets.values()]
        return {
            **state,
            "adapter": "ml_mecha",
            "current": {
                "dataset_id": self._last_dataset_id() or "",
                "pipeline_id": self.pipeline_id_for(self._last_dataset_id() or ""),
                "method": self._last_method() or "",
                "seed": self._seed,
                "device": self._device.describe(),
                "export_root": self._export_root,
                "resource_guard": dict(self._resource_guard),
            },
            "datasets": datasets,
            "run_count": len(state.get("records", [])),
        }

    def run(self, spec: Mapping[str, Any], *, context: ExecutionContext | None = None,
            **opts: Any) -> Mapping[str, Any]:
        """受控命令分派。

        ``context`` 由框架传入（可能为 None——只读命令不需要取消语义）；
        取消只在批量/对比里按方法边界检查（ML 单次 fit 是同步长计算，
        本层不假装能中断）。
        """
        try:
            self._validator.validate_spec(spec, commands=command_map())
        except MechaError as exc:
            return self._failure(str(spec.get("command", "")), exc, [])
        command = str(spec["command"])
        try:
            handler = self._handlers()[command]
            return dict(handler(dict(spec), context))
        except MechaError as exc:
            return self._failure(command, exc, [])
        except SessionError as exc:
            return self._failure(command, MechaError(
                str(exc), kind="session_error",
                hint="ML 会话拒绝了这次请求（方法不可用/配置非法）"), [])
        except Exception as exc:  # noqa: BLE001 - 内部错误也要进回执，不逃逸
            return self._failure(command, MechaError(
                f"{type(exc).__name__}: {exc}", kind="internal_error",
                hint="栈留在宿主日志；回执只给结构化失败"), [])

    # ------------------------------------------------------------ 命令实现
    def _handlers(self) -> dict[str, Callable[[dict[str, Any], Any], Mapping[str, Any]]]:
        return {
            "prepare_dataset": self._cmd_prepare_dataset,
            "run_method": self._cmd_run_method,
            "run_method_batch": self._cmd_run_method_batch,
            "compare_methods": self._cmd_compare_methods,
        }

    def _cmd_prepare_dataset(self, spec: dict[str, Any],
                             context: Any) -> Mapping[str, Any]:
        dataset, source_desc = self._materialize_source(spec)
        name = str(spec.get("name") or getattr(dataset, "name", "") or "dataset")
        dataset.name = name
        target = spec.get("target") or dataset.target
        if target:
            dataset.target = str(target)
        time_col = spec.get("time_col") or dataset.time_col
        if time_col:
            dataset.time_col = str(time_col)
        diag = bool(spec.get("diag", False))
        self._validator.check_row_count(int(len(dataset.frame)))

        fitted = self.session.prepare_fitted(dataset, pipeline=self._default_pipeline,
                                             diag=diag)
        dataset_id = self._next_dataset_id(dataset.name)
        entry = self._build_entry(dataset_id, dataset, fitted.spec, source_desc, diag)
        # 写权门拒绝时**不留半拉子状态**：登记 → 写门 → 失败即回滚（真题试过，
        # 校验器必须在登记之后才认识这个 id；回滚让被拒的请求彻底无痕）。
        self._commit_dataset(entry)
        try:
            self._write_state(
                "ml_mecha:prepare_dataset",
                {"current.dataset_id": dataset_id,
                 "current.pipeline_id": entry.pipeline_id},
                spec,
            )
        except MechaError:
            self._rollback_dataset(dataset_id)
            raise
        warnings = self._dataset_warnings(entry)
        return {
            "ok": True,
            "command": "prepare_dataset",
            "run_ids": [],
            "primary_metric": "",
            "metrics": {"n_rows": entry.n_rows, "n_features": entry.n_features},
            "warnings": warnings,
            "dataset_id": dataset_id,
            "pipeline_id": entry.pipeline_id,
            "dataset": entry.facts(),
            "error": "",
        }

    def _cmd_run_method(self, spec: dict[str, Any], context: Any) -> Mapping[str, Any]:
        dataset_id = str(spec.get("dataset_id") or self._last_dataset_id() or "")
        self._validator.check_dataset_ref(dataset_id)
        entry = self.dataset_entry(dataset_id)
        assert entry is not None
        method = str(spec.get("method", ""))
        descriptor = self._validator.check_method_ref(method, dataset_id=dataset_id)
        request, options = self._build_request(
            spec, method=method, param_schema=self.method_param_schema(method))
        self._write_state(
            "ml_mecha:run_method",
            {"current.dataset_id": dataset_id,
             "current.method": method,
             "current.overrides": dict(request.overrides),
             "current.seed": int(request.seed),
             "current.device": clean_device(
                 options.get("device", spec.get("device", self._device))).to_state(),
             "current.resource_guard": dict(
                 request.extras.get("resource_guard", {}))},
            spec,
        )
        summary = self._execute_one(entry, request, call_id=_call_id())
        self._append_command_event("ml.command.run_method", summary, spec)
        warning = summary.warning
        return {
            "ok": summary.ok,
            "command": "run_method",
            "run_ids": [summary.run_id],
            "primary_metric": summary.primary_metric,
            "metrics": dict(summary.metrics),
            "warnings": [warning] if warning else [],
            "run_id": summary.run_id,
            "elapsed_s": summary.elapsed_s,
            "error": warning,
        }

    def _cmd_run_method_batch(self, spec: dict[str, Any],
                              context: Any) -> Mapping[str, Any]:
        entries = self._resolve_dataset(spec)
        methods = self._requested_methods(spec)
        self._validator.check_run_count(len(methods))
        for name in methods:
            self._validator.check_method_ref(name, dataset_id=entries.dataset_id)
        base_spec = {k: v for k, v in spec.items() if k != "methods"}
        request, options = self._build_request(base_spec, method="")
        overrides = clean_overrides(spec.get("overrides"))
        for name in methods:
            # 每个方法各自对表：批量里给 A 的覆写不该被 B 静默忽略
            clean_overrides(overrides, schema=self.method_param_schema(name))
        self._write_state(
            "ml_mecha:run_method_batch",
            {"current.dataset_id": entries.dataset_id,
             "current.seed": int(request.seed),
             "current.overrides": dict(request.overrides),
             "current.resource_guard": dict(
                 request.extras.get("resource_guard", {}))},
            spec,
        )
        runs: list[dict[str, Any]] = []
        warnings: list[str] = []
        for i, name in enumerate(methods):
            if context is not None and getattr(context, "cancelled", False):
                warnings.append(f"取消发生在第 {i + 1} 个方法之前：{name} 未运行")
                break
            if context is not None:
                try:
                    context.report_progress((i + 1) / max(1, len(methods)), name)
                except MechaError:
                    pass
            req = RunRequest(method=name, overrides=dict(request.overrides),
                             extras=dict(request.extras), diag=request.diag,
                             seed=request.seed, persist=request.persist)
            item = self._execute_isolated(entries, req, call_id=_call_id())
            runs.append(item.compact())
            if item.warning:
                warnings.append(f"{name}: {item.warning}")
            self._append_command_event("ml.command.run_method", item, spec)
        primary = next((r["primary_metric"] for r in runs if r["ok"]), "")
        metrics: dict[str, Any] = {}
        for r in runs:
            metrics[r["method"]] = {r["primary_metric"] or "primary": r["metrics"].get(
                r["primary_metric"])}
        return {
            "ok": bool(runs) and all(r["ok"] for r in runs),
            "command": "run_method_batch",
            "run_ids": [r["run_id"] for r in runs],
            "primary_metric": primary,
            "metrics": metrics,
            "warnings": warnings,
            "runs": runs,
            "error": "" if all(r["ok"] for r in runs) else "部分方法失败",
        }

    def _cmd_compare_methods(self, spec: dict[str, Any],
                             context: Any) -> Mapping[str, Any]:
        entries = self._resolve_dataset(spec)
        methods = self._requested_methods(spec)
        self._validator.check_run_count(len(methods))
        overrides = clean_overrides(spec.get("overrides"))
        seed = clean_seed(spec.get("seed", self._seed))
        guard = (clean_resource_guard(spec["resource_guard"])
                 if "resource_guard" in spec else dict(self._resource_guard))
        for name in methods:
            self._validator.check_method_ref(name, dataset_id=entries.dataset_id)
            clean_overrides(overrides, schema=self.method_param_schema(name))
        self._write_state(
            "ml_mecha:compare_methods",
            {"current.dataset_id": entries.dataset_id, "current.seed": seed,
             "current.resource_guard": dict(guard)},
            spec,
        )
        rows: list[dict[str, Any]] = []
        runs: list[dict[str, Any]] = []
        warnings: list[str] = []
        primary = ""
        for i, name in enumerate(methods):
            if context is not None and getattr(context, "cancelled", False):
                warnings.append(f"取消发生在第 {i + 1} 个方法之前：{name} 未运行")
                break
            req = RunRequest(method=name, overrides=dict(overrides), seed=seed,
                             extras=(dict({"resource_guard": dict(guard)})
                                     if guard else {}))
            item = self._execute_isolated(entries, req, call_id=_call_id())
            runs.append(item.compact())
            primary = primary or (item.primary_metric if item.ok else "")
            score = item.metrics.get(item.primary_metric) if item.primary_metric else None
            rows.append({"method": name, "run_id": item.run_id, "ok": item.ok,
                         "primary_metric": item.primary_metric,
                         "score": score,
                         "elapsed_s": item.elapsed_s,
                         "error": item.warning})
            if item.warning:
                warnings.append(f"{name}: {item.warning}")
            self._append_command_event("ml.command.run_method", item, spec)
        winner = _pick_winner(rows, primary)
        return {
            "ok": bool(runs) and all(r["ok"] for r in runs),
            "command": "compare_methods",
            "run_ids": [r["run_id"] for r in runs],
            "primary_metric": primary,
            "metrics": {r["method"]: r["score"] for r in rows},
            "warnings": warnings,
            "table": rows,
            "winner": winner,
            "error": "" if all(r["ok"] for r in runs) else "部分方法失败",
        }

    # ------------------------------------------------------------ 内部：数据
    def _materialize_source(self, spec: Mapping[str, Any]) -> tuple[Dataset, str]:
        source = spec.get("source")
        if isinstance(source, Dataset):
            return source, f"dataset:{source.name}"
        if isinstance(source, pd.DataFrame):
            return Dataset(source, name="dataset"), "frame:inline"
        if source is None:
            raise MechaError(
                "prepare_dataset 缺 source",
                kind="missing_arguments",
                hint="source 见 schema()['commands']['prepare_dataset']",
            )
        if not isinstance(source, Mapping):
            raise MechaError(
                f"source 需要映射或 Dataset/DataFrame，收到 {type(source).__name__}",
                kind="bad_source",
                hint="source={'kind': 'csv', 'path': ...} / 'frame' / 'arrays' / 'synthetic'",
            )
        kind = str(source.get("kind", "")).strip().lower()
        if kind in ("frame", "dataframe"):
            frame = source.get("frame")
            if not isinstance(frame, pd.DataFrame):
                raise MechaError(
                    "source.kind='frame' 需要 source['frame'] 是 DataFrame",
                    kind="bad_source",
                    hint="仅用于进程内调用；跨进程用 'csv'",
                )
            target = source.get("target")
            return Dataset(frame.copy(), name=str(source.get("name", "dataset")),
                           target=str(target) if target else None,
                           source="frame:inline"), "frame:inline"
        if kind == "csv":
            path = self._resolve_data_path(source.get("path"))
            target = source.get("target")
            ds = Dataset.load(str(path), target=str(target) if target else None)
            ds.source = f"csv:{path}"
            return ds, f"csv:{path.name}"
        if kind == "arrays":
            X = source.get("X")
            if X is None:
                raise MechaError("source.kind='arrays' 需要 source['X']",
                                 kind="bad_source", hint="X 为二维数组/DataFrame")
            arr = np.asarray(X, dtype=float)
            if arr.ndim != 2:
                raise MechaError(
                    f"source['X'] 需要二维数组，收到 ndim={arr.ndim}",
                    kind="bad_source", hint="形状 (n_samples, n_features)")
            y = source.get("y")
            ds = Dataset.from_arrays(arr, None if y is None else np.asarray(y),
                                     name=str(source.get("name", "dataset")))
            ds.source = "arrays:inline"
            return ds, "arrays:inline"
        if kind == "synthetic":
            ds = _synthetic_dataset(
                task=str(source.get("task", "classification")),
                n_samples=int(source.get("n_samples", 120)),
                n_features=int(source.get("n_features", 4)),
                n_informative=int(source.get("n_informative", 2)),
                noise=float(source.get("noise", 0.1)),
                seed=clean_seed(source.get("seed", self._seed)),
            )
            return ds, f"synthetic:{source.get('task', 'classification')}"
        raise MechaError(
            f"未知数据来源 kind={kind!r}",
            kind="bad_source",
            hint="支持 'csv' / 'frame' / 'arrays' / 'synthetic'（不自动猜）",
        )

    def _resolve_data_path(self, raw: Any) -> Path:
        if not isinstance(raw, (str, Path)):
            raise MechaError(
                f"source.path 需要路径字符串，收到 {type(raw).__name__}",
                kind="bad_source_path", hint="示例 source={'kind':'csv','path':'data/x.csv'}",
            )
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute():
            roots = [Path(r) for r in self._dataset_roots]
            candidate = roots[0] / candidate
        candidate = candidate.resolve()
        # Fail closed by default: file-backed datasets are AI-visible input, so
        # no explicit dataset root means no file reads at all.  The old
        # ``Path.cwd()`` fallback let ``csv`` read arbitrary absolute paths.
        if not self._dataset_roots:
            raise MechaError(
                f"未声明数据根，拒绝读取文件：{str(candidate)!r}",
                kind="dataset_roots_required",
                hint="装配时传 dataset_roots=[...] 才允许 csv 数据源；"
                     "未声明即不允许 AI 读取任意路径（fail closed）",
                suggest="assemble_ml_mecha(root=..., dataset_roots=[r'D:\\data'])",
            )
        allowed = [Path(r).resolve() for r in self._dataset_roots]
        if not any(candidate == r or r in candidate.parents for r in allowed):
            raise MechaError(
                f"数据路径 {str(candidate)!r} 不在允许的数据根内",
                kind="dataset_path_denied",
                hint=f"允许的根：{[str(r) for r in allowed]}",
                suggest=str(allowed[0]),
            )
        if not candidate.exists():
            raise MechaError(
                f"数据文件不存在：{str(candidate)!r}",
                kind="dataset_not_found",
                hint="相对路径按装配时声明的 dataset_roots[0] 解析",
            )
        return candidate

    def _next_dataset_id(self, name: str) -> str:
        """预分配数据集引用（写权门拒绝时该号不回滚，但**不登记**任何状态）。"""
        with self._lock:
            self._dataset_seq += 1
            slug = "".join(ch if ch.isalnum() else "-" for ch in name)[:24]
            return f"ds-{self._dataset_seq:03d}-{slug or 'dataset'}"

    def _build_entry(self, dataset_id: str, dataset: Dataset, spec: DataSpec,
                     source: str, diag: bool) -> DatasetEntry:
        return DatasetEntry(
            dataset_id=dataset_id, name=dataset.name, target=dataset.target,
            time_col=dataset.time_col, source=source,
            n_rows=int(len(dataset.frame)),
            n_features=int(spec.X.shape[1]),
            target_kind=spec.target_kind,
            pipeline_id=str(spec.meta.get("pipeline_id", "")),
            diag=diag, spec=spec,
        )

    def _commit_dataset(self, entry: DatasetEntry) -> None:
        with self._lock:
            self._datasets[entry.dataset_id] = entry

    def _rollback_dataset(self, dataset_id: str) -> None:
        """写权门/校验拒绝后的回滚：内存里也不留这条数据集。"""
        with self._lock:
            self._datasets.pop(dataset_id, None)

    def _resolve_dataset(self, spec: Mapping[str, Any]) -> DatasetEntry:
        dataset_id = str(spec.get("dataset_id") or self._last_dataset_id() or "")
        self._validator.check_dataset_ref(dataset_id)
        entry = self.dataset_entry(dataset_id)
        assert entry is not None
        return entry

    def _last_dataset_id(self) -> str | None:
        with self._lock:
            return next(reversed(self._datasets), None) if self._datasets else None

    def _last_method(self) -> str | None:
        record = self.session.history()
        return record[-1].method if record else None

    def _dataset_warnings(self, entry: DatasetEntry) -> list[str]:
        out: list[str] = []
        if entry.spec.meta.get("schema_mismatch"):
            out.append("训练/测试列集合不一致，测试列已按训练列对齐（补 0）")
        if entry.n_rows > self._max_dataset_rows * 0.8:
            out.append(f"行数 {entry.n_rows} 接近单次运行上限 {self._max_dataset_rows}")
        return out

    # ------------------------------------------------------------ 内部：执行
    def _build_request(self, spec: Mapping[str, Any], *,
                       method: str,
                       param_schema: Sequence[Mapping[str, Any]] | None = None,
                       ) -> tuple[RunRequest, dict[str, Any]]:
        seed_source = spec.get("seed", self._seed)
        if seed_source is None:
            seed_source = self._seed
        seed = clean_seed(seed_source)
        overrides = clean_overrides(spec.get("overrides"), schema=param_schema)
        guard = (clean_resource_guard(spec["resource_guard"])
                 if "resource_guard" in spec else dict(self._resource_guard))
        device = clean_device(spec.get("device", self._device))
        extras: dict[str, Any] = {}
        if guard:
            extras["resource_guard"] = {k: v for k, v in guard.items()
                                        if k != "enabled"}
            if guard.get("enabled") and not extras["resource_guard"]:
                extras["resource_guard"] = {}
        request = RunRequest(method=method, overrides=overrides, extras=extras,
                             diag=bool(spec.get("diag", False)), seed=seed,
                             persist=bool(spec.get("persist", False)))
        return request, {"device": device}

    def _execute_one(self, entry: DatasetEntry, request: RunRequest,
                     *, call_id: str) -> RunSummary:
        record = self.session.run(entry.spec, request)
        return _summarize_record(record, entry.dataset_id, request)

    def _execute_isolated(self, entry: DatasetEntry, request: RunRequest,
                          *, call_id: str) -> RunSummary:
        """批量用：一个方法派发失败不拖垮其余方法（与 ML runner 同一纪律）。"""
        try:
            return self._execute_one(entry, request, call_id=call_id)
        except (SessionError, MechaError, KeyError, TypeError, ValueError) as exc:
            return RunSummary(
                run_id="",
                method=request.method, dataset_id=entry.dataset_id,
                pipeline_id=entry.pipeline_id, ok=False,
                primary_metric="", metrics={}, elapsed_s=0.0,
                warning=f"派发失败：{type(exc).__name__}: {exc}",
            )

    def _write_state(self, reason: str, updates: Mapping[str, Any],
                     spec: Mapping[str, Any]) -> list[Any]:
        """经注入的写口把状态键落到 Gate；没装配写口就 fail loud，不静默降级。"""
        payload = {k: v for k, v in updates.items() if v is not None}
        if not payload:
            return []
        if self._state_writer is None:
            raise MechaError(
                "引擎未接写入门，无法落地域状态",
                kind="missing_state_writer",
                hint="装配点必须调 assemble_ml_mecha（唯一装配点）注入写口；"
                     "裸引擎只支持只读命令与估算",
            )
        return self._state_writer(reason, payload, _call_id())

    def _append_command_event(self, key: str, summary: RunSummary,
                              spec: Mapping[str, Any]) -> None:
        """把**紧凑**命令事件追加进 History（大对象一律只留 run_id 引用）。

        归因取自**本次请求的通道**（不是装配期的固定标签）：人类侧发起的运行
        记 actor='ml-gui'，AI 侧发起的记 actor='ml-ai'——否则归因是假的。
        """
        writer = self._command_event_writer
        if writer is None:
            return
        channel = active_channel()
        writer(key, summary.compact(), _call_id(),
               channel.actor if channel is not None else "unknown")

    #: 装配点注入：直接 ``History.append``（命令事件不是域状态，不走 Gate 快照）。
    _command_event_writer: Callable[[str, Mapping[str, Any], str, str], None] | None = None

    def bind_command_events(
            self,
            writer: Callable[[str, Mapping[str, Any], str, str], None]) -> None:
        """注入命令事件写口：``(key, value, call_id, actor) -> None``。"""
        self._command_event_writer = writer

    def bind_state_writer(
            self,
            writer: Callable[[str, Mapping[str, Any], str], list[Any]]) -> None:
        """注入域状态写口（唯一实现是 ``make_state_writer``：经 mecha Gate）。"""
        self._state_writer = writer

    def bind_surface_query(self, query_fn: Callable[..., Any]) -> None:
        """注入 ``Surface.query``：只读查询统一走 mecha 注册项分派。"""
        self._surface_query = query_fn

    def _failure(self, command: str, exc: MechaError,
                 warnings: list[str]) -> Mapping[str, Any]:
        return {
            "ok": False,
            "command": command,
            "run_ids": [],
            "primary_metric": "",
            "metrics": {},
            "warnings": warnings + [exc.teaching_text()],
            "error": exc.message,
            "error_kind": exc.kind,
            "error_hint": exc.hint,
            "error_suggest": exc.suggest,
        }

    # ------------------------------------------------------------ 内部：估算
    def _estimate_run(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        method = str(spec.get("method", ""))
        if not method:
            return {"est_sec": 0.5, "peak_mib": 0.0, "ops": 0.0,
                    "primary_cost": "unknown_method",
                    "warnings": ["缺少 method，按最小估时"], "violations": []}
        entry = self.dataset_entry(str(spec.get("dataset_id") or
                                       self._last_dataset_id() or ""))
        if entry is None:
            return {"est_sec": 0.5, "peak_mib": 0.0, "ops": 0.0,
                    "primary_cost": "no_dataset",
                    "warnings": ["尚无数据集：先 prepare_dataset 才能估时"],
                    "violations": []}
        guard = (clean_resource_guard(spec["resource_guard"])
                 if "resource_guard" in spec else dict(self._resource_guard))
        estimate = self._estimate_entry(entry, method, spec)
        violations = list(estimate.violations)
        est_sec = min(MAX_EST_SEC, max(0.05, estimate.operation_units / self._ops_per_sec))
        return {
            "est_sec": round(est_sec, 3),
            "peak_mib": round(estimate.working_set_mb, 1),
            "ops": estimate.operation_units,
            "complexity": estimate.complexity,
            "primary_cost": "method_fit",
            "n_samples": estimate.n_samples,
            "n_features": estimate.n_features,
            "warnings": list(estimate.warnings),
            "violations": violations,
            "budget": guard,
        }

    def _estimate_prepare(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        source = spec.get("source")
        n_rows = 1
        if isinstance(source, Mapping):
            n_rows = int(source.get("n_samples", 1) or 1)
        elif isinstance(source, pd.DataFrame):
            n_rows = int(len(source))
        est = round(max(0.1, min(MAX_EST_SEC, n_rows / 5000.0)), 3)
        return {"est_sec": est, "peak_mib": 0.0, "ops": float(n_rows),
                "primary_cost": "pipeline_prepare",
                "warnings": ([] if n_rows <= self._max_dataset_rows
                             else [f"行数 {n_rows} 超过单次运行上限"]),
                "violations": []}

    def _estimate_entry(self, entry: DatasetEntry, method: str,
                        spec: Mapping[str, Any]) -> ResourceEstimate:
        guard = (clean_resource_guard(spec["resource_guard"])
                 if "resource_guard" in spec else dict(self._resource_guard))
        limit_map = {k: (None if v == "unlimited" else v)
                     for k, v in guard.items()
                     if k in ("max_kernel_mb", "max_working_set_mb",
                              "max_quantum_state_mb", "max_operation_units",
                              "max_qubits")}
        budget = ResourceBudget.from_mapping(limit_map) if limit_map else None
        params: Mapping[str, Any] = {}
        try:
            method_obj = registry.get(method)
            params = method_obj.params(
                RunRequest(method=method, overrides=clean_overrides(
                    spec.get("overrides")), seed=self._seed).config())
        except Exception:  # noqa: BLE001 - 估算不该因为参数非法而炸
            params = dict(clean_overrides(spec.get("overrides")))
        return estimate_resources(
            method, entry.n_rows, entry.n_features, params=params,
            n_qubits=guard.get("n_qubits"), budget=budget,
            safety_factor=float(guard.get("safety_factor", 1.0)),
        )

    def _requested_methods(self, spec: Mapping[str, Any]) -> list[str]:
        raw = spec.get("methods")
        if isinstance(raw, str):
            names = [raw]
        elif isinstance(raw, Sequence):
            names = [str(m) for m in raw]
        else:
            raise MechaError(
                f"methods 需要方法名列表，收到 {type(raw).__name__}",
                kind="bad_methods", hint="示例 ['logistic', 'svc']",
            )
        if not names:
            raise MechaError("methods 不能为空", kind="bad_methods",
                             hint="至少给一个方法名")
        seen: list[str] = []
        for n in names:
            if n not in seen:
                seen.append(n)
        return seen


# ---------------------------------------------------------------- 模块工具
def _call_id() -> str:
    """读 mecha 的请求作用域 call_id（AI 侧互引；人类侧为空字符串）。"""
    try:
        from mecha.tools import CURRENT_CALL_ID
    except Exception:  # noqa: BLE001 - 未装 mecha 时不该炸
        return ""
    return CURRENT_CALL_ID.get()


def _state_keys(validator: MLValidator) -> set[str]:
    from .validator import STATE_KEYS
    return set(STATE_KEYS)


def kinds_of(target_kind: str | None) -> list[str]:
    """目标类型 → 校验器用的候选列表（None = 不限，空列表）。"""
    return [target_kind] if target_kind else []


def _jsonable_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if k == "cv_scores":
            continue
        if isinstance(v, (int, float, np.floating, np.integer)):
            fv = float(v)
            out[str(k)] = fv if np.isfinite(fv) else None
        elif isinstance(v, str):
            out[str(k)] = v
    return out


def _jsonable_scalars(values: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (values or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[str(k)] = v
        else:
            out[str(k)] = str(v)
    return out


def _summarize_record(record: Any, dataset_id: str,
                      request: RunRequest) -> RunSummary:
    """把 ``RunRecord`` 压成紧凑投影——**只**留引用与标量指标。"""
    result = record.result
    metrics = _jsonable_metrics(result.metrics)
    warning = ""
    if not result.ok:
        warning = (result.error or "运行失败").splitlines()[-1][:200]
    elif not metrics:
        warning = "运行成功但没有产出可比较的数值指标"
    return RunSummary(
        run_id=record.run_id, method=record.method, dataset_id=dataset_id,
        pipeline_id=record.pipeline_id, ok=bool(result.ok),
        primary_metric=str(result.primary_metric or ""),
        metrics=metrics,
        elapsed_s=round(float(getattr(result, "elapsed", 0.0) or 0.0), 4),
        warning=warning,
    )


def _pick_winner(rows: Sequence[Mapping[str, Any]], primary: str) -> dict[str, Any]:
    """按主指标方向挑最优行（``is_lower_better`` 是 ML 核心的单一事实来源）。"""
    scored = [r for r in rows if r.get("ok") and isinstance(r.get("score"), (int, float))]
    if not scored or not primary:
        return {"method": "", "run_id": "", "score": None, "reason": "无可比较的成功运行"}
    lower = is_lower_better(primary)
    best = (min if lower else max)(scored, key=lambda r: float(r["score"]))
    return {"method": best["method"], "run_id": best["run_id"],
            "score": best["score"],
            "reason": f"{primary} {'越小越好' if lower else '越大越好'}"}


def _synthetic_dataset(*, task: str, n_samples: int, n_features: int,
                       n_informative: int, noise: float, seed: int) -> Dataset:
    """离线可复现的合成数据（测试与教学演示用；不读任何外部文件）。"""
    rng = np.random.RandomState(seed)
    n_samples = max(20, int(n_samples))
    n_features = max(1, int(n_features))
    n_informative = max(1, min(int(n_informative), n_features))
    X = rng.randn(n_samples, n_features)
    signal = X[:, :n_informative].sum(axis=1)
    if task == "regression":
        y = signal + noise * rng.randn(n_samples)
        return Dataset.from_arrays(X, y, name="synthetic_regression")
    if task == "timeseries":
        t = np.arange(n_samples, dtype=float)
        y = np.sin(2 * np.pi * t / 12.0) + noise * rng.randn(n_samples)
        ds = Dataset.from_arrays(y.reshape(-1, 1), y, name="synthetic_timeseries",
                                 time=t)
        return ds
    y = (signal + noise * rng.randn(n_samples) > 0).astype(int)
    return Dataset.from_arrays(X, y, name="synthetic_classification")


__all__ = [
    "MLEngine", "DatasetEntry", "RunSummary", "RECEIPT_REQUIRED_KEYS",
    "ChannelRequired", "make_state_writer", "use_channel", "clear_channel",
    "active_channel", "DEFAULT_MAX_DATASET_ROWS", "DEFAULT_OPS_PER_SEC",
]
