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
from ml_toolbox.core.contracts import AUX_GRAPH, DataSpec, is_lower_better
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

#: 核心 CommandSpec 的默认估时声明（秒）。真实估算仍走 :meth:`MLEngine.estimate`；
#: 核心 ``estimate`` 只做"声明存在 + 形状对"的守卫。
DEFAULT_EST_SEC = 0.5

#: 回执/History 里最多列几个列名（宽表不截断 = 把表头塞进事件）。
MAX_LISTED_COLUMNS = 64

#: 当前执行通道（供工具层把 actor/side 带进来；未设置则写状态 fail closed）。
_EXEC_CHANNEL: contextvars.ContextVar[Channel | None] = contextvars.ContextVar(
    "ml_mecha_exec_channel", default=None)


#: **自定义网络的注册槽**（由方法层在 import 时填）。
#:
#: ⚠ 为什么是槽而不是 `from ml_toolbox.methods.neural import custom`：
#: 本模块住在边界层（`ml_mecha/`），**不许**顶层 import 上层
#: （结构性纪律，`ast.walk` 静态守着）。而"自定义网络的类/名字/参数表"这件事
#: 的事实面在方法层 ⇒ 用本仓既有做法：**方法层注册、边界层代转**
#: （同 `core.persistence` 的 opt 槽）。取不到 ⇒ 命令照常存在，
#: 调用时给一条可读错误（fail loud，不是静默降级）。
_CUSTOM_SLOT: dict[str, Any] = {}

#: 兜底名字/键名：与方法层**同名**（两边都写同一事实，取值时以槽为准）。
_CUSTOM_METHOD_NAME_FALLBACK = "custom_network"
_CUSTOM_CODE_KEY_FALLBACK = "custom_network_code"


def register_custom_network(*, method_name: str, code_key: str,
                            cls: type, param_schema: Sequence[Any]) -> None:
    """由方法层（`methods/neural/custom.py`）在 import 时调用（幂等）。"""
    _CUSTOM_SLOT.update({"method_name": str(method_name), "code_key": str(code_key),
                         "cls": cls, "param_schema": list(param_schema)})


def _custom_method_name() -> str:
    return str(_CUSTOM_SLOT.get("method_name") or _CUSTOM_METHOD_NAME_FALLBACK)


def _custom_code_key() -> str:
    return str(_CUSTOM_SLOT.get("code_key") or _CUSTOM_CODE_KEY_FALLBACK)


def _custom_param_schema() -> list:
    return list(_CUSTOM_SLOT.get("param_schema") or [])


#: "这一次是在**后台 job** 里跑的"标记——由 :meth:`MLMecha.submit` 的 worker 放进
#: spec，:meth:`MLEngine.run` 在**校验之前**摘走（它不在命令参数面里），
#: 转成 `self._job_run` 供 `_build_request` 打开 `persist`。
#: ⚠ 单一真源在这里（引擎是被标记方）；装配点从本模块 import，不各写一份字面量。
JOB_RUN_FLAG = "job_run"

#: 会产生 run 的命令（只有它们值得为"后台"打开落盘）。
RUN_COMMANDS = frozenset({"run_method", "run_method_batch", "compare_methods",
                          "run_custom"})

#: 进度**富载荷**挂在 `ExecutionContext` 上的属性名（见 `_progress_hook`）。
#: ⚠ 为什么用 sidecar 而不是改框架：`mecha.surface.ExecutionContext.report_progress`
#: 只收 `(frac, stage)`，而 `_progress` 存的是**裸元组** ⇒ 想带 loss/run_id 只有三条路：
#: ① 改框架（**不许**：本仓只消费框架）；② 往 stage 字符串里编码（脏，且会破坏既有
#: "stage 是人读的阶段名"契约）；③ **在 context 上挂一个附加 dict**（本方案）。
#: ⇒ ③ 既不动框架、也不污染既有形状，而富载荷只在"有值时才出现"。
_PROGRESS_SIDECAR = "ml_progress_extra"

#: job 进度里最多留多少条**逐代数值**（有界，防长训练下线性膨胀）。
PROGRESS_EPOCH_WINDOW = 50


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
        self._commands: Any | None = None
        self._command_gate: Any | None = None
        #: 审批通道（``Software.approval``）：只有声明 ``approval_required=True`` 的命令会用到；
        #: 未注入时框架 fail closed（拒绝），不会静默放行。
        self._command_approval: Any | None = None
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
            "training_detail": self._query_training_detail,
            "network_catalog": self._query_network_catalog,
            "network_source": self._query_network_source,
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
                 "title": str(c.get("step", {}).get("title", "") or ""),
                 "enabled": c.get("step", {}).get("enabled", True),
                 "skipped": bool(c.get("skipped")),
                 "big_num": (c.get("summary") or {}).get("big_num", ""),
                 # ⭐ **参数**与**标量摘要**一并透出（原先只有 key + big_num）。
                 #   为什么必须给：AI 能看到的 pipeline_id 只是一个指纹，
                 #   而"这条链到底按什么参数跑的"（缺失值策略、编码模式、
                 #   缩放方式……）全在 `step.params` 里 ⇒ 不给她就只能猜
                 #   （实测缺口：AI 无法回答"类别列是怎么编码的"）。
                 "params": _jsonable_scalars(c.get("step", {}).get("params")),
                 "summary": _jsonable_scalars(
                     _chain_summary(c.get("summary")))}
                for c in chain
            ],
            # ⭐ **实际的划分口径**。优先用**这份数据集自己的**记录
            #   （`DataSpec.meta["split_policy"]`，取自 Pipeline 本体）——
            #   因为 AI 现在可以逐数据集指定切分（`pipeline.split` 声明），
            #   报引擎默认值会说错。老档没有这个键 ⇒ 回落到引擎默认（诚实兜底）。
            "split_policy": dict(entry.spec.meta.get("split_policy")
                                 or self._default_pipeline._split_policy()),
            "schema_mismatch": bool(entry.spec.meta.get("schema_mismatch", False)),
            "fit_scope": entry.spec.meta.get("fit_scope", ""),
        }

    def _query_network_catalog(self, *, name: str = "",
                               limit: int = 50) -> dict[str, Any]:
        """⭐ 已保存的自定义网络清单（**从磁盘存档汇总**，不另建索引）。

        为什么必须有它（用户点破的实际风险）：如果没有"按名字复用"，AI 每次改个
        学习率都要**把网络代码重贴一遍** ⇒ token 全烧在复制粘贴上，自由度换来的是
        低效。有了它，第一次花 token 写、之后按名字取。

        ## 为什么从存档汇总、不建索引文件

        - **零新增真相**：`runs/<id>/record.json` 已经是"那次跑的是什么"的唯一真源，
          另建 `networks.json` 就是第二份会漂的账；
        - **代价**：要扫存档。所以① 只扫最近的 `limit` 个 run；② 同名的**后写覆盖**
          （与"最近一次"语义一致，且留 `runs` 计数说明它被用过几次）。

        ⚠ 结果里**不带源码**（可能很大）：要看源码用 `describe_network`。
        """
        from . import tb_series                      # noqa: F401 - 保持与本模块同一读面
        from ml_toolbox.core.persistence import RUNS_DIR
        from pathlib import Path
        import json as _json

        root = Path(RUNS_DIR)
        found: dict[str, dict[str, Any]] = {}
        if not root.is_dir():
            return {"networks": [], "scanned": 0,
                    "note": "runs/ 目录不存在（本会话还没跑过任何东西）"}
        dirs = sorted((d for d in root.iterdir() if d.is_dir()),
                      key=lambda d: d.stat().st_mtime if d.exists() else 0.0,
                      reverse=True)[: max(1, int(limit))]
        scanned = 0
        for d in dirs:
            rec_path = d / "record.json"
            if not rec_path.is_file():
                continue
            scanned += 1
            try:
                payload = _json.loads(rec_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            extras = ((payload.get("config") or {}).get("extras") or {})
            code = extras.get(_custom_code_key())
            if not code:
                continue
            net_name = str(extras.get("network_name") or "").strip() or "custom"
            if name and net_name != name:
                continue
            metrics = payload.get("metrics") or {}
            primary = str(payload.get("primary_metric") or "")
            found[net_name] = {
                "name": net_name,
                "run_id": payload.get("run_id") or d.name,
                "primary_metric": primary,
                "score": metrics.get(primary) if isinstance(metrics, dict) else None,
                "saved_at": payload.get("saved_at") or "",
                "n_lines": len(str(code).splitlines()),
                "uses": int(found.get(net_name, {}).get("uses", 0)) + 1,
            }
        rows = sorted(found.values(), key=lambda r: r["saved_at"], reverse=True)
        return {"networks": rows, "scanned": scanned,
                "source_key": _custom_code_key(),
                "note": "源码不在本清单里（可能很大）；用 describe_network 取"}

    def _query_network_source(self, *, name: str, limit: int = 200) -> dict[str, Any]:
        """按名字取回**已保存网络的源码**（供"改一版再跑"用，免重贴）。

        ⚠ 读不到 ⇒ `available=false` + 可读原因（**不返回空源码假装成功**）。
        """
        cat = self._query_network_catalog(name=name, limit=int(limit))
        rows = cat.get("networks") or []
        if not rows:
            return {"available": False, "name": str(name), "code": "",
                    "reason": f"存档里没有名为 {name!r} 的自定义网络"
                              f"（扫了最近 {cat.get('scanned')} 个 run）"}
        rid = str(rows[0]["run_id"])
        from pathlib import Path
        import json as _json

        from ml_toolbox.core.persistence import RUNS_DIR
        try:
            payload = _json.loads((Path(RUNS_DIR) / rid / "record.json")
                                  .read_text(encoding="utf-8"))
            code = ((payload.get("config") or {}).get("extras") or {}).get(
                _custom_code_key()) or ""
        except Exception as exc:                          # noqa: BLE001
            return {"available": False, "name": str(name), "code": "",
                    "reason": f"读 {rid} 的存档失败：{type(exc).__name__}: {exc}"}
        if not code:
            return {"available": False, "name": str(name), "code": "",
                    "reason": f"{rid} 的存档里没有源码（该 run 不是 run_custom 跑的？）"}
        return {"available": True, "name": str(name), "run_id": rid,
                "code": str(code), "reason": ""}

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

    def _query_training_detail(self, *, run_id: str = "",
                               include_series: bool = False,
                               limit: int = 200) -> dict[str, Any]:
        """训练的**逐 epoch 观测数值**（只读；数据源 = TensorBoard 事件文件）。

        ## 为什么必须是只读查询 + 不走会话内存

        观测数据是**训练时逐 epoch 落盘**的（`tb_log.log_epoch` → `runs/<id>/tb/`），
        而 ``run_detail`` 只看本会话的内存记录 ⇒ 进程一换就"描述不到"，
        事件文件却还在。⇒ 本条**只按 run_id 读磁盘**，因此对下列两种情形同样有效：
        未落盘的运行（`persist=false`，AI 默认）与跨重启的历史运行。

        ## 为什么默认只给"最新一轮"

        `include_series` 默认 False：整条序列是"趋势"问题的答案，代价是响应大
        （11 个 tag × 最多 500 点）。要看趋势时显式打开，并按需调 `limit`。

        ⚠ **读不到就说读不到**（`available=false` + `reason`），绝不回空数组假装
        "跑到第 0 个 epoch"——这是本仓反复在守的那条纪律。
        """
        from . import tb_series

        payload = tb_series.read_run(str(run_id or ""),
                                     limit=max(1, int(limit or 200)))
        out: dict[str, Any] = {
            "run_id": str(run_id or ""),
            "available": bool(payload.get("available")),
            "reason": str(payload.get("reason") or ""),
            "latest": tb_series.latest_of(payload),
            "tags": sorted((payload.get("scalars") or {}).keys()),
            "histograms": sorted((payload.get("histograms") or {}) or []),
            "images": sorted((payload.get("images") or {}) or []),
            "text": sorted((payload.get("text") or {}) or []),
            # 同一 step 写了多个值时的留痕（重跑沿用同一 run_id 会走到这里）
            "duplicate_steps": payload.get("duplicate_steps") or [],
        }
        if not include_series:
            return out
        scal = payload.get("scalars") or {}
        # ⚠ 每个 tag **各自**带 steps：tag 可以晚开始（`val_split=0` 时 val_loss
        #   整条不写），拿全局步轴去对会错位。
        out["series"] = {
            tag: {"steps": list((got or {}).get("steps") or []),
                  "values": list((got or {}).get("values") or [])}
            for tag, got in scal.items()}
        out["max_points"] = int(tb_series.MAX_POINTS)
        return out

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
                        "run_detail", "training_detail", "network_catalog",
                        "network_source"],
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

        装配后命令声明注册进 mecha 核心 ``CommandRegistry``（唯一命令审计 +
        result_ref）；未绑定的裸引擎回退本地分派，便于单测。

        取消语义（审查 P1-3 残余）：``context.cancelled`` 在**命令入口**先查一次
        ——已取消就当场返回结构化失败回执（``error_kind="cancelled"``），
        **不写域状态、不产生 run、不写 ``ml.run``、不写 ``command.<name>`` 审计**。
        批量/对比另在每个方法边界复查（见对应命令体）。

        **诚实声明**：取消是协作式的，检查点只有"命令入口"和"方法与方法之间"。
        ``Session.run`` 的一次 fit 计算内部**不可中断**——ML 方法层没有协作式
        取消点，本层不假装能打断正在跑的训练（不谎报可中断性）。
        """
        command = str(spec.get("command", ""))
        if _is_cancelled(context):
            return self._cancelled_receipt(command)
        # ⚠ **后台标记必须在任何校验之前摘掉**：它由装配点的 job worker 放进 spec，
        #   而 `validate_spec` 只认**命令声明的参数** ⇒ 留在里面会被判成
        #   "接受了未声明的参数"（实测：job 立刻以 ok=False 结束，看着像瞬间跑完）。
        #   ⇒ 摘出来走**显式形参**，既不进命令参数面、也不丢信息。
        self._job_run = bool(spec.get(JOB_RUN_FLAG, False))
        clean_spec = {k: v for k, v in spec.items() if k != JOB_RUN_FLAG}
        if self._commands is not None and command in self._commands.names():
            return self._invoke_core_command(command, clean_spec, context)
        return self._dispatch(clean_spec, context)

    def _invoke_core_command(self, command: str, spec: Mapping[str, Any],
                             context: ExecutionContext | None) -> Mapping[str, Any]:
        """经核心 CommandRegistry 调用命令（审计/result_ref 由核心写）。"""
        args = {k: v for k, v in spec.items() if k != "command"}
        args["reason"] = f"ml_mecha:{command}"
        args["call_id"] = _call_id()
        result = self._commands.invoke(
            command, args, context=context, gate=self._command_gate,
            channel=active_channel(), approval=self._command_approval)
        if result.get("is_error"):
            failure = result.get("error") or {}
            info = failure.get("info") or {}
            payload = result.get("value")
            if isinstance(payload, Mapping) and payload:
                # 宿主回执原样返回：run_ids / warnings 不能在失败路径丢失，
                # 否则被守卫拦下的运行无法与 ML run store 对账。
                out = dict(payload)
                out.setdefault("error_kind", str(info.get("kind") or ""))
                out.setdefault("error", str(failure.get("message") or ""))
                out.setdefault("error_hint", str(info.get("hint") or ""))
                out.setdefault("error_suggest", str(info.get("suggest") or ""))
                return out
            exc = MechaError(
                str(failure.get("message") or "命令失败"),
                kind=str(info.get("kind") or "command_failed"),
                hint=str(info.get("hint") or ""),
                suggest=str(info.get("suggest") or ""))
            return self._failure(command, exc, [])
        return dict(result["value"])

    def run_core_command(self, command_name: str, *, context: Any,
                         **kwargs: Any) -> Any:
        """核心命令 handler：把受控命令体包成 ``CommandResult`` + result_ref。"""
        from mecha.commands import CommandResult

        receipt = self._dispatch({"command": command_name, **kwargs}, context)
        ref = {"kind": "ml_command", "command": command_name,
               "run_ids": list(receipt.get("run_ids", []))}
        if receipt.get("ok"):
            return CommandResult(ok=True, values=dict(receipt), result_ref=ref)
        exc = MechaError(
            str(receipt.get("error") or "命令失败"),
            kind=str(receipt.get("error_kind") or "command_failed"),
            hint=str(receipt.get("error_hint") or ""),
            suggest=str(receipt.get("error_suggest") or ""))
        return CommandResult(ok=False, values=dict(receipt), error=exc,
                             result_ref=ref)

    def _dispatch(self, spec: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
        command_hint = str(spec.get("command", "")) if isinstance(spec, Mapping) else ""
        if _is_cancelled(context):
            # 命令入口的第二道（核心 ``CommandRegistry.invoke`` 直连 handler 时
            # 走不到 ``run``）：取消态绝不做域工作。此处核心审计已由 invoke
            # 写过一条 ``ok=False`` 的记录——那是核心如实记账，不是本层伪造。
            return self._cancelled_receipt(command_hint)
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
            "run_custom": self._cmd_run_custom,
        }

    def _cmd_run_custom(self, spec: dict[str, Any], context: Any) -> Mapping[str, Any]:
        """跑一段**网络定义代码**（AI 写结构，系统管训练与观测）。

        ## 形状：它本质就是"跑一个方法"

        收编方式（用户裁决）不是另开一条执行面，而是：**临时注册**一个
        `CustomNNMethod`（它是一台真正的 `MLMethod`）→ 走与 `run_method`
        完全相同的执行路径 → 跑完**立刻撤除**注册项。

        ⇒ 白拿的东西：`Session` 的校验/快照、`runner` 的切分与评估、
        `artifacts`、`runs/` 存档、五页检视、`describe_training` 观测面、
        Gate 写权与审计。

        ## ⚠ 为什么必须"跑完就撤"

        代码是**逐请求变化**的，而 `Session.run` 只认注册表里的名字
        （`registry.get(request.method)`）。留着不撤 ⇒ 注册表里恒有一个
        **代码已经不对**的 `custom_network`，之后谁取到它都会拿到上一次的代码。
        `finally` 里撤，连失败路径也不留垃圾。
        """
        code = str(spec.get("code") or "")
        if not code.strip():
            raise MechaError("run_custom 需要 code（定义 `class Network` 的代码）",
                             kind="missing_code",
                             hint="code 是字符串；只需要实现 build(n_in, n_out, *, nn)")
        dataset_id = str(spec.get("dataset_id") or self._last_dataset_id() or "")
        self._validator.check_dataset_ref(dataset_id)
        entry = self.dataset_entry(dataset_id)
        assert entry is not None
        name = str(spec.get("name") or "").strip() or "custom"
        method_name = _custom_method_name()
        # ⚠ **先注册再按注册表取 schema**：这样覆写校验用的就是"运行时真正那个方法"
        #   的声明，而不是另抄一份；且里外只有一个来源。
        #   `method_param_schema` 回的是**dict 形状**（`clean_overrides` 要的形状），
        #   而槽里那份是 `ParamSpec` 对象 ⇒ 直接塞进 `_build_request` 会
        #   `'ParamSpec' object has no attribute 'get'`（实测踩过）。
        self._register_custom_method(code)
        try:
            request, options = self._build_request(
                {**spec, "method": method_name}, method=method_name,
                param_schema=self.method_param_schema(method_name))
            # ⭐ 代码经 extras 下传（字符串 ⇒ JSON 可序列化 ⇒ 能原样进 record.json）
            request.extras[_custom_code_key()] = code
            request.extras["network_name"] = name
            _attach_progress(request, context)
            self._write_state(
                "ml_mecha:run_custom",
                # ⚠ `current.method` 必须是**已注册的方法名**（状态校验器按注册表查，
                #   实测：写用户给的网络名会被拒 "未注册的方法 …"）。运行级那个
                #   人类可读的名字走回执与 `extras["network_name"]`，不占这个键。
                {"current.dataset_id": dataset_id,
                 "current.method": method_name,
                 "current.overrides": dict(request.overrides),
                 "current.seed": int(request.seed),
                 "current.device": clean_device(
                     options.get("device", spec.get("device", self._device))).to_state()},
                spec,
            )
            summary = self._execute_one(entry, request, call_id=_call_id(),
                                        context=context)
        finally:
            self._unregister_custom_method()
        self._append_command_event("ml.run", summary, spec)
        warning = summary.warning
        return {
            "ok": summary.ok,
            "command": "run_custom",
            "run_ids": [summary.run_id],
            "primary_metric": summary.primary_metric,
            "metrics": dict(summary.metrics),
            "warnings": [warning] if warning else [],
            "run_id": summary.run_id,
            "network_name": name,
            "elapsed_s": summary.elapsed_s,
            "error": warning,
        }

    @staticmethod
    def _register_custom_method(code: str = "") -> None:
        """把适配器临时注册进方法注册表（幂等：先撤再注册）。

        ⚠ **代码要绑在类上，不是实例上**：`Session.run` 走的是
        `registry.get(name)`，而它**无参构造**（`cls()`）⇒ 绑在实例上等于没绑
        （实测：`SandboxError: 没有拿到网络代码`）。这里造一个**带代码的子类**
        注册进去，`registry.get` 造出来的每个实例都带着它。
        """
        from ml_toolbox.core import registry

        base = _CUSTOM_SLOT.get("cls")
        if base is None:
            raise MechaError(
                "自定义网络适配器未就绪",
                kind="custom_network_unavailable",
                hint="方法层没把适配器注册进边界层的槽（`methods/neural/custom.py` "
                     "import 时会调 `register_custom_network`）",
                suggest="确认 ml_toolbox.methods 已加载，且 neural 族可用（需 torch）",
            )
        name = _custom_method_name()
        registry.unregister(name)                     # 上一次的残留先清掉
        bound = type(name, (base,), {"custom_code": str(code or ""),
                                     "name": name})
        registry.register(bound)

    @staticmethod
    def _unregister_custom_method() -> None:
        from ml_toolbox.core import registry
        registry.unregister(_custom_method_name())

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

        # ⭐ **AI 组装预处理链**（用户裁决：预处理是"自由的那一半"）。
        #   声明形状见 `core.pipeline` 的"管道装配"一节；省略整份 ⇒ 用默认链
        #   （**零行为变化**——既有调用方一行不用改）。
        pipeline = self._pipeline_from_spec(spec)
        fitted = self.session.prepare_fitted(dataset, pipeline=pipeline, diag=diag)
        # ⭐ **图结构**（可选）：边表进 `spec.aux`，供声明了 `accepts_aux` 的方法消费。
        #   为什么放 aux 而不是塞进特征表：它是"观测之间的关系"，不是特征。
        graph_facts = self._attach_graph(fitted.spec, spec, dataset)
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
            # 回执里带上"这条链长什么样"（不含代码，代码在存档里）：
            # 调用方不用再查一次就知道自己装配的结果，且**划分口径报的是真值**。
            "pipeline_chain": [{"step": c.get("step", {}).get("key", "?"),
                                "enabled": c.get("step", {}).get("enabled", True),
                                "skipped": bool(c.get("skipped"))}
                               for c in entry.spec.meta.get("chain", [])],
            "split_policy": dict(entry.spec.meta.get("split_policy") or {}),
            # 图事实（节点/边/度分布/孤立点）——带了图才有；让人和 AI 一眼看见
            # "这份数据的拓扑长什么样"（图的坑大多在这儿）。
            "graph": dict(graph_facts or {}),
            "error": "",
        }

    def _pipeline_from_spec(self, spec: Mapping[str, Any]) -> Pipeline:
        """按 `spec["pipeline"]` 声明装配一条管道；没声明就用引擎默认。

        ⚠ **自定义步骤的编译在这里做**（不在 core）：沙箱住在边界层，而 core
        不许反向依赖它（依赖方向单向：ui/methods → core）⇒ core 提供
        "声明 → 管道"的机械层，`compile_fn` 由本层注入（见 `core.pipeline` 的说明）。

        ⚠ 校验失败一律转成**可教学**错误（`StepDeclError` 带 kind/detail），
        不让它变成 `internal_error` —— 那是"你写错了"和"框架坏了"的区别。
        """
        decl = spec.get("pipeline")
        if decl in (None, {}, []):
            return self._default_pipeline
        from ml_toolbox.core.pipeline import StepDeclError, build_pipeline

        def _compile(code: str):
            from .sandbox import SandboxError, compile_step
            try:
                return compile_step(code)
            except SandboxError as exc:
                # ⚠ **沙箱的错误要保住它的 kind**：不然"你写的代码错了"会跟
                #   "框架坏了"混成一个 `internal_error`（实测被自己的判据抓到）。
                raise MechaError(exc.teaching_text(), kind=exc.kind) from exc

        try:
            return build_pipeline(decl, compile_fn=_compile)
        except StepDeclError as exc:
            raise MechaError(exc.teaching_text(), kind=exc.kind) from exc

    def _attach_graph(self, spec: Any, request: Mapping[str, Any],
                      dataset: Any) -> dict:
        """把 `request["graph"]` 声明的边表装进 `spec.aux["graph"]`（可选）。

        ## 声明形状（JSON 可序列化 ⇒ 能进存档）

        ```jsonc
        "graph": {
          "edges": [[0,1],[1,2],[2,0]],          // 内联边表（小图/示例）
          "edges_path": "edges.csv",             // 或：CSV（相对数据集根）
          "src": "src", "dst": "dst", "weight": "w",   // CSV 的列名（可省）
          "directed": false                      // false（默认）⇒ 自动补反向边
        }
        ```

        ⚠ **度数/自环等事实进回执**（`graph_facts`）：图数据的坑大多在"边指向了
        不存在的节点""有孤立点"这类事上，让它们可见比事后猜便宜得多。
        ⚠ 边索引**越界即拒绝**（不静默裁剪）——裁掉一条边等于悄悄改拓扑。
        """
        decl = request.get("graph")
        if not decl:
            return {}
        if not isinstance(decl, Mapping):
            raise MechaError("graph 需要对象", kind="bad_graph_declaration",
                             hint='例如 {"edges": [[0,1],[1,0]]}')
        import pandas as pd
        n_nodes = int(len(spec.X))
        rows: list = []
        if decl.get("edges") is not None:
            for item in decl["edges"] or []:
                if isinstance(item, Mapping):
                    rows.append((int(item["src"]), int(item["dst"]),
                                 float(item.get("weight", 1.0))))
                else:
                    seq = list(item)
                    if len(seq) < 2:
                        raise MechaError(
                            "graph.edges 的每一项至少要有 [src, dst]",
                            kind="bad_graph_edge", hint=f"收到 {item!r}")
                    rows.append((int(seq[0]), int(seq[1]),
                                 float(seq[2]) if len(seq) > 2 else 1.0))
        elif decl.get("edges_path"):
            path = self._resolve_data_path(str(decl["edges_path"]))
            frame = pd.read_csv(path)
            src_col = str(decl.get("src") or frame.columns[0])
            dst_col = str(decl.get("dst") or frame.columns[1])
            w_col = decl.get("weight") or ""
            for _, r in frame.iterrows():
                rows.append((int(r[src_col]), int(r[dst_col]),
                             float(r[w_col]) if w_col else 1.0))
        else:
            raise MechaError("graph 声明要给 `edges`（内联）或 `edges_path`（CSV）",
                             kind="bad_graph_declaration")
        edge = pd.DataFrame(rows, columns=["src", "dst", "weight"])
        if not bool(decl.get("directed", False)) and len(edge):
            rev = edge.rename(columns={"src": "dst", "dst": "src"})
            edge = pd.concat([edge, rev], ignore_index=True)
        # 越界检查（**不裁剪**：裁边等于悄悄改拓扑）
        if len(edge):
            bad = edge[(edge["src"] < 0) | (edge["src"] >= n_nodes)
                       | (edge["dst"] < 0) | (edge["dst"] >= n_nodes)]
            if len(bad):
                raise MechaError(
                    f"边表里有 {len(bad)} 条边指向不存在的节点"
                    f"（节点数 {n_nodes}，行号应在 [0, {n_nodes - 1}]）",
                    kind="graph_edge_out_of_range",
                    hint="edge_index 用的是**行号**（指向观测表的行）",
                    suggest=f"最大合法行号 {n_nodes - 1}")
        spec.aux = dict(spec.aux or {})
        spec.aux[AUX_GRAPH] = edge
        try:
            from ml_toolbox.core.contracts import Problem
            return Problem(spec, allowed=(AUX_GRAPH,)).graph_facts()
        except Exception:                                     # noqa: BLE001
            return {"n_nodes": n_nodes, "n_edges": int(len(edge))}

    def _cmd_run_method(self, spec: dict[str, Any], context: Any) -> Mapping[str, Any]:
        """在已准备的数据视图上运行一个方法。

        取消检查点：**命令入口**（:meth:`run` / :meth:`_dispatch`）与批量/对比的
        **方法边界**。单方法体内部**没有**取消检查——``Session.run`` 的一次 fit
        计算不可中断（方法层无协作式取消点），这里如实写清，不假装可中断。
        """
        dataset_id = str(spec.get("dataset_id") or self._last_dataset_id() or "")
        self._validator.check_dataset_ref(dataset_id)
        entry = self.dataset_entry(dataset_id)
        assert entry is not None
        method = str(spec.get("method", ""))
        descriptor = self._validator.check_method_ref(method, dataset_id=dataset_id)
        request, options = self._build_request(
            spec, method=method, param_schema=self.method_param_schema(method))
        _attach_progress(request, context)         # ⚠ 必须在 session.run 之前
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
        summary = self._execute_one(entry, request, call_id=_call_id(),
                                    context=context)
        self._append_command_event("ml.run", summary, spec)
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
        # ⚠ 挂一次就够：循环里每个 `RunRequest(extras=dict(request.extras))`
        #   会把钩子**复制**给每个方法（各自报自己的 epoch 进度）。
        #   必须在任何 `session.run` 之前挂（`config()` 是拷贝语义）。
        _attach_progress(request, context)
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
            item = self._execute_isolated(entries, req, call_id=_call_id(),
                                          context=context)
            runs.append(item.compact())
            if item.warning:
                warnings.append(f"{name}: {item.warning}")
            self._append_command_event("ml.run", item, spec)
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
            _attach_progress(req, context)         # ⚠ 必须在 session.run 之前
            item = self._execute_isolated(entries, req, call_id=_call_id(),
                                          context=context)
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
            self._append_command_event("ml.run", item, spec)
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
                suggest="assemble_ml_mecha(root=..., dataset_roots=[<数据根>])",
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
        # ⭐ 设备声明**真透传**给方法层（`cfg.extras["device"]`）。
        #   修的是一个"声明不生效"的真缺陷（曾经的诚实自述见模块顶部 Limitations）：
        #   顶层 `device` 一路被 `clean_device` 清洗、也被写进 `current.device`
        #   域状态，却**从不进 RunRequest** ⇒ 方法层永远收不到它，
        #   于是"AI 传了 device=cuda 而实际在 CPU 上跑"没有任何提示。
        #   ⇒ 现在它是真参数：方法层可读（`torch_mlp` 就读，见它的 `_device`）。
        #   ⚠ 不是所有方法都消费它（sklearn 族本来也没有设备概念）——
        #   这一点如实写在这里，不假装"所有方法都支持"。
        extras["device"] = device.describe()
        if guard:
            extras["resource_guard"] = {k: v for k, v in guard.items()
                                        if k != "enabled"}
            if guard.get("enabled") and not extras["resource_guard"]:
                extras["resource_guard"] = {}
        request = RunRequest(method=method, overrides=overrides, extras=extras,
                             diag=bool(spec.get("diag", False)), seed=seed,
                             # ⭐ **后台任务一律落盘**（用户裁决：别让"意外"白训练）：
                             #   job 状态只在进程内存里，权威一重启 `job_id`/`result_ref`
                             #   就没了 ⇒ 训练跑完了却**没有路标**。装配点的 submit
                             #   worker 会给后台请求打 `_JOB_RUN_FLAG`，这里据此打开
                             #   persist（调用方显式给了 persist 也照样打开）。
                             persist=bool(spec.get("persist", False)
                                          or getattr(self, "_job_run", False)))
        return request, {"device": device}

    def _execute_one(self, entry: DatasetEntry, request: RunRequest,
                     *, call_id: str,
                     context: Any = None) -> RunSummary:
        """跑一次方法。

        ⚠ 进度钩子**不在这里挂**：``RunRequest.config()`` 进 `session.run` 就拷贝了
        extras ⇒ 那时再挂已经太晚（实测假绿一次）。挂载点在三个命令体里
        （``_attach_progress``），它们的 ``context`` 与 job 的
        ``ExecutionContext`` 是同一个对象 ⇒ `read_job` 读得到。
        ``context`` 形参保留：批量/对比的**方法边界**取消检查用它。
        """
        record = self.session.run(entry.spec, request)
        return _summarize_record(record, entry.dataset_id, request)

    def _execute_isolated(self, entry: DatasetEntry, request: RunRequest,
                          *, call_id: str,
                          context: Any = None) -> RunSummary:
        """批量用：一个方法派发失败不拖垮其余方法（与 ML runner 同一纪律）。"""
        try:
            return self._execute_one(entry, request, call_id=call_id,
                                     context=context)
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

    def bind_command_registry(self, commands: Any, gate: Any,
                              approval: Any = None) -> None:
        """注入核心 ``CommandRegistry`` + ``Gate`` + ``approval``：受控命令经核心命令面调用。

        ``approval`` 是**审批通道**（框架 ADR「命令面治理闸前置与声明式 opt-in」§Decision 2）：
        ``approval_required=True`` 的命令在**任何副作用之前**要一次 one-shot 授权，
        **没接通道即拒绝**（fail closed）。⇒ 本适配层把装配点的 ``Software.approval``
        一路传下去；今天 ML **没有任何命令声明** ``approval_required=True``，
        所以这是"把闸接上电"（将来声明导出/删除类命令时它立刻生效），
        不是"多走一遍审批通道"。

        ⚠ 框架**刻意不做自动接线**（ADR §Consequences：「自动接线会让没声明审批的命令也走一遍
        审批通道，属于扩大行为面」）⇒ 注入由本装配点显式完成。
        """
        self._commands = commands
        self._command_gate = gate
        self._command_approval = approval

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

    def _cancelled_receipt(self, command: str) -> Mapping[str, Any]:
        """命令入口即被取消时的结构化失败回执（键齐 ``RECEIPT_REQUIRED_KEYS``）。

        取消发生在**任何副作用之前**，所以这里只有话，没有事实要撤销：
        不写域状态、不产生 run、不写 ``ml.run``、不写 ``command.<name>`` 审计
        （核心审计由 ``CommandRegistry.invoke`` 在 handler 之前写，因此本检查
        必须挡在 ``invoke`` **之前**——见 :meth:`run`）。
        """
        return self._failure(command, MechaError(
            "命令在入口处已被取消：本次调用没有执行",
            kind="cancelled",
            hint="取消是协作式的——命令入口与批量/对比的每个方法边界各查一次；"
                 "单次 fit 计算中不可中断（诚实声明，不假装能打断）",
            suggest="清掉取消标志后重新提交；被取消的调用不留域状态、不产生 run、"
                    "也不写命令审计",
        ), [])

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
def _is_cancelled(context: Any) -> bool:
    """读执行上下文的取消标志（``context`` 可为 None 或鸭子类型替身）。

    只在命令入口/方法边界查——取消是**协作式**的，没有抢占式中断。
    """
    return bool(getattr(context, "cancelled", False))


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


def _progress_hook(context: Any, *, run_id: str = "") -> Callable[..., bool] | None:
    """把 ``ExecutionContext`` 包成方法层能用的进度钩子。

    签名 ``(epoch, total, stage, loss=None, val_loss=None) -> bool``：
    **返回真值 = 调用方请求停止**（协作式取消的唯一落点，见 `trainer.train`）。

    ## 为什么要带数值（用户裁决）

    原先只报 ``(epoch, total, stage)`` ⇒ AI 每次轮询只看到"跑到第 13/25 轮"，
    判断"该不该早停"还得再拉一次完整曲线。带上 loss/val_loss 之后，
    **一次轮询就能决策**。数值同时被有界地存进 job 的 `epochs` 窗口（见 `_read_job`）。

    ## 为什么要带 run_id

    `result_ref`（含 `run_id`）**只在终态才有** ⇒ AI 在训练中途拿不到句柄，
    "边跑边查曲线"实际不可用。把 `run_id` 随进度一起报上来，第一次轮询就有了。

    ⚠ **绝不让"报进度失败"把训练搞崩**：本钩子吞掉自己的异常（观测是附加的，
    主结果优先）；而"返回 True"这条**不能**被吞掉——那是取消语义。
    """
    if context is None:
        return None

    def hook(epoch: int, total: int, stage: str = "",
             loss: Any = None, val_loss: Any = None) -> bool:
        try:
            t = max(1, int(total))
            fraction = min(1.0, max(0.0, (int(epoch) + 1) / t))
            label = f"{stage or 'epoch'} {int(epoch) + 1}/{t}"
            context.report_progress(fraction, label)
        except Exception:                                   # noqa: BLE001
            pass
        # 富载荷（loss / val_loss / run_id）进 sidecar：不污染框架的 (frac, stage)
        try:
            side = getattr(context, _PROGRESS_SIDECAR, None)
            if side is None:
                side = {"run_id": "", "latest": {}, "epochs": []}
                try:
                    setattr(context, _PROGRESS_SIDECAR, side)
                except Exception:                           # noqa: BLE001
                    side = None
            if side is not None:
                if run_id:
                    side["run_id"] = str(run_id)
                point = {"epoch": int(epoch)}
                if loss is not None:
                    point["loss"] = _finite_or_none(loss)
                if val_loss is not None:
                    point["val_loss"] = _finite_or_none(val_loss)
                side["latest"] = point
                side["epochs"].append(point)
                # 有界窗口：只留最近 N 条（完整曲线按需从 describe_training 拉）
                if len(side["epochs"]) > PROGRESS_EPOCH_WINDOW:
                    del side["epochs"][:-PROGRESS_EPOCH_WINDOW]
        except Exception:                                   # noqa: BLE001
            pass
        try:
            return bool(getattr(context, "cancelled", False))
        except Exception:                                   # noqa: BLE001
            return False

    return hook


def _finite_or_none(value: Any) -> float | None:
    """给进度载荷的数值把关：非有限（NaN/inf）**回 None**，不塞进 JSON。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _attach_progress(request: Any, context: Any) -> None:
    """把逐 epoch 进度钩子挂到 **RunRequest** 上——**必须在 `session.run` 之前**。

    ⚠ 实测踩过（2026-10-03，一次假绿）：钩子原本挂在 `_execute_one` 里，而那已经
    太晚 —— `Session.run` 一进来就调 `request.config()`，而 ``RunRequest.config()``
    是 **``extras=dict(self.extras)`` 的一次性拷贝**（`api.py:89-94`）⇒ 之后往
    `request.extras` 里塞东西**传不到 `cfg`**，方法层永远收不到钩子。
    症状极具欺骗性：**同步** `run_method` 那条路（`Session.run` 之外的 spy）看起来
    一切正常，只有**后台 job** 拿不到进度 ⇒ 正好是"人看得见、AI 看不见"的那个缺口。

    ⇒ 统一在三个命令体里、`_build_request` 之后立刻挂上（那里 request 刚造出来）。
    """
    hook = _progress_hook(context) if context is not None else None
    if hook is None:
        return
    # ⚠ `run_id` 必须在**这里**就取：它由 `runner.run_one` 在跑之前定死并写回
    #   `cfg.extras["run_id"]`，但那只在方法层可见；引擎这一侧要提前知道它，
    #   好让进度载荷带上句柄（否则 AI 训练中途拿不到 run_id，`result_ref` 只在终态有）。
    #   两条来源：① `request.extras` 里已有（复跑同一 run_id 的情形）；
    #   ② 没有就**按 runner 的同一规则算一个**并写进 extras（让两边逐字一致）。
    run_id = str(request.extras.get("run_id") or "").strip()
    if not run_id:
        import time as _time
        run_id = f"{request.method}-{int(_time.time() * 1000) % 10_000_000}-{request.seed}"
        request.extras["run_id"] = run_id
    request.extras["on_progress"] = _progress_hook(context, run_id=run_id)


def _chain_summary(summary: Any) -> dict[str, Any]:
    """管道步骤摘要 → 可上线形状（**剔除已知误导项**）。

    目前只剔除一处：``test_size``。`SplitStep.apply` 把它写成**步骤参数**
    （默认 0.25），而真正决定划分的是 ``Pipeline.test_size``（见
    ``pipeline.py::_split``）⇒ 两者可以不一致，留着就是让人照错的数做判断。
    真值改由 ``_query_dataset_detail`` 的 ``split_policy`` 提供（取自 Pipeline 本体）。

    ⚠ 只做"剔除 + 透传"，**不在这里补默认值**：补默认值等于把"步骤默认"当成
    "实际生效"，那是同一个错的镜像。
    """
    if not isinstance(summary, Mapping):
        return {}
    return {k: v for k, v in summary.items() if k != "test_size"}


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
    "DEFAULT_EST_SEC",
]
