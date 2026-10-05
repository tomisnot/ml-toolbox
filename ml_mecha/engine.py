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
  ⚠ 其中 **`predict_new` 是最容易误判的一个**：会话层**已有** `Session.predict`
  （`api.py:281`，GUI 的"对新数据预测"就在用它），只是**没有命令/工具入口**
  ⇒ 属于"能调但没接线"，不是"能力不存在"。接它是一条命令 + 一个工具的量。
- 数据集/管道/记录都**只在内存**（会话级）。跨进程找回要等 mecha 的
  Artifact / Job Contract 补齐（见汇报里的接口缺口清单）。
- ~~``device`` 只做声明与校验，不真的切设备~~ ⇒ **已修（2026-10-03）**：
  `device` 现在一路透传到方法层（优先级"具体 > auto"），判据
  `test_ml_mecha.py::test_device_declaration_reaches_the_method_layer` 钉住它。
"""
from __future__ import annotations

import contextvars
import hashlib
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mecha.errors import MechaError
from mecha.gate import Channel
from mecha.surface import Engine, ExecutionContext

from ml_toolbox.api import RunRequest, Session, SessionError
from ml_toolbox.core import registry
from ml_toolbox.core.contracts import (AUX_GRAPH, AUX_PAYLOAD, DataSpec,
                                       ParamValueError, is_lower_better)
from ml_toolbox.core.dataset import Dataset
from ml_toolbox.core.pipeline import Pipeline
from ml_toolbox.core.resources import (ResourceBudget, ResourceEstimate,
                                       estimate_resources)

from .commands import COMMANDS, command_map
from .ledger import MAX_DECL_BYTES, Ledger, decl_fits, fingerprint
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
    #: ⭐ **内容指纹**（`ds-<fp12>` 里的那 12 位）与它据以计算的**声明指纹**。
    #: ⚠ 必须排在 `spec` **之后**（dataclass 不许"无默认字段跟在有默认字段后面"）；
    #: 默认空 ⇒ 既有构造点一行不用改；两条都由 `prepare_dataset` 填。
    #: `fp_decl` 是"同一份数据"的**去重键**（不含拟合结果），`fp` 才进 id
    #: （再叠上 `pipeline_id`，这样库升级改了拟合链也会得到新 id，是如实而非漂移）。
    fp: str = ""
    fp_decl: str = ""
    #: 用户给的名字（**只作可读标签**，不进 id——进 id 就会破坏幂等）。
    label: str = ""

    def facts(self) -> dict[str, Any]:
        """给校验器/只读工具用的**紧凑**事实面（无数组、无大对象）。

        列名按 ``MAX_LISTED_COLUMNS`` 截断：这个 dict 会随回执进 History，
        宽表（几千列）不截断就等于把表头整个塞进事件（宪章 §8 明令禁止）。
        """
        columns = [str(c) for c in self.spec.X.columns]
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            # ⭐ 台账三件（2026-10-05）：**机器对齐靠 id（内容寻址），人读靠 label**。
            "label": self.label or self.name,
            "fp": self.fp,
            "fp_decl": self.fp_decl,
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
    #: ⭐ 失败的**可教学三件套**（2026-10-03 补；默认空 ⇒ 既有构造点一行不用改）。
    #: 为什么走 RunSummary：它是"一次运行的紧凑投影"，而回执是由它拼的
    #: ⇒ 不加在这里，engine 就只能从 `warning` 那句自由文本里反推失败性质
    #: （实测：反推不出来 ⇒ AI 拿到 `command_failed` + 万能 hint）。
    #: ⚠ **不进 `compact()`**：History 事件形状有判据钉着，加键属行为变更，
    #:   而失败归类只在**回执**这条路上需要（要进 History 时另开一笔改）。
    error_kind: str = ""
    error_hint: str = ""
    error_suggest: str = ""

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


#: 命令 → 人话兜底模板（**调用方没给 reason 时**用；`{k}` 取该命令声明的参数值）。
#:
#: ## 为什么需要（用户实测反馈）
#
# 面板"飞行记录仪"的折叠行显示的是 **`reason`**（资产里是 `reason || target`）。
# 而这里原先是 `args["reason"] = f"ml_mecha:{command}"` —— **无条件覆盖**：
# 既抹掉了调用方/AI 写下的意图，写出来的又只是"模块:命令名"这种机械串
# （每次 custom 运行都显示 `ml_mecha:eval_custom`，既不说明做了什么、也不说明为什么）。
#
# 对照另一家的做法（GP）：**"AI 可给 reason + 人话兜底"**
# （`reason=reason or f"落库引文边 {aid}"`）。本表就是那条兜底。
#
# ⚠ 兜底**只描述"这条命令干了什么"**，不替调用方编"为什么"。
_REASON_FALLBACK: dict = {
    "prepare_dataset": "准备数据集「{name}」",
    "run_method": "运行方法 {method}",
    "run_method_batch": "批量运行 {n} 个方法",
    "compare_methods": "对比 {n} 个方法",
    "run_custom": "运行自定义网络「{name}」",
    "eval_custom": "评估自定义网络（{which} 权重）",
    "clear_current_state": "清空当前配置台面",
}


def _reason_fallback(command: str, spec: Mapping[str, Any]) -> str:
    """按"这条命令干了什么"补一句人话（拿不到关键参数时退回最朴素的描述）。"""
    tpl = _REASON_FALLBACK.get(command)
    if not tpl:
        return "执行命令 " + str(command)
    ctx: dict = {"n": 0}
    try:
        if isinstance(spec.get("methods"), (list, tuple)):
            ctx["n"] = len(spec["methods"])
        for key in ("name", "method", "which"):
            if spec.get(key) not in (None, ""):
                ctx[key] = str(spec[key])
    except Exception:
        pass
    try:
        return tpl.format(**ctx)
    except Exception:
        return tpl.replace("「{name}」", "").replace("{name}", "") \
                  .replace("{method}", "").replace("{which}", "") \
                  .replace("{n}", "若干").strip()


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
                 ops_per_sec: float = DEFAULT_OPS_PER_SEC,
                 ledger_dir: str | Path | None = None) -> None:
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
        #: ⭐ **身份台账**（R32/R38）：内容寻址的 dataset id + 落盘 + 可复原。
        #: `ledger_dir` 由装配点给（`<工作区>/.ml-mecha/ml_ledger`）；不给 ⇒
        #: 台账只在内存（纯库用法/单测），**id 仍然是内容寻址的**（幂等不依赖落盘）。
        self._ledger = Ledger(ledger_dir)
        #: `fp_decl → dataset_id`（会话内的幂等索引；台账是它的落盘镜像）。
        self._fp_index: dict[str, str] = {}
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

    def _query_handler_map(self) -> dict[str, Callable[..., Any]]:
        """只读查询的 op 名 → 实现（**单一真源**：分派与参数预检共用这一张）。"""
        return {
            "method_catalog": self._query_method_catalog,
            "method_detail": self._query_method_detail,
            "dataset_detail": self._query_dataset_detail,
            "run_detail": self._query_run_detail,
            "training_detail": self._query_training_detail,
            "network_catalog": self._query_network_catalog,
            "network_source": self._query_network_source,
            "checkpoint_detail": self._query_checkpoint_detail,
        }

    def _query_checkpoint_detail(self, *, run_id: str) -> dict[str, Any]:
        """某个 run 的 **checkpoint 摘要**（只读；**不内联权重**）。

        为什么单开一条（而不是塞进 `run_detail`）：
        - `run_detail` 读的是**会话内存**，而 checkpoint 在**磁盘**上
          ⇒ 跨重启照样查得到（这正是"训练完把模型拿回来"的入口）；
        - 权重是几十万个数 ⇒ 回执只给**定位符与结构摘要**，
          真要用它就走 `eval_custom`（系统装载后交给你的代码）。
        """
        from ml_toolbox.methods.neural.custom import describe_checkpoint
        return describe_checkpoint(str(run_id or ""))

    def query_arg_spec(self, op_name: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        """某个只读查询的 ``(可给参数, 必给参数)``；op 不存在 ⇒ ``None``。

        ⚠ **从真实签名派生，不手写表**：手写的参数表一定会与实现漂移，而漂移的
        那一边（预检说"可以"，实现说"不行"）会变成运行期 `TypeError`
        —— 那条错误消息还会**泄露内部符号**（如 `MLEngine._query_method_detail`），
        既不可教学、又违反"模型可见文本只含任务概念"。派生出来的这份不会漂移。
        """
        fn = self._query_handler_map().get(str(op_name))
        if fn is None:
            return None
        import inspect
        sig = inspect.signature(fn)
        allowed, required = [], []
        for pname, p in sig.parameters.items():
            if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                continue
            allowed.append(pname)
            if p.default is p.empty:
                required.append(pname)
        return tuple(allowed), tuple(required)

    def _query_direct(self, op_name: str, **args: Any) -> Any:
        handlers = self._query_handler_map()
        fn = handlers.get(op_name)
        if fn is None:
            raise MechaError(
                f"未注册的只读查询 {op_name!r}",
                kind="unknown_query",
                hint=f"已注册查询：{sorted(handlers)}",
            )
        # ⭐ **参数预检**（同步、可教学），而不是等 Python 抛 TypeError：
        #   后者的消息里带着内部方法名，对 AI 既没用又把实现细节漏出去了。
        allowed, required = self.query_arg_spec(op_name) or ((), ())
        unknown = sorted(set(args) - set(allowed))
        if unknown:
            raise MechaError(
                f"查询 {op_name!r} 不认识参数 {unknown}",
                kind="unknown_query_arg",
                hint=f"{op_name!r} 可给：{list(allowed)}；必给：{list(required)}",
                suggest=f"例如 {{'op': {op_name!r}, 'args': "
                        f"{{{', '.join(repr(r) + ': ...' for r in required)}}}}}",
            )
        missing = sorted(set(required) - set(args))
        if missing:
            raise MechaError(
                f"查询 {op_name!r} 缺必给参数 {missing}",
                kind="missing_query_arg",
                hint=f"{op_name!r} 必给：{list(required)}；可给：{list(allowed)}",
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
            # ⭐ **台账视图**（R38 的"运行清单"，2026-10-05）：把
            #   数据（指纹） → 名字（label/aliases） → 用它的运行（run_ids）串起来。
            #   为什么挂在这里而不是另开一个 `describe_ledger` 工具：工具数每加一个
            #   都要同步 SKILL/能力表/判据，而"看这份数据的来龙去脉"本来就是
            #   `describe_dataset` 的语义——**同一件事不开第二个入口**。
            "ledger": self._dataset_ledger_view(entry),
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

    def _dataset_ledger_view(self, entry: DatasetEntry) -> dict[str, Any]:
        """这份数据在台账里的**来龙去脉**（run manifest 的一格）。

        三件事一次给全，回答"上一轮用的哪份数据、跑出了什么"：
        `fp`（内容身份）/ `label`+`aliases`（人读的名字，含**历史上用过的所有名字**）/
        `recent_runs`（用过这份数据的运行，新→旧，**有界**）。
        ⚠ 运行清单**从既有存档读**（`runs/<id>/record.json` 里的 `dataset`），
        不另建索引——否则就是第二份可以各自漂移的真值。
        """
        rec = self._ledger.lookup(entry.fp_decl) or {}
        runs: list[dict[str, Any]] = []
        # 匹配口径：**优先按内容身份**（`record.json` 的 `dataset_id`，2026-10-05 起写入）；
        # 老档没有它 ⇒ 退回到"显示名属于这份数据的别名"，且**只认属于本 fp 的名字**
        # （名字会重复/会改，所以这条路是兜底而非常规）。
        names = {entry.label or entry.name, *(rec.get("aliases") or [])}
        names.discard("")
        try:
            from ml_toolbox.core.persistence import list_records
            for meta in list_records():                       # 已按 saved_at 新→旧
                rid = str(meta.get("dataset_id") or "")
                if rid:
                    if rid != entry.dataset_id:
                        continue
                elif str(meta.get("dataset") or "") not in names:
                    continue
                primary = str(meta.get("primary_metric") or "")
                runs.append({
                    "run_id": str(meta.get("run_id") or ""),
                    "method": str(meta.get("method") or ""),
                    "primary_metric": primary,
                    "score": (meta.get("metrics") or {}).get(primary, ""),
                    "saved_at": str(meta.get("saved_at") or ""),
                    "matched_by": "dataset_id" if rid else "alias(老档无 dataset_id)",
                    "error": meta.get("error") or "",
                })
                if len(runs) >= 10:
                    break
        except Exception:                                      # noqa: BLE001
            runs = []
        return {
            "fp": entry.fp, "fp_decl": entry.fp_decl,
            "dataset_id": entry.dataset_id,
            "label": entry.label or entry.name,
            "aliases": list(rec.get("aliases") or [entry.label or entry.name]),
            "first_seen": str(rec.get("first_seen") or ""),
            "registrations": int(rec.get("registrations") or 1),
            "restorable": bool(rec.get("restorable")),
            "restore_note": str(rec.get("restore_note") or ""),
            "n_runs": len(runs), "recent_runs": runs,
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
                               limit: int = 200,
                               grad_layers: str = "summary",
                               best_by: str = "",
                               compare: Sequence[str] | None = None,
                               tags: Sequence[str] | None = None) -> dict[str, Any]:
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
        from . import curve as _curve
        from . import tb_series

        # ⭐⭐ **多 run 曲线对齐**（R40，2026-10-05）：用户实测的误判是"跨 run 在
        #   固定 epoch 上比**终值**"——他要的是**同一 tag、同一根 step 轴**的两条
        #   轨迹。⇒ 给 `compare=[run_a, run_b, …]` 就回对齐后的表（缺失处 `null`，
      #   **绝不插值**：插值会造出一段没算过的曲线）。
        compare_ids = [str(r) for r in (compare or []) if str(r or "").strip()]
        if compare_ids:
            picked = [str(t) for t in (tags or []) if str(t or "").strip()]
            series: dict[str, Any] = {}
            for rid in compare_ids:
                payload_i = tb_series.read_run(rid, limit=max(1, int(limit or 200)))
                series[rid] = dict(payload_i.get("scalars") or {})
            if not picked:
                picked = _curve.default_tags(series)
            aligned = _curve.align(series, picked, limit=max(1, int(limit or 200)))
            return {
                "run_id": compare_ids[0], "available": True, "reason": "",
                "latest": {}, "tags": picked, "histograms": [], "images": [],
                "text": [], "duplicate_steps": [],
                "compare": {"runs": compare_ids, **aligned},
                "curve": {}, "convergence": [],
                "best_by": {"tag": "", "best": {}, "by_val_loss": {},
                            "note": "compare 模式下不做单 run 的 best_by"},
            }

        # ⭐ **留空 = 最近写过训练事件的那次**（实现描述里的承诺）。
        #   ⚠ 曾经的真相：这句话只写在工具描述里，引擎是直接透传空 run_id 的
        #   ⇒ 留空一律 `available=false`（描述与行为不符）。实测抓到后补上。
        resolved = str(run_id or "")
        if not resolved:
            resolved = tb_series.latest_run_id()
            if not resolved:
                return {
                    "run_id": "", "available": False,
                    "reason": "没有任何运行写过训练过程事件（只有神经网络类方法会写）",
                    "latest": {}, "tags": [], "histograms": [], "images": [],
                    "text": [], "duplicate_steps": [], "max_points": 0,
                    "curve": {}, "convergence": [],
                    "best_by": {"tag": "", "best": {}, "by_val_loss": {},
                                "note": "没有可读的运行"},
                    "compare": {"runs": [], "tags": {},
                                "note": "没有给 compare（本条只报单 run）"},
                }
        payload = tb_series.read_run(resolved,
                                     limit=max(1, int(limit or 200)))
        out: dict[str, Any] = {
            "run_id": resolved,
            "available": bool(payload.get("available")),
            "reason": str(payload.get("reason") or ""),
            "latest": tb_series.latest_of(payload, grad_layers=grad_layers),
            "tags": sorted((payload.get("scalars") or {}).keys()),
            "histograms": sorted((payload.get("histograms") or {}) or []),
            "images": sorted((payload.get("images") or {}) or []),
            "text": sorted((payload.get("text") or {}) or []),
            # 同一 step 写了多个值时的留痕（重跑沿用同一 run_id 会走到这里）
            "duplicate_steps": payload.get("duplicate_steps") or [],
        }
        # ⭐⭐ **曲线形状 + 收敛判读**（R42/R43，**默认就给**）：用户实测的误判是
        #   "只读 `latest`（一个点）⇒ 跨 run 比终值 ⇒ 以为饱和了"。形状是**便宜**的
        #   （每 tag 六个数），所以不该等 `include_series` 才给。
        scal = payload.get("scalars") or {}
        shapes: dict[str, Any] = {}
        notes: list[dict[str, Any]] = []
        for tag, got in scal.items():
            sh = _curve.shape(got.get("steps"), got.get("values"),
                              tail=_curve.TAIL)
            if sh.get("n"):
                shapes[str(tag)] = sh
            note = _curve.convergence_note(str(tag), sh)
            if note.get("kind"):
                notes.append({"tag": str(tag), **note})
        out["curve"] = shapes
        #: ⚠ 排序：**先报"欠训练/发散"里的欠训练**（那是用户的误判点），
        #:   但两类都要在 ⇒ 一个词盖住两种病是最坏的做法。
        notes.sort(key=lambda n: (n.get("kind") != "underfit", str(n.get("tag"))))
        out["convergence"] = notes
        # ⭐ **按任意 tag 挑最佳点**（R41）：`best.pt` 是按 `val_loss` 选的，而
        #   "val_loss 还在抖、fid 早平了"时两者**不是同一个 epoch**（用户实测踩过）。
        want = str(best_by or "").strip()
        best: dict[str, Any] = {"tag": want, "best": {}, "by_val_loss": {},
                                "note": ""}
        if want:
            got = scal.get(want)
            if not got:
                best["note"] = (f"这个 run 没有 tag {want!r}；有的 tag 见 `tags`"
                                f"（注意逐层梯度是 `grad/<层名>.weight`）")
            else:
                best["best"] = _curve.best_point(got.get("steps"), got.get("values"),
                                                 tag=want)
                vl = scal.get("val_loss")
                if vl:
                    best["by_val_loss"] = _curve.best_point(
                        vl.get("steps"), vl.get("values"), tag="val_loss")
                    b1, b2 = best["best"], best["by_val_loss"]
                    if b1.get("step") is not None and b2.get("step") is not None \
                            and b1["step"] != b2["step"]:
                        best["note"] = (
                            f"⚠ 按 {want} 的最佳点在 step {b1['step']:.0f}"
                            f"（{b1['value']:.4g}），而按 val_loss 的最佳点在 step "
                            f"{b2['step']:.0f} ⇒ **两者不是同一轮**：`best.pt` 是按 "
                            f"val_loss 选的，用它做 {want} 的口径评估会不一致")
                    else:
                        best["note"] = "两者落在同一轮（这个 run 上口径一致）"
                else:
                    best["note"] = "这个 run 没写 val_loss（val_split=0？）⇒ 无从对照"
        out["best_by"] = best
        #: `compare` 是**必填键**（回执契约）⇒ 单 run 模式下也要在：空表 + 一句说明
        #: （"没给 compare"与"给了但没数据"是两件事，别混成一个空对象）。
        out["compare"] = {"runs": [], "tags": {},
                          "note": "没有给 `compare` ⇒ 本条只报单 run；"
                                  "要比多条曲线给 compare=[run_a, run_b, …]"}
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
        # ⭐ **显存预检**（2026-10-05，L3）：用户声明了预期峰值（`max_peak_mb`）而
        #   卡上明显装不下 ⇒ **可读地拒**（`cuda_oom_predicted`，带三个数字）。
        #   放在**命令入口**的理由与别的守卫一致：同步路径（run_method/run_custom）
        #   与后台路径都在这里汇合 ⇒ 一处生效、两条路同口径；而且它发生在**写域
        #   状态之前**（不留半拉子状态、不产生 run）。
        #   ⚠ 它**只认声明**、不猜你网络的峰值（猜出来的数字会假装精确）。
        from . import vram as _vram
        _vram.precheck(dict(spec))
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
        # ⭐ **保留调用方给的 reason**；没给才由本层按动作补一句人话。
        #   ⚠ 原先写死 `f"ml_mecha:{command}"` —— 那会**抹掉** AI 写下的意图，
        #   面板"飞行记录仪"折叠行上就只剩机械串（用户实测反馈的那条）。
        #   现在与另一家的做法一致："AI 可给 + 人话兜底"。
        given = args.get("reason")
        args["reason"] = (str(given).strip() if given else "") or \
            _reason_fallback(command, spec)
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
            # ⭐ **把"接不住"翻译成人话**：会话层只会说"方法不可用/配置非法"，
            #   而最常见的一种其实是**任务的标签形态不对**（NQS 需求 §3.3：
            #   "我连我到底在哪一步失败都问不出来……怀疑是自己参数写错了"）。
            #   ⇒ 这里按可判别的条件**加具体的 kind/hint**，别让它落进万能话
            #   （pitfalls A14：可教学最常死在最后一公里）。
            hint = "ML 会话拒绝了这次请求（方法不可用/配置非法）"
            kind = "session_error"
            if self._spec_has_no_label(spec):
                kind = "no_label_but_not_custom"
                hint = ("这份数据**没有标签列**，而该方法是监督任务 ⇒ 我们不知道拿什么"
                        "算损失与指标。若标签其实在**逐行负载**里（变长/复数，例如每个"
                        "样本采样出的振幅），请在 `prepare_dataset` 里显式写 "
                        "`target_kind='custom'`；若确实有标签，就在那里给 `target` 列名。")
            return self._failure(command, MechaError(
                str(exc), kind=kind, hint=hint), [])
        except ParamValueError as exc:
            # ⭐ **可教学的参数错误**：用户/AI 把参数值写错了 ⇒ 原样保住它的
            #   `kind`/`hint`/`suggest`，**绝不落到下面的 internal_error**
            #   （那是"框架坏了"，会把"我写错了"读成"这个模型不行"）。
            #   ⚠ 必须排在 `except Exception` **之前**。
            return self._failure(command, MechaError(
                str(exc), kind=exc.kind, hint=exc.hint, suggest=exc.suggest), [])
        except Exception as exc:  # noqa: BLE001 - 内部错误也要进回执，不逃逸
            # ⚠ **注册表查不到 `custom_network`** 是"方法没就绪"，不是"框架坏了"
            #   （用户实测 §2.3 的兜底要求原话："报错要能自解释 —— '自定义网络方法
            #   尚未就绪，请重试'，而不是把 59 个可用方法列一遍（那反而误导，
            #   让人以为方法名写错了）"）。
            #   ⚠ 与 `check_method_ref` 的用户输入校验**不是一回事**：走到这里说明
            #   是我们自己在跑 `custom_network` 时它不在注册表里（导入顺序/旧版竞态）。
            if _is_missing_custom_method(exc):
                return self._failure(command, MechaError(
                    "自定义网络方法（custom_network）在方法表里查不到 ⇒ "
                    "**这不是你写错了方法名**，是适配器尚未就绪",
                    kind="custom_network_not_ready",
                    hint="方法层（ml_toolbox.methods.neural.custom）没能在运行前把"
                         "适配器注册进方法表：多半是 neural 族尚未 import（需 torch）"
                         "或旧版的并发注册竞态",
                    suggest="重试一次；仍失败则确认 ml_toolbox.methods 已加载"
                            "（描述里列出的 59 个经典方法与本方法无关）"), [])
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
            "clear_current_state": self._cmd_clear_current_state,
            "eval_custom": self._cmd_eval_custom,
        }

    def _cmd_eval_custom(self, spec: dict[str, Any],
                         context: Any) -> Mapping[str, Any]:
        """**把训练好的模型交给调用方的代码**（NQS 的"外部函数口"）。

        为什么是受控命令而不是只读查询：它**在权威进程里执行调用方的代码**
        （可能与外部物理引擎交互）——与 `run_custom` 同一口径：走命令面、受审计与
        Gate 管辖。⚠ 沙箱负责"别把宿主搞崩 + 错误可读"，**不是权限边界**。
        """
        from ml_toolbox.methods.neural.custom import run_eval_code
        run_id = str(spec.get("run_id") or "").strip()
        if not run_id:
            raise MechaError("eval_custom 需要 run_id（要装载哪一次的模型）",
                             kind="missing_run_id",
                             hint="`checkpoint_detail(run_id=…)` 可先确认它有没有 checkpoint")
        code = str(spec.get("code") or "")
        if not code.strip():
            raise MechaError("eval_custom 需要 code（定义 `evaluate(model, ctx)`）",
                             kind="missing_eval_code",
                             hint="契约：`def evaluate(model, ctx) -> dict`")
        # 数据：① 复用会话里的某份数据集；② 或现场读一份新负载（"换个 N 评估"的用法）
        #
        # ⚠ **这里曾经是个确定性 bug**（反馈 §2-缺口2 定位准确）：
        #   `check_dataset_ref` 返回的是 **facts 字典**，不是带 `.spec` 的 DatasetEntry
        #   ⇒ `getattr(entry.spec, …)` 必然 `AttributeError: 'dict' object has no
        #   attribute 'spec'`。而**判据没抓到它**，因为我的判据只走了 `payload` 那条路
        #   —— "文档推荐的用法没人跑过"正是典型的假绿。现在两条路都有判据。
        payload: dict[str, Any] = {}
        ds_id = str(spec.get("dataset_id") or "").strip()
        if ds_id:
            entry = self.ensure_dataset(ds_id)          # 懒复原优先（重启后仍可用）
            if entry is None:
                self._validator.check_dataset_ref(ds_id)   # 先按既有口径校验存在性
                entry = self.dataset_entry(ds_id)
            if entry is not None:
                payload = dict(getattr(entry.spec, "aux", {}).get("payload") or {})
        decl = spec.get("payload")
        if decl:
            payload = {**payload, **self._parse_payload_decl(decl, n_rows=None)}
        # ⭐ **默认继承训练时的负载**（反馈建议 4，他两次提到）：不显式给数据时，
        #   从 checkpoint 里那份**声明**重建（`arrays` 直接用；`hdf5` 按路径重读，
        #   受同一套数据根授权管辖 ⇒ 跨重启也成立）。
        #   ⚠ 为什么这次改主意（上一轮我刻意不继承）：他的用法是"在原数据上复评"，
        #   每次重传几十 KB 是纯粹的浪费。但**继承必须响亮**——回执会写
        #   `payload_source`，免得"评估用的到底是哪份数据"变成猜的。
        payload_source = ("given" if payload else "")
        #: ⭐ **为什么是这个来源**（反馈 §3.1：他只看到"没有负载"，只能靠猜是
        #: "声明没落盘"还是"HDF5 读不到"）⇒ 把判别结果直接给出来。
        payload_source_reason = "explicit" if payload else ""
        warn: list[str] = []
        # ⚠ **"显式说了"与"没说"要区分**：调用方给了 `payload`/`dataset_id` 就是明确表态，
        #   哪怕解析出来是空的（例如 `{"arrays": {}}`）—— 那时**不去继承**，
        #   否则"我明确说了不要数据"会被悄悄改成"用训练那份"（实测踩过）。
        explicit = bool(spec.get("payload")) or bool(ds_id)
        if not payload and not explicit:
            from ml_toolbox.methods.neural.custom import load_checkpoint_meta
            meta_ck = load_checkpoint_meta(run_id, which=str(spec.get("which") or "best"))
            decl_ck = meta_ck.get("payload_decl") or {}
            if not meta_ck:
                payload_source_reason = "no_checkpoint"
            elif not decl_ck:
                # ⚠ 最常见的一种：那次训练的存档**早于**"声明随存档走"这次改动
                #   （旧 checkpoint 的键集里根本没有 `payload_decl`）。
                payload_source_reason = "no_declaration"
            else:
                try:
                    payload = self._parse_payload_decl(decl_ck, n_rows=None)
                    if not payload:
                        # ⚠ 内联负载：声明里只剩形状/dtype ⇒ **真值在存档的
                        #   `inline_payload` 里**（不让它静默失效）。
                        payload = dict(meta_ck.get("inline_payload") or {})
                    payload_source = "inherited_from_training"
                    payload_source_reason = "inherited"
                except Exception as exc:                       # noqa: BLE001
                    payload_source_reason = "declaration_unreadable"
                    warn.append(
                        f"继承训练负载失败（{type(exc).__name__}: {exc}）⇒ 这次评估"
                        "**没有负载**；要评估请显式给 `dataset_id` 或 `payload`。")
        if not payload:
            warn.append(
                "这次评估**没有负载**：若代码要读 `ctx['payload']`，请给 "
                "`dataset_id=…`（复用会话里某份数据集）或 `payload=…`（现场给新数据）。"
                "⚠ 正常情况下会**自动继承训练时的负载**；这次没有，多半是"
                "**那次训练本身没给负载**，或声明里的 HDF5 已经读不到。")
        out = run_eval_code(run_id, code, which=str(spec.get("which") or "best"),
                            payload=payload, dataset_id=ds_id)
        return {
            "ok": bool(out.get("ok")), "command": "eval_custom",
            "run_ids": [run_id] if out.get("ok") else [],
            "primary_metric": "", "metrics": dict(out.get("metrics") or {}),
            "warnings": ([out["hint"]] if out.get("hint") else []) + warn,
            "run_id": run_id, "which": out.get("which", ""),
            "extra_keys": sorted((out.get("extra") or {}).keys()),
            "n_payload_arrays": len(payload),
            "payload_keys": sorted(payload),
            # ⭐ 数据**从哪来**要看得见（given / inherited_from_training / 空），
            #   以及**为什么**是这个来源（别让 AI 靠猜 —— 反馈 §3.1）
            "payload_source": payload_source,
            "payload_source_reason": payload_source_reason,
            "note": out.get("note", ""),
            "error": out.get("error", ""),
            "error_kind": out.get("error_kind", ""),
            "error_hint": out.get("hint", ""),
        }

    def _cmd_run_custom(self, spec: dict[str, Any], context: Any) -> Mapping[str, Any]:
        """跑一段**网络定义代码**（AI 写结构，系统管训练与观测）。

        ## 形状：它本质就是"跑一个方法"

        收编方式（用户裁决）不是另开一条执行面，而是：**注册**一台真正的
        `CustomNNMethod`（装配期常驻）→ 走与 `run_method` 完全相同的执行路径。

        ⇒ 白拿的东西：`Session` 的校验/快照、`runner` 的切分与评估、
        `artifacts`、`runs/` 存档、五页检视、`describe_training` 观测面、
        Gate 写权与审计。

        ## ⚠ 为什么**常驻注册**（"跑完就撤"已被证明是错的）

        旧实现每次跑完在 `finally` 里撤除注册项，理由是"代码逐请求变化，
        留着会拿到上一次的代码"。**那个理由的前提已经不存在**：代码现在随
        `request.extras` 逐请求下传，注册项本身**不带代码**（见
        `_ensure_custom_method` 的竞态复盘）。
        ⇒ 撤除不再防任何东西，只会**撤掉并发中的另一次运行**：实测"连续提交三个
        `run_custom`，第 1 个成功、后两个死于 `未注册的方法: custom_network`"。
        """
        code = str(spec.get("code") or "")
        # ⭐ **代码可以从文件读**（反馈建议 3）：调参要跑 N 个组合时，
        #   同一份 9 KB 代码不必发 N 遍。⚠ 受**已有**的 `dataset_roots` 管辖
        #   （不新增授权面：读文件的口子只有一个）。
        code_path = str(spec.get("code_path") or "")
        # ⚠ **二选一，禁止同时给**（反馈 §2.2 实测：同时给时 `code` **静默覆盖**
        #   `code_path`，而回执又把 `code_path` 原样回显 ⇒ "看起来像读到了文件"，
        #   实际跑的是内联那段 —— 最容易骗人的那种组合。他给了 3 行占位符 +
        #   156 行的真文件，回执写着 `code_path` 与 `code_lines: 3`）。
        if code.strip() and code_path.strip():
            raise MechaError(
                "`code` 与 `code_path` **只能给一个**（同时给时无法判断你想跑哪一份）",
                kind="bad_code_source",
                hint="要跑文件里那份：只给 `code_path`；要跑内联那段：只给 `code`。"
                     "⚠ 同时给时旧行为是**静默取 code**，那会让'我改了文件'变成假的")
        code_source = "inline"
        if not code.strip() and code_path:
            code = self._read_code_file(code_path)
            code_source = "file"
        if not code.strip():
            raise MechaError("run_custom 需要 code（定义 `class Network` 的代码）"
                             "或 code_path（指向代码文件）",
                             kind="missing_code",
                             hint="code 是字符串；只需要实现 build(n_in, n_out, *, nn)")
        dataset_id = str(spec.get("dataset_id") or self._last_dataset_id() or "")
        # ⭐ 懒复原（同上）：重启后可复原的 id 就地重建，不必重新登记
        entry = self.ensure_dataset(dataset_id)
        if entry is None:
            self._validator.check_dataset_ref(dataset_id)
            entry = self.dataset_entry(dataset_id)
        assert entry is not None
        name = str(spec.get("name") or "").strip() or "custom"
        # ⭐⭐ **试跑**（反馈建议 1，他评为最高性价比）：编译 + 建模型 + 用**真实负载**
        #   跑几次一次前向，回逐模块形状轨迹与结构化归因。
        #   ⚠ 三条边界：**不反传、不落盘、不计入实验记录**（不写状态、不建 run）。
        #   为什么值得单开一条：他这次约 15 个往返都花在形状错上，而其中一类
        #   （批次相关）**只有用真实负载才抓得到**。
        if bool(spec.get("dry_run", False)):
            return self._cmd_dry_run_custom(spec, entry, code, code_path, name,
                                            context, code_source)
        method_name = _custom_method_name()
        # ⚠ **先确保注册项在**再按注册表取 schema：这样覆写校验用的就是"运行时
        #   真正那个方法"的声明，而不是另抄一份；且里外只有一个来源。
        #   `method_param_schema` 回的是**dict 形状**（`clean_overrides` 要的形状），
        #   而槽里那份是 `ParamSpec` 对象 ⇒ 直接塞进 `_build_request` 会
        #   `'ParamSpec' object has no attribute 'get'`（实测踩过）。
        # ⚠ **不撤除**（撤除 = 并发竞态的根）：代码随 extras 走，见
        #   `_ensure_custom_method` 的说明。
        self._ensure_custom_method()
        request, options = self._build_request(
            {**spec, "method": method_name}, method=method_name,
            param_schema=self.method_param_schema(method_name))
        # ⭐ 代码经 extras 下传（字符串 ⇒ JSON 可序列化 ⇒ 能原样进 record.json）
        request.extras[_custom_code_key()] = code
        # ⭐ **负载的"声明"也下传**：checkpoint 要把它存起来，`eval_custom` 才能
        #   默认继承训练时的数据（反馈建议 4）。只存声明（HDF5 路径/小数组），
        #   大数组留在原处按需读。
        #   ⚠ 声明来自**登记数据那一步**（`spec.meta`），不在本次 `run_custom`
        #     的参数里 —— 实测踩过：只看 `spec["payload"]` 会永远是空的。
        request.extras["payload_decl"] = dict(
            spec.get("payload") or entry.spec.meta.get("payload_decl") or {})
        request.extras["code_path"] = code_path
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
            # ⭐ 代码从文件读时**把路径带出去**：AI 才能把它和文件版本对齐
            #   （"我改了文件、跑的是不是新版本"是反馈里那类反复出现的疑问）
            "code_path": code_path,
            # ⭐ **这次实际用的是哪一份源码**（`file` / `inline`）——反馈 §2.2 的
            #   教训：只回显 `code_path` 会让人以为"读的是文件"。
            "code_source": code_source,
            "code_lines": len(code.splitlines()),
            "elapsed_s": summary.elapsed_s,
            "error": warning,
            # 同 `run_method`：失败的可教学三件套（自定义网络的失败同样要能教）
            "error_kind": summary.error_kind,
            "error_hint": summary.error_hint,
            "error_suggest": summary.error_suggest,
        }

    def _read_code_file(self, code_path: str) -> str:
        """从文件读网络代码（反馈建议 3：`code_path`）。

        ⚠ **受既有 `dataset_roots` 管辖**：读文件的口子只有一个（不新增授权面）。
        路径不在允许根下 ⇒ 沿用 `_resolve_data_path` 的 fail-closed 拒绝。
        """
        path = self._resolve_data_path(code_path)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="gbk")
            except (OSError, UnicodeDecodeError) as exc:
                raise MechaError(
                    f"读代码文件失败（要从 UTF-8 或 GBK 读）：{code_path}",
                    kind="code_path_unreadable",
                    hint="把源码存成 UTF-8 再试") from exc
        except OSError as exc:
            raise MechaError(
                f"读代码文件失败：{code_path}（{exc}）",
                kind="code_path_unreadable",
                hint="确认路径存在；它必须在**已授权的数据根**之下"
                     "（`dataset_roots.txt` / `--dataset-roots`）") from exc
        if not text.strip():
            raise MechaError(f"代码文件是空的：{code_path}",
                             kind="code_path_empty",
                             hint="文件里要定义 `class Network`")
        return text

    def _cmd_dry_run_custom(self, spec: dict[str, Any], entry: Any, code: str,
                            code_path: str, name: str, context: Any,
                            code_source: str = "inline"
                            ) -> Mapping[str, Any]:
        """**试跑**：不反传、不落盘、不计入实验记录（反馈建议 1）。

        为什么它不是"另开一条执行面"：它复用真跑的**同一套**准备与调用约定
        （`prepare_supervised` / `_build_ctx` / `forward_step`）⇒ 试跑通过
        **真的意味着**真跑的形状对得上。否则试跑只是另一种"看着对"。

        ⚠ 它**仍然受写权门管辖**（同一个 `run_custom` 命令）：它在权威进程里
        执行调用方的代码 —— "不写盘"不等于"无需授权"。
        """
        import torch                                          # noqa: F401
        from ml_toolbox.methods.neural import dryrun as _dry
        from ml_toolbox.methods.neural.custom import _network_of
        from ml_toolbox.core.contracts import Problem

        try:
            net = _network_of(code)
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "command": "run_custom", "dry_run": True,
                    "run_ids": [], "primary_metric": "", "metrics": {},
                    "warnings": [str(exc)], "run_id": "", "network_name": name,
                    "error": str(exc),
                    "error_kind": getattr(exc, "kind", "") or "code_error",
                    "error_hint": getattr(exc, "detail", ""),
                    "shapes": [], "batch_sizes_checked": [], "runs": [],
                    "n_params": 0, "modules": [], "code_path": code_path,
                    "code_lines": len(code.splitlines()),
                    "elapsed_s": 0.0}
        payload = dict(getattr(entry.spec, "aux", {}).get("payload") or {})
        problem = Problem(entry.spec, allowed=tuple(entry.spec.aux or ()))
        options = spec.get("overrides") or {}
        # ⚠ `method_param_schema` 回的是**列表**（每项一个 dict），不是映射
        #   （实测：当映射用会 `'list' object has no attribute 'items'`）
        defaults = {str(d["key"]): d.get("default")
                    for d in self.method_param_schema(_custom_method_name())
                    if isinstance(d, dict) and "key" in d}
        params = {**defaults, **dict(options)}
        params["batch"] = int(params.get("batch", 64) or 64)
        params["val_split"] = float(params.get("val_split", 0.2))
        t0 = time.time()
        # ⚠ 设备要**过清洗再 describe()**：`self._device` 是 DeviceSpec 对象，
        #   直接 `str()` 会得到 `devicespec(kind='auto', …)`；而 `to_state()` 回的是
        #   **dict**（`str(dict)` 同样不是 torch 认的字符串）——两条都是实测踩过的坑。
        #   `describe()` 才是那个"能给 torch 的字符串"。
        dev_state = clean_device(spec.get("device") or self._device).describe()
        out = _dry.dry_run_network(
            net=net, problem=problem, params=params, torch=torch,
            device=str(dev_state), payload=payload,
            batch_probes=spec.get("batch_probes"),
            warmup_steps=int(spec.get("warmup_steps") or 0),
            # ⭐ 回执大小可调（反馈 §2.3）：他 65 个模块 × 3 档批量 ⇒ 顶层 `shapes`
            #   与 `runs[].shapes` 各来一份，同一批数据重复两遍。
            verbosity=str(spec.get("shapes_verbosity") or "trace"))
        out.update({"command": "run_custom", "dry_run": True, "run_ids": [],
                    "primary_metric": "", "metrics": {},
                    "warnings": [], "run_id": "", "network_name": name,
                    "code_path": code_path, "code_source": code_source,
                    "code_lines": len(code.splitlines()),
                    "shapes_verbosity": str(spec.get("shapes_verbosity") or "trace"),
                    "warmup_steps": int(spec.get("warmup_steps") or 0),
                    "elapsed_s": round(time.time() - t0, 3)})
        if out.get("ok"):
            out["warnings"] = [] if out.get("loss_first_batch") is not None else [
                "这次试跑**没能算损失**（网络没有 `loss` 钩子，或 target_kind=custom）"
                "—— 形状对了，但损失语义没验"]
        return out

    @staticmethod
    def _ensure_custom_method() -> None:
        """确保 `custom_network` 在方法注册表里（**幂等；永不撤除**）。

        ## 为什么不再"注册→跑→撤"（用户实测复现的竞态，2026-10-05）

        旧实现每次 `run_custom` 造一个**带代码的子类**注册进方法注册表，
        跑完在 `finally` 里 `unregister`。并发提交两次时：

        .. code-block:: text

            A: register(code_a) ────────────► 跑 ──► finally unregister  ← 把 B 的项撤了
            B:        register(code_b) ──► get(name)  ✗ KeyError 未注册的方法

        ⇒ 使用者实测：连续提交三个 `run_custom`，**第 1 个成功、后两个**全部
        死于 `未注册的方法: custom_network`；**取消后立刻提交**也复现。
        这条路还有个更隐蔽的错法：A 撤、B 注册、A 再注册 ⇒ `方法名重复`；
        更坏的是两份**不同代码**并发时，谁后注册谁覆盖 ⇒ 可能**跑错代码**。

        ⇒ 改为：注册表里恒有一个**不带代码**的适配器（方法层 import 时已常驻
        注册，见 `methods/neural/custom.py::_register_into_registry`），
        代码随 `request.extras` 逐请求下传。这里只做**幂等兜底**——
        覆盖"方法层尚未 import 到"的导入顺序情形（那时注册表里没有它，
        而 `_CUSTOM_SLOT` 可能有），仍然**不撤除**：撤除正是竞态的根。
        """
        from ml_toolbox.core import registry

        name = _custom_method_name()
        if name in registry.names():
            return
        base = _CUSTOM_SLOT.get("cls")
        if base is None:
            raise MechaError(
                "自定义网络适配器未就绪",
                kind="custom_network_unavailable",
                hint="方法层没把适配器注册进边界层的槽（`methods/neural/custom.py` "
                     "import 时会调 `register_custom_network`）",
                suggest="确认 ml_toolbox.methods 已加载，且 neural 族可用（需 torch）",
            )
        # 注册**基类本身**（不带代码）：`registry.get(name)` 无参构造出来的实例
        # 从 `cfg.extras` 取代码 ⇒ 并发安全。
        registry.register(base)

    def _cmd_clear_current_state(self, spec: dict[str, Any],
                          context: Any) -> Mapping[str, Any]:
        """把配置台面清零：每个**已设置**的状态键写回 `None`（= 回默认）。

        ## 为什么必须有这个入口（指导意见 §1-P1-5）

        状态是跨会话残留的，而没有清零入口时，AI 是在**别人（或上一轮自己）没收拾
        的台面**上干活：上一轮留下的 `current.method` / `current.overrides` 会让
        "我以为我在跑默认配置"变成假话，而且**没有任何办法回到干净状态**。

        ## 三条纪律（照指导意见）

        1. **不另造第二条落地路径**：走同一个 `_write_state` → Gate 四步
           （写权门、校验、预检、记史），只是显式 `keep_none=True`
           —— 因为 `None` 在这条路上**有语义**（回默认），不能被默认的过滤器吃掉。
        2. **不走"删史/重置快照"旁路**：History 是 append-only 的硬纪律；
           清零是**再写一批 None**，史实完整与台面干净不冲突。
        3. **逐键留痕**：回执列出每个键的 `before` 与 `readback`
           （写前值 / 写后从快照读回的值）⇒ 调用方不必再问一次"到底清干净没有"。

        ⚠ **写的是全部状态键，不是"当前有值的那些"**：命令声明里 `writes_state`
        就是全部 8 个，而判据 `test_writes_state_declaration_matches_actual_history`
        要求"声明 == 实际写入"**双向相等**（多声明=谎报会写、少声明=偷偷写）。
        只写"有值的"会让声明变成半真——那种"声明与实现不一致"正是本仓反复在防的。
        代价是台面本来就空时也会写 8 条 None（幂等、留痕，可接受）。
        """
        gate = self._command_gate
        before = dict(getattr(gate, "snapshot", {}) or {}) if gate is not None else {}
        keys = sorted(_state_keys(self._validator))
        if keys:
            self._write_state("ml_mecha:clear_current_state",
                              {k: None for k in keys}, spec, keep_none=True)
        after = dict(getattr(gate, "snapshot", {}) or {}) if gate is not None else {}
        entries = [{"key": k, "before": _state_value_jsonable(before.get(k)),
                    "readback": _state_value_jsonable(after.get(k))} for k in keys]
        # ⚠ 区分"这次真清掉的"与"本来就是空的"：两者混在一起会让
        #   "n_cleared=N"读成"我清掉了 N 个东西"，而实际可能一个都没动过。
        cleared = [e for e in entries if e["before"] is not None]
        already = [e["key"] for e in entries if e["before"] is None]
        warnings = [f"{e['key']} 读回不是空（= {e['readback']!r}）"
                    for e in cleared if e["readback"] is not None]
        return {
            "ok": True, "command": "clear_current_state", "run_ids": [],
            "primary_metric": "", "metrics": {"n_cleared": len(cleared),
                                              "n_already_clean": len(already)},
            # ⚠ 读回不为空就**响亮 warn**：那说明有键没被真正清掉
            #   （半清状态比不清更坏 —— AI 会以为台面干净了）。
            "warnings": warnings,
            "cleared": cleared, "already_clean": already,
            "n_cleared": len(cleared), "n_already_clean": len(already),
            "readback": {e["key"]: e["readback"] for e in entries},
            "error": "",
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
        # ⭐⭐ **幂等**（R32 的正解，用户实测：每次登记返回新 id ⇒ 只能靠记忆对齐）：
        #   先把"你给了什么"算成一个**声明指纹**；命中会话索引或台账 ⇒ **直接复用**
        #   那一份（连管道都不必重新拟合），回执如实标 `reused: true`。
        fp_decl = self._dataset_fp_decl(spec, dataset, source_desc)
        hit = self._dataset_by_fp_decl(fp_decl)
        if hit is not None:
            aux = self._aux_facts(hit)
            # ⭐ **复用也要记一笔**：`aliases` 要**累加**（同一份数据用过的所有名字），
            #   `registrations` 要能看出"这份数据被登记过几次"。不记的话，用户换了名字
            #   再登记一次，台账里就查不到那个名字——那正是"只能靠记忆对齐"的病根。
            self._ledger.remember(
                fp_decl=hit.fp_decl, dataset_id=hit.dataset_id, label=name,
                facts=self._ledger_facts(hit),
                restore=self._dataset_restore_decl(spec), at=_now_iso())
            return self._prepare_receipt(
                hit, spec, reused=True,
                restore_note=("这份数据此前登记过（同指纹）⇒ 直接复用同一个 id；"
                              "没有重新拟合管道"),
                extra={"pipeline_chain": [{"step": c.get("step", {}).get("key", "?"),
                                           "enabled": c.get("step", {}).get("enabled", True),
                                           "skipped": bool(c.get("skipped"))}
                                          for c in hit.spec.meta.get("chain", [])],
                       "split_policy": dict(hit.spec.meta.get("split_policy") or {}),
                       "graph": aux["graph"], "payload": aux["payload"]})
        # ⭐ **数据形态体检**（NQS 需求 §3.3 的正解）：接不住的形态要**当场、可教学**地拒。
        #   ⚠ 实测过的两个失败：
        #   - 复数列：pipeline 里 `to_numpy(float)` **静默丢掉虚部**，回执还说"成功"
        #     （只在 stderr 有一条 ComplexWarning）⇒ 用户拿到的是被毁掉的数据；
        #   - 变长数组列（object dtype）：冒出一句裸 numpy 广播错
        #     `operands could not be broadcast together with shapes (256,) (1024,)`。
        #   两者都让用户"怀疑自己参数写错了"（他的原话），而不是知道"这个形态不支持"。
        self._validator.check_frame_shapes(dataset.frame)

        # ⭐ **AI 组装预处理链**（用户裁决：预处理是"自由的那一半"）。
        #   声明形状见 `core.pipeline` 的"管道装配"一节；省略整份 ⇒ 用默认链
        #   （**零行为变化**——既有调用方一行不用改）。
        pipeline = self._pipeline_from_spec(spec)
        fitted = self.session.prepare_fitted(dataset, pipeline=pipeline, diag=diag)
        # ⭐ **图结构**（可选）：边表进 `spec.aux`，供声明了 `accepts_aux` 的方法消费。
        #   为什么放 aux 而不是塞进特征表：它是"观测之间的关系"，不是特征。
        graph_facts = self._attach_graph(fitted.spec, spec, dataset)
        # ⭐ **逐行负载**（可选）：变长/复数的大数组进 `spec.aux["payload"]`。
        #   与图的区别：图是"行之间怎么连"，负载是"每行自带的东西"
        #   ⇒ 图要节点掩码划分，负载**按行切**就行（保住 row-level 语义）。
        payload_facts = self._attach_payload(fitted.spec, spec, dataset)
        # ⭐ 自定义任务（无标准标签，指标由方法自己上报）
        self._apply_target_kind(fitted.spec, spec)
        fp = self._dataset_fp(fp_decl, str(fitted.spec.meta.get("pipeline_id", "")))
        dataset_id = self._next_dataset_id(fp)
        # `DatasetEntry` 是 frozen ⇒ 用 `replace` 补三个台账字段（不另造构造点）。
        entry = replace(
            self._build_entry(dataset_id, dataset, fitted.spec, source_desc, diag),
            fp=fp, fp_decl=fp_decl, label=name)
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
        # ⭐ **台账落盘**：这份数据的身份、可读标签、以及"重启后怎么重建"。
        #   落盘失败**不吞**（身份错了比没有身份更贵）：让异常照常冒出去。
        record = self._ledger.remember(
            fp_decl=entry.fp_decl, dataset_id=entry.dataset_id, label=entry.label,
            facts=self._ledger_facts(entry),
            restore=self._dataset_restore_decl(spec),
            at=_now_iso())
        return self._prepare_receipt(
            entry, spec, reused=False,
            restore_note=("这份数据可跨重启懒复原（来源可重建）" if record["restorable"]
                          else f"⚠ 重启后无法自动复原：{record['restore_note']}"),
            extra={"graph": dict(graph_facts or {}),
                   "payload": dict(payload_facts or {}),
                   "pipeline_chain": record["chain"],
                   "split_policy": record["split_policy"]})

    @staticmethod
    def _ledger_facts(entry: DatasetEntry) -> dict[str, Any]:
        """进台账的**事实面**（不含代码、不含大数组）。"""
        return {"n_rows": entry.n_rows, "n_features": entry.n_features,
                "target": entry.target, "target_kind": entry.target_kind,
                "pipeline_id": entry.pipeline_id, "source": entry.source,
                "split_policy": dict(entry.spec.meta.get("split_policy") or {}),
                "chain": [{"step": c.get("step", {}).get("key", "?"),
                           "enabled": c.get("step", {}).get("enabled", True),
                           "skipped": bool(c.get("skipped"))}
                          for c in entry.spec.meta.get("chain", [])]}

    def _prepare_receipt(self, entry: DatasetEntry, spec: Mapping[str, Any], *,
                         reused: bool, restore_note: str,
                         extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """`prepare_dataset` 的回执（**新建与复用共用一份形状**）。

        ⚠ 为什么抽出来：复用那条路跳过拟合，但它必须给出**同样完整**的事实
        （`split_policy`/`pipeline_chain`/`graph`/`payload`）——只回一个 id 会让
        "复用"看起来像"少了信息的降级"。形状只有一份 ⇒ 两条路不可能漂移。
        """
        extra = dict(extra or {})
        ledger_rec = self._ledger.lookup(entry.fp_decl) or {}
        return {
            "ok": True,
            "command": "prepare_dataset",
            "run_ids": [],
            "primary_metric": "",
            "metrics": {"n_rows": entry.n_rows, "n_features": entry.n_features},
            "warnings": self._dataset_warnings(entry),
            "dataset_id": entry.dataset_id,
            # ⭐ 台账三件：**机器对齐靠 id（内容寻址），人读靠 label/aliases**
            "fp": entry.fp,
            "fp_decl": entry.fp_decl,
            "label": entry.label,
            "reused": bool(reused),
            "aliases": list(ledger_rec.get("aliases") or [entry.label]),
            "registrations": int(ledger_rec.get("registrations") or 1),
            "restorable": bool(ledger_rec.get("restorable")),
            "restore_note": restore_note,
            "pipeline_id": entry.pipeline_id,
            "dataset": entry.facts(),
            # 回执里带上"这条链长什么样"（不含代码，代码在存档里）：
            # 调用方不用再查一次就知道自己装配的结果，且**划分口径报的是真值**。
            "pipeline_chain": extra.get("pipeline_chain") or [],
            "split_policy": dict(extra.get("split_policy")
                                 or entry.spec.meta.get("split_policy") or {}),
            # 图事实（节点/边/度分布/孤立点）——带了图才有；让人和 AI 一眼看见
            # "这份数据的拓扑长什么样"（图的坑大多在这儿）。
            "graph": dict(extra.get("graph") or {}),
            # 逐行负载的形状/dtype（NQS：复振幅、位串、mask）——**别只回键名**，
            # 形状与 dtype 才是"我的复数还在不在"的答案。
            "payload": dict(extra.get("payload") or {}),
            "target_kind": str(entry.spec.target_kind or ""),
            "error": "",
        }

    def _aux_facts(self, entry: DatasetEntry) -> dict[str, dict]:
        """从**已拟合的数据视图**重算图/负载事实（复用那条路要用）。

        ⚠ 为什么可以重算：这两份事实是 `spec.aux` 的**纯函数**
        （`Problem(..., allowed=…)` 现场算），不依赖拟合时的临时变量
        ⇒ 复用分支不必再拟合一次，也能给出与新建**同样完整**的事实。
        """
        from ml_toolbox.core.contracts import Problem

        out: dict[str, dict] = {"graph": {}, "payload": {}}
        for key, allowed, field in (("graph", (AUX_GRAPH,), "graph"),
                                    ("payload", (AUX_PAYLOAD,), "payload")):
            if key not in (entry.spec.aux or {}):
                continue
            try:
                out[field] = dict(Problem(entry.spec, allowed=allowed)
                                  .__getattribute__(f"{field}_facts")())
            except Exception:                                  # noqa: BLE001
                out[field] = {}
        return out

    def _dataset_by_fp_decl(self, fp_decl: str) -> DatasetEntry | None:
        """会话里有没有同一份数据（按**声明指纹**查；这是幂等的判据）。"""
        with self._lock:
            dataset_id = self._fp_index.get(str(fp_decl))
            if dataset_id and dataset_id in self._datasets:
                return self._datasets[dataset_id]
            for entry in self._datasets.values():
                if entry.fp_decl and entry.fp_decl == str(fp_decl):
                    return entry
        return None

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

        def _edge_num(raw: Any, what: str, item: Any) -> float:
            """边表里的数字字段：**写错要可教学**（不是 `internal_error`）。

            ⚠ 实测：`{"edges": [["a","b"]]}` 原先抛裸 `ValueError`
            （`invalid literal for int() with base 10: 'a'`）⇒ 到 AI 那里是
            `internal_error`，看起来像框架坏了。而它明明是"你给的边表不是数字"。
            """
            try:
                return float(raw)
            except (TypeError, ValueError):
                raise MechaError(
                    f"边的 {what} 需要数字，收到 {raw!r}",
                    kind="bad_graph_edge",
                    hint="边表是 [[src, dst], …]（行号），可选第三项是权重；"
                         "src/dst 是**行的下标**，不是节点名字",
                    suggest='例如 {"edges": [[0, 1], [1, 2]]}',
                ) from None

        rows: list = []
        feat_cols: list[str] = []
        if decl.get("edges") is not None:
            for item in decl["edges"] or []:
                if isinstance(item, Mapping):
                    if "src" not in item or "dst" not in item:
                        raise MechaError(
                            f"边 {item!r} 缺 src/dst",
                            kind="bad_graph_edge",
                            hint="写成 {'src': 0, 'dst': 1} 或 [0, 1]",
                        )
                    # ⭐ **映射形态的边可以带任意特征键**（多分量耦合）：
                    #   `{"src":0,"dst":1,"V_re":…,"V_im":…,"dist":…}`
                    #   ⇒ 除 src/dst/weight 外的键都成为**边特征列**（`edge_attr`）。
                    #   旧行为只认 src/dst/weight，多出来的键被**静默丢掉**
                    #   —— 那会逼人把物理量压成一个数（需求原文："硬压会毁掉物理量"）。
                    extra_keys = [k for k in item
                                  if k not in ("src", "dst", "weight")]
                    for k in extra_keys:
                        if str(k) not in feat_cols:
                            feat_cols.append(str(k))
                    row = {"src": int(_edge_num(item["src"], "src", item)),
                           "dst": int(_edge_num(item["dst"], "dst", item)),
                           "weight": _edge_num(item.get("weight", 1.0), "weight", item)}
                    for k in extra_keys:
                        row[str(k)] = _edge_num(item[k], str(k), item)
                    rows.append(row)
                else:
                    seq = list(item)
                    if len(seq) < 2:
                        raise MechaError(
                            "graph.edges 的每一项至少要有 [src, dst]",
                            kind="bad_graph_edge", hint=f"收到 {item!r}")
                    rows.append((int(_edge_num(seq[0], "src", item)),
                                 int(_edge_num(seq[1], "dst", item)),
                                 _edge_num(seq[2], "weight", item)
                                 if len(seq) > 2 else 1.0))
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
        if rows and isinstance(rows[0], Mapping):
            # 映射形态：列 = src/dst/weight + **所有出现过的特征键**（列序 = 首现序）
            cols = ["src", "dst", "weight"] + [c for c in feat_cols]
            edge = pd.DataFrame([{c: r.get(c, 0.0) for c in cols} for r in rows],
                                columns=cols)
        else:
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

    def _attach_payload(self, spec: Any, request: Mapping[str, Any],
                        dataset: Any) -> dict:
        """把 `request["payload"]` 声明的**逐行负载**装进 `spec.aux["payload"]`。

        ## 声明形状（JSON 可序列化 ⇒ 能进存档）

        ```jsonc
        "payload": {
          // ① HDF5（**大数组/复数的主路径**：原生复数，且不必过 JSON）
          "hdf5": {"path": "data.h5",
                   "datasets": {"amps": "/config/amps", "bits": "/config/bits"}},
          // ② 内联数组（小数据、且**只有实数**——JSON 表达不了复数）
          "arrays": {"n_atoms": [6, 8, 10]}
        }
        ```

        ## 三条纪律

        1. **第 0 维必须 = 观测表的行数**（逐行负载的定义）——否则"切一批行"
           就对不上，且错误会以极难查的广播错形式出现。**越界即拒，不裁剪。**
        2. **保住 dtype**：复数就是复数、位串就是整数、mask 就是 bool。
           ⚠ 绝不 float 强转——那正是"复振幅被静默变成实数"的根因。
        3. 变长部分由调用方**先 pad 再配 mask**（`core.paddings.pad_rows`）；
           我们**不猜**怎么 pad，也不替它补 mask。
        """
        decl = request.get("payload")
        if not decl:
            return {}
        out = self._parse_payload_decl(decl, n_rows=int(len(spec.X)))
        if not out:
            return {}
        spec.aux = dict(spec.aux or {})
        spec.aux[AUX_PAYLOAD] = out
        # ⭐ **声明本身也留一份**（`spec.meta`）：`run_custom` 要把它写进 checkpoint，
        #   `eval_custom` 才能**默认继承训练时的数据**（反馈建议 4）。
        #   ⚠ 存的只是**声明**（HDF5 路径 / 小数组），不是大数组。
        #   ⚠⚠ **必须 JSON 安全**：这份声明会进 `request.extras`，而后台 job 会**序列化**
        #   它 ⇒ 里面留着 ndarray 会让**一次成功的运行**在 job 路径上被判失败
        #   （实测：`TypeError: Object of type ndarray is not JSON serializable`，
        #   只在 `submit_run` 路径复现、同步路径看不出来）。⇒ 内联数组只留形状/dtype。
        spec.meta = dict(spec.meta or {})
        spec.meta["payload_decl"] = _jsonable_payload_decl(decl)
        try:
            from ml_toolbox.core.contracts import Problem
            return Problem(spec, allowed=(AUX_PAYLOAD,)).payload_facts()
        except Exception:                                      # noqa: BLE001
            return {k: {"shape": None, "dtype": None} for k in out}

    def _parse_payload_decl(self, decl: Mapping[str, Any],
                            *, n_rows: int | None) -> dict[str, Any]:
        """解析负载声明 → ``{名字: 数组}``。

        ``n_rows=None`` ⇒ **不校验行数**（评估场景：负载不跟任何数据集绑定，
        例如"换个没见过的 N 评估"——那份数据是现场给的）。
        """
        if not isinstance(decl, Mapping):
            raise MechaError("payload 需要对象", kind="bad_payload_declaration",
                             hint='例如 {"hdf5": {"path": "d.h5", "datasets": {...}}}')
        out: dict[str, Any] = {}
        arrays = decl.get("arrays")
        if arrays:
            if not isinstance(arrays, Mapping):
                raise MechaError("payload.arrays 需要对象", kind="bad_payload_declaration")
            import numpy as np
            for k, v in arrays.items():
                arr = np.asarray(v)
                if arr.ndim == 0:
                    raise MechaError(
                        f"负载 {k!r} 是标量 ⇒ 它属于**列**，不属于逐行负载",
                        kind="bad_payload_array",
                        hint="逐行负载要求第 0 维 = 行数；单值请放进数据表的一列")
                if n_rows is not None and arr.shape[0] != n_rows:
                    raise MechaError(
                        f"负载 {k!r} 的第 0 维是 {arr.shape[0]}，而观测表有 {n_rows} 行",
                        kind="bad_payload_rows",
                        hint="逐行负载要求**第 0 维 = 行数**（切一批行 = 切前几维）",
                        suggest="检查是否按行排好，或是否该先转置")
                out[str(k)] = arr
        h5 = decl.get("hdf5")
        if h5:
            if not isinstance(h5, Mapping) or not h5.get("path"):
                raise MechaError("payload.hdf5 需要 {'path': …, 'datasets': {…}}",
                                 kind="bad_payload_declaration")
            from ml_toolbox.core.paddings import pack_hdf5
            path = self._resolve_data_path(str(h5["path"]))
            if n_rows is None:
                n_rows = _hdf5_first_dim(str(path), dict(h5.get("datasets") or {}))
            try:
                out.update(pack_hdf5(str(path), dict(h5.get("datasets") or {}),
                                     n_rows=n_rows))
            except MechaError:
                raise
            except Exception as exc:                           # noqa: BLE001
                raise MechaError(
                    f"读 HDF5 负载失败：{type(exc).__name__}: {exc}",
                    kind="bad_payload_hdf5",
                    hint="检查 path / datasets 里的数据集路径是否正确；"
                         "读 HDF5 需要 h5py",
                ) from exc
        return out

    def _apply_target_kind(self, spec: Any, request: Mapping[str, Any]) -> None:
        """允许 ``target_kind="custom"``：**没有标准标签**的任务（指标由方法上报）。

        为什么需要（NQS 需求）：他的"标签"是每个配置采样的 64 个**复数振幅**
        —— 变长、复数、且**在负载里**，不是表里的一列。若强行要求一个标准标签列，
        要么得造一个假标签（标准指标随之变成假话），要么就做不了。

        ⇒ 显式声明 `target_kind="custom"`：`spec.y` 保持 `None`，
        `spec.target_kind="custom"`；**标准指标不计算、也不假装**
        （`runner` 跳过 `evaluate`，回执里写明"自定义指标"）。
        """
        want = str(request.get("target_kind") or "").strip()
        if not want:
            return
        if want != "custom":
            raise MechaError(
                f"target_kind 只接受 'custom'（自动推断的两种不用写），收到 {want!r}",
                kind="bad_target_kind",
                hint="回归/分类的 kind 由数据自动推断；只有'无标准标签'才需要显式写 custom")
        spec.target_kind = "custom"
        spec.y = None

    def _spec_has_no_label(self, spec: Mapping[str, Any]) -> bool:
        """这次请求用的数据**没有标签、也没声明自定义任务**吗？

        用来把"会话层那句万能拒绝"翻译成人话（pitfalls A14：可教学最常死在最后一公里）。
        ⚠ 一切都在 try 里：这是**诊断辅助**，它自己绝不能把失败路径搞崩。
        """
        try:
            entry = self._resolve_dataset(dict(spec))
            s = entry.spec
            return (getattr(s, "y", None) is None
                    and str(getattr(s, "target_kind", "") or "") != "custom")
        except Exception:                                      # noqa: BLE001
            return False

    def _cmd_run_method(self, spec: dict[str, Any], context: Any) -> Mapping[str, Any]:
        """在已准备的数据视图上运行一个方法。

        取消检查点：**命令入口**（:meth:`run` / :meth:`_dispatch`）与批量/对比的
        **方法边界**。单方法体内部**没有**取消检查——``Session.run`` 的一次 fit
        计算不可中断（方法层无协作式取消点），这里如实写清，不假装可中断。
        """
        dataset_id = str(spec.get("dataset_id") or self._last_dataset_id() or "")
        entry = self.ensure_dataset(dataset_id)         # 懒复原优先（重启后仍可用）
        if entry is None:
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
            # ⭐ 失败的**可教学三件套**：少了它们，工具层只能退化成
            #   `command_failed` + 万能 hint（实测：AI 因此学不到怎么改）。
            "error_kind": summary.error_kind,
            "error_hint": summary.error_hint,
            "error_suggest": summary.error_suggest,
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

    def _next_dataset_id(self, fp: str) -> str:
        """内容寻址的 dataset id：``ds-<fp12>``（**同一个 fp 必然同一个 id**）。

        ⚠ **不带用户给的名字**：名字进 id 会让"同一份数据换个名字登记"裂成两个 id，
        而那正是本条要消灭的病（用户实测：每次登记返回新 id ⇒ 只能靠记忆对齐）。
        可读性由 `label`/`aliases` 承担（随回执与 `describe_dataset` 一起给）。
        """
        return f"ds-{str(fp)}"

    def _source_decl(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        """来源的**可复核声明**：既能算指纹（内容寻址），也能在可复原时照着重建。

        ## 为什么按 kind 分别处理（而不是把 `spec["source"]` 原样哈希）

        - `frame` / `arrays` 是**进程内**的大对象：声明里必须换成**内容哈希**
          （否则 `default=str` 会把它们变成 `"<DataFrame …>"` ⇒ 两份不同的数据
          指纹相同 ⇒ **幂等变成了谎**，那是比"id 每次变"更坏的错）。
        - `csv` 要带 `size`/`mtime`：源文件改了 ⇒ 指纹变 ⇒ **得到新 id**（如实），
          不假装幂等（设计文档 §3.2）。
        - `synthetic` 只由参数决定 ⇒ 声明即全部，天然可复原。
        """
        source = spec.get("source")
        if isinstance(source, Dataset):
            return {"kind": "dataset", "name": source.name,
                    "rows": int(len(source.frame))}
        if isinstance(source, pd.DataFrame):
            return self._frame_decl(source, {"kind": "frame", "inline": True})
        if not isinstance(source, Mapping):
            return {"kind": "unknown", "type": type(source).__name__}
        kind = str(source.get("kind", "")).strip().lower()
        if kind in ("frame", "dataframe"):
            frame = source.get("frame")
            if not isinstance(frame, pd.DataFrame):
                return {"kind": "frame", "bad": True}
            return self._frame_decl(frame, {"kind": "frame", "inline": True,
                                            "target": source.get("target") or ""})
        if kind == "csv":
            try:
                path = self._resolve_data_path(source.get("path"))
                stat = path.stat()
                return {"kind": "csv", "path": str(path), "size": int(stat.st_size),
                        "mtime": int(stat.st_mtime),
                        "target": source.get("target") or ""}
            except (MechaError, OSError):
                # 路径解不出/文件不在 ⇒ 用原始声明兜底（指纹仍然稳定）
                return {"kind": "csv", "path": str(source.get("path")),
                        "target": source.get("target") or ""}
        if kind == "arrays":
            return {"kind": "arrays", "x_hash": _hash_array(source.get("X")),
                    "y_hash": _hash_array(source.get("y")),
                    "name": source.get("name") or ""}
        if kind == "synthetic":
            return {"kind": "synthetic",
                    "task": str(source.get("task", "classification")),
                    "n_samples": int(source.get("n_samples", 120)),
                    "n_features": int(source.get("n_features", 4)),
                    "n_informative": int(source.get("n_informative", 2)),
                    "noise": float(source.get("noise", 0.1)),
                    "seed": clean_seed(source.get("seed", self._seed))}
        return {"kind": kind or "unknown"}

    @staticmethod
    def _frame_decl(frame: pd.DataFrame, extra: Mapping[str, Any]) -> dict[str, Any]:
        """DataFrame 的内容声明：**行内容哈希** + 形状 + 列名（顺序敏感）。"""
        import hashlib as _h

        try:
            values = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
            digest = _h.sha1(values).hexdigest()[:16]
        except Exception:                                     # noqa: BLE001
            digest = ""            # 哈希不了 ⇒ 空串（**不假装**同一个内容）
        return {**dict(extra), "hash": digest, "rows": int(len(frame)),
                "cols": [str(c) for c in frame.columns],
                "dtypes": [str(d) for d in frame.dtypes]}

    def _payload_fp_input(self, spec: Mapping[str, Any]) -> Any:
        """把 `spec["payload"]` / `spec["graph"]` 压成**内容敏感**的指纹输入。

        ⚠ 不能直接用 `_jsonable_payload_decl`：它把内联数组压成**只剩形状/dtype**
        ⇒ 两份"形状相同、数值不同"的负载会得到相同指纹（假幂等）。
        这里对数组取**内容哈希**，对路径型声明保留路径。
        """
        def _walk(node: Any) -> Any:
            if isinstance(node, Mapping):
                return {str(k): _walk(v) for k, v in sorted(node.items(),
                                                            key=lambda kv: str(kv[0]))}
            if isinstance(node, (list, tuple)):
                return [_walk(v) for v in node]
            if isinstance(node, np.ndarray) or hasattr(node, "tobytes"):
                return {"array_hash": _hash_array(node),
                        "shape": list(getattr(node, "shape", ()) or []),
                        "dtype": str(getattr(node, "dtype", ""))}
            if isinstance(node, (str, int, float, bool)) or node is None:
                return node
            return str(node)

        return {"payload": _walk(spec.get("payload")),
                "graph": _walk(spec.get("graph"))}

    def _dataset_fp_decl(self, spec: Mapping[str, Any], dataset: Any,
                         source_desc: str) -> str:
        """**去重键**：只由"你给了什么"决定，不含拟合结果。

        为什么不含 `pipeline_id`：它要靠**拟合**才得到，而幂等的价值恰恰在于
        "命中就不必再拟合一遍"。拟合结果对 id 的贡献在 `_dataset_fp` 里补上。
        """
        pipeline_decl = spec.get("pipeline")
        return fingerprint({
            "source": self._source_decl(spec),
            "source_desc": str(source_desc),
            "pipeline": pipeline_decl if decl_fits(pipeline_decl) else "too-big",
            "target": str(spec.get("target") or ""),
            "time_col": str(spec.get("time_col") or ""),
            "target_kind": str(spec.get("target_kind") or ""),
            "rows": int(len(dataset.frame)) if hasattr(dataset, "frame") else 0,
            "aux": self._payload_fp_input(spec),
        })

    def _dataset_fp(self, fp_decl: str, pipeline_id: str) -> str:
        """进 id 的指纹 = 声明指纹 ⊕ 拟合出的管道指纹（库升级改了链 ⇒ 新 id）。"""
        return fingerprint({"decl": fp_decl, "pipeline": str(pipeline_id)})

    def _dataset_restore_decl(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        """**照着重建**这份数据集所需要的一切（存进台账，供重启后懒复原）。"""
        source = spec.get("source")
        pipeline = spec.get("pipeline")
        payload = spec.get("payload")
        graph = spec.get("graph")
        restorable, note = self._restorable(source)
        big = [name for name, decl in (("pipeline", pipeline), ("payload", payload),
                                       ("graph", graph))
               if decl is not None and not decl_fits(decl)]
        if big:
            restorable, note = False, (f"声明过大（{'+'.join(big)} > "
                                       f"{MAX_DECL_BYTES} 字节）⇒ 没有存全")
        return {
            "restorable": bool(restorable), "note": note,
            "source": _jsonable_decl(source),
            "pipeline": _jsonable_decl(pipeline),
            "payload": _jsonable_decl(payload),
            "graph": _jsonable_decl(graph),
            "target": spec.get("target") or "",
            "time_col": spec.get("time_col") or "",
            "target_kind": spec.get("target_kind") or "",
            "diag": bool(spec.get("diag", False)),
        }

    @staticmethod
    def _restorable(source: Any) -> tuple[bool, str]:
        """这份来源在**新进程**里能不能按声明重建？（不能就如实说为什么）"""
        if isinstance(source, Mapping):
            kind = str(source.get("kind", "")).strip().lower()
            if kind == "csv":
                return True, ""
            if kind == "synthetic":
                return True, ""
            if kind in ("frame", "dataframe"):
                return False, "source.kind='frame' 是**进程内**的 DataFrame（没落盘）⇒ 重启后无法重建；要跨重启用 'csv'"
            if kind == "arrays":
                return False, "source.kind='arrays' 是**进程内**的数组（没落盘）⇒ 重启后无法重建；要跨重启用 'csv' 或 hdf5 负载"
        return False, ("来源是进程内对象（Dataset/DataFrame）⇒ 重启后无法重建；"
                       "要跨重启用 'csv'")

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
            if entry.fp_decl:
                # 幂等索引：同一个**声明指纹**永远指回同一个 id（台账是它的落盘镜像）
                self._fp_index[entry.fp_decl] = entry.dataset_id

    def _rollback_dataset(self, dataset_id: str) -> None:
        """写权门/校验拒绝后的回滚：内存里也不留这条数据集。"""
        with self._lock:
            entry = self._datasets.pop(dataset_id, None)
            if entry is not None and entry.fp_decl:
                self._fp_index.pop(entry.fp_decl, None)

    def ensure_dataset(self, dataset_id: str) -> DatasetEntry | None:
        """取一个数据集；**不在会话里就按台账懒复原**（R32 的"重启后自动复原"）。

        ## 为什么值得做（用户实测 §3.4）

        "重启后 `dataset_id` 全失效，必须重新 `prepare_dataset`。本轮我因此重复登记了
        七八次。"⇒ 台账里存了这份数据的**来源/管道/负载声明**，于是在**可复原**的来源
        （csv / synthetic）上，重启后第一次引用那个 id 时就地重建，用户**一行不用改**。

        ## 三条纪律

        1. **只复原可复原的**：`frame`/`arrays` 是进程内对象，重启后无从重建
           ⇒ 回 `None`，由校验器给"为什么不能 + 怎么改"的可教学错误
           （**不假装复原成功**）；
        2. **复原是纯内存动作**：不额外写域状态（`current.*` 由本次命令自己写）
           ⇒ 不会在用户的台面上留下第二次写入；
        3. **复原失败要说出来**：异常转成 `dataset_restore_failed`（带原始原因），
           而不是让它变成 `unknown_dataset`（那会让人以为"id 不存在"，而不是
           "你的源文件不在了"）。
        """
        with self._lock:
            entry = self._datasets.get(str(dataset_id))
        if entry is not None:
            return entry
        wanted = str(dataset_id or "")
        if not wanted.startswith("ds-"):
            return None
        # 台账里按 **id** 找（新进程的 fp_index 是空的 ⇒ 只能按 id 反查）
        rec = next((r for r in self._ledger.all()
                    if str(r.get("dataset_id")) == wanted), None)
        if rec is None:
            return None
        if not rec.get("restorable"):
            return None
        restore = dict(rec.get("restore") or {})
        try:
            spec = {
                "source": restore.get("source"),
                "pipeline": restore.get("pipeline"),
                "payload": restore.get("payload"),
                "graph": restore.get("graph"),
                "target": restore.get("target") or None,
                "time_col": restore.get("time_col") or None,
                "target_kind": restore.get("target_kind") or None,
                "name": str(rec.get("label") or ""),
                "diag": bool(restore.get("diag")),
            }
            dataset, source_desc = self._materialize_source(spec)
            dataset.name = str(rec.get("label") or dataset.name)
            dataset.target = str(spec["target"]) if spec["target"] else dataset.target
            time_col = spec.get("time_col")
            if time_col:
                dataset.time_col = str(time_col)
            pipeline = self._pipeline_from_spec(spec)
            fitted = self.session.prepare_fitted(dataset, pipeline=pipeline,
                                                diag=bool(spec["diag"]))
            self._attach_graph(fitted.spec, spec, dataset)
            self._attach_payload(fitted.spec, spec, dataset)
            self._apply_target_kind(fitted.spec, spec)
            entry = replace(
                self._build_entry(wanted, dataset, fitted.spec, source_desc,
                                  bool(spec["diag"])),
                fp=self._dataset_fp(str(rec.get("fp_decl") or ""),
                                    str(fitted.spec.meta.get("pipeline_id", ""))),
                fp_decl=str(rec.get("fp_decl") or ""),
                label=str(rec.get("label") or dataset.name))
            self._commit_dataset(entry)
            return entry
        except MechaError:
            raise
        except Exception as exc:                               # noqa: BLE001
            raise MechaError(
                f"数据集 {wanted} 的**懒复原**失败：{type(exc).__name__}: {exc}",
                kind="dataset_restore_failed",
                hint="台账里有这份数据的声明，但按声明重建失败了（源文件被移动/删除、"
                     "或依赖变了都在此列）",
                suggest="重新 prepare_dataset 一次（会得到同一个 id）",
            ) from exc

    def ledger_hint(self, dataset_id: str) -> str:
        """台账里关于这个 id 的**出路**（给 `unknown_dataset` 当可教学补充）。

        ⚠ 为什么要它：重启后 id 认不出时，旧回执只说"未知 dataset_id（本会话已准备的
        数据集：（空））"——用户只能靠记忆重建。现在告诉他：这份数据**台账里有**、
        能不能自动复原、以及**怎么用一次调用拿回来**。
        """
        wanted = str(dataset_id or "")
        rec = next((r for r in self._ledger.all()
                    if str(r.get("dataset_id")) == wanted), None)
        if rec is None:
            return ""
        alias = "、".join(rec.get("aliases") or []) or "（无名）"
        if rec.get("restorable"):
            return (f"⚠ 这份数据（label={alias}）在**台账里有**且**可自动复原**："
                    f"下一次在 run_method/run_custom/eval_custom 里引用 {wanted} "
                    f"时会按声明就地重建。")
        return (f"⚠ 这份数据（label={alias}）在**台账里有**，但**无法自动复原**："
                f"{rec.get('restore_note') or '来源不可重建'} ⇒ 请重新 prepare_dataset "
                f"（同样的声明会得到**同一个 id** {wanted}）")

    def _resolve_dataset(self, spec: Mapping[str, Any]) -> DatasetEntry:
        dataset_id = str(spec.get("dataset_id") or self._last_dataset_id() or "")
        # ⭐ **重启后懒复原**：id 不在会话里但台账里有且可复原 ⇒ 就地重建
        #   （用户实测：重启后必须重新登记，一晚上重复了七八次）。
        restored = self.ensure_dataset(dataset_id)
        if restored is not None:
            return restored
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
        # ⭐ **把内容身份写进数据视图的 meta**：`runner` 会把它带进 `RunRecord`
        #   ⇒ `record.json` 里留下 `dataset_id`（`dataset` 只是显示名，会重复也会改）。
        #   必须在 `session.run` **之前**（run 一开始就把 meta 读进 RunRecord）。
        entry.spec.meta = dict(entry.spec.meta or {})
        entry.spec.meta["dataset_id"] = entry.dataset_id
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
                     spec: Mapping[str, Any], *, keep_none: bool = False) -> list[Any]:
        """经注入的写口把状态键落到 Gate；没装配写口就 fail loud，不静默降级。

        ``keep_none``：默认 **丢弃值为 None 的键**（调用方的便利——"有这个值才写"，
        如 `pipeline_id` 可能为空）。但 **`None` 本身是有语义的**（= 回默认），
        `clear_current_state` 要的就是"把每个键写回 None" ⇒ 那条路必须显式 `keep_none=True`。
        ⚠ 这两种语义**必须由调用方说清**：不分明的话，"清零"会被这个过滤器
        静默吃掉（写了个空 dict、什么都不发生），而这正是最难查的一类失败。
        """
        payload = {k: v for k, v in updates.items() if keep_none or v is not None}
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


#: ML adapter 的只读查询面（注册进 ``Surface``，经 ``Surface.query`` 分派）。
#: ⚠ **这里是单一真源**：`_query_handler_map` 的键必须与它一致（有判据钉住），
#: 编排入口 `run_queries` 的"可用 op"也从它派生。
READ_QUERY_NAMES: tuple[str, ...] = (
    "method_catalog", "method_detail", "dataset_detail", "run_detail",
    "training_detail", "network_catalog", "network_source", "checkpoint_detail",
)


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


def _empty_frame():
    """一张 0 行的空表（`eval_custom` 现场读负载时当占位：那份负载不跟任何数据集绑定）。"""
    import pandas as pd
    return pd.DataFrame()


def _jsonable_payload_decl(decl: Mapping[str, Any]) -> dict:
    """把负载**声明**转成 **JSON 安全**的形式（会被后台 job 序列化）。

    ## 为什么要它（实测：一次**成功**的运行被判失败）

    `payload_decl` 会经 `request.extras` 下传，而后台 job 会把 extras/回执**序列化**
    ⇒ 声明里留着 **ndarray** 会让 `submit_run` 路径报
    `TypeError: Object of type ndarray is not JSON serializable`，
    而那次运行**其实跑成功了**（有 run_id、有进度）。⚠ 同步 `run_custom` 看不到这个错
    （同步路径不序列化 extras）⇒ 只有后台任务才会暴露。

    ⇒ 规则：`hdf5`（全是**路径与数据集名**，本来就安全）**原样保留**；
    内联 `arrays` **只留形状与 dtype**（值不需要——真值在 `spec.aux` 里供训练用，
    而"继承"要的是重读路径或小数组，后者由 checkpoint 单独携带）。
    """
    out: dict[str, Any] = {}
    h5 = decl.get("hdf5")
    if isinstance(h5, Mapping) and h5:
        out["hdf5"] = {str(k): (dict(v) if isinstance(v, Mapping) else v)
                       for k, v in h5.items()}
    arrays = decl.get("arrays")
    if isinstance(arrays, Mapping) and arrays:
        import numpy as np
        out["inline"] = {}
        for k, v in arrays.items():
            try:
                a = np.asarray(v)
                out["inline"][str(k)] = {"shape": list(a.shape),
                                         "dtype": str(a.dtype)}
            except Exception:                                  # noqa: BLE001
                out["inline"][str(k)] = {"shape": None, "dtype": None}
    return out


def _hdf5_first_dim(path: str, datasets: Mapping[str, Any]) -> int:
    """读 HDF5 里第一个数据集的第 0 维（评估场景：负载不绑数据集，行数由数据自己定）。"""
    try:
        import h5py
    except ImportError as exc:                                 # pragma: no cover
        raise MechaError("读 HDF5 需要 h5py", kind="bad_payload_hdf5",
                         hint="`pip install h5py`") from exc
    with h5py.File(str(path), "r") as f:
        for _name, h5path in dict(datasets or {}).items():
            if str(h5path) in f:
                return int(f[str(h5path)].shape[0])
    raise MechaError(
        "HDF5 里没找到任何声明的数据集",
        kind="bad_payload_hdf5",
        hint=f"声明的是 {list((datasets or {}).values())}")


def _state_value_jsonable(value: Any) -> Any:
    """状态键的值 → **可进 JSON 的形态**（`clear_current_state` 回执的 before/readback 用）。

    ⚠ 为什么不能拿 `_finite_or_none` 顶：那是**给指标数值**把关的
    （非数值一律回 None）⇒ 用它来报状态值，字符串/字典键会被统统写成 `None`，
    回执看着像"这些键本来就没设置"——**把真相说反了**（实测踩过：
    `current.dataset_id='ds-001-d'` 被报成 `None`）。
    状态键的值本来就有字符串/字典/None 三种形态，如实带出去即可。
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, Mapping):
        return {str(k): _state_value_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_state_value_jsonable(v) for v in value]
    return str(value)


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
    kind = hint = suggest = ""
    if not result.ok:
        warning = (result.error or "运行失败").splitlines()[-1][:200]
        # ⭐ 把失败归类**原样带出去**（见 `RunSummary.error_kind` 的说明）：
        #   没有它，engine 只能拿一句自由文本去猜，猜不出就退化成 `command_failed`
        #   + 万能 hint ⇒ AI 学不到任何东西。
        kind = str(getattr(result, "error_kind", "") or "")
        hint = str(getattr(result, "error_hint", "") or "")
        suggest = str(getattr(result, "error_suggest", "") or "")
    elif not metrics:
        warning = "运行成功但没有产出可比较的数值指标"
    # ⭐⭐ **CUDA 上下文被断言毒化**：一次 `device-side assert` 之后，**本进程内**所有
    #   后续 CUDA 运行都会在**第一次与显卡同步**处报同一个错（实测：栈指向
    #   `custom._fit_table` 的 `rec.set_probe(...)`，而那位置是**假的**）。
    #   ⇒ 记下**第一次**那次（它才是有信息量的），之后每次都在 hint 里点明"这是回声"，
    #   否则使用者会去修一个不存在的问题（实测烧掉一小时）。
    if not result.ok:
        # ⚠ 传**完整** error，不只 `warning`：`warning` 只取**最后一行**，而 CUDA 断言的
        #   关键字在第一行（`device-side assert triggered`）⇒ 只看最后一行认不出来
        #   （实测：判据当场红，`error_kind` 还是空串）。
        # ⭐⭐ 返回值多了第三个：**回执首行**。用户的实测反馈是"这条提示写了，但放在
        #   `hint` 末尾，我读了好几轮才注意到"——假位置/回声这类事实必须**置顶**
        #   （`warning` 就是回执的 `error`，也是 job 失败时 message 里那句话）。
        kind, hint, head = _cuda_poison_note(
            kind, hint, f"{getattr(result, 'error', '') or ''}\n{warning}",
            record.run_id)
        if head:
            # **结论置顶**（用户的实测反馈：埋在 `hint` 末尾读不到），原文末行保留
            # ——它是唯一的"机器原文"，两者都不能丢。
            warning = f"{head} ｜（原始报错末行）{warning}"
    return RunSummary(
        run_id=record.run_id, method=record.method, dataset_id=dataset_id,
        pipeline_id=record.pipeline_id, ok=bool(result.ok),
        primary_metric=str(result.primary_metric or ""),
        metrics=metrics,
        elapsed_s=round(float(getattr(result, "elapsed", 0.0) or 0.0), 4),
        warning=warning,
        error_kind=kind, error_hint=hint, error_suggest=suggest,
    )


def _now_iso() -> str:
    """本地时间的 ISO 串（台账里给人看的时间戳；秒级足够）。"""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _hash_array(value: Any) -> str:
    """数组/序列的**内容哈希**（指纹要认内容，不认形状）。

    ⚠ 形状相同、数值不同的负载**必须**得到不同指纹，否则"幂等"会变成
    "把两份不同的数据认成同一份"——那比"id 每次都变"更坏（用户会以为在跑 A，
    实际拿到的是 B 的拟合与划分）。拿不到内容就回空串（**不假装相同**）。
    """
    if value is None:
        return ""
    try:
        arr = np.ascontiguousarray(value)
        if arr.dtype == object or arr.dtype.kind in "US":
            payload = np.asarray([str(v) for v in arr.ravel()], dtype=object).tobytes()
        else:
            payload = arr.tobytes()
        return hashlib.sha1(payload).hexdigest()[:16]
    except Exception:                                          # noqa: BLE001
        try:
            return hashlib.sha1(repr(value).encode("utf-8")).hexdigest()[:16]
        except Exception:                                      # noqa: BLE001
            return ""


def _jsonable_decl(value: Any) -> Any:
    """把声明压成**能进 jsonl** 的形状（数组/DataFrame → 形状+内容哈希）。

    用途只有一处：台账里的 `restore` 声明（重启后照着重建）。
    ⚠ 内联大数组**不存真值**（会导致台账膨胀），只存形状+哈希 ⇒ 那种来源的
    `restorable` 由 `_restorable` 判为 false（**如实**，而不是存一份半真的东西）。
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): _jsonable_decl(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_decl(v) for v in value]
    if isinstance(value, np.ndarray) or hasattr(value, "tobytes"):
        return {"__array__": _hash_array(value),
                "shape": list(getattr(value, "shape", ()) or []),
                "dtype": str(getattr(value, "dtype", ""))}
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _is_missing_custom_method(exc: BaseException) -> bool:
    """这个异常是不是"注册表里没有 `custom_network`"？

    ⚠ 判据要**窄**：只认"KeyError 且消息里同时提到未注册与自定义网络"
    （`registry.get` 的 KeyError 文案是 `未注册的方法: <name>（可用: [...]）`）。
    宽判会把它当万能兜底，反而盖住真正的内部错误——那是另一种"假位置"。
    """
    if not isinstance(exc, KeyError):
        return False
    text = f"{exc}"
    return (_custom_method_name() in text
            and ("未注册" in text or "not registered" in text.lower()))


#: 本进程**首次** `device-side assert` 的留痕：`{"run_id", "at"}`（空 = 还没被毒化）。
#: ⚠ **进程级**：CUDA 上下文一旦被毒化就无法在本进程内恢复 ⇒ 这个事实必须跟着进程走，
#: 而不是跟着某一次运行（典型的"一次事故、之后全是回声"）。
_CUDA_POISON: dict[str, Any] = {}

#: 显存不足的判据串（torch/cuBLAS 各版本的措辞都要认）。
_OOM_MARKERS = ("out of memory", "outofmemoryerror", "cuda oom",
                "cublas_status_alloc_failed", "cuda_error_out_of_memory")


def _oom_headline(blob: str) -> str:
    """从 OOM 原文里抠出**数字**，拼一句能决策的首行（抠不到就只报类别）。

    ## 为什么必须一等化（用户实测 §3.3）

    6 GB 卡上他的配置峰值 5.30 GB、运行时只剩 3.6 GB 可用 ⇒ OOM，而 CUDA 层把它
    报成 `device-side assert`（栈指向 `set_probe`）⇒ "我事前完全不知道会不够"，
    事后也看不出是显存。⇒ 至少要把 OOM **认出来并置顶**，把 torch 里本来就有的
    数字（要多少 / 剩多少 / 总量）提出来，因为那三个数直接决定"减 batch 还是分块"。
    """
    import re
    want = re.search(r"tried to allocate ([\d.]+)\s*([KMGT]?i?B)", blob, re.I)
    free = re.search(r"([\d.]+)\s*([KMGT]?i?B)\s*free", blob, re.I)
    total = re.search(r"total capacity of ([\d.]+)\s*([KMGT]?i?B)", blob, re.I)

    def _fmt(m):
        return f"{m.group(1)} {m.group(2)}" if m else "?"

    return (f"⚠ 显存不足（CUDA OOM）：本次还要 {_fmt(want)}，"
            f"而卡上只剩 {_fmt(free)}（总 {_fmt(total)}）"
            f"⇒ 减 `batch`／开分块／换更小的负载。"
            f"⚠ 别再调参重试同一个配置（它一定还会 OOM）")


def _cuda_poison_note(kind: str, hint: str, warning: str,
                      run_id: str) -> tuple[str, str, str]:
    """识别 CUDA 断言/OOM 并**区分"第一次"与"回声"**，回 ``(kind, hint, headline)``。

    ## 为什么值得单列（实测代价：一小时）

    使用者遇到的现象：通过工具箱提交必在 `custom.py:238` 的 `rec.set_probe(...)`
    报 `CUDA error: device-side assert triggered` ⇒ 他把怀疑对象定成"探针路径与
    多组数组负载的交互"，并写了一份分析。**实测复核**：那一行只是**第一次与显卡同步**
    的位置（`torch.tensor(..., device=dev)` 会同步）；真正的断言发生在它**之前**
    （很可能在上一次运行里），而上下文一旦被毒化，**之后每次运行都在这里报同一个错**。

    ⇒ 本函数把这件事挑明：
    - **第一次**：记为"毒化起点"，并给"要看真位置就 `CUDA_LAUNCH_BLOCKING=1`"的指引；
    - **之后每次**：明确说"这是**上一次断言的回声**、位置是假的、**重启进程**才恢复"。

    ## 第三个返回值 = **首行**（2026-10-05 补，用户实测反馈）

    提示原先只进 `hint`（回执的次要从属字段），而使用者读的是 `error`/message
    那一行 ⇒ 他"读了好几轮才注意到"。⇒ 现在把**结论**摆到 `headline`
    （= 回执的 `error` 首行），`hint` 里保留完整解释与操作步骤。
    ⚠ OOM 优先于断言回声：OOM **不毒化**上下文，把它说成回声会让人去重启
    （而真正该做的是减 batch）。
    """
    blob = f"{warning}\n{hint}".lower()
    if any(s in blob for s in _OOM_MARKERS):
        return ("cuda_oom", (hint + "；" if hint else "") +
                "显存不足**不会**毒化 CUDA 上下文（不必重启）；"
                "减 batch 或让网络按块算，然后重跑即可。", _oom_headline(blob))
    if not any(s in blob for s in ("device-side assert", "cuda error",
                                   "cuda kernel error", "cuda_launch_blocking",
                                   "cuda_launch_blocking=1")):
        # ⚠ 没话说就回**空首行**（不是"原样回 warning"）：调用方据此决定要不要
        #   拼首行，回原值会让它把同一句话拼两遍。
        return kind, hint, ""
    if not _CUDA_POISON:
        _CUDA_POISON.update({"run_id": str(run_id), "at": time.time()})
        note = (
            "⚠ 这是一次 **CUDA 断言（device-side assert）**：报错位置**不一定**是真的出错点"
            "（CUDA 内核错误是**异步**上报的，栈会指向**下一次与显卡同步**的地方）。"
            "⇒ 真位置：设 `CUDA_LAUNCH_BLOCKING=1` 后**重启**再重跑；排查期也可先用 "
            "`device='cpu'` 确认逻辑。⚠ 本进程的 CUDA 上下文**已被毒化**：重启之前"
            "**所有** CUDA 运行都会在第一次同步处报同一个错（那个位置是假的）。")
        head = ("⚠ CUDA 断言（device-side assert）：报错位置**不一定是真出错点**"
                "（异步上报，栈指向下一次同步处）；且本进程 CUDA 上下文**已被毒化**，"
                "重启前所有 CUDA 运行都会在这里报同一个假错")
        return (kind or "cuda_device_assert"), (hint + "；" if hint else "") + note, head
    prev = str(_CUDA_POISON.get("run_id") or "?")
    note = (
        f"⚠⚠ **这是回声，不是新故障**：本进程的 CUDA 上下文早在 run `{prev}` 就已被一次 "
        f"`device-side assert` 毒化 ⇒ 当前报错的位置（例如 `custom._fit_table` 里的 "
        f"`rec.set_probe(...)`）**是假的**——它只是下一次与显卡同步的地方。"
        f"⇒ ① 立刻**重启权威/看门人**；② 要看真位置就先设 `CUDA_LAUNCH_BLOCKING=1` 再重启；"
        f"③ **别在这里改代码**（那是在修一个不存在的问题）。")
    head = (f"⚠⚠ **这是回声，不是新故障**：本进程 CUDA 上下文早在 run `{prev}` 已被 "
            f"device-side assert 毒化 ⇒ 当前位置是假的（只是下一次同步处）。"
            f"重启权威才恢复；**别在这里改代码**")
    return ((kind or "cuda_context_poisoned"),
            (hint + "；" if hint else "") + note, head)


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
