# -*- coding: utf-8 -*-
"""教学化校验：ML 状态键面、受控取值与命令 spec 的显式契约。

本模块是宪章 §5.2 / R2 的落点：``current.dataset_id`` / ``current.method`` /
``current.resource_guard`` 这类键**属于 ML adapter**，绝不允许沉淀进 mecha
核心；mecha 只知道"有个 key/value 要过校验器"。

校验分两层，各有明确的调用方：

1. :func:`make_state_validator` → ``Callable[[key, value], None]``
   形状对齐 ``mecha.gate.Validator``，交给 ``Gate`` 在**落地之前**调用；
   失败抛 ``GateDenied``，快照一个字节都不动。
2. :meth:`MLValidator.validate_spec` → 命令 spec 的取面校验（未知命令 /
   未知参数 / 缺必填参数），由引擎在**任何副作用之前**调用。

错误一律走 mecha 的 ``MechaError`` 家族（``UnknownKey`` / ``GateDenied`` /
``InvalidSpec``），保证 ``kind`` 机读、``hint`` / ``suggest`` 可教学。
"""
from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mecha.errors import GateDenied, InvalidSpec, MechaError, UnknownKey

#: ML adapter 声明并可写的状态键全表（R2：这些键只属于 ML，不进 mecha 核心）。
STATE_KEYS: frozenset[str] = frozenset({
    "current.dataset_id",
    "current.pipeline_id",
    "current.method",
    "current.overrides",
    "current.seed",
    "current.device",
    "current.resource_guard",
    "current.export_root",
})

#: 允许的计算设备声明（ML 词汇；mecha 核心不认识 device 概念）。
DEVICE_KINDS: tuple[str, ...] = ("auto", "cpu", "cuda")

#: seed 允许区间：非负整数，且能进 sklearn 的 32 位随机种子。
MAX_SEED = 2 ** 31 - 1

#: 「不限制」资源预算时显式写的哨兵值（``None`` 在 History JSON 里落成 null）。
UNLIMITED = "unlimited"


@dataclass(frozen=True)
class DeviceSpec:
    """一次运行的设备声明：种类 + 可选设备号。"""

    kind: str = "auto"
    index: int | None = None

    def to_state(self) -> dict[str, Any]:
        return {"kind": self.kind, "index": self.index}

    def describe(self) -> str:
        return self.kind if self.index is None else f"{self.kind}:{self.index}"


class EngineFacts(Protocol):
    """校验器需要的**只读事实面**（引擎实现；不反向依赖引擎模块）。

    只要求"能回答事实"，不要求引擎是实现类——测试可传最小替身。
    """

    def method_names(self) -> list[str]: ...

    def method_descriptor(self, name: str) -> Mapping[str, Any] | None: ...

    def dataset_ids(self) -> list[str]: ...

    def pipeline_ids(self) -> list[str]: ...

    def pipeline_id_for(self, dataset_id: str) -> str: ...

    def dataset_facts(self, dataset_id: str) -> Mapping[str, Any] | None: ...

    def allowed_export_roots(self) -> Sequence[str]: ...


# ---------------------------------------------------------------- 单值校验
def _as_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateDenied(
            f"{field} 需要整数，收到 {type(value).__name__}（{value!r}）",
            kind="bad_seed",
            hint="seed 写非负整数，例如 42；布尔值不算整数",
            suggest="set current.seed 42",
        )
    return int(value)


def clean_seed(value: Any, *, kind: str = "bad_seed") -> int:
    """校验并归一 seed（非负整数）。命令层与状态层共用同一口径。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MechaError(
            f"seed 需要整数，收到 {type(value).__name__}（{value!r}）",
            kind=kind,
            hint="seed 取 [0, 2**31-1] 的整数",
            suggest="seed=42",
        )
    if isinstance(value, float):
        if not math.isfinite(value) or value != int(value):
            raise MechaError(f"seed 需要整数，收到 {value!r}", kind=kind,
                             hint="小数 seed 不可复现；请给整数")
        value = int(value)
    if value < 0 or value > MAX_SEED:
        raise MechaError(
            f"seed={value} 越界（允许 0..{MAX_SEED}）",
            kind=kind,
            hint="超范围 seed 会被下游静默取模，失去可复现性",
        )
    return int(value)


def clean_device(value: Any, *, kind: str = "bad_device") -> DeviceSpec:
    """校验并归一设备声明：``"auto"`` / ``"cpu"`` / ``"cuda"`` / ``"cuda:1"``。"""
    if value is None:
        return DeviceSpec()
    if isinstance(value, DeviceSpec):
        return value
    if isinstance(value, Mapping):
        raw = value.get("kind", "auto")
        index = value.get("index")
    elif isinstance(value, str):
        raw, _, tail = value.strip().partition(":")
        index = int(tail) if tail.strip().isdigit() else None
        if tail.strip() and index is None:
            raise MechaError(
                f"device={value!r} 的设备号不是整数",
                kind=kind,
                hint="写法为 'cuda:0' / 'cpu' / 'auto'",
                suggest="device='cuda:0'",
            )
    else:
        raise MechaError(
            f"device 需要字符串或映射，收到 {type(value).__name__}",
            kind=kind,
            hint="可选值见 ml_mecha.validator.DEVICE_KINDS",
            suggest="device='cpu'",
        )
    raw = str(raw).strip().lower()
    if raw not in DEVICE_KINDS:
        raise MechaError(
            f"device={raw!r} 不在允许集合 {list(DEVICE_KINDS)}",
            kind=kind,
            hint="设备声明是 ML adapter 的词汇；不支持的值 fail loud，不静默回退",
            suggest="、".join(DEVICE_KINDS),
        )
    if index is None:
        return DeviceSpec(raw)
    idx = _as_int(index, field="device.index")
    if raw != "cuda":
        raise MechaError(
            f"device={raw!r} 不接受设备号 index={idx}",
            kind=kind,
            hint="只有 'cuda' 需要设备号（如 'cuda:1'）",
        )
    if idx < 0:
        raise MechaError(f"device index 不能为负（收到 {idx}）", kind=kind,
                         hint="GPU 编号从 0 开始")
    return DeviceSpec(raw, idx)


def _float_or_none(value: Any, *, field: str) -> float | None:
    if value is None or (isinstance(value, str) and value == UNLIMITED):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MechaError(
            f"{field} 需要非负有限数、'{UNLIMITED}' 或 null，收到 {value!r}",
            kind="bad_resource_guard",
            hint="数值上限用数字；显式不限用 'unlimited'（诚实声明，不靠省略）",
        )
    out = float(value)
    if not math.isfinite(out) or out < 0:
        raise MechaError(
            f"{field}={value!r} 必须是非负有限数",
            kind="bad_resource_guard",
            hint="负值/NaN/inf 会让守卫失去判别力",
        )
    return out


def clean_resource_guard(value: Any, *,
                         kind: str = "bad_resource_guard") -> dict[str, Any]:
    """校验资源守卫声明，产出**落史安全**的 dict（值只能是数 / None / str）。

    形状与 ``ml_toolbox.core.resources.ResourceBudget`` 对齐，并额外接受
    ``safety_factor`` 与 ``n_qubits``（runner 的 ``_resource_guard`` 消费它们）。
    """
    if value is None or value is False:
        return {}
    if value is True:
        return {"enabled": True}
    if not isinstance(value, Mapping):
        raise MechaError(
            f"resource_guard 需要映射、true/false 或 null，收到 {type(value).__name__}",
            kind=kind,
            hint="示例 {'max_kernel_mb': 128, 'max_operation_units': 1e11}",
        )
    limits = ("max_kernel_mb", "max_working_set_mb", "max_quantum_state_mb",
              "max_operation_units")
    unknown = sorted(set(value) - set(limits) - {"safety_factor", "n_qubits",
                                                 "enabled"})
    if unknown:
        raise MechaError(
            f"resource_guard 含未知字段 {unknown}",
            kind=kind,
            hint=f"可写字段：{list(limits)} + safety_factor / n_qubits",
            suggest="max_kernel_mb",
        )
    out: dict[str, Any] = {}
    for name in limits:
        if name in value:
            limit = _float_or_none(value[name], field=name)
            out[name] = UNLIMITED if limit is None else limit
    if "safety_factor" in value:
        sf = _float_or_none(value["safety_factor"], field="safety_factor")
        if sf is None or sf <= 0:
            raise MechaError(
                "safety_factor 必须是正有限数",
                kind=kind,
                hint="safety_factor 是估算放大系数，1.0=不放大",
            )
        out["safety_factor"] = sf
    if "n_qubits" in value:
        nq = _as_int(value["n_qubits"], field="n_qubits")
        if nq < 0:
            raise MechaError("n_qubits 不能为负", kind=kind,
                             hint="量子模拟比特数从 0 起")
        out["n_qubits"] = nq
    if "enabled" in value:
        out["enabled"] = bool(value["enabled"])
    return out


def clean_overrides(value: Any, *, schema: Sequence[Mapping[str, Any]] | None = None,
                    kind: str = "bad_overrides") -> dict[str, Any]:
    """校验运行级覆写：必须是映射；给出参数 schema 时拒绝未知参数名。"""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MechaError(
            f"overrides 需要映射，收到 {type(value).__name__}",
            kind=kind,
            hint="示例 {'C': 1.0, 'max_iter': 200}",
        )
    out = dict(value)
    for k in out:
        if not isinstance(k, str) or not k.strip():
            raise MechaError(f"覆写参数名必须是非空字符串，收到 {k!r}", kind=kind,
                             hint="参数名与方法的 param_schema.key 对齐")
    if schema is not None:
        known = {str(p.get("key")) for p in schema}
        unknown = sorted(set(out) - known)
        if unknown:
            raise UnknownKey.typo(
                unknown[0], sorted(known),
                message=f"覆写参数 {unknown} 不在该方法声明的参数表里",
                hint="可用参数见 describe_method；参数名拼错会被静默忽略是最坏情况",
            )
    return out


def clean_export_root(value: Any, *, allowed_roots: Sequence[str]) -> str:
    """校验导出目录：必须落在宿主声明的白名单根之内（防任意文件导出）。"""
    if value in (None, ""):
        return ""
    if not isinstance(value, (str, os.PathLike)):
        raise MechaError(
            f"export_root 需要路径字符串，收到 {type(value).__name__}",
            kind="bad_export_root",
            hint="导出目录必须落在宿主声明的白名单根内",
        )
    roots = [Path(r).resolve() for r in allowed_roots]
    if not roots:
        raise GateDenied(
            "宿主未声明任何导出白名单根，拒绝任何导出目录",
            kind="export_root_not_allowed",
            hint="装配时传 allowed_export_roots；不声明即不允许导出（fail closed）",
        )
    candidate = Path(value).expanduser().resolve()
    for root in roots:
        if candidate == root or root in candidate.parents:
            return str(candidate)
    raise GateDenied(
        f"export_root={str(value)!r} 不在白名单根内",
        kind="export_root_not_allowed",
        hint=f"允许的根：{[str(r) for r in roots]}",
        suggest=str(roots[0]),
    )


# ---------------------------------------------------------------- 校验器
class MLValidator:
    """ML 状态键 + 命令 spec 的校验器。

    ``facts`` 提供只读事实面（方法表 / 已准备数据集 / 白名单根），因此校验
    可以给出**可教学**的失败（"方法名拼错 → 你是不是想写 logistic"），而不是
    一句"非法值"。
    """

    def __init__(self, facts: EngineFacts, *,
                 allowed_export_roots: Sequence[str] = (),
                 max_batch_methods: int = 8,
                 max_batch_rows: int = 200_000) -> None:
        self._facts = facts
        self._allowed_export_roots = tuple(str(r) for r in allowed_export_roots)
        self._max_batch_methods = int(max_batch_methods)
        self._max_batch_rows = int(max_batch_rows)

    # ---- 事实面 ----

    @property
    def allowed_export_roots(self) -> tuple[str, ...]:
        return self._allowed_export_roots

    @property
    def max_batch_methods(self) -> int:
        return self._max_batch_methods

    # ---- Gate 注入的 validate（key, value） ----

    def validate_state(self, key: str, value: Any) -> None:
        """``mecha.gate.Validator`` 实现：只认 ML 状态键；命令键当场拒。"""
        if not isinstance(key, str) or not key.strip():
            raise UnknownKey.typo(key, STATE_KEYS,
                                  message="状态键必须是非空字符串",
                                  hint="可写键见 MLValidator.STATE_KEYS")
        if key.startswith("ml.") or key.startswith("command."):
            raise GateDenied(
                f"键 {key!r} 是命令/事件词汇，不是域状态键",
                kind="key_not_writable",
                hint="命令副作用走引擎命令面；域状态键见 STATE_KEYS",
                suggest="current.method",
            )
        if key not in STATE_KEYS:
            raise UnknownKey.typo(
                key, sorted(STATE_KEYS),
                message=f"键 {key!r} 不在 ML 可写清单里",
                hint="ML 状态键由 ml_mecha.validator.STATE_KEYS 声明"
                     "（这些键不进 mecha 核心）",
            )
        self._validate_value(key, value)

    def _validate_value(self, key: str, value: Any) -> None:
        if key == "current.dataset_id":
            self._check_dataset_id(value)
        elif key == "current.pipeline_id":
            self._check_pipeline_id(value)
        elif key == "current.method":
            self._check_method(value)
        elif key == "current.overrides":
            self._check_overrides_state(value)
        elif key == "current.seed":
            clean_seed(value)
        elif key == "current.device":
            clean_device(value)
        elif key == "current.resource_guard":
            clean_resource_guard(value)
        elif key == "current.export_root":
            clean_export_root(value, allowed_roots=self._allowed_export_roots)

    def _check_dataset_id(self, value: Any) -> None:
        if value in (None, ""):
            return
        if not isinstance(value, str):
            raise MechaError(
                f"dataset_id 需要字符串，收到 {type(value).__name__}",
                kind="bad_dataset_id",
                hint="dataset_id 由 prepare_dataset 回执给出",
            )
        if value not in self._facts.dataset_ids():
            raise MechaError(
                f"未知 dataset_id {value!r}",
                kind="unknown_dataset",
                hint=f"本会话已准备的数据集：{self._facts.dataset_ids() or '（空）'}",
                suggest="先调 prepare_dataset",
            )

    def _check_pipeline_id(self, value: Any) -> None:
        if value in (None, ""):
            return
        if not isinstance(value, str):
            raise MechaError(
                f"pipeline_id 需要字符串，收到 {type(value).__name__}",
                kind="bad_pipeline_id",
                hint="pipeline_id 由 prepare_dataset / describe_dataset 回执给出",
            )
        known = self._facts.pipeline_ids()
        if value not in known:
            raise MechaError(
                f"未知 pipeline_id {value!r}",
                kind="unknown_pipeline",
                hint=f"本会话已拟合的管道指纹：{known or '（空）'}",
                suggest="先调 prepare_dataset 让管道在训练段上拟合",
            )

    def _check_method(self, value: Any) -> None:
        if value in (None, ""):
            return
        if not isinstance(value, str):
            raise MechaError(
                f"method 需要字符串，收到 {type(value).__name__}",
                kind="bad_method",
                hint="方法名见 describe_methods",
            )
        self.check_method_ref(value)

    def _check_overrides_state(self, value: Any) -> None:
        """状态层只要求形状正确；具体参数名等有 method 时再对表。"""
        clean_overrides(value, schema=None)

    # ---- 命令层可复用的具名校验 ----

    def check_method_ref(self, name: str, *, dataset_id: str | None = None) -> Mapping[str, Any]:
        """方法名必须是已注册方法，且能处理当前数据视图（如果给了 dataset）。"""
        descriptor = self._facts.method_descriptor(name)
        if descriptor is None:
            raise UnknownKey.typo(
                name, self._facts.method_names(),
                message=f"未注册的方法 {name!r}",
                hint="方法名必须能在 ML 方法注册表里查到（不猜、不近似匹配）",
            )
        if dataset_id:
            facts = self._facts.dataset_facts(dataset_id)
            if facts is not None:
                if not facts.get("y_available") and descriptor.get("task") == "supervised":
                    raise MechaError(
                        f"方法 {name!r} 是监督方法，但数据集 {dataset_id!r} 没有目标列",
                        kind="method_cannot_handle",
                        hint="监督方法需要 prepare_dataset 时声明 target 列",
                        suggest="prepare_dataset target='<列名>'",
                    )
                allowed = facts.get("target_kinds") or []
                tk = descriptor.get("target_kind")
                if allowed and tk and tk not in allowed:
                    raise MechaError(
                        f"方法 {name!r} 只处理 target_kind={tk!r}，"
                        f"数据集 {dataset_id!r} 推断为 {allowed[0]!r}",
                        kind="method_cannot_handle",
                        hint="目标类型不匹配的方法在该数据上会直接失败，先换方法",
                    )
        return descriptor

    def check_dataset_ref(self, dataset_id: str) -> Mapping[str, Any]:
        if not isinstance(dataset_id, str) or not dataset_id:
            raise MechaError(
                f"dataset_id 必须是非空字符串，收到 {dataset_id!r}",
                kind="bad_dataset_id",
                hint="dataset_id 由 prepare_dataset 回执给出",
            )
        facts = self._facts.dataset_facts(dataset_id)
        if facts is None:
            raise MechaError(
                f"未知 dataset_id {dataset_id!r}",
                kind="unknown_dataset",
                hint=f"本会话已准备的数据集：{self._facts.dataset_ids() or '（空）'}",
                suggest="先调 prepare_dataset",
            )
        return facts

    def check_run_ref(self, run_id: str) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise MechaError(
                f"run_id 必须是非空字符串，收到 {run_id!r}",
                kind="bad_run_ref",
                hint="run_id 由 run_method 回执给出（大对象一律用引用）",
            )

    def check_run_count(self, count: int) -> None:
        """批量方法的数量上限：中风险操作要有闸，不是"能跑多少跑多少"。"""
        if count > self._max_batch_methods:
            raise GateDenied(
                f"一次批量请求 {count} 个方法，超过上限 {self._max_batch_methods}",
                kind="batch_too_large",
                hint="批量是闸控操作；拆成多次、或由人类侧调整 max_batch_methods",
            )

    def check_row_count(self, n_rows: int) -> None:
        if n_rows > self._max_batch_rows:
            raise GateDenied(
                f"数据行数 {n_rows} 超过单次运行上限 {self._max_batch_rows}",
                kind="dataset_too_large",
                hint="大表先抽样或降维；上限由宿主装配时声明",
            )

    def check_export_root(self, value: Any) -> str:
        return clean_export_root(value, allowed_roots=self._allowed_export_roots)

    # ---- 命令 spec 取面校验 ----

    def validate_spec(self, spec: Mapping[str, Any], *, commands: Mapping[str, Any]) -> str:
        """校验命令 spec，返回命令名；失败抛 ``InvalidSpec``（在任何副作用之前）。"""
        if not isinstance(spec, Mapping):
            raise InvalidSpec(
                f"run 的 spec 需要映射（含 command 键），收到 {type(spec).__name__}",
                kind="bad_spec",
                hint="示例 {'command': 'run_method', 'method': 'logistic', ...}",
            )
        command = spec.get("command")
        if not isinstance(command, str) or not command.strip():
            raise InvalidSpec(
                "spec 缺少非空字符串键 'command'",
                kind="missing_command",
                hint=f"受控命令：{sorted(commands)}",
                suggest="command='run_method'",
            )
        declared = commands.get(command)
        if declared is None:
            raise UnknownKey.typo(
                command, sorted(commands),
                message=f"未知命令 {command!r}",
                kind="unknown_command",
                hint="命令面由 ml_mecha.engine 声明；未声明的命令 fail loud，不猜测",
            )
        params = {p.name: p for p in declared.parameters}
        unknown = sorted(set(spec) - {"command"} - set(params))
        if unknown:
            raise UnknownKey.typo(
                unknown[0], sorted(params),
                message=f"命令 {command!r} 不接受参数 {unknown}",
                hint="未知参数会被静默忽略是最坏情况；参数表见 schema()",
            )
        missing = [name for name, p in params.items()
                   if p.required and name not in spec]
        if missing:
            raise InvalidSpec(
                f"命令 {command!r} 缺必填参数 {missing}",
                kind="missing_arguments",
                hint="必填参数见 schema()['commands'] 的 required 标记",
            )
        return command

    def runnable_method_names(self, dataset_id: str | None = None) -> list[str]:
        """给只读工具用：当前数据视图上可跑的方法（稳定排序）。"""
        out: list[str] = []
        for name in self._facts.method_names():
            try:
                self.check_method_ref(name, dataset_id=dataset_id)
            except MechaError:
                continue
            out.append(name)
        return out


def make_state_validator(validator: MLValidator) -> Callable[[str, Any], None]:
    """产出 ``mecha.gate.Validator`` 形状的函数（装配点透传给 Gate）。"""
    return validator.validate_state


__all__ = [
    "STATE_KEYS", "DEVICE_KINDS", "MAX_SEED", "UNLIMITED", "DeviceSpec",
    "MLValidator", "make_state_validator", "clean_seed", "clean_device",
    "clean_overrides", "clean_resource_guard", "clean_export_root",
]
