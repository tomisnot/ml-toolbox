# -*- coding: utf-8 -*-
"""命令面声明：ML 自己的受控命令（宪章 R1：不复制 EL 命令名）。

这里的每条命令是**声明**，不是实现：``name`` / ``parameters`` / ``required``
显式写清，引擎据此校验 spec，工具层据此投影出模型可见参数表。声明集中在一处
是为了"声明与执行不漂移"——命令名只在本模块的字面量里出现一次。

纪律（宪章 §5.2 / R1）：

- 命令名是 ML 词汇（``prepare_dataset`` / ``run_method`` / ...），**不是**
  EL 的 ``set_config`` / ``run_sim`` / ``set_drive``；
- ``side_effect`` 显式声明（none / domain_state / host_artifacts），让"这次
  调用会不会改东西"可审计；
- ``risk`` 是本 adapter 自己的分级，不要求 mecha 核心认识风险概念。

本模块只依赖标准库与 ``ml_mecha`` 自身：不 import mecha，也不 import
``ml_toolbox``，因此可以被 schema 投影/静态检查单独消费。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

#: 副作用分类（本 adapter 词汇，不进 mecha 核心）。
SIDE_EFFECT_NONE = "none"                 # 只读：不改域状态、不产生产物
SIDE_EFFECT_DOMAIN_STATE = "domain_state"  # 写 ML 状态键（经 Gate）
SIDE_EFFECT_HOST_ARTIFACTS = "host_artifacts"  # 在 ML 会话里产生 run 记录

#: 风险分级（本 adapter 词汇；mecha 核心不认风险，只认 Gate/审批）。
RISK_READ = "read"
RISK_LOW = "low"
RISK_MEDIUM = "medium"


@dataclass(frozen=True)
class CommandParam:
    """一个命令参数的显式声明。"""

    name: str
    kind: str = "string"
    required: bool = False
    doc: str = ""
    default: Any = None

    def projection(self) -> dict[str, Any]:
        """投影成模型可见的参数声明（给工具层与 schema 视图共用）。"""
        out: dict[str, Any] = {"type": self.kind, "description": self.doc}
        if not self.required and self.default is not None:
            out["default"] = self.default
        return out


@dataclass(frozen=True)
class CommandSpec:
    """一条 ML 受控命令的声明。"""

    name: str
    summary: str
    parameters: tuple[CommandParam, ...] = ()
    #: 执行后写入哪些 ML 状态键（空 = 只读命令）
    writes_state: tuple[str, ...] = ()
    side_effect: str = SIDE_EFFECT_NONE
    risk: str = RISK_READ
    #: 回执里除 run_ids/metrics 外还会带哪些紧凑键（供 schema 自描述）
    extra_receipt_keys: tuple[str, ...] = ()

    def param(self, name: str) -> CommandParam | None:
        for p in self.parameters:
            if p.name == name:
                return p
        return None

    @property
    def required_params(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters if p.required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary,
            "side_effect": self.side_effect,
            "risk": self.risk,
            "writes_state": list(self.writes_state),
            "required": list(self.required_params),
            "parameters": {p.name: p.projection() for p in self.parameters},
        }


#: 命令面全表：ML 侧唯一声明点（顺序即投影顺序）。
COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        name="prepare_dataset",
        summary="登记一份数据并在训练部分上拟合预处理链，返回数据集引用与管道指纹；数据来源可给文件路径、进程内表格、数组或合成规格。",
        parameters=(
            CommandParam("source", "object", True,
                         "数据来源声明，例如 {'kind': 'csv', 'path': 'x.csv'}、{'kind': 'frame', 'frame': <表格>}、{'kind': 'arrays', 'X': [...], 'y': [...]}、{'kind': 'synthetic', 'task': 'classification'}"),
            CommandParam("name", "string", False, "数据集显示名", "dataset"),
            CommandParam("target", "string", False, "目标列名（监督/时序任务需要）", ""),
            CommandParam("time_col", "string", False, "时序任务的时间列名", ""),
            CommandParam("diag", "boolean", False, "是否保留预处理链的诊断中间产物", False),
        ),
        writes_state=("current.dataset_id", "current.pipeline_id"),
        side_effect=SIDE_EFFECT_DOMAIN_STATE,
        risk=RISK_LOW,
        extra_receipt_keys=("dataset_id", "pipeline_id", "dataset"),
    ),
    CommandSpec(
        name="run_method",
        summary="在已准备的数据视图上运行一个机器学习方法并返回运行编号与指标摘要；预测数组等大产物留在结果侧，此处只给运行编号。",
        parameters=(
            CommandParam("method", "string", True, "已注册的方法名"),
            CommandParam("dataset_id", "string", False, "数据集引用；留空用最近准备的一份", ""),
            CommandParam("overrides", "object", False, "运行级参数覆写（不改全局默认）"),
            CommandParam("seed", "integer", False, "随机种子（整数）"),
            CommandParam("diag", "boolean", False, "是否保留诊断信息", False),
            CommandParam("persist", "boolean", False, "是否把本次运行写入 ML 运行存档", False),
            CommandParam("resource_guard", "object", False, "资源预算声明，例如 {'max_kernel_mb': 128}"),
            CommandParam("device", "string", False, "计算设备声明：auto / cpu / cuda:0", ""),
        ),
        # 声明必须**逐键等于**实际写入：run_method 的 `_write_state` 也写
        # `current.dataset_id`（隐含数据集解析），漏声明就是声明漂移
        # （第三轮审查 P2-7）。双向对账见
        # `tests/test_ml_mecha.py::test_writes_state_declaration_matches_actual_history`。
        writes_state=("current.dataset_id", "current.method", "current.overrides",
                      "current.seed", "current.device", "current.resource_guard"),
        side_effect=SIDE_EFFECT_HOST_ARTIFACTS,
        risk=RISK_LOW,
        extra_receipt_keys=("run_id", "elapsed_s", "error"),
    ),
    CommandSpec(
        name="run_method_batch",
        summary="在同一份数据视图上顺序运行多个方法，逐方法隔离失败；返回每个方法的运行编号与指标，失败项只给编号与原因。",
        parameters=(
            CommandParam("methods", "array", True, "按顺序运行的方法名列表（受数量上限约束）"),
            CommandParam("dataset_id", "string", False, "数据集引用；留空用最近准备的一份", ""),
            CommandParam("overrides", "object", False, "所有方法共用的运行级参数覆写"),
            CommandParam("seed", "integer", False, "随机种子（整数）"),
            CommandParam("resource_guard", "object", False, "资源预算声明，例如 {'max_kernel_mb': 128}"),
        ),
        writes_state=("current.dataset_id", "current.seed", "current.overrides",
                      "current.resource_guard"),
        side_effect=SIDE_EFFECT_HOST_ARTIFACTS,
        risk=RISK_MEDIUM,
        extra_receipt_keys=("runs",),
    ),
    CommandSpec(
        name="compare_methods",
        summary="在同一份数据视图上对比多个方法的主指标，返回可排序对比表与最优方法；逐方法隔离失败，失败项只给运行编号与失败原因。",
        parameters=(
            CommandParam("methods", "array", True, "参与对比的方法名列表"),
            CommandParam("dataset_id", "string", False, "数据集引用；留空用最近准备的一份", ""),
            CommandParam("seed", "integer", False, "随机种子（整数）"),
            CommandParam("overrides", "object", False, "所有方法共用的运行级参数覆写"),
            CommandParam("resource_guard", "object", False, "资源预算声明，例如 {'max_kernel_mb': 128}"),
        ),
        writes_state=("current.dataset_id", "current.seed",
                      "current.resource_guard"),
        side_effect=SIDE_EFFECT_HOST_ARTIFACTS,
        risk=RISK_MEDIUM,
        extra_receipt_keys=("table", "winner"),
    ),
)


def command_map() -> dict[str, CommandSpec]:
    """命令名 → 声明（供引擎校验 spec、工具层投影参数表）。"""
    return {c.name: c for c in COMMANDS}


def command_names() -> list[str]:
    return [c.name for c in COMMANDS]


def command_doc(name: str) -> Mapping[str, Any] | None:
    spec = command_map().get(name)
    return spec.to_dict() if spec is not None else None


__all__ = [
    "SIDE_EFFECT_NONE", "SIDE_EFFECT_DOMAIN_STATE", "SIDE_EFFECT_HOST_ARTIFACTS",
    "RISK_READ", "RISK_LOW", "RISK_MEDIUM",
    "CommandParam", "CommandSpec", "COMMANDS", "command_map", "command_names",
    "command_doc",
]
