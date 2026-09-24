# -*- coding: utf-8 -*-
"""模型可见工具投影：从 ML 命令面/查询面生成工具（宪章 §11.2 第 1—4 阶段）。

纪律（宪章 R4 + mecha ``tools.py`` 命名律）：

- 工具名是**动词_宾语** snake_case，且不带项目前缀（``mecha_`` 收编）；
  ``get_*`` / ``list_*`` 这类"视图名当工具"也收编——只读枚举并进
  ``describe_*`` 或走 ``output_schema`` 投影；
- 每个工具的 ``output_schema["required"]`` **显式声明**回执必填键；
- 错误经 ``MechaError → ToolFailure {message, info}`` 归一化（``ToolRegistry``
  的 execute 负责，不在工具体里手抄错误字典）；
- 大结果一律走**引用**：``run_id`` / ``dataset_id``，绝不把 ``y_pred``、
  ``artifacts``、完整配置内联进工具输出。

工具只是"投影"：所有写路径仍然经 :class:`ml_mecha.engine.MLEngine` 的受控
命令 + mecha ``Gate``；这里不新增第二条写通道。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mecha.errors import MechaError
from mecha.gate import Channel
from mecha.surface import ExecutionContext
from mecha.tools import ToolDefinition, define_tool

from .engine import MLEngine, clear_channel, use_channel

#: 本地宿主的默认通道标签（细标签归因用；side 才定写权）。
LOCAL_ACTOR = "ml-local"


def _raise_on_failure(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    """把引擎的 ``ok=False`` 回执转成 ``ToolRegistry`` 认识的结构化失败。

    引擎回执是领域层返回值（``{"ok": False, "error_kind": ...}``）；工具层若不
    转异常，``ToolRegistry.execute`` 会误判为 ``is_error=False``——AI 看到的
    是"成功但结果失败"，而不是可教学的 ``ToolFailure {message, info}``。
    """
    if receipt.get("ok"):
        return receipt
    warnings = [str(w) for w in receipt.get("warnings", []) if w]
    message = str(receipt.get("error") or "")
    if not message and warnings:
        message = "；".join(warnings)
    raise MechaError(
        message or "ML 命令失败",
        kind=str(receipt.get("error_kind") or "command_failed"),
        hint="命令未生效；检查参数、权限与数据引用后重试",
        suggest=str(receipt.get("suggest") or ""),
    )


def _local_context() -> ExecutionContext:
    """工具路径的执行上下文。

    工具不由 ``Surface.run`` 调用，所以这里显式造一个上下文（而不是传 None）：
    命令体拿到的 contract 与框架路径一致（取消/进度是显式字段，消 D-16）。
    本地同步工具暂不暴露取消，但契约形状先对齐。
    """
    return ExecutionContext()


def build_ml_tools(engine: MLEngine, *, actor: str = LOCAL_ACTOR,
                   side: str = "human",
                   banned_words: tuple[str, ...] = ()) -> list[ToolDefinition]:
    """构造 ML 工具面（顺序稳定；宿主注册进自己的 ToolRegistry）。

    工具按**内容**分类，不按传输形态分类：本地 Python 宿主、dsh/MCP 宿主
    都用同一批 :class:`ToolDefinition`（mecha ``tools.py`` 的白名单投影）。
    """
    channel = Channel(actor, side)

    def _with_channel(fn):
        """把操作者通道绑到执行上下文，再调受控命令（fail closed 的地基）。"""
        def wrapped(**kwargs):
            token = use_channel(channel)
            try:
                return fn(**kwargs)
            finally:
                clear_channel(token)
        return wrapped

    return [
        _describe_methods(engine, banned_words),
        _describe_method(engine, banned_words),
        _describe_dataset(engine, banned_words),
        _describe_run(engine, banned_words),
        _prepare_dataset(engine, _with_channel, banned_words),
        _run_method(engine, _with_channel, banned_words),
        _compare_methods(engine, _with_channel, banned_words),
    ]


# ---------------------------------------------------------------- 只读投影
def _describe_methods(engine: MLEngine,
                      banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, family: str = "", dataset_id: str = "") -> Mapping[str, Any]:
        result = engine.query("method_catalog", family=family, dataset_id=dataset_id)
        rows = [{"name": m["name"], "family": m["family"], "task": m["task"],
                 "target_kind": m["target_kind"], "purposes": list(m["purposes"]),
                 "supports_predict": m["supports_predict"]}
                for m in result["methods"]]
        return {"ok": True, "method_count": len(rows), "methods": rows,
                "error": ""}

    return define_tool(
        name="describe_methods",
        description="列出当前可用的机器学习方法及其族、任务类型与用途；"
                    "可选按族筛选，或按已准备的数据集筛选出真正能跑的方法。",
        parameters={
            "family": {"type": "string", "description": "按方法族筛选；留空表示不过滤"},
            "dataset_id": {"type": "string",
                           "description": "已准备数据集引用；给了就只列能处理它的方法"},
        },
        output_schema={"type": "object",
                       "required": ["ok", "method_count", "methods", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


def _describe_method(engine: MLEngine,
                     banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, name: str) -> Mapping[str, Any]:
        detail = engine.query("method_detail", name=name)
        return {"ok": True, "name": detail["name"], "family": detail["family"],
                "task": detail["task"], "target_kind": detail["target_kind"],
                "purposes": list(detail["purposes"]),
                "purposes_cn": detail["purposes_cn"],
                "supports_predict": detail["supports_predict"],
                "param_schema": detail["param_schema"], "error": ""}

    return define_tool(
        name="describe_method",
        description="说明一个机器学习方法的可调参数、默认值、取值范围与建模用途，"
                    "用于在运行前确认参数名与合法取值。",
        parameters={"name": {"type": "string", "description": "方法名"}},
        output_schema={"type": "object",
                       "required": ["ok", "name", "task", "param_schema", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


def _describe_dataset(engine: MLEngine,
                      banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, dataset_id: str = "") -> Mapping[str, Any]:
        dataset_id = dataset_id or (engine.dataset_ids() or [""])[-1]
        detail = engine.query("dataset_detail", dataset_id=dataset_id)
        return {"ok": True, "dataset_id": detail["dataset_id"],
                "name": detail["name"], "target": detail["target"] or "",
                "n_rows": detail["n_rows"], "n_features": detail["n_features"],
                "target_kind": detail["target_kind"] or "",
                "pipeline_id": detail["pipeline_id"],
                "columns": list(detail["columns"]),
                "pipeline_chain": detail["pipeline_chain"],
                "schema_mismatch": detail["schema_mismatch"], "error": ""}

    return define_tool(
        name="describe_dataset",
        description="说明一份已准备数据视图的行列规模、目标列、预处理链摘要与管道指纹；"
                    "用于判断数据是否适合某个方法。",
        parameters={"dataset_id": {"type": "string",
                                   "description": "数据集引用；留空用最近准备的一份"}},
        output_schema={"type": "object",
                       "required": ["ok", "dataset_id", "n_rows", "n_features",
                                    "pipeline_id", "pipeline_chain", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


def _describe_run(engine: MLEngine,
                  banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, run_id: str = "") -> Mapping[str, Any]:
        if not run_id:
            records = engine.session.export_state()["records"]
            run_id = records[-1]["run_id"] if records else ""
        detail = engine.query("run_detail", run_id=run_id)
        return {"ok": True, "run_id": detail["run_id"], "method": detail["method"],
                "task": detail["task"], "run_ok": detail["ok"],
                "primary_metric": detail["primary_metric"],
                "metrics": detail["metrics"], "params": detail["params"],
                "dataset": detail["dataset"], "pipeline_id": detail["pipeline_id"],
                "artifact_keys": detail["artifact_keys"],
                "error": detail["error"]}

    return define_tool(
        name="describe_run",
        description="按运行编号回看一次机器学习运行的方法、指标、生效参数与产物类别；"
                    "大规模数值产物只给类别名与运行编号，不内联。",
        parameters={"run_id": {"type": "string",
                               "description": "运行编号；留空用最近一次运行"}},
        output_schema={"type": "object",
                       "required": ["ok", "run_id", "method", "primary_metric",
                                    "metrics", "artifact_keys", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


# ---------------------------------------------------------------- 受控命令
def _prepare_dataset(engine: MLEngine, wrap,
                     banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, source: Mapping[str, Any], name: str = "",
                target: str = "", time_col: str = "",
                diag: bool = False) -> Mapping[str, Any]:
        spec: dict[str, Any] = {"command": "prepare_dataset", "source": source,
                                "diag": bool(diag)}
        if name:
            spec["name"] = name
        if target:
            spec["target"] = target
        if time_col:
            spec["time_col"] = time_col
        receipt = _raise_on_failure(
            engine.run(spec, context=_local_context()))
        dataset = receipt.get("dataset") or {}
        return {"ok": bool(receipt["ok"]),
                "dataset_id": receipt.get("dataset_id", ""),
                "pipeline_id": receipt.get("pipeline_id", ""),
                "n_rows": dataset.get("n_rows", 0),
                "n_features": dataset.get("n_features", 0),
                "target": dataset.get("target") or "",
                "warnings": list(receipt.get("warnings", [])),
                "error": receipt.get("error", "")}

    return define_tool(
        name="prepare_dataset",
        description="登记一份数据并在训练部分上拟合预处理链，返回数据集引用与管道指纹；"
                    "数据来源可给文件路径、进程内表格、数组或合成规格。",
        parameters={
            "source": {"type": "object",
                       "description": "数据来源声明，例如 {'kind': 'csv', 'path': 'x.csv'}、"
                                      "{'kind': 'frame', 'frame': <表格>}、"
                                      "{'kind': 'arrays', 'X': [...], 'y': [...]}、"
                                      "{'kind': 'synthetic', 'task': 'classification'}"},
            "name": {"type": "string", "description": "数据集显示名"},
            "target": {"type": "string", "description": "目标列名（监督/时序任务需要）"},
            "time_col": {"type": "string", "description": "时序任务的时间列名"},
            "diag": {"type": "boolean", "description": "是否保留预处理链的诊断中间产物",
                     "default": False},
        },
        output_schema={"type": "object",
                       "required": ["ok", "dataset_id", "pipeline_id", "n_rows",
                                    "n_features", "warnings", "error"]},
        execute=wrap(execute),
        banned_words=banned_words,
    )


def _run_method(engine: MLEngine, wrap,
                banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, method: str, dataset_id: str = "",
                overrides: Mapping[str, Any] | None = None, seed: int | None = None,
                resource_guard: Mapping[str, Any] | None = None,
                device: str = "", persist: bool = False) -> Mapping[str, Any]:
        spec: dict[str, Any] = {"command": "run_method", "method": method,
                                "persist": bool(persist)}
        if dataset_id:
            spec["dataset_id"] = dataset_id
        if overrides:
            spec["overrides"] = dict(overrides)
        if seed is not None:
            spec["seed"] = seed
        if resource_guard:
            spec["resource_guard"] = dict(resource_guard)
        if device:
            spec["device"] = device
        receipt = _raise_on_failure(
            engine.run(spec, context=_local_context()))
        return {"ok": bool(receipt["ok"]), "run_id": receipt.get("run_id", ""),
                "method": method,
                "primary_metric": receipt.get("primary_metric", ""),
                "metrics": dict(receipt.get("metrics", {})),
                "elapsed_s": receipt.get("elapsed_s", 0.0),
                "warnings": list(receipt.get("warnings", [])),
                "error": receipt.get("error", "")}

    return define_tool(
        name="run_method",
        description="在已准备的数据视图上运行一个机器学习方法并返回运行编号与指标摘要；"
                    "预测数组等大产物留在结果侧，此处只给运行编号。",
        parameters={
            "method": {"type": "string", "description": "方法名"},
            "dataset_id": {"type": "string",
                           "description": "数据集引用；留空用最近准备的一份"},
            "overrides": {"type": "object", "description": "运行级参数覆写（不改全局默认）"},
            "seed": {"type": "integer", "description": "随机种子（整数）"},
            "resource_guard": {"type": "object",
                               "description": "资源预算声明，例如 {'max_kernel_mb': 128}"},
            "device": {"type": "string", "description": "计算设备声明：auto / cpu / cuda:0"},
            "persist": {"type": "boolean", "description": "是否把这次运行写入运行存档",
                        "default": False},
        },
        output_schema={"type": "object",
                       "required": ["ok", "run_id", "method", "primary_metric",
                                    "metrics", "elapsed_s", "warnings", "error"]},
        execute=wrap(execute),
        banned_words=banned_words,
    )


def _compare_methods(engine: MLEngine, wrap,
                     banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, methods: list[str], dataset_id: str = "",
                seed: int | None = None,
                overrides: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        spec: dict[str, Any] = {"command": "compare_methods",
                                "methods": list(methods)}
        if dataset_id:
            spec["dataset_id"] = dataset_id
        if seed is not None:
            spec["seed"] = seed
        if overrides:
            spec["overrides"] = dict(overrides)
        receipt = _raise_on_failure(
            engine.run(spec, context=_local_context()))
        return {"ok": bool(receipt["ok"]),
                "run_ids": list(receipt.get("run_ids", [])),
                "primary_metric": receipt.get("primary_metric", ""),
                "table": list(receipt.get("table", [])),
                "winner": dict(receipt.get("winner", {})),
                "warnings": list(receipt.get("warnings", [])),
                "error": receipt.get("error", "")}

    return define_tool(
        name="compare_methods",
        description="在同一份数据视图上对比多个方法的主指标，返回可排序对比表与最优方法；"
                    "逐方法隔离失败，失败项只给运行编号与失败原因。",
        parameters={
            "methods": {"type": "array", "description": "参与对比的方法名列表"},
            "dataset_id": {"type": "string",
                           "description": "数据集引用；留空用最近准备的一份"},
            "seed": {"type": "integer", "description": "随机种子（整数）"},
            "overrides": {"type": "object", "description": "所有方法共用的运行级参数覆写"},
        },
        output_schema={"type": "object",
                       "required": ["ok", "run_ids", "primary_metric", "table",
                                    "winner", "warnings", "error"]},
        execute=wrap(execute),
        banned_words=banned_words,
    )


def register_ml_tools(registry: Any, engine: MLEngine, **kwargs) -> list[Any]:
    """把工具面注册进宿主的 ``ToolRegistry``，返回 disposer 列表（可全撤）。"""
    disposers = []
    for tool in build_ml_tools(engine, **kwargs):
        disposers.append(registry.register(tool))
    return disposers


__all__ = ["build_ml_tools", "register_ml_tools", "LOCAL_ACTOR"]
