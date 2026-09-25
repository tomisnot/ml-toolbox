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

## 长任务：job 类工具（P1）

``submit_run`` / ``read_job`` / ``cancel_run`` 是**调度面**，不是新的写通道：
``submit_run`` 把 ``run_method`` 提交成 mecha job（走 ``MLMecha.submit``，
由它把操作者通道与 ``call_id`` 带进后台线程），真正的域写入仍由那条受控命令
经 ``Gate`` 完成，审计落 ``command.run_method`` + ``ml.run``。因此：

- 这三个工具**不套 ``_with_channel``**（它们的写路径不在本进程的调用栈上，
  而在 job 线程里）。这同时是硬约束：``capabilities.declared_side_effect`` 用
  "闭包是否持有 ``Channel``"与"有无同名写命令"两条证据交叉判定，而
  ``submit_run`` **没有同名命令**；若这里给它绑一个 ``Channel``，两条证据就
  不一致，能力清单会当场 ``CapabilityDrift``（fail loud，方向正确）。
- 代价是能力清单把它们记为 ``side_effect=False``（**调用当刻**不写域状态），
  而"提交即产生一次运行"这个事实写在各自的 description 与
  ``docs/mecha/04`` 的专门小节里，不靠那个 bool 表达。
- ``host`` 是 job 服务 seam（``MLMecha``）；省略时（如 ``capabilities.py``
  只为取名而构造工具、不执行）job 工具仍能构造，调用时才结构化报错。
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

#: ``read_job`` 单次最多阻塞多久（秒）。阻塞是**可选便利**，不是"等到跑完"的
#: 承诺：MCP 侧单次调用有超时（dsh 默认 60 s），模型应轮询。
READ_JOB_MAX_WAIT_S = 30.0

#: job 工具在宿主未接 job 服务时的结构化失败（不是静默假成功）。
_NO_JOB_HOST_HINT = ("job 工具需要一个接好 Job 服务的宿主（MLMecha）；"
                     "capabilities.py 只为取名而构造工具，不执行它们")


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


def _raise_on_total_failure(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    """批量/对比类回执：**全部**失败才转异常；有任一成功则原样返回。

    为什么不能对批量直接用 :func:`_raise_on_failure`：批量与对比的语义是
    "**逐方法隔离失败**"，部分成功是**正常结果**，不是工具失败。若一律转异常，
    已经成功跑出来的那些 ``run_id`` 会连同失败项一起被丢掉——工具自己的
    description 承诺了"返回每个方法的运行编号与指标，失败项只给编号与原因"，
    那样就与实现自相矛盾（模型拿不到任何一个 run_id，还得重跑）。

    "全部失败"= 什么都没达成，仍按写工具纪律 fail loud（不制造假成功）；
    而"部分失败"用 ``ok=False`` + 逐项明细如实表达，调用方读得到。
    """
    if receipt.get("ok"):
        return receipt
    items = receipt.get("runs") or receipt.get("table") or []
    if any(isinstance(item, Mapping) and item.get("ok") for item in items):
        return receipt
    return _raise_on_failure(receipt)


def _local_context() -> ExecutionContext:
    """工具路径的执行上下文。

    工具不由 ``Surface.run`` 调用，所以这里显式造一个上下文（而不是传 None）：
    命令体拿到的 contract 与框架路径一致（取消/进度是显式字段，消 D-16）。
    本地同步工具暂不暴露取消，但契约形状先对齐。
    """
    return ExecutionContext()


def build_ml_tools(engine: MLEngine, *, actor: str = LOCAL_ACTOR,
                   side: str = "human",
                   banned_words: tuple[str, ...] = (),
                   host: Any = None) -> list[ToolDefinition]:
    """构造 ML 工具面（顺序稳定；宿主注册进自己的 ToolRegistry）。

    工具按**内容**分类，不按传输形态分类：本地 Python 宿主、dsh/MCP 宿主
    都用同一批 :class:`ToolDefinition`（mecha ``tools.py`` 的白名单投影）。

    ``host`` 只给 job 类工具用（它们要 ``MLMecha.submit`` / Job 注册表）；
    省略时那三个工具照样构造得出来（能力清单只取名不执行），调用时才报
    ``no_job_host``。
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
        _run_method_batch(engine, _with_channel, banned_words),
        _compare_methods(engine, _with_channel, banned_words),
        _submit_run(host, side, banned_words),
        _read_job(host, banned_words),
        _cancel_run(host, banned_words),
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


def _run_method_batch(engine: MLEngine, wrap,
                      banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, methods: list[str], dataset_id: str = "",
                overrides: Mapping[str, Any] | None = None,
                seed: int | None = None,
                resource_guard: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        spec: dict[str, Any] = {"command": "run_method_batch",
                                "methods": list(methods)}
        if dataset_id:
            spec["dataset_id"] = dataset_id
        if overrides:
            spec["overrides"] = dict(overrides)
        if seed is not None:
            spec["seed"] = seed
        if resource_guard:
            spec["resource_guard"] = dict(resource_guard)
        receipt = _raise_on_total_failure(
            engine.run(spec, context=_local_context()))
        runs = list(receipt.get("runs", []))
        return {"ok": bool(receipt["ok"]),
                "run_ids": list(receipt.get("run_ids", [])),
                "primary_metric": receipt.get("primary_metric", ""),
                "runs": runs,
                "failed_methods": [str(r.get("method", "")) for r in runs
                                   if isinstance(r, Mapping) and not r.get("ok")],
                "warnings": list(receipt.get("warnings", [])),
                "error": receipt.get("error", "")}

    return define_tool(
        name="run_method_batch",
        description="在同一份数据视图上顺序运行多个方法，逐方法隔离失败；"
                    "返回每个方法的运行编号与指标，失败项只给编号与原因。",
        parameters={
            "methods": {"type": "array", "description": "按顺序运行的方法名列表"},
            "dataset_id": {"type": "string",
                           "description": "数据集引用；留空用最近准备的一份"},
            "overrides": {"type": "object", "description": "所有方法共用的运行级参数覆写"},
            "seed": {"type": "integer", "description": "随机种子（整数）"},
            "resource_guard": {"type": "object",
                               "description": "资源预算声明，例如 {'max_kernel_mb': 128}"},
        },
        output_schema={"type": "object",
                       "required": ["ok", "run_ids", "primary_metric", "runs",
                                    "failed_methods", "warnings", "error"]},
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
        # 与 run_method_batch 同一规则：部分成功不是工具失败（否则成功项的
        # run_id / table 会被一起丢掉，与 description 承诺的"失败项只给编号与
        # 原因"自相矛盾）；全部失败才 fail loud。
        receipt = _raise_on_total_failure(
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


# ---------------------------------------------------------------- job 调度面
def _require_job_host(host: Any) -> Any:
    """job 工具的宿主 seam：没有就结构化拒（不静默假成功）。"""
    if host is None:
        raise MechaError(
            "job 工具没有可用的 Job 服务",
            kind="no_job_host",
            hint=_NO_JOB_HOST_HINT,
            suggest="经 assemble_ml_mecha 装配的实例才带 Job 服务",
        )
    return host


def _submit_run(host: Any, side: str,
                banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, method: str, dataset_id: str = "",
                overrides: Mapping[str, Any] | None = None,
                seed: int | None = None,
                device: str = "") -> Mapping[str, Any]:
        ml = _require_job_host(host)
        spec: dict[str, Any] = {"command": "run_method", "method": method}
        if dataset_id:
            spec["dataset_id"] = dataset_id
        if overrides:
            spec["overrides"] = dict(overrides)
        if seed is not None:
            spec["seed"] = seed
        if device:
            spec["device"] = device
        # 通道与 call_id 由 MLMecha.submit 带进后台线程（contextvars 不跨线程）；
        # 这里**不**用 _with_channel —— 见模块 docstring「长任务：job 类工具」。
        job = ml.submit(spec, side=side)
        return {"ok": True, "job_id": job.id, "command": job.command,
                "cancel_supported": bool(job.cancel_supported),
                "state": job.state.value, "error": ""}

    return define_tool(
        name="submit_run",
        description="把一个方法运行提交为后台任务并立刻返回任务编号；"
                    "提交即开始一次真实运行（写入域状态与运行存档），"
                    "但本次调用不等它跑完——用 read_job 轮询进度与结果。",
        parameters={
            "method": {"type": "string", "description": "方法名"},
            "dataset_id": {"type": "string",
                           "description": "数据集引用；留空用最近准备的一份"},
            "overrides": {"type": "object", "description": "运行级参数覆写（不改全局默认）"},
            "seed": {"type": "integer", "description": "随机种子（整数）"},
            "device": {"type": "string", "description": "计算设备声明：auto / cpu / cuda:0"},
        },
        output_schema={"type": "object",
                       "required": ["ok", "job_id", "command", "cancel_supported",
                                    "state", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


def _read_job(host: Any,
              banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, job_id: str, wait_s: float = 0.0) -> Mapping[str, Any]:
        ml = _require_job_host(host)
        try:
            timeout = max(0.0, min(float(wait_s or 0.0), READ_JOB_MAX_WAIT_S))
        except (TypeError, ValueError):
            raise MechaError(
                f"wait_s 必须是数字，收到 {wait_s!r}",
                kind="bad_wait_s",
                hint=f"wait_s 是可选阻塞秒数，上限 {READ_JOB_MAX_WAIT_S:g}",
            ) from None
        status = ml.jobs.wait(job_id, timeout)
        raw = status.get("progress") or []
        progress = [{"frac": float(p[0]), "stage": str(p[1])}
                    for p in raw if isinstance(p, (list, tuple)) and len(p) == 2]
        return {"ok": True, "job_id": str(status.get("id") or job_id),
                "state": str(status.get("state") or ""),
                "cancel_supported": bool(status.get("cancel_supported", False)),
                "progress": progress,
                "progress_latest": progress[-1] if progress else {},
                "result_ref": dict(status.get("result_ref") or {}),
                "error": str(status.get("error") or ""),
                "ref_error": str(status.get("ref_error") or "")}

    return define_tool(
        name="read_job",
        description="按任务编号查一次后台任务的进度与结果：状态、最近进度、"
                    "结果引用（含运行编号）、失败原因；不阻塞（除非给 wait_s）。",
        parameters={
            "job_id": {"type": "string", "description": "submit_run 返回的任务编号"},
            "wait_s": {"type": "number",
                       "description": f"可选：最多阻塞多少秒等它推进（上限 "
                                      f"{READ_JOB_MAX_WAIT_S:g}）；默认 0 立即返回",
                       "default": 0},
        },
        output_schema={"type": "object",
                       "required": ["ok", "job_id", "state", "cancel_supported",
                                    "progress", "progress_latest", "result_ref",
                                    "error", "ref_error"]},
        execute=execute,
        banned_words=banned_words,
    )


def _cancel_run(host: Any,
                banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, job_id: str) -> Mapping[str, Any]:
        ml = _require_job_host(host)
        # 先读一次状态：回执里同时给"请求前"与"请求后"的状态，不谎报"已取消"
        # （取消是协作式的——设置取消标志不等于训练当场停下）。
        before = str(ml.jobs.wait(job_id, 0.0).get("state") or "")
        ml.jobs.cancel(job_id)
        after = str(ml.jobs.wait(job_id, 0.0).get("state") or "")
        return {"ok": True, "job_id": job_id, "cancel_requested": True,
                "state_before": before, "state": after, "error": ""}

    return define_tool(
        name="cancel_run",
        description="请求取消一个后台任务。取消是协作式的：任务的取消标志被置位，"
                    "它会在下一个检查点停止；不可取消的任务会被明确拒绝。",
        parameters={
            "job_id": {"type": "string", "description": "submit_run 返回的任务编号"},
        },
        output_schema={"type": "object",
                       "required": ["ok", "job_id", "cancel_requested",
                                    "state_before", "state", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


def register_ml_tools(registry: Any, engine: MLEngine, **kwargs) -> list[Any]:
    """把工具面注册进宿主的 ``ToolRegistry``，返回 disposer 列表（可全撤）。"""
    disposers = []
    for tool in build_ml_tools(engine, **kwargs):
        disposers.append(registry.register(tool))
    return disposers


__all__ = ["build_ml_tools", "register_ml_tools", "LOCAL_ACTOR",
           "READ_JOB_MAX_WAIT_S"]
