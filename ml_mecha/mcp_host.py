# -*- coding: utf-8 -*-
"""ML Toolbox 的 MCP 服务端适配（交付形态 P0：AI 模式经 dsh 指挥 ML 工具）。

归属：本模块住在 **ML 仓**（同 ``ml_mecha`` 包）。框架仓**刻意不含 MCP**——
``mecha/tools.py`` 的模块 docstring 写明工具注册进宿主 ``ctx.tools``
"**不另起 MCP server**（ADR 另起-MCP-server）"，``mecha/toolhost.py`` 的
Known Limitations 也把 dsh/MCP Provider 列为未实现。因此"谁来起 MCP 服务"
是宿主的事，本模块就是 ML 这个宿主的实现。

形态照 EL 的 ``mecha_v2/hub.py::to_mcp`` + ``mecha_v2/bridge.py::V2Bridge.serve_mcp``
（**同进程后台线程** + 起完写端口文件供发现），但**不照抄其内部细节**：本机
MCP SDK 是 2.2.0，工具登记与 schema 覆写的落点与 EL 记录的版本不同（见下）。

三条纪律
--------

1. **工具来源不手抄**：唯一来源是 ``software.tools.schemas()``（``ToolRegistry``
   的白名单投影：只有 name/description/parameters）。调用统一走
   ``software.toolhost.call(name, args)``——ToolHost 契约的 transport seam，
   失败已在那里归一化为 ``{"is_error": bool, ...}`` 回执。本模块**不新增**
   第二张工具表、第二道权限门（可见性归 ``visibility``，写权归 Gate）。
2. **顶层 ``required`` 从权威声明派生**：``schemas()`` 的 ``parameters`` 只有
   properties（参数名 → 声明），"哪些必填"要看两处——**写工具**的必填在
   **命令声明**里（``ml_mecha/commands.py::COMMANDS`` → ``core_command_specs()``
   → 核心 ``CommandRegistry`` 的 ``CommandSpec.parameters["required"]``），
   **只读工具**没有同名命令，必填在 ``execute`` 的闭包签名里（无默认值 = 必填）。
   照 EL ``hub._register`` 的做法合成真正的 ``inputSchema``，并把声明里的
   type/description/default **覆写回去**（"Phase 4 的 schema 成果过线不丢"）。
   另保留 EL 的垫片：默认值非 ``None`` 的参数若收到显式 ``null``，丢弃该键，
   让真实默认生效（否则会出现"说明书说默认 X，但客户端必须显式传"）。
   ⚠ 这里**不能只靠闭包签名**：写工具是 ``execute=wrap(execute)``，
   ``wrap`` 的签名是 ``(**kwargs)`` ⇒ 从签名推必填会得到空集，模型看到的
   ``required`` 就成了 ``[]``（必填项在 MCP 面静默消失）。见
   :func:`command_declared_required`。
3. **失败不伪装成功**（见 ``_handler`` 的注释：三种返回姿势的实测结论）。

## Known Limitations and Deferred Work

- 工具是**同步**的（``ml_mecha/tools.py`` 的既有形状），长任务会撞 dsh 侧
  ``toolCallTimeoutMs``；P1 才补 job 类工具。本模块不做超时兜底（不发明
  第二套调度）。
- 本模块不提供监控面（那是 ``monitor_http.py`` 的事），也不做端口发现之外的
  编排（那是 ``launcher.py`` 的事）。
- ``server._tool_manager`` 是 SDK 私有面：本模块**只在一处**用它把声明覆写回
  ``Tool.parameters``（EL 同款垫片）。SDK 升级破坏时改这一个函数，不外泄进
  ``ml_mecha`` 其它模块。
"""
from __future__ import annotations

import inspect
import json
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, TextContent

from mecha.errors import MechaError
from mecha.tools import CURRENT_CALL_ID

#: MCP 服务名 → dsh 侧工具名形如 ``mcp__mltoolbox__describe_methods``。
MCP_SERVER_NAME = "mltoolbox"

#: streamable-http 的挂载路径（dsh 内置客户端连 ``http://host:port/mcp``）。
MCP_PATH = "/mcp"

#: 服务端说明（模型可见）。只含任务相关概念，不含 UI/传输/实现词汇；
#: 控制体积（dsh 侧 ``maxInstructionBytes`` 默认 32768，超出会让连接失败）。
INSTRUCTIONS = (
    "ML Toolbox 的机器学习操作工具集。"
    "先用 describe_methods 看有哪些方法、用 describe_dataset 看已准备的数据视图；"
    "prepare_dataset 登记数据并拟合预处理链，run_method 跑单个方法，"
    "compare_methods 在同一份数据上对比多个方法。"
    "运行结果一律用运行编号引用，大数组不会内联返回。"
)

#: JSON Schema 类型 → Python 注解（用于合成签名，让 pydantic 层同样强制必填）。
_PY_TYPES: dict[str, type] = {
    "string": str, "integer": int, "number": float,
    "boolean": bool, "object": dict, "array": list,
}


def _py_type(json_type: object) -> type:
    return _PY_TYPES.get(str(json_type or "string"), str)


def _signature_required(execute: Callable[..., object]) -> list[str]:
    """闭包签名里的必填项（无默认值 = 必填）。"""
    sig = inspect.signature(execute)
    return [p.name for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]


def _signature_defaulted(execute: Callable[..., object]) -> set[str]:
    """闭包签名里"默认值非 None"的参数名。"""
    sig = inspect.signature(execute)
    return {p.name for p in sig.parameters.values()
            if p.default is not inspect.Parameter.empty and p.default is not None}


def _declaration_defaulted(declared: Mapping[str, object]) -> set[str]:
    """声明里带非 None ``default`` 的参数名（包装过的写工具唯一可用来源）。"""
    out = set()
    for name, decl in declared.items():
        if isinstance(decl, Mapping) and decl.get("default") is not None:
            out.add(str(name))
    return out


def command_declared_required(ml, tool_name: str) -> list[str] | None:
    """同名**受控命令声明**里的必填项（写工具的权威来源）；无同名命令 → ``None``。

    为什么不能只靠闭包签名：``ml_mecha/tools.py`` 的**写工具**是
    ``execute=wrap(execute)``，而 ``wrap`` 是 ``def wrapped(**kwargs)``——
    它的签名里没有任何命名参数，于是"从签名推必填"会得到**空集**，
    模型看到的 ``required`` 就成了 ``[]``（"必填项在 MCP 面静默消失"，
    正是 EL 接管报告 P1-8 的同一类缺陷）。真正的权威来源是命令声明：
    ``core_command_specs()`` 把 ``ml_mecha/commands.py::COMMANDS`` 的
    ``required_params`` 投影成 ``CommandSpec.parameters["required"]``，
    装配时已注册进核心 ``CommandRegistry``——本函数只是把它取出来。
    只读工具没有同名命令（它们走 ``Surface`` 查询，不是命令），故退回闭包签名。
    """
    software = getattr(ml, "software", None)
    commands = getattr(software, "commands", None)
    if commands is None:                       # pragma: no cover - 裸桥/判据装配
        return None
    spec = commands.spec(tool_name)
    if spec is None:
        return None
    params = getattr(spec, "parameters", None)
    if not isinstance(params, Mapping):
        return None
    declared = params.get("required")
    return None if declared is None else [str(k) for k in declared]


def tool_input_schema(tool_name: str, schema: Mapping[str, object],
                      execute: Callable[..., object],
                      declared_required: list[str] | None = None,
                      ) -> tuple[dict, set[str]]:
    """由**声明**（properties）与必填来源合成 inputSchema。

    返回 ``(input_schema, defaulted)``：

    - ``input_schema`` = ``{"type": "object", "properties": <声明逐字>, "required": [...]}``；
    - ``defaulted`` = 默认值非 ``None`` 的参数名（调用侧据此丢弃显式 ``null``，
      让真实默认生效）。两个来源取并集：闭包签名（只读工具）+ 声明
      ``default``（包装过的写工具，签名里没有命名参数）。

    必填的**优先来源是命令声明**（``declared_required``，见
    :func:`command_declared_required`）；没有同名命令时才用闭包签名。
    声明里不存在的必填项当场 fail loud——不静默产出一个模型永远填不出来的字段。
    """
    declared: dict = dict(schema.get("parameters") or {})  # type: ignore[arg-type]
    sig_required = _signature_required(execute)
    if declared_required is not None:
        required = list(declared_required)
        # 漂移检查：签名**确实**暴露了命名必填项时，两侧必须一致（包装器下的
        # sig_required 为空是预期的，故只在非空时比对）。
        if sig_required and sig_required != required:
            raise MechaError(
                f"工具 {tool_name!r} 的闭包签名必填 {sig_required} 与命令声明 "
                f"{required} 不一致",
                kind="tool_declaration_drift",
                hint="必填的单一来源是命令声明（core_command_specs 的投影）",
                suggest="改 ml_mecha/commands.py 的声明或工具闭包，使两侧一致",
            )
    else:
        required = sig_required
    undeclared = [n for n in required if n not in declared]
    if undeclared:
        raise MechaError(
            f"工具 {tool_name!r} 的必填项 {undeclared} 不在参数声明里",
            kind="tool_declaration_drift",
            hint="required 里的名字必须都能在声明（properties）里找到",
            suggest="在 ml_mecha/tools.py 的 parameters 里补上这些参数",
        )
    defaulted = _signature_defaulted(execute) | _declaration_defaulted(declared)
    input_schema = {
        "type": "object",
        "properties": {k: (dict(v) if isinstance(v, Mapping) else {"type": "string"})
                       for k, v in declared.items()},
        "required": required,
    }
    return input_schema, defaulted


def _synthesize_signature(schema: Mapping[str, object],
                          required: list[str]) -> inspect.Signature:
    """给 handler 一个"真实命名参数"的合成签名。

    为什么必须合成：MCP 从**函数签名**建 pydantic 入参模型；用 ``**kwargs``
    会得到一个名为 ``kwargs`` 的必填字段（错）。合成后模型字段 = 参数名，
    必填项不给默认值 ⇒ pydantic 层同样强制。可选项标注为 ``<type> | None``，
    这样客户端送来显式 ``null`` 也能过校验（再由 ``defaulted`` 垫片丢弃）。
    """
    declared: dict = dict(schema.get("parameters") or {})  # type: ignore[arg-type]
    params = []
    for name, decl in declared.items():
        annot = _py_type((decl or {}).get("type") if isinstance(decl, Mapping) else None)
        if name in required:
            params.append(inspect.Parameter(
                name, inspect.Parameter.KEYWORD_ONLY, annotation=annot))
        else:
            params.append(inspect.Parameter(
                name, inspect.Parameter.KEYWORD_ONLY,
                annotation=annot | None, default=None))
    return inspect.Signature(params, return_annotation=str)


def _error_result(error: object) -> CallToolResult:
    """领域失败的**唯一**返回形状：``is_error=True`` + 结构化 JSON 正文。

    实测（mcp 2.2.0）三条姿势的区别——这不是风格选择，是正确性问题：

    ==========================================  ====================================
    做什么                                       结果
    ==========================================  ====================================
    ``return CallToolResult(..., is_error=True)``  ✅ 逐字透传：``is_error=True``、正文不改
    ``raise ToolError(json)``                     ❌ 正文被加上 ``Error executing tool ...`` 前缀
    ``return json_str``                           ❌ ``is_error=False``（"成功文本里写着错误"的假成功）
    ==========================================  ====================================

    正文就是 ``{"ok": false, "error": {message, info:{kind,hint,suggest}}}``——
    模型读 ``kind`` 决定怎么改，不从散文里猜。
    """
    payload = {"ok": False, "error": error}
    return CallToolResult(
        content=[TextContent(type="text",
                             text=json.dumps(payload, ensure_ascii=False, default=str))],
        is_error=True,
    )


def _make_handler(ml, tool_name: str, defaulted: set[str]):
    """一个工具的处理函数：归一化入参 → 走 ToolHost → 结构化回执。

    ``ctx`` 由 SDK 注入（``find_context_parameter`` 按注解找），用于把本次
    MCP 请求的 ``request_id`` 写成 ``CURRENT_CALL_ID``——于是"AI 做了什么"
    （dsh 行为史）与"域变成了什么"（History 事件）能按同一个 call_id 互引
    （mecha 审查 2.7 / 总纲 §5.2②）。``ctx`` 不进模型可见参数（合成签名里没有它）。
    """
    def _handler(ctx: Context = None, **kwargs):
        # 垫片：闭包默认非 None 的参数收到显式 null ⇒ 丢弃，让真实默认生效。
        args = {k: v for k, v in kwargs.items()
                if not (v is None and k in defaulted)}
        call_id = ""
        if ctx is not None:
            rid = getattr(ctx, "request_id", None)
            call_id = "" if rid is None else str(rid)
        token = CURRENT_CALL_ID.set(call_id)
        try:
            result = ml.toolhost.call(tool_name, args)
        finally:
            CURRENT_CALL_ID.reset(token)
        if not isinstance(result, Mapping):
            return _error_result({"message": "工具宿主回执不是映射",
                                  "info": {"kind": "bad_host_result"}})
        if result.get("is_error"):
            return _error_result(result.get("error") or {
                "message": "工具失败", "info": {"kind": "unknown_failure"}})
        return json.dumps(result.get("value"), ensure_ascii=False, default=str)

    _handler.__name__ = tool_name
    return _handler


def _override_declared_schema(server: MCPServer, name: str, input_schema: dict) -> None:
    """把**声明**的 inputSchema 覆写回注册项（EL 同款、全模块唯一的 SDK 私有面访问）。

    pydantic 派生出的 schema 类型正确但会丢掉声明里的 description 与真实默认值
    （可选项一律变成 ``anyOf: [T, null], default: null``）。声明是"模型唯一的
    用法来源"，所以覆写回去。``MCPServer.list_tools`` 正是用 ``info.parameters``
    产出协议层 ``input_schema``（已核对源码），故覆写即生效。
    """
    manager = getattr(server, "_tool_manager", None)
    tool = None if manager is None else manager.get_tool(name)
    if tool is None:
        raise MechaError(
            f"刚注册的工具 {name!r} 在 SDK 注册表里找不到",
            kind="mcp_sdk_shape_changed",
            hint="mcp SDK 的 MCPServer 内部结构变了（_tool_manager.get_tool）",
            suggest="改 ml_mecha/mcp_host.py 的这一处垫片，别改调用方",
        )
    tool.parameters = input_schema


def register_tool(server: MCPServer, ml, schema: Mapping[str, object]) -> None:
    """把一条 ``schemas()`` 投影注册成 MCP 工具（声明与签名双向对齐）。"""
    name = str(schema["name"])
    tool_def = ml.tools.get(name)
    if tool_def is None:
        raise MechaError(
            f"schemas() 声明了工具 {name!r}，注册表里却取不到本体",
            kind="mcp_sdk_shape_changed",
            hint="schemas() 与 get() 必须来自同一张注册表",
        )
    input_schema, defaulted = tool_input_schema(
        name, schema, tool_def.execute,
        declared_required=command_declared_required(ml, name))
    handler = _make_handler(ml, name, defaulted)
    handler.__signature__ = _synthesize_signature(schema, input_schema["required"])
    server.add_tool(handler, name=name, description=str(schema.get("description") or ""))
    _override_declared_schema(server, name, input_schema)


def build_mcp_server(ml, *, name: str = MCP_SERVER_NAME,
                     instructions: str = INSTRUCTIONS) -> MCPServer:
    """按当前工具面构造 MCP 服务（纯构造，不起线程、不碰端口）。"""
    server = MCPServer(name=name, instructions=instructions)
    for schema in ml.tools.schemas():
        register_tool(server, ml, schema)
    return server


class McpHost:
    """在**本进程后台线程**跑 MCP 服务，起完把实际端口写进端口文件。

    与 EL 的 ``V2Bridge.serve_mcp`` 同形态：权威进程与 MCP 服务**共一个 Gate、
    一份 History**（不是另起进程、不是第二真值）。dsh 是纯客户端。

    - ``start()`` 返回实际端口（``port=0`` 时由 OS 分配，再 bind 后回读，
      无 TOCTOU 竞争窗口），并写 ``port_file``（默认不写，由调用方显式给）。
    - ``stop()`` 干净退出：置 uvicorn 的 ``should_exit`` → join 线程 → 删端口文件。
      模式交接时"旧权威真退出、写租约释放、新权威再起"依赖这里的干净收尾。
    """

    def __init__(self, ml, *, host: str = "127.0.0.1", port: int = 0,
                 port_file: str | Path | None = None,
                 log: Callable[[str], None] | None = None) -> None:
        self._ml = ml
        self._host = str(host)
        self._want_port = int(port)
        self._port_file = Path(port_file) if port_file is not None else None
        self._log = log or (lambda _m: None)
        self._server: MCPServer | None = None
        self._uvicorn = None
        self._thread: threading.Thread | None = None
        self._port: int | None = None

    # ------------------------------------------------------------ 事实面
    @property
    def port(self) -> int | None:
        return self._port

    @property
    def url(self) -> str:
        return "" if self._port is None else f"http://{self._host}:{self._port}{MCP_PATH}"

    @property
    def server(self) -> MCPServer | None:
        """底层 MCP 服务（判据/调试用；不要拿它当第二条调用通道）。"""
        return self._server

    # ------------------------------------------------------------ 生命周期
    def start(self, timeout: float = 30.0) -> int:
        """起服务并返回实际端口；失败抛错（不静默留一个连不上的端点）。"""
        if self._thread is not None:
            raise MechaError("MCP 服务已启动", kind="mcp_already_started",
                             hint="一个 McpHost 只 start 一次；要重起先 stop()")
        import uvicorn

        self._unlink_stale_port_file()
        self._server = build_mcp_server(self._ml)
        # 直接用 streamable_http_app + uvicorn.Server（而不是 server.run(...)）：
        # 后者内部自己 asyncio.run，拿不到句柄 ⇒ 无法干净 stop。
        starlette_app = self._server.streamable_http_app(
            streamable_http_path=MCP_PATH, host=self._host)
        config = uvicorn.Config(starlette_app, host=self._host, port=self._want_port,
                                log_level="warning")
        self._uvicorn = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._uvicorn.run, daemon=True,
                                        name="ml-mecha-mcp")
        self._thread.start()
        self._port = self._await_ready(timeout)
        if self._port_file is not None:
            self._port_file.write_text(str(self._port), encoding="utf-8")
        self._log(f"[ml-mecha-mcp] {self.url}（工具 {len(self._ml.tools.schemas())} 个）")
        return self._port

    def _await_ready(self, timeout: float) -> int:
        """等 uvicorn 真正 listen（不是等线程起来）；失败当场报清楚。"""
        deadline = time.time() + max(1.0, float(timeout))
        while time.time() < deadline:
            if getattr(self._uvicorn, "started", False):
                return self._bound_port()
            if self._thread is not None and not self._thread.is_alive():
                raise MechaError(
                    f"MCP 服务线程退出，未能 listen {self._host}:{self._want_port}",
                    kind="mcp_start_failed",
                    hint="端口被占 / 依赖缺失 / uvicorn 启动即失败；栈见上一条日志",
                    suggest="换端口（port=0 让 OS 分配）后重试",
                )
            time.sleep(0.05)
        raise MechaError(
            f"等 MCP 服务 listen 超时（{timeout}s）",
            kind="mcp_start_timeout", hint="uvicorn 未在限期内 started=True",
        )

    def _bound_port(self) -> int:
        """回读真正 bind 到的端口（port=0 时这是唯一可靠的来源）。"""
        for srv in getattr(self._uvicorn, "servers", None) or []:
            for sock in getattr(srv, "sockets", None) or []:
                try:
                    return int(sock.getsockname()[1])
                except OSError:      # pragma: no cover - 平台差异
                    continue
        return self._want_port

    def stop(self, timeout: float = 10.0) -> None:
        """干净退出（幂等）：置 should_exit → join → 删端口文件。"""
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        if self._thread is not None:
            self._thread.join(max(0.1, float(timeout)))
        self._thread = None
        self._clear_port_file()
        self._port = None

    def _unlink_stale_port_file(self) -> None:
        """起服务前清掉上一次留下的端口文件（陈旧文件是"假绿"的温床）。

        **为什么必须有这一步**（实测得出）：Windows 上 ``terminate()`` /
        ``taskkill /F`` **不会**跑 Python 的 ``finally``，所以"起写止删"不能只靠
        :meth:`stop`。留下一个陈旧 ``.mcp-port`` 的危害很具体：看门人若只判
        "文件存在"就认为权威就绪（EL ``launcher._wait_mcp_port`` 正是这么写的），
        它会立刻去起 dsh，而 dsh 连的是**已经死掉的端口**——"起得来但看不见"。

        **安全性论证**：本方法只在 :meth:`start` 里、**装配成功之后**被调用；
        装配成功意味着本进程已持有该数据根的写租约（``.ml-mecha/.writer.lock``，
        跨进程独占），即同一根上不可能还有别的活写宿主 ⇒ 这个端口文件必然属于
        一个已经退出的进程，删它不会误伤活实例。
        """
        if self._port_file is None:
            return
        try:
            if self._port_file.exists():
                self._port_file.unlink()
        except OSError:
            pass                          # 删不掉不致命：真端口随后会被覆写

    def _clear_port_file(self) -> None:
        """收尾时删端口文件——只在内容仍是**我们自己**的端口时删。"""
        if self._port_file is None:
            return
        try:
            if not self._port_file.exists():
                return
            if self._port is None or self._port_file.read_text(
                    encoding="utf-8").strip() in ("", str(self._port)):
                self._port_file.unlink()
        except OSError:
            pass

    def __enter__(self) -> "McpHost":
        self.start()
        return self

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False


__all__ = ["McpHost", "build_mcp_server", "register_tool", "tool_input_schema",
           "command_declared_required", "MCP_SERVER_NAME", "MCP_PATH", "INSTRUCTIONS"]
