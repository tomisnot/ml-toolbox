# -*- coding: utf-8 -*-
"""ML Toolbox 的 MCP 服务端适配（**薄壳**）：投影与传输全在 ``mecha.providers.mcp``。

归属：本文件住在 ML 仓，但只保留**ML 词汇与 ML 策略**——服务名、服务说明、以及
"必填项来自哪张声明表"。其余（三步投影、required 解析、null 垫片、失败归一化、
uvicorn 后台线程与端口回读、SDK 私有面垫片、instructions 体积守卫）**逐条**已是
框架行为，本文件不再有第二份实现。

## 迁移记录（P3：两层传输上提框架，2026-09-25）

迁移前本文件 445 行，自持上述全部机制。上提的依据是契约
``_mecha-extraction/mcp-contract.md``（本仓是「MCP 投影契约」的权威方），框架按该规格
实现：§1 工具枚举与三步投影、§2 必填优先级（R2-1..R2-5）、§3 SDK 私有面单点隔离、
§4 调用路径与 ``CURRENT_CALL_ID``、§5 失败归一化（R5-1 部分成功不算失败）、§6 服务装配、
§7 instructions 体积纪律、§8 null 垫片（R8-1/R8-2/**R8-2a 声明唯一权威**）。

## 保留在 ML 侧的差异（不是重复实现）

- **必填来源是"受控命令声明"**：4 条写命令的 ``CommandSpec.parameters["required"]``
  （``ml_mecha/commands.py`` → ``ml_mecha/core_commands.py`` 机械投影 → 核心
  ``CommandRegistry``）比闭包签名权威——写工具体是 ``execute=wrap(execute)``，
  ``def wrapped(**kwargs)`` 让签名恒为空集。框架用 ``RequiredSource`` 注入点承载它，
  并在签名**非空**时做漂移检查（``tool_declaration_drift``）。
- **只读工具与 job 调度面没有同名命令** ⇒ ``required_for`` 返回 ``None``
  ⇒ 框架回退闭包签名（这三类工具未被包装，签名可用）。
- **服务名与说明是 ML 词汇**（``mltoolbox`` / 下面的 ``INSTRUCTIONS``）。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from mecha.providers.mcp import McpEndpoint as _McpEndpoint
from mecha.providers.mcp import build_mcp_server as _framework_build_mcp_server

#: MCP 服务名 → dsh 侧工具名形如 ``mcp__mltoolbox__describe_methods``。
MCP_SERVER_NAME = "mltoolbox"

#: 服务端说明（模型可见）。只含任务相关概念，不含 UI/传输/实现词汇；体积由框架的
#: ``check_instructions`` 在构造期守卫（dsh 侧 ``maxInstructionBytes`` 默认 32768，
#: 超出会让连接失败——报错离病因很远，所以要在构造期拦）。
INSTRUCTIONS = (
    "ML Toolbox 的机器学习操作工具集。"
    "先用 describe_methods 看有哪些方法、用 describe_dataset 看已准备的数据视图；"
    "prepare_dataset 登记数据并拟合预处理链，run_method 跑单个方法，"
    "compare_methods 在同一份数据上对比多个方法。"
    "运行结果一律用运行编号引用，大数组不会内联返回。"
)


class CommandRequiredSource:
    """必填项来源：同名**受控命令**的声明（核心 ``CommandRegistry`` 的投影）。

    只回答"这条工具的同名命令声明了哪些必填"；没有同名命令时返回 ``None``
    —— 那是"我不知道"，由框架回退闭包签名（协议约定，见 ``RequiredSource``）。
    """

    def __init__(self, commands: Any) -> None:
        self._commands = commands

    def required_for(self, name: str, tool: Any = None) -> list[str] | None:
        spec = self._commands.spec(name)
        if spec is None:
            return None
        params = getattr(spec, "parameters", None)
        if not isinstance(params, Mapping):
            return None
        declared = params.get("required")
        return None if declared is None else [str(k) for k in declared]


def required_source_for(ml: Any) -> CommandRequiredSource | None:
    """装配一台 ML 实例的必填来源；裸桥/判据装配（无 ``software``）返回 ``None``。"""
    software = getattr(ml, "software", None)
    commands = getattr(software, "commands", None)
    return None if commands is None else CommandRequiredSource(commands)


def command_declared_required(ml: Any, tool_name: str) -> list[str] | None:
    """单点查询：该工具的**命令声明**必填项（无同名命令 → ``None``）。

    与装配路径共用同一份逻辑（``CommandRequiredSource``），判据不必自己重推。
    """
    source = required_source_for(ml)
    if source is None:
        return None
    return source.required_for(tool_name, ml.tools.get(tool_name))


def build_mcp_server(ml: Any, **kwargs: Any):
    """按 ML 的默认值**纯构造** MCP 服务（不起线程、不碰端口；判据可直接调）。

    ML 的默认值只有三样：服务名、服务说明、必填来源；其余走框架。
    """
    return _framework_build_mcp_server(
        ml.tools, server_name=MCP_SERVER_NAME, instructions=INSTRUCTIONS,
        toolhost=ml.toolhost, required_source=required_source_for(ml), **kwargs)


class McpHost:
    """在**权威进程后台线程**跑 MCP 服务（薄壳：委托框架 ``McpEndpoint``）。

    - ``start()`` 返回**实际**端口（``port=0`` 由 OS 分配后回读，无 TOCTOU 窗口），
      并写 ``port_file`` 供发现；
    - ``stop()`` 干净退出：置 uvicorn 的 ``should_exit`` → join → 条件删端口文件
      （模式交接依赖"旧权威真退出、写租约释放、新权威再起"）；
    - ``server`` 暴露底层 MCPServer（判据/调试用；**不是**第二条调用通道）。
    """

    def __init__(self, ml: Any, *, host: str = "127.0.0.1", port: int = 0,
                 port_file: Any = None,
                 log: Callable[[str], None] | None = None) -> None:
        self._ml = ml
        self._endpoint = _McpEndpoint(
            ml.tools, server_name=MCP_SERVER_NAME, instructions=INSTRUCTIONS,
            toolhost=ml.toolhost, required_source=required_source_for(ml),
            host=host, port=port, port_file=port_file, log=log)

    # ------------------------------------------------------------ 事实面
    @property
    def port(self) -> int | None:
        return self._endpoint.port

    @property
    def url(self) -> str:
        return self._endpoint.url

    @property
    def server(self):
        return self._endpoint.server

    # ------------------------------------------------------------ 生命周期
    def build_server(self):
        return self._endpoint.build_server()

    def start(self, timeout: float = 30.0) -> int:
        return self._endpoint.start(timeout)

    def stop(self, timeout: float = 10.0) -> None:
        self._endpoint.stop(timeout)

    def __enter__(self) -> "McpHost":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.stop()
        return False


#: 公开面**只有 ML 自己的东西**：端点壳、纯构造壳、必填来源（+其访问器）。
#: 框架的 ``tool_input_schema`` / ``resolve_required`` / ``check_instructions`` /
#: ``RequiredSource`` / ``MCP_PATH`` 等**不在这里再导出**——它们没有本仓的消费者
#: （判据要用就直接 ``from mecha.providers.mcp import …``），转一手只会多一层
#: 需要同步的转发面。
__all__ = ["McpHost", "build_mcp_server", "CommandRequiredSource", "INSTRUCTIONS",
           "MCP_SERVER_NAME", "command_declared_required", "required_source_for"]
