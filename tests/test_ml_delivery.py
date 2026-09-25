# -*- coding: utf-8 -*-
"""交付形态 P0 判据：AI 模式（headless 权威 + 同进程 MCP 服务）真的能被调通。

每条断言都指向"机制真的接上了"，不是形状猜测：

1. ``required`` 来自 ``execute`` 签名（无默认值=必填），且**声明**里的
   type/description 覆写回了协议层（消"pydantic 派生把声明成果冲掉"）；
2. 起服务后 ``list_tools`` 含**全部** 7 个 ML 工具，名字与 ``schemas()`` 逐字相同；
3. 只读工具经 HTTP 调用返回真值（``method_count > 0``）；
4. **LOCKED 下写工具 fail closed**：``is_error=True`` +
   ``error.info.kind == "authority_locked"``，正文是**纯 JSON**（钉住"raise
   会被加上 ``Error executing tool`` 前缀"这个坑），且被拒的写一个字都没进史；
5. 开闸后写工具真跑通（``dataset_id`` / ``run_id`` / 指标），且 History 事件带
   **非空 call_id**——这是 MCP 的 ``request_id`` 经 ``Context`` 注入、
   落到 mecha 归因链（"AI 做了什么"↔"域变成了什么"互引）的真实证据；
6. 端口文件起写止删（看门人/发现机制依赖它），且**被强杀后留下的陈旧端口文件
   会被启动时清掉**（Windows 的强杀不跑 ``finally``——这是实测得出的坑）。

运行：``python -m pytest tests/test_ml_delivery.py -q``
也可直接 ``python tests/test_ml_delivery.py``（自带 main，与仓内其它 test_*.py 同形；
**故意不 import pytest**，直跑入口不需要它）。

## 环境要求

- mecha 在 ``sys.path`` 上（``MECHA_ROOT`` 或默认 ``D:\\code-nosync\\mecha``）；
- ``mcp`` 已安装（服务端 ``mcp.server.mcpserver`` + 客户端
  ``mcp.client.streamable_http``）；
- 只在**回环**上起临时端口（``port=0``，OS 分配）：不占固定端口、不碰仓内
  ``.ml-mecha`` / ``.mcp-port``（全部落在临时数据根）。
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ---- mecha 上 sys.path（不 vendor；环境变量优先，默认走同级仓） ----
MECHA_ROOT = Path(os.environ.get("MECHA_ROOT", r"D:\code-nosync\mecha"))
if MECHA_ROOT.is_dir() and str(MECHA_ROOT) not in sys.path:
    sys.path.insert(0, str(MECHA_ROOT))

from mcp import ClientSession                                   # noqa: E402
from mcp.client.streamable_http import streamable_http_client   # noqa: E402

from ml_mecha.assembly import assemble_ml_mecha                 # noqa: E402
from ml_mecha.mcp_host import McpHost, build_mcp_server         # noqa: E402

#: 工具面全量（``ToolRegistry.schemas()`` 按 name 排序）。
TOOL_NAMES = ("compare_methods", "describe_dataset", "describe_method",
              "describe_methods", "describe_run", "prepare_dataset", "run_method")

#: 每个工具的必填项（唯一来源 = 各自 ``execute`` 的签名，无默认值即必填）。
EXPECTED_REQUIRED = {
    "compare_methods": ["methods"],
    "describe_dataset": [],
    "describe_method": ["name"],
    "describe_methods": [],
    "describe_run": [],
    "prepare_dataset": ["source"],
    "run_method": ["method"],
}

_SYNTHETIC = {"kind": "synthetic", "task": "classification", "n_samples": 120}

#: 单次端到端调用的墙钟上限（防"挂住"把判据变成超时噪音）。
_TIMEOUT = 90.0


# ---------------------------------------------------------------- 夹具
@contextlib.contextmanager
def _authority(root, *, open_gate: bool = False, **kw):
    """起一台临时权威（数据根 = ``root``），yield ``(ml, host)``，退出即收尾。

    ``open_gate=False`` 保持出厂 ``LOCKED``（判据 4 靠它）；``True`` 模拟看门人
    代理启动意图开闸（判据 5 靠它）。
    """
    ml = assemble_ml_mecha(root=root, seed=13)
    host = McpHost(ml, port_file=Path(root) / ".mcp-port", **kw)
    try:
        host.start(timeout=30.0)
        if open_gate:
            ml.switch_ai()
        yield ml, host
    finally:
        host.stop()
        ml.close()


def _run(coro, timeout: float = _TIMEOUT):
    return asyncio.run(asyncio.wait_for(coro, timeout))


def _list_tools(url: str):
    async def body():
        async with streamable_http_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return list(result.tools)
    return _run(body())


def _call_tools(url: str, calls) -> list[tuple[bool, str]]:
    """在**同一个** MCP 会话里按顺序调多个工具，返回 ``[(is_error, text), ...]``。"""
    async def body():
        out = []
        async with streamable_http_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for name, args in calls:
                    res = await session.call_tool(name, args)
                    text = ""
                    if res.content:
                        text = getattr(res.content[0], "text", "") or ""
                    out.append((bool(res.is_error), text))
        return out
    return _run(body())


# ---------------------------------------------------------------- 1. 投影
def test_required_from_signature_and_declaration_survives():
    """inputSchema：required 来自闭包签名；声明的 type/description 覆写回协议层。"""
    with tempfile.TemporaryDirectory(prefix="ml-delivery-schema-") as td:
        ml = assemble_ml_mecha(root=td, seed=13)
        try:
            server = build_mcp_server(ml)
            tools = asyncio.run(server.list_tools())
            by_name = {t.name: t for t in tools}
            assert sorted(by_name) == list(TOOL_NAMES), sorted(by_name)

            # 参数名逐字来自 schemas() 的声明（不手抄、不多不少）
            declared = {s["name"]: s["parameters"] for s in ml.tools.schemas()}
            for name, required in EXPECTED_REQUIRED.items():
                schema = by_name[name].input_schema
                assert schema["type"] == "object", name
                assert schema["required"] == required, (name, schema["required"])
                assert list(schema["properties"]) == list(declared[name]), name
                for param, decl in declared[name].items():
                    assert schema["properties"][param]["type"] == decl["type"], (name, param)
                    assert (schema["properties"][param]["description"]
                            == decl["description"]), (name, param)

            # 声明成果过线不丢：properties 带我们的 description，且没有 pydantic
            # 派生会加的 "title"（有 title 说明覆写没生效，声明被冲掉了）
            method = by_name["run_method"].input_schema["properties"]["method"]
            assert method["type"] == "string", method
            assert method["description"] == "方法名", method
            assert "title" not in method, method

            # 可选项带真实默认值（声明里的 default），不是派生出来的 null
            persist = by_name["run_method"].input_schema["properties"]["persist"]
            assert persist.get("default") is False, persist

            for name in TOOL_NAMES:
                assert by_name[name].description.strip(), name
        finally:
            ml.close()


# ---------------------------------------------------------------- 2. 工具面到线
def test_mcp_host_exposes_every_declared_tool_over_http():
    with tempfile.TemporaryDirectory(prefix="ml-delivery-list-") as td:
        with _authority(td) as (_ml, host):
            assert host.port and host.port > 0
            assert host.url.endswith("/mcp"), host.url
            tools = _list_tools(host.url)
            names = sorted(t.name for t in tools)
            assert names == list(TOOL_NAMES), names
            for tool in tools:
                assert tool.description, tool.name
                assert tool.input_schema["type"] == "object", tool.name


# ---------------------------------------------------------------- 3. 只读真值
def test_readonly_tool_returns_real_value_over_http():
    with tempfile.TemporaryDirectory(prefix="ml-delivery-read-") as td:
        with _authority(td) as (_ml, host):
            (is_error, text), = _call_tools(host.url, [("describe_methods", {})])
            assert is_error is False, text
            payload = json.loads(text)
            assert payload["ok"] is True, payload
            assert payload["method_count"] > 0, payload
            assert len(payload["methods"]) == payload["method_count"]
            # 只读工具不写域状态
            assert _ml.history.events() == []


# ---------------------------------------------------------------- 4. LOCKED
def test_write_tool_is_fail_closed_while_locked():
    """出厂 LOCKED：写工具被结构化拒，正文纯 JSON，且不留痕。"""
    with tempfile.TemporaryDirectory(prefix="ml-delivery-locked-") as td:
        with _authority(td) as (ml, host):        # 刻意不开闸
            (is_error, text), = _call_tools(
                host.url, [("prepare_dataset", {"source": dict(_SYNTHETIC)})])
            assert is_error is True, text
            # 钉住"raise 会被加前缀"的坑：正文必须是纯 JSON
            assert "Error executing tool" not in text, text
            payload = json.loads(text)            # 解析失败即形状不对
            assert payload["ok"] is False, payload
            assert payload["error"]["info"]["kind"] == "authority_locked", payload
            # 被拒的写：快照/内存/史都不留半拉子状态
            assert ml.history.events() == [], [e.key for e in ml.history.events()]
            assert ml.engine.dataset_ids() == []


# ---------------------------------------------------------------- 5. 开闸写入
def test_open_gate_write_tools_run_and_call_id_is_attributed():
    """开闸后写工具真跑通；History 带非空 call_id（MCP request_id 进了归因）。"""
    with tempfile.TemporaryDirectory(prefix="ml-delivery-ai-") as td:
        with _authority(td, open_gate=True) as (ml, host):
            (p_err, p_txt), (r_err, r_txt) = _call_tools(host.url, [
                ("prepare_dataset", {"source": dict(_SYNTHETIC)}),
                ("run_method", {"method": "logistic"}),
            ])

            assert p_err is False, p_txt
            prepared = json.loads(p_txt)
            assert prepared["dataset_id"], prepared
            assert prepared["n_rows"] > 0, prepared

            assert r_err is False, r_txt
            run = json.loads(r_txt)
            assert run["run_id"], run
            assert run["primary_metric"], run
            assert isinstance(run["metrics"], dict), run
            # 大结果不内联（只给 run_id 与指标）
            assert "y_pred" not in r_txt, r_txt
            assert len(r_txt) < 4000, f"工具回执过大（{len(r_txt)} 字符）"

            # call_id 互引：ctx 注入没生效的话这里会是空串
            events = ml.history.events()
            ds_events = [e for e in events if e.key == "current.dataset_id"]
            assert ds_events, [e.key for e in events]
            assert ds_events[-1].actor == "ml-ai", ds_events[-1]
            assert ds_events[-1].call_id != "", "MCP request_id 未进 call_id（ctx 没注入？）"


# ---------------------------------------------------------------- 6. 端口发现
def test_port_file_written_on_start_and_removed_on_stop():
    with tempfile.TemporaryDirectory(prefix="ml-delivery-port-") as td:
        root = Path(td)
        pf = root / ".mcp-port"
        ml = assemble_ml_mecha(root=root, seed=13)
        host = McpHost(ml, port_file=pf)
        try:
            port = host.start(timeout=30.0)
            assert port > 0
            assert pf.exists(), "起服务后必须写端口文件（看门人靠它发现端点）"
            assert pf.read_text(encoding="utf-8").strip() == str(port)
        finally:
            host.stop()
            ml.close()
        assert not pf.exists(), "stop 后端口文件必须删掉（否则下次会连到死端点）"


def test_stale_port_file_from_killed_run_is_replaced_on_start():
    """上一次被强杀留下的端口文件必须被清掉/覆写（否则看门人会连死端点）。

    这条是**实测得出的**：Windows 上 ``terminate()``/``taskkill /F`` 不会跑
    Python 的 ``finally``，所以"起写止删"靠不住——必须在启动时兜一次。
    """
    with tempfile.TemporaryDirectory(prefix="ml-delivery-stale-") as td:
        root = Path(td)
        pf = root / ".mcp-port"
        pf.write_text("1", encoding="utf-8")          # 假装是死掉进程留下的
        ml = assemble_ml_mecha(root=root, seed=13)
        host = McpHost(ml, port_file=pf)
        try:
            port = host.start(timeout=30.0)
            assert port > 1, port
            assert pf.read_text(encoding="utf-8").strip() == str(port), \
                "陈旧端口文件没有被覆写 ⇒ 发现机制会连到死端点"
        finally:
            host.stop()
            ml.close()


_TESTS = [
    test_required_from_signature_and_declaration_survives,
    test_mcp_host_exposes_every_declared_tool_over_http,
    test_readonly_tool_returns_real_value_over_http,
    test_write_tool_is_fail_closed_while_locked,
    test_open_gate_write_tools_run_and_call_id_is_attributed,
    test_port_file_written_on_start_and_removed_on_stop,
    test_stale_port_file_from_killed_run_is_replaced_on_start,
]


def main() -> int:
    """不装 pytest 时的直跑入口（与仓内其它 test_*.py 同形）。"""
    # 默认控制台可能是 GBK（本机 cp936）：✓/✗ 会让直跑入口在**第一个测试之前**
    # 就崩掉（exit 1），真实失败被编码崩溃掩盖。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    failures = []
    for fn in _TESTS:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except Exception as exc:                       # noqa: BLE001
            failures.append((fn.__name__, exc))
            print(f"  ✗ {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(_TESTS) - len(failures)}/{len(_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
