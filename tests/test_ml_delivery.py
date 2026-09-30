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

P1 追加（工具面补齐 + 看门人机械件）：

7. 批量工具：**部分成功不是工具失败**（成功项的 ``run_id`` 不许被丢掉），
   全部失败才 fail loud；
8. job 三件套：``submit_run`` → ``read_job`` 拿到终态与 ``result_ref``；
   ``cancel_run`` 对不可取消命令结构化拒（``job_not_cancellable``）；
9. ``--dataset-roots`` 授权面（**真子进程** + MCP）：不给 roots ⇒ csv 来源
   fail closed；给了才允许；
10. 看门人机械件：dsh 端口 3080 硬跳过；**D13 就绪判据拒绝陈旧端口文件**；
    交接（terminate + 等退出）后写租约真的释放、同根可再起一台权威。

迁移前置（P3，给 mecha extras 的 MCP 投影迁移当安全网）：

11. **null 垫片逐参数判据**：``defaulted``（真实默认非 None 的参数）的**唯一权威
    来源是声明**（R8-2a，包装 `**kwargs` 会遮住内层闭包默认）——显式 ``null`` 被剥
    （真实默认生效）、真值原样过线、真默认本就是 ``None`` 的参数**必须保留**显式
    null；并把规则当全函数穷举（每个可选参数落进唯一一格）+ 单来源/滥收突变体的
    灵敏度（致死性）判据。契约见框架仓的 mcp-contract 文档
    §8.1。

运行：``python -m pytest tests/test_ml_delivery.py -q``
也可直接 ``python tests/test_ml_delivery.py``（自带 main，与仓内其它 test_*.py 同形；
**故意不 import pytest**，直跑入口不需要它）。

## 环境要求

- mecha 在 ``sys.path`` 上（``MECHA_ROOT`` 或与之同级的框架仓）；
- ``mcp`` 已安装（服务端 ``mcp.server.mcpserver`` + 客户端
  ``mcp.client.streamable_http``）；
- 只在**回环**上起临时端口（``port=0``，OS 分配）：不占固定端口、不碰仓内
  ``.ml-mecha`` / ``.mcp-port``（全部落在临时数据根）。
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ---- mecha 上 sys.path（不 vendor；环境变量优先，默认走同级仓） ----
MECHA_ROOT = Path(os.environ.get("MECHA_ROOT") or os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "mecha")))
if MECHA_ROOT.is_dir() and str(MECHA_ROOT) not in sys.path:
    sys.path.insert(0, str(MECHA_ROOT))

from mcp import ClientSession                                   # noqa: E402
from mcp.client.streamable_http import streamable_http_client   # noqa: E402

from ml_mecha.assembly import assemble_ml_mecha                 # noqa: E402
from ml_mecha.mcp_host import (McpHost, build_mcp_server,       # noqa: E402
                               command_declared_required)
# 框架助手直接用框架入口——shim 不再转一手（理由见 ml_mecha/mcp_host.py:151-155）。
from mecha.providers.mcp import tool_input_schema               # noqa: E402

#: 工具面全量（``ToolRegistry.schemas()`` 按 name 排序）。
TOOL_NAMES = ("cancel_run", "compare_methods", "describe_dataset",
              "describe_method", "describe_methods", "describe_run",
              "prepare_dataset", "read_job", "run_method", "run_method_batch",
              "submit_run")

#: 每个工具的必填项（写工具取**命令声明**，只读与 job 调度面取闭包签名）。
EXPECTED_REQUIRED = {
    "cancel_run": ["job_id"],
    "compare_methods": ["methods"],
    "describe_dataset": [],
    "describe_method": ["name"],
    "describe_methods": [],
    "describe_run": [],
    "prepare_dataset": ["source"],
    "read_job": ["job_id"],
    "run_method": ["method"],
    "run_method_batch": ["methods"],
    "submit_run": ["method"],
}

_SYNTHETIC = {"kind": "synthetic", "task": "classification", "n_samples": 120}

#: 单次端到端调用的墙钟上限（防"挂住"把判据变成超时噪音）。
_TIMEOUT = 90.0


# ---------------------------------------------------------------- 夹具
@contextlib.contextmanager
def _authority(root, *, open_gate: bool = False, toolhost=None, **kw):
    """起一台临时权威（数据根 = ``root``），yield ``(ml, host)``，退出即收尾。

    ``open_gate=False`` 保持出厂 ``LOCKED``（判据 4 靠它）；``True`` 模拟看门人
    代理启动意图开闸（判据 5 靠它）。``toolhost`` 给了就**在起服务之前**换掉传输
    seam（null 垫片判据用它做"只记录、不执行"的观测宿主，见 :class:`_StubToolHost`）；
    替换前先把真宿主交给它 ``attach``——声明面必须照抄真宿主，否则框架的 reach 守卫
    （"端点列得出、toolhost 声明 0 个"＝绑错注册表）会如实报错。
    """
    ml = assemble_ml_mecha(root=root, seed=13)
    if toolhost is not None:
        toolhost.attach(ml.software.toolhost)
        ml.software.toolhost = toolhost
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


def _frame(n: int = 160, d: int = 5, seed: int = 11):
    """判据用的分类小表（csv 授权面测试要真的落一个文件）。"""
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(seed)
    x = rng.randn(n, d)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(d)])
    frame["target"] = y
    return frame


def _spawn_authority_proc(root, *, extra_args=()):
    """起一台**真子进程**权威（``app_entry.py --authority``）；调用方负责收尾。"""
    cmd = [sys.executable, str(REPO / "app_entry.py"), "--authority",
           "--root", str(root), *extra_args]
    return subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


@contextlib.contextmanager
def _spawned_authority(root, *, open_gate: bool = True, extra_args=()):
    """真子进程权威 + 就绪等待（走 launcher 的 D13 判据），yield ``(url, proc)``。

    用真子进程而不是进程内装配，是因为 ``--dataset-roots`` 是 **CLI 授权面**：
    只有经 ``app_entry.py`` 的参数路径才证明它真的接上了。
    """
    import launcher

    args = list(extra_args)
    if not open_gate:
        args.append("--no-open-gate")
    proc = _spawn_authority_proc(root, extra_args=args)
    try:
        ok, port, reason = launcher.await_authority(proc, Path(root) / ".mcp-port",
                                                    timeout=90.0)
        assert ok is True, f"权威未就绪：{reason}"
        yield f"http://127.0.0.1:{port}/mcp", proc
    finally:
        launcher._terminate_group([proc])
        launcher.await_exit([proc], timeout=20)


def _http_get(url: str, method: str = "GET"):
    """裸 HTTP 请求（监控端点判据用 stdlib，不引第三方）。返回 (code, headers, body)。"""
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, dict(resp.headers), resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers or {}), err.read().decode("utf-8")


def _strip_js_comments(text: str) -> str:
    """剥掉 JS/TS 的块注释与整行注释（判据用；不做完整词法分析）。

    只删 ``/* ... */`` 块与**整行** ``//`` 注释——刻意不删行尾 ``//``：那会把
    ``http://127.0.0.1:8767`` 这类字面量从中间截断，正好让"硬编码端口"漏网
    （对"必须不存在"的判据来说，假阴性最危险）。
    """
    import re

    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith(("//", "*")))


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
            assert method["description"] == "已注册的方法名", method
            assert "title" not in method, method

            # 可选项带真实默认值（声明里的 default），不是派生出来的 null
            persist = by_name["run_method"].input_schema["properties"]["persist"]
            assert persist.get("default") is False, persist

            for name in TOOL_NAMES:
                assert by_name[name].description.strip(), name
        finally:
            ml.close()


# ------------------------------------------------- 1a2. 「假契约不得回归」判据
def test_mcp_wire_declares_no_output_schema():
    """线上 ``tools/list`` 的 ``outputSchema`` **必须为空**（假契约不得回归）。

    为什么要有这条：MCP SDK 会**从 handler 的返回注解派生** ``outputSchema``，并把它
    放进 ``tools/list``。历史上合成签名带 ``return_annotation=str`` ⇒ 线上真的出现了
    ``{"properties":{"result":…},"required":["result"]}`` —— 那是**谎称**"返回
    ``{result: string}``"，而工具实际返回各自领域回执的 JSON。它没有任何消费者，
    纯属把实现痕迹写进模型可见面（契约 §1.2 的 F 更正）。

    这条判据**不看源码**（源码里 grep `outputSchema` 是零命中，正是漏检点），只看
    **线上协议输出**；不需要模型、不依赖 dsh 实现，因此适合当回归门。
    """
    with tempfile.TemporaryDirectory(prefix="ml-delivery-outschema-") as td:
        with _authority(td) as (_ml, host):
            tools = _list_tools(host.url)
            assert sorted(t.name for t in tools) == list(TOOL_NAMES)
            offenders = [(t.name, t.output_schema) for t in tools
                         if getattr(t, "output_schema", None) is not None]
            assert not offenders, f"线上出现了 outputSchema（假契约）：{offenders}"


def test_mcp_output_schema_judge_can_go_red():
    """上一条的**灵敏度证据**：让派生注解回来，它就必然红。

    复刻历史的成因（``return_annotation=str``）即可让 ``tools/list`` 带上
    ``outputSchema``；本判据断言"确实带上了"——即上面那条在这具突变体上会失败。
    不起端口（纯构造），因此这条证据几乎零成本。
    """
    from unittest import mock

    import mecha.providers.mcp as _mcp_provider

    original = _mcp_provider.synthesize_signature

    def _legacy(schema, required, **kwargs):
        sig = original(schema, required, **kwargs)
        return sig.replace(return_annotation=str)      # 历史成因

    with tempfile.TemporaryDirectory(prefix="ml-delivery-outschema-mut-") as td:
        ml = assemble_ml_mecha(root=td, seed=13)
        try:
            with mock.patch.object(_mcp_provider, "synthesize_signature",
                                   _legacy):
                server = build_mcp_server(ml)
                mutated = asyncio.run(server.list_tools())
            derived = {t.name: t.output_schema for t in mutated
                       if t.output_schema is not None}
            assert derived, "突变体没有让 outputSchema 回来 ⇒ 上一条判据抓不到这类回归"
            # 形状必须是**历史上真实出现过**的那一个（不是随便什么非空值）：
            # 派生自 ``return_annotation=str`` ⇒ required: ["result"]
            sample = derived[sorted(derived)[0]]
            assert sample.get("required") == ["result"], sample
            assert "result" in (sample.get("properties") or {}), sample
            # 去掉突变体后必须干净（否则上一条会误绿）
            server2 = build_mcp_server(ml)
            clean = asyncio.run(server2.list_tools())
            assert all(t.output_schema is None for t in clean)
        finally:
            ml.close()


# ------------------------------------------------- 1a3. 「公开名单不得撒谎」判据
#: 至少要覆盖这两个模块（d561e13 踩的坑就在这里）。加别的模块欢迎，但不做大扫除。
_PUBLIC_SURFACE_MODULES = ("ml_mecha.mcp_host", "ml_mecha.monitor_http")


def _public_surface_violations(names, attributes) -> list[str]:
    """``__all__`` 里"实际不存在"的名字（含重复项）。纯函数，便于自测灵敏度。"""
    existing = set(attributes)
    missing = sorted(n for n in names if n not in existing)
    seen: set = set()
    dupes: list = []
    for n in names:
        if n in seen and n not in dupes:
            dupes.append(n)
        seen.add(n)
    return missing + [f"{n}（重复）" for n in dupes]


def test_public_surface_all_names_exist():
    """``__all__`` ⊆ 实际属性：**对外声明的名单不许撒谎**。

    踩过的坑（commit d561e13）：``monitor_http.__all__`` 里留着已不再 import 的
    ``CORE_ROUTES`` ⇒ **模块导入不报错、全部判据全绿**，但 ``import *`` 会
    ``AttributeError``、按 ``__all__`` 枚举公开面的工具会列出不存在的名字。
    静默失败 ⇒ 只能靠判据拦。
    """
    import importlib

    problems = {}
    for mod_name in _PUBLIC_SURFACE_MODULES:
        mod = importlib.import_module(mod_name)
        violations = _public_surface_violations(getattr(mod, "__all__", []),
                                                dir(mod))
        if violations:
            problems[mod_name] = violations
    assert not problems, f"__all__ 里有不存在的名字：{problems}"


def test_public_surface_judge_can_go_red():
    """上一条的**灵敏度证据**：名单里塞一个假名字必须被算出来。"""
    assert _public_surface_violations(["ok", "不存在的东西"], ["ok"]) == \
        ["不存在的东西"]
    assert _public_surface_violations(["ok", "ok"], ["ok"]) == ["ok（重复）"]
    assert _public_surface_violations(["ok"], ["ok"]) == []


# ------------------------------------------- 1b. null 垫片（迁移前置判据）
#
# 契约（`_mecha-extraction/mcp-contract.md` §8.1 R8-1/R8-2/R8-2a）：
# ``defaulted`` = 「**声明**了非 None default」的参数集合；handler 收到显式 null 时
# 丢弃该键，让被调函数的真实默认生效。
#
# 为什么单列一节 + 为什么权威必须是**声明**：早期实现用「声明 ∪ 闭包签名」双来源，
# 而写工具的 ``execute=wrap(execute)`` 包装（``def wrapped(**kwargs)``）会让**内层
# 闭包默认全部不可见** ⇒ 有一类参数两个来源同时落空，显式 null 会以 ``None`` 到达
# 工具体（症状是模型侧行为变差而非报错，端到端判据抓不到）。R8-2a 之后每个"真实默认
# 非 None"的参数都在声明里写出 default，于是：**来源唯一、包装是否透明不再影响正确性**。
#
# 这一节把规则钉成**全函数**（每个可选参数落进唯一一格）并给出判据灵敏度证据；
# ``test_null_shim_single_source_mutant_is_detected`` 是致死性证据。

#: ``_StubToolHost.call`` 的固定回执（判据只关心 seam 收到的 args，不关心结果值）。
_STUB_VALUE = {"ok": True, "stub": True}


class _StubToolHost:
    """传输 seam 的观测宿主：``call`` **只记录、不执行**；声明面照抄真宿主。

    为什么必须 stub 而不是用真的 ``LocalToolHost``：真执行会带来副作用（起后台
    job、写 Gate、落 runs/、跑真训练），让"参数去留"的断言被无关噪声拖累。本类在
    ``call`` 处截断，于是判据测的正是**投影层**（pydantic 入参 → 合成签名 → 垫片 →
    seam）的契约，且零副作用：11 个工具（含 ``submit_run``）都不会真跑。

    ⚠ **声明面必须照抄真宿主**（``attach``）：框架有一条 reach 守卫
    （``mecha.providers.mcp._check_toolhost_reach``）——"端点列得出 N 个工具、而
    toolhost 声明 0 个"＝绑错了注册表，当场 fail loud。本类只改 ``call`` 的行为，
    **不改** ``schemas()/declare()/teardown()`` 的语义（照抄才是忠实的观测点）；
    早先这里让 ``schemas()`` 返回 ``[]``，被那条守正如实抓到（是 stub 在撒谎，
    不是守卫误报）。
    """

    def __init__(self, inner=None):
        self.inner = inner
        self.calls: list[tuple[str, dict]] = []

    def attach(self, inner):
        """接上真宿主（``_authority`` 在替换 seam 之前调用）。"""
        self.inner = inner
        return self

    def declare(self, schemas):
        if self.inner is not None:
            return self.inner.declare(schemas)

    def schemas(self):
        return [] if self.inner is None else self.inner.schemas()

    def teardown(self):
        if self.inner is not None:
            return self.inner.teardown()

    def call(self, name, args=None):
        self.calls.append((name, dict(args or {})))
        return {"is_error": False, "value": dict(_STUB_VALUE), "content": []}


def _defaulted_of(ml, name: str):
    """该工具**真实**算出来的 ``defaulted``（与注册路径同一条调用）。

    ⚠ 迁移后断言对象是**框架行为**（``mecha.providers.mcp.tool_input_schema``，
    ``ml_mecha.mcp_host`` 只是再导出）：`defaulted` 的权威来源是声明、必填来自
    ML 的 ``RequiredSource``（命令声明）。签名形状也随之改为关键字传 ``execute``。
    """
    schema = next(s for s in ml.tools.schemas() if s["name"] == name)
    _input_schema, defaulted = tool_input_schema(
        name, schema, execute=ml.tools.get(name).execute,
        declared_required=command_declared_required(ml, name))
    return set(defaulted), schema


def _tool_input_schema(ml, name: str):
    """该工具协议层的 ``inputSchema``（同一条调用；穷举判据用它取必填集合）。"""
    schema = next(s for s in ml.tools.schemas() if s["name"] == name)
    input_schema, _defaulted = tool_input_schema(
        name, schema, execute=ml.tools.get(name).execute,
        declared_required=command_declared_required(ml, name))
    return input_schema


def _real_default(ml, tool: str, param: str):
    """该参数**声明**的真实默认（R8-2a 之后声明是唯一权威来源）。

    ``None`` 有两种含义，判据里分开处理：① 真实默认就是 ``None``（该保留显式 null）；
    ② 声明里没有 ``default`` 键（视同真默认 ``None``，同样保留显式 null）。两者在
    null 垫片规则里落在同一格，但语义不同，所以这里如实区分（闭包签名只作为兜底读取，
    仅用于诊断——本项目 11 个工具的 defaulted 已全部由声明承载，见
    ``test_null_shim_authority_is_declaration_only``）。
    """
    _defaulted, schema = _defaulted_of(ml, tool)
    decl = schema["parameters"].get(param) or {}
    if "default" in decl:
        return decl["default"]
    sig = inspect.signature(ml.tools.get(tool).execute)
    if param in sig.parameters:
        return sig.parameters[param].default
    return None


#: ``defaulted`` 全表快照（实测；迁移不得静默增删——多了会吞掉"清空语义"的显式 null，
#: 少了会让某类工具的默认值失效）。键是工具名，值是**排序后**的参数名列表。
#: 16 项 = 全部「声明了非 None default」的参数（R8-2a：声明是唯一权威来源）。
EXPECTED_DEFAULTED = {
    "cancel_run": [],
    "compare_methods": ["dataset_id"],
    "describe_dataset": ["dataset_id"],
    "describe_method": [],
    "describe_methods": ["dataset_id", "family"],
    "describe_run": ["run_id"],
    "prepare_dataset": ["diag", "name", "target", "time_col"],
    "read_job": ["wait_s"],
    "run_method": ["dataset_id", "device", "diag", "persist"],
    "run_method_batch": ["dataset_id"],
    "submit_run": ["dataset_id", "device"],
}

#: 逐参数行为判据：(工具, 最小必填参数, 参数名, 真值)。
#: **覆盖全部 16 个 defaulted 参数**，真值必须是该参数类型的合法取值。
#: 备注一栏说明该参数的默认值原本来自哪一侧（R8-2a 之后一律由**声明**承载）：
#: - 写工具的 `**kwargs` 包装会遮住内层闭包 ⇒ 只能靠声明（`diag`/`persist`/`name`/
#:   `target`/`time_col`/`dataset_id`/`device`）；
#: - 只读与 job 调度面未被包装，闭包默认本来可见（`family`/`run_id`/`wait_s`），
#:   现同样显式声明，让"声明 = 唯一权威"成为事实。
_DEFAULTED_PROBES = (
    ("describe_methods", {}, "family", "linear"),
    ("describe_methods", {}, "dataset_id", "ds-null-shim"),
    ("describe_dataset", {}, "dataset_id", "ds-null-shim"),
    ("describe_run", {}, "run_id", "run-null-shim"),
    ("prepare_dataset", {"source": dict(_SYNTHETIC)}, "name", "my-ds"),
    ("prepare_dataset", {"source": dict(_SYNTHETIC)}, "target", "target"),
    ("prepare_dataset", {"source": dict(_SYNTHETIC)}, "time_col", "t"),
    ("prepare_dataset", {"source": dict(_SYNTHETIC)}, "diag", True),
    ("run_method", {"method": "logistic"}, "dataset_id", "ds-null-shim"),
    ("run_method", {"method": "logistic"}, "device", "cpu"),
    ("run_method", {"method": "logistic"}, "persist", True),
    ("run_method_batch", {"methods": ["logistic"]}, "dataset_id", "ds-null-shim"),
    ("compare_methods", {"methods": ["logistic"]}, "dataset_id", "ds-null-shim"),
    ("read_job", {"job_id": "job-null-shim"}, "wait_s", 5.0),
    ("submit_run", {"method": "logistic"}, "dataset_id", "ds-null-shim"),
    ("submit_run", {"method": "logistic"}, "device", "cpu"),
)

#: 真实默认**本就是 None** 的参数（10 项）：显式 null 必须**原样保留**（这是"清空/
#: 不指定"的唯一表达方式，也是 EL P0-1 注释里点名不许误伤的一类）。垫片必须是"按
#: ``defaulted`` 精确剥"，**不是**"见 null 就剥"。
_NONE_DEFAULTED_PROBES = (
    ("run_method", {"method": "logistic"}, "overrides", {"k": 1}),
    ("run_method", {"method": "logistic"}, "seed", 7),
    ("run_method", {"method": "logistic"}, "resource_guard", {"max_kernel_mb": 64}),
    ("run_method_batch", {"methods": ["logistic"]}, "overrides", {"k": 1}),
    ("run_method_batch", {"methods": ["logistic"]}, "seed", 7),
    ("run_method_batch", {"methods": ["logistic"]}, "resource_guard", {"max_kernel_mb": 64}),
    ("compare_methods", {"methods": ["logistic"]}, "overrides", {"k": 1}),
    ("compare_methods", {"methods": ["logistic"]}, "seed", 7),
    ("submit_run", {"method": "logistic"}, "overrides", {"k": 1}),
    ("submit_run", {"method": "logistic"}, "seed", 7),
)


def test_null_shim_drops_explicit_null_for_every_defaulted_param():
    """``defaulted`` 的每个参数：显式 null 被剥（真实默认生效），真值原样过线。

    三条断言缺一不可：**省略**与**显式 null** 都不许把该键传给被调函数（合成签名
    给可选项的 ``default=None`` 只是为了让显式 null 过 pydantic，不是让它落进闭包）；
    **真值**必须逐字过线（垫片不能变成"见可选就剥"）。
    """
    with tempfile.TemporaryDirectory(prefix="ml-null-shim-") as td:
        stub = _StubToolHost()
        with _authority(td, toolhost=stub) as (ml, host):
            got = {name: sorted(_defaulted_of(ml, name)[0]) for name in TOOL_NAMES}
            assert got == EXPECTED_DEFAULTED, got

            calls = []
            for tool, base, param, real in _DEFAULTED_PROBES:
                calls.append((tool, dict(base)))
                calls.append((tool, dict(base, **{param: None})))
                calls.append((tool, dict(base, **{param: real})))
            results = _call_tools(host.url, calls)
            # 显式 null 必须能过 pydantic（合成签名的可选项注解为 ``T | None``）
            assert all(not err for err, _ in results), [r for r in results if r[0]]
            assert len(stub.calls) == len(calls), (len(stub.calls), len(calls))

            for i, (tool, _base, param, real) in enumerate(_DEFAULTED_PROBES):
                omit = stub.calls[3 * i][1]
                null = stub.calls[3 * i + 1][1]
                value = stub.calls[3 * i + 2][1]
                # 前提守卫：defaulted 的语义就是"真实默认非 None"（挡住把 None 默认滥收）
                assert _real_default(ml, tool, param) is not None, (tool, param)
                assert param not in omit, (tool, param, omit)
                assert param not in null, (tool, param, null)
                assert value.get(param) == real, (tool, param, value)


def test_null_shim_keeps_explicit_null_when_real_default_is_none():
    """真默认是 ``None`` 的参数：显式 null **原样保留**（垫片必须精确、不搞一刀切）。"""
    with tempfile.TemporaryDirectory(prefix="ml-null-keep-") as td:
        stub = _StubToolHost()
        with _authority(td, toolhost=stub) as (ml, host):
            calls = []
            for tool, base, param, real in _NONE_DEFAULTED_PROBES:
                calls.append((tool, dict(base, **{param: None})))
                calls.append((tool, dict(base, **{param: real})))
            results = _call_tools(host.url, calls)
            assert all(not err for err, _ in results), [r for r in results if r[0]]

            for i, (tool, _base, param, real) in enumerate(_NONE_DEFAULTED_PROBES):
                null = stub.calls[2 * i][1]
                value = stub.calls[2 * i + 1][1]
                assert param not in _defaulted_of(ml, tool)[0], (tool, param)
                assert param in null and null[param] is None, (tool, param, null)
                assert value.get(param) == real, (tool, param, value)


#: 穷举判据每个工具的最小必填入参（工具不同，必填项也不同）。
_MIN_ARGS = {
    "cancel_run": {"job_id": "job-null-shim"},
    "compare_methods": {"methods": ["logistic"]},
    "describe_dataset": {},
    "describe_method": {"name": "logistic"},
    "describe_methods": {},
    "describe_run": {},
    "prepare_dataset": {"source": dict(_SYNTHETIC)},
    "read_job": {"job_id": "job-null-shim"},
    "run_method": {"method": "logistic"},
    "run_method_batch": {"methods": ["logistic"]},
    "submit_run": {"method": "logistic"},
}

#: 可选参数总数（26 = 全部声明参数 − 必填参数）；迁移后必须仍是这个数才说明
#: "每个可选参数都被判据看过一遍"（新增参数会在这里红，提醒补探针）。
_EXPECTED_OPTIONAL_PARAMS = 28


def test_null_shim_rule_partitions_every_optional_param():
    """把规则当**全函数**穷举：每个可选参数落进唯一一格，不存在第三态。

    三格（契约 R8-2a）：
      声明 ``default`` 非 None  → 进 ``defaulted`` → **丢**显式 null（让真默认生效）
      声明 ``default`` 为 None  → **保留**显式 null
      声明没有 ``default`` 键    → 视同真默认 None → **保留**显式 null

    动机：旧实现用「声明 ∪ 闭包签名」双来源，而 ``**kwargs`` 包装会让内层闭包默认
    **不可见** ⇒ 有一类参数同时落空（静默缺口）。本条判据逐参数走过全部可选参数，
    任何"第三态"（既不在 defaulted、真默认又不是 None）都会当场红——**这条取代了
    早期那条"把 7 项缺口钉成数据"的迁移期判据**（缺口已在声明层修掉）。
    """
    with tempfile.TemporaryDirectory(prefix="ml-null-partition-") as td:
        stub = _StubToolHost()
        with _authority(td, toolhost=stub) as (ml, host):
            optional = []          # [(tool, param, decl)]
            defaulted_all = {}     # {tool: set}
            for name in TOOL_NAMES:
                defaulted, schema = _defaulted_of(ml, name)
                defaulted_all[name] = defaulted
                required = set(_tool_input_schema(ml, name)["required"])
                for param, decl in schema["parameters"].items():
                    if param in required:
                        continue
                    optional.append((name, param, decl))

            assert len(optional) == _EXPECTED_OPTIONAL_PARAMS, \
                f"可选参数 {len(optional)} != {_EXPECTED_OPTIONAL_PARAMS}（新增/删除参数请补探针）"

            results = _call_tools(host.url, [(tool, {**_MIN_ARGS[tool], param: None})
                                             for tool, param, _d in optional])
            # 显式 null 对**每个**可选参数都必须能过 pydantic（注解为 ``T | None``）
            assert all(not err for err, _ in results), [r for r in results if r[0]]
            assert len(stub.calls) == len(optional), (len(stub.calls), len(optional))

            for i, (tool, param, decl) in enumerate(optional):
                null = stub.calls[i][1]
                declared_non_none = decl.get("default") is not None
                # 格归属必须由**声明**唯一决定（R8-2a：签名侧不得再承重）
                assert (param in defaulted_all[tool]) is declared_non_none, \
                    (tool, param, decl, sorted(defaulted_all[tool]))
                if declared_non_none:
                    assert param not in null, (tool, param, null)
                else:
                    assert param in null and null[param] is None, (tool, param, null)


def test_null_shim_authority_is_declaration_only():
    """R8-2a 的核心不变量：``defaulted`` **等于**「声明了非 None default 的参数」。

    两条都要成立，缺一条就说明有参数在依赖看不见的东西：
    ① **完备**：声明了非 None default ⇒ 必在 ``defaulted``（否则"默认值生效"是空话）；
    ② **唯一**：``defaulted`` 里的每一项都必须有声明来源（否则它靠的是运行期闭包——
       包装一变就失效，正是 R8-2a 要消灭的隐式依赖）。

    纯集合比较、不打端口：行为面已由上面两条判据与 ``_DEFAULTED_PROBES`` 覆盖。
    """
    with tempfile.TemporaryDirectory(prefix="ml-null-authority-") as td:
        ml = assemble_ml_mecha(root=td, seed=13)
        try:
            declared_only = {}
            for name in TOOL_NAMES:
                _defaulted, schema = _defaulted_of(ml, name)
                declared_only[name] = sorted(
                    p for p, d in schema["parameters"].items()
                    if d.get("default") is not None)
            measured = {name: sorted(_defaulted_of(ml, name)[0]) for name in TOOL_NAMES}
            assert measured == declared_only, measured
            assert measured == EXPECTED_DEFAULTED, measured
        finally:
            ml.close()


def test_null_shim_single_source_mutant_is_detected():
    """**致死性证据**：把 ``defaulted`` 算错，正常判据必然变红。

    这不是测实现细节，而是测**判据的灵敏度**——用同一个 seam 观测点证明两种典型
    假实现会被抓到：

    - **杀掉声明侧**（``_declaration_defaulted`` → 空）⇒ 全部 16 项失去垫片，
      ``persist`` 的显式 null 会以 ``None`` 到达被调函数；
    - **滥收**（把真默认为 ``None`` 的 ``overrides`` 也算进 defaulted）⇒
      ``test_null_shim_keeps_explicit_null_when_real_default_is_none`` 失守
      （"按 defaulted 精确剥"退化成"一刀切剥"，清空语义被吞）。

    "只取签名侧"这一方向**不再是突变体**：R8-2a 之后签名侧对本项目的 11 个工具不再
    贡献任何一项，这一点由 ``test_null_shim_authority_is_declaration_only`` 永久断言
    （那种假实现只会让 ``defaulted`` 变小，已被第一条突变体覆盖）。

    ⚠ 迁移后本判据打的是**框架函数** ``mecha.providers.mcp._declaration_defaulted``
    （``ml_mecha.mcp_host`` 只是再导出，自己不再有这份实现）——断言对象从"自家实现"
    改为"框架行为"，强度不变（仍是同一个 seam 观测点 + 同一对突变方向）。
    """
    from unittest import mock

    import mecha.providers.mcp as _mcp_provider

    def probe(mutant_patch, tool, base, param, *, normally_defaulted, mutant_keeps_null):
        """在突变体下探测：归属是否**翻转**、观测是否与正常判据相反。

        ``normally_defaulted`` = 正常实现下该参数是否在 ``defaulted`` 里；
        ``mutant_keeps_null`` = 突变体下显式 null 是否**留着**（= 正常判据会红的证据）。
        """
        with tempfile.TemporaryDirectory(prefix="ml-null-mutant-") as td:
            stub = _StubToolHost()
            with mutant_patch, _authority(td, toolhost=stub) as (ml, host):
                defaulted, _schema = _defaulted_of(ml, tool)
                assert (param in defaulted) is not normally_defaulted, \
                    (tool, param, sorted(defaulted))
                _call_tools(host.url, [(tool, dict(base, **{param: None}))])
                null = stub.calls[-1][1]
                kept = param in null
                assert kept is mutant_keeps_null, (tool, param, null)
                if kept:
                    assert null[param] is None, (tool, param, null)

    # 杀掉声明侧：所有 defaulted 失守（剥 null 的判据会红）
    probe(mock.patch.object(_mcp_provider, "_declaration_defaulted", lambda _d: set()),
          "run_method", {"method": "logistic"}, "persist",
          normally_defaulted=True, mutant_keeps_null=True)
    # 滥收（真默认为 None 的也进 defaulted）：清空语义失守（保 null 的判据会红）
    probe(mock.patch.object(_mcp_provider, "_declaration_defaulted",
                            lambda _d: {"overrides"}),
          "run_method", {"method": "logistic"}, "overrides",
          normally_defaulted=False, mutant_keeps_null=False)


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
            assert ml.history.events() == [], [(e.op, e.target) for e in ml.history.events()]
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
            ds_events = [e for e in events if e.target == "current.dataset_id"]
            assert ds_events, [(e.op, e.target) for e in events]
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


# ------------------------------------------------------- P1：job 调度面
def test_batch_tool_preserves_partial_success():
    """批量工具：部分成功**不是**工具失败（成功项的 run_id 不许被丢掉）。

    怎么造"部分成功"：``ridge`` 之类的**事前**不兼容会整批被拒（那是总失败），
    真正逐方法隔离的是**运行期**失败——这里用极小的 kernel 预算卡住 ``svc``，
    ``logistic`` 照跑。engine 的 ``_execute_isolated`` 把后者隔离成 ok=False。
    """
    with tempfile.TemporaryDirectory(prefix="ml-delivery-batch-") as td:
        with _authority(td, open_gate=True) as (_ml, host):
            (p_err, p_txt), = _call_tools(host.url, [
                ("prepare_dataset", {"source": dict(_SYNTHETIC)})])
            assert p_err is False, p_txt
            dataset_id = json.loads(p_txt)["dataset_id"]

            err, txt = _call_tools(host.url, [
                ("run_method_batch", {"methods": ["logistic", "svc"],
                                      "dataset_id": dataset_id,
                                      "resource_guard": {"max_kernel_mb": 0.000001}})])[0]
            body = json.loads(txt)
            assert err is False, txt                  # 部分成功 ⇒ 不是 is_error
            assert body["ok"] is False, body           # 但如实报"有失败"
            assert body["failed_methods"] == ["svc"], body
            # 成功项的 run_id 真的在（这正是"不整批转异常"的意义）
            assert len(body["run_ids"]) == 2, body
            assert body["runs"][0]["run_id"], body["runs"][0]
            assert body["runs"][0]["ok"] is True, body["runs"][0]
            assert body["runs"][1]["ok"] is False, body["runs"][1]

            # 全部失败（事前不兼容 ⇒ 整批被拒）仍是写工具纪律的 fail loud
            all_bad = _call_tools(host.url, [
                ("run_method_batch", {"methods": ["ridge"],
                                      "dataset_id": dataset_id})])[0]
            assert all_bad[0] is True, all_bad
            assert json.loads(all_bad[1])["error"]["info"]["kind"], all_bad[1]


def test_job_tools_submit_read_and_result_ref():
    """``submit_run`` → ``read_job`` 拿到终态与 ``result_ref``（含 run_id）。"""
    with tempfile.TemporaryDirectory(prefix="ml-delivery-job-") as td:
        with _authority(td, open_gate=True) as (_ml, host):
            (p_err, p_txt), = _call_tools(host.url, [
                ("prepare_dataset", {"source": dict(_SYNTHETIC)})])
            assert p_err is False, p_txt
            dataset_id = json.loads(p_txt)["dataset_id"]

            err, txt = _call_tools(host.url, [
                ("submit_run", {"method": "logistic", "dataset_id": dataset_id})])[0]
            assert err is False, txt
            submitted = json.loads(txt)
            assert submitted["job_id"], submitted
            assert submitted["command"] == "run_method", submitted
            assert submitted["cancel_supported"] is True, submitted

            # 轮询到终态（read_job 的 wait_s 是可选便利，这里用它少转几圈）
            job = submitted["job_id"]
            final = {}
            for _ in range(60):
                err, txt = _call_tools(host.url, [
                    ("read_job", {"job_id": job, "wait_s": 1})])[0]
                assert err is False, txt
                final = json.loads(txt)
                if final["state"] in ("done", "failed", "cancelled"):
                    break
            assert final["state"] == "done", final
            assert final["result_ref"].get("run_ids"), final
            assert final["result_ref"]["command"] == "run_method", final
            assert final["error"] == "", final

            # 未知任务编号：结构化拒，不是假成功
            err, txt = _call_tools(host.url, [("read_job", {"job_id": "job-9999"})])[0]
            assert err is True, txt
            assert json.loads(txt)["error"]["info"]["kind"] == "unknown_job", txt


def test_cancel_run_rejects_uncancellable_job():
    """``cancel_run`` 对不可取消的命令**结构化拒**（job_not_cancellable）。

    ``prepare_dataset`` 在核心命令面声明 ``cancel_supported=False``。要拿到一个
    "不可取消的 job"，用宿主自己的 ``submit``（同一个 seam，不是绕过工具面）。
    """
    with tempfile.TemporaryDirectory(prefix="ml-delivery-cancel-") as td:
        with _authority(td, open_gate=True) as (ml, host):
            job = ml.submit({"command": "prepare_dataset", "name": "u",
                             "source": dict(_SYNTHETIC)}, side="ai")
            assert job.cancel_supported is False, job.to_dict()
            ml.wait_job(job, 30)

            err, txt = _call_tools(host.url, [("cancel_run", {"job_id": job.id})])[0]
            assert err is True, txt
            body = json.loads(txt)
            assert body["error"]["info"]["kind"] == "job_not_cancellable", body

            # 可取消的 job：cancel_run 受理并**如实**回报（不谎报"已取消"）
            ok_job = ml.submit({"command": "run_method", "method": "logistic",
                                "dataset_id": ml.engine.dataset_ids()[-1]}, side="ai")
            err, txt = _call_tools(host.url, [("cancel_run", {"job_id": ok_job.id})])[0]
            assert err is False, txt
            body = json.loads(txt)
            # ⚠ **不要断言"运行中 ⇒ cancel_requested=True"**：那是**竞态**——小数据集上的
            # `run_method` 可能在这次调用前就跑完（实测在门禁负载下真的发生了）。
            # 断**契约本身**（两个方向都成立、且与速度无关）：
            #   终态 ⇒ 什么都没取消（cancel_requested=False）；非终态 ⇒ 确实请求了取消。
            assert body["state_before"], body           # 请求前的状态（不是编的）
            assert body["terminal"] in (True, False), body
            assert body["cancel_requested"] is (not body["terminal"]), body

            # ⭐ **对偶（这条才是"说真话"的要害，且确定性）**：先**等它跑完**，再调 cancel_run，
            #    必须如实答"我什么都没取消"——`cancel_requested=False` + `terminal=True`。
            #    ⚠ 从前这里**硬编** `cancel_requested: True`，于是对终态 job 也说"已请求取消"
            #      （框架 ADR 点名的那个谎，在我们自己的工具面上活着）⇒ 本对偶就是它的守卫。
            done_job = ml.submit({"command": "run_method", "method": "logistic",
                                  "dataset_id": ml.engine.dataset_ids()[-1]}, side="ai")
            ml.wait_job(done_job, 300)
            err, txt = _call_tools(host.url, [("cancel_run", {"job_id": done_job.id})])[0]
            assert err is False, txt
            after = json.loads(txt)
            assert after["terminal"] is True, after
            assert after["cancel_requested"] is False, (
                f"对**已结束**的 job 仍报 cancel_requested=True ⇒ 这正是要防的谎：{after}")


# ------------------------------------------------------- P1：数据根授权面
def test_dataset_roots_authorization_is_fail_closed():
    """``--dataset-roots``：不给就 fail closed，给了才允许 csv（真子进程 + MCP）。"""
    with tempfile.TemporaryDirectory(prefix="ml-delivery-roots-") as td:
        root = Path(td)
        data_dir = root / "data"
        data_dir.mkdir()
        csv_path = data_dir / "train.csv"
        _frame().to_csv(csv_path, index=False)

        # (a) 不给 roots：csv 来源必须在副作用前被拒
        with _spawned_authority(root, open_gate=True) as (url, _proc):
            err, txt = _call_tools(url, [("prepare_dataset", {
                "source": {"kind": "csv", "path": str(csv_path),
                           "target": "target"}})])[0]
            assert err is True, txt
            assert json.loads(txt)["error"]["info"]["kind"] == \
                "dataset_roots_required", txt

        # (b) 给了 roots：同一份 csv 可以准备（授权面是人显式给的）
        with tempfile.TemporaryDirectory(prefix="ml-delivery-roots-ok-") as td2:
            with _spawned_authority(Path(td2), open_gate=True,
                                    extra_args=["--dataset-roots", str(data_dir)]) as (url, _p):
                err, txt = _call_tools(url, [("prepare_dataset", {
                    "source": {"kind": "csv", "path": str(csv_path),
                               "target": "target"}})])[0]
                assert err is False, txt
                body = json.loads(txt)
                assert body["dataset_id"], body
                assert body["n_rows"] > 0, body


# ------------------------------------------------------- P1：看门人机械件
def test_launcher_pick_dsh_port_skips_official_3080():
    """dsh 端口从 3081 起找，**3080 硬跳过**（官方实例的端口永不占）。"""
    import launcher

    for _ in range(5):
        port = launcher.pick_dsh_port()
        assert port != 3080, port
        assert 3081 <= port <= 3090, port


def test_launcher_await_authority_rejects_stale_port_file():
    """D13 就绪判据：**陈旧端口文件不算就绪**——必须内容新鲜 + HTTP 在听。

    这条直接钉住"只判文件存在"的假绿：先放一个陈旧 ``.mcp-port``（指向一个
    没人听的端口），若看门人只看存在性，它会立刻返回就绪并把 dsh 指向死端口。
    """
    import launcher

    with tempfile.TemporaryDirectory(prefix="ml-delivery-ready-") as td:
        root = Path(td)
        pf = root / ".mcp-port"
        # 陈旧文件：内容是 1（几乎肯定没人听），且 mtime 早于 spawn
        pf.write_text("1", encoding="utf-8")
        time.sleep(0.05)
        pf.write_text("1", encoding="utf-8")          # 内容不变，只更新 mtime
        stale_mtime = pf.stat().st_mtime

        proc = _spawn_authority_proc(root, extra_args=["--no-open-gate"])
        try:
            ok, port, reason = launcher.await_authority(proc, pf, timeout=90.0)
            assert ok is True, reason
            assert port > 1, port
            assert port != 1, "把陈旧端口当成就绪了"
            assert pf.read_text(encoding="utf-8").strip() == str(port)
            assert pf.stat().st_mtime >= stale_mtime
            assert launcher._probe_http("127.0.0.1", port), port

            # 交接纪律：terminate + 等退出 ⇒ 写租约释放（同根可再起一台权威）
            launcher._terminate_group([proc])
            launcher.await_exit([proc], timeout=20)
            assert proc.poll() is not None, "等退出后进程必须真的没了"
            second = _spawn_authority_proc(root, extra_args=["--no-open-gate"])
            try:
                assert second.poll() is None
                ok2, port2, reason2 = launcher.await_authority(second, pf, timeout=90.0)
                assert ok2 is True, reason2
                assert port2 != port, (port, port2)
            finally:
                launcher._terminate_group([second])
                launcher.await_exit([second], timeout=20)
        finally:
            launcher._terminate_group([proc])
            launcher.await_exit([proc], timeout=10)

        # await_authority 对"起不来的权威"必须如实失败，不假装就绪
        class _Dead:
            returncode = 3

            @staticmethod
            def poll():
                return 3

        ok3, port3, reason3 = launcher.await_authority(_Dead(), pf, timeout=5.0)
        assert ok3 is False and port3 is None
        assert "退出" in reason3, reason3


# ------------------------------------------------------- P2：只读监控端点
def test_monitor_shapes_match_el_cockpit_contract():
    """监控端点**逐字段**对齐 EL `_MonitorHandler`——面板才能不改渲染逻辑。

    只读端点是"面板能不能直接用"的契约面，所以断言的是**字段名与取值语义**，
    不是"有响应就算过"。
    ⚠ **2026-09-30 随 mecha `926b90d` 更新期望**（事件形状改名）：`/history` 现在是
    `seq/kind/actor/op/target/after/before/reason/call_id/ts`（**`value` 已取消**、
    新增 `op` 与 `ts`；`op` 对 state 事件恒为 `"set"`）；`/activity` 仍是 6 键
    `kind/actor/target/value/seq/call_id`（**它的 `value` 没改名**，那是另一条 wire）。
    `before` 由全史 fold 重建；`/config` 必须 `has_schema=False` + 快照键全进 `orphans`
    （**不许造树**）。
    """
    from ml_mecha.monitor_http import MonitorEndpoint
    from ml_mecha.runtime import MONITOR_PORT_FILE, read_port

    with tempfile.TemporaryDirectory(prefix="ml-delivery-mon-") as td:
        root = Path(td)
        pf = root / MONITOR_PORT_FILE
        with _authority(root, open_gate=True) as (ml, host):
            monitor = MonitorEndpoint(ml, mcp_port=host.port, port_file=pf)
            try:
                mon_port = monitor.start()
                assert mon_port > 0
                assert read_port(pf) == mon_port, "必须写 .ml-monitor-port 供发现"
                base = f"http://127.0.0.1:{mon_port}"

                # 先经 MCP 真跑一次，让 /history /config /summary 有内容
                _call_tools(host.url, [
                    ("prepare_dataset", {"source": dict(_SYNTHETIC)}),
                    ("run_method", {"method": "logistic"})])

                code, headers, body = _http_get(base + "/status")
                assert code == 200, body
                assert headers.get("Access-Control-Allow-Origin") == "*"
                status = json.loads(body)
                assert status["ok"] is True and status["mode"] == "ai", status
                assert status["mcp_port"] == host.port, status
                assert status["monitor_port"] == mon_port, status

                code, _, body = _http_get(base + "/activity?since_seq=0")
                activity = json.loads(body)
                assert code == 200 and activity["mode"] == "ai", body[:200]
                assert activity["events"], "跑过运行就该有事件"
                # EL 的 activity 形状：**无** before/after/reason
                assert set(activity["events"][0]) == {
                    "kind", "actor", "target", "value", "seq", "call_id"}, \
                    activity["events"][0]

                code, _, body = _http_get(base + "/history?since_seq=0")
                hist = json.loads(body)
                assert code == 200 and hist["mode"] == "ai"
                assert hist["events"], "跑过运行就该有事件"
                for ev in hist["events"]:
                    # ⚠ 新形状（mecha `926b90d`）：`value` 取消、改 `after`；新增 `op`/`ts`
                    assert set(ev) == {"seq", "kind", "actor", "op", "target", "after",
                                       "before", "reason", "call_id", "ts"}, ev
                    assert ev["kind"] == "set", ev
                    # `op` 与 `target` 的分工（新形状的要害）：
                    #   **只有域状态事件带 `target`**（且 `op` 恒为 `"set"`）；
                    #   其余（审计 `command.<name>`、本仓的域事件 `ml.run`）**`target` 为空**
                    #   ——它们的"名字"在 `op` 里（这正是类 C：按名字筛要用 `op`）。
                    assert ev["op"], ev            # op 恒非空（新形状的必填）
                    assert (ev["op"] == "set") == bool(ev["target"]), ev
                    if ev["op"] == "set":
                        assert ev["target"].startswith("current."), ev
                assert hist["events"][0]["before"] is None, \
                    f"首写 before 必须是 None：{hist['events'][0]}"
                assert [e for e in hist["events"] if e["before"] is not None], \
                    "同键第二次写必须有重建出来的 before"

                code, _, body = _http_get(base + "/config")
                cfg = json.loads(body)
                assert code == 200, body[:200]
                assert cfg["has_schema"] is False, cfg
                assert cfg["groups"] == {}, cfg
                assert cfg["n_keys"] == 0, cfg
                names = {r["name"] for r in cfg["orphans"]}
                assert "current.dataset_id" in names, names
                assert cfg["n_snapshot_keys"] == len(cfg["orphans"]), cfg
                row = next(r for r in cfg["orphans"]
                           if r["name"] == "current.dataset_id")
                assert set(row) == {"name", "value", "set"}, row
                assert row["set"]["actor"] == "ml-ai" and row["set"]["seq"] > 0, row

                # ML 附加面：/summary（可对账概括）与 /runs（磁盘存档）
                code, _, body = _http_get(base + "/summary")
                summary = json.loads(body)
                assert code == 200 and summary["ok"] is True, body[:200]
                assert "disputed" in summary and "summary" in summary, summary
                assert summary["summary"]["run_count"] >= 1, summary["summary"]

                code, _, body = _http_get(base + "/runs?limit=5")
                runs = json.loads(body)
                assert code == 200 and runs["source"] == "runs/", runs
                # 行取 `rows`（共享面板的附加页数据契约；迁移前叫 `runs`）
                assert isinstance(runs["rows"], list), runs

                # 只读：未知路径 404、POST 一律 404
                code, _, body = _http_get(base + "/nope")
                assert code == 404 and json.loads(body) == {
                    "ok": False, "error": "unknown endpoint"}, body
                code, _, body = _http_get(base + "/status", method="POST")
                assert code == 404 and json.loads(body) == {
                    "ok": False, "error": "read-only monitor endpoint"}, body

                # 跨源预检：204 + CORS
                code, headers, _ = _http_get(base + "/history", method="OPTIONS")
                assert code == 204, code
                assert headers.get("Access-Control-Allow-Origin") == "*"
                assert headers.get("Access-Control-Allow-Methods") == "GET, OPTIONS"
            finally:
                monitor.stop()
                ml.close()
            assert not pf.exists(), "stop 后监控端口文件必须删掉"


def test_monitor_endpoint_rejects_stale_port_file():
    """监控端点的端口文件走同一清理规则（陈旧文件必须被替换）。"""
    from ml_mecha.monitor_http import MonitorEndpoint
    from ml_mecha.runtime import read_port

    with tempfile.TemporaryDirectory(prefix="ml-delivery-mon-stale-") as td:
        root = Path(td)
        pf = root / ".ml-monitor-port"
        pf.write_text("1", encoding="utf-8")
        ml = assemble_ml_mecha(root=root, seed=13)
        monitor = MonitorEndpoint(ml, port_file=pf)
        try:
            port = monitor.start()
            assert port > 1 and read_port(pf) == port, (port, pf.read_text())
        finally:
            monitor.stop()
            ml.close()


def test_monitor_runs_route_fails_loud_instead_of_faking_empty():
    """D6（迁移裁决）：``/runs`` 取数失败 ⇒ **500 + 结构化错误体**，不得伪造空表。

    迁移前本路由是 ``try: … except Exception: return []``——"读不到"与"确实没有数据"
    变得不可区分，正是两家判据纪律反复在防的假绿。现在取数异常由框架归一化成
    ``500 {ok:false, error:{message, info:{kind,…}}}``，面板据此显示可读错误。

    两件事必须**同时**成立才算这条判据合格（只测 500 会漏掉"合法空表也被当错误"）：
    ``runs/`` 不存在 ⇒ 200 + 空表（``list_meta`` 的合法语义）；取数抛错 ⇒ 500。
    """
    from unittest import mock

    from ml_mecha.monitor_http import MonitorEndpoint
    from ml_mecha.runtime import MONITOR_PORT_FILE

    with tempfile.TemporaryDirectory(prefix="ml-delivery-runs-") as td:
        root = Path(td)
        pf = root / MONITOR_PORT_FILE
        with _authority(root, open_gate=True) as (ml, host):
            monitor = MonitorEndpoint(ml, mcp_port=host.port, port_file=pf)
            try:
                monitor.start()
                base = f"http://127.0.0.1:{monitor.port}"

                # ① 合法空表：把存档根指到不存在的目录 ⇒ 200 + 空表（不是错误）
                #    （不能靠"临时数据根"来制造空表：runs/ 是**进程级**路径，
                #     由 ML_TOOLBOX_RUNS/项目根决定，与权威的数据根无关。）
                from ml_toolbox.core import persistence
                with mock.patch.object(persistence, "RUNS_DIR",
                                       str(root / "no-such-runs")):
                    code, _, body = _http_get(base + "/runs?limit=5")
                    runs = json.loads(body)
                    assert code == 200 and runs["rows"] == [], body[:200]
                    assert runs["source"] == "runs/", runs

                # ② 取数失败：响亮失败（500 + 结构化 kind），绝不降级成空表
                with mock.patch.object(persistence, "list_records",
                                       side_effect=OSError("磁盘读不到")):
                    code, _, body = _http_get(base + "/runs")
                assert code == 500, (code, body[:300])
                payload = json.loads(body)
                assert payload["ok"] is False, payload
                assert payload["error"]["info"]["kind"] == "monitor_internal_error", \
                    payload["error"]
                assert "磁盘读不到" in payload["error"]["message"], payload["error"]
                assert "runs" not in payload, \
                    "错误回执里不得再出现「看起来正常」的空表键"
            finally:
                monitor.stop()
                ml.close()


def test_monitor_extra_routes_are_registered_and_core_routes_untouched():
    """两条 ML 附加路由经框架 ``add_route`` 注册，且不覆盖四个核心路由。"""
    from ml_mecha.monitor_http import MonitorEndpoint

    with tempfile.TemporaryDirectory(prefix="ml-delivery-routes-") as td:
        root = Path(td)
        with _authority(root, open_gate=True) as (ml, host):
            monitor = MonitorEndpoint(ml, mcp_port=host.port,
                                      port_file=root / ".ml-monitor-port")
            try:
                assert monitor.routes() == ["/runs", "/summary"], monitor.routes()
                # 重名 fail loud（核心路由不可覆盖；两条附加路由不可重注册）
                for path in ("/status", "/summary"):
                    try:
                        monitor._endpoint.add_route(path, lambda _q: {})
                    except Exception as exc:                     # noqa: BLE001
                        assert "duplicate_route" in str(exc) or "重名" in str(exc) \
                            or "已注册" in str(exc), exc
                    else:                                        # pragma: no cover
                        raise AssertionError(f"{path} 重名注册竟然成功了")
            finally:
                monitor.stop()
                ml.close()


# ------------------------------------------------------- P2：dsh overlay 生成
def test_generated_dsh_patch_composes_with_real_dsh():
    """生成的 overlay 必须真能被 dsh 组合（`- insert:` + 端口 + 超时）。

    这条跑**真的 dsh CLI**（`--dump-config` 只组合配置、不 mount 服务），因为
    patch 形态错误恰恰是"静默跳过"（裸 id 被判目标不存在）——只看我们自己生成的
    文本会漏掉这类假绿。dsh 不在 PATH 时如实说明并跳过。
    """
    import shutil as _shutil

    import launcher

    dsh_exe = _shutil.which("dsh")
    if not dsh_exe:
        print("    （跳过：PATH 里没有 dsh——patch 组合未验证）")
        return

    with tempfile.TemporaryDirectory(prefix="ml-delivery-patch-") as td:
        patch = launcher.write_dsh_patch(Path(td) / "dsh-ml-mcp.patch.yml", 59999)
        text = patch.read_text(encoding="utf-8")
        assert "- insert:" in text, "patch 条目必须用 `- insert:` 包住"
        for token in ("mcp-mltoolbox", "serverName: mltoolbox",
                      "transport: streamable-http",
                      "url: http://127.0.0.1:59999/mcp",
                      "failOnStartupError: true",
                      f"toolCallTimeoutMs: {launcher.TOOL_CALL_TIMEOUT_MS}"):
            assert token in text, token

        proc = subprocess.run(
            [dsh_exe, "--profile", "web", "--patch", str(patch), "--dump-config"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=180)
        assert proc.returncode == 0, (proc.returncode, (proc.stdout or "")[-800:],
                                      (proc.stderr or "")[-800:])
        dumped = proc.stdout or ""
        assert "mcp-mltoolbox" in dumped, "组合树里没有我们的条目（被跳过了？）"
        assert "serverName: mltoolbox" in dumped, dumped[-500:]
        assert "url: http://127.0.0.1:59999/mcp" in dumped, dumped[-500:]
        assert f"toolCallTimeoutMs: {launcher.TOOL_CALL_TIMEOUT_MS}" in dumped


def test_dsh_panel_source_has_no_el_leftovers_and_no_hardcoded_port():
    """静态：ML 的 `dsh/` 不得残留 EL 专属字样，也不得有硬编码监控端口。

    `MONITOR_URL` 是 EL 的硬编码常量（写死在两端 ⇒ 漂移即"面板空白但没人报错"）。
    ML 的地址必须走 `resolveMonitorBase()`（host 半读运行期描述符）。

    判据只看**代码**（先剥掉注释/文档串）：那些"EL 当时怎么写的、为什么不能照抄"
    的说明是有价值的文档，不该因为出现 `8767` 三个字就被判红。
    """
    import re

    dsh_dir = REPO / "dsh"
    # ⚠ **扫描范围排除 `src/dsh-panel/`**：那是**逐字复制的共享资产**（判据
    # `dsh/test/panel-assets.test.ts` 用常量指纹守着它），里面的命名/历史说明是上游的，
    # 本仓**不许就地改**（改了指纹判据就红）。本条判据的对象是**本仓自己的代码**。
    src_files = [p for p in (dsh_dir / "src").rglob("*")
                 if p.suffix in (".ts", ".tsx")
                 and "dsh-panel" not in p.parts]
    assert src_files, "dsh/src 不存在或为空——面板源码丢了？"

    code = "\n".join(_strip_js_comments(p.read_text(encoding="utf-8"))
                     for p in src_files)
    for token in ("re0", "机甲", "Rb-87", "8767", "mecha_v2", "energy-level",
                  "Energy Level", "MONITOR_URL"):
        assert token not in code, f"dsh 源码（去注释后）仍含 EL 专属字样 {token!r}"

    client_code = _strip_js_comments(
        (dsh_dir / "src" / "client" / "index.ts").read_text(encoding="utf-8"))
    host_code = _strip_js_comments(
        (dsh_dir / "src" / "index.ts").read_text(encoding="utf-8"))
    # 地址**单级、绝不回落**（迁移后由共享资产的 monitor-url.ts 实现）：本仓自己的代码里
    # 不许再出现 loopback 端口字面量，也不许自己拼地址。
    for name, src in (("client", client_code), ("host", host_code)):
        assert not re.search(r"127\.0\.0\.1:\d+", src), \
            f"{name} 半里出现了硬编码的 loopback 端口"
    assert "resolveMonitorBase" not in client_code, \
        "客户端不该自己解析地址（取址归共享资产的 fetchMonitorBase）"
    assert "commands.execute" not in client_code, \
        "客户端不得经 commands.execute 取地址（会污染会话 + 形状脆弱）"
    assert "remote.commands" not in client_code, "客户端不该再依赖 remote.commands"
    # host 半必须用**共享的**地址实现（参数块给路由与端口文件名）
    assert "makeMonitorUrlHandler" in host_code, "host 半没有用共享的地址路由实现"
    assert "monitor_url" not in host_code and ".ml-monitor-port" not in host_code, \
        "host 半不该自己写端口文件名（单一来源是参数块 PANEL_CONFIG.PORT_FILE）"

    # 页签：标准两页的文案默认在资产里；ML 的两条附加页文案在本仓 client 数据层里
    panel_data = (dsh_dir / "src" / "dsh-panel" / "panel-data.ts").read_text(
        encoding="utf-8")
    for label in ("飞行记录仪", "配置态"):
        assert label in panel_data, f"共享面板缺少标准页签 {label!r}"
    # ⚠ 附加页声明住在 `client/extraPages.ts`（无 react 的数据层，可 node 直测）
    extra = (dsh_dir / "src" / "client" / "extraPages.ts").read_text(
        encoding="utf-8")
    for label in ("运行记录", "磁盘存档"):
        assert label in extra, f"本仓 client 未声明附加页签 {label!r}"


def test_panel_runs_page_shows_readable_error_instead_of_blank():
    """D6 的面板侧：``/runs`` 取数失败要显示**可读错误**，不能渲染成空表。

    ⚠ **迁移后本条判据的对象变了**（强度不变）：这条性质现在由**共享资产**实现——
    `panel-view.ts::extraPageHtml` 的失败分支画 `unreadableText` + 真因，而"每条分支都
    返回非空可读 HTML"由资产自己的 `panel-view.test.ts` 逐档断言（随 `npm test` 执行）。
    本仓侧要钉的是**声明对不对**：两条附加页各自声明了 `emptyText` 与 `unreadableText`，
    且**两者必须不同**（"确实没有数据"与"读不到"是两件事）。
    """
    client = (REPO / "dsh" / "src" / "client" / "extraPages.ts").read_text(
        encoding="utf-8")
    view = (REPO / "dsh" / "src" / "dsh-panel" / "panel-view.ts").read_text(
        encoding="utf-8")
    # 资产把"读不到"渲染成 `unreadableText` + 真因（不是空表）
    assert "spec.unreadableText" in view and "result.failure.detail" in view, \
        "资产的附加页失败分支没有画可读错误（读不到 ≠ 空表）"
    # 我的两条声明都必须给"空"与"读不到"两套文案
    for key in ("emptyText", "unreadableText"):
        assert client.count(key + ":") >= 2, f"两条附加页都该声明 {key}"
    assert "不是空的" in client, "错误文案没有与「确实没有数据」区分开"
    assert "磁盘存档读取失败" in client, "附加页缺少可读的失败文案"
    # 资产自测覆盖"每条分支都非空"（这条性质在 npm test 里跑；此处确认判据存在，不是被删了）
    view_test = (REPO / "dsh" / "src" / "dsh-panel" / "panel-view.test.ts").read_text(
        encoding="utf-8")
    assert "每条分支都返回**非空**可读 HTML" in view_test, \
        "资产自测里少了「每条分支都非空」那条（性质没被守）"
    # 附加页失败**不牵连**标准两页：取数把它们单独返回（不塞进 panelState 的 extra）
    data = (REPO / "dsh" / "src" / "dsh-panel" / "panel-data.ts").read_text(
        encoding="utf-8")
    assert "extraRoutes: ExtraRoute[] = []" in data and "extras[r.id] =" in data, \
        "loadPanel 没有把附加页取数单独返回（失败会牵连标准两页）"


# ------------------------------------------------------- P3：面板的自动化代理验收
def _node_available():
    import shutil as _shutil
    return _shutil.which("node")


def _run_node(script: str, *args, timeout: float = 120.0):
    """把一段 JS 写到临时文件跑掉，返回 ``(rc, stdout, stderr)``。

    用文件而不是 ``node -e``：脚本里有多行模板与顶层 await，命令行转义太脆。
    """
    node = _node_available()
    assert node, "PATH 里没有 node"
    with tempfile.TemporaryDirectory(prefix="ml-delivery-node-") as td:
        path = Path(td) / "probe.mjs"
        path.write_text(script, encoding="utf-8")
        proc = subprocess.run([node, str(path), *[str(a) for a in args]],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout,
                              env={**os.environ,
                                   "MLTB_BUNDLE": str(REPO / "dsh" / "lib")})
        return proc.returncode, proc.stdout or "", proc.stderr or ""


def _payload(stdout: str, marker: str = "RESULT "):
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise AssertionError(f"JS 没有输出 {marker!r} 行；stdout=\n{stdout}")


_HOST_HALF_JS = r"""
import { pathToFileURL } from 'node:url'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const bundle = join(process.env.MLTB_BUNDLE, 'index.mjs')
const mod = await import(pathToFileURL(bundle).href)
const root = process.argv[2]
const out = { name: mod.name, inject: mod.inject, hasApply: typeof mod.apply === 'function' }

// --- apply：捕获 webServer.register 的路由 + disposer ---
let route = null
let disposed = false
let disposer = null
const ctx = {
  effect: async (fn) => { disposer = fn(); return disposer },
  webServer: { register: (r) => { route = r; return () => { disposed = true } } },
  logger: { info() {}, warn() {}, error() {} },
}
await mod.apply(ctx)
out.routeKind = route && route.kind
out.routePath = route && route.path
out.handlerIsFn = !!(route && typeof route.handler === 'function')

// --- stub req/res：handler 拥有完整响应生命周期 ---
function callHandler() {
  const chunks = []
  const res = {
    statusCode: 0,
    headers: {},
    setHeader(k, v) { this.headers[k] = v },
    end(body) { if (body) chunks.push(Buffer.from(body)) },
  }
  route.handler({ method: 'GET', url: route.path, headers: {} }, res)
  return { status: res.statusCode, headers: res.headers,
           body: Buffer.concat(chunks).toString('utf8') }
}

// (a) 描述符就位 → 200 + {base}
process.env.MLTB_ML_ROOT = root
const okRes = callHandler()
out.okStatus = okRes.status
out.okContentType = okRes.headers['Content-Type']
out.okCacheControl = okRes.headers['Cache-Control']
try { out.okBase = JSON.parse(okRes.body).base } catch { out.okBase = null }

// (b) 描述符缺失（把根指到空目录）→ 非 200 + 可读错误 JSON
const empty = mkdtempSync(join(tmpdir(), 'mltb-empty-'))
process.env.MLTB_ML_ROOT = empty
const badRes = callHandler()
out.badStatus = badRes.status
try {
  const parsed = JSON.parse(badRes.body)
  out.badHasError = typeof parsed.error === 'string' && parsed.error.length > 0
  out.badError = parsed.error
  out.badHasBase = 'base' in parsed
} catch { out.badHasError = false }

// (c) 环境变量与 cwd 都读不到 → 直接函数必须抛错（不许回落硬编码端口）
delete process.env.MLTB_ML_ROOT
process.chdir(empty)
try {
  out.emptyDirUrl = mod.resolveMonitorBase()
} catch { out.emptyDirThrew = true }

// (d) disposer 真的撤掉路由
out.disposedBefore = disposed
if (typeof disposer === 'function') disposer()
out.disposedAfter = disposed

console.log('RESULT ' + JSON.stringify(out))
"""

_CLIENT_HALF_JS = r"""
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

const registry = {}
globalThis.window = { __ModuleLoader__: { load: (m) => { registry.mod = m } } }
const code = readFileSync(join(process.env.MLTB_BUNDLE, 'client.js'), 'utf8')
new Function('window', code)(globalThis.window)      // classic-script 形态
const mod = registry.mod
if (!mod) { console.log('RESULT ' + JSON.stringify({ loaded: false })); process.exit(0) }

const react = {
  createElement: (t, p, ...c) => ({ t, p, c }),
  useState: (v) => [v, () => {}],
  useEffect: () => {},
  useRef: (v) => ({ current: v }),
  useCallback: (f) => f,
  Fragment: 'fragment',
}
const jsx = { jsx: (t, p) => ({ t, p }), jsxs: (t, p) => ({ t, p }) }
const fakeRequire = (id) =>
  id === 'react' ? react : (id === 'react/jsx-runtime' ? jsx : {})

const exported = mod.factory(fakeRequire)
const regs = []
let bodyDesc = null
const ctx = {
  effect: async (fn) => fn(),
  slots: {
    inject: (_slot, cb) => cb(),
    register: (desc, comp) => {
      regs.push({ slot: desc.name, id: desc.id ?? null, key: desc.key ?? null,
                  comp: comp && comp.name,
                  hasInject: typeof desc.inject === 'function' })
      if (desc.name === 'sidebar.right.pane.tab') bodyDesc = desc
      return () => {}
    },
  },
  layout: { openRightbar: () => {} },
  sidebarRight: { openTab: () => {} },
  sidebarRightTabs: { register: (d) => { regs.push({ tab: d.id, kind: d.kind }); return () => {} } },
  logger: { info() {}, warn() {}, error() {} },
}
await exported.apply(ctx)

const out = { loaded: true, name: exported.name, inject: exported.inject, regs,
              hasApply: typeof exported.apply === 'function' }
out.injectedKeys = Object.keys(bodyDesc.inject() || {})
// 附加页声明必须真的进到组件 props 里（R17：声明了就得被读——不是"看起来能注入"）
out.injectedExtraIds = (bodyDesc.inject().extraPages || []).map((p) => p.id)
console.log('RESULT ' + JSON.stringify(out))
"""


def test_panel_host_half_serves_monitor_url_route():
    """HOST 半（真产物）：注册**同源只读路由**，200 给 `{base}`、读不到就非 200。

    这是真机 bug 的回归判据：地址不再经 command 返回（那个形状
    `value.result.text` 曾让我们读成 `value.text`，面板永远"监控端点未知"）。
    """
    if not _node_available():
        print("    （跳过：PATH 里没有 node）")
        return
    if not (REPO / "dsh" / "lib" / "index.mjs").exists():
        print("    （跳过：dsh/lib 未构建——cd dsh && npm install && npm run bundle）")
        return
    from ml_mecha.monitor_http import MonitorEndpoint
    from ml_mecha.runtime import MONITOR_PORT_FILE, write_runtime

    with tempfile.TemporaryDirectory(prefix="ml-delivery-host-") as td:
        root = Path(td)
        with _authority(root, open_gate=True) as (ml, host):
            monitor = MonitorEndpoint(ml, mcp_port=host.port,
                                      port_file=root / MONITOR_PORT_FILE)
            try:
                mon_port = monitor.start()
                write_runtime(root, mode="ai", mcp_port=host.port,
                              monitor_port=mon_port)
                rc, out, err = _run_node(_HOST_HALF_JS, root)
                assert rc == 0, (rc, out[-1200:], err[-1200:])
                got = _payload(out)

                assert got["name"] == "ml-toolbox-monitor", got
                # 依赖换成 webserver（不再有 commands）
                assert got["inject"] == ["webServer"], got
                assert got["hasApply"] is True, got
                # 路由形状必须精确（dsh 契约：绝对路径、无尾斜杠、exact）
                assert got["routeKind"] == "exact", got
                assert got["routePath"] == "/ml-toolbox/monitor-url", got
                assert got["handlerIsFn"] is True, got

                # (a) 描述符就位 → 200 + {base} == 权威真写的端口
                assert got["okStatus"] == 200, got
                assert got["okBase"] == f"http://127.0.0.1:{mon_port}", got
                assert "application/json" in (got["okContentType"] or ""), got
                assert got["okCacheControl"] == "no-store", got

                # (b) 描述符缺失 → 非 200 + 可读错误（**不许回落硬编码端口**）
                assert got["badStatus"] == 503, got
                assert got["badHasError"] is True, got
                assert got["badHasBase"] is False, got
                assert "监控端点未知" in got["badError"], got
                assert got["emptyDirThrew"] is True, got

                # (d) disposer 撤路由
                assert got["disposedBefore"] is False, got
                assert got["disposedAfter"] is True, got
            finally:
                monitor.stop()
                ml.close()


def test_panel_client_half_uses_same_origin_fetch_not_command():
    """CLIENT 半（真产物）：地址走**共享资产**的同源 `fetch`；旧 command 形状必须失败。

    ⚠ **迁移后本条判据的对象变了**（强度不变）：
    * 注册面仍断言**本仓 client**（按钮 / tab / 注入面 / 附加页声明真的进到 props）；
    * 取址行为改断言**共享资产**的管线源码 + **真产物 bundle**——`fetchMonitorBase`
      不再是本仓 client 的导出（它是资产内部实现），而"喂真形状得 base / 喂旧 command
      形状必须失败 / 空 base 必须失败"这三条负例由资产的 `monitor-client.test.ts`
      逐条覆盖（随 `npm test` 执行）。
    """
    if not _node_available():
        print("    （跳过：PATH 里没有 node）")
        return
    if not (REPO / "dsh" / "lib" / "client.js").exists():
        print("    （跳过：dsh/lib 未构建）")
        return
    rc, out, err = _run_node(_CLIENT_HALF_JS)
    assert rc == 0, (rc, out[-1500:], err[-1500:])
    got = _payload(out)
    assert got.get("loaded") is True, got
    assert got["name"] == "ml-toolbox-monitor", got

    # 依赖面：不再需要 remote.commands（那正是污染会话与形状脆弱的来源）
    assert "slots" in got["inject"], got
    assert "remote" not in got["inject"], got
    assert "remote.commands" not in got["inject"], got

    regs = got["regs"]
    header = [r for r in regs if r.get("slot") == "conversation.session.header.actions"]
    assert header and header[0]["comp"] == "CockpitButton", regs
    assert header[0]["id"] == "ml-toolbox-monitor", header
    tab = next(r for r in regs if r.get("tab"))
    assert tab["tab"] == "ml-toolbox-monitor", regs
    body = [r for r in regs if r.get("slot") == "sidebar.right.pane.tab"]
    # tab body 现在**就是共享面板本体**（本仓不再自绘）
    assert body and body[0]["comp"] == "MonitorTabBody", regs
    assert body[0]["key"] == "ml-toolbox-monitor", body

    # R17：声明了就必须被读——注入面只有 extraPages，且它**真的**是那两条声明
    assert got["injectedKeys"] == ["extraPages"], got
    assert got["injectedExtraIds"] == ["summary", "runs"], got

    # 取址走共享资产的管线：同源 fetch + `cache: no-store` + 失败落 `address` 档
    pipeline = (REPO / "dsh" / "src" / "dsh-panel" / "monitor-client.ts").read_text(
        encoding="utf-8")
    assert "doFetch(routePath, { cache: 'no-store' })" in pipeline, \
        "共享管线没有同源 fetch 地址路由"
    assert "tier: 'address'" in pipeline, "取址失败没有落 address 档（会被误报成离线）"
    assert "isOriginLike" in pipeline, "缺少地址形状自检（垃圾地址会被当成 base）"
    # 真产物 bundle：路由字面量来自参数块（单一来源），且**不含**任何命令调用
    bundle = (REPO / "dsh" / "lib" / "client.js").read_text(encoding="utf-8")
    assert "ml-toolbox/monitor-url" in bundle, \
        "bundle 里没有地址路由字面量 ⇒ 参数块没被打进产物（bundle 陈旧？）"
    assert "commands.execute" not in bundle and "remote.commands" not in bundle, \
        "bundle 里出现了命令调用（地址必须走同源 fetch）"


def test_dsh_mounts_profile_with_all_overlays_and_stays_alive():
    """真实两进程 mount：权威 + dsh（三份 overlay）——R19 的最强代理。

    断言：dsh 起来后**进程仍活着**、在挑到的端口上**真的在听 HTTP**、输出里
    **没有插件加载失败 / 语法错 / 未处理拒绝**；且**不碰官方 :3080**。
    收尾用进程树杀（``launcher._kill_tree``）——只 terminate 会留下持有端口的
    node 孤儿（P3 实测踩过，见 launcher 里 ``_kill_tree`` 的注释）。
    """
    import shutil as _shutil
    import socket

    import launcher

    dsh_exe = _shutil.which("dsh")
    if not dsh_exe:
        print("    （跳过：PATH 里没有 dsh——本项未能自动化，见 dsh/README.md 手动清单）")
        return

    with tempfile.TemporaryDirectory(prefix="ml-delivery-mount-") as td:
        root = Path(td)
        authority = _spawn_authority_proc(root)
        dsh_proc = None
        try:
            ok, mcp_port, reason = launcher.await_authority(
                authority, root / ".mcp-port", timeout=90.0)
            assert ok is True, reason

            patches = launcher.dsh_overlays(mcp_port)
            names = [Path(p).name for p in patches]
            # 三份都要在：插件（面板）+ 项目隔离（让 --port 生效）+ 工具接入
            assert "cordis.patch.yml" in names, names
            assert "cordis.project.patch.yml" in names, names
            assert "dsh-ml-mcp.patch.yml" in names, names

            dsh_port = launcher.pick_dsh_port()
            assert dsh_port != 3080, "绝不能占官方 dsh 的端口"
            cmd = [dsh_exe, "--profile", "web"]
            for patch in patches:
                cmd += ["--patch", patch]
            cmd += ["--port", str(dsh_port), "--no-open"]
            dsh_proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace",
                                        env={**os.environ,
                                             "MLTB_ML_ROOT": str(root)})

            deadline = time.time() + 120
            serving = False
            while time.time() < deadline:
                if dsh_proc.poll() is not None:
                    break
                try:
                    with socket.create_connection(("127.0.0.1", dsh_port),
                                                  timeout=1.0):
                        serving = True
                        break
                except OSError:
                    time.sleep(0.5)
            assert dsh_proc.poll() is None, f"dsh 提前退出 rc={dsh_proc.returncode}"
            assert serving, f"dsh 起来了但 :{dsh_port} 没有在听"

            time.sleep(3)                      # 给插件加载/工具同步留出时间
            assert dsh_proc.poll() is None, "dsh 在启动后几秒内退出了"

            # ---- 端到端真证据：dsh **同源路由**返回的 base 必须指向权威真写的端口 ----
            from ml_mecha.runtime import MONITOR_PORT_FILE, read_port
            real_monitor_port = read_port(root / MONITOR_PORT_FILE)
            assert real_monitor_port, "权威没写 .ml-monitor-port"
            code, _, body = _http_get(
                f"http://127.0.0.1:{dsh_port}/ml-toolbox/monitor-url")
            assert code == 200, (code, body[:300])
            route_payload = json.loads(body)
            assert route_payload.get("base") == \
                f"http://127.0.0.1:{real_monitor_port}", route_payload
            # 顺着这个 base 真去读一次监控面（证明它可直接用，不只是字符串对）
            code, _, body = _http_get(f"{route_payload['base']}/status")
            assert code == 200, (code, body[:300])
            assert json.loads(body)["mode"] == "ai", body[:200]

            launcher._kill_tree(dsh_proc)
            try:
                out, _ = dsh_proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:                  # pragma: no cover
                launcher._kill_tree(dsh_proc)
                out, _ = dsh_proc.communicate(timeout=15)
            output = out or ""
            lowered = output.lower()
            for bad in ("failed to load", "cannot find module", "unhandled rejection",
                        "syntaxerror", "plugin load error"):
                assert bad not in lowered, f"dsh 输出里有 {bad!r}：\n{output[-1500:]}"
        finally:
            if dsh_proc is not None:
                launcher._kill_tree(dsh_proc)
                try:
                    dsh_proc.wait(timeout=20)
                except subprocess.TimeoutExpired:              # pragma: no cover
                    pass
            launcher._terminate_group([authority])
            launcher.await_exit([authority], timeout=20)


_TESTS = [
    test_required_from_signature_and_declaration_survives,
    test_mcp_wire_declares_no_output_schema,
    test_mcp_output_schema_judge_can_go_red,
    test_public_surface_all_names_exist,
    test_public_surface_judge_can_go_red,
    test_null_shim_drops_explicit_null_for_every_defaulted_param,
    test_null_shim_keeps_explicit_null_when_real_default_is_none,
    test_null_shim_rule_partitions_every_optional_param,
    test_null_shim_authority_is_declaration_only,
    test_null_shim_single_source_mutant_is_detected,
    test_mcp_host_exposes_every_declared_tool_over_http,
    test_readonly_tool_returns_real_value_over_http,
    test_write_tool_is_fail_closed_while_locked,
    test_open_gate_write_tools_run_and_call_id_is_attributed,
    test_port_file_written_on_start_and_removed_on_stop,
    test_stale_port_file_from_killed_run_is_replaced_on_start,
    test_batch_tool_preserves_partial_success,
    test_job_tools_submit_read_and_result_ref,
    test_cancel_run_rejects_uncancellable_job,
    test_dataset_roots_authorization_is_fail_closed,
    test_launcher_pick_dsh_port_skips_official_3080,
    test_launcher_await_authority_rejects_stale_port_file,
    test_monitor_shapes_match_el_cockpit_contract,
    test_monitor_endpoint_rejects_stale_port_file,
    test_monitor_runs_route_fails_loud_instead_of_faking_empty,
    test_monitor_extra_routes_are_registered_and_core_routes_untouched,
    test_generated_dsh_patch_composes_with_real_dsh,
    test_dsh_panel_source_has_no_el_leftovers_and_no_hardcoded_port,
    test_panel_runs_page_shows_readable_error_instead_of_blank,
    test_panel_host_half_serves_monitor_url_route,
    test_panel_client_half_uses_same_origin_fetch_not_command,
    test_dsh_mounts_profile_with_all_overlays_and_stays_alive,
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
