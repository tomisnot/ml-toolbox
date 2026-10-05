# -*- coding: utf-8 -*-
"""交付形态 P0 判据：AI 模式（headless 权威 + 同进程 MCP 服务）真的能被调通。

每条断言都指向"机制真的接上了"，不是形状猜测：

1. ``required`` 的来源是**两条路**：写工具取**命令声明**
   （``CommandSpec.parameters["required"]``，经 ``mcp_host`` 的必填来源），
   只读与 job 调度面取 ``execute`` 闭包签名（无默认值=必填）；
   且**声明**里的 type/description 覆写回了协议层（消"pydantic 派生把声明成果冲掉"）。
   ⚠ 两条路**可能不一致**（`run_custom` 曾经：闭包签名 `code=""` 是非必填、
   命令声明却是 `required=True`）—— 本判据正是钉它们一致的地方；
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
#: ⚠ 2026-10-05 长任务编排：等待原语**折进 `read_job`**（不新增工具），
#: 但"热改并发上限"是**新能力**、没有既有入口 ⇒ 加了 `set_job_limit`（19→20）。
TOOL_NAMES = ("cancel_run", "checkpoint_detail", "clear_current_state",
              "compare_methods", "describe_dataset",
              "describe_method", "describe_methods", "describe_network",
              "describe_networks", "describe_run", "describe_training",
              "eval_custom", "prepare_dataset", "read_job", "run_custom",
              "run_method", "run_method_batch", "run_queries", "set_job_limit",
              "submit_run")

#: 每个工具的必填项（写工具取**命令声明**，只读与 job 调度面取闭包签名）。
EXPECTED_REQUIRED = {
    "cancel_run": ["job_id"],
    "compare_methods": ["methods"],
    "describe_dataset": [],
    "describe_method": ["name"],
    "describe_methods": [],
    "describe_network": ["name"],
    "describe_networks": [],
    "describe_run": [],
    "describe_training": [],
    "prepare_dataset": ["source"],
    # ⚠ `job_id` **不是**必填（2026-10-05 长任务编排）：它与 `job_ids` 二选一
    #   （"看一个" vs "一次等多个/等任意一个"），而 JSON-schema 的 required
    #   表达不了"二选一"⇒ 两者都声明为非必填，由 `tools._job_handles` 校验
    #   （都不给报 `missing_job_id`、同时给报 `bad_job_selection`）。
    #   这与 `run_custom.code`/`code_path` 是**同一套**做法（见下）。
    "read_job": [],
    # ⭐ 2026-10-05：`set_job_limit` 的 `limit` 是**必填**（没有"默认上限"可说——
    #   默认来自环境变量/装配点，不在这里再写一份）。
    "set_job_limit": ["limit"],
    # ⚠ `code` **不是**必填（2026-10-04 修）：`code` 与 `code_path` 二选一，
    #   而 JSON-schema 的 required 表达不了"二选一"⇒ 两者都声明为非必填，
    #   由引擎在 `engine.py:1113-1126` 校验（都不给报 `missing_code`、
    #   同时给报 `bad_code_source`）。
    #   旧行为把 `code` 声明成必填，导致 `code_path`-only 的调用
    #   **在参数校验层就被拒**（实测 `1 validation error … code Field required`），
    #   而声明里的注释却写着"也可用 code_path（二选一）"——自相矛盾。
    "run_custom": [],
    # ⚠ `method` **不是**必填：`submit_run` 允许 `method` / `code` 二选一
    #   （给了 code 就走自定义网络）。必填项由工具自己判并给可读错误。
    "run_method": ["method"],
    "run_method_batch": ["methods"],
    "submit_run": [],
}

_SYNTHETIC = {"kind": "synthetic", "task": "classification", "n_samples": 120}

#: `run_custom` 探针用的最小网络代码（**真能跑**：单层线性就是合法的 Network）。
_MIN_NET_CODE = ("class Network:\n"
                 "    def build(self, n_in, n_out, *, nn):\n"
                 "        return nn.Linear(n_in, n_out)\n")

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
    "describe_training": ["best_by", "grad_layers", "include_series", "limit", "run_id"],
    "describe_network": ["limit"],
    "describe_networks": ["limit"],
    # ⚠ `pipeline` / `payload` 是**可选但无默认**（None = 用默认值）⇒ 不进这张表，
    #   但仍是可选参数（计入 `_EXPECTED_OPTIONAL_PARAMS`）。
    "prepare_dataset": ["diag", "name", "target", "target_kind", "time_col"],
    "read_job": ["detail", "job_id", "tail", "timeout_s", "wait", "wait_s"],
    # ⚠ 只列**声明了非 None default** 的（`overrides`/`seed`/`resource_guard`
    #   在 `run_custom` 上是"无默认"⇒ 不进这张表，但它们仍是**可选**参数）。
    # `device` 声明了 `""` 默认 ⇒ 进表（"未给"= 空串 = auto）；
    # `dry_run`（False）与 `code_path`（""）也进了这张表。
    # ⚠ `code` 现在**也有**默认（`""`）：为的是 `code_path` 能单独用（二选一）。
    # `shapes_verbosity`（"trace"）同样进表。
    "run_custom": ["code", "code_path", "dataset_id", "device", "diag", "dry_run",
                   "name", "persist", "shapes_verbosity", "warmup_steps"],
    "run_method": ["dataset_id", "device", "diag", "persist"],
    "run_method_batch": ["dataset_id"],
    # `method`/`code`/`name` 都声明了 `""` 默认 ⇒ 进这张表（"未给"= 空串）。
    "submit_run": ["code", "code_path", "dataset_id", "device", "diag",
                   "method", "name", "persist"],
    # ---- 2026-10-03 新增（AI 调用体验）----
    # `clear_current_state` 无参数（语义就是"全部清零"，不给"清一部分"的语义面）。
    "clear_current_state": [],
    # ⚠ `actions`/`pick` 都是**无默认**（None = 必给的 actions / 不投影）
    #   ⇒ 不进 defaulted 表，但仍是**可选**参数（`actions` 由闭包签名把关必给）。
    "run_queries": [],
    # ---- 2026-10-03 新增（NQS 路径）----
    # `checkpoint_detail` 的 `run_id` 有 `""` 默认（留空 = 没有目标，会明确报错）
    "checkpoint_detail": ["run_id"],
    # `eval_custom` 的 `which` 有 `"best"` 默认、`dataset_id` 有 `""` 默认；
    # `run_id`/`code` 是必填。
    "eval_custom": ["dataset_id", "which"],
    # ---- 2026-10-05 长任务编排（L2 调度器）----
    # `set_job_limit` 唯一的参数 `limit` 是**必填**（无默认）⇒ 本表记空列表。
    "set_job_limit": [],
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
    ("describe_training", {}, "run_id", "run-null-shim"),
    ("describe_training", {}, "include_series", True),
    ("describe_training", {}, "limit", 50),
    ("describe_training", {}, "grad_layers", "full"),
    # ---- R41（2026-10-05）：按任意 tag 挑最佳 step ----
    ("describe_training", {}, "best_by", "fid/sub1"),
    ("describe_networks", {}, "limit", 10),
    ("describe_network", {"name": "n"}, "limit", 10),
    ("prepare_dataset", {"source": dict(_SYNTHETIC)}, "name", "my-ds"),
    ("prepare_dataset", {"source": dict(_SYNTHETIC)}, "target", "target"),
    ("prepare_dataset", {"source": dict(_SYNTHETIC)}, "time_col", "t"),
    ("prepare_dataset", {"source": dict(_SYNTHETIC)}, "diag", True),
    # ---- NQS 路径新增 ----
    ("prepare_dataset", {"source": dict(_SYNTHETIC)}, "target_kind", "custom"),
    ("checkpoint_detail", {}, "run_id", "run-null-shim"),
    ("eval_custom", {"run_id": "run-null-shim",
                     "code": "def evaluate(model, ctx):\n    return {}"},
     "which", "last"),
    ("eval_custom", {"run_id": "run-null-shim",
                     "code": "def evaluate(model, ctx):\n    return {}"},
     "dataset_id", "ds-null-shim"),
    ("run_method", {"method": "logistic"}, "dataset_id", "ds-null-shim"),
    ("run_method", {"method": "logistic"}, "device", "cpu"),
    ("run_method", {"method": "logistic"}, "persist", True),
    ("run_method", {"method": "logistic"}, "diag", True),
    # ⚠ `run_custom` 只探**声明了默认值**的那几个：`overrides`/`seed`/`resource_guard`
    #   没有默认值（是可选但无默认）⇒ 进这张表会被前提守卫挡下（实测）。
    ("run_custom", {"code": _MIN_NET_CODE}, "dataset_id", "ds-null-shim"),
    ("run_custom", {"code": _MIN_NET_CODE}, "name", "my-net"),
    ("run_custom", {"code": _MIN_NET_CODE}, "diag", True),
    ("run_custom", {"code": _MIN_NET_CODE}, "persist", True),
    ("run_custom", {"code": _MIN_NET_CODE}, "device", "cpu"),
    ("run_custom", {"code": _MIN_NET_CODE}, "dry_run", True),
    ("run_custom", {"code": _MIN_NET_CODE}, "code_path", "net.py"),
    ("run_custom", {"code": _MIN_NET_CODE}, "shapes_verbosity", "summary"),
    ("run_custom", {"code": _MIN_NET_CODE}, "warmup_steps", 5),
    ("run_method_batch", {"methods": ["logistic"]}, "dataset_id", "ds-null-shim"),
    ("compare_methods", {"methods": ["logistic"]}, "dataset_id", "ds-null-shim"),
    ("read_job", {"job_id": "job-null-shim"}, "wait_s", 5.0),
    ("read_job", {"job_id": "job-null-shim"}, "detail", "full"),
    ("read_job", {"job_id": "job-null-shim"}, "tail", 3),
    # ---- 长任务编排（2026-10-05）：等待原语折进 read_job（不新增工具）----
    ("read_job", {"job_id": "job-null-shim"}, "wait", "terminal"),
    ("read_job", {"job_id": "job-null-shim"}, "timeout_s", 5.0),
    # `job_id` 变成可选（与 `job_ids` 二选一）⇒ 它的默认在同一张表的基准里
    # 不能再出现，否则探针的 base 与被探参数会撞在一起。
    ("read_job", {}, "job_id", "job-null-shim"),
    ("submit_run", {"method": "logistic"}, "dataset_id", "ds-null-shim"),
    ("submit_run", {"method": "logistic"}, "device", "cpu"),
    ("submit_run", {"method": "logistic"}, "persist", True),
    ("submit_run", {"method": "logistic"}, "diag", True),
    # ⚠ 不探 `submit_run.method`：它现在声明成默认 `""`（与 `code` 二选一），
    #   而本探针表的 `base` 里已经带了 `method="logistic"` ⇒ 两者会撞在一起
    #   （实测：断言 "method 不该出现" 失败）。它的"二选一"语义由
    #   `test_tool_params_stay_within_command_declaration` 与端到端判据覆盖。
    ("submit_run", {"method": "logistic"}, "code", _MIN_NET_CODE),
    ("submit_run", {"method": "logistic"}, "name", "my-net"),
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
    # ---- 长任务编排（2026-10-05）：`job_ids` 是**无默认**的可选参数
    #      （None = 用 `job_id`）⇒ 显式 null 必须原样保留（"不指定多个"）。
    ("read_job", {"job_id": "job-null-shim"}, "job_ids", ["job-null-shim"]),
    # ---- R40（2026-10-05）：多 run 曲线对齐的两个参数（**无默认**）
    ("describe_training", {}, "compare", ["run-a", "run-b"]),
    ("describe_training", {}, "tags", ["fid/sub1"]),
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
    "describe_training": {},
    "describe_networks": {},
    "describe_network": {"name": "n"},
    "prepare_dataset": {"source": dict(_SYNTHETIC)},
    "read_job": {"job_id": "job-null-shim"},
    "run_custom": {"code": _MIN_NET_CODE},
    "run_method": {"method": "logistic"},
    "run_method_batch": {"methods": ["logistic"]},
    "submit_run": {"method": "logistic"},
    # ---- 2026-10-03 新增（AI 调用体验）----
    # `clear_current_state` 无参数：最小必填集就是空集。
    "clear_current_state": {},
    # `run_queries` 的 `actions` 是必给；`pick` 可选。给一个**真能跑通**的最小动作
    # （`summary` 是内置动作、零参数）——判据只关心 null 垫片的传输行为，
    # 但用真动作能顺带证明"编排入口本身可用"。
    "run_queries": {"actions": [{"op": "summary"}]},
    # `set_job_limit`：最小必填就是一个合法名额（**真的会改上限**——本判据只关心
    # 传输行为，但顺手用一个合法值不制造额外副作用面）。
    "set_job_limit": {"limit": 4},
    # ---- 2026-10-03 新增（NQS 路径）----
    # `checkpoint_detail` 只读、`run_id` 有 `""` 默认 ⇒ 最小必填集是空。
    "checkpoint_detail": {},
    # `eval_custom` 必给 run_id + code。这里给一对**跑不出结果但有教学错误**的值即可
    # （这条判据只关心 null 垫片的传输行为，不关心那次评估成不成）。
    "eval_custom": {"run_id": "custom_network-0-0",
                    "code": "def evaluate(model, ctx):\n    return {}"},
}

#: 可选参数总数（= 全部声明参数 − 必填参数）；迁移后必须仍是这个数才说明
#: "每个可选参数都被判据看过一遍"（新增参数会在这里红，提醒补探针）。
#: 2026-10-03 `describe_training` 带来 3 个（run_id / include_series / limit）；
#: 同日补齐 `diag`/`persist` 的**工具面缺口**（`run_method.diag` 曾声明却调不通，
#: `submit_run` 缺 persist/diag ⇒ 长训练跑完人回 GUI 看不到）再 +3；
#: 同日 `run_custom`（7 个可选）+ `describe_networks`/`describe_network`（各 1~2 个）
#: —— 这是"AI 写网络代码 + 按名字复用"那条路。
#: 2026-10-05 长任务编排 +4：`read_job` 的 `wait`/`timeout_s`（阻塞到终态）、
#: `job_ids`（一次等多个/等任意一个）与 `job_id`（二选一 ⇒ 不再是必填）。
#: 2026-10-05 长任务编排 +4：`read_job` 的 `wait`/`timeout_s`（阻塞到终态）、
#: `job_ids`（一次等多个/等任意一个）与 `job_id`（二选一 ⇒ 不再是必填）。
#: 同日曲线分析（R40–R43）+3：`describe_training` 的 `best_by` / `compare` / `tags`。
_EXPECTED_OPTIONAL_PARAMS = 74


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


def test_tool_params_stay_within_command_declaration():
    """⭐ 工具面的参数必须落在**命令声明**里，且 `run_method.diag` 要真通。

    钉的是一个实测到的真缺陷（2026-10-03）：`run_method` 的 `parameters` 里
    **声明了** `diag`，但处理函数签名里**没有**它 ⇒ 调用方按声明传 `diag`
    会撞 "unexpected keyword argument" ⇒ **该参数在工具面上从来调不通**。
    这类"声明与实现漂移"用别的判据看不出来：`EXPECTED_DEFAULTED` 只看声明、
    null 垫片探针只看"显式 null 被丢掉"，**没有一条真把 `diag=True` 发下去**。

    两个方向一起钉：
    ① 工具参数 ⊆ 命令声明参数（超集会被 `validate_spec` 以"未声明的参数"拒）；
    ② 端到端真发一次 `run_method(diag=True, persist=True)`：必须成功，且
       `diag` 真的到达方法层（`nn_attr` 归因工件存在——那是 `diag` 的唯一产物）。
    """
    from ml_mecha.commands import COMMANDS

    with tempfile.TemporaryDirectory(prefix="ml-toolparam-") as td:
        ml = assemble_ml_mecha(root=td, seed=13)
        try:
            declared = {c.name: {p.name for p in c.parameters} for c in COMMANDS}
            #: 工具 → 它**可能**落到哪些命令（一套参数可以喂多条命令）。
            #: ⚠ `submit_run` 是两条：给 `method` 走 `run_method`、给 `code` 走
            #:   `run_custom`（2026-10-03 起，长训练的自定义网络才跑得起来）。
            #:   ⇒ 它的参数只要被**其中任一条**命令接受即可。
            routes = {"prepare_dataset": ("prepare_dataset",),
                      "run_method": ("run_method",),
                      "run_method_batch": ("run_method_batch",),
                      "compare_methods": ("compare_methods",),
                      "run_custom": ("run_custom",),
                      "submit_run": ("run_method", "run_custom")}
            for tool_name, commands in routes.items():
                tool = ml.tools.get(tool_name)
                assert tool is not None, tool_name
                got = set((tool.parameters or {}).keys())
                allowed = set().union(*(declared[c] for c in commands))
                extra = got - allowed
                assert not extra, (
                    f"{tool_name} 声明了 {'/'.join(commands)} 都不认识的参数 "
                    f"{sorted(extra)}（工具参数必须是命令声明参数的子集；"
                    f"要加就先改 commands.py）")

            # ---- ② 真发一次：`diag` 必须通，并且真的产出归因工件
            if not _torch_ready():
                print("    （torch 不可用：跳过 diag 端到端那一半）")
                return
            ml.switch_ai()          # 开闸：写命令需要它（出厂是 LOCKED）
            got = _tool_ok(ml, "run_method",
                           {"method": "torch_mlp", "dataset_id": _tool_ok(
                               ml, "prepare_dataset",
                               {"source": {"kind": "synthetic",
                                           "task": "classification",
                                           "n_samples": 90},
                                "name": "params", "target": "target"})["dataset_id"],
                            "diag": True, "persist": True,
                            "overrides": {"hidden": "8", "epochs": 5}})
            rid = got["run_id"]
            try:
                import shutil
                from pathlib import Path

                from ml_toolbox.core.persistence import RUNS_DIR, load_run_record
                record = load_run_record(rid)
                assert record is not None, "persist=True 却没落盘"
                assert "nn_attr" in record.result.artifacts, \
                    "diag=True 没有产出归因工件 ⇒ diag 没真的到达方法层"
            finally:
                import shutil
                from pathlib import Path

                from ml_toolbox.core.persistence import RUNS_DIR
                shutil.rmtree(Path(RUNS_DIR) / rid, ignore_errors=True)
        finally:
            ml.close()


def _torch_ready() -> bool:
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def _tool_ok(ml, name: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
    """经**工具面**调一次工具（走的就是 AI 的路径：自带通道绑定与参数默认值）。

    ⚠ 名字带 `_tool_` 是刻意的：本文件另有一个 `_run`（跑协程的那个），
    重名会把它所有调用方静默换成这个签名 —— 实测撞过一次，两条判据一起红。
    """
    result = ml.tools.execute(name, dict(args))
    assert result.get("is_error") is False, result
    return result["value"]


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


# ============================================================ 长任务编排（2026-10-05）
# 依据：`docs/建议-ML工具长任务编排与失败可读性.md` §2.1（等待原语）。
def test_read_job_wait_terminal_blocks_and_returns_terminal():
    """⭐ **等待原语**：**一次** `read_job` 就等到终态，不再靠调用方轮询。

    用户实测（建议 §2.1）：`wait_s` 上限 30 s ⇒ 为等一个 250 epoch 的训练反复轮询，
    最后**绕过工具**用本地 `pwsh` 睡 9.5 分钟。"绕过工具 = 缺能力最硬的证据"。
    本判据只调一次 `read_job` 就必须拿到终态；若实现退回"立刻返回"，它会红。
    """
    with tempfile.TemporaryDirectory(prefix="ml-wait-terminal-") as td:
        with _authority(td, open_gate=True) as (_ml, host):
            (p_err, p_txt), = _call_tools(host.url, [
                ("prepare_dataset", {"source": dict(_SYNTHETIC)})])
            assert p_err is False, p_txt
            dataset_id = json.loads(p_txt)["dataset_id"]

            err, txt = _call_tools(host.url, [
                ("submit_run", {"method": "logistic", "dataset_id": dataset_id})])[0]
            assert err is False, txt
            job = json.loads(txt)["job_id"]

            # ⭐ **只有这一次** read_job：它必须一直阻塞到终态
            err, txt = _call_tools(host.url, [
                ("read_job", {"job_id": job, "wait": "terminal",
                              "timeout_s": 120})])[0]
            assert err is False, txt
            got = json.loads(txt)
            assert got["state"] == "done", got
            assert got["timed_out"] is False, got
            assert got["wait"] == "terminal", got
            assert got["wait_cap_s"] >= 540, got          # 上限从 30 提到 540
            assert got["wait_note"] == "", got            # 没夹、没超时 ⇒ 不必解释
            assert [j["job_id"] for j in got["jobs"]] == [job], got["jobs"]
            assert got["result_ref"].get("run_ids"), got


def test_read_job_wait_timeout_is_a_clean_answer_not_an_error():
    """⭐ 等到超时**不是错误**：回"仍在跑" + 进度，调用方可以**循环调用**。

    用户明确要求（本轮裁决）："超时必须干净返回'仍在运行 + 当前进度摘要'，
    **不能报错**——我要能循环调用它。"⇒ 本条用宿主自己的 submit seam 造一个
    **受控地不结束**的 job，再对一个 1 s 预算调 `wait="terminal"`。
    """
    import threading

    with tempfile.TemporaryDirectory(prefix="ml-wait-timeout-") as td:
        with _authority(td, open_gate=True) as (ml, host):
            gate = threading.Event()
            sleepy = ml.jobs.submit(lambda _ctx: gate.wait(20), command="sleepy",
                                    cancel_supported=True)
            try:
                err, txt = _call_tools(host.url, [
                    ("read_job", {"job_id": sleepy.id, "wait": "terminal",
                                  "timeout_s": 1})])[0]
                # ① **不是错误**（is_error=False）
                assert err is False, txt
                got = json.loads(txt)
                # ② 如实说"还在跑"，且 timed_out 为真（这是等待的正常终点）
                assert got["timed_out"] is True, got
                assert got["state"] not in ("done", "failed", "cancelled"), got
                assert got["waited_s"] >= 1, got
                # ③ 必须**告诉调用方下一步**（能循环调用 / 或用 watcher 等更久）
                assert "watch" in got["wait_note"], got["wait_note"]
            finally:
                gate.set()


def test_read_job_wait_any_reports_whichever_finished_first():
    """⭐ `job_ids` + `wait="any"`：**谁先完先报谁**（不是"报第一个句柄"）。

    顺序刻意把"不会结束的"放**前面**：若实现只是取第一个句柄，这条会红。
    """
    import threading

    with tempfile.TemporaryDirectory(prefix="ml-wait-any-") as td:
        with _authority(td, open_gate=True) as (ml, host):
            (p_err, p_txt), = _call_tools(host.url, [
                ("prepare_dataset", {"source": dict(_SYNTHETIC)})])
            assert p_err is False, p_txt
            dataset_id = json.loads(p_txt)["dataset_id"]
            err, txt = _call_tools(host.url, [
                ("submit_run", {"method": "logistic", "dataset_id": dataset_id})])[0]
            assert err is False, txt
            fast = json.loads(txt)["job_id"]

            gate = threading.Event()
            sleepy = ml.jobs.submit(lambda _ctx: gate.wait(20), command="sleepy",
                                    cancel_supported=True)
            try:
                err, txt = _call_tools(host.url, [
                    ("read_job", {"job_ids": [sleepy.id, fast], "wait": "any",
                                  "timeout_s": 120})])[0]
                assert err is False, txt
                got = json.loads(txt)
                assert got["job_id"] == fast, got          # ← 先完的那个，不是第一个
                assert got["state"] == "done", got
                assert got["timed_out"] is False, got
                # 两个句柄的状态都要能看见（不是只看主角）
                states = {j["job_id"]: j["state"] for j in got["jobs"]}
                assert set(states) == {sleepy.id, fast}, states
                assert states[sleepy.id] not in ("done", "failed", "cancelled"), states
            finally:
                gate.set()


def test_read_job_wait_clamps_to_cap_and_says_so():
    """⭐ 超上限**不静默**：夹到上限并在 `wait_note` 里明说，且给出更长的替代。

    "静默夹住"是旧实现的行为（要 3600 得 30）——那让人以为"等够了"。
    """
    with tempfile.TemporaryDirectory(prefix="ml-wait-clamp-") as td:
        with _authority(td, open_gate=True) as (_ml, host):
            (p_err, p_txt), = _call_tools(host.url, [
                ("prepare_dataset", {"source": dict(_SYNTHETIC)})])
            assert p_err is False, p_txt
            dataset_id = json.loads(p_txt)["dataset_id"]
            err, txt = _call_tools(host.url, [
                ("submit_run", {"method": "logistic", "dataset_id": dataset_id})])[0]
            assert err is False, txt
            job = json.loads(txt)["job_id"]

            err, txt = _call_tools(host.url, [
                ("read_job", {"job_id": job, "wait": "terminal",
                              "timeout_s": 99999})])[0]
            assert err is False, txt
            got = json.loads(txt)
            assert got["wait_clamped"] is True, got
            assert f"{got['wait_cap_s']:g}" in got["wait_note"], got["wait_note"]
            assert "watch" in got["wait_note"], got["wait_note"]
            assert got["state"] == "done", got            # 夹的是等待，不是结论


def test_read_job_wait_selection_rejects_bad_input_teachably():
    """句柄给法冲突 / `wait` 写错 ⇒ **可教学**拒（不是猜一个用）。"""
    with tempfile.TemporaryDirectory(prefix="ml-wait-bad-") as td:
        with _authority(td, open_gate=True) as (_ml, host):
            err, txt = _call_tools(host.url, [
                ("read_job", {"job_id": "job-1", "job_ids": ["job-1"]})])[0]
            assert err is True, txt
            assert json.loads(txt)["error"]["info"]["kind"] == "bad_job_selection", txt

            err, txt = _call_tools(host.url, [
                ("read_job", {"job_id": "job-1", "wait": "someday"})])[0]
            assert err is True, txt
            assert json.loads(txt)["error"]["info"]["kind"] == "bad_wait", txt


# ============================================================ 看门狗（进程外等待）
def test_watch_summary_lines_are_pure_ascii():
    """看门狗的摘要行必须是**纯 ASCII** —— 否则它在 GBK 控制台上**打印就崩**。

    实测（2026-10-05 22:0x）：摘要行里放了 emoji/⇒ ⇒
    `UnicodeEncodeError: 'gbk' codec can't encode character` ⇒
    **看门狗拿到了终态却退出码 1**（新一轮"假失败"）。本判据把这条钉死。
    """
    from ml_mecha import watch

    cases = [
        {"timed_out": True, "job_state": "running", "run_id": "", "elapsed_s": 30,
         "polls": 3},
        {"ok": False, "evidence": ["handle_unknown"], "error": "no such job",
         "state": "unknown_handle"},
        {"ok": False, "evidence": ["record_missing_after_settle"], "state": "done",
         "run_id": "r1"},
        {"ok": True, "state": "done", "run_id": "r1", "primary_metric": "m",
         "metrics": {"m": 1.0}, "elapsed_s": 61, "polls": 5},
    ]
    for case in cases:
        line = watch.summary_line(case)
        assert line.isascii(), (line, "摘要行必须纯 ASCII（GBK 控制台会崩在打印上）")


def test_watch_reports_terminal_from_disk_record_and_ignores_wreck():
    """看门狗：① 有 `record.json` ⇒ 直接终态（**不需要权威**）；
    ② 盘上"没有 record.json 的旧残骸目录"**不许**被当成在跑。

    ② 不是假想：本仓 `runs/` 里真的有一个 22 小时前的残骸目录，
    没有新鲜度过滤时 `:latest` 会一头扎进去、永远等一个不会出现的记录。
    """
    from ml_mecha import watch

    with tempfile.TemporaryDirectory(prefix="ml-watch-") as td:
        runs = os.path.join(td, "runs")
        os.makedirs(runs)
        wreck = os.path.join(runs, "custom_network-1-1")
        os.makedirs(wreck)
        old = time.time() - 22 * 3600
        os.utime(wreck, (old, old))
        assert watch.newest_unfinished(runs, fresh_s=watch.DEFAULT_FRESH_S) is None, \
            "22 小时没写过的残骸目录不许被当成'在跑'"

        rid = "custom_network-2-2"
        d = os.path.join(runs, rid)
        os.makedirs(d)
        with open(os.path.join(d, "record.json"), "w", encoding="utf-8") as fh:
            json.dump({"run_id": rid, "method": "custom_network",
                       "primary_metric": "m", "metrics": {"m": 0.5},
                       "error": "", "saved_at": "2026-10-05T22:00:00"}, fh)

        got = watch.watch(handle=rid, url="", runs=runs, timeout_s=5, poll_s=0.1)
        assert got["ok"] is True and got["state"] == "done", got
        assert got["metrics"] == {"m": 0.5}, got
        assert "record.json" in got["evidence"], got["evidence"]
        assert got["record"].endswith("record.json"), got
        assert got["timed_out"] is False, got


def test_watch_timeout_never_fakes_a_state():
    """等不到终态时不许**替权威下结论**：`state` 是 `unreported`，`timed_out` 为真。"""
    from ml_mecha import watch

    with tempfile.TemporaryDirectory(prefix="ml-watch-to-") as td:
        runs = os.path.join(td, "runs")
        os.makedirs(os.path.join(runs, "custom_network-1-1"))
        got = watch.watch(handle="custom_network-1-1", url="", runs=runs,
                          timeout_s=1, poll_s=0.1)
        assert got["timed_out"] is True and got["ok"] is False, got
        assert got["state"] == "unreported", got
        assert "unreported" in watch.summary_line(got), watch.summary_line(got)


def test_watch_flags_terminal_job_without_any_run():
    """⭐ `state=done` + **没有 run_id** 这个组合：看门狗必须**报出来**，不许傻等。

    这正是用户点名的那个"不该存在"的组合（`done && error=="" && run_id==""`）——
    它无法区分"成功但慢"与"从未启动"。看门狗用假 MCP 客户端注入这个形状。
    """
    from ml_mecha import watch

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def call_tool(self, name, args):
            return {"ok": True, "job_id": args.get("job_id", ""), "state": "done",
                    "run_id": "", "error": "", "progress_latest": {}}

    with tempfile.TemporaryDirectory(prefix="ml-watch-silent-") as td:
        runs = os.path.join(td, "runs")
        os.makedirs(runs)
        real = watch.McpClient
        watch.McpClient = _FakeClient
        try:
            got = watch.watch(handle="job-7", url="http://127.0.0.1:1/mcp",
                              runs=runs, timeout_s=5, poll_s=0.1)
        finally:
            watch.McpClient = real
        assert got["ok"] is False, got
        assert "terminal_without_run" in got["evidence"], got["evidence"]
        assert got["error"], "静默终态必须带一句人能读的说明"


# ============================================================ 失败真因（2026-10-05）
# 依据：`docs/建议-ML工具长任务编排与失败可读性.md` §2.2（静默死亡 / 假位置）。
def test_empty_receipt_becomes_silent_terminal_not_done():
    """⭐ ``state=done`` + 无 run + 无失败原因 这个组合**不许存在**。

    用户原话（§2.2）："`state=done && error=="" && run_id==""` 这个组合**不应该存在**
    —— 它无法区分'成功但慢'和'从未启动'。"

    做法：把**真实命令处理链**里的 `run_method` 换成一个"回空回执"的桩
    （走的仍是 `submit → worker → engine.run` 这条真路），断言它被改写成
    `silent_terminal`，而且 `read_job` 会把它报成**失败**（不是 done）。
    """
    with tempfile.TemporaryDirectory(prefix="ml-silent-") as td:
        with _authority(td, open_gate=True) as (ml, host):
            real = ml.engine._handlers()
            patched = dict(real)
            patched["run_method"] = lambda spec, ctx: {"ok": True}     # ← 空回执
            ml.engine._handlers = lambda: patched
            try:
                job = ml.submit({"command": "run_method", "method": "logistic",
                                 "dataset_id": "ds-x"}, side="ai")
                ml.wait_job(job, 20)
            finally:
                ml.engine._handlers = lambda: real

            assert job.result["ok"] is False, job.result
            assert job.result["error_kind"] == "silent_terminal", job.result
            assert "silent_terminal" in job.result["error"], job.result
            assert job.result["error_hint"], job.result

            # 用户可见的那条路：`read_job` 必须报失败（不是"done 但什么都没有"）
            err, txt = _call_tools(host.url, [("read_job", {"job_id": job.id})])[0]
            assert err is True, txt
            info = json.loads(txt)["error"]["info"]
            assert info["kind"] == "silent_terminal", info


def test_failure_without_a_kind_still_gets_one():
    """`ok=False` 但没归类 ⇒ 补一个**具体** kind（"不知道哪一类"本身要看得见）。"""
    from ml_mecha.assembly import enforce_terminal_invariant

    got = enforce_terminal_invariant({"ok": False, "error": "炸了"}, "run_method")
    assert got["error_kind"] == "job_command_failed", got
    # 正常形状**原样过**（不变量不许把成功改坏）
    ok_shape = {"ok": True, "run_ids": ["r1"], "error": ""}
    assert enforce_terminal_invariant(ok_shape, "run_method") == ok_shape
    # 不产 run 的命令不受这条不变量约束（prepare_dataset 正常成功就是没有 run_ids）
    prep = {"ok": True, "dataset_id": "ds-1", "error": ""}
    assert enforce_terminal_invariant(prep, "prepare_dataset") == prep


def test_cuda_oom_and_poison_are_classified_and_pinned_to_first_line():
    """⭐ 显存不足要**一等化**、CUDA 断言/回声要**置顶**（不许埋在 hint 末尾）。

    两条实测反馈（§2.2 / §3.3）：
    - 显存不足被 CUDA 报成 `device-side assert`，栈指向 `set_probe` ⇒ "事前完全
      不知道会不够"；⇒ 认出来 + 把**数字**（要多少/剩多少/总量）摆到首行；
    - "毒化提示写了，但放在 `hint` 末尾，我读了好几轮才注意到" ⇒ 回声必须**置顶**。

    ⚠ 走的是**真实回执构造入口** `_summarize_record`（那条路上曾经有过
    "`warning` 只取最后一行 ⇒ 认不出 CUDA 断言"的失误，只测纯函数会漏掉它）。
    """
    from ml_mecha import engine as E
    from ml_toolbox.api import RunRequest
    from ml_toolbox.core.contracts import MLResult
    from ml_toolbox.core.runner import RunRecord

    def _receipt(error: str, run_id: str = "r-x"):
        res = MLResult(method_name="custom_network", task="supervised",
                       metrics={"m": 1.0}, primary_metric="m",
                       params={}, elapsed=0.1, error=error)
        rec = RunRecord(run_id=run_id, method="custom_network", family="neural",
                        task="supervised", target_kind=None, dataset="ds-1",
                        pipeline_id="p-1", config={}, result=res)
        return E._summarize_record(rec, "ds-1", RunRequest(method="custom_network"))

    saved = dict(E._CUDA_POISON)
    E._CUDA_POISON.clear()
    try:
        # ① 显存不足：**一等**错误 + 数字进首行 + 明说"不必重启"
        oom = _receipt("CUDA out of memory. Tried to allocate 2.00 GiB "
                       "(GPU 0 has a total capacity of 5.99 GiB of which 1.21 GiB "
                       "is free)")
        assert oom.error_kind == "cuda_oom", oom
        assert oom.warning.startswith("⚠ 显存不足"), oom.warning
        assert "2.00 GiB" in oom.warning and "1.21 GiB" in oom.warning, oom.warning
        assert "不必重启" in oom.error_hint, oom.error_hint

        # ② 第一次 device-side assert：置顶 + 首次记号
        first = _receipt("CUDA error: device-side assert triggered", run_id="r-first")
        assert first.error_kind == "cuda_device_assert", first
        assert first.warning.startswith("⚠ CUDA 断言"), first.warning
        assert "毒化" in first.warning, first.warning

        # ③ 之后的每一次：**回声**必须置顶，并点名那次 run + "别在这里改代码"
        echo = _receipt("CUDA error: device-side assert triggered", run_id="r-second")
        assert echo.error_kind == "cuda_context_poisoned", echo
        assert echo.warning.startswith("⚠⚠"), echo.warning
        assert "回声" in echo.warning and "r-first" in echo.warning, echo.warning
        assert "别在这里改代码" in echo.warning, echo.warning
        assert "别在这里改代码" in echo.error_hint, echo.error_hint
    finally:
        E._CUDA_POISON.clear()
        E._CUDA_POISON.update(saved)


def test_missing_custom_method_is_teaching_not_internal_error():
    """注册表里没有 `custom_network` ⇒ **自解释**，不许报成 `internal_error`。

    用户 §2.3 的兜底要求原话："报错要能自解释 —— '自定义网络方法尚未就绪，请重试'，
    而不是把 59 个可用方法列一遍（那反而误导，让人以为方法名写错了）"。
    """
    with tempfile.TemporaryDirectory(prefix="ml-notready-") as td:
        with _authority(td, open_gate=True) as (ml, _host):
            real = ml.engine._handlers()
            patched = dict(real)

            def _boom(spec, ctx):
                raise KeyError("未注册的方法: custom_network"
                               "（可用: ['adaboost', 'logistic', 'ridge']）")

            patched["run_method"] = _boom
            ml.engine._handlers = lambda: patched
            try:
                got = ml.engine.run({"command": "run_method", "method": "logistic",
                                     "dataset_id": "ds-x"})
            finally:
                ml.engine._handlers = lambda: real
            assert got["ok"] is False, got
            assert got["error_kind"] == "custom_network_not_ready", got
            assert "不是你写错了" in got["error"], got
            # 别的内部错误**不许**被这条兜底吃掉（窄判据）
            assert not _is_missing_custom_method_public(RuntimeError("boom"))
            assert not _is_missing_custom_method_public(KeyError("未注册的方法: ridge"))


def _is_missing_custom_method_public(exc):
    from ml_mecha.engine import _is_missing_custom_method
    return _is_missing_custom_method(exc)


# ============================================================ L1 身份台账（2026-10-05）
# 依据：`docs/建议-ML工具长任务编排与失败可读性.md` §3.4 / §5（运行清单）。
def test_dataset_id_is_content_addressed_and_idempotent():
    """⭐ **同内容 ⇒ 同一个 id**（幂等）——用户"上一轮用的哪份数据只能靠记忆对齐"的正解。

    实测原文（§3.4）："重启后 `dataset_id` 全失效，必须重新 `prepare_dataset`。本轮
    我因此**重复登记了七八次**。更麻烦的是：每次登记**返回新 id**，于是'上一轮用的
    哪份数据'在会话里丢失，只能靠**我的记忆**对齐。"

    本判据钉三条：
    ① 同一份数据（**换个名字**）再登记 ⇒ 同一个 id、`reused: true`；
    ② `aliases` **累加**（两个名字都查得到）、`registrations` 计数可见；
    ③ **不同**内容 ⇒ **不同** id（否则"幂等"会退化成"把所有数据认成一份"）。
    """
    with tempfile.TemporaryDirectory(prefix="ml-ledger-idem-") as td:
        with _authority(td, open_gate=True) as (_ml, host):
            src = {"kind": "synthetic", "task": "classification", "n_samples": 60}
            (a_err, a_txt), = _call_tools(host.url, [
                ("prepare_dataset", {"source": dict(src), "name": "口径对照"})])
            assert a_err is False, a_txt
            a = json.loads(a_txt)
            assert a["dataset_id"].startswith("ds-"), a
            assert a["reused"] is False, a
            assert a["fp"] and a["fp"] in a["dataset_id"], a       # id 由指纹承载
            assert a["label"] == "口径对照", a

            # ① 同内容、**换个名字** ⇒ 同 id + reused
            (b_err, b_txt), = _call_tools(host.url, [
                ("prepare_dataset", {"source": dict(src), "name": "改了个名字"})])
            assert b_err is False, b_txt
            b = json.loads(b_txt)
            assert b["dataset_id"] == a["dataset_id"], (a["dataset_id"], b["dataset_id"])
            assert b["reused"] is True, b
            # ② 别名累加 + 登记次数（"这份数据被登记过几次"要看得见）
            assert set(b["aliases"]) >= {"口径对照", "改了个名字"}, b["aliases"]
            assert b["registrations"] == 2, b

            # ③ 不同内容 ⇒ 不同 id（防"幂等"变成"全都认成一份"）
            (c_err, c_txt), = _call_tools(host.url, [
                ("prepare_dataset", {"source": {"kind": "synthetic",
                                                "task": "classification",
                                                "n_samples": 61},
                                     "name": "口径对照"})])
            assert c_err is False, c_txt
            c = json.loads(c_txt)
            assert c["dataset_id"] != a["dataset_id"], c

        # ④ ⭐ **跨重启**再登记（**换个名字**）：仍然必须是同一个 id。
        #    ⚠ 这一条才是真正钉死"名字不许进 id"的地方：会话内的幂等索引
        #    （`fp_decl → id`）会把"名字进了 id"这种实现**掩盖**掉（第一次登记返回
        #    的 id 本来就是它自己算的）——只有跨会话重新计算 id 才暴露它。
        #    实测教训：我第一版判据漏了这一步，于是"把 name 掺进 fp"这个变异
        #    **杀不死**（等价变异）。
        ml2 = assemble_ml_mecha(root=td, seed=13)
        try:
            ml2.switch_ai()
            again = ml2.tools.execute("prepare_dataset", {
                "source": {"kind": "synthetic", "task": "classification",
                           "n_samples": 60},
                "name": "重启后换的名字"})["value"]
            assert again["dataset_id"] == a["dataset_id"], (
                "跨重启+换名字之后 id 变了 ⇒ 名字掺进了身份（'上一轮用的哪份数据'"
                "又会失配）", again["dataset_id"], a["dataset_id"])
            assert again["reused"] is False, again      # 新会话里是**新建**（不是复用）
            assert "重启后换的名字" in again["aliases"], again["aliases"]
        finally:
            ml2.close()


def test_dataset_survives_restart_by_lazy_restore():
    """⭐ **重启后同一个 id 仍然可用**（按台账声明就地重建，不必重新登记）。

    用户要的正是这条（§3.4 "登记表落盘，重启后按指纹自动复原"）。
    判据做**真的两次装配**（同一 root = 模拟重启）：第二次装配的会话里**没有**任何
    数据集，直接拿第一次的 id 跑 `run_method` ⇒ 必须成功，且事后那个 id 出现在会话里。
    """
    with tempfile.TemporaryDirectory(prefix="ml-ledger-restart-") as td:
        src = {"kind": "synthetic", "task": "classification", "n_samples": 60}

        def _fresh():
            """同一 root 的**新装配**（= 模拟"重启权威"）。"""
            ml = assemble_ml_mecha(root=td, seed=13)
            ml.switch_ai()
            return ml

        ml1 = _fresh()
        try:
            got = ml1.tools.execute("prepare_dataset",
                                    {"source": dict(src), "name": "跨重启"})["value"]
            dataset_id, fp = got["dataset_id"], got["fp"]
            assert got["restorable"] is True, got        # synthetic 可重建
        finally:
            ml1.close()

        # ---- 模拟重启：新装配、空会话，只有盘上的台账 ----
        ml2 = _fresh()
        try:
            assert ml2.engine.dataset_ids() == [], "新装配不该凭空有数据集（前提守卫）"
            receipt = ml2.tools.execute("run_method", {
                "method": "logistic", "dataset_id": dataset_id})
            assert receipt.get("is_error") is not True, receipt
            assert receipt["value"]["run_id"], receipt
            back = ml2.engine.dataset_entry(dataset_id)
            assert back is not None, "懒复原之后这个 id 必须真的在会话里"
            assert back.fp == fp, (back.fp, fp)          # 复原出来的是**同一份**
        finally:
            ml2.close()


def test_dataset_ledger_view_links_fingerprint_label_and_runs():
    """`describe_dataset` 的 `ledger`：**指纹 → 名字 → 用它的运行**串成一条链（R38）。

    用户要的"运行清单"：*"数据集（按指纹） → 负载 → 作业 → run_id → 判据与结果"*。
    这里钉前两环 + 运行计数：跑一次 `run_method` 之后，`recent_runs` 里必须出现它
    （说明台账与 `runs/` 存档**对得上**，不是两本各说各话的账）。
    """
    with tempfile.TemporaryDirectory(prefix="ml-ledger-view-") as td:
        with _authority(td, open_gate=True) as (_ml, host):
            (p_err, p_txt), = _call_tools(host.url, [
                ("prepare_dataset", {"source": dict(_SYNTHETIC),
                                     "name": "清单用"})])
            assert p_err is False, p_txt
            dataset_id = json.loads(p_txt)["dataset_id"]
            (r_err, r_txt), = _call_tools(host.url, [
                ("run_method", {"method": "logistic", "dataset_id": dataset_id,
                                "persist": True})])       # 清单要读的是**盘上**的存档
            assert r_err is False, r_txt
            run_id = json.loads(r_txt)["run_id"]

            (d_err, d_txt), = _call_tools(host.url, [
                ("describe_dataset", {"dataset_id": dataset_id})])
            assert d_err is False, d_txt
            ledger = json.loads(d_txt)["ledger"]
            assert ledger["fp"] and ledger["fp"] in ledger["dataset_id"], ledger
            assert ledger["label"] == "清单用", ledger
            assert ledger["restorable"] is True, ledger
            ids = [r["run_id"] for r in ledger["recent_runs"]]
            assert run_id in ids, (run_id, ids)
            entry = next(r for r in ledger["recent_runs"] if r["run_id"] == run_id)
            assert entry["method"] == "logistic", entry
            assert entry["score"] != "", entry            # 结果也在（不只是 run_id）


def test_dataset_needs_reprepare_says_why_and_how():
    """不可复原的来源：回**可教学**的 `unknown_dataset` + 为什么 + 怎么办。

    ⚠ 这条是"诚实边界"的判据：`frame`/`arrays` 是进程内对象，重启后**真的**重建不了
    ⇒ 不许假装复原成功；但必须比"未知 dataset_id（本会话已准备的数据集：（空））"
    多给两句（台账里有它、为什么不能复原、重新登记会得到同一个 id）。
    """
    with tempfile.TemporaryDirectory(prefix="ml-ledger-prep-") as td:
        frame = _frame(40, 3)
        ml = assemble_ml_mecha(root=td, seed=13)
        try:
            ml.switch_ai()
            got = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": frame, "target": "target"},
                "name": "进程内"})["value"]
            assert got["restorable"] is False, got
            assert "无法自动复原" in got["restore_note"], got["restore_note"]
            dataset_id = got["dataset_id"]
        finally:
            ml.close()

        # 台账里有它，但**新装配**的会话里没有（frame 重建不了）⇒ 可教学地拒
        ml2 = assemble_ml_mecha(root=td, seed=13)
        try:
            ml2.switch_ai()
            # ⚠ 命令层的失败留在**回执**里（`ok=false` + kind/hint），不抛异常
            #   ——这是既有契约（`read_job` 就是据此把失败说给人听的）。
            got = ml2.engine.run({"command": "run_method", "method": "logistic",
                                  "dataset_id": dataset_id})
            assert got["ok"] is False, "不可复原的 id 不该静默跑通"
            assert got["error_kind"] == "unknown_dataset", got
            hint = got["error_hint"] or ""
            assert "台账" in hint, hint
            assert "同一个 id" in hint, hint          # 告诉用户"重登会得到同一个 id"
        finally:
            ml2.close()


# ============================================================ L2 调度器（队列）（2026-10-05）
# 依据：`docs/建议-ML工具长任务编排与失败可读性.md` §3.2（上限即报错、无队列）。
def _slow_handler(seconds: float = 1.2):
    """把 `run_method` 换成一个**慢但合法**的桩（回执形状仍满足终态不变量）。

    ⚠ 为什么必须慢：队列只在"跑得比提交慢"时才显现。用真的 logistic（~0.1 s）
    根本排不上队——判据会**假绿**（我第一版探针就是这么被骗的）。
    """
    def handler(spec, ctx):
        time.sleep(seconds)
        return {"ok": True, "command": "run_method", "run_ids": ["r-slow"],
                "primary_metric": "m", "metrics": {"m": 1.0}, "warnings": [],
                "error": ""}
    return handler


def test_queue_admits_over_limit_reports_position_and_is_hot_editable():
    """⭐ **超上限不再报错，而是入队** + 回执给 `queue_position` + 上限**热改**。

    用户实测（§3.2）："上限 4 个并发。超出直接**报错**，而不是**排队**。我因此必须
    '投一个、等一个、再投一个'……上限调整**不需要重启**（重启会清空 dataset 与会话态，
    代价很大）。"

    本判据一次钉三步（都在**工具面**上走，AI 看到的就是这些）：
    ① 第 2、3 个回执带 `queued: true` 与**递增的 queue_position**（不是报错）；
    ② `read_job` 把"排队"与"框架 state"**分开报**（排队时框架仍是 running）；
    ③ `set_job_limit` 调大 ⇒ 立刻放行，最终**三个都跑成**（队列不是"吞任务"）。
    """
    with tempfile.TemporaryDirectory(prefix="ml-queue-") as td:
        with _authority(td, open_gate=True) as (ml, host):
            # 先压低上限（热改，不需要重启）：只放 1 个跑
            err, txt = _call_tools(host.url, [("set_job_limit", {"limit": 1})])[0]
            assert err is False, txt
            assert json.loads(txt)["limit"] == 1, txt

            real = ml.engine._handlers()
            patched = dict(real)
            patched["run_method"] = _slow_handler(1.2)
            ml.engine._handlers = lambda: patched
            try:
                sub = []
                for i in range(3):
                    e, t = _call_tools(host.url, [
                        ("submit_run", {"method": "logistic",
                                        "dataset_id": "ds-x"})])[0]
                    assert e is False, t
                    sub.append(json.loads(t))
                # ① 后两个入队（**不是**报错），位次递增
                assert sub[0]["queued"] is False, sub[0]
                assert sub[1]["queued"] is True and sub[1]["queue_position"] == 1, sub[1]
                assert sub[2]["queued"] is True and sub[2]["queue_position"] == 2, sub[2]
                assert sub[2]["queue_note"], sub[2]

                # ② 排队与框架状态分开报（read_job 一次看清两件事）
                e, t = _call_tools(host.url, [
                    ("read_job", {"job_id": sub[2]["job_id"]})])[0]
                assert e is False, t
                got = json.loads(t)
                assert got["queued"] is True, got
                assert got["queue_position"] >= 1, got
                assert got["state"] == "running", got          # 框架：线程已起
                assert "排队" in got["wait_note"], got["wait_note"]

                # ③ 热改上限 ⇒ 排队者被唤醒
                e, t = _call_tools(host.url, [("set_job_limit", {"limit": 3})])[0]
                assert e is False, t
                up = json.loads(t)
                assert up["limit"] == 3 and up["limit_before"] == 1, up
            finally:
                ml.engine._handlers = lambda: real

            # 三个都必须跑成（队列是排队，不是丢任务）
            for item in sub:
                status = ml.jobs.wait(item["job_id"], 30)
                assert status["state"] == "done", (item["job_id"], status)
            assert ml.queue_state()["queued"] == 0, ml.queue_state()

            # 坏上限：可教学地拒（0 会把任务永久堵在队列里）
            err, txt = _call_tools(host.url, [("set_job_limit", {"limit": 0})])[0]
            assert err is True, txt
            assert json.loads(txt)["error"]["info"]["kind"] == "bad_job_limit", txt


def test_queued_job_is_cancellable_before_it_ever_runs():
    """⭐ **队列里的任务可以取消**（用户明确要求："我会投多、然后改主意"）。

    取消发生在**开跑之前** ⇒ 必须如实说"它还没开始跑、没有产生任何运行"
    （`cancelled_in_queue`），而不是假装"训练被中断"（那会让人以为跑过一段）。
    """
    with tempfile.TemporaryDirectory(prefix="ml-queue-cancel-") as td:
        with _authority(td, open_gate=True) as (ml, host):
            _call_tools(host.url, [("set_job_limit", {"limit": 1})])
            real = ml.engine._handlers()
            patched = dict(real)
            patched["run_method"] = _slow_handler(1.5)
            ml.engine._handlers = lambda: patched
            try:
                first, second = [], []
                for sink in (first, second):
                    e, t = _call_tools(host.url, [
                        ("submit_run", {"method": "logistic",
                                        "dataset_id": "ds-x"})])[0]
                    assert e is False, t
                    sink.append(json.loads(t))
                assert second[0]["queued"] is True, second[0]   # 前提：它真的在排队

                e, t = _call_tools(host.url, [
                    ("cancel_run", {"job_id": second[0]["job_id"]})])[0]
                assert e is False, t
                assert json.loads(t)["cancel_requested"] is True, t   # 排队中 ⇒ 真的受理

                status = ml.jobs.wait(second[0]["job_id"], 30)
                # 框架侧如实记成 `cancelled`（取消确实生效了，不是"跑完了"）
                assert status["state"] == "cancelled", status
                got = ml.jobs.get(second[0]["job_id"]).result
                assert got["ok"] is False, got
                assert got["error_kind"] == "cancelled_in_queue", got
                assert not got["run_ids"], got            # **没有**产生运行
                # 取消的那条**没有占用名额**：第一个照常跑完，队列清空
                assert ml.jobs.wait(first[0]["job_id"], 30)["state"] == "done"
                state = ml.queue_state()
                assert state["queued"] == 0 and state["running"] == 0, state
            finally:
                ml.engine._handlers = lambda: real


# ============================================================ L3 资源真相（2026-10-05）
# 依据：`docs/建议-ML工具长任务编排与失败可读性.md` §3.3（显存预检 / 框架分块）。
def test_vram_precheck_rejects_only_when_the_declared_peak_cannot_fit():
    """⭐ **提交前**显存预检：声明了峰值**且明显装不下** ⇒ 可读拒绝（不是 CUDA 报错）。

    用户实测（§3.3）：6 GB 卡、配置峰值 5.30 GB、运行时只剩 3.6 GB ⇒ OOM，
    而 CUDA 把它报成 `device-side assert`（栈指向假位置）。他的原话：
    "**我事前完全不知道会不够。**"

    判据钉四条边界（**都是"诚实"问题**）：
    ① 声明 > 可用×1.05 ⇒ 拒，kind=`cuda_oom_predicted`，且**三个数字都在话里**；
    ② 声明 ≤ 可用 ⇒ 放行（不能把正常任务吓回去）；
    ③ **没声明** ⇒ 不判（框架不猜你网络的峰值——猜出来会假装精确）；
    ④ `device='cpu'` ⇒ 不判（跑 CPU 不涉及显存）。
    ⚠ 显存探针被**注入**（`assembly.vram.probe`），所以这条判据在**没有 GPU** 的
    机器上也确定地跑（否则它只能靠有卡时才能验，等于没判据）。
    """
    from ml_mecha import assembly as _asm

    real = _asm.vram.probe
    _asm.vram.probe = lambda device=None: {"total_mb": 6141.0, "free_mb": 1024.0,
                                           "used_mb": 5117.0, "device": "cuda:0"}
    try:
        with tempfile.TemporaryDirectory(prefix="ml-vram-") as td:
            ml = assemble_ml_mecha(root=td, seed=13)
            ml.switch_ai()
            try:
                # ① 明显不够 ⇒ 可读拒绝（带三个数字）
                out = ml.tools.execute("run_method", {
                    "method": "logistic", "dataset_id": "ds-x",
                    "resource_guard": {"max_peak_mb": 4096}})
                assert out["is_error"] is True, out
                info = out["error"]["info"]
                assert info["kind"] == "cuda_oom_predicted", info
                msg = out["error"]["message"]
                assert "4096" in msg and "1024" in msg and "6141" in msg, msg
                assert "OOM" in msg or "显存" in msg, msg
                # 出路必须给全（减 batch / 分块 / 换 cpu）
                assert "batch" in info["hint"] + info["suggest"], info

                # ② 装得下 ⇒ 不拒（这里只验"没过预检"这一层：方法不存在之类的
                #    后置失败不算；用 minimize 的办法——直接调内部函数）
                assert _asm.vram.precheck(
                    {"resource_guard": {"max_peak_mb": 512}, "device": "cuda:0"})
                # ③ 没声明 ⇒ 不判
                assert _asm.vram.precheck({"device": "cuda:0"}) == {}
                # ④ 明确 CPU ⇒ 不判（即使声明很大）
                assert _asm.vram.precheck(
                    {"resource_guard": {"max_peak_mb": 99999},
                     "device": "cpu"}) == {}
                # ⑤ 坏声明 ⇒ 可教学拒（不是静默忽略）
                bad = ml.tools.execute("run_method", {
                    "method": "logistic", "dataset_id": "ds-x",
                    "resource_guard": {"max_peak_mb": "很大"}})
                assert bad["is_error"] is True, bad
                assert bad["error"]["info"]["kind"] == "bad_peak_mb", bad
            finally:
                ml.close()
    finally:
        _asm.vram.probe = real


def test_peak_vram_is_not_faked_on_cpu():
    """CPU 上**不许**留"峰值显存"字段（不假装有数字）。

    与上一条配对：**没有 CUDA 就回空**——一个 0 或一个编造的数都会误导决策
    （用户拿它去设 `max_peak_mb`，编造等于害他再 OOM 一次）。
    """
    from ml_mecha import vram

    if vram.probe():
        print("    （跳过：本机有 CUDA，这条判据只在无 CUDA 时才有意义）")
        return
    assert vram.peak_mb() is None, vram.peak_mb()
    assert vram.probe() == {}, vram.probe()


def test_watch_latest_is_labelled_as_a_heuristic():
    """⭐ 无句柄模式（`:latest`）必须**自称启发式**——它可能选中启动器的探针 run。

    用户实测的纠正（2026-10-05）："那个 `-42` **不是我的**……大概是**启动器自己的
    探针 run**。你的新鲜度过滤把它认成'在跑'了——建议你按 `job_id` 反查 `run_id`
    来判定归属，而不是靠 `:latest` 猜。"
    ⇒ 在没有句柄的时候，看门狗**只能**猜；那就把"这是猜的、以及怎么不猜"说出来
    （旧版会一本正经地报"盯住 xxx"，读起来像一个确定的事实）。
    """
    from ml_mecha import watch

    with tempfile.TemporaryDirectory(prefix="ml-watch-latest-") as td:
        runs = os.path.join(td, "runs")
        fresh = os.path.join(runs, "custom_network-9-9")
        os.makedirs(fresh)                      # 新鲜、未落盘 ⇒ 会被启发式挑中
        got = watch.watch(handle=":latest", url="", runs=runs,
                          timeout_s=1, poll_s=0.1)
        assert "newest_unfinished_run" in got["evidence"], got["evidence"]
        notes = " ".join(got["notes"])
        assert "启发式" in notes, got["notes"]
        assert "job_id" in notes, got["notes"]     # 指路：要精确就传句柄


# ============================================================ R40–R43 曲线分析（2026-10-05）
# 依据：用户实测的一次真实误判（代价一轮 1500 epoch 的训练）：
#   "我一直只读 `latest`（一个点），靠跨 run 比终值推断曲线已饱和，据此把 epoch 从
#    900 砍到 250。后来拉了同一 run 内的曲线，发现 `fid/sub1` 从 0.367 一路升到 0.682
#    且**在 epoch 250 时仍在爬** —— 根本没收敛，是严重欠训练。"
# 锚点：**两条真实 run** 的逐点序列（抽成仓内夹具；判据不依赖 runs/ 目录存亡）。
_CURVE_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _curve_fixture(run_id: str, tag: str) -> dict:
    path = _CURVE_FIXTURE_DIR / f"curve_{run_id}_{tag.replace('/', '_')}.json"
    assert path.is_file(), f"缺夹具 {path.name}（曲线判据的锚点）"
    return json.loads(path.read_text(encoding="utf-8"))


def test_convergence_warns_on_the_real_underfit_run_and_not_on_the_converged_one():
    """⭐ R42：**250 epoch 那条必须报警、1500 epoch 那条不许报**（判别力判据）。

    ⚠ 这条判据的价值全在"**两条真实曲线的差别**"上：
    - `custom_network-9008175-1002`（250 epoch）：用户据它判断"已饱和"并砍了 epoch
      ⇒ 当时 `fid/sub1` **末段仍在爬** ⇒ 必须给 `underfit` 警告；
    - `custom_network-1422969-1010`（1500 epoch）：真的到平台了（末 10 点甚至在下行）
      ⇒ **不许报欠训练**（否则警告会变成噪声，人就会学会忽略它）。

    ⚠ 如实记录：这两条曲线的判读**没有调过阈值**（`TAIL=15`/`TREND_FRAC=1%` 都是
    先定的）。第一版判据在**真实曲线上一度红**——因为它要求"严格单调上升"，
    而真实曲线有 ±0.005 抖动 ⇒ 判成 `mixed`，**恰好漏掉这个警告**。
    抗噪判据（末段"前三分位均值 vs 后三分位均值"）才是对的。
    """
    sys.path.insert(0, str(REPO))
    from ml_mecha import curve

    under = _curve_fixture("custom_network-9008175-1002", "fid/sub1")
    conv = _curve_fixture("custom_network-1422969-1010", "fid/sub1")
    sh_under = curve.shape(under["steps"], under["values"])
    sh_conv = curve.shape(conv["steps"], conv["values"])

    note_under = curve.convergence_note("fid/sub1", sh_under)
    assert sh_under["trend"] == "rising", sh_under
    assert note_under["kind"] == "underfit", note_under
    assert "欠训练" in note_under["note"], note_under
    assert "别按终值" in note_under["note"], note_under      # 指路：别按终值判断
    # 用户的原始观察：0.367 → 0.682（这两个数要能在形状里看到）
    assert abs(sh_under["first"] - 0.3670) < 5e-3, sh_under
    assert abs(sh_under["last"] - 0.6823) < 5e-3, sh_under

    note_conv = curve.convergence_note("fid/sub1", sh_conv)
    assert sh_conv["trend"] == "flat", sh_conv               # 1500 epoch 已到平台
    assert note_conv["kind"] == "", note_conv                # ⇒ **不许报警**

    # ⭐ **方向也要分对**：同一条 250 epoch run 的 `val_loss` 末段也在爬，但那是
    #   **发散**不是欠训练（两者的处置相反）——第一版正是把它报成了欠训练。
    vl = _curve_fixture("custom_network-9008175-1002", "val_loss")
    note_vl = curve.convergence_note("val_loss",
                                     curve.shape(vl["steps"], vl["values"]))
    assert note_vl["kind"] == "diverging", note_vl
    # 方向未知的量（梯度范数等）**不给诊断**（趋势照给，但不硬凑一句话）
    assert curve.convergence_note("grad_norm", {"trend": "rising", "tail": 15,
                                                "first": 1.0, "last": 2.0})["kind"] == ""
    assert curve.direction("grad_norm") == "unknown"
    assert curve.direction("val_loss") == "lower"
    assert curve.direction("fid/sub1") == "higher"


def test_best_by_any_tag_compares_against_val_loss_choice():
    """⭐ R41：按**任意 tag** 挑最佳 step，并与按 `val_loss` 的选择**对照**。

    用户实测（§R41）："`best.pt` 是按 `val_loss` 选的，但我实测同一 run 里
    `val_loss` 一直在 155–170 抖动，而 `fid` 早就平了 ⇒ **按 val_loss 选出的 best
    未必是 fid 最好的那轮**。我做 `eval_custom` 时就踩了这个（口径不一致）。"
    """
    sys.path.insert(0, str(REPO))
    from ml_mecha import curve

    fid = _curve_fixture("custom_network-9008175-1002", "fid/sub1")
    got = curve.best_point(fid["steps"], fid["values"], tag="fid/sub1")
    assert got["lower_better"] is False, got
    assert got["step"] is not None and got["value"] is not None, got
    # fid 仍在爬 ⇒ 最佳点应当在**最末**（这本身就是"欠训练"的另一面）
    assert got["index"] >= got["n"] - 3, got

    vl = _curve_fixture("custom_network-9008175-1002", "val_loss")
    b_vl = curve.best_point(vl["steps"], vl["values"], tag="val_loss")
    assert b_vl["lower_better"] is True, b_vl
    # 两者的最佳轮**不是同一轮**（这正是用户踩到的口径不一致）
    assert abs(got["step"] - b_vl["step"]) > 5, (got, b_vl)


def test_compare_aligns_multiple_runs_without_interpolating():
    """⭐ R40：多 run 曲线**对齐**（同一根 step 轴；缺失处 `null`，**不插值**）。

    用户实测（§R40）："我要比'逐对耦合 vs 聚合量'两条轨迹，而不是只比终值。"
    ⚠ `null` 而不是插值是**硬要求**：插值会造出一段没算过的曲线，而"看起来更整齐"
    正是它危险的地方。
    """
    sys.path.insert(0, str(REPO))
    from ml_mecha import curve

    a = _curve_fixture("custom_network-9008175-1002", "fid/sub1")
    b = _curve_fixture("custom_network-1422969-1010", "fid/sub1")
    series = {
        "run-250": {"fid/sub1": {"steps": a["steps"], "values": a["values"]}},
        # 故意截掉前面一段 ⇒ 制造"某个 step 上 A 有、B 没有"的情形
        "run-1500": {"fid/sub1": {"steps": b["steps"][100:],
                                  "values": b["values"][100:]}},
    }
    got = curve.align(series, ["fid/sub1"], limit=500)["tags"]["fid/sub1"]
    assert got["n_runs"] == 2, got
    axis = got["steps"]
    vals_a = got["runs"]["run-250"]
    vals_b = got["runs"]["run-1500"]
    assert len(axis) == len(vals_a) == len(vals_b), (len(axis), len(vals_a))
    # ① 轴是**并集**且升序
    assert axis == sorted(axis), axis[:5]
    # ② 缺失处是 `None`（**不是 0、也不是插值出来的数**）
    assert any(v is None for v in vals_a) or any(v is None for v in vals_b), \
        "这条判据需要至少一处缺失（夹具被改动过？）"
    missing = [i for i, v in enumerate(vals_b) if v is None]
    assert missing, "run-1500 截掉了前 100 点 ⇒ 那段必须是 None"
    # ③ 真实点**逐字**保留（对齐不许改数）
    for i, step in enumerate(axis):
        if vals_b[i] is not None:
            idx = b["steps"].index(step)
            assert vals_b[i] == b["values"][idx], (step, vals_b[i])
    # ④ 默认 tag 选取要跳过逐层梯度（几十条会把回执灌满）
    wide = {"r1": {"loss": {"steps": [1], "values": [1.0]},
                   "grad/x.weight": {"steps": [1], "values": [1.0]},
                   "fid": {"steps": [1], "values": [1.0]}},
            "r2": {"loss": {"steps": [1], "values": [2.0]},
                   "grad/x.weight": {"steps": [1], "values": [2.0]},
                   "fid": {"steps": [1], "values": [2.0]}}}
    picked = curve.default_tags(wide)
    assert "grad/x.weight" not in picked, picked
    assert set(picked) == {"loss", "fid"}, picked


def test_watch_prefers_job_ownership_over_guessing():
    """⭐ 无句柄时**先按作业归属找**（权威的 `jobs` 查询），找不到才退回启发式。

    用户实测的纠正（2026-10-05）："盘上那个 `-42` **不是我的**，大概是启动器自己的
    探针 run …… 建议你按 `job_id` 反查 `run_id` 来判定归属，而不是靠 `:latest` 猜。"
    ⇒ 判据用假 MCP 客户端给出"作业归属表"，断言：
    ① 选中**归属表里那个** run（哪怕盘上还有更"新鲜"的目录）；
    ② `evidence` 里是 `owned_by_job:<job_id>`（可追溯），而**不是** `newest_unfinished_run`；
    ③ 归属表里**没有**在跑的 run 时，才退回启发式并**自称启发式**。
    """
    from ml_mecha import watch

    with tempfile.TemporaryDirectory(prefix="ml-watch-own-") as td:
        runs = os.path.join(td, "runs")
        mine = os.path.join(runs, "custom_network-111-1")
        probe = os.path.join(runs, "custom_network-222-2")
        os.makedirs(mine)
        os.makedirs(probe)
        # 让"探针"那个更晚写 ⇒ 纯启发式会挑错（这正是用户踩到的情形）
        os.utime(mine, (time.time() - 20, time.time() - 20))

        class _FakeClient:
            def __init__(self, *a, **k):
                pass

            def call_tool(self, name, args):
                assert name == "run_queries", name
                return {"ok": True, "steps": [{"index": 0, "op": "jobs", "ok": True,
                                               "output": {
                                                   "available": True, "n": 1,
                                                   "jobs": [{"job_id": "job-7",
                                                             "state": "running",
                                                             "run_ids": ["custom_network-111-1"]}]}}]}

        real = watch.McpClient
        watch.McpClient = _FakeClient
        try:
            got = watch.watch(handle=":latest", url="http://127.0.0.1:1/mcp",
                              runs=runs, timeout_s=1, poll_s=0.1,
                              report_every_s=0)
        finally:
            watch.McpClient = real
        assert got["run_id"] == "custom_network-111-1", got["run_id"]
        assert "owned_by_job:job-7" in got["evidence"], got["evidence"]
        assert "newest_unfinished_run" not in got["evidence"], got["evidence"]
        assert "启发式" not in " ".join(got["notes"]), got["notes"]

        # ③ 归属表里没有在跑的 run ⇒ 退回启发式，并且**自称启发式**
        class _EmptyClient(_FakeClient):
            def call_tool(self, name, args):
                return {"ok": True, "steps": [{"index": 0, "op": "jobs", "ok": True,
                                               "output": {"available": True, "n": 0,
                                                          "jobs": []}}]}

        watch.McpClient = _EmptyClient
        try:
            got2 = watch.watch(handle=":latest", url="http://127.0.0.1:1/mcp",
                               runs=runs, timeout_s=1, poll_s=0.1, report_every_s=0)
        finally:
            watch.McpClient = real
        assert "newest_unfinished_run" in got2["evidence"], got2["evidence"]
        assert "启发式" in " ".join(got2["notes"]), got2["notes"]


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


def test_audit_reason_is_the_callers_and_falls_back_to_human_text():
    """⭐ 命令审计的 ``reason``：**调用方给的原话要保住**，没给则**人话兜底**。

    ## 为什么有这条（用户实测反馈）

    面板"飞行记录仪"折叠行显示的是 ``reason``（资产里 ``reason || target``）。
    原先 ``engine._invoke_core_command`` 写死 ``args["reason"] = f"ml_mecha:{command}"``
    ⇒ **无条件覆盖**：AI 写下的意图被抹掉，面板上只剩 ``ml_mecha:eval_custom`` 这种机械串
    （既不说明做了什么、也不说明为什么）。

    现在与另一家的做法一致：**AI 可给 + 人话兜底**。本判据四件事都钉：
    1. **每条命令都声明了 ``reason``**（框架规则：只在命令里声明过才保留，未声明会被剥掉）；
    2. **它会投影到 AI 侧工具面**（`CommandParam` → `core_commands` → 框架 `project`），
       否则模型想给也给不了；
    3. **调用方给的原话原样落进审计**（不被覆盖）；
    4. **没给时是人话兜底**，且**不含** `ml_mecha:` 那种机械前缀。
    """
    from mecha.commands import CommandSpec as CoreCommandSpec

    from ml_mecha.commands import COMMANDS
    from ml_mecha.core_commands import core_command_specs
    from ml_mecha.engine import _reason_fallback

    # 1. 声明齐
    missing = [c.name for c in COMMANDS
               if not any(p.name == "reason" for p in c.parameters)]
    assert not missing, f"这些命令没声明 reason ⇒ 模型说的话会被框架剥掉：{missing}"

    # 2. 投影到工具面（AI 看得到）
    cores = {s.name: s for s in core_command_specs()}
    assert cores, "拿不到核心命令声明"
    for name, s in cores.items():
        assert isinstance(s, CoreCommandSpec)
        props = (getattr(s, "parameters", {}) or {}).get("properties", {})
        assert "reason" in props, f"{name} 的 AI 工具面没有 reason ⇒ 模型给不了理由"

    # 3. 兜底是人话、且不是老那种机械串
    fb_eval = _reason_fallback("eval_custom", {"which": "best"})
    assert "ml_mecha:" not in fb_eval, f"兜底还是机械串：{fb_eval!r}"
    assert "评估" in fb_eval and "best" in fb_eval, fb_eval
    assert _reason_fallback("run_method", {"method": "ridge"}) == "运行方法 ridge"
    assert "2" in _reason_fallback("compare_methods", {"methods": ["a", "b"]}), \
        "批量类命令的兜底应带上数量"
    # 未知命令也要有兜底（不能 KeyError）
    assert _reason_fallback("没见过的命令", {}), "未知命令的兜底是空的"


def test_monitor_extra_routes_are_registered_and_core_routes_untouched():
    """ML 的附加路由经框架 ``add_route`` 注册，且不覆盖四个核心路由。

    ## 为什么这里钉的是**全集**（而不是"至少包含某几条"）

    多注册一条 = 面板/AI 多一个能读的面（**面是承诺**），漏注册一条 = 承诺了却调不到。
    两者都该被看见 ⇒ 钉全集，**有意增删时同步改这里并说明**。

    变更记录：
    * — → `{"/runs", "/summary"}`（迁移到框架 cockpit 时的两条附加页数据源）。
    * — → 加 `{"/latest", "/series", "/dashboard"}`（2026-10-02 TensorBoard 集成）：
      `/latest` 与 `/series` 给 AI 与人读训练数值（读事件文件）；
      `/dashboard` 起 TensorBoard 辅进程并回它的地址（**起不来会回可读 reason，不假装**）。
    """
    from ml_mecha.monitor_http import MonitorEndpoint

    expected = {"/runs", "/summary", "/latest", "/series", "/dashboard"}

    with tempfile.TemporaryDirectory(prefix="ml-delivery-routes-") as td:
        root = Path(td)
        with _authority(root, open_gate=True) as (ml, host):
            monitor = MonitorEndpoint(ml, mcp_port=host.port,
                                      port_file=root / ".ml-monitor-port")
            try:
                got = set(monitor.routes())
                assert got == expected, (
                    f"附加路由全集对不上。缺={sorted(expected - got)} "
                    f"多={sorted(got - expected)}（有意增删请同步改本判据并留变更记录）")
                # 重名 fail loud（核心路由不可覆盖；已注册的附加路由不可重注册）
                for path in ("/status", "/summary", "/latest", "/dashboard"):
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
    panel_data = (dsh_dir / "node_modules" / "@mecha" / "dsh-panel" / "panel" / "panel-data.ts").read_text(
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
    view = (REPO / "dsh" / "node_modules" / "@mecha" / "dsh-panel" / "panel" / "panel-view.ts").read_text(
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
    view_test = (REPO / "dsh" / "node_modules" / "@mecha" / "dsh-panel" / "panel" / "panel-view.test.ts").read_text(
        encoding="utf-8")
    assert "每条分支都返回**非空**可读 HTML" in view_test, \
        "资产自测里少了「每条分支都非空」那条（性质没被守）"
    # 附加页失败**不牵连**标准两页：取数把它们单独返回（不塞进 panelState 的 extra）
    data = (REPO / "dsh" / "node_modules" / "@mecha" / "dsh-panel" / "panel" / "panel-data.ts").read_text(
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
    pipeline = (REPO / "dsh" / "node_modules" / "@mecha" / "dsh-panel" / "panel" / "monitor-client.ts").read_text(
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


#: ⭐ **内省收集**（2026-10-03 改）：此前这里是**手写**的判据清单，而门禁用
#: **脚本**方式跑本文件 ⇒ 新判据忘了加进来就**永远不会进闸**。实锤：
#: `test_ml_delivery.py::test_tool_params_stay_within_command_declaration`
#: （"工具声明与实现漂移"那条）从未在门禁里跑过。
#: ⇒ 定义即入闸，漂移不可能。（前提：本文件的 `test_*` 都是无参函数。）
_TESTS = [fn for name, fn in sorted(globals().items())
          if name.startswith("test_") and callable(fn)]


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
