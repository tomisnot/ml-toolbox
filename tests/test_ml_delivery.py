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
MECHA_ROOT = Path(os.environ.get("MECHA_ROOT", r"D:\code-nosync\mecha"))
if MECHA_ROOT.is_dir() and str(MECHA_ROOT) not in sys.path:
    sys.path.insert(0, str(MECHA_ROOT))

from mcp import ClientSession                                   # noqa: E402
from mcp.client.streamable_http import streamable_http_client   # noqa: E402

from ml_mecha.assembly import assemble_ml_mecha                 # noqa: E402
from ml_mecha.mcp_host import McpHost, build_mcp_server         # noqa: E402

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

            # 可取消的 job：cancel_run 受理并给出请求前后状态（不谎报"已取消"）
            ok_job = ml.submit({"command": "run_method", "method": "logistic",
                                "dataset_id": ml.engine.dataset_ids()[-1]}, side="ai")
            err, txt = _call_tools(host.url, [("cancel_run", {"job_id": ok_job.id})])[0]
            assert err is False, txt
            body = json.loads(txt)
            assert body["cancel_requested"] is True, body
            assert "state_before" in body and "state" in body, body


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
    不是"有响应就算过"：`/history` 必须含 `seq/kind/actor/target/value/before/
    after/reason/call_id`，且 `kind == "set"`、`before` 由全史 fold 重建；
    `/config` 必须 `has_schema=False` + 快照键全进 `orphans`（**不许造树**）。
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
                    assert set(ev) == {"seq", "kind", "actor", "target", "value",
                                       "before", "after", "reason", "call_id"}, ev
                    assert ev["kind"] == "set", ev
                    assert ev["after"] == ev["value"], ev
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
                assert isinstance(runs["runs"], list), runs

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
    src_files = [p for p in (dsh_dir / "src").rglob("*")
                 if p.suffix in (".ts", ".tsx")]
    assert src_files, "dsh/src 不存在或为空——面板源码丢了？"

    code = "\n".join(_strip_js_comments(p.read_text(encoding="utf-8"))
                     for p in src_files)
    for token in ("re0", "机甲", "Rb-87", "8767", "mecha_v2", "energy-level",
                  "Energy Level", "MONITOR_URL"):
        assert token not in code, f"dsh 源码（去注释后）仍含 EL 专属字样 {token!r}"

    client_code = _strip_js_comments(
        (dsh_dir / "src" / "client" / "index.ts").read_text(encoding="utf-8"))
    assert not re.search(r"127\.0\.0\.1:\d+", client_code), \
        "客户端里出现了硬编码的 loopback 端口"
    assert "resolveMonitorBase" in client_code, "监控地址必须经注入的解析器取"
    assert "/ml-monitor-url" in client_code, "必须经 host 半命令取地址"

    body = (dsh_dir / "src" / "client" / "MonitorTabBody.tsx").read_text(
        encoding="utf-8")
    for label in ("飞行记录仪", "配置态", "运行记录"):
        assert label in body, f"面板缺少页签 {label!r}"


_TESTS = [
    test_required_from_signature_and_declaration_survives,
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
    test_generated_dsh_patch_composes_with_real_dsh,
    test_dsh_panel_source_has_no_el_leftovers_and_no_hardcoded_port,
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
