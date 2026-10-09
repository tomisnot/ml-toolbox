# -*- coding: utf-8 -*-
"""写权自省 `read_write_authority` 在**能力面**上的判据（真起 `--mcp` + 真 MCP 客户端）。

## 它守的是什么（设计 §7.1.1 的"三处"之①）

`read_write_authority` 要在三处出现：① **软件自己的能力面** ② 控制端点 ③ 聚合端点。
⚠ 设计专门警告：**最容易漏的是①** —— 只想着"面板要读"，就漏掉"**AI 自己也要能自查**"。
漏了的现象：AI 写被拒后**无法自查**、只能盲撞（框架的 hint 会把工具名指给 AI，
而表里没有它 ⇒ **等于白指**）。

## 为什么必须真起进程 + 真 MCP 客户端

"注册进能力面"这件事的**唯一证据是线上工具表里有它、且真能调通**。
用替身测 = 测我自己写的替身（本工程最贵的一课）。
⚠ 判据**必须断言能真调**（不只是"名字在清单里"）——名字在而调不动正是"列得出、调不动"。

## Known Limitations and Deferred Work

- ⚠ 慢（真进程 + 装配 ML 栈）；⚠ **不测**聚合端点那一处（那是框架的判据）。
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mecha.authority import READ_AUTHORITY_TOOL  # noqa: E402

_TIMEOUT = 240.0


def _wait_file(pf: Path, proc: subprocess.Popen, deadline_s: float) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if proc.poll() is not None:
            return False
        if pf.is_file() and pf.read_text(encoding="utf-8").strip():
            return True
        time.sleep(0.5)
    return False


class _Live:
    """真 `--mcp` 子进程 + 它的 MCP 端口（`with` 收尾）。"""

    def __init__(self, *, declared: str = "ai"):
        self.root = Path(tempfile.mkdtemp(prefix="ml-authread-"))
        self.log = self.root / "mcp.log"
        self.proc: subprocess.Popen | None = None
        self._declared = declared
        self._fh = None

    def __enter__(self) -> "_Live":
        env = dict(os.environ)
        env.update({"MECHA_INITIAL_MODE": self._declared, "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8"})
        #: ⚠ 日志落**文件**，不用 `stdout=PIPE`（管道满 ⇒ 现象像挂死）。
        self._fh = open(self.log, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, str(REPO / "app_entry.py"), "--mcp",
             "--root", str(self.root)],
            cwd=str(REPO), env=env, stdout=self._fh, stderr=subprocess.STDOUT)
        assert _wait_file(self.root / ".mcp-port", self.proc, _TIMEOUT), (
            "MCP 端口文件没出来；日志尾部="
            f"{self.log.read_text(encoding='utf-8', errors='replace')[-500:]!r}")
        port = int((self.root / ".mcp-port").read_text(encoding="utf-8").strip())
        self.url = f"http://127.0.0.1:{port}/mcp"
        return self

    def __exit__(self, *exc) -> bool:
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        finally:
            if self._fh is not None:
                self._fh.close()
            shutil.rmtree(self.root, ignore_errors=True)
        return False

    # ---- 真 MCP 往返 ----
    def tools(self) -> list[str]:
        async def go():
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
            async with streamable_http_client(self.url) as (r, w):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    return sorted(t.name for t in (await s.list_tools()).tools)
        return asyncio.run(asyncio.wait_for(go(), timeout=_TIMEOUT))

    def call(self, name: str, args: dict | None = None) -> tuple[bool, str]:
        """真调一个工具：返回 ``(is_error, 文本)``。"""
        async def go():
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
            async with streamable_http_client(self.url) as (r, w):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    res = await s.call_tool(name, args or {})
                    text = res.content[0].text if res.content else ""
                    return bool(getattr(res, "is_error", False)), str(text)
        return asyncio.run(asyncio.wait_for(go(), timeout=_TIMEOUT))


# ============================================================ ① 表里有它 + 真能调
def test_read_write_authority_is_on_the_capability_surface_and_callable():
    """⭐ 能力面上**有** `read_write_authority`，且**真能调通**（不是"列得出、调不动"）。"""
    with _Live(declared="ai") as live:
        names = live.tools()
        assert READ_AUTHORITY_TOOL in names, (
            f"能力面上没有 {READ_AUTHORITY_TOOL!r} ⇒ AI 被拒后无法自查"
            f"（设计 §7.1.1 最容易漏的那一处）：{names}")

        err, text = live.call(READ_AUTHORITY_TOOL)
        assert err is False, f"调不通（名字在而调不动）：{text[:300]}"
        got = json.loads(text)
        #: ⚠ 断言**真对象**回来的值（不是替身里写死的）：字段来自 authority_view
        for key in ("mode", "ai_can_write", "human_holds"):
            assert key in got, (key, got)
        assert got["mode"] == "ai", got
        assert got["ai_can_write"] is True and got["human_holds"] is False, got


# ============================================================ ② 只读，不给写权操作
def test_ai_surface_does_not_expose_write_authority_operations():
    """⛔ AI 的能力面上**只有 read**：`set_write_authority` / `open_manual_surface` 不在。

    ⚠ §7.1.1：**不许 AI 给自己解锁** ⇒ 那两个只在**控制端点**上。
    """
    with _Live(declared="ai") as live:
        names = set(live.tools())
        assert READ_AUTHORITY_TOOL in names, names
        for banned in ("set_write_authority", "open_manual_surface"):
            assert banned not in names, (
                f"{banned!r} 漏进了 AI 的能力面 ⇒ AI 能给自己解锁：{sorted(names)}")


def test_guard_catches_a_leaked_write_operation():
    """⭐ 红证：把控制操作塞进能力面，上面那条断言**必须能红**（R16：扰动真落上）。

    ⚠ 直接在**真 ToolRegistry** 上注册一个同名工具（用假 execute，不碰真写权），
    再跑**与上面完全同形**的那段判断 —— 它必须报出这个泄漏。
    """
    from mecha.authority import register_authority_read  # noqa: F401  (契约来源)
    from mecha.tools import ToolRegistry, define_tool

    reg = ToolRegistry()
    reg.register(define_tool(name="set_write_authority",
                             description="（红证用假工具）",
                             parameters={},
                             #: ⚠ 框架要求 `required` **非空**（消 D-15 的隐式契约键）
                             output_schema={"type": "object", "required": ["ok"]},
                             execute=lambda **_kw: {"ok": True}))

    def _leak_check(names: set[str]) -> list[str]:
        return [b for b in ("set_write_authority", "open_manual_surface")
                if b in names]

    leaked = _leak_check({s["name"] for s in reg.schemas()})
    assert leaked == ["set_write_authority"], (
        f"守卫抓不到泄漏 ⇒ 它是哑的：{leaked}")


# ============================================================ ③ 与启动者声明一致
@pytest.mark.parametrize("declared", ["ai", "human"])
def test_mode_read_matches_the_declared_initial_mode(declared: str):
    """⭐ 能力面读回的 `mode` == 启动者声明的 `MECHA_INITIAL_MODE`（不是猜的）。"""
    with _Live(declared=declared) as live:
        err, text = live.call(READ_AUTHORITY_TOOL)
        assert err is False, text[:300]
        got = json.loads(text)
        assert got["mode"] == declared, (
            f"声明了 {declared!r} 而读回 {got['mode']!r} ⇒ `MECHA_INITIAL_MODE`"
            "没被读/没被应用")


# ============================================================ ④ 闭环：被拒时的 hint
def test_rejection_hint_points_at_the_self_check_tool():
    """⭐ **闭环**：写被拒时给 AI 的 hint 指向 `read_write_authority`（"先查后写"）。

    ⚠ 这就是"能力面上必须有它"的**理由本身**：hint 指了一个表里没有的工具 = 白指。
    """
    with _Live(declared="human") as live:
        #: `human` 模式下 AI 侧**写必被拒**（`authority_mode_mismatch`）
        err, text = live.call("prepare_dataset",
                              {"source": {"kind": "synthetic",
                                          "task": "classification"}})
        assert err is True, f"human 模式下 AI 不该能写：{text[:300]}"
        assert READ_AUTHORITY_TOOL in text, (
            f"被拒的 hint 没指向 {READ_AUTHORITY_TOOL!r} ⇒ 自查闭环断了：{text[:400]}")
        #: 而且那个工具**真在**能力面上（否则 hint 是空头支票）
        assert READ_AUTHORITY_TOOL in live.tools()
