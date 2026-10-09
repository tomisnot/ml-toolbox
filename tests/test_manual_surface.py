# -*- coding: utf-8 -*-
"""ML **人手面（GUI 窗口开在软件进程内）** 的判据 —— ⚠ **真起 `--mcp --with-gui` 子进程**。

## 为什么必须真起进程（本工程最贵的一课）

"按需开窗"横跨**三样东西**：控制端点（服务器线程）→ 队列 → **Qt 主线程**。
用替身只能测到我自己的替身（框架侧曾"29 项零件判据全绿，而 web 面板从来起不来"）。
⇒ 真进程、真 `.control-port`、真 MCP 客户端、**真 Qt**（`QT_QPA_PLATFORM=offscreen`，
无头机器上也能跑真 Qt 事件循环）。

## 它钉住什么（每条都对着一个"会静默出错"的形态）

| 判据 | 防的是什么 |
|---|---|
| ① 起来后窗口是**关**的 | 防"启动即开窗"（用户裁决是**按需**开） |
| ② `open_manual_surface()` ⇒ 窗口**真开了** | 防"返回 ok 但不开窗"（**静默假成功**） |
| ③ 幂等：再调一次仍 ok，**窗口数还是 1** | 防"每次调用都新建窗口" |
| ④ 不带 `--with-gui` ⇒ **响亮失败** | 防"悄悄也能开"（形态是启动者选的） |
| ⑤ 开窗请求**记账**在同一本账 | 防"绕过桥直接改内部状态" |
| ⑥ **能力面看不到**那两个操作 | 防控制操作漏进 AI 的工具表 |
| ⑦ 切写权后**立刻生效**（AI 侧下一次写被拒） | 防"等一拍"的模糊语义 |

⚠ 红线（本工程踩过，写判据时必须避开）：
- **不许吞异常**：线程/回调里抛的错必须带出来（否则真因被盖掉、表现成"崩溃"）；
- **子进程日志落文件，不用 `stdout=PIPE`**（没人读会写满管道 ⇒ 卡在 `write`，
  现象像"挂死"，真因是管道满 —— 两个施工方独立踩过）；
- **红证要自证"扰动真落上了"**（R16），别只跑一次看到没报错就收工。

## Known Limitations and Deferred Work

- ⚠ 慢（真进程 + 装配 ML 栈）；⚠ **不测**窗口里的交互（那是 `test_ui.py` 的事）。
- ⚠ `QT_QPA_PLATFORM=offscreen` 的覆盖范围：证明**开窗这条路真的通到 Qt**，
  不断言"在真显示器上长什么样"。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mecha.software_control import CONTROL_PORT_FILE, SoftwareControl  # noqa: E402

_TIMEOUT = 240.0


def _wait_port_file(pf: Path, proc: subprocess.Popen, deadline_s: float) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if proc.poll() is not None:
            return False
        if pf.is_file() and pf.read_text(encoding="utf-8").strip():
            return True
        time.sleep(0.5)
    return False


class _Run:
    """一台真子进程 + 它的日志 + 控制端点客户端（`with` 收尾）。"""

    def __init__(self, *, with_gui: bool, declared: str = "ai"):
        self.root = Path(tempfile.mkdtemp(prefix="ml-surface-"))
        self.log = self.root / "mcp.log"
        self.proc: subprocess.Popen | None = None
        self._with_gui = with_gui
        self._declared = declared
        self._fh = None

    def __enter__(self) -> "_Run":
        env = dict(os.environ)
        env["MECHA_INITIAL_MODE"] = self._declared
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        #: ⚠ 真 Qt，但**跑到离屏平台**上（无头机器也能验"开窗这条路通不通"）。
        env["QT_QPA_PLATFORM"] = "offscreen"
        #: ⚠ **不带 `--no-open-gate`**：那一档会把写权**钉在 LOCKED**
        #: （它压过 `MECHA_INITIAL_MODE`）⇒ 判据 ⑦ 就测不出"切完立刻生效"。
        #: 这里要的是"按声明装配 + 运行中可切"。
        cmd = [sys.executable, str(REPO / "app_entry.py"), "--mcp",
               "--root", str(self.root)]
        if self._with_gui:
            cmd.append("--with-gui")
        #: ⚠⚠ 日志落**文件**，绝不用 `stdout=PIPE`（管道满 ⇒ 现象像挂死）。
        self._fh = open(self.log, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                                     stdout=self._fh, stderr=subprocess.STDOUT)
        pf = self.root / CONTROL_PORT_FILE
        assert _wait_port_file(pf, self.proc, _TIMEOUT), (
            "`.control-port` 没出来；日志尾部="
            f"{self.log.read_text(encoding='utf-8', errors='replace')[-600:]!r}")
        self.ctl = SoftwareControl.connect("ML", self.root)
        avail, why = self.ctl.available()
        assert avail, f"控制端点连不上：{why}"
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

    # ---- 事实面 ----
    def log_text(self) -> str:
        try:
            return self.log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def monitor_port(self) -> int:
        return int((self.root / ".ml-monitor-port").read_text(encoding="utf-8").strip())

    def history(self) -> list[dict]:
        url = f"http://127.0.0.1:{self.monitor_port()}/history?since_seq=0"
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.loads(r.read().decode("utf-8")).get("events", [])


# ============================================================ ① + ② + ③ + ⑤
def test_with_gui_opens_window_on_demand_and_is_idempotent():
    """⭐ 按需开窗：起来是**关**的 → 调用后**真开** → 再调**幂等**；且开窗**记账**。"""
    with _Run(with_gui=True) as run:
        #: ① ⭐ 起来后**窗口是关的**（按需开，不是启动就开）
        #:    证据：进程活着 + 日志里有"人手面：**按需开**（尚未开窗）"
        log = run.log_text()
        assert "按需开" in log and "尚未开窗" in log, (
            f"启动时不该开窗（日志应写明'按需开'）：{log[-500:]!r}")

        #: ② 调用 ⇒ 窗口**真开了**
        got = run.ctl.open_manual_surface()
        assert got.get("ok") is True, got
        assert got.get("status") == "opened", f"第一次应当 opened，实得 {got!r}"

        #: ③ 幂等：再调一次**不报错**，且**不重复开**（窗口数还是 1）
        again = run.ctl.open_manual_surface()
        assert again.get("ok") is True, again
        assert again.get("status") == "already-open", (
            f"第二次应当 already-open（幂等），实得 {again!r} —— 每次都新建窗口？")

        #: ⑤ 两个操作都**记账**在同一本账上（同门同账），且**状态串如实**：
        #: `after` 依次是 `opened` / `already-open` ⇒ 一条判据同时证明
        #: "真的开了一次" + "第二次是幂等而不是又开一个"。
        #: ⚠ 为什么不用日志行计数：客户端返回时主线程的 print **可能还没刷出来**
        #:   （竞态）⇒ 那是 flaky。账是**同步**写下的，断言它才稳。
        events = [e for e in run.history()
                  if str(e.get("op")).startswith("control.open_manual_surface")]
        assert [e.get("after") for e in events] == ["opened", "already-open"], (
            f"开窗的状态串不对（应当 exactly opened → already-open）：{events}")


# ============================================================ ④
def test_without_the_flag_open_surface_fails_loudly():
    """⭐ 不带 `--with-gui` ⇒ `open_manual_surface()` **响亮失败**（不假装开了窗）。

    ⚠ 这条**也是红证的对偶**：证明"形态是启动者选的"，不是"总能开"。
    """
    with _Run(with_gui=False) as run:
        with pytest.raises(Exception) as ei:
            run.ctl.open_manual_surface()
        assert "no_manual_surface" in str(ei.value) or "手动面" in str(ei.value), (
            f"应当响亮失败（没有手动面），实得：{ei.value}")


# ============================================================ ⑥
def test_control_ops_are_not_on_the_ai_capability_surface():
    """⭐ AI 的能力面上**看不到**那两个控制操作（防线=地址只给框架）。

    ⚠ 靠"没人告诉 AI 那个地址"是**薄**的；这条把"真没漏进工具表"钉住。
    """
    import asyncio

    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    with _Run(with_gui=True) as run:
        #: 工具面在 MCP 端口上（控制端点是**另一个**端口）
        mcp_port = int((run.root / ".mcp-port").read_text(encoding="utf-8").strip())
        url = f"http://127.0.0.1:{mcp_port}/mcp"

        async def names():
            async with streamable_http_client(url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return sorted(t.name for t in (await session.list_tools()).tools)

        got = asyncio.run(asyncio.wait_for(names(), timeout=_TIMEOUT))
        for banned in ("set_write_authority", "open_manual_surface",
                       "read_write_authority"):
            assert banned not in got, (
                f"控制操作 {banned!r} 漏进了 AI 的能力面：{got}")


# ============================================================ ⑦
def test_switch_write_authority_takes_effect_immediately():
    """⭐ 切写权**立刻生效**：切 human 后 AI 侧下一次写**当场被拒**；切回反向。

    ⚠ "等一拍"式的模糊语义在这里会被抓住：切完立刻用 MCP 面调一个**写工具**。
    """
    import asyncio

    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    with _Run(with_gui=True, declared="ai") as run:
        mcp_port = int((run.root / ".mcp-port").read_text(encoding="utf-8").strip())
        url = f"http://127.0.0.1:{mcp_port}/mcp"

        async def prepare():                     # 一个会真写的调用
            async with streamable_http_client(url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    res = await session.call_tool(
                        "prepare_dataset", {"source": {"kind": "synthetic",
                                                       "task": "classification"}})
                    return bool(getattr(res, "is_error", False)), str(res.content)[:200]

        err, txt = asyncio.run(asyncio.wait_for(prepare(), timeout=_TIMEOUT))
        assert err is False, f"AI 侧本该能写：{txt}"

        #: 切成 human ⇒ AI 侧**下一次写当场被拒**（不重连、不等一拍）
        run.ctl.set_write_authority("human")
        assert run.ctl.read_write_authority().get("mode") == "human"
        err2, txt2 = asyncio.run(asyncio.wait_for(prepare(), timeout=_TIMEOUT))
        assert err2 is True, f"切到 human 后 AI 侧还能写 ⇒ 没立刻生效：{txt2}"
        assert "authority" in txt2 or "写权" in txt2, txt2

        #: 切回 ⇒ 又能写（反向同理）
        run.ctl.set_write_authority("ai")
        err3, txt3 = asyncio.run(asyncio.wait_for(prepare(), timeout=_TIMEOUT))
        assert err3 is False, f"切回 ai 后仍写不了：{txt3}"
