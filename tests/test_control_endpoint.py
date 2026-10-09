# -*- coding: utf-8 -*-
"""ML 的**控制端点** + **启动者声明**判据 —— ⚠ **真起 `--mcp` 子进程**，不用替身。

## 为什么必须真起进程（本工程最贵的一课）

控制端点与 env 读取**都在软件进程内**。用替身测 = "测我自己写的替身"：
框架侧曾出现"**29 项零件判据全绿，而那个 web 面板从来起不来**"
（判据只调 `render_page(stub)`，替身还与真实现不同形）。
⇒ 所以这一条走真路：真进程 → 真 `.control-port` → **真 MCP 客户端**调三个操作。

## 它钉住什么

1. ⭐ `.control-port` 真出来（框架的 `start_control_endpoint` 接对了）；
2. ⭐ **`MECHA_INITIAL_MODE` 真被读并应用**：给 `human` ⇒ 软件起来后写权就是 `human`
   （⚠ 这条直接钉住"声明了却不读"那个坑）；
3. ⭐ 三个操作都在，`set_write_authority` 切完**立刻生效**；
4. ⭐ `open_manual_surface` **响亮失败**（ML 的 GUI 不在本进程 ⇒ `open_surface=None`）
   ⇒ 不假装开了窗。

## Known Limitations and Deferred Work

- ⚠ 慢（要起真进程、装配 ML 栈，几十秒）⇒ **别放进秒级档**。
- ⚠ 只测"控制端点 + 初始写权"，**不测**跑训练（那是别的判据的事）。
- ⚠ 本判据**不改功能代码**：它只读 `.control-port` 并调框架客户端。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mecha.software_control import CONTROL_PORT_FILE, SoftwareControl  # noqa: E402
from mecha.site import MECHA_MODE_ENV  # noqa: E402


def _wait_port_file(pf: Path, proc: subprocess.Popen, deadline_s: float) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if proc.poll() is not None:
            return False
        if pf.is_file() and pf.read_text(encoding="utf-8").strip():
            return True
        time.sleep(0.5)
    return False


@pytest.mark.parametrize("declared", ["human", "ai"])
def test_control_endpoint_and_declared_mode(declared: str) -> None:
    """⭐ 真起 `--mcp`：`.control-port` 出来、三个操作可用、**初始写权按声明装配**。"""
    root = Path(tempfile.mkdtemp(prefix="ml-ctl-"))
    pf = root / CONTROL_PORT_FILE
    env = dict(os.environ)
    env[MECHA_MODE_ENV] = declared
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = None
    log = root / "mcp.log"
    try:
        #: ⚠⚠ **日志必须落文件，不能用 `stdout=PIPE`**（实测踩过）：
        #: 判据正阻塞在 MCP 调用上、没人读管道 ⇒ 子进程**日志写满管道**、卡在 `write`
        #: ⇒ 客户端看到的是"**协议错** / TaskGroup 异常"，**现象像"控制端点挂死"**
        #: —— 而真因是**管道满**。（EL 侧施工方独立踩过同一条，两边都记档。）
        with open(log, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen(
                [sys.executable, str(REPO / "app_entry.py"), "--mcp", "--root", str(root)],
                cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT)
            ok = _wait_port_file(pf, proc, deadline_s=180.0)
            assert ok, ("`.control-port` 没出来（控制端点没起来）；"
                        f"日志尾部={log.read_text(encoding='utf-8', errors='replace')[-400:]!r}")

        ctl = SoftwareControl.connect("ML", root)
        avail, why = ctl.available()
        assert avail, f"控制端点连不上：{why}"

        #: ① ⭐ 三个操作都在，且**初始写权 = 启动者声明的那个**
        #:   （⚠ 这条就是"声明了却不读"那个坑的判据）
        got = ctl.read_write_authority()
        assert got.get("mode") == declared, (
            f"初始写权应为声明的 {declared!r}，实得 {got.get('mode')!r} —— "
            "`MECHA_INITIAL_MODE` 没被读/没被应用")

        #: ② 切写权**立刻生效**（再读回来必须变）
        other = "ai" if declared == "human" else "human"
        ctl.set_write_authority(other)
        assert ctl.read_write_authority().get("mode") == other, "切完没立刻生效"
        ctl.set_write_authority(declared)          # 还原
        assert ctl.read_write_authority().get("mode") == declared

        #: ③ ⭐ `open_manual_surface` **响亮失败**（ML 的 GUI 不在本进程 ⇒ open_surface=None）
        with pytest.raises(Exception) as ei:
            ctl.open_manual_surface()
        assert "no_manual_surface" in str(ei.value) or "手动面" in str(ei.value), (
            f"应当**响亮失败**（没有手动面），实得：{ei.value}")
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        import shutil
        shutil.rmtree(root, ignore_errors=True)
