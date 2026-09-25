# -*- coding: utf-8 -*-
"""ML Toolbox 看门人（launcher）：两模式启动与交接的**唯一编排者**。

心智模型（见 ``docs/mecha/07-交付形态-双模式与dsh桥.md``）：

- **本地 GUI 模式**：纯本地软件（``app_entry.py --gui``），写权在人，不起任何端点。
- **AI 模式**：headless 权威（``app_entry.py --authority``，写权自动落 AI）
  + dsh 前台（AI 界面 + 驾驶舱监控面板）。

看门人常驻（无窗口，日志进本控制台），职责：

1. 启动弹窗选一次模式 → spawn 对应进程组。
2. AI 模式：等权威就绪（**D13 双条件**）→ 生成 dsh overlay → 起 dsh。
3. 轮询 ``.mode-request``（任一子进程写它 = 切换意图）→ 交接：terminate 当前组、
   **等进程真退出**（写租约随进程释放，否则新权威被 ``writer_lease_held`` 拒）、
   按新模式 spawn、清请求文件。
4. 主进程自然退出且无切换请求 → 收尾退出。

## 就绪判据（D13：**不照抄** EL 的"只判端口文件存在"）

Windows 上 ``terminate()`` / ``taskkill /F`` **不跑** Python 的 ``finally``，强杀后
``.mcp-port`` 会残留；只判"文件存在"会让看门人立刻去起 dsh 并连**已死的端口**
——"起得来但看不见"。三条一起用：

1. spawn 前**快照**端口文件（内容 + mtime）；
2. 轮询到**新鲜**（文件先消失过 / mtime 变新 / 内容变化）**且** ``/mcp`` 上有 HTTP
   响应（TCP 连得上还不够——同端口上可能是别的东西在听）；
3. 超时（``READY_TIMEOUT``）→ **明确报错且不 spawn dsh**（不静默退回默认端口）。

探活用 stdlib（``urllib``）而非 MCP SDK：看门人不耦合 SDK；完整 MCP ``initialize``
握手由交付判据的真客户端负责（``tests/test_ml_delivery.py``）。

## 运行期文件

文件名与清理规则收在 :mod:`ml_mecha.runtime`（**单一真源**），本模块不重抄字面量。
dsh 的 overlay 是**生成物**（``runtime/dsh-ml-mcp.patch.yml``，每次启动按实际端口
重写，不入库）。
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from ml_mecha.runtime import (MCP_PORT_FILE, MODE_REQUEST, MODE_STATE, RUNTIME_JSON,
                              clear_port_file, clear_runtime, read_port)
from ml_mecha.runtime import write_runtime as _write_runtime

HERE = Path(__file__).resolve().parent
PY = sys.executable
APP = str(HERE / "app_entry.py")

MODE_REQUEST_FILE = HERE / MODE_REQUEST
MODE_STATE_FILE = HERE / MODE_STATE
RUNTIME_FILE = HERE / RUNTIME_JSON
MCP_PORT_PATH = HERE / MCP_PORT_FILE
#: 生成的 dsh overlay（不入库；每次启动重写）。
DSH_PATCH_FILE = HERE / "runtime" / "dsh-ml-mcp.patch.yml"
#: 入库的插件 overlay（加载驾驶舱面板插件；相对本文件解析 `./lib/index.mjs`）。
DSH_PLUGIN_PATCH = HERE / "dsh" / "cordis.patch.yml"
#: 插件的构建产物：**没有它就不能挂插件 overlay**（否则 dsh 加载失败）。
DSH_PLUGIN_BUNDLE = HERE / "dsh" / "lib" / "index.mjs"

GUI, AI = "gui", "ai"

#: 等权威就绪的上限（秒）：装配 + cache + 起 uvicorn，装配要十几秒。
READY_TIMEOUT = 90.0
#: 交接时等旧进程真退出的上限（秒）；超时才 kill（防租约没放干净）。
EXIT_TIMEOUT = 15.0
#: MCP 单次工具调用的超时（dsh 客户端侧）。10 分钟：长训练不假失败，
#: 又不至于让"卡住"看起来像在跑（07 §5 R2 / D4）。
TOOL_CALL_TIMEOUT_MS = 600000

_FLAGS = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0


def _log(msg: str) -> None:
    print("[launcher] %s" % msg, flush=True)


# ---------------------------------------------------------------- 进程组
def _spawn(cmd) -> subprocess.Popen:
    """起一个子进程，输出继承本控制台（权威日志/dsh 进度可见）。"""
    return subprocess.Popen(cmd, cwd=str(HERE), creationflags=_FLAGS)


def start_group(mode: str):
    """按模式 spawn 进程组，返回 ``(procs, primary, ready)``。

    ``primary`` = 主进程索引：它自然退出且无切换请求 ⇒ 看门人收尾。
    ``ready`` = 就绪信息 dict（AI 模式下含权威 MCP 端口）。
    """
    if mode == GUI:
        return [_spawn([PY, APP, "--gui"])], 0, {}
    authority = _spawn([PY, APP, "--authority"])
    ok, port, reason = await_authority(authority, MCP_PORT_PATH)
    if not ok:
        # 不静默降级：权威没就绪就不起 dsh，并把原因说清楚（D13 第 3 条）。
        _log("⚠ 权威未就绪：%s" % reason)
        return [authority], 0, {"ready": False, "reason": reason}
    dsh = _spawn_dsh(port)
    procs = [authority] if dsh is None else [authority, dsh]
    # primary：有 dsh 时是 dsh（关掉 AI 界面即结束整组）；缺 dsh 时退回权威。
    return procs, (1 if dsh is not None else 0), {"ready": True, "mcp_port": port}


def dsh_patch_text(mcp_port: int) -> str:
    """生成的 dsh overlay 文本。

    **必须是 ``- insert:`` 形态**：Lead 实测过裸 ``- id: mcp-mltoolbox`` 会被 dsh
    判 ``patch: entry "mcp-mltoolbox" not found`` 并**静默跳过**（组合树里没有它）。
    ``toolCallTimeoutMs`` 显式给 10 分钟：默认 60 s 会让长训练"假失败"。
    ``failOnStartupError: true`` 让"连不上权威"变响，而不是留一个空工具表。
    """
    return (
        "# 由 launcher.py 生成（每次启动按实际端口重写）——**不入库**。\n"
        "# 用法：dsh --profile web --patch <本文件> --port <dsh 端口>\n"
        "# 注意：patch 条目必须用 `- insert:` 包住；裸 id 会被判「目标不存在」并跳过。\n"
        "- insert:\n"
        "    - id: mcp-mltoolbox\n"
        "      name: '@deepseek-ai/dsh-mcp-client'\n"
        "      config:\n"
        "        serverName: mltoolbox\n"
        "        transport: streamable-http\n"
        "        url: http://127.0.0.1:%d/mcp\n"
        "        failOnStartupError: true\n"
        "        toolCallTimeoutMs: %d\n" % (int(mcp_port), TOOL_CALL_TIMEOUT_MS)
    )


def write_dsh_patch(path: str | Path, mcp_port: int) -> Path:
    """写生成的 overlay（父目录按需创建）；返回写入路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dsh_patch_text(mcp_port), encoding="utf-8")
    return target


def dsh_overlays(mcp_port: int) -> list[str]:
    """起 dsh 要带的两份 overlay（顺序：插件 → 工具接入）。

    插件 overlay 只在**构建产物存在**时带上：`dsh/cordis.patch.yml` 指向
    ``./lib/index.mjs``，没构建过就不能把它塞给 dsh（否则插件加载失败）。
    降级要说清楚，不静默少挂东西。
    """
    patches: list[str] = []
    if DSH_PLUGIN_PATCH.exists() and DSH_PLUGIN_BUNDLE.exists():
        patches.append(str(DSH_PLUGIN_PATCH))
    else:
        _log("（面板插件未构建：先 `cd dsh && npm install && npm run bundle`；"
             "本次不带面板 overlay，AI 模式照常可用，只是没有右栏监控面板）")
    try:
        patches.append(str(write_dsh_patch(DSH_PATCH_FILE, mcp_port)))
    except OSError as exc:
        _log("⚠ 生成 dsh overlay 失败（%s）；只用已有 overlay。" % exc)
    return patches


def _spawn_dsh(mcp_port: int):
    """起 dsh（AI 界面 + 面板）：生成 overlay → 挑端口 → ``dsh --profile web ...``。

    ``dsh`` 不在 PATH 时**清晰降级**：只打日志并返回 ``None``（AI 模式仍可用，
    只是没有 AI 界面——比静默起不来强）。
    """
    dsh_exe = shutil.which("dsh")
    if not dsh_exe:
        _log("⚠ PATH 里没找到 dsh —— AI 模式只有权威在跑，没有 AI 界面/监控面板。")
        _log("  手动运行：dsh --profile web --patch %s --port <端口>" % DSH_PATCH_FILE)
        return None
    patches = dsh_overlays(mcp_port)
    port = pick_dsh_port()
    cmd = [dsh_exe, "--profile", "web"]
    for patch in patches:
        cmd += ["--patch", patch]
    cmd += ["--port", str(port)]
    _log("起 dsh（本实例 :%d，避开官方 3080）；overlay=%s"
         % (port, " + ".join(patches) or "(无)"))
    return _spawn(cmd)


# ---------------------------------------------------------------- 就绪判据
def _port_snapshot(port_file: Path):
    """``(port|None, mtime|None)``——供 D13 的"新鲜度"判定（读端口用 runtime）。"""
    try:
        mtime = Path(port_file).stat().st_mtime
    except OSError:
        return None, None
    return read_port(port_file), mtime


def _probe_http(host: str, port: int, timeout: float = 1.0) -> bool:
    """``/mcp`` 上有 HTTP 响应即算在听（4xx 也是响应；连不上/超时不算）。

    TCP 连得上还不够——同一端口上可能是别的东西在听。这里用 stdlib 做一次
    真实 HTTP 请求；MCP 层握手（``initialize``）由交付判据的真客户端验。
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError:
        return False
    try:
        urllib.request.urlopen("http://%s:%d/mcp" % (host, port), timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True                      # 有 HTTP 响应 = 服务在听
    except Exception:                    # noqa: BLE001 - 网络层的任何失败都算没就绪
        return False


def await_authority(proc, port_file: Path, *, timeout: float = READY_TIMEOUT,
                    host: str = "127.0.0.1"):
    """等权威就绪：端口文件**新鲜** + ``/mcp`` 有 HTTP 响应。

    返回 ``(ok, port, reason)``；``ok=False`` 时 ``reason`` 是给人看的短句。
    """
    before_port, before_mtime = _port_snapshot(port_file)
    saw_missing = before_port is None
    deadline = time.time() + max(1.0, float(timeout))
    last = "尚未写入端口文件"
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False, None, "权威进程已退出（exit=%s）" % proc.returncode
        port, mtime = _port_snapshot(port_file)
        if port is None:
            saw_missing = True
            last = "尚未写入端口文件"
        else:
            fresh = (saw_missing or mtime != before_mtime or port != before_port)
            if not fresh:
                last = "端口文件还是上一次的旧值（%s）" % port
            elif _probe_http(host, port):
                return True, port, ""
            else:
                last = "端口 %d 上还没有 HTTP 响应" % port
        time.sleep(0.25)
    return False, None, "等权威就绪超时（%.0fs）：%s" % (timeout, last)


# ---------------------------------------------------------------- dsh 端口
def pick_dsh_port(preferred: int = 3081, tries: int = 10) -> int:
    """从 ``preferred`` 起找第一个可绑定回环端口；**3080 硬跳过**。

    3080 是官方 dsh 的端口，项目实例永不占（注释即机制，不靠默认值兜底）。
    """
    for port in range(preferred, preferred + tries):
        if port == 3080:
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
    return preferred


# ---------------------------------------------------------------- 交接
def _terminate_group(procs) -> None:
    for p in procs:
        if p is None:
            continue
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:                                  # noqa: BLE001
            pass


def await_exit(procs, timeout: float = EXIT_TIMEOUT) -> None:
    """等进程**真退出**（写租约随进程释放）。超时才 kill。"""
    deadline = time.time() + timeout
    for p in procs:
        if p is None:
            continue
        while p.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if p.poll() is None:
            try:
                p.kill()
            except Exception:                              # noqa: BLE001
                pass


def read_request():
    """读切换意图；非法内容返回 ``None``（不猜）。"""
    try:
        if MODE_REQUEST_FILE.exists():
            value = MODE_REQUEST_FILE.read_text(encoding="utf-8").strip().lower()
            return value if value in (GUI, AI) else None
    except OSError:
        pass
    return None


def clear_request() -> None:
    try:
        if MODE_REQUEST_FILE.exists():
            MODE_REQUEST_FILE.unlink()
    except OSError:
        pass


def write_state(mode: str) -> None:
    try:
        MODE_STATE_FILE.write_text(mode, encoding="utf-8")
    except OSError:
        pass


def write_runtime(**fields) -> dict:
    """写运行期描述符（合并语义：权威写的端口不会被看门人抹掉）。"""
    return _write_runtime(HERE, **fields)


# ---------------------------------------------------------------- 模式选择
def choose_mode():
    """弹窗选模式。PyQt5 **惰性导入**——本模块被 import（判据）时不得碰 Qt。"""
    from PyQt5.QtWidgets import QApplication, QMessageBox

    app = QApplication.instance() or QApplication(sys.argv)
    box = QMessageBox()
    box.setWindowTitle("ML Toolbox · 选择启动模式")
    box.setIcon(QMessageBox.Question)
    box.setText("请选择本次启动的模式：")
    box.setInformativeText(
        "AI 模式：软件在后台当权威（写权自动交给 AI），前台起 AI 界面——\n"
        "你在对话里驱动 ML 工具，右侧只读监控看它每笔操作；想收回控制权就把\n"
        "控制权切回 GUI 模式。\n\n"
        "本地 GUI 模式：纯本地软件，写权在你手上，AI 完全看不见这台软件；\n"
        "想用 AI 就经看门人切到 AI 模式。\n\n"
        "两种模式共用同一份实验记录（runs/），切换时状态原样交接、不断档。")
    ai_button = box.addButton("AI 模式", QMessageBox.AcceptRole)
    gui_button = box.addButton("本地 GUI 模式", QMessageBox.ActionRole)
    box.addButton("退出", QMessageBox.RejectRole)
    box.setDefaultButton(gui_button)
    box.exec_()
    clicked = box.clickedButton()
    if clicked is ai_button:
        return AI
    if clicked is gui_button:
        return GUI
    return None


# ---------------------------------------------------------------- 主循环
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    clear_request()
    # 前一次被强杀留下的端口文件：权威自己也会清，这里先清一次让 D13 的
    # "文件消失过"信号从一开始就是干净的。
    clear_port_file(MCP_PORT_PATH)
    mode = choose_mode()
    if mode is None:
        _log("未选模式，退出。")
        return 0
    _log("启动 %s 模式。" % ("GUI（人在开）" if mode == GUI else "AI（AI 在开）"))
    procs, primary, ready = start_group(mode)
    write_state(mode)
    write_runtime(mode=mode,
                  authority_pid=(procs[0].pid if mode == AI and procs else None),
                  mcp_port=ready.get("mcp_port"), dsh_port=pick_dsh_port())
    if mode == AI and not ready.get("ready"):
        _log("AI 模式未能就绪（%s）；收尾。" % ready.get("reason"))
        _terminate_group(procs)
        await_exit(procs, timeout=8)
        return 1

    try:
        while True:
            time.sleep(0.5)
            requested = read_request()
            if requested and requested != mode:
                _log("收到切换请求：%s → %s，交接中…" % (mode, requested))
                _terminate_group(procs)
                await_exit(procs)
                clear_request()
                procs, primary, ready = start_group(requested)
                mode = requested
                write_state(mode)
                write_runtime(mode=mode,
                              authority_pid=(procs[0].pid
                                             if mode == AI and procs else None),
                              mcp_port=ready.get("mcp_port"),
                              dsh_port=pick_dsh_port())
                _log("已切到 %s 模式。" % mode)
                continue
            prim = procs[primary] if primary < len(procs) else procs[0]
            if prim is not None and prim.poll() is not None:
                _log("主进程已退出（%s 模式结束），收尾。" % mode)
                break
    except KeyboardInterrupt:
        _log("\nCtrl+C —— 收尾所有子进程。")
    finally:
        _terminate_group(procs)
        await_exit(procs, timeout=8)
        clear_request()
        try:
            if MODE_STATE_FILE.exists():
                MODE_STATE_FILE.unlink()
        except OSError:
            pass
        clear_runtime(HERE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
