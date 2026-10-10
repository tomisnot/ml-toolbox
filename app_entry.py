# -*- coding: utf-8 -*-
"""ML Toolbox 的两个**软件入口**：本地 GUI / MCP 服务端点。

⚠ **本文件不编排任何东西**（不起 dsh、不等别人、不选模式）。
进程编排整块归 mecha 的统一启动器（supervisor）：声明在 ``ml_mecha/mecha_site.py``，
入口是 ``python -m mecha``。这里只负责"起本软件自己的一个形态"。

- ``--gui``  本地 GUI：走 ``app.py`` 的路径（不订阅 MCP、不起端点、不持写租约）。
             ⚠ 三层架构下"人那条腿"是**软件进程内的窗口**（不是 supervisor 的子进程）：
             今天 ``--gui`` 起的还是**另一个进程**，把它并进 ``--mcp`` 那个进程
             属**后续步骤**（见 `docs/v2/软件-交互-写权-三层架构.md` §3 裁决 1）。
- ``--mcp``  起 MCP 服务端点（原 ``--authority``）：headless 装配 ``assemble_ml_mecha``
             （出厂 ``Mode.LOCKED``）→ 起 MCP 服务（同进程后台线程）+
             只读监控端点 → 人类侧开闸 ``switch_ai()`` → 阻塞到 Ctrl+C / 被 terminate。
             ⚠ 名字说的是"**对外提供什么**"，不是"在编排里扮什么角色"。

## 两条实现纪律（都不是风格问题）

1. **PyQt5 只在 ``--gui`` 分支里惰性导入**。``app.py`` 记录的导入顺序铁律是
   "lightgbm（经 methods 顶层导入）必须先于任何 PyQt5 模块，否则其 OpenMP 与
   Qt 冲突 → fit 时 access violation"。``--mcp`` 是长跑训练进程，绝不能
   因为"顺手 import 了一下 Qt"把 native 初始化顺序搞坏，所以本模块**顶层不
   import app、不 import PyQt5**。
2. **写权落位是"启动者的意图"**，不是 AI 自解锁：装配出厂
   ``LOCKED``（mecha 的 "nothing ships enabled"），起完服务后由**人类侧**
   调 ``switch_ai()``——打开闸门的动作发生在启动序列里，AI 面永远拿不到
   ``side='human'`` 的通道（``mecha/authority.py::switch_mode`` 的硬判定）。

无参数运行时**明确报错**并指向 ``python -m mecha``，**不静默降级成 GUI**
（那会让人以为"AI 没生效"是个 bug）。
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
from pathlib import Path

#: 本文件所在目录 = 项目根。``.ml-mecha``（journal + 写租约）与 ``.mcp-port``
#: 都挂在这里 —— ⚠ **"同源"这条关系仍然成立**（发现文件、写租约、数据根必须在
#: 同一个根上，否则"起得来但看不见"），只是**机制换了**：
#: 今天这个根由框架按声明传进来（``ml_mecha/mecha_site.py`` 的 ``data_root`` /
#: ``cwd``），不再由项目的看门人自己编排（旧 ``launcher.py`` 已删）。
PROJ_ROOT = Path(__file__).resolve().parent

_USAGE = """用法：
  python app_entry.py --gui                     本地 GUI（= python app.py）
  python app_entry.py --mcp [选项]              起 MCP 服务端点（headless；不给 dsh 编排）

⚠ 进程编排（起谁、等谁、收谁）**已整块归 mecha 的 supervisor**
  ⇒ 用 `python -m mecha` 起整站；本文件只负责"起一个软件自己的端点"。
  ⇒ 声明在哪：`ml_mecha/mecha_site.py`

--mcp 选项：
  --root DIR            数据根（默认本文件所在目录）；``.ml-mecha`` 与 ``.mcp-port`` 落此
  --host HOST           绑定地址（默认 127.0.0.1，仅回环）
  --port N              MCP 端口；0 = 由系统分配（默认 0）
  --port-file PATH      端口文件（默认 <root>/.mcp-port）
  --dataset-roots DIR   允许 AI 读取的数据目录（可重复）；不给则 csv 来源 fail closed
  --no-open-gate        起来后不开闸（保持 LOCKED，供判据/维护用）
  --with-gui            ⭐ **本进程带"人面"**：收到 `open_manual_surface()`（控制端点）
                        才把 GUI 窗口开出来（**按需开，不是启动就开**）。
                        ⚠ 不带它 ⇒ 控制端点传 `open_surface=None` ⇒ 调它**响亮失败**。
"""


class _ManualSurface:
    """"手动面" = **本进程内的那个 GUI 窗口**（三层架构 A1/A2：人那条腿不是子进程）。

    ## 线程模型（⚠ 这里是最容易挂死的地方）

    控制请求到达的是 **MCP 服务器线程**，而 **Qt 只能在主线程建窗** ⇒
    直接建 = **挂死**（EL 第一版判据已实测）。所以分工是：

    .. code-block:: text

        服务器线程  ── open() ──►  队列（threading.Queue）
                                      │
        Qt 主线程   ◄── QTimer ───────┘  drain() ⇒ 真正 MainWindow()

    `open()` 会**等主线程兑现后的回执**（`_ACK`）再返回 ⇒ 判据能断言
    "真的开了"（窗口对象在），而不是只看到一句乐观的字符串。

    ## 幂等

    已经开着 ⇒ 回 ``"already-open"``（**不重复开、不报错**）⇒ 判据可以断言窗口数还是 1。
    """

    #: `open()` 的最长等待（秒）。主线程在跑（drain 每 50ms 一次）⇒ 正常毫秒级。
    OPEN_TIMEOUT_S = 20.0

    def __init__(self, app) -> None:
        self._app = app
        self._q: queue.Queue = queue.Queue()
        self._window = None
        self._last_error = ""
        #: 驱动 `drain()` 的 QTimer（由 `_attach_manual_surface` 装；收尾时停掉）。
        self.timer = None

    # ---- 判据用的只读事实面（**不加锁**：只在断言里读，且值单调） ----
    @property
    def is_open(self) -> bool:
        return self._window is not None

    @property
    def window_count(self) -> int:
        """开过几个窗口（幂等判据：必须恒为 0 或 1）。"""
        return 1 if self._window is not None else 0

    @property
    def last_error(self) -> str:
        return self._last_error

    # ---- 服务器线程侧 ----
    def open(self) -> str:
        """请求开窗（**可在服务器线程调用**）；返回状态字符串给控制端点。

        ⚠ **失败必须响亮**：主线程建窗抛了 ⇒ 这里 `raise`（把真因带出去），
        绝不回一句"开好了"——"静默假成功"正是本条要防的形态。
        """
        if self._window is not None:
            return "already-open"
        ack: queue.Queue = queue.Queue(maxsize=1)
        self._q.put((ack, threading.get_ident()))
        try:
            opened = ack.get(timeout=self.OPEN_TIMEOUT_S)
        except queue.Empty:
            raise RuntimeError(
                "主线程没有在 %.0fs 内兑现开窗请求（Qt 事件循环没在跑？）"
                % self.OPEN_TIMEOUT_S)
        if isinstance(opened, BaseException):
            raise RuntimeError(f"建窗失败：{opened!r}") from opened
        if not opened:
            raise RuntimeError(f"建窗未生效（last_error={self._last_error!r}）")
        return "opened"

    # ---- Qt 主线程侧（由 QTimer 驱动） ----
    def drain(self) -> None:
        while True:
            try:
                ack, _tid = self._q.get_nowait()
            except queue.Empty:
                return
            if self._window is not None:
                ack.put(True)                 # 幂等：别人已经开好了
                continue
            try:
                self._build_window()
                #: 打印一行**可据以断言**的证据（判据要证明"真建了窗"，不是只看返回值）。
                print(f"[mcp] 人手面已开：主窗口 {type(self._window).__name__} "
                      f"（在本进程内，与 MCP 面共用同一份门/账）", flush=True)
                ack.put(True)
            except BaseException as exc:       # noqa: BLE001 - **必须带出去**
                # ⚠ 不吞异常：把真因放进 ack，让服务器线程抛给调用方；
                #   否则主线程抛一个没有信息量的错，表现成"崩溃"而不是"干净地红"。
                self._last_error = f"{type(exc).__name__}: {exc}"
                ack.put(exc)

    def _build_window(self) -> None:
        """**只在 Qt 主线程调用**：建主窗口并 `show()`。

        ⚠ 与 MCP 服务**同一个进程** ⇒ 复用同一份 Gate/Authority/History；
        窗口自己**不建桥**（`ml_toolbox.ui` 不许 import `ml_mecha`，那条边界有判据守着）。
        """
        # ⚠ 导入顺序铁律：lightgbm（经 registry 顶层导入）必须先于 PyQt5 模块。
        #   `run_mcp` 在起服务前已经 `load_builtin()` ⇒ 这里只补 Qt 侧导入。
        from ml_toolbox.ui.main_window import MainWindow

        self._window = MainWindow()
        self._window.show()


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gui", action="store_true")
    #: ⚠ 名字是 ``--mcp``（原 ``--authority``）：它描述的是"**本进程对外提供什么**"
    #: （一个 MCP 服务端点），而不是"它在编排里扮演什么角色" ——
    #: 编排（起谁、等谁、收谁）已整块归 mecha 的 supervisor（见 `ml_mecha/mecha_site.py`）。
    parser.add_argument("--mcp", action="store_true")
    parser.add_argument("--root", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", default="")
    parser.add_argument("--dataset-roots", action="append", default=[])
    parser.add_argument("--no-open-gate", action="store_true")
    #: ⭐ **本进程带不带"人面"**（三层架构 A1：人那条腿是**软件进程内的窗口**）。
    #: ⚠ **启动时不开窗**：收到 `open_manual_surface()`（控制端点）才开 —— 按需开。
    #: ⚠ 不带这一档 ⇒ 控制端点传 `open_surface=None` ⇒ 调它**响亮失败**
    #:   （"这个进程有没有人手面"是**启动者选的形态**，不是猜的）。
    parser.add_argument("--with-gui", action="store_true")
    #: ⭐ **本进程带不带"只读监视窗"**（`PanelWeb`：人看方法结果那只眼；独立进程 + `.mlweb-port`）。
    #: 与 `--with-gui` 同一条纪律：**形态由启动者选**，不是猜的。
    #: ⚠ 不带它 ⇒ **不起**（判据/维护/纯 AI 形态都不需要弹一个面板 + 一个浏览器标签）；
    #: 框架那条路在声明里带上它（见 `mecha_site.py` 的 `_service_argv`）。
    parser.add_argument("--with-panel", action="store_true")
    #: ⭐ **别自动开浏览器**（词汇表口径：`--mcp` / `--with-gui` / `--with-web` / `--with-panel` /
    #: `--no-open`）。只影响"有网页面那可开"的形态（本项目是 `--with-panel` 起的只读监视窗）；
    #: ⚠ 与 `--with-panel` **正交**：`--with-panel --no-open` = 起面板但不弹浏览器
    #: （无人值守/判据/自动化要的就是这个）。
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    return parser.parse_args(argv)


def run_gui() -> int:
    """本地 GUI 模式：完整委托给 ``app.py``（保持唯一一份 GUI 装配代码）。

    ``import app`` 是**有意**放在函数内的：见模块 docstring 纪律 1。
    ``app.main()`` 末尾 ``sys.exit(code)``，SystemExit 直接向上传播（exit code
    原样保留），所以本函数正常情况下不返回。
    """
    import app                        # noqa: PLC0415 - 见纪律 1：必须惰性导入
    app.main()
    return 0                          # pragma: no cover - app.main 不返回


def _drain_control_queue(surface) -> None:
    """把控制线程排队的"开窗"请求**在主线程上兑现**。

    ⚠ **本函数必须在 Qt 主线程上跑**（由 `QTimer` 驱动，见 `_attach_manual_surface`）：
    控制请求到达的是 **MCP 服务器线程**，在那里直接 `MainWindow()` ⇒ **挂死**
    （EL 第一版判据就这么写的，已实测）。⇒ 分工是"**线程只入队，主线程建窗**"。
    """
    surface.drain()


def _pump_until_quit(app, stop_event) -> None:
    """在**主线程**上跑 Qt 事件循环，直到 `stop_event` 被置位（收尾信号）。

    ⚠ 为什么不用 `while not stop: app.processEvents()` 那种忙转：
    那会把 CPU 打满、且拿不到 Qt 的"窗口关闭"语义。这里用一个 200ms 的定时器
    去看收尾信号，正常路径仍由 `app.exec_()` 事件循环驱动。
    """
    from PyQt5.QtCore import QTimer

    watchdog = QTimer()
    watchdog.setInterval(200)
    watchdog.timeout.connect(lambda: app.quit() if stop_event.is_set() else None)
    watchdog.start()
    app.exec_()


def _attach_manual_surface(app):
    """建人手面并接线；返回 :class:`_ManualSurface`（**不建窗**）。

    ⚠ 起进程时**先不开窗**（用户裁决：按需开，不是启动就开）——
    这里只装一个 50ms 的 `QTimer`：它每次在主线程上跑，把服务器线程排进队列的
    开窗请求兑现（`_drain_control_queue`）。真正的 `MainWindow()` 在第一次
    `open_manual_surface()` 到达时才建。
    ⚠ 建窗与 MCP 服务**同一个进程** ⇒ 天然复用同一份 Gate/Authority/History
    （不是第二个装配）；ML 的窗口本来就不自建桥（`ml_toolbox.ui` 不 import `ml_mecha`）。
    """
    from PyQt5.QtCore import QTimer

    surface = _ManualSurface(app)
    timer = QTimer()
    timer.setInterval(50)
    timer.timeout.connect(lambda: _drain_control_queue(surface))
    timer.start()
    surface.timer = timer                      # 收尾时停掉（防"收尾后又来一次请求"）
    return surface


def run_mcp(args: argparse.Namespace) -> int:
    """起本项目的 MCP 服务端点：装配 → 起 MCP（带只读监控）→ 开闸 → 阻塞。

    ⚠ 名字是 ``--mcp``（原 ``--authority``）：它说的是"**本进程对外提供什么**"。
    "权威（authority）"那个词描述的是**它在编排里的角色**，而编排已整块归
    mecha 的 supervisor ⇒ 留着旧名会让人以为本进程还在编排什么。**它不编排任何东西。**
    """
    root = Path(args.root).resolve() if args.root else PROJ_ROOT
    #: ⭐ **框架声明的数据根优先**（三层架构 §5.1）：框架 spawn 本进程时注入
    #: `MECHA_SOFTWARE_ROOT`；⚠ 读它才不会出现"框架以为软件在用 A 根、软件实际在写 B 根"
    #: （那个坑的现象是"看不到刚才的改动"，**且无任何报错**）。
    from mecha.site import software_root_from_env
    declared_root = software_root_from_env(None)
    if declared_root is not None:
        root = Path(declared_root)
    root.mkdir(parents=True, exist_ok=True)

    # 训练核心（lightgbm/torch 等）先于一切 GUI 栈导入——headless 路径本就不
    # import Qt，这里显式先 load_builtin 是把 app.py 的顺序铁律在同一进程里也钉住。
    from ml_toolbox.core import registry
    registry.load_builtin()

    from ml_mecha.assembly import assemble_ml_mecha
    from ml_mecha.mcp_host import McpHost
    from ml_mecha.monitor_http import MonitorEndpoint
    from ml_mecha.runtime import (MCP_PORT_FILE, MONITOR_PORT_FILE, resolve,
                                  write_runtime)

    #: ⭐ 只读监视窗的**进程外壳**（人看方法结果那只眼）：`PanelWeb` 的独立进程入口。
    from ml_mecha.panel_web import WEB_PORT_FILE, PanelWebSidecar

    from mecha.authority import Mode
    from mecha.software_control import (CONTROL_PORT_FILE, build_control_registry,
                                        start_control_endpoint)
    from mecha.site import initial_mode_from_env

    #: ⭐ 给 AI 看的"怎么切写权"（框架**不写死**这句：那是**项目知识**）。
    #: ⚠ **别写端口/凭据**（它会给 AI 读）。
    #: ⚠ **文案必须与形态一致**（`--with-gui` 决定本进程有没有人手面）——
    #:   声明与实现不一致会让 AI 照着错的前提推理。
    if args.with_gui:
        how_to_open = (
            "在启动面板（常驻控制台）上点本软件的「切到手动」：写权即归人"
            "（本机可信、无口令），再点「交还给 AI」交还模型。"
            "本进程带人手面（`--with-gui`）⇒ 同一页的「开手动面」会**真的把 "
            "GUI 窗口开出来**（开在本进程内，与 MCP 面共用同一份门/账）。"
        )
    else:
        how_to_open = (
            "在启动面板（常驻控制台）上点本软件的「切到手动」：写权即归人"
            "（本机可信、无口令），再点「交还给 AI」交还模型。"
            "⚠ 本进程**没有人手面**（没带 `--with-gui`）⇒ 「开手动面」会**响亮失败**"
            "（不假装开了窗）；要看窗口请用 `app_entry.py --gui` 另起一个进程。"
        )

    #: ⭐ **人手面**（三层架构 A1：人那条腿是**软件进程内的窗口**）。
    #: ⚠ 只有 `--with-gui` 才建 —— "这个进程有没有人手面"是**启动者选的形态**。
    #: ⚠ 建窗**要在 Qt 主线程**做 ⇒ 这里只建 QApplication，真正开窗留给
    #:   `_ManualSurface.drain()`（由主线程的 QTimer 驱动）。
    surface = None
    qt_app = None
    if args.with_gui:
        from PyQt5.QtWidgets import QApplication
        qt_app = QApplication.instance() or QApplication(sys.argv[:1])
        surface = _attach_manual_surface(qt_app)

    mcp_port_file = (Path(args.port_file) if args.port_file
                     else resolve(root, MCP_PORT_FILE))
    #: 收尾信号（Ctrl+C / 启动方终止）：主线程跑 Qt 事件循环时由定时器看它。
    stop_event = threading.Event()
    ml = assemble_ml_mecha(root=root, dataset_roots=args.dataset_roots or None)
    host = McpHost(ml, host=args.host, port=args.port, port_file=mcp_port_file,
                   log=lambda m: print(m, flush=True))
    monitor = None
    control = None
    #: ⭐ 只读监视窗（独立进程）；收尾时 `finally` 会引用它 ⇒ 先钉成 None。
    panel = None
    try:
        port = host.start()
        # 监控端点在同一权威进程内（与 MCP 共一份 History/Gate）——这正是
        # "AI 模式不开 GUI 也能看见谁改了什么"的数据来源（07 §7.3）。
        monitor = MonitorEndpoint(ml, host=args.host, mcp_port=port,
                                  port_file=resolve(root, MONITOR_PORT_FILE),
                                  log=lambda m: print(m, flush=True))
        monitor.start()
        #: ⭐⭐ **只读监视窗随服务起来**（2026-10-10 用户实测："ML 只读窗口还是没成功起来"）。
        #: 本项目那只眼是 `ml_mecha/panel_web.py` 那个**独立只读 web**
        #: （`.mlweb-port`，三个 view：ml / opt / nn；`nn` 那一页自己按需起 TensorBoard）。
        #: 它的 `main()` 本来就写着"**给启动方起窗口用**"——缺的就是那个启动方，就是这里。
        #: ⚠ **只在 `--with-panel` 时起**（与 `--with-gui` 同一条纪律：形态由启动者选）；
        #: ⚠ 三条纪律：**best-effort**（起不来不许拖垮 AI 那条腿）·**响亮**（把 reason 打出来，
        #: 不假装起了）·**收尾时只收自己起的那个**（见下面 finally 里的 stop）。
        if args.with_panel:
            panel = PanelWebSidecar(
                port_file=resolve(root, WEB_PORT_FILE),
                #: ⭐ 预载优化侧：`opt` 那一页的读档能力靠它 import 时注册（见 CLI 的 `--preload`）。
                #: ⚠ 名字在这里是**数据** ⇒ 边界层不因此依赖 opt 层（同 `preload_modules` 的说明）。
                preload=("ml_toolbox.opt.persistence",),
                log=lambda m: print(m, flush=True))
            got_panel = panel.start()
            if got_panel.get("ready"):
                print(f"[panel] 只读监视窗（人看方法结果）：{got_panel['url']}", flush=True)
                if args.no_open:
                    print("[panel] --no-open：不自动开浏览器（面板已起，地址见上一行）",
                          flush=True)
                else:
                    try:
                        import webbrowser
                        webbrowser.open(str(got_panel["url"]))  # "把面板带到眼前"（本项目安排）
                    except Exception as exc:              # noqa: BLE001 - 打不开浏览器不算失败
                        print(f"[panel] 浏览器没打开（不算失败）：{type(exc).__name__}: {exc}",
                              flush=True)
            else:
                print(f"[panel] 只读监视窗起不来（不影响 AI 那条腿）：{got_panel.get('reason')}",
                      flush=True)
        else:
            print("[panel] 未带 --with-panel ⇒ 不起只读监视窗（纯 AI 形态）", flush=True)
        if args.no_open_gate:
            print("[mcp] 写权保持 LOCKED（--no-open-gate）。", flush=True)
        else:
            #: ⭐ **启动者声明的初始写权**（三层架构 §5.1）：框架 spawn 本进程时注入
            #: `MECHA_INITIAL_MODE`；读取口是框架的 `initial_mode_from_env`
            #: （**同一个契约的两半住在一起**：注入在 `Service.child()`、读取在 `mecha.site`）。
            #: ⚠ 空串 = 显式取消 ⇒ 回落默认；⚠ 非法值 ⇒ 回落**并出声**（`'Mode.AI'` 那个坑）。
            #: ⚠ 开闸发生在**启动序列里**，AI 面永远拿不到 `side='human'` 的通道
            #:   （`mecha/authority.py::switch_mode` 的硬判定）⇒ 这不是 AI 自解锁。
            declared = initial_mode_from_env(Mode.AI)      # 缺省 = 本项目声明的 AI
            ml.authority.switch_mode(declared, "human")
            print(f"[mcp] 写权按启动者声明装配：{declared.value}", flush=True)

        #: ⭐ **控制端点**（人侧专用那条腿，三层架构 §7.1.1）。
        #: ⚠ 框架给**机制**，本项目只交 3 个回调 + 2 个值 —— 不必再手写一遍
        #:   （EL/GP 曾各写 ~134 行，ML 此前完全没有 ⇒ 切不了手动）。
        #: ⚠ **记账走 `gate.record`（只记史、不过写权门）**：切写权本身就是要改的东西，
        #:   过门会**自锁**（切到 ai 之后人侧再也记不了账、也切不回来）。
        #: ⭐ **人手面回调**：`--with-gui` 时是"**入队 → 主线程建窗**"那条路
        #: （⚠ 直接在 MCP 服务器线程建 Qt 窗口会**挂死**，EL 已实测）；
        #: 不带那一档 ⇒ `None` ⇒ `open_manual_surface()` **响亮失败**（诚实）。
        reg = build_control_registry(
            ml.authority,
            switch=lambda v: ml.authority.switch_mode(Mode(v), "human"),
            audit=lambda op, val, why: ml.gate.record(op, "authority", val,
                                                      actor="human", reason=why),
            open_surface=(surface.open if surface is not None else None),
            how_to_open=how_to_open,
        )
        control, control_port = start_control_endpoint(
            reg, port_file=resolve(root, CONTROL_PORT_FILE),
            server_name="ml-toolbox-control", log=lambda m: print(m, flush=True))
        # 运行期描述符 = 端口的单一真源（启动方、监控页、故障排查都读它）。
        # ⚠ **不写 mode**：写权是**运行中可切**的（任何本机客户端都能请求切），
        #   "启动时那个值"不是值得持久化的真相；要读当前写权走控制端点。
        write_runtime(root, authority_pid=os.getpid(),
                      mcp_port=port, monitor_port=monitor.port)
        print(f"[mcp] 端点就绪：写权 {'LOCKED' if args.no_open_gate else '按启动者声明'}，"
              f"MCP {host.url}，只读监控 {monitor.url}，"
              f"控制端点 {'.control-port'}（文件 "
              f"{mcp_port_file.name} / {MONITOR_PORT_FILE} / {CONTROL_PORT_FILE}）"
              + ("，人手面：**按需开**（尚未开窗）" if surface is not None else
                 "，人手面：无（未带 --with-gui）")
              + "（Ctrl+C 或启动方收尾时退出，写租约随进程释放）。", flush=True)
        if surface is not None:
            #: 主线程跑 Qt 事件循环（收尾信号由 200ms 定时器看）——**不忙转**。
            _pump_until_quit(qt_app, stop_event)
        else:
            stop_event.wait()
    except KeyboardInterrupt:
        print("\n[authority] 收到 Ctrl+C，收尾（释放写租约）。", flush=True)
    finally:
        stop_event.set()
        timer = getattr(surface, "timer", None) if surface is not None else None
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        if control is not None:
            control.stop()
        if monitor is not None:
            monitor.stop()
        #: ⭐ 收尾**只收自己起的那个**面板（复用到的"别人那个"不动，见 `PanelWebSidecar.stop`）。
        if panel is not None:
            panel.stop()
        host.stop()
        ml.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    # ⚠ **stdout 与 stderr 都要配**（A7②）：只配 stdout ⇒ 一句
    #   `print(…, file=sys.stderr)`（或一条 warning）在 cp936 管道里仍会抛
    #   `UnicodeEncodeError`；而"输出被重定向"时默认编码是 **cp936**，不是 UTF-8
    #   （用户实测：网络代码里的 `print('⚠ …')` 让整个后台任务崩掉）。
    #   ⚠ 这里只是**兜底**：真正的强制在**框架的 spawn 层**（supervisor）——
    #   它统一给所有子进程注入 `PYTHONIOENCODING=utf-8`（不依赖"是否走到 main()"）。
    #   本项目声明里把该变量**显式置空**就是为了让那个注入生效
    #   （见 `ml_mecha/mecha_site.py::_NO_CODEC`）。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass
    args = _parse(argv)
    if args.help:
        print(_USAGE)
        return 0
    if args.gui and args.mcp:
        print("--gui 与 --mcp 互斥。\n" + _USAGE)
        return 2
    if args.gui:
        return run_gui()
    if args.mcp:
        return run_mcp(args)
    print("未指定：本文件只起【一个软件自己的形态】（--gui 或 --mcp）。\n"
          "要起整站（含 harness 与进程编排）用：python -m mecha\n" + _USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
