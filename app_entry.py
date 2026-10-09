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
"""


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


def run_mcp(args: argparse.Namespace) -> int:
    """起本项目的 MCP 服务端点：装配 → 起 MCP（带只读监控）→ 开闸 → 阻塞。

    ⚠ 名字是 ``--mcp``（原 ``--authority``）：它说的是"**本进程对外提供什么**"。
    "权威（authority）"那个词描述的是**它在编排里的角色**，而编排已整块归
    mecha 的 supervisor ⇒ 留着旧名会让人以为本进程还在编排什么。**它不编排任何东西。**
    """
    root = Path(args.root).resolve() if args.root else PROJ_ROOT
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

    mcp_port_file = (Path(args.port_file) if args.port_file
                     else resolve(root, MCP_PORT_FILE))
    ml = assemble_ml_mecha(root=root, dataset_roots=args.dataset_roots or None)
    host = McpHost(ml, host=args.host, port=args.port, port_file=mcp_port_file,
                   log=lambda m: print(m, flush=True))
    monitor = None
    try:
        port = host.start()
        # 监控端点在同一权威进程内（与 MCP 共一份 History/Gate）——这正是
        # "AI 模式不开 GUI 也能看见谁改了什么"的数据来源（07 §7.3）。
        monitor = MonitorEndpoint(ml, host=args.host, mcp_port=port,
                                  port_file=resolve(root, MONITOR_PORT_FILE),
                                  log=lambda m: print(m, flush=True))
        monitor.start()
        if args.no_open_gate:
            print("[mcp] 写权保持 LOCKED（--no-open-gate）。", flush=True)
        else:
            # ⚠ **启动者声明的初始写权**：声明在 ``ml_mecha/mecha_site.py``
            # （``initial_mode``），框架按《三层架构》§5.1 用 ``MECHA_INITIAL_MODE``
            # **送**进来。⚠ 今天这里还**没读**那个环境变量（见 mecha_site.py 的
            # Known Limitations）⇒ 仍旧按"启动即授权 AI 写"这条**旧口径**开闸，
            # 这不是 AI 自解锁——AI 侧拿不到 side='human' 的通道。
            ml.switch_ai()
        # 运行期描述符 = 端口的单一真源（启动方、监控页、故障排查都读它）。
        # ⚠ **不写 mode**：写权是**运行中可切**的（任何本机客户端都能请求切），
        #   "启动时那个值"不是值得持久化的真相；要读当前写权走控制端点。
        write_runtime(root, authority_pid=os.getpid(),
                      mcp_port=port, monitor_port=monitor.port)
        print(f"[mcp] 端点就绪：写权 {'LOCKED' if args.no_open_gate else 'AI'}，"
              f"MCP {host.url}，只读监控 {monitor.url}（端口文件 "
              f"{mcp_port_file.name} / {MONITOR_PORT_FILE}）"
              f"（Ctrl+C 或启动方收尾时退出，写租约随进程释放）。", flush=True)
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\n[authority] 收到 Ctrl+C，收尾（释放写租约）。", flush=True)
    finally:
        if monitor is not None:
            monitor.stop()
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
