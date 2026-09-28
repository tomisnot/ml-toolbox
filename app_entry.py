# -*- coding: utf-8 -*-
"""ML Toolbox 双入口：本地 GUI 模式 / AI 模式（headless 权威）。

交付形态的两条腿（设计见 内部记录（未随仓发布，存档在仓外））：

- ``--gui``        本地 GUI 模式：走现有 ``app.py`` 的 GUI 路径（行为逐位不变，
                   不订阅 MCP、不起任何端点、不持有 mecha 写租约）。
- ``--authority``  AI 模式的**后台权威**：headless 装配 ``assemble_ml_mecha``
                   （出厂 ``Mode.LOCKED``）→ 起 MCP 服务（同进程后台线程）→
                   人类侧开闸 ``switch_ai()`` → 阻塞到 Ctrl+C / 被 terminate。
                   dsh 由看门人另起（P2），是纯客户端。

## 两条实现纪律（都不是风格问题）

1. **PyQt5 只在 ``--gui`` 分支里惰性导入**。``app.py`` 记录的导入顺序铁律是
   "lightgbm（经 methods 顶层导入）必须先于任何 PyQt5 模块，否则其 OpenMP 与
   Qt 冲突 → fit 时 access violation"。``--authority`` 是长跑训练进程，绝不能
   因为"顺手 import 了一下 Qt"把 native 初始化顺序搞坏，所以本模块**顶层不
   import app、不 import PyQt5**。
2. **写权落位是"看门人代理人的启动意图"**，不是 AI 自解锁：装配出厂
   ``LOCKED``（mecha 的 "nothing ships enabled"），起完服务后由**人类侧**
   调 ``switch_ai()``——打开闸门的动作发生在权威的启动序列里，AI 面永远拿不到
   ``side='human'`` 的通道（``mecha/authority.py::switch_mode`` 的硬判定）。

无参数运行时**明确报错**而不是静默变成 GUI：看门人（``launcher.py`` 的弹窗与
进程交接）是 P2 的交付物，现在还没接上；静默降级成 GUI 会让人以为"AI 模式
没生效"是个 bug。
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

#: 本文件所在目录 = 项目根。``.ml-mecha``（journal + 写租约）与 ``.mcp-port``
#: 都挂在这里——与看门人（P2 的 ``launcher.py``）必须同源，否则"起得来但看不见"。
PROJ_ROOT = Path(__file__).resolve().parent

_USAGE = """用法：
  python app_entry.py --gui                     本地 GUI 模式（同 python app.py）
  python app_entry.py --authority [选项]        AI 模式的后台权威（headless）
  python app_entry.py                           看门人两模式弹窗（P2 未接入）

--authority 选项：
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
    parser.add_argument("--authority", action="store_true")
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


def run_authority(args: argparse.Namespace) -> int:
    """AI 模式后台权威：装配 → 起 MCP → 开闸 → 阻塞。"""
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
            print("[authority] 写权保持 LOCKED（--no-open-gate）。", flush=True)
        else:
            # 看门人代理人的启动意图开闸：人选了 AI 模式即授权 AI 写。
            # 这不是 AI 自解锁——AI 侧拿不到 side='human' 的通道。
            ml.switch_ai()
        # 运行期描述符 = 端口/模式的单一真源（看门人、监控页、故障排查都读它）。
        write_runtime(root, mode=ml.mode.value, authority_pid=os.getpid(),
                      mcp_port=port, monitor_port=monitor.port)
        print(f"[authority] headless 权威就绪：写权 {'LOCKED' if args.no_open_gate else 'AI'}，"
              f"MCP {host.url}，只读监控 {monitor.url}（端口文件 "
              f"{mcp_port_file.name} / {MONITOR_PORT_FILE}）"
              f"（Ctrl+C 或看门人交接时退出，写租约随进程释放）。", flush=True)
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
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    args = _parse(argv)
    if args.help:
        print(_USAGE)
        return 0
    if args.gui and args.authority:
        print("--gui 与 --authority 互斥。\n" + _USAGE)
        return 2
    if args.gui:
        return run_gui()
    if args.authority:
        return run_authority(args)
    print("未指定模式：看门人（launcher.py，两模式弹窗）尚未接入（P2）。\n" + _USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
