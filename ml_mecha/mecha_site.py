# -*- coding: utf-8 -*-
"""ML Toolbox 的**站点声明**：把"这个项目怎么被起来"交给 mecha 的统一启动器。

## 这个文件是什么

mecha 的 supervisor 按约定发现本文件（模块级 ``SITE``）并把声明翻成进程编排。
⇒ **本项目不再自己编排进程**（原 ``launcher.py`` 的那套是它的前身，逐步作废）。

⚠ **它是数据，不是逻辑**：不许在这里起线程、连网络、读业务状态。
唯一允许的"计算"是下面那段**读 ``dataset_roots.txt``**（那是宿主策略的内容，
按设计文档 §5.5(3) 归项目所有）。

## 为什么每一行是现在这样（改之前先读）

| 声明 | 依据 |
|---|---|
| ``ready.port_file = ".mcp-port"`` | 与 ``ml_mecha/runtime.py`` 的 ``MCP_PORT_FILE`` **同一个名字**（跨语言单一事实：dsh 插件也读它） |
| ``ready.ok_codes = ()`` | ⚠ 见下方 `_READY_ANY_RESPONSE` 的说明 —— 实测 `GET /mcp` 回 **406** |
| ``env["ML_TOOLBOX_RUNS"]`` | ⚠ **必须显式钉死**：它是 ``runs/`` 的唯一开关（``core/runs.py``），两条腿若不一致，切手动后会看到**另一份** runs（"切了但看着是空的"） |
| ``env["PYTHONIOENCODING"] = ""`` | 让 supervisor 的 ``setdefault("utf-8")`` 生效（见下方说明） |
| ``DSHSpec.env["MLTB_ML_ROOT"]`` | 它是 **dsh 插件**读的数据根（``dsh/src/index.ts:56``），不是 Python 侧读的 |
| ``manual.stop_timeout = 20.0`` | GUI 是 Qt 事件循环；强杀会让 ``finally`` 不跑（A7/A56 的同类教训），给足时间正常退出 |
| ``manual.ready`` 只能判存活 | ⚠ GUI **没有任何可探信号**（无端口文件、无握手）⇒ 如实声明，不编假判据 |

## 三处"不许静默"的底线

1. **`dataset_roots` 搬迁要保真**：下面 ``dataset_roots()`` 的行为与旧
   ``launcher._dataset_roots_argv`` **逐项等价**（判据里直接对拍）——
   尤其是"**每个路径带一次旗标**"那条（曾经不带旗标 ⇒ 权威 exit=2 ⇒ 打不开）。
2. **`runs/` 两腿同源**：svc 与 manual 都由 supervisor 注同一个 ``ML_TOOLBOX_RUNS``。
3. **`ok_codes=()`` 不许滥用**：它表达的是"**有 HTTP 响应即证明服务在听**"，
   不是"错误码无所谓"。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mecha.site import DSHSpec, ReadySpec, Service, Site

#: 本仓根（声明文件位于 ``ml_mecha/`` 下一层 ⇒ 上跳一层）。
_SITE_ROOT = Path(__file__).resolve().parent.parent

#: ``dataset_roots.txt`` 的位置（人写的白名单；内容是本机绝对路径，不入库）。
_DATASET_ROOTS_FILE = _SITE_ROOT / "dataset_roots.txt"

#: 解释器：用**当前解释器**，与旧 launcher 的 ``PY = sys.executable`` 同源
#: （用别的 python 会换掉 site-packages，是在另一个环境里跑）。
_PY = sys.executable

#: ⚠ **实测依据**：本项目 MCP 端点的 ``GET /mcp`` 回 **406**（不是 2xx），
#: 而该端点（`:mcp-port`）上**没有**任何回 2xx 的只读路由
#: （``/summary``/``/runs`` 等属**监控端点**，在另一个端口 ``.ml-monitor-port``）。
#:
#: "就绪"要表达的是"**这个服务真的在听**" —— 旧判据正是
#: "有 HTTP 响应即算在听（4xx 也是响应）"，理由是"TCP 连得上还不够：
#: 同一端口上可能是别的东西在听"。
#: ⇒ 故这里用 ``ok_codes=()``（**任意 HTTP 响应都算在听**）而不是 ``tcp=True``，
#:   为的是保住那条"**必须是 HTTP 在听，不只是 TCP 有人**"的精度。
_READY_ANY_RESPONSE: tuple[int, ...] = ()


def dataset_roots() -> list[str]:
    """读 ``dataset_roots.txt`` → 允许的数据根绝对路径列表。

    不存在 / 读不出 / 全是注释 ⇒ **回空列表**（维持 fail closed，不猜任何默认根）。

    相对路径按**本文件所在目录**解析后取绝对路径 —— 权威侧校验的是
    "resolve 之后必须在允许根之下"，交相对路径会随 cwd 漂移。

    ⚠ **行为必须与旧 ``launcher._dataset_roots_argv`` 逐项等价**（判据里直接对拍）。
    """
    if not _DATASET_ROOTS_FILE.is_file():
        return []
    try:
        text = _DATASET_ROOTS_FILE.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        # 本机历史文件有 GBK 的；读不出就**放弃**而不是猜。
        try:
            text = _DATASET_ROOTS_FILE.read_text(encoding="gbk")
        except OSError:
            return []
    except OSError:
        return []
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(os.path.expandvars(os.path.expanduser(line)))
        if not path.is_absolute():
            path = _SITE_ROOT / path
        out.append(str(path.resolve()))
    return out


def _service_argv() -> tuple[str, ...]:
    """起本项目服务（headless 权威）的命令行。

    ⚠ **每个数据根带一次旗标**：``app_entry.py`` 那边是 ``action="append"``
    （一个 ``--dataset-roots`` 配一个值）。旧实现曾经只回**路径**（没有旗标），
    调用方直接展开 ⇒ argparse 收到裸参数 ⇒ ``unrecognized arguments``
    ⇒ **权威 exit=2 ⇒ 看门人 exit=1**，现象是"改完打不开了"。
    """
    argv = [_PY, "app_entry.py", "--mcp", "--root", str(_SITE_ROOT)]
    for root in dataset_roots():
        argv += ["--dataset-roots", root]
    return tuple(argv)


#: ⚠ **这一行是"两腿同源"的唯一防线**：``runs/`` 的根由它开关
#: （``ml_toolbox/core/runs.py``：``os.environ.get("ML_TOOLBOX_RUNS") or <包根>/runs``）。
#: 若 svc 与 manual 两侧不一致，切手动之后人会看到**另一份 runs** ——
#: 现象是"切了但看着是空的/老的"，且**没有任何报错**。
_RUNS_ENV = {"ML_TOOLBOX_RUNS": str((_SITE_ROOT / "runs").resolve())}

#: ⚠ 把 ``PYTHONIOENCODING`` **显式置空**，让 supervisor 的
#: ``child_env.setdefault("PYTHONIOENCODING", "utf-8")`` 生效
#: （``setdefault`` 只在键**不存在**时补 ⇒ 置空串会挡住它，故传空串表示"不设这个键"）。
#:
#: 为什么必须由框架统一注入（而不是这里写死 "utf-8"）：见旧 ``launcher._spawn`` 的注释 ——
#: 子进程 stdout 一旦不是真控制台（被重定向到文件/管道），编码就退回 **cp936**，
#: 网络代码里一句 ``print("⚠ …")`` 会抛 ``UnicodeEncodeError``，**让整个后台任务崩掉**
#: （用户实测、本机可复现）。那是**所有子进程**的共同需求 ⇒ 归 supervisor。
_NO_CODEC = {"PYTHONIOENCODING": ""}


def _manual_child():
    """human adapter 的 ``ChildSpec``（"切手动"要起的那个 child）。

    ⚠ **为什么只能是"存活"这一档**（不是偷懒）：
    GUI 的最后一步就是 ``win.show(); app.exec_()``（``app.py``）——
    它**不监听端口、不写端口文件、不握手**（全仓 grep
    ``QLocalServer|QLocalSocket|socket|http.server`` 在 UI 层零命中）。
    ⇒ **今天没有任何可探信号**。能诚实做到的只有"进程还活着且一段时间没退出"。
    ⇒ ⚠ **判不了"窗口真的可用"**（Qt 可能在缺字体/插件/无显示时窗口空白而不崩）。
      真要 ``ready`` 就得让 GUI 侧加一条**显式握手**（写标记文件或起 QLocalServer），
      那是第二段的事，见设计文档 §5.3.1。

    ⚠ 本轮**不实现**换腿本身（那是步骤 3）：这里只声明，让框架知道"这条腿存在、
    怎么起、怎么收"。``id`` 必须是 ``<svc_id>:manual``（框架会校验）。
    """
    from mecha.supervise import ChildSpec

    return ChildSpec(
        id="ML:manual",
        argv=(_PY, "app_entry.py", "--gui"),
        cwd=_SITE_ROOT,
        env={**_RUNS_ENV, **_NO_CODEC},
        # ⚠ **必须与 svc 的 data_root 相同**（框架会校验）：换腿的定义就是
        # "同一份账、同一个写租约根，只换操作者" —— 不同的根就不是换腿，是另一个实例了。
        data_root=_SITE_ROOT,
        # ⚠ 不是 ReadySpec(...)：见上面"只能判存活"。`ready=None` ⇒ 起完即算就绪，
        # 若进程当场崩，supervisor 仍能从退出码看出（`state()` 里有 exit_code）。
        ready=None,
        # GUI 自然退出**不该**触发整组收尾（它不是这个站点的"主心骨"）。
        significant=False,
        # Qt 事件循环要时间；强杀会让 finally 不跑（A7/A56 的同类教训）。
        stop_timeout=20.0,
    )


SITE = Site(
    name="ML Toolbox",
    services=(
        Service(
            id="ML",
            brief="传统机器学习 / 神经网络 / 自动调参",
            argv=_service_argv(),
            cwd=_SITE_ROOT,
            # 判据 = 端口文件相对 spawn 前**发生过变化** 且 **探活通过**
            # （框架的两条一起看，缺一条就会重现"陈旧端口文件算就绪"的假绿）。
            ready=ReadySpec(
                port_file=".mcp-port",
                probe="http://127.0.0.1:{port}/mcp",
                ok_codes=_READY_ANY_RESPONSE,
                timeout_s=90.0,
            ),
            #: 写租约根 = 本仓根（每个软件各自持自己的；**不跨软件共享**）。
            data_root=_SITE_ROOT,
            #: ⭐ human adapter（"接手动"要起的那条腿）。与本体**互斥**：
            #: 同一 ``data_root`` 同一时刻只能一个写宿主（框架的写租约是 OS 级独占锁）。
            manual=_manual_child(),
            env={**_RUNS_ENV, **_NO_CODEC},
        ),
    ),
    dsh=DSHSpec(
        command=("dsh",),
        cwd=_SITE_ROOT,
        #: 入库的两层 patch（顺序即命令行顺序）。
        #: ⚠ 第三层（指向 MCP 端点的 ``runtime/dsh-ml-mcp.patch.yml``）是**生成物**，
        #: 由框架按聚合端点地址生成 —— 见设计文档 §5.5(4)。
        patches=(
            _SITE_ROOT / "dsh" / "cordis.patch.yml",
            _SITE_ROOT / "dsh" / "cordis.project.patch.yml",
        ),
        #: 3080 是官方 dsh 的端口，项目实例永不占（框架从 3081 起试）。
        port_preference=3081,
        port_tries=10,
        #: 插件产物：不存在 ⇒ 响亮失败（而不是起一个没有工具面的 dsh）。
        plugin_artifacts=(_SITE_ROOT / "dsh" / "lib" / "index.mjs",),
        #: ⚠ 它是 **dsh 插件**读的数据根（``dsh/src/index.ts:56``），不是 Python 侧读的。
        env={"MLTB_ML_ROOT": str(_SITE_ROOT), **_NO_CODEC},
        ready=None,   # dsh 的就绪由框架用「端口偏好 + 回读」判，见 DSHSpec 注释
    ),
)


__all__ = ["SITE", "dataset_roots"]
