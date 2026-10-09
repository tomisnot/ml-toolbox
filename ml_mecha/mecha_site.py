# -*- coding: utf-8 -*-
"""ML Toolbox 的**站点声明**：把"这个项目怎么被起来"交给 mecha 的统一启动器。

## 这个文件是什么

mecha 的 supervisor 按约定发现本文件（模块级 ``SITE``）并把声明翻成进程编排。
⇒ **本项目不再自己编排进程**（旧 ``launcher.py`` 是它的前身，**已删**；
只有几处踩出来的坑与实测常量被抽成 ``docs/旧启动器-活知识存档.md``）。

⚠ **它是数据，不是逻辑**：不许在这里起线程、连网络、读业务状态。
唯一允许的"计算"是下面那段**读 ``dataset_roots.txt``**（那是宿主策略的内容，
按设计文档 §5.5(3) 归项目所有）。

## ⚠ 三层架构换代（2026-10-08）：``manual`` 已删，别再往回加

本声明**曾经**把"切手动"建成 ``Service.manual: ChildSpec``（**要 spawn 的进程**）。
三层架构（``docs/v2/软件-交互-写权-三层架构.md`` §3 裁决 1 / §6）**推翻了它**：

| 旧说法（已废） | 定论 |
|---|---|
| 人那条腿是 ``manual: ChildSpec``（要 spawn 的进程） | ⭐ **错** —— 它是**软件进程内的一个窗口** |
| "切手动 = 换腿（停一条起一条）" | ⭐ **切手动 = 只切写权**（``Mode``）；不动进程、不动连接、不动状态 |

⇒ 那条腿现在由两个**声明**表达：``has_gui``（面板据此决定显不显示"切到手动"）
与 ``initial_mode``（启动者声明的初始写权）。**判定依据是"窗口在软件进程内"，
所以它根本不是 supervisor 的 child** —— 写在这里只会逼 supervisor 去 spawn 一个
它不该管的东西。

## 为什么每一行是现在这样（改之前先读）

| 声明 | 依据 |
|---|---|
| ``ready.port_file = ".mcp-port"`` | 与 ``ml_mecha/runtime.py`` 的 ``MCP_PORT_FILE`` **同一个名字**（跨语言单一事实：dsh 插件也读它） |
| ``ready.ok_codes = ()`` | ⚠ 见下方 `_READY_ANY_RESPONSE` 的说明 —— 实测 `GET /mcp` 回 **406** |
| ``has_gui = True`` | ML 有 PyQt5 窗口（``app.py`` → ``ml_toolbox/ui/main_window.py``）⇒ 面板**该**显示"切到手动" |
| ``how_to_open`` | 面板/聚合面把它**原话**回给 AI（``mecha/aggregate.py``）⇒ 由项目给，框架不写死 |
| ``initial_mode = Mode.AI`` | 从 mecha 启动 = **永远是 AI 组**；启动者（人）即声明"我起它就是要用"（理由与保留一并写在下面） |
| ``env["ML_TOOLBOX_RUNS"]`` | ⚠ **必须显式钉死**：它是 ``runs/`` 的唯一开关（``core/runs.py``）⇒ 不许留给继承（§5.6 环境变量陷阱） |
| ``env["PYTHONIOENCODING"] = ""`` | 让 supervisor 的 ``setdefault("utf-8")`` 生效（见下方说明） |
| ``DSHSpec.env["MLTB_ML_ROOT"]`` | 它是 **dsh 插件**读的数据根（``dsh/src/index.ts:56``），不是 Python 侧读的 |

## 三处"不许静默"的底线

1. **`dataset_roots` 搬迁要保真**：下面 ``dataset_roots()`` 的行为与旧
   ``launcher._dataset_roots_argv`` **逐项等价**（判据里直接对拍）——
   尤其是"**每个路径带一次旗标**"那条（曾经不带旗标 ⇒ 权威 exit=2 ⇒ 打不开）。
2. **`runs/` 根不许留给继承**：``ML_TOOLBOX_RUNS`` 由本项目显式钉死
   （三层架构下"两条腿"已不是两个进程，但**软件进程与 GUI 内窗口仍必须看同一份**
   —— 窗口如果自己另算一遍 runs 根，人看到的就是另一份）。
3. **`ok_codes=()`` 不许滥用**：它表达的是"**有 HTTP 响应即证明服务在听**"，
   不是"错误码无所谓"。

## Known Limitations and Deferred Work

- ✅ **`initial_mode` 已接通**（本轮）：`app_entry.py::run_mcp` 现在用框架的读取口
  ``mecha.site.initial_mode_from_env`` 读 ``MECHA_INITIAL_MODE`` 并**应用到闸门**
  （缺省回落本项目声明的 AI；空串=显式取消；⚠ 非法值回落**并出声**）。
  实测日志：``[mcp] 写权按启动者声明装配：ai``。
  ⚠ 读取口上提到框架是刻意的：**注入（``Service.child()``）与读取（``mecha.site``）
  是同一个契约的两半，必须住在一起**，否则名字/语义/空串规则会在两边各自漂。
- ✅ **控制端点已接通**（本轮）：``app_entry.py::run_mcp`` 用框架的
  ``software_control.build_control_registry`` / ``start_control_endpoint`` 起了
  ``.control-port``（三个操作：``read_write_authority`` / ``set_write_authority`` /
  ``open_manual_surface``）。⚠ **机制在框架，本项目只交 3 个回调 + 2 个值**。
- ✅ **"人手面"已接通**（本轮 = **ML 门化**）：``--mcp --with-gui`` 那一档让**本进程**
  按需开窗（⭐ 收到 ``open_manual_surface()`` 才建窗口，**不是启动就开** —— 设计 §7.1）；
  ⚠ **不带那一档 ⇒ ``open_surface=None`` ⇒ 响亮失败**（"这个进程有没有人手面"是
  **启动者选的形态**，不是猜的）。从 mecha 启动那条路**默认带**它（见下面的 ``argv``）。
  ⚠ 窗口**复用装配好的那份桥/门**（设计 §11 的 A4：各建一份桥 ⇒ ``writer_lease_held``）。
  ⇒ 行为判据：``tests/test_manual_surface.py``（**真起子进程**：按需开窗 / 幂等 /
  不带档响亮失败 / 控制操作不在 AI 能力面上 / 切写权立刻生效）。
  ⚠ 实测附带发现（已修在**框架**）：``open_manual_surface`` 要 **5.8 秒**建窗
  ⇒ 框架原先把控制超时**一刀切 3 秒** ⇒ 窗口**开成功了却报"协议错"**；
  现在**按操作分别定超时**（查询/切写权 3s，**开窗 60s**）。
- ⚠ **数据根同源**：``app_entry.py`` 现在读 ``MECHA_SOFTWARE_ROOT``（框架声明的根），
  读到了就以它为准。⚠ 它与本项目的 ``ML_TOOLBOX_RUNS`` **各有各的用途**
  （前者是本进程的数据根/写租约根，后者决定 ``runs/`` 落在哪）——
  **两者是否该同源，留待下一段核清**（不同源时现象是"看不到刚才的运行记录"，且**无报错**）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mecha.authority import Mode
from mecha.site import DSHSpec, ExtraColumn, ExtraPage, MonitorSpec, ReadySpec, Service, Site

#: 本仓根（声明文件位于 ``ml_mecha/`` 下一层 ⇒ 上跳一层）。
_SITE_ROOT = Path(__file__).resolve().parent.parent

#: ``dataset_roots.txt`` 的位置（人写的白名单；内容是本机绝对路径，不入库）。
_DATASET_ROOTS_FILE = _SITE_ROOT / "dataset_roots.txt"

#: 解释器：用**当前解释器**。理由：换一个 python 就等于换掉 site-packages，
#: 是在**另一个环境**里跑（旧 launcher 当年也是这么取的：``PY = sys.executable``）。
_PY = sys.executable

#: ⚠ **实测依据**：本项目 MCP 端点的 ``GET /mcp`` 回 **406**（不是 2xx），
#: 而该端点（`:mcp-port`）上**没有**任何回 2xx 的只读路由
#: （``/summary``/``/runs`` 等属**监控端点**，在另一个端口 ``.ml-monitor-port``）。
#:
#: "就绪"要表达的是"**这个服务真的在听**" —— 这条口径来自旧看门人的 D13 双条件
#: （spawn 前**快照**端口文件 + **真探活**），理由是"TCP 连得上还不够：
#: 同一端口上可能是别的东西在听"。
#: ⚠ 快照那一条**不用**在这里表达：框架的 ``ReadySpec`` 已经把它做进去了
#: （它自己比对"端口文件相对 spawn 前变化过"），声明只负责说清**探活**怎么做。
#: ⇒ 故这里用 ``ok_codes=()``（**任意 HTTP 响应都算在听**）而不是 ``tcp=True``，
#:   为的是保住"**必须是 HTTP 在听，不只是 TCP 有人**"的精度。
_READY_ANY_RESPONSE: tuple[int, ...] = ()


def dataset_roots() -> list[str]:
    """读 ``dataset_roots.txt`` → 允许的数据根绝对路径列表。

    不存在 / 读不出 / 全是注释 ⇒ **回空列表**（维持 fail closed，不猜任何默认根）。

    相对路径按**本文件所在目录**解析后取绝对路径 —— 权威侧校验的是
    "resolve 之后必须在允许根之下"，交相对路径会随 cwd 漂移。

    ⚠ **行为必须与旧 ``launcher._dataset_roots_argv`` 逐项等价**（判据里直接对拍）——
    那段实现**已随 launcher 删除**，所以这里的等价性是**冻结的期望值**在守
    （`tests/test_mecha_site.py` 内嵌 12 个用例 + GBK 兜底），不是"再去 import 它"。
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
    """起本项目服务（headless + **人手面**）的命令行。

    ⚠ **每个数据根带一次旗标**：``app_entry.py`` 那边是 ``action="append"``
    （一个 ``--dataset-roots`` 配一个值）。旧实现曾经只回**路径**（没有旗标），
    调用方直接展开 ⇒ argparse 收到裸参数 ⇒ ``unrecognized arguments``
    ⇒ **权威 exit=2 ⇒ 看门人 exit=1**，现象是"改完打不开了"。

    ⭐ ``--with-gui``：**从 mecha 启动的那条路默认带人手面**（三层架构 A1：
    人那条腿是**软件进程内的窗口**）⇒ 控制端点的「开手动面」**真能开窗**。
    ⚠ 它是**按需**开（不是启动就开）；⚠ 不带它的形态（例如手工
    `python app_entry.py --mcp`）⇒ `open_manual_surface()` **响亮失败**——
    "这个进程有没有人手面"是**启动者选的形态**，不是猜的。
    """
    argv = [_PY, "app_entry.py", "--mcp", "--with-gui", "--root", str(_SITE_ROOT)]
    for root in dataset_roots():
        argv += ["--dataset-roots", root]
    return tuple(argv)


#: ⚠ **这一行是"看到同一份账"的唯一防线**：``runs/`` 的根由它开关
#: （``ml_toolbox/core/runs.py``：``os.environ.get("ML_TOOLBOX_RUNS") or <包根>/runs``）。
#: ⚠ **不许留给继承**：三层架构下"两条腿"不再是两个进程，但**软件进程与它要开的
#: GUI 窗口仍必须看同一份** —— 窗口若自己另算一遍 runs 根，人看到的就是另一份，
#: 现象是"切了但看着是空的/老的"，且**没有任何报错**。
_RUNS_ENV = {"ML_TOOLBOX_RUNS": str((_SITE_ROOT / "runs").resolve())}

#: ⚠ 把 ``PYTHONIOENCODING`` **显式置空**，让 supervisor 的
#: ``child_env.setdefault("PYTHONIOENCODING", "utf-8")`` 生效
#: （``setdefault`` 只在键**不存在**时补 ⇒ 置空串会挡住它，故传空串表示"不设这个键"）。
#:
#: 为什么必须由框架统一注入（而不是这里写死 "utf-8"）：子进程 stdout 一旦不是真控制台
#: （被重定向到文件/管道），编码就退回 **cp936**，网络代码里一句 ``print("⚠ …")``
#: 会抛 ``UnicodeEncodeError``，**让整个后台任务崩掉**（用户实测、本机可复现）。
#: 那是**所有子进程**的共同需求 ⇒ 归 supervisor（活知识：``docs/旧启动器-活知识存档.md`` §3）。
_NO_CODEC = {"PYTHONIOENCODING": ""}



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
            #: ⭐ 三层架构：**"切手动"不是另一条要 spawn 的腿**，而是**写权切换**
            #: （``Mode``）—— 窗口开在**本进程内**（旧文的 ``manual: ChildSpec`` 已删，
            #: 那条路会被判据 ⑬ 当场 ``TypeError``）。
            #: ``has_gui`` 只回答"**本项目有没有人操作的那个窗口**"（面板据此显示按钮）。
            has_gui=True,
            #: ⭐ 启动者（人）声明的初始写权。⚠ 从 mecha 启动 = **永远是 AI 组**。
            #: ⚠ ML 的软件进程**有门**（``--mcp`` 侧装配了 Gate/Authority），
            #: 且**已读框架注入的** ``MECHA_INITIAL_MODE`` 并应用（本轮接通）。
            #: ⚠ 但**没有人手面**：GUI 是另一个进程 ⇒ 面板「切到手动」会
            #: "**写权成功、窗口开不了**"（框架分两步如实报告）。
            #: ⇒ 见本文件顶部 Known Limitations 里"仍然缺的一段"。
            initial_mode=Mode.AI,
            #: ⭐ 给 AI 看的"怎么切写权"：框架只**转交**、不改写，它出现在写权自省的回执里。
            #: ⚠ **别写端口/凭据** —— 这句话 AI 会读到。
            how_to_open=(
                "在启动面板（常驻控制台）上点该软件的「切到手动」：写权即归人"
                "（本机可信、无口令），再点「交还给 AI」交还模型。"
                "⚠ 但 ML 的 GUI 是**独立进程**、不在软件进程内 ⇒ 「切到手动」会"
                "**写权成功、窗口开不了**（框架分两步如实报告，不假装成功，"
                "见本文件顶部 Known Limitations 的『仍然缺的一段』）。"
            ),
            #: ⭐ **监控面声明**（三层架构 §7.3.6）：路由 / 端口文件名 / 标题 / 附加页
            #: **只在这里写一次** —— 它就是"声明 ↔ 面板"之间的唯一那份真值。
            #: ⚠ 框架把它生成成一份参数（**不含端口号**），经**框架自己的 patch** 带上的
            #: 插件交给浏览器 ⇒ 本项目 dsh 侧只留一行 bootstrap（`dsh/src/client/index.ts`）。
            #: ⭐ 2026-10-09：**监控地址路由也由框架注册**（框架的 host 插件按 `port_file` 的
            #: **绝对路径**现取端口）⇒ 这是那三个值的**唯一家**（项目侧那份参数常量已删）。
            monitor=MonitorSpec(
                route_path="/ml-toolbox/monitor-url",
                port_file=".ml-monitor-port",      # 与 ml_mecha/runtime.py 的 MONITOR_PORT_FILE 同名
                title="ML 监控",
                #: 两条附加页：⚠ 这里只放**数据**（页集 / 路由 / 文案 / 列名与取值字段）。
                #: 取值**函数**（`metricCell` 要按来源各取一次 `score` / `metrics[主指标]`）
                #: 是**代码** ⇒ 留在 `dsh/src/client/extraPages.ts`，走"自定义 UI"那条路。
                extra_pages=(
                    ExtraPage(
                        id="summary",
                        label="运行记录",
                        path="/summary",
                        title="运行记录（操作史 ml.run 事件）",
                        empty_text="本次会话还没有运行记录——AI/人每跑一次方法都会出现在这里。",
                        unreadable_text="运行记录读取失败：这一格不是「没有运行」，是没读到",
                        note="来自操作史 ml.run 事件：AI 跑过就一定在这里（与磁盘存档是两件事）。",
                        columns=(
                            ExtraColumn("运行编号", "run_id"),
                            ExtraColumn("方法", "method"),
                            ExtraColumn("主指标", "primary_metric"),
                            ExtraColumn("操作者", "actor"),
                        ),
                    ),
                    ExtraPage(
                        id="runs",
                        label="磁盘存档",
                        path="/runs?limit=50",
                        title="磁盘存档 runs/（跨重启可见）",
                        empty_text=(
                            "磁盘上没有落盘的运行。**注意：存档为空不等于没跑过**——"
                            "运行只在 persist=True（或 GUI 批量成功）时写入 runs/。"
                        ),
                        # ⚠「读不到 ≠ 空表」：真因摆成人能读的文本（旧面板在这点上的措辞原样保留）。
                        unreadable_text=(
                            "磁盘存档读取失败：这一格**不是空的**，是没读到（权威的存档读取出错了）"
                        ),
                        note="跨重启可见；与「运行记录」是两件事（那份来自操作史，这份来自磁盘）。",
                        columns=(
                            ExtraColumn("运行编号", "run_id"),
                            ExtraColumn("方法", "method"),
                            ExtraColumn("主指标", "primary_metric"),
                            ExtraColumn("时间", "saved_at"),
                        ),
                    ),
                ),
            ),
            env={**_RUNS_ENV, **_NO_CODEC},
        ),
    ),
    dsh=DSHSpec(
        command=("dsh", "web"),
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
