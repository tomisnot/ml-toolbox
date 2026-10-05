# -*- coding: utf-8 -*-
"""唯一装配点：把 ML host adapter 接到 mecha 的四件正职上（宪章 §11.1）。

``assemble`` 是 mecha 的唯一具名装配点（ADR 装配统一-assemble单点），本模块
只是**调用它**并补齐 ML 侧的注入：项目名、数据目录名、校验器、概括器、
工具投影与命令事件写口。

装配后的形态：

.. code-block:: text

    人类操作者 ── Host UI / 脚本 ─┐
                                  ├─► mecha Gate/History/Monitor/Tools
    AI 操作者 ── ToolHost / MCP ──┘        │
                                            ▼
                             ml_mecha.MLEngine（受控命令）
                                            │
                                            ▼
                            ml_toolbox.api.Session（ML 真值）

纪律：

- 出厂 ``Mode.LOCKED``（nothing ships enabled）；人类侧显式 ``switch_ai`` 才放闸；
- 校验器**必填**（mecha Gate 不静默放行），由 :class:`ml_mecha.validator.MLValidator`
  提供；
- 概括器是纯函数（:func:`ml_mecha.summarizer.ml_summarizer`），只读事件、无写回。

## Known Limitations and Deferred Work

- 长任务现在可以经 :meth:`MLMecha.submit` 异步跑（mecha Job Contract），但
  **取消是协作式的**：检查点在命令入口与批量/对比的方法边界，单次 fit 计算
  不可中断；job 元数据也**不跨进程**（进程重启后找回 job 仍需落 History）。
  声明为不可取消的命令（``cancel_supported=False``，如数据准备类）经
  :meth:`MLMecha.cancel_job` 会被结构化拒（``job_not_cancellable``）——
  闸门只负责"拒收做不到的取消"，不代表可取消的命令真能被中途打断。
- 跨进程写租约由 mecha local Log Provider 提供（``.ml-mecha`` 目录）；多进程
  同时写同一数据目录会被拒，这是有意的 fail closed。
- **Approval 已接线，但今天没有命令用到它**：框架 ADR
  （``docs/v2/notes/implemented/2026-09-26-命令面治理闸前置与声明式opt-in.md``）之后，
  ``approval_required=True`` 的命令会**真的被拦**（在任何副作用之前要一次 one-shot 授权，
  没接审批通道即拒绝，fail closed）。本装配点把 ``Software.approval`` 注入到
  :meth:`MLEngine.bind_command_registry`（``invoke(..., approval=…)``），
  但**ML 目前四条命令都是 ``approval_required=False``**（``core_commands.py``，
  且没有导出/删除类命令）⇒ 这道闸**今天不挡任何东西**。
  ⇒ 将来给导出/删除类命令声明 ``approval_required=True`` 时，闸**立刻生效**：
  默认的 ``LocalApproval(decider=None)`` 会**拒绝**（fail closed），要放行必须显式给决策器
  （装配参数 ``approval_decider=``）。
- ``wants_channel``（ADR §Decision 3）**本适配层不采用**：ML 的通道是**显式**绑在调用路径上的
  （工具层 ``_with_channel`` / ``MLMecha.submit`` 的 worker 各自 ``use_channel``），
  引擎从 contextvar 读；声明 ``wants_channel=True`` 会把同一个事实搬成 handler 形参，
  属于另一处重构（要配自己的判据），本批不做——不是遗漏。
  ⚠ **框架已有声明式替代品**：``CommandSpec.wants_channel``（声明后 handler 直接收
  ``channel=``，不声明则调用形状一字不变）⇒ 本仓用的是"自有的调用路径绑定"。
  **若将来要统一到框架那套**，改动面 = **handler 签名**（`_cmd_*` 与命令 handler 工厂）
  **+ 判据**（通道归因那几条，`test_ml_mecha.py` 的 actor/call_id 组）——换过去的收益是
  "归因来源收进声明面"，代价是上面两处；下一个人不必重新论证这一遍。
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mecha.assembly import Software, assemble
from mecha.authority import Mode
from mecha.errors import MechaError
from mecha.gate import Channel
from mecha.history import History
from mecha.monitor import Summarizer
from mecha.providers.codeact import Action, CodeAct
from mecha.scopes import ScopePolicy
from mecha.surface import EntryKind
from mecha.tools import CURRENT_CALL_ID, ToolRegistry

from ml_toolbox.api import Session

from .commands import COMMANDS as HOST_COMMANDS
from .core_commands import core_command_specs
from .engine import (JOB_RUN_FLAG, READ_QUERY_NAMES, RUN_COMMANDS, MLEngine,
                     clear_channel, make_state_writer, use_channel)
from . import vram
from .summarizer import ml_summarizer
from .tools import BANNED_WORDS, register_ml_tools

#: ML adapter 的只读查询面（注册进 ``Surface``，经 ``Surface.query`` 分派）。
#: ⚠ **不在这里另立一份字面量**：查询名、`_query_handler_map` 的键、编排入口的
#: "可用 op" 必须是**同一份**（指导意见 §4-2：工具名与 op 名别搞成两套账，
#: 否则模型每次都得试一次）。⇒ 单一真源在 `engine.READ_QUERY_NAMES`。
READ_QUERIES = READ_QUERY_NAMES

#: ML adapter 自己的数据目录名（mecha 不提供默认值，必须由项目注入）。
DATA_DIR_NAME = ".ml-mecha"

#: 后台 job 的**默认并发上限**。
#: ⚠ 为什么项目必须自己定（§3-1）：mecha 的 local job 实现是"每次 submit 新起一个
#: daemon 线程、**不排队、无上限**"——那是**机制**；"该允许多少个同时跑"是**项目
#: 策略**（机器核数、内存、模型规模都是项目才知道的事）。不兜住的话，AI 连续提交
#: 十几个训练就会把机器打满（而它自己毫无察觉）。
DEFAULT_MAX_JOBS = 4
MAX_JOBS_ENV = "MLTB_MAX_JOBS"


def _max_jobs() -> int:
    """并发上限：`MLTB_MAX_JOBS` 可调；非法值回默认（**不因为环境变量写错就崩**）。"""
    raw = os.environ.get(MAX_JOBS_ENV, "")
    try:
        return max(1, int(str(raw).strip()))
    except (TypeError, ValueError):
        return DEFAULT_MAX_JOBS

#: 人类侧默认通道标签（细标签归因用；side 才定写权）。
HUMAN_ACTOR = "ml-gui"
AI_ACTOR = "ml-ai"

#: 排队状态挂在 `ExecutionContext` 上的属性名（与进度 sidecar 同一手法）。
#:
#: ⚠ 为什么不改框架的 job 状态：框架的 `running` 表示"**线程已经起了**"，
#: 那是它的真话；"在排队、还没开始跑命令"是**调度层**的事实。两件事都要看得见，
#: 但**不能互相冒充**（把框架状态改写成 queued 就是在报假位置）。
QUEUE_SIDECAR = "ml_queue_extra"


def _ctx_cancelled(ctx: Any) -> bool:
    """这个执行上下文被请求取消了吗？（**两种形状都要认**）

    ⚠ 实测踩过：mecha 的 `ExecutionContext.cancelled` 是 **property**（回 bool），
    不是方法 ⇒ 直接 `ctx.cancelled()` 会 `TypeError: 'bool' object is not callable`
    （而且它发生在**排队线程**里，症状是"排队中被取消的任务变成 failed"——
    看起来像取消没生效）。⇒ 这里两种形状都认。
    """
    flag = getattr(ctx, "cancelled", None)
    if flag is None:
        return False
    try:
        return bool(flag() if callable(flag) else flag)
    except Exception:                                          # noqa: BLE001
        return False


def _attach_queue_note(ctx: Any, *, queued: bool, position: int,
                       waited_s: float) -> None:
    """把调度层的排队事实挂到执行上下文（`read_job` 从那里读）。"""
    if ctx is None:
        return
    try:
        setattr(ctx, QUEUE_SIDECAR, {"queued": bool(queued),
                                     "position": int(position),
                                     "waited_s": round(float(waited_s), 3)})
    except Exception:                                          # noqa: BLE001
        pass


def enforce_terminal_invariant(receipt: Any, command: str) -> Any:
    """**终态不变量**：终态的 job 必须"要么有 run、要么有可读的失败"。

    用户点名的那个"不该存在"的组合（实测原文）：
    ``state=done && error=="" && run_id==""``（既无 run、也无日志、也无 run 目录）——
    它**无法区分"成功但慢"和"从未启动"**。⇒ 在这里把它变成一次**框架自检失败**
    （``silent_terminal``），而不是让调用方看到"done 但什么都没有"。

    两条蕴含式（**不是严格 XOR**：``error`` 同时是 ``ok=True`` 时的告警通道，
    见 `RunSummary.warning` —— 所以"恰好一个非空"在这里不可实现，能实现且真正
    有用的是下面两条）：

    1. ``ok=True``  ⇒ ``run_ids`` 非空（说成功就得说成功在哪个 run 上）；
    2. ``ok=False`` ⇒ ``error`` 非空**且** ``error_kind`` 具体（说失败就得说哪一类）。

    ⚠ 只对**会产生 run 的命令**（:data:`ml_mecha.engine.RUN_COMMANDS`）成立：
    ``prepare_dataset`` 本来就不产 run，用同一条不变量会把正常的成功判成失败。
    ⚠ 违反时**改写回执**（``ok=False`` + ``error_kind="silent_terminal"``）而不是
    抛异常：抛异常会让 job 变成 ``failed`` + 一句栈，反而丢掉"是哪条命令"的上下文；
    改写后 `read_job` 会照既有路径把原因说给人听。
    """
    if not isinstance(receipt, Mapping) or command not in RUN_COMMANDS:
        return receipt
    out = dict(receipt)
    run_ids = [r for r in (out.get("run_ids") or []) if str(r or "").strip()]
    ok = bool(out.get("ok"))
    err = str(out.get("error") or "").strip()
    kind = str(out.get("error_kind") or "").strip()
    if ok and run_ids:
        return out
    if not ok and err:
        if kind:
            return out
        # 失败但没归类 ⇒ 补一个**具体**的 kind（"不知道哪一类"本身要看得见）
        out["error_kind"] = "job_command_failed"
        return out
    reason = ("命令回了 ok=True 但没有任何 run_id" if ok
              else "命令没成功，却也没给出一句失败原因")
    out.update({
        "ok": False,
        "run_ids": [],
        "error": f"silent_terminal：{reason}（命令 {command}）——"
                 f"既拿不到运行、也拿不到可读的失败，这是**框架自检失败**，"
                 f"不是「这次实验没结果」",
        "error_kind": "silent_terminal",
        "error_hint": "这条不变量存在的理由：`state=done && error=='' && run_id==''` "
                      "无法区分'成功但慢'与'从未启动'。请把这次调用上报给开发侧；"
                      "同时看一眼权威日志（`logs/`）与 `runs/` 最近写入时间",
        "error_suggest": "重跑一次；若重复出现，改用 read_job 看 job 的 progress/epochs",
    })
    return out


class MLMecha:
    """装配结果句柄：mecha ``Software`` + ML 引擎 + 薄帮助。

    不复制 mecha 的任何机制：``gate`` / ``history`` / ``monitor`` / ``tools``
    都是同一个 ``Software`` 实例里的对象，本类只做"少打几个字"的转发。
    """

    def __init__(self, software: Software, engine: MLEngine,
                 scopes: ScopePolicy) -> None:
        self.software = software
        self.engine = engine
        self.scopes = scopes
        self._closed = False
        #: 后台 job 的**并发名额**（§3-1：框架不排队、无上限 ⇒ 项目自己兜）。
        #: ⚠ 2026-10-05 起语义变了：它不再"超额就拒"，而是"**同时能跑几个**"，
        #: 超出的**入队**（用户实测：上限 4 时只能"投一个、等一个、再投一个"，
        #: 而他要的是"一次排满 24 小时"）。⇒ 用条件变量自己管，不用 BoundedSemaphore：
        #: 后者**不能热改容量**（而"改上限要重启"正是用户点名的代价）。
        self._max_jobs = _max_jobs()
        self._cond = threading.Condition()
        self._running_jobs = 0
        #: 排队表：``ticket → {"at": 入队时刻, "job_id": 提交后回填}``。
        #: 顺序 = ticket 顺序 = **入队顺序**（`queue_position` 由它算出来）。
        self._waiting: dict[int, dict] = {}
        self._ticket_seq = 0
        #: 提交那一刻的调度事实（`job_id → {queued, position, waited_s}`）。
        #: 提交到 worker 真正开跑之间有窗口，那期间只有这份初始值是真的。
        self._queue_info: dict[str, dict] = {}
        self._queue_info_lock = threading.Lock()
    # ---- 转发（同一实例，不是第二套） ----
    @property
    def gate(self):
        return self.software.gate

    @property
    def history(self):
        return self.software.history

    @property
    def monitor(self):
        return self.software.monitor

    @property
    def tools(self) -> ToolRegistry:
        return self.software.tools

    @property
    def toolhost(self):
        """工具面的宿主传输 seam（ToolHost 契约的真实消费者）。

        本适配层同时持有 ``tools``（定义）与 ``toolhost``（传输）：装配时用
        ``toolhost.declare(tools.schemas())`` 校验投影，宿主调工具走
        :meth:`call_tool`。两者指向同一张注册表（``toolhost.registry is tools``），
        不造第二真值。
        """
        return self.software.toolhost

    @property
    def artifacts(self):
        """宿主产物托管（Artifact 契约的真实消费者：``ml.run`` 事件落 locator）。"""
        return self.software.artifacts

    def call_tool(self, name: str, args: Mapping[str, Any] | None = None) -> dict:
        """经 ToolHost 调一次工具（失败已归一化为 ``is_error`` 回执，不抛穿）。"""
        return dict(self.software.toolhost.call(name, dict(args or {})))

    @property
    def authority(self):
        return self.software.authority

    @property
    def layout(self):
        return self.software.layout

    @property
    def mode(self) -> Mode:
        return self.software.authority.mode

    @property
    def human_channel(self) -> Channel:
        return self.software.channels["human"]

    @property
    def ai_channel(self) -> Channel:
        return self.software.channels["ai"]

    # ---- 薄帮助 ----
    def switch_ai(self, side: str = "human") -> None:
        """人类侧把写权交给 AI 通道（side 只能是 'human'，AI 不能自解锁）。"""
        self.software.authority.switch_mode(Mode.AI, side)

    def switch_human(self, side: str = "human") -> None:
        """人类侧收回写权。"""
        self.software.authority.switch_mode(Mode.HUMAN, side)

    def lock(self, side: str = "human") -> None:
        """人类侧锁死写权（任何通道都不可写域状态）。"""
        self.software.authority.switch_mode(Mode.LOCKED, side)

    # ---- Job Contract：长命令提交为后台 job（可轮询/可取消/带 result_ref） ----
    @property
    def jobs(self):
        return self.software.jobs

    def bind_codeact(self, codeact: Any) -> None:
        """接上 CodeAct 编排器（模型入口 `run_queries` 用它）。

        ⚠ 为什么是"注入"而不是在本类里 new 一个：`CodeAct` 需要**同一个**
        ``Surface`` 与 ``History``（同一实例，不是第二套）——那两样在装配点才有。
        把实例注入进来，保证"编排里跑的查询"与"单独调查询"走的是同一条路。
        """
        self._codeact = codeact

    @property
    def codeact(self) -> Any:
        """编排器；未装配 ⇒ ``None``（`run_queries` 会据此给教学拒绝）。"""
        return getattr(self, "_codeact", None)

    def submit(self, spec: Mapping[str, Any], *, side: str = "ai"):
        """把一条 ML 受控命令提交成 mecha job。

        ## ⭐ 两道**提交前**闸（§3：框架给机制，"上限/前置条件"由项目自己补）

        1. **写权预检**（`authority.gate(side)`，只读、无副作用）：
           让调用方**同步**拿到拒绝，而不是"提交一个注定被拒的 job、再去轮询失败"
           —— 后者会让 AI 白等一轮，还可能把失败读成"实验结论"。
        2. **并发上限**（`BoundedSemaphore`）：框架的 local job 实现是
           "每次 submit 新起一个 daemon 线程、不排队、无上限" ⇒ 不兜住的话
           连续提交 N 个训练会把机器打满。**超额响亮拒绝**（不排队、不静默丢弃）。
           ⚠ 顺序有讲究：预检在 `acquire` **之前**（失败不留占用）。

        可用 `MLTB_MAX_JOBS` 环境变量调上限（默认 4）。

        **submit 不扩权**：``side`` 只是"这次用哪条**既有**通道"的选择器，写权
        仍由 ``Authority`` 的当前 mode 决定——``mode=AI`` 时选 ``side="human"``
        照样被 Gate 以 ``authority_mode_mismatch`` 拒（本方法不碰 mode、不开关闸）。
        ``side`` 只接受 ``{"human", "ai"}``，其它值当场 ``bad_side``（见下）。

        命令在**后台线程**里跑；操作者通道与本次请求的 ``call_id`` 必须显式
        带过去（contextvars 不跨线程），因此 worker 自己在两个上下文里调
        ``engine.run``：通道定写权与归因，``call_id`` 让 job 路径的命令审计
        与操作者行为史互引（第三轮审查 P2-5）。

        **可取消性随命令声明走**（第三轮审查：声明不被消费就只是注释）：装配点
        从核心命令面取该命令的 ``CommandSpec.cancel_supported`` 交给 Job
        Registry——``False`` 的命令（如数据准备类）调 ``cancel_job`` 会被结构化
        拒（``job_not_cancellable``），不静默接受一个做不到的取消。命令不在核心
        命令面时**不在这里发明新拒绝**：既有失败路径是 worker 内的分派产
        ``ok=False`` 的教学回执（含 did-you-mean），故取不到声明就沿用 Job 默认。
        """
        if side not in ("human", "ai"):
            # 选 MechaError 而不是 ValueError：与库内其它拒绝同形（结构化
            # ``kind`` + hint + suggest），调用方（含 AI 可达面）拿到的是可
            # 教学失败而不是裸异常；装配点的其它拒绝也都是 MechaError。
            raise MechaError(
                f"submit 的 side 只能是 'human' 或 'ai'，收到 {side!r}",
                kind="bad_side",
                hint="side 选择用哪条既有通道（写权仍由 Authority 的 mode 决定）；"
                     "它不是权限等级，写 'root' 之类的值不会获得更多权限",
                suggest="side='ai'（AI 通道）或 side='human'（人类通道）",
            )
        command = str(spec.get("command", ""))
        declared = self.software.commands.spec(command)
        cancel_supported = True if declared is None else declared.cancel_supported
        channel = self.software.channels[side]
        # 提交线程里捕获本次请求的 call_id；后台线程的 contextvar 是空的。
        # 没设过就是 ""（人类/脚本路径），不是伪造一个。
        call_id = CURRENT_CALL_ID.get()

        # ⭐ 闸 ①：**写权预检**（只读、无副作用、同步拒绝）
        #   框架本来会在 worker 里经 Gate 拒；但那时 job 已经建了 ⇒ 调用方要
        #   "提交成功 → 轮询 → 发现失败"。改为**提交前**当场拒（同样经 Gate 的
        #   判断逻辑，不另造一套）：`authority.gate(side)` 只读 mode 并抛 GateDenied。
        self.software.authority.gate(side)
        # ⭐ 闸 ③：**显存预检**（2026-10-05，L3；用户实测 §3.3 的代价是一整晚）。
        #   这里是**快速失败**（提交前就拒，连 job 都不建）；同一个函数也在
        #   `MLEngine.run`（命令入口）上兜一次 ⇒ **同步路径与后台路径同口径**。
        vram.precheck(dict(spec))
        # ⭐ 闸 ②：**并发名额 → 超额入队**（2026-10-05 行为变更，见类说明）
        #   旧行为：超额**响亮拒绝**（`too_many_jobs`）⇒ 用户只能"投一个等一个"，
        #   而"一次排满 24 小时"是刚需。新行为：**照收，排队**，并在回执里给
        #   `queued` / `queue_position`（AI 立刻知道"前面还有几个"）。
        #   ⚠ 取消仍然有效：排队的 worker 会轮询框架的取消信号
        #     （`ExecutionContext.cancelled()`）⇒ **队列里的任务可以取消**
        #     （用户明确要求：他会投多、然后改主意）。
        with self._cond:
            self._ticket_seq += 1
            ticket = self._ticket_seq
            queued = self._running_jobs >= self._max_jobs
            if queued:
                self._waiting[ticket] = {"at": time.time(), "job_id": ""}
                queue_position = len(self._waiting)
            else:
                self._running_jobs += 1
                queue_position = 0

        def worker(ctx):
            channel_token = use_channel(channel)
            call_token = CURRENT_CALL_ID.set(call_id)
            held = not queued
            try:
                if not held:
                    started_wait = time.time()
                    _attach_queue_note(ctx, queued=True,
                                       position=self._queue_position(ticket),
                                       waited_s=0.0)
                    while True:
                        with self._cond:
                            if self._running_jobs < self._max_jobs:
                                self._running_jobs += 1
                                held = True
                                self._waiting.pop(ticket, None)
                                break
                            self._cond.wait(0.2)
                        if _ctx_cancelled(ctx):
                            # ⭐ **在队列里就被取消**：连命令都没跑 ⇒ 如实说，
                            #   不假装"训练被中断"（那会让人以为跑过一段）。
                            with self._cond:
                                self._waiting.pop(ticket, None)
                            return {
                                "ok": False, "command": command, "run_ids": [],
                                "primary_metric": "", "metrics": {}, "warnings": [],
                                "error": "任务在**排队期间**被取消：它还没开始跑，"
                                         "没有产生任何运行",
                                "error_kind": "cancelled_in_queue",
                                "error_hint": "取消发生在等并发名额的阶段；"
                                              "要跑请重新提交",
                                "error_suggest": "",
                            }
                    _attach_queue_note(ctx, queued=False, position=0,
                                       waited_s=time.time() - started_wait)
                # ⭐ **后台任务的终态自动落盘**（用户裁决：别让"意外"白训练）。
                #   为什么在这里标：`submit` 是**后台 job 的唯一入口** ⇒ 打这个标
                #   等价于"这次是在后台跑的"。而 job 状态只在**进程内存**里
                #   （注册表无淘汰、也无持久化）⇒ 权威一重启，`job_id` 与
                #   `result_ref` 全没了：训练其实跑完了、结果也在，但**没有路标**。
                #   ⇒ 后台任务默认落盘，等于给结果留一个不依赖"进程活到取回"的锚。
                #   ⚠ 只在**有 run 的命令**上打标（`prepare_dataset` 之类没有 run）。
                job_spec = dict(spec)
                if job_spec.get("command") in RUN_COMMANDS:
                    job_spec[JOB_RUN_FLAG] = True
                receipt = self.engine.run(job_spec, context=ctx)
                # ⭐ **终态不变量**（用户点名的那条）：终态必须"要么有 run、
                #   要么有可读的失败"。挂在这里的原因与落盘兜底**同一个**：
                #   `submit` 是后台 job 的唯一入口 ⇒ 不变量对所有命令自动生效，
                #   不需要每加一条命令就记得补一次。
                return enforce_terminal_invariant(receipt, command)
            finally:
                CURRENT_CALL_ID.reset(call_token)
                clear_channel(channel_token)
                # ⭐ 释放并发名额：**job 真跑完**（含失败/取消）才放。
                #   ⚠ `held` 而不是"总是释放"：排队期间被取消的那条**从没拿到名额**
                #   ⇒ 释放它会把名额凭空变多（上限形同虚设）。
                if held:
                    with self._cond:
                        self._running_jobs = max(0, self._running_jobs - 1)
                        self._cond.notify_all()

        def ref(receipt):
            if not isinstance(receipt, Mapping):
                return None
            if not receipt.get("run_ids"):
                return None
            return {"kind": "ml_command", "command": command,
                    "run_ids": list(receipt.get("run_ids", []))}

        try:
            job = self.software.jobs.submit(worker, command=command,
                                            cancel_supported=cancel_supported,
                                            result_ref_fn=ref)
        except Exception:
            # 提交本身失败（框架拒绝/注册表异常）⇒ **把名额还回去**，
            # 否则这两份账会永久错位（跑着的少了、名额却被占着）。
            with self._cond:
                self._waiting.pop(ticket, None)
                if not queued:
                    self._running_jobs = max(0, self._running_jobs - 1)
                self._cond.notify_all()
            raise
        with self._cond:
            if ticket in self._waiting:
                self._waiting[ticket]["job_id"] = job.id
        with self._queue_info_lock:
            self._queue_info[job.id] = {"queued": bool(queued),
                                        "position": int(queue_position),
                                        "waited_s": 0.0}
        return job

    def job_queue_info(self, job_id: str) -> dict:
        """这个 job 在**调度层**的状态：``{queued, position, waited_s}``。

        优先读**活的** sidecar（worker 会随排队/开跑更新它）；没有就回提交时记下的
        初始值（提交到线程真正开跑之间有个极小的窗口，那时只有初始值是真的）。
        """
        live: dict = {}
        try:
            ctx = self.software.jobs.get(job_id).context
            live = dict(getattr(ctx, QUEUE_SIDECAR, None) or {})
        except Exception:                                      # noqa: BLE001
            live = {}
        if live:
            return live
        with self._queue_info_lock:
            return dict(self._queue_info.get(job_id)
                        or {"queued": False, "position": 0, "waited_s": 0.0})

    # ---- 调度器（队列/热改上限）------------------------------------------------
    def _queue_position(self, ticket: int) -> int:
        """排队位次（1 = 下一个跑）；已不在队列 ⇒ 0。"""
        with self._cond:
            if ticket not in self._waiting:
                return 0
            return sorted(self._waiting).index(ticket) + 1

    def queue_state(self) -> dict:
        """调度器的当前账目：``{limit, running, queued, waiting:[job_id,…]}``。

        ⚠ 与框架的 job 状态**是两件事**，都要看得见：框架的 `running` 表示
        "线程已经起了"，而这里表示"**真的在跑命令**"——排队的那些在线程上也是
        `running`。两个都真，只是问的不是同一件事（`read_job` 会把两者分开报）。
        """
        with self._cond:
            ordered = sorted(self._waiting)
            return {"limit": int(self._max_jobs), "running": int(self._running_jobs),
                    "queued": len(ordered),
                    "waiting": [str(self._waiting[t].get("job_id") or "") for t in ordered],
                    "queue_limit_editable": True}

    def set_job_limit(self, limit: int) -> dict:
        """**热改**并发名额（不必重启权威——用户点名的代价就在这里）。

        - 调**大**：立刻唤醒排队的 job（能上几个上几个）；
        - 调**小**：**不打断**已在跑的（训练中断不是"改配置"该有的副作用），
          只是后续不再放新的进去 ⇒ 实际并发会在它们收尾后收敛到新上限。
        """
        try:
            n = int(limit)
        except (TypeError, ValueError):
            raise MechaError(
                f"并发名额需要整数，收到 {limit!r}", kind="bad_job_limit",
                hint=f"当前 {self._max_jobs}；例如 set_job_limit 8") from None
        if n < 1:
            raise MechaError(
                f"并发名额至少为 1，收到 {n}", kind="bad_job_limit",
                hint="0 会把所有后台任务永久堵在队列里（那不是「暂停」的语义）",
            )
        with self._cond:
            before = int(self._max_jobs)
            self._max_jobs = n
            self._cond.notify_all()
            after = self.queue_state()
        return {"before": before, **after}

    def wait_job(self, job, timeout: float | None = None) -> dict:
        """等待 job 结束并返回状态（``result_ref`` 在其中）。"""
        return self.software.jobs.wait(job.id, timeout)

    def cancel_job(self, job) -> dict:
        """请求取消并**返回核心说的真话**（``state_before`` / ``terminal`` / ``cancel_requested``）。

        ⚠ **行为变更（2026-09-27）**：从前返回 ``None`` —— 框架后来把取消的真实结果写进了
        返回值（终态 ⇒ ``cancel_requested=False``），丢掉它等于**把已经拿到的真话扔掉**，
        人类侧（控制台/GUI）于是只能自己编"已取消"。现在原样透出。
        返回值从 ``None`` 变成 ``dict`` 是**加法**：忽略返回值的调用方不受影响。
        """
        return self.software.jobs.cancel(job.id)

    def list_jobs(self) -> list[dict]:
        return self.software.jobs.list()

    def close(self) -> None:
        """关停：释放 History 写租约（幂等）。"""
        if self._closed:
            return
        self._closed = True
        self.software.close()

    def __enter__(self) -> "MLMecha":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False


def assemble_ml_mecha(*, root: Path | str | None = None,
                      session: Session | None = None,
                      summarizer: Summarizer | None = None,
                      mode: Mode = Mode.LOCKED,
                      project: str = "ml_toolbox",
                      data_dir_name: str = DATA_DIR_NAME,
                      dataset_roots: Sequence[str] | None = None,
                      allowed_export_roots: Sequence[str] | None = None,
                      seed: int = 42,
                      device: str = "auto",
                      resource_guard: Mapping[str, Any] | None = None,
                      export_root: str = "",
                      max_batch_methods: int = 8,
                      max_dataset_rows: int | None = None,
                      register_tools: bool = True,
                      approval_decider=None,
                      migration=None) -> MLMecha:
    """装配一台 ML×mecha 软件实例（**唯一装配点**）。

    ``root`` 省略时用 ``Path.cwd()``；数据目录固定为 ``data_dir_name``
    （``.ml-mecha``，ML adapter 自己的名字——mecha 不猜目录名）。
    """
    from .engine import DEFAULT_MAX_DATASET_ROWS

    root_path = Path(root) if root is not None else Path.cwd()
    roots = tuple(str(r) for r in (dataset_roots or ()))
    export_roots = tuple(str(r) for r in (allowed_export_roots or ()))
    max_rows = int(max_dataset_rows or DEFAULT_MAX_DATASET_ROWS)

    # 通道标签：side 定写权（human/ai），actor 只做归因（mecha 硬纪律 4）。
    human = Channel(HUMAN_ACTOR, "human")
    ai = Channel(AI_ACTOR, "ai")

    engine = MLEngine(
        session=session, state_writer=None,
        dataset_roots=roots, allowed_export_roots=export_roots,
        seed=seed, device=device, resource_guard=resource_guard,
        export_root=export_root, max_batch_methods=max_batch_methods,
        max_dataset_rows=max_rows,
        # ⭐ **身份台账落盘位置**（R32/R38）：`<工作区>/.ml-mecha/ml_ledger`。
        #   与 mecha 自己的操作账本（`journal.jsonl`）**同一个数据目录**：
        #   都是"这个工作区的事实"，不该分裂成两个根。
        ledger_dir=root_path / data_dir_name / "ml_ledger",
    )

    software = assemble(
        root=root_path, data_dir_name=data_dir_name, engine=engine,
        summarizer=summarizer or ml_summarizer,
        validate=engine.validator.validate_state,
        project=project, mode=mode, approval_decider=approval_decider,
        migration=migration,
    )
    # 换掉 mecha 出厂的同名通道：装配点钉死 ML 侧标签（请求内容改不了 side）。
    software.channels["human"] = human
    software.channels["ai"] = ai

    # 只读查询注册进 Surface（机制层注册项，不经写权门）：命令面之外的
    # 第二条标准路径也被真实消费，不再让 ``Surface.entries('query')`` 为空。
    for name in READ_QUERIES:
        software.surface.register(
            EntryKind.QUERY, name,
            (lambda _name=name, **kwargs: engine._query_direct(_name, **kwargs)),
            called_by="ml_mecha:read_query")
    engine.bind_surface_query(software.surface.query)

    # 引擎的写路径：唯一入口是 Gate（引擎自己无快照、无第二条写口）。
    engine.bind_state_writer(make_state_writer(software.gate))
    # 域运行摘要（ml.run）不是命令审计：直接进 History（只存摘要与 run_id 引用）。
    # 每条的紧凑 JSON 同时经 Artifact Contract 落盘，事件里只留**不透明 locator**
    # ——这是 Artifact 机制在第二宿主上的真实消费者（大结果不进 History）。
    engine.bind_command_events(
        _history_event_writer(software.history, software.artifacts))

    # 受控命令注册进核心 Command Surface：命令审计（command.<name>）与
    # result_ref 由核心写；这是"第二宿主真实消费核心通用机制"的落点。
    for core_spec in core_command_specs():
        software.commands.register(
            core_spec, _core_command_handler(engine, core_spec.name))
    # 审批通道一并注入（``invoke(..., approval=…)``）：框架**刻意不做自动接线**
    # （ADR：自动接线会让没声明审批的命令也走一遍审批通道 ⇒ 扩大行为面），
    # 由本装配点显式给。今天四条命令都没声明 ``approval_required`` ⇒ 这道闸不挡任何东西；
    # 将来声明了它立刻生效（默认 decider=None ⇒ fail closed 拒绝）。
    engine.bind_command_registry(software.commands, software.gate,
                                approval=software.approval)

    # 作用域写权：命令 scope 用 ML 自己的命令名（不透明字符串）。
    # ⚠ **实情（2026-09-30 校正注释）**：两条 side 通道**同授全部已注册命令**——
    # `all_scopes` 就是 `HOST_COMMANDS` 的**那四条**（prepare_dataset / run_method /
    # run_method_batch / compare_methods）⇒ 今天这个闸**不区分两侧**。
    # 这与能力图的声明**一致**（`capabilities.py` 的
    # `CONSUMERS_BY_KIND[KIND_COMMAND] = (adapter.ai, adapter.gui)`，注明"两条 side 通道都有 scope"）；
    # 工具面才是 AI 独有的（`register_ml_tools(..., side="ai")` + `KIND_TOOL: (adapter.ai,)`）。
    # ⇒ 本闸的价值是**运行期可撤销**：`scopes.revoke("ai", "run_method")` 之后，AI 的下一次调用
    #   会在副作用之前被 `scope_denied` 挡住（判据 `test_scope_policy_denies_unlisted_command_scope`
    #   钉的就是这条：先拒、再 `grant` 回来即放行）。
    # ⚠ **此前的注释在这里说了假话**：它写"未来 export/predict 等新命令必须显式加授，否则被
    #   `scope_denied`"——但 `all_scopes` 是**装配时**从 `HOST_COMMANDS` 算出来的 ⇒ 新增命令会
    #   **自动**进 AI 的授权集，不会出现"必须显式加授"。若要"新命令默认不给 AI"，那是**代码变更**
    #   （把 AI 侧改成显式白名单，如 `scopes.grant("ai", "prepare_dataset", …)` 逐条列），
    #   属行为变更，不是注释能表达的。
    scopes = ScopePolicy()
    all_scopes = tuple(c.name for c in HOST_COMMANDS)
    scopes.grant("human", *all_scopes)
    scopes.grant("ai", *all_scopes)
    software.commands.bind_scope_policy(scopes)

    # 先构造宿主句柄：job 类工具（submit_run/read_job/cancel_run）需要它作
    # Job 服务 seam（通道与 call_id 由 MLMecha.submit 带进后台线程）。构造是
    # 纯赋值、无副作用，故提前到注册工具之前不影响其它装配顺序。
    ml = MLMecha(software, engine, scopes)

    # ⭐ **把 CodeAct 接上电**（指导意见 §4：缺一步就等于没有）。
    #
    # 框架只登记了 `provider.codeact` 这个角色，**不负责造实例**；项目要自己
    # `CodeAct(surface, history, actor_provider)` 并把模型入口暴露出去。
    # 它够得到的恰好是：4 个内置动作（run/summary/estimate/schema）+ **Surface 里
    # 注册的 QUERY 项** —— 而我们的 7 条只读查询刚好都注册进去了（见上面的
    # `READ_QUERIES` 循环）。⇒ 两边**天然对齐**，不需要再造动作集。
    #
    # ⚠ 边界（框架自带说明，照抄以免误读）：CodeAct **不触达 command/tool 面**，
    #   它只到 query 那一层。所以"多步**写**编排"不在它的能力里——这也正是我们
    #   想要的：写操作仍是一条一条经受控命令面（每条各自过闸、各自留痕）。
    codeact = CodeAct(software.surface, software.history, lambda: AI_ACTOR)
    ml.bind_codeact(codeact)

    if register_tools:
        # ⭐ **真词表**（不是空表）：`define_tool` 会在构造期校验 description，
        #   撞上实现词就 `bad_description` ⇒ 工具**根本注册不出来**（fail loud）。
        #   ⚠ 此前传的是默认空表 ⇒ 纪律铺了却没启用（见 `BANNED_WORDS` 的说明）。
        register_ml_tools(software.tools, engine, actor=AI_ACTOR, side="ai",
                          host=ml, banned_words=BANNED_WORDS, codeact=codeact)
    # ToolHost 契约的真实消费者：装配时把模型可见投影交给宿主传输面校验
    # （白名单形状不对会当场 fail loud），宿主调工具走 MLMecha.call_tool。
    software.toolhost.declare(software.tools.schemas())
    return ml


def _core_command_handler(engine: MLEngine, name: str):
    """核心命令 handler：只接收声明过的参数 + 显式 context。"""
    def handler(*, context, **kwargs):
        return engine.run_core_command(name, context=context, **kwargs)
    return handler


def _history_event_writer(history: History, artifacts):
    """``ml.run`` 写入器：紧凑 JSON 落 Artifact，事件只带不透明 locator。"""
    def write(key: str, value: Mapping[str, Any], call_id: str = "",
              actor: str = "unknown") -> None:
        payload = dict(value)
        run_id = str(payload.get("run_id") or "")
        if run_id:
            locator = artifacts.put(
                json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
                kind="ml_run_summary", name=run_id)
            payload["artifact_locator"] = locator
        # ⚠ 新事件形状（mecha `926b90d`）：`History.append` 已被
        # `append_state` / `append_history` 取代。域运行摘要**只记史**（`ml.run` 不进快照）
        # ⇒ 用 `append_history(op=…, target="", after=…)`：`op` 承载事件名，`target` 留空。
        history.append_history("ml.run", after=payload, actor=actor,
                               reason="ml_mecha:command", call_id=call_id)
    return write


__all__ = ["assemble_ml_mecha", "MLMecha", "DATA_DIR_NAME", "HUMAN_ACTOR",
           "AI_ACTOR"]
