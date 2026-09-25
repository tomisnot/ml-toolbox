# -*- coding: utf-8 -*-
"""唯一装配点：把 ML host adapter 接到 mecha 的稳定脊椎上（宪章 §11.1）。

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
- Approval 未接命令：``approval_required`` 只是核心声明，导出/删除类命令尚未
  出现，本装配点不接审批闸（诚实推迟）。
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mecha.assembly import Software, assemble
from mecha.authority import Mode
from mecha.errors import MechaError
from mecha.gate import Channel
from mecha.history import History
from mecha.monitor import Summarizer
from mecha.scopes import ScopePolicy
from mecha.surface import EntryKind
from mecha.tools import CURRENT_CALL_ID, ToolRegistry

from ml_toolbox.api import Session

from .commands import COMMANDS as HOST_COMMANDS
from .core_commands import core_command_specs
from .engine import MLEngine, clear_channel, make_state_writer, use_channel
from .summarizer import ml_summarizer
from .tools import register_ml_tools

#: ML adapter 的只读查询面（注册进 ``Surface``，经 ``Surface.query`` 分派）。
READ_QUERIES = ("method_catalog", "method_detail", "dataset_detail", "run_detail")

#: ML adapter 自己的数据目录名（mecha 不提供默认值，必须由项目注入）。
DATA_DIR_NAME = ".ml-mecha"

#: 人类侧默认通道标签（细标签归因用；side 才定写权）。
HUMAN_ACTOR = "ml-gui"
AI_ACTOR = "ml-ai"


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

    def submit(self, spec: Mapping[str, Any], *, side: str = "ai"):
        """把一条 ML 受控命令提交成 mecha job。

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

        def worker(ctx):
            channel_token = use_channel(channel)
            call_token = CURRENT_CALL_ID.set(call_id)
            try:
                return self.engine.run(dict(spec), context=ctx)
            finally:
                CURRENT_CALL_ID.reset(call_token)
                clear_channel(channel_token)

        def ref(receipt):
            if not isinstance(receipt, Mapping):
                return None
            if not receipt.get("run_ids"):
                return None
            return {"kind": "ml_command", "command": command,
                    "run_ids": list(receipt.get("run_ids", []))}

        return self.software.jobs.submit(worker, command=command,
                                         cancel_supported=cancel_supported,
                                         result_ref_fn=ref)

    def wait_job(self, job, timeout: float | None = None) -> dict:
        """等待 job 结束并返回状态（``result_ref`` 在其中）。"""
        return self.software.jobs.wait(job.id, timeout)

    def cancel_job(self, job) -> None:
        self.software.jobs.cancel(job.id)

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
    engine.bind_command_registry(software.commands, software.gate)

    # 作用域写权：命令 scope 用 ML 自己的命令名（不透明字符串）。人类侧全给；
    # AI 侧只给当前开放的四条命令——未来 export/predict 等新命令必须显式加授，
    # 否则 AI 在副作用之前被 scope_denied（fail closed）。
    scopes = ScopePolicy()
    all_scopes = tuple(c.name for c in HOST_COMMANDS)
    scopes.grant("human", *all_scopes)
    scopes.grant("ai", *all_scopes)
    software.commands.bind_scope_policy(scopes)

    # 先构造宿主句柄：job 类工具（submit_run/read_job/cancel_run）需要它作
    # Job 服务 seam（通道与 call_id 由 MLMecha.submit 带进后台线程）。构造是
    # 纯赋值、无副作用，故提前到注册工具之前不影响其它装配顺序。
    ml = MLMecha(software, engine, scopes)

    if register_tools:
        register_ml_tools(software.tools, engine, actor=AI_ACTOR, side="ai",
                          host=ml)
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
        history.append(key, payload, actor, "ml_mecha:command", call_id)
    return write


__all__ = ["assemble_ml_mecha", "MLMecha", "DATA_DIR_NAME", "HUMAN_ACTOR",
           "AI_ACTOR"]
