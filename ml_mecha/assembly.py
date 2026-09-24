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

- 长任务仍是**同步**执行：ML 的 fit 在调用线程里跑完。mecha 的 Job Contract
  尚未与宿主引擎对接（见汇报的接口缺口），本层不假装有 job 语义。
- 跨进程写租约由 mecha local Log Provider 提供（``.ml-mecha`` 目录）；多进程
  同时写同一数据目录会被拒，这是有意的 fail closed。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mecha.assembly import Software, assemble
from mecha.authority import Mode
from mecha.gate import Channel
from mecha.history import History
from mecha.monitor import Summarizer
from mecha.scopes import ScopePolicy
from mecha.surface import EntryKind
from mecha.tools import ToolRegistry

from ml_toolbox.api import Session

from .commands import COMMANDS as HOST_COMMANDS
from .core_commands import core_command_specs
from .engine import MLEngine, make_state_writer
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
    engine.bind_command_events(_history_event_writer(software.history))

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

    if register_tools:
        register_ml_tools(software.tools, engine, actor=AI_ACTOR, side="ai")
    return MLMecha(software, engine, scopes)


def _core_command_handler(engine: MLEngine, name: str):
    """核心命令 handler：只接收声明过的参数 + 显式 context。"""
    def handler(*, context, **kwargs):
        return engine.run_core_command(name, context=context, **kwargs)
    return handler


def _history_event_writer(history: History):
    def write(key: str, value: Mapping[str, Any], call_id: str = "",
              actor: str = "unknown") -> None:
        history.append(key, dict(value), actor, "ml_mecha:command", call_id)
    return write


__all__ = ["assemble_ml_mecha", "MLMecha", "DATA_DIR_NAME", "HUMAN_ACTOR",
           "AI_ACTOR"]
