# -*- coding: utf-8 -*-
"""ML 能力清单：从**单一来源**机械派生，并与 mecha 出厂 capability map 并表。

依据：宪章 §10.8（capability map 升级为 command map）与 §12.2（"EL 与 ML 的
能力声明能生成可合并 map"）。判据在 ``tests/test_ml_capabilities.py``；生成方式
与"仍是部分并表"的边界记录在 ``docs/mecha/04-ML能力面与双宿主并表.md``。

## 单一来源（本模块不手抄任何命令 / 查询 / 工具名）

=====================  ===================================================
能力类别                声明来源（唯一）
=====================  ===================================================
命令名 / risk           ``ml_mecha.commands.COMMANDS``
命令的 scope /          ``ml_mecha.core_commands.core_command_specs()``
approval_required /     （真正注册进核心 ``CommandRegistry`` 的那份投影；
副作用 bool              两处声明不一致即 fail loud）
只读查询                ``ml_mecha.assembly.READ_QUERIES``
模型可见工具            ``ml_mecha.tools.build_ml_tools(...)`` 的
                        ``ToolDefinition`` 声明
=====================  ===================================================

工具的副作用**不靠名字猜**：由"同名命令声明"与"工具体是否绑定操作者通道"
两条独立证据交叉判定（:func:`declared_side_effect`），不一致即 fail loud——
不生成一份"看起来对"的清单。

## 能力名 = ``kind:declared``（与 mecha 的运行时能力名同构）

mecha 的 ``Surface.register`` / ``CommandRegistry.register`` 把注册项登记为
``query:<name>`` / ``command:<name>``（见 ``mecha/surface.py``、
``mecha/commands.py``）。本模块沿用同一命名律，使两宿主的能力行落在**同一
命名空间**里机械可比：``command:<name>`` 与 ``tool:<name>`` 是两条不同的能力
（一条受控命令、一条模型可见投影），因此不会互相误判为重名冲突。

## 字段（与 mecha ``CapabilityEntry`` 并表）

===================  ====================================================
字段                 含义
===================  ====================================================
name                 能力名（``kind:declared``；mecha 出厂行用它自己的 seam 名）
declared_name        宿主自己的声明名（命令名 / 查询名 / 工具名，逐字可比）
kind                 command / query / tool / seam（seam = mecha 角色声明行）
role                 provider / consumer / definition（mecha 角色词汇）
provider             ML 行固定 ``ml_toolbox``；mecha 行取其 ``providers``
consumer             消费方（``adapter.ai`` / ``adapter.gui``，逗号连接）
side_effect          命令与工具为 bool；查询恒 False；mecha seam 行为 None（未声明）
scope                命令 scope（来自核心投影）；写工具继承同名命令的 scope
risk                 宿主风险分级（``ml_mecha.commands`` 的词汇）
approval_required    是否需审批（来自核心投影）
source               该行由哪一处声明生成（冲突/漂移时用于定位）
===================  ====================================================

## Known Limitations and Deferred Work

- **部分并表**：mecha 出厂 map 由 ``mecha.assembly.build_roles()``（角色表）
  生成，ML 行由本模块从宿主声明生成；两者字段已对齐、可机械 diff，但**还没有
  统一生成器**（mecha 侧的第三路 diff 判据未加，本仓不改 mecha）。
- **工具行没有 mecha 侧对应登记**：``ToolRegistry`` 不逐工具登记角色，故
  ``tool:<name>`` 命名空间是 ML 侧约定（与 ``command:``/``query:`` 同律），
  等 ToolHost 契约落地后由 mecha 侧统一口径。
- **运行时角色行与声明行口径不同**（只对名字集合做对账，不对角色列做 diff）：
  ``command:<name>`` 在 mecha 侧登记为 CONSUMER/consumer=Surface，
  ``query:<name>`` 登记为 CONSUMER/provider=<查询名>；本模块声明的是"谁提供、
  谁消费"（provider=ml_toolbox），二者是同一能力的两面，见 §6 of 04 文档。
- ``consumer`` 里的 ``adapter.gui`` 是**声明**：按框架口径（ADR
  「工具命名放宽与接入口径三条」§Decision 3）它是"**面向人的适配面**"，
  **不要求是一个独立 GUI**。本仓的"人面"就是**人类通道**（assembly 把命令/查询
  scope 同授给 human 通道，判据以 human 通道**真实跑过**命令与查询）⇒ 声明成立。
  ⚠ 早先这里的措辞是"Qt GUI 面板尚未接线"——那句容易被读成"还不合规"，
  按上述口径改成现在这句（Qt 面板是否走这条通道，是**我们**的接线选择，与角色定义无关）；
  工具面只注册给 ai 通道，故工具行只标 ``adapter.ai``。
- 不校验"命令声明的 risk 分得对不对"（那是宿主分类内容，机械不可判）。
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mecha.assembly import build_roles
from mecha.errors import MechaError
from mecha.gate import Channel

from .assembly import READ_QUERIES
from .commands import COMMANDS, RISK_READ, SIDE_EFFECT_NONE
from .core_commands import core_command_specs
from .engine import MLEngine
from .tools import build_ml_tools

#: 本 adapter 的 provider 名（mecha capability map 的 provider 列词汇）。
PROVIDER = "ml_toolbox"

#: 消费方（mecha 出厂 map 的 ``adapter.*`` 词汇）。
CONSUMER_AI = "adapter.ai"      # AI 操作通道（工具面 + 命令/查询面）
CONSUMER_GUI = "adapter.gui"    # 面向人的适配面（本仓 = 人类通道的命令/查询面）

#: 能力分类。``seam`` 专指 mecha 出厂 map 的角色声明行（不是命令/查询/工具）。
KIND_COMMAND = "command"
KIND_QUERY = "query"
KIND_TOOL = "tool"
KIND_SEAM = "seam"

#: ML 行的角色：能力由本 adapter 提供（mecha 行取自己的 definition/provider/consumer）。
ROLE_PROVIDER = "provider"

#: 按 kind 的消费方规则（声明；依据见模块 docstring 的 Known Limitations）。
CONSUMERS_BY_KIND: dict[str, tuple[str, ...]] = {
    KIND_COMMAND: (CONSUMER_AI, CONSUMER_GUI),   # 两条 side 通道都有 scope
    KIND_QUERY: (CONSUMER_AI, CONSUMER_GUI),     # 只读查询不经写权门
    KIND_TOOL: (CONSUMER_AI,),                   # 工具只注册给 ai 通道
}

#: 并表行的完整字段集（三处来源都必须投影成这一套；缺/多字段即 fail loud）。
CAPABILITY_FIELDS: tuple[str, ...] = (
    "name", "declared_name", "kind", "role", "provider", "consumer",
    "side_effect", "scope", "risk", "approval_required", "source",
)

#: mecha map 的表格头（与 ``scripts/gen_capability_map.py`` 逐字同格式）。
TABLE_HEADER = "| 能力 | 角色 | provider | 消费方 |\n|---|---|---|---|\n"

#: mecha 出厂行的 source 标签（并表时用于定位"这一行是谁生成的"）。
MECHA_SOURCE = "mecha.assembly.build_roles"


@dataclass(frozen=True)
class CapabilityConflict(MechaError):
    """并表重名冲突：同一能力名来自两个来源（fail loud，不静默覆盖）。"""

    kind: str = "capability_conflict"


@dataclass(frozen=True)
class CapabilityDrift(MechaError):
    """能力清单与声明漂移（两处声明互相矛盾，或行 schema 不合）。"""

    kind: str = "capability_drift"


# ------------------------------------------------------------------ 命名律
def capability_name(kind: str, declared_name: str) -> str:
    """能力名 = ``kind:declared``（与 mecha 运行时 ``command:<name>`` 同律）。"""
    return f"{kind}:{declared_name}"


# ------------------------------------------------------------------ 派生
def _command_core_specs() -> dict[str, Any]:
    """核心命令投影（唯一来源：``core_command_specs()``），并校验集合一致。"""
    core = {spec.name: spec for spec in core_command_specs()}
    declared = {c.name for c in COMMANDS}
    if set(core) != declared:
        raise CapabilityDrift(
            f"命令声明与核心投影不是同一集合：声明 {sorted(declared)}、"
            f"投影 {sorted(core)}",
            hint="核心 CommandSpec 必须由 ml_mecha.commands.COMMANDS 机械投影"
                 "（core_commands.core_command_specs）；手写的核心命令会让并表"
                 "读到两份不同的命令面",
        )
    return core


def _command_rows(core: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for command in COMMANDS:
        spec = core[command.name]
        host_side_effect = command.side_effect != SIDE_EFFECT_NONE
        if bool(spec.side_effect) != host_side_effect:
            raise CapabilityDrift(
                f"命令 {command.name!r} 的副作用声明不一致：宿主声明 "
                f"{command.side_effect!r}，核心投影 side_effect="
                f"{bool(spec.side_effect)}",
                hint="ml_mecha.commands 与 ml_mecha.core_commands 必须同口径；"
                     "不一致时清单拒绝生成，先裁定哪一处是对的",
            )
        rows.append({
            "name": capability_name(KIND_COMMAND, command.name),
            "declared_name": command.name,
            "kind": KIND_COMMAND,
            "role": ROLE_PROVIDER,
            "provider": PROVIDER,
            "consumer": _join(CONSUMERS_BY_KIND[KIND_COMMAND]),
            "side_effect": host_side_effect,
            "scope": tuple(spec.scope),
            "risk": command.risk,
            "approval_required": bool(spec.approval_required),
            "source": "ml_mecha.commands.COMMANDS",
        })
    return rows


def _query_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in READ_QUERIES:
        rows.append({
            "name": capability_name(KIND_QUERY, name),
            "declared_name": name,
            "kind": KIND_QUERY,
            "role": ROLE_PROVIDER,
            "provider": PROVIDER,
            "consumer": _join(CONSUMERS_BY_KIND[KIND_QUERY]),
            # 只读查询注册进 Surface.query，不经写权门（assembly 的注册路径）
            "side_effect": False,
            "scope": (),
            "risk": RISK_READ,
            "approval_required": False,
            "source": "ml_mecha.assembly.READ_QUERIES",
        })
    return rows


def _binds_operator_channel(tool: Any) -> bool:
    """工具体是否绑定了操作者通道（写路径的**独立**证据）。

    ``ml_mecha.tools.build_ml_tools`` 只给走受控命令的写工具套
    ``_with_channel``（闭包持有 ``Channel``）；只读投影直接用裸函数。这里读
    闭包里的 ``Channel`` 实例，而不是靠工具名前缀猜——名字会漂，闭包不会
    悄悄说谎：若某天读工具也绑通道，判定即 red（fail closed 方向）。
    """
    fn = getattr(tool, "execute", None)
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:      # 未绑定的空 cell
            continue
        if isinstance(value, Channel):
            return True
    return False


def declared_side_effect(tool: Any, *, commands: Sequence[Any] = COMMANDS) -> bool:
    """工具的副作用声明：两条独立证据交叉判定，不一致即 fail loud。

    证据 A：有没有同名受控命令声明（``ml_mecha.commands``），且它声明了副作用；
    证据 B：工具体有没有绑定操作者通道（写路径的唯一入口是受控命令）。

    ``tools.py`` 的纪律是"工具只是投影，不新增第二条写通道"，因此 A、B 必须
    同真同假；任何不一致都说明声明漂移，此时**不生成清单**（不猜、不静默取一方）。
    """
    by_name = {c.name: c for c in commands}
    command = by_name.get(str(getattr(tool, "name", "")))
    from_command = command is not None and command.side_effect != SIDE_EFFECT_NONE
    binds_channel = _binds_operator_channel(tool)
    if binds_channel != from_command:
        raise CapabilityDrift(
            f"工具 {getattr(tool, 'name', '?')!r} 的副作用自相矛盾："
            f"同名命令声明={'有副作用' if from_command else '无（未声明或只读）'}，"
            f"工具体{'绑定了操作者通道' if binds_channel else '未绑定操作者通道'}",
            hint="写工具必须是同名受控命令的投影（ml_mecha.tools 纪律：不新增"
                 "第二条写通道）；两条证据不一致时人工裁定后再生成清单",
        )
    return from_command


def _tool_rows(core: Mapping[str, Any], engine: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    commands = {c.name: c for c in COMMANDS}
    for tool in build_ml_tools(engine):
        side_effect = declared_side_effect(tool)
        command = commands.get(tool.name)
        spec = core.get(tool.name) if command is not None else None
        rows.append({
            "name": capability_name(KIND_TOOL, tool.name),
            "declared_name": tool.name,
            "kind": KIND_TOOL,
            "role": ROLE_PROVIDER,
            "provider": PROVIDER,
            "consumer": _join(CONSUMERS_BY_KIND[KIND_TOOL]),
            "side_effect": side_effect,
            # 写工具继承同名命令的 scope（写权门按命令 scope 判）；只读工具无 scope
            "scope": tuple(spec.scope) if spec is not None else (),
            "risk": command.risk if command is not None else RISK_READ,
            "approval_required": bool(spec.approval_required) if spec is not None else False,
            "source": "ml_mecha.tools.build_ml_tools",
        })
    return rows


def ml_capability_map(*, engine: Any = None) -> list[dict[str, Any]]:
    """ML 能力清单（顺序：命令 → 查询 → 工具，各自按声明顺序）。

    ``engine`` 只用于 ``build_ml_tools`` 的闭包绑定（派生声明、**不执行**任何
    命令）；省略时造一个内存态 ``MLEngine()``，不碰磁盘、不需要装配。
    """
    core = _command_core_specs()
    rows = (_command_rows(core) + _query_rows()
            + _tool_rows(core, engine if engine is not None else MLEngine()))
    _check_unique_names(rows, where="ML 能力清单")
    return rows


def _check_unique_names(rows: Iterable[Mapping[str, Any]], *, where: str) -> None:
    seen: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        name = str(row["name"])
        if name in seen:
            raise CapabilityConflict(
                f"{where}里能力名 {name!r} 重复："
                f"{seen[name]['source']} 与 {row['source']}",
                hint="同一 kind 下的能力名必须唯一；重复声明会让并表静默丢一行",
            )
        seen[name] = row


# ------------------------------------------------------------------ mecha 行
def _join(values: Sequence[str]) -> str:
    return ", ".join(v for v in values if v)


def _as_tuple(value: Any) -> tuple[str, ...]:
    """``providers``/``consumers`` 的三种形态（序列 / 逗号串 / "—"）归一。"""
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
    elif isinstance(value, Sequence):
        parts = [str(v).strip() for v in value]
    else:
        parts = [str(value).strip()]
    return tuple(p for p in parts if p and p != "—")


def _role_value(role: Any) -> str:
    return str(getattr(role, "value", role) or "")


def _entry_name(entry: Any) -> str:
    """取能力行的名字（``CapabilityEntry`` 或同形映射都吃）。"""
    if isinstance(entry, Mapping):
        return str(entry.get("name", ""))
    return str(getattr(entry, "name", ""))


def mecha_capability_rows(entries: Any = None) -> list[dict[str, Any]]:
    """把 mecha 出厂 capability map 行适配到同一 schema（只读 mecha，不改它）。

    ``entries`` 省略时取 ``mecha.assembly.build_roles().entries()``（出厂声明）；
    也可传 EL / 第三宿主生成的真表条目（``CapabilityEntry`` 或同形映射）。
    """
    if entries is None:
        entries = build_roles().entries()
    return [_adapt_mecha_row(entry) for entry in entries]


def _adapt_mecha_row(entry: Any) -> dict[str, Any]:
    if isinstance(entry, Mapping):
        name = str(entry.get("name", ""))
        role = _role_value(entry.get("role", ""))
        providers = _as_tuple(entry.get("providers", entry.get("provider")))
        consumers = _as_tuple(entry.get("consumers", entry.get("consumer")))
    else:
        name = str(getattr(entry, "name", ""))
        role = _role_value(getattr(entry, "role", ""))
        providers = _as_tuple(getattr(entry, "providers", None))
        consumers = _as_tuple(getattr(entry, "consumers", None))
    if not name:
        raise CapabilityDrift(
            "mecha 能力行缺 name，无法并表",
            hint="CapabilityEntry 的 name 必须非空（RoleRegistry.register 的守卫）",
        )
    return {
        "name": name,
        "declared_name": name,
        "kind": KIND_SEAM,
        "role": role,
        "provider": _join(providers),
        "consumer": _join(consumers),
        "side_effect": None,        # mecha 出厂行是角色声明，不是命令声明
        "scope": (),
        "risk": "",
        "approval_required": None,
        "source": MECHA_SOURCE,
    }


# ------------------------------------------------------------------ 并表
def _check_row_schema(row: Any) -> dict[str, Any]:
    """并表前提：行必须与 ``CAPABILITY_FIELDS`` 逐字同 schema（缺/多都拒）。"""
    if not isinstance(row, Mapping):
        raise CapabilityDrift(
            f"并表行不是映射：{type(row).__name__}",
            hint="并表只接受 ml_capability_map()/mecha_capability_rows() 同形行",
        )
    keys = set(row)
    expected = set(CAPABILITY_FIELDS)
    if keys != expected:
        raise CapabilityDrift(
            f"并表行 {row.get('name', '?')!r} 字段不合 schema："
            f"多 {sorted(keys - expected)}、缺 {sorted(expected - keys)}",
            hint="两宿主并表的前提是同一字段集（CAPABILITY_FIELDS）；"
                 "字段漂移时先对齐 schema，不要就地补默认值",
        )
    return dict(row)


def merge_with_mecha(capability_rows: Any = None, *,
                     mecha_entries: Any = None) -> list[dict[str, Any]]:
    """把 ML 行与 mecha 出厂行按同一 schema 合并，按 ``name`` 排序。

    - ``capability_rows`` 省略时用 :func:`ml_capability_map()`；
    - ``mecha_entries`` 省略时用 mecha 出厂角色表（``build_roles().entries()``）；
      显式传 ``[]`` = 只要 ML 行（测试与"先看 ML 一侧"的场景）；
    - **重名即 fail loud**（:class:`CapabilityConflict`）：并表不许静默覆盖，
      同名意味着两个来源都声称拥有该能力，必须人工裁定归属。
    """
    rows = list(ml_capability_map() if capability_rows is None else capability_rows)
    merged: dict[str, dict[str, Any]] = {}
    for row in rows + mecha_capability_rows(mecha_entries):
        checked = _check_row_schema(row)
        name = str(checked["name"])
        existing = merged.get(name)
        if existing is not None:
            raise CapabilityConflict(
                f"能力名 {name!r} 重名冲突："
                f"{existing['kind']}/{existing['role']} "
                f"provider={existing['provider'] or '—'}（{existing['source']}）与 "
                f"{checked['kind']}/{checked['role']} "
                f"provider={checked['provider'] or '—'}（{checked['source']}）",
                hint="重名必须人工裁定归属（哪个宿主拥有该能力），"
                     "并表不许静默覆盖；同一能力的两条投影请用不同 kind 前缀",
            )
        merged[name] = checked
    return [merged[name] for name in sorted(merged)]


def render_rows(rows: Any = None, *, with_mecha_rows: bool = False) -> str:
    """渲染成 mecha 同格式的 markdown 表（``gen_capability_map.render`` 逐字同格式）。

    列 = 能力 / 角色 / provider / 消费方；额外字段（kind/side_effect/scope/risk/
    approval_required）留在行 dict 里，供程序化 diff 用（markdown 只承载并表四列）。
    """
    if rows is None:
        rows = merge_with_mecha() if with_mecha_rows else ml_capability_map()
    elif with_mecha_rows:
        rows = merge_with_mecha(rows)
    lines = [TABLE_HEADER]
    for row in sorted(rows, key=lambda r: str(r["name"])):
        lines.append(f"| `{row['name']}` | {row['role']} | "
                     f"{row['provider'] or '—'} | {row['consumer'] or '—'} |\n")
    return "".join(lines)


def render_markdown(rows: Any = None, *, with_mecha_rows: bool = False) -> str:
    """完整文档（含"禁止手工编辑"头 + 表格），供未来入库 ``capabilities.md``。"""
    head = (
        "<!-- 本文件由 ml_mecha/capabilities.py 从 ML 声明生成，禁止手工编辑。\n"
        "     字段格式与 mecha 出厂 map 相同（gen_capability_map.render 同格式），\n"
        "     两宿主并表见 docs/mecha/04-ML能力面与双宿主并表.md。 -->\n"
        "# ML Capability Map（生成）\n\n"
    )
    return head + render_rows(rows, with_mecha_rows=with_mecha_rows)


# ------------------------------------------------------------------ 运行时对账
def diff_with_runtime(entries: Any, *, rows: Any = None) -> dict[str, list[str]]:
    """把 ML 声明行与装配实例的角色表（``Software.roles.entries()``）对账。

    只对 ``command:`` / ``query:`` 两个命名空间做**名字集合**对账：

    - ``declared_not_registered``：声明了但运行时没注册 = 幽灵行（清单撒谎）；
    - ``registered_not_declared``：运行时注册了但清单没声明 = 漏声明。

    工具不逐条进 mecha ``RoleRegistry``（没有 ``tool:`` 登记口径），工具行的
    对账由 ``ToolRegistry.schemas()`` 承担（见 tests/test_ml_capabilities.py）。
    角色列不做 diff：运行时登记的是"谁在消费"（CONSUMER/Surface），本清单声明的
    是"谁提供、谁消费"，口径不同（见 04 文档 §6），强行比会制造假红。
    """
    declared = {
        str(row["name"]) for row in (ml_capability_map() if rows is None else rows)
        if row["kind"] in (KIND_COMMAND, KIND_QUERY)
    }
    registered = {_entry_name(entry) for entry in entries}
    host_registered = {
        name for name in registered
        if name.startswith((f"{KIND_COMMAND}:", f"{KIND_QUERY}:"))
    }
    return {
        "declared_not_registered": sorted(declared - host_registered),
        "registered_not_declared": sorted(host_registered - declared),
        "registered": sorted(host_registered),
    }


__all__ = [
    "PROVIDER", "CONSUMER_AI", "CONSUMER_GUI",
    "KIND_COMMAND", "KIND_QUERY", "KIND_TOOL", "KIND_SEAM", "ROLE_PROVIDER",
    "CONSUMERS_BY_KIND", "CAPABILITY_FIELDS", "TABLE_HEADER", "MECHA_SOURCE",
    "CapabilityConflict", "CapabilityDrift",
    "capability_name", "declared_side_effect", "ml_capability_map",
    "mecha_capability_rows", "merge_with_mecha", "render_rows", "render_markdown",
    "diff_with_runtime",
]
