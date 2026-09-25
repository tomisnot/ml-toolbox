# -*- coding: utf-8 -*-
"""ML 能力清单与双宿主并表的判据（宪章 §10.8 / §12.2 的可执行版本）。

九组断言：

1. 清单恰好覆盖 4 条命令 + 4 条只读查询 + 7 个工具，无缺无多——且与"声明源、
   核心注册表、Surface 注册项、ToolRegistry.schemas()"多条独立路径交叉核对；
2. 命令行的 scope / side_effect / approval_required 与 ``core_command_specs()``
   （以及真正注册进核心的那份 spec）逐字一致；
3. 工具行与 ``ToolRegistry.schemas()`` 的工具名一致；副作用由"同名命令声明"与
   "工具体是否绑定操作者通道"两条独立证据交叉判定；
4. 交叉判定守卫不是哑守卫（L7 对偶）：合成的反例两个方向都必须被抓住；
5. 并表：与 mecha 出厂行合并、按 name 排序、字段集同一、行数守恒；
6. 渲染格式与 mecha 自己的 ``gen_capability_map.render`` **逐字同格式**（并表
   的文本层前提，不是"看起来像"）；
7. 重名冲突 fail loud（含"与第二宿主行重名""ML 行内部重名""schema 不合"三种）；
8. 运行时对账：声明的能力行必须真在装配实例的角色表里（幽灵行 = 红），且运行时
   为 ML 登记的 command:/query: 行没有漏声明；
9. 静态：``ml_mecha/capabilities.py`` 不手抄命令/查询/工具名单，也不含 ML 领域
   方法名（logistic / random_forest）。

运行：``python -m pytest tests/test_ml_capabilities.py -q``
也可直接 ``python tests/test_ml_capabilities.py``（自带 main）。

## 环境要求

mecha 必须在 ``sys.path`` 上：优先用环境变量 ``MECHA_ROOT``，否则回退到
``D:\\code-nosync\\mecha``。**不 vendor mecha**（两个仓各自独立演进）。
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ---- mecha 上 sys.path（不 vendor；环境变量优先，默认走同级仓） ----
MECHA_ROOT = Path(os.environ.get("MECHA_ROOT", r"D:\code-nosync\mecha"))
if MECHA_ROOT.is_dir() and str(MECHA_ROOT) not in sys.path:
    sys.path.insert(0, str(MECHA_ROOT))
# gen_capability_map.py 是 mecha 自己的生成器：它的 render 就是并表格式的判据，
# 借它来证明"我们的表与 mecha 的表逐字同格式"，而不是自己写个"看起来像"的解析。
if (MECHA_ROOT / "scripts").is_dir() and str(MECHA_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(MECHA_ROOT / "scripts"))

import gen_capability_map                              # noqa: E402
from mecha.assembly import build_roles                 # noqa: E402
from mecha.gate import Channel                         # noqa: E402
from mecha.roles import CapabilityEntry, Role          # noqa: E402
from mecha.tools import define_tool                    # noqa: E402

from ml_mecha.assembly import READ_QUERIES, assemble_ml_mecha   # noqa: E402
from ml_mecha.capabilities import (CAPABILITY_FIELDS,           # noqa: E402
                                   CapabilityConflict, CapabilityDrift,
                                   KIND_COMMAND, KIND_QUERY, KIND_SEAM,
                                   KIND_TOOL, declared_side_effect,
                                   diff_with_runtime, merge_with_mecha,
                                   ml_capability_map, render_markdown,
                                   render_rows)
from ml_mecha.commands import COMMANDS, SIDE_EFFECT_NONE        # noqa: E402
from ml_mecha.core_commands import core_command_specs           # noqa: E402
from ml_mecha.engine import MLEngine                            # noqa: E402
from ml_mecha.tools import build_ml_tools                       # noqa: E402

CAPABILITIES_PY = REPO / "ml_mecha" / "capabilities.py"


# ---------------------------------------------------------------- 夹具
@pytest.fixture()
def app(tmp_path):
    """装配一台 LOCKED 的 ML×mecha 实例（每个测试独立数据目录 → 独立写租约）。"""
    instance = assemble_ml_mecha(root=tmp_path, seed=13)
    try:
        yield instance
    finally:
        instance.close()


def _rows(kind: str) -> list[dict]:
    return [row for row in ml_capability_map() if row["kind"] == kind]


def _tool_names() -> list[str]:
    return [tool.name for tool in build_ml_tools(MLEngine())]


def _write_command_names() -> set[str]:
    return {c.name for c in COMMANDS if c.side_effect != SIDE_EFFECT_NONE}


def _fake_tool(name: str, *, binds: bool):
    """合成工具声明：``binds=True`` 时闭包真的持有 ``Channel``（写路径证据）。

    注意 ``binds=False`` 的分支必须在**另一个函数体**里：Python 的自由变量按
    词法判定，同一个函数体里提到 ``channel`` 就会让它进闭包，合成反例就废了。
    """
    if not binds:
        def execute(**kwargs):
            return {"ok": True}
    else:
        channel = Channel("ml-cap-test", "ai")

        def execute(**kwargs):
            return {"ok": True, "actor": channel.actor}

    return define_tool(
        name=name, description="能力清单判据用的合成工具声明", parameters={},
        output_schema={"type": "object", "required": ["ok"]}, execute=execute)


# ---------------------------------------------------------------- 1. 覆盖面
def test_map_covers_commands_queries_tools_exactly(app):
    """清单 = 4 命令 + 4 查询 + 11 工具；每条都与独立来源交叉核对，无缺无多。"""
    rows = ml_capability_map()

    for row in rows:
        assert set(row) == set(CAPABILITY_FIELDS), row["name"]
    names = [row["name"] for row in rows]
    assert len(names) == len(set(names)), "清单内部重名"

    # ---- 命令：与声明表 / 核心投影 / 已注册核心注册表三路交叉 ----
    declared = [c.name for c in COMMANDS]
    projected = [spec.name for spec in core_command_specs()]
    registered = app.software.commands.names()
    command_rows = _rows(KIND_COMMAND)
    assert [row["declared_name"] for row in command_rows] == declared
    assert set(declared) == set(projected) == set(registered)
    assert len(command_rows) == 4

    # ---- 查询：与 READ_QUERIES / Surface 注册项两路交叉 ----
    query_rows = _rows(KIND_QUERY)
    assert [row["declared_name"] for row in query_rows] == list(READ_QUERIES)
    assert len(query_rows) == 4
    assert {e.name for e in app.software.surface.entries("query")} == set(READ_QUERIES)
    for row in query_rows:
        assert row["side_effect"] is False and row["scope"] == ()

    # ---- 工具：与 build_ml_tools 声明 / ToolRegistry.schemas() 两路交叉 ----
    tool_rows = _rows(KIND_TOOL)
    tool_names = _tool_names()
    schema_names = [schema["name"] for schema in app.tools.schemas()]
    assert [row["declared_name"] for row in tool_rows] == tool_names
    assert sorted(tool_names) == schema_names
    assert len(tool_rows) == 11

    assert len(rows) == 19


# ---------------------------------------------------------------- 2. 命令字段
def test_command_rows_match_core_projection(app):
    """命令行的 scope/side_effect/approval 与核心投影、与已注册 spec 一致。"""
    core = {spec.name: spec for spec in core_command_specs()}
    risks = {c.name: c.risk for c in COMMANDS}
    for row in _rows(KIND_COMMAND):
        spec = core[row["declared_name"]]
        assert row["scope"] == tuple(spec.scope)
        assert row["side_effect"] == bool(spec.side_effect)
        assert row["approval_required"] == bool(spec.approval_required)
        assert row["risk"] == risks[row["declared_name"]]

        # 真正注册进核心注册表的那份 spec 也要同口径（不是只跟投影函数对齐）
        live = app.software.commands.spec(row["declared_name"])
        assert live is not None, row["declared_name"]
        assert tuple(live.scope) == row["scope"]
        assert bool(live.side_effect) == row["side_effect"]
        assert bool(live.approval_required) == row["approval_required"]

        # 有副作用的命令必须有 scope（否则 AI 写权无处可拒——fail closed 的前提）
        if row["side_effect"]:
            assert row["scope"], f"有副作用的命令 {row['name']} 没有 scope"
        assert row["provider"] == "ml_toolbox"
        consumers = set(row["consumer"].split(", "))
        assert consumers and consumers <= {"adapter.ai", "adapter.gui"}


# ---------------------------------------------------------------- 3. 工具字段
def test_tool_rows_match_registry_schemas(app):
    """工具行与 ToolRegistry.schemas() 一致；写工具继承同名命令的 scope/风险。"""
    schema_names = {schema["name"] for schema in app.tools.schemas()}
    core = {spec.name: spec for spec in core_command_specs()}
    write_commands = _write_command_names()

    tool_rows = _rows(KIND_TOOL)
    for row in tool_rows:
        assert row["declared_name"] in schema_names
        assert row["side_effect"] == (row["declared_name"] in write_commands)
        assert row["consumer"] == "adapter.ai"       # 工具面只注册给 ai 通道
        if row["side_effect"]:
            spec = core[row["declared_name"]]
            assert row["scope"] == tuple(spec.scope)
            assert row["approval_required"] == bool(spec.approval_required)
        else:
            assert row["scope"] == ()
    write_rows = {row["declared_name"] for row in tool_rows if row["side_effect"]}
    assert write_rows == write_commands & set(_tool_names())
    assert len(write_rows) == 4                       # 4 写工具 / 7 只读或调度面
    # job 调度面（submit_run/read_job/cancel_run）**没有同名命令**，故不计入
    # write_rows；submit_run 的写路径在它提交的 run_method 命令里（见 04 文档
    # 「job 调度面」小节）。这里显式钉住这个事实，避免哪天被误当成只读。
    assert {"submit_run", "read_job", "cancel_run"} <= set(_tool_names())
    assert not ({"submit_run", "read_job", "cancel_run"} & write_rows)


# ---------------------------------------------------------------- 4. L7 对偶
def test_declared_side_effect_cross_check_fails_loud():
    """副作用的两条证据必须互相印证；合成反例两个方向都要被抓住。"""
    tools = {tool.name: tool for tool in build_ml_tools(MLEngine())}
    assert declared_side_effect(tools["run_method"]) is True
    assert declared_side_effect(tools["describe_run"]) is False

    # 反例 A：绑定了操作者通道却没有同名命令声明（疑似第二条写通道）
    ghost_write = _fake_tool("audit_widget", binds=True)
    with pytest.raises(CapabilityDrift) as exc_a:
        declared_side_effect(ghost_write)
    assert exc_a.value.kind == "capability_drift"
    assert "audit_widget" in exc_a.value.message

    # 反例 B：名字撞上有副作用的命令，却根本没绑通道（声明漂移）
    fake_read = _fake_tool("run_method", binds=False)
    with pytest.raises(CapabilityDrift) as exc_b:
        declared_side_effect(fake_read)
    assert exc_b.value.kind == "capability_drift"

    # 不误报：真只读投影（未绑通道、无同名写命令）判为无副作用
    honest_read = _fake_tool("describe_widget", binds=False)
    assert declared_side_effect(honest_read) is False


# ---------------------------------------------------------------- 5. 并表
def test_merge_with_mecha_merges_and_sorts():
    """与 mecha 出厂行按同一 schema 合并：按 name 排序、行数守恒、mecha 行可读。"""
    ml_rows = ml_capability_map()
    merged = merge_with_mecha()
    names = [row["name"] for row in merged]

    assert names == sorted(names)
    assert len(merged) == len(ml_rows) + len(build_roles().entries())
    assert set(names) >= {row["name"] for row in ml_rows}
    # mecha 出厂脊椎件在同表里（两宿主真的进了同一张表）
    assert {"gate", "history", "surface", "monitor", "authority"} <= set(names)
    for row in merged:
        assert set(row) == set(CAPABILITY_FIELDS), row["name"]

    # mecha 行是角色声明行：没有副作用/scope/审批声明（None，不是猜的 bool）
    gate = next(row for row in merged if row["name"] == "gate")
    assert gate["kind"] == KIND_SEAM
    assert gate["role"] == "definition"
    assert gate["provider"] == "mecha"
    assert gate["side_effect"] is None
    assert gate["scope"] == ()
    assert gate["approval_required"] is None
    adapter_ai = next(row for row in merged if row["name"] == "adapter.ai")
    assert adapter_ai["role"] == "consumer" and adapter_ai["consumer"] == "ai"

    # 只要 ML 行时显式给空表（不静默回退到 mecha 出厂表）
    only_ml = merge_with_mecha(mecha_entries=[])
    assert [row["name"] for row in only_ml] == sorted(r["name"] for r in ml_rows)


# ---------------------------------------------------------------- 6. 文本格式
def test_render_rows_is_same_format_as_mecha_renderer():
    """渲染出的表与 mecha 自己的 render 逐字同格式（文本层并表的前提）。"""
    rows = ml_capability_map()
    ours = [line for line in render_rows(rows).splitlines() if line.startswith("|")]

    entries = [
        CapabilityEntry(
            str(row["name"]), Role(str(row["role"])),
            tuple(p for p in str(row["provider"]).split(", ") if p),
            tuple(c for c in str(row["consumer"]).split(", ") if c))
        for row in sorted(rows, key=lambda r: r["name"])
    ]
    theirs = [line for line in gen_capability_map.render(entries).splitlines()
              if line.startswith("|")]
    assert ours == theirs, (ours[:3], theirs[:3])

    # 生成的文档带"禁止手工编辑"标记（未来入库时不许手改）
    doc = render_markdown()
    assert "禁止手工编辑" in doc
    assert [line for line in doc.splitlines() if line.startswith("|")] == ours


# ---------------------------------------------------------------- 7. 冲突 fail loud
def test_merge_conflict_fails_loud():
    """重名/schema 不合一律 fail loud，不静默覆盖。"""
    rows = ml_capability_map()

    # A. 与第二宿主（EL）的行重名：同名能力来自两个 provider
    clash = CapabilityEntry("command:run_method", Role.PROVIDER,
                            ("el_project",), ("adapter.gui",))
    with pytest.raises(CapabilityConflict) as exc:
        merge_with_mecha(rows, mecha_entries=[clash])
    assert exc.value.kind == "capability_conflict"
    assert "command:run_method" in exc.value.message
    assert "el_project" in exc.value.message        # 冲突双方都进消息（不静默取一方）
    assert "ml_toolbox" in exc.value.message

    # B. 传入行内部重名（同一能力被声明两次）
    with pytest.raises(CapabilityConflict):
        merge_with_mecha(list(rows) + [dict(rows[0])], mecha_entries=[])

    # C. 字段不合 schema 的行不许混入并表
    with pytest.raises(CapabilityDrift) as exc_c:
        merge_with_mecha([{"name": "command:whatever"}], mecha_entries=[])
    assert exc_c.value.kind == "capability_drift"

    # D. 不误报：合法并表照常成功（守卫不是永远红的哑守卫）
    ok = merge_with_mecha(rows, mecha_entries=[])
    assert len(ok) == len(rows)


# ---------------------------------------------------------------- 8. 运行时对账
def test_runtime_role_table_matches_declaration(app):
    """声明行必须真在装配实例的角色表里（幽灵行 = 红）；运行时登记无漏声明。"""
    diff = diff_with_runtime(app.software.roles.entries())
    declared = {row["name"] for row in ml_capability_map()
                if row["kind"] in (KIND_COMMAND, KIND_QUERY)}
    assert diff["declared_not_registered"] == []
    assert diff["registered_not_declared"] == []
    assert set(diff["registered"]) == declared
    assert len(diff["registered"]) == 8              # 4 命令 + 4 查询

    # L7 对偶：合成一条运行时没有的声明，必须被抓住（守卫不哑）
    ghost = dict(ml_capability_map()[0])
    ghost["name"] = "command:no_such_command"
    ghost_diff = diff_with_runtime(app.software.roles.entries(), rows=[ghost])
    assert ghost_diff["declared_not_registered"] == ["command:no_such_command"]


# ---------------------------------------------------------------- 9. 静态
def test_static_capabilities_module_does_not_hand_copy_names():
    """静态：capabilities.py 不手抄命令/查询/工具名单，也不含 ML 领域方法名。"""
    text = CAPABILITIES_PY.read_text(encoding="utf-8")
    for word in ("logistic", "random_forest"):
        assert word not in text, f"capabilities.py 出现领域方法名 {word!r}"

    declared = ({c.name for c in COMMANDS} | set(READ_QUERIES) | set(_tool_names()))
    assert declared, "声明面为空？"

    tree = ast.parse(text, filename=str(CAPABILITIES_PY))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.value not in docstrings}
    offenders = sorted(literals & declared)
    assert offenders == [], f"清单模块里出现了硬编码的能力名：{offenders}"


# ---------------------------------------------------------------- 直跑入口
def main() -> int:
    """不装 pytest 时的直跑入口（与仓内其他 test_*.py 同形）。"""
    # 默认控制台可能是 GBK（本机 cp936）：✓/✗ 会让直跑入口在第一个测试之前
    # 就崩掉（exit 1），真实失败被编码崩溃掩盖（最终验收审查 R18/D14）。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import tempfile

    tests = [
        test_map_covers_commands_queries_tools_exactly,
        test_command_rows_match_core_projection,
        test_tool_rows_match_registry_schemas,
        test_declared_side_effect_cross_check_fails_loud,
        test_merge_with_mecha_merges_and_sorts,
        test_render_rows_is_same_format_as_mecha_renderer,
        test_merge_conflict_fails_loud,
        test_runtime_role_table_matches_declaration,
        test_static_capabilities_module_does_not_hand_copy_names,
    ]
    failures = []
    for fn in tests:
        try:
            if "app" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory(prefix="ml-cap-test-") as td:
                    instance = assemble_ml_mecha(root=td, seed=13)
                    try:
                        fn(instance)
                    finally:
                        instance.close()
            else:
                fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((fn.__name__, exc))
            print(f"  ✗ {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ✓ {fn.__name__}")
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
