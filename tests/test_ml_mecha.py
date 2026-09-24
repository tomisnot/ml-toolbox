# -*- coding: utf-8 -*-
"""ml_mecha 薄 host adapter 的判据（宪章 §12 验收判据的可执行版本）。

八组断言（对应任务书第 6 条）：

1. 只读工具投影（命名律、白名单、required、错误归一化）；
2. human / ai Authority 门（LOCKED fail closed、side 不扩权）；
3. ``run_method`` 端到端（回执必填键 + 紧凑性）；
4. 资源守卫（预算拒绝不逃逸）；
5. History 归因（actor / call_id / 只存引用）；
6. Monitor 对账（健康读数不存疑；故意不一致的 Claim 必须 disputed）；
7. 静态断言（不 import EL、不复制 EL 命令名、ML 键不进 mecha 核心）。

运行：``python -m pytest tests/test_ml_mecha.py -q``
也可直接 ``python tests/test_ml_mecha.py``（自带 main）。

## 环境要求

mecha 必须在 ``sys.path`` 上：优先用环境变量 ``MECHA_ROOT``，否则回退到
``D:\\code-nosync\\mecha``。**不 vendor mecha**（两个仓各自独立演进）。
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ---- mecha 上 sys.path（不 vendor；环境变量优先，默认走同级仓） ----
MECHA_ROOT = Path(os.environ.get("MECHA_ROOT", r"D:\code-nosync\mecha"))
if MECHA_ROOT.is_dir() and str(MECHA_ROOT) not in sys.path:
    sys.path.insert(0, str(MECHA_ROOT))

from mecha.authority import Mode                    # noqa: E402
from mecha.errors import GateDenied                 # noqa: E402
from mecha.gate import Channel                      # noqa: E402
from mecha.monitor import Claim, Monitor            # noqa: E402
from mecha.tools import CURRENT_CALL_ID, check_tool_name  # noqa: E402

from ml_mecha.assembly import READ_QUERIES, assemble_ml_mecha  # noqa: E402
from ml_mecha.engine import (RECEIPT_REQUIRED_KEYS, clear_channel,  # noqa: E402
                             use_channel)
from ml_mecha.summarizer import ml_summarizer       # noqa: E402
from ml_mecha.validator import STATE_KEYS           # noqa: E402

TOOL_NAMES = ("describe_methods", "describe_method", "describe_dataset",
              "describe_run", "prepare_dataset", "run_method",
              "compare_methods")


# ---------------------------------------------------------------- 夹具
def _frame(n=160, d=5, seed=11):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    frame["target"] = y
    return frame


@pytest.fixture()
def app(tmp_path):
    """装配一台 LOCKED 的 ML×mecha 实例（每个测试独立数据目录 → 独立写租约）。"""
    instance = assemble_ml_mecha(root=tmp_path, seed=13)
    try:
        yield instance
    finally:
        instance.close()


def _as(app, channel, spec, context=None):
    """以指定通道执行一条受控命令（显式绑定通道，不走任何默认值）。"""
    token = use_channel(channel)
    try:
        return app.engine.run(spec, context=context)
    finally:
        clear_channel(token)


def _prepare(app, channel, **source):
    spec = {"command": "prepare_dataset", "name": "demo",
            "source": {"kind": "frame", "frame": _frame(), "target": "target"}}
    spec["source"].update(source)
    return _as(app, channel, spec)


def _prepared(app):
    """人类侧准备好数据并开 AI 闸，返回 dataset_id。"""
    app.switch_human()
    receipt = _prepare(app, app.human_channel)
    assert receipt["ok"], receipt
    app.switch_ai()
    return receipt["dataset_id"]


# ---------------------------------------------------------------- 1. 工具投影
def test_tool_projection_is_read_only_capable(app):
    """工具面：命名合法、白名单投影、read 工具在 LOCKED 下可用。"""
    names = [t["name"] for t in app.tools.schemas()]
    assert names == sorted(TOOL_NAMES), names
    for name in names:
        check_tool_name(name)            # 命名律：动词_宾语，无 get_/list_/mecha_
        assert not name.startswith(("get_", "list_", "mecha_"))

    # schemas() 只出白名单字段（execute/output/render 绝不泄漏）
    for schema in app.tools.schemas():
        assert set(schema) == {"name", "description", "parameters"}

    # 每个工具显式声明 required
    for name in TOOL_NAMES:
        tool = app.tools.get(name)
        assert tool is not None
        required = tool.output.schema.get("required")
        assert isinstance(required, list) and required, f"{name} 缺 required"

    # LOCKED 下只读工具照常可用（只读不经写权门）
    assert app.mode is Mode.LOCKED
    result = app.tools.execute("describe_methods", {"family": "linear"})
    assert result["is_error"] is False
    assert result["value"]["method_count"] > 0
    result = app.tools.execute("describe_method", {"name": "logistic"})
    assert result["is_error"] is False
    assert result["value"]["param_schema"]


def test_tool_errors_normalized_to_tool_failure(app):
    """错误归一化：未知方法/未知参数/缺参数都是结构化 ToolFailure，不抛穿。"""
    # 方法名拼错 → 可教学的 unknown_key（带 did_you_mean）
    bad = app.tools.execute("describe_method", {"name": "logistc"})
    assert bad["is_error"] is True
    assert set(bad["error"]) == {"message", "info"}
    assert bad["error"]["info"]["kind"] == "unknown_key"
    assert bad["error"]["info"]["suggest"]

    # 缺必填参数 → missing_arguments（裸 TypeError 不许逃）
    missing = app.tools.execute("run_method", {})
    assert missing["is_error"] is True
    assert missing["error"]["info"]["kind"] == "missing_arguments"

    # 只读 tool 的回执必填键由 output.schema.required 声明
    tool = app.tools.get("describe_run")
    assert "run_id" in tool.output.schema["required"]


# ---------------------------------------------------------------- 2. Authority 门
def test_authority_gate_locked_denies_writes(app):
    """出厂 LOCKED：写命令 fail closed，且被拒后不留半拉子状态。"""
    receipt = _prepare(app, app.human_channel, frame=_frame())
    assert receipt["ok"] is False
    assert receipt["error_kind"] == "authority_locked"
    assert app.engine.dataset_ids() == []            # 回滚：内存里也没有

    # 人类侧先正常准备一份数据，再锁回去：证明"被拒"是写权门在拦，
    # 不是别的地方先报错（否则这条断言是假绿）。
    app.switch_human()
    dataset_id = _prepare(app, app.human_channel)["dataset_id"]
    events_before = len(app.history.events())
    app.lock()
    assert app.mode is Mode.LOCKED

    run = _as(app, app.ai_channel,
              {"command": "run_method", "method": "logistic",
               "dataset_id": dataset_id})
    assert run["ok"] is False
    assert run["error_kind"] == "authority_locked"
    assert len(app.history.events()) == events_before   # 被拒的写不进史


def test_authority_side_switch_cannot_be_forged(app):
    """AI 侧不能自解锁；开闸只有人类侧能做（side 在通道构造时钉死）。"""
    with pytest.raises(GateDenied) as exc:
        app.software.authority.switch_mode(Mode.AI, app.ai_channel.actor)
    assert exc.value.kind == "authority_switch_denied"

    app.switch_human()
    dataset_id = _prepare(app, app.human_channel)["dataset_id"]
    app.switch_ai()

    # AI 通道可写
    ai_receipt = _as(app, app.ai_channel,
                     {"command": "run_method", "method": "logistic",
                      "dataset_id": dataset_id})
    assert ai_receipt["ok"] is True

    # 人类通道在此刻被拒（mode=AI 只接受 ai 侧）
    human_receipt = _as(app, app.human_channel,
                        {"command": "run_method", "method": "svc",
                         "dataset_id": dataset_id})
    assert human_receipt["ok"] is False
    assert human_receipt["error_kind"] == "authority_mode_mismatch"

    # 标签不扩写权面：mode=HUMAN 时，标签写成人类但 side='ai' 的通道仍被拒
    app.switch_human()
    assert app.mode is Mode.HUMAN
    labelled_human = Channel("ml-human", "ai")
    labelled = _as(app, labelled_human,
                   {"command": "run_method", "method": "svc",
                    "dataset_id": dataset_id})
    assert labelled["ok"] is False
    assert labelled["error_kind"] == "authority_mode_mismatch"

    # 人类侧正常可写（证明上一条拒绝来自 side 而不是别处）
    assert _as(app, app.human_channel,
               {"command": "run_method", "method": "svc",
                "dataset_id": dataset_id})["ok"] is True


# ---------------------------------------------------------------- 3. 端到端
def test_read_queries_are_registered_in_surface(app):
    """只读查询注册进 mecha Surface，经 ``Surface.query`` 分派（不经写权门）。"""
    names = {e.name for e in app.software.surface.entries("query")}
    assert names == set(READ_QUERIES), names
    # 注册项可读，且引擎查询统一走这条路（回退路径仅裸引擎/单测使用）
    catalog = app.software.surface.query("method_catalog")
    assert catalog["method_count"] > 0
    assert app.engine.query("method_catalog")["method_count"] == catalog["method_count"]
    assert app.engine.query("method_detail", name="logistic")["name"] == "logistic"


def test_commands_registered_in_core_registry(app):
    """ML 4 条受控命令注册进核心 CommandRegistry（双宿主消费核心机制）。"""
    names = set(app.software.commands.names())
    assert names == {"prepare_dataset", "run_method", "run_method_batch",
                     "compare_methods"}
    spec = app.software.commands.spec("run_method")
    assert spec is not None and spec.side_effect is True
    assert set(spec.output_schema["required"]) == set(RECEIPT_REQUIRED_KEYS)
    # prepare_dataset 的必填参数来自 ML 声明，核心不持有 ML 领域语义
    prep = app.software.commands.spec("prepare_dataset")
    assert "source" in prep.parameters["required"]


def test_write_tool_failures_are_tool_failure(app):
    """写工具失败必须是 ``is_error=True`` + ToolFailure kind，不是假成功。"""
    csv_path = Path(app.layout.root) / "outside.csv"
    _frame().to_csv(csv_path, index=False)
    app.switch_human()
    result = app.tools.execute("prepare_dataset", {
        "source": {"kind": "csv", "path": str(csv_path), "target": "target"}})
    assert result["is_error"] is True
    assert result["error"]["info"]["kind"] == "dataset_roots_required"

    # 未知方法同样归一化（不是 value.ok=False 的假成功）
    dataset_id = _prepare(app, app.human_channel)["dataset_id"]
    app.switch_ai()
    missing = app.tools.execute("run_method", {"method": "no_such_method",
                                                "dataset_id": dataset_id})
    assert missing["is_error"] is True
    assert missing["error"]["info"]["kind"] == "unknown_key"


def test_run_method_end_to_end_and_receipt_contract(app):
    """``run_method`` 端到端：回执必填键齐、指标可读、大对象不内联。"""
    dataset_id = _prepared(app)
    receipt = _as(app, app.ai_channel,
                  {"command": "run_method", "method": "logistic",
                   "dataset_id": dataset_id, "seed": 13,
                   "overrides": {"C": 1.0}})
    assert receipt["ok"] is True, receipt
    for key in RECEIPT_REQUIRED_KEYS:
        assert key in receipt, key
    assert receipt["run_ids"] == [receipt["run_id"]]
    assert receipt["primary_metric"] == "f1"
    assert 0.0 <= receipt["metrics"]["f1"] <= 1.0

    # 大对象只在 ML 侧：回执里没有 y_pred / artifacts / 完整配置
    blob = json.dumps(receipt, ensure_ascii=False, default=str)
    for banned in ("y_pred", "y_true", "artifacts", "npz"):
        assert banned not in blob
    assert len(blob) < 4000, f"回执过大（{len(blob)} 字符），大对象疑似内联"

    # 描述工具能按 run_id 回看（仍然只给类别名，不内联数组）
    described = app.tools.execute("describe_run", {"run_id": receipt["run_id"]})
    assert described["is_error"] is False
    assert "y_pred" in described["value"]["artifact_keys"]
    assert isinstance(described["value"]["metrics"], dict)

    # 批量 + 对比：仍是引用与标量
    batch = _as(app, app.ai_channel,
                {"command": "run_method_batch",
                 "methods": ["logistic", "svc"], "dataset_id": dataset_id})
    assert batch["ok"] is True
    assert len(batch["run_ids"]) == 2

    compare = _as(app, app.ai_channel,
                  {"command": "compare_methods", "methods": ["logistic", "svc"],
                   "dataset_id": dataset_id, "seed": 13})
    assert compare["ok"] is True
    assert compare["winner"]["method"] in ("logistic", "svc")
    assert set(compare["metrics"]) == {"logistic", "svc"}

    # 未知命令 fail loud（不猜测、不静默降级）
    unknown = _as(app, app.ai_channel, {"command": "tune_method", "method": "svc"})
    assert unknown["ok"] is False
    assert unknown["error_kind"] == "unknown_command"
    assert "prepare_dataset" in unknown["error_suggest"] or \
           "run_method" in unknown["error_suggest"]

    # estimate 走 mecha Surface 的 ops 管道（est_sec 必填）
    estimate = app.software.surface.estimate(
        {"command": "run_method", "method": "svc", "dataset_id": dataset_id})
    assert estimate["est_sec"] > 0
    assert estimate["command"] == "run_method"


# ---------------------------------------------------------------- 4. 资源守卫
def test_resource_guard_rejects_without_escaping(app):
    """资源守卫：预算超限是**结构化拒绝**，不抛穿、不拖垮会话。"""
    dataset_id = _prepared(app)
    receipt = _as(app, app.ai_channel,
                  {"command": "run_method", "method": "svc",
                   "dataset_id": dataset_id,
                   "resource_guard": {"max_kernel_mb": 0.000001}})
    assert receipt["ok"] is False
    for key in RECEIPT_REQUIRED_KEYS:
        assert key in receipt
    assert receipt["run_ids"], "被守卫拦下的运行仍要有 run_id 供对账"
    assert any("资源预算超限" in w for w in receipt["warnings"])

    # 会话没被打坏：随后的正常运行照跑
    ok = _as(app, app.ai_channel,
             {"command": "run_method", "method": "logistic",
              "dataset_id": dataset_id})
    assert ok["ok"] is True

    # 估算也吃守卫：预算太小时 estimate 报 violations（事前就看得见）
    estimate = app.engine.estimate(
        {"command": "run_method", "method": "svc", "dataset_id": dataset_id,
         "resource_guard": {"max_kernel_mb": 0.000001}})
    assert estimate["est_sec"] > 0
    assert estimate["violations"]

    # 非法守卫声明 fail loud（不静默取默认）
    bad = _as(app, app.ai_channel,
              {"command": "run_method", "method": "logistic",
               "dataset_id": dataset_id, "resource_guard": {"max_kernel_mb": -1}})
    assert bad["ok"] is False
    assert bad["error_kind"] == "bad_resource_guard"


def test_history_resource_guard_is_effective_value(app):
    """History 里的 resource_guard 必须是本次实际生效值，不是装配期默认值。"""
    dataset_id = _prepared(app)
    receipt = _as(app, app.ai_channel,
                  {"command": "run_method", "method": "logistic",
                   "dataset_id": dataset_id,
                   "resource_guard": {"max_kernel_mb": 123.0}})
    assert receipt["ok"] is True, receipt
    guard_events = [e for e in app.history.events()
                    if e.key == "current.resource_guard"]
    assert guard_events, "resource_guard 没进 History"
    assert guard_events[-1].value == {"max_kernel_mb": 123.0}, guard_events[-1].value


# ---------------------------------------------------------------- 5. History 归因
def test_history_attribution_actor_and_call_id(app):
    """History 归因：谁（actor）、哪次调用（call_id）、做了什么（命令键）。"""
    dataset_id = _prepared(app)

    CURRENT_CALL_ID.set("call-abc123")
    try:
        receipt = _as(app, app.ai_channel,
                      {"command": "run_method", "method": "logistic",
                       "dataset_id": dataset_id, "seed": 13})
    finally:
        CURRENT_CALL_ID.set("")
    assert receipt["ok"] is True

    events = app.history.events()
    # 命令审计由 mecha 核心写（第二宿主真实消费 Command Surface）
    audit_events = [e for e in events if e.key == "command.run_method"]
    assert audit_events, "核心命令审计没进 History"
    audit = audit_events[-1]
    assert audit.actor == "ml-ai"
    assert audit.call_id == "call-abc123"
    assert audit.value["result_ref"]["run_ids"] == [receipt["run_id"]]

    # 域运行摘要（ml.run）保留 run 引用与标量指标，供 Monitor 复述
    run_events = [e for e in events if e.key == "ml.run"]
    assert run_events, "域运行摘要没进 History"
    run_event = run_events[-1]
    assert run_event.actor == "ml-ai"
    assert run_event.call_id == "call-abc123"
    assert run_event.value["run_id"] == receipt["run_id"]

    state_events = [e for e in events if e.key.startswith("current.")]
    assert state_events, "状态键没进 History"
    for e in state_events:
        assert e.key in STATE_KEYS, e.key
        assert e.actor in ("ml-gui", "ml-ai")

    # 准备阶段是人类侧发起的 → 归因必须是人类标签
    prepare_events = [e for e in events if e.key == "current.pipeline_id"]
    assert prepare_events and prepare_events[0].actor == "ml-gui"

    # History 只存引用与标量：事件体序列化后不得出现大数组/大对象
    for e in events:
        payload = json.dumps(e.value, ensure_ascii=False, default=str)
        assert len(payload) < 2000, f"事件 {e.key} 体过大，疑似内联大对象"
        assert "y_pred" not in payload
        assert "artifacts.npz" not in payload

    # 基线：ML 侧）确实握着真数组（证明"引用"不是因为没有数据）
    record = app.engine.session.record(receipt["run_id"])
    assert record is not None
    assert isinstance(record.result.artifacts.get("y_pred"), np.ndarray)


# ---------------------------------------------------------------- 6. Monitor 对账
def test_monitor_reconciles_claims_against_raw_history(app):
    """概括层可证伪：健康读数不 disputed，且每条 Claim 都能对到原始史。"""
    dataset_id = _prepared(app)
    receipt = _as(app, app.ai_channel,
                  {"command": "run_method", "method": "logistic",
                   "dataset_id": dataset_id, "seed": 13})
    assert receipt["ok"] is True

    view = app.monitor.read()
    assert view.disputed is False, view.dispute_reason
    summary = view.summary
    assert summary["run_count"] == 1
    assert summary["current_method"] == Claim("logistic", target="current.method")
    assert summary["current_dataset"] == Claim(dataset_id, target="current.dataset_id")
    assert summary["current_seed"] == Claim(13, target="current.seed")

    # 每个 Claim 都能回原始史核（这是 Monitor 的做法，这里显式再做一遍）
    for key, claim in summary.items():
        if isinstance(claim, Claim):
            target = claim.target or key
            assert target in view.raw_snapshot
            assert claim.value == view.raw_snapshot[target]

    # 人话里的 run_id 能在原始史与 ML 侧同时查到（双向可核）
    assert receipt["run_id"] in summary["headline"]
    assert any(e.value.get("run_id") == receipt["run_id"]
               for e in app.history.events()
               if isinstance(e.value, dict))


def test_monitor_marks_deliberate_mismatch_as_disputed(app):
    """故意不一致的 Claim 必须 disputed，且原始层赢（概括不得掩盖原史）。"""
    dataset_id = _prepared(app)
    _as(app, app.ai_channel, {"command": "run_method", "method": "logistic",
                              "dataset_id": dataset_id, "seed": 13})
    raw = app.engine.summary()["records"][-1]
    assert raw["method"] == "logistic"

    honest = app.monitor.read()
    assert honest.disputed is False

    def lying_summarizer(events):
        summary = dict(ml_summarizer(events))
        summary["current_method"] = Claim("random_forest", target="current.method")
        return summary

    lying = Monitor(app.history.events, lying_summarizer).read()
    assert lying.disputed is True
    assert "current.method" in lying.dispute_reason
    # 原始赢：读数里的 raw_snapshot 仍是 logistic，概括的谎话不改变原史
    assert lying.raw_snapshot["current.method"] == "logistic"
    assert app.monitor.read().disputed is False

    # 谎报一个原始史里压根没有的键：同样 disputed（不许编值）
    def phantom_summarizer(events):
        summary = dict(ml_summarizer(events))
        summary["current_pipeline"] = Claim("nope", target="current.pipeline")
        return summary

    phantom = Monitor(app.history.events, phantom_summarizer).read()
    assert phantom.disputed is True
    assert "current.pipeline" in phantom.dispute_reason


# ---------------------------------------------------------------- 7. 静态断言
def _ml_mecha_sources() -> list[Path]:
    return sorted((REPO / "ml_mecha").glob("*.py"))


#: EL 的命令名（宪章 R1：ML 不许复制它们）。
EL_COMMAND_NAMES = ("set_config", "run_sim", "set_drive", "reset_config",
                    "read_waves", "run_actions")

#: EL 的状态键（宪章 R2：ML 不许复制它们）。
#: 只收**足够特异**的键——``power`` 这类普通英文词会让静态断言变成误报机器。
EL_STATE_KEYS = ("field.B_gauss", "drive.pulse_end_ns", "config_head",
                 "src.adapter")

#: EL 侧参考实现模块名（出现在 import 里即违规）。
EL_MODULE_HINTS = ("mecha_v2", "energy_level", "el_adapter")


def test_static_ml_mecha_does_not_import_el_modules():
    """静态：ml_mecha 不 import EL 模块，也不 import ML 的 GUI/优化层。"""
    banned = EL_MODULE_HINTS + ("ml_toolbox.opt", "ml_toolbox.ui")
    offenders: list[str] = []
    for path in _ml_mecha_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if any(name == b or name.startswith(b + ".") for b in banned):
                    offenders.append(f"{path.name}: import {name}")
    assert offenders == [], offenders


def test_static_ml_mecha_does_not_copy_el_command_or_state_names():
    """静态：ml_mecha 里不出现 EL 命令名 / EL 状态键。"""
    offenders: list[str] = []
    for path in _ml_mecha_sources():
        if path.name in ("tools.py", "validator.py", "commands.py"):
            continue    # 这三处在"讲解规矩"时会引用 mecha/EL 的反例名字
        text = path.read_text(encoding="utf-8")
        for token in EL_COMMAND_NAMES + EL_STATE_KEYS:
            if token in text:
                offenders.append(f"{path.name}: 出现 EL 词汇 {token!r}")
    assert offenders == [], offenders

    # 命令面声明里也不许有 EL 名（用 AST 取值，避免被注释/文档干扰）
    from ml_mecha.commands import command_names
    assert not set(command_names()) & set(EL_COMMAND_NAMES)
    # 状态键同理：ML 的键必须全是 ML 词汇
    assert not set(STATE_KEYS) & set(EL_STATE_KEYS)
    assert all(k.startswith("current.") for k in STATE_KEYS)


def test_static_ml_mecha_does_not_load_gui_or_opt_layers():
    """静态（运行时）：import ml_mecha 不把 ML 的 GUI / 优化层拖进来。"""
    import subprocess

    code = (
        "import sys;"
        "import ml_mecha.engine, ml_mecha.tools, ml_mecha.assembly, ml_mecha.summarizer;"
        "bad=[m for m in sys.modules if m.startswith('ml_toolbox.ui')"
        " or m.startswith('ml_toolbox.opt') or m.startswith('PyQt')];"
        "print(','.join(sorted(bad)))"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", f"ml_mecha 拖进了 {proc.stdout.strip()}"


def test_static_ml_domain_keys_absent_from_mecha_core():
    """静态：ML 领域键/方法名没有写进 mecha 核心（宪章 §5.2 硬约束）。

    命令名单独放宽一档：``mecha/commands.py`` 的**教学提示**里会把
    ``run_method`` / ``prepare_dataset`` 当"合法命名长什么样"的例子（框架教
    宿主怎么起名，不是框架拥有这两个命令）。因此命令名只在**代码字面量**里
    查，文档串里出现不算违规——这也让"真写进核心"仍会被抓住。
    """
    if not MECHA_ROOT.is_dir():
        pytest.skip(f"mecha 仓不在位：{MECHA_ROOT}")
    core = sorted((MECHA_ROOT / "mecha").rglob("*.py"))
    assert core, "mecha 核心源码为空？"

    # 领域状态键与领域方法名：源码里一个都不许有
    forbidden = set(STATE_KEYS) | {"resource_guard", "dataset_id", "pipeline_id",
                                   "random_forest", "logistic"}
    offenders: list[str] = []
    for path in core:
        text = path.read_text(encoding="utf-8")
        for token in sorted(forbidden):
            if re.search(rf"(?<![\w.]){re.escape(token)}(?![\w])", text):
                offenders.append(f"{path.relative_to(MECHA_ROOT)}: 含 {token!r}")
    assert offenders == [], offenders

    # 命令名：只查 AST 里的字符串字面量（排除 docstring），命中即"核心注册了 ML 命令"
    command_names = {"prepare_dataset", "run_method", "run_method_batch",
                     "compare_methods"}
    hits: list[str] = []
    for path in core:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docstrings.add(doc)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in docstrings:
                    continue
                if node.value in command_names:
                    hits.append(f"{path.relative_to(MECHA_ROOT)}: {node.value!r}")
    assert hits == [], hits


def test_static_command_surface_is_ml_owned():
    """静态：命令面是 ML 自己的词汇，且每条命令都声明副作用与风险。"""
    from ml_mecha.commands import COMMANDS, SIDE_EFFECT_NONE

    assert {c.name for c in COMMANDS} == {"prepare_dataset", "run_method",
                                          "run_method_batch", "compare_methods"}
    for command in COMMANDS:
        assert command.side_effect, command.name
        assert command.risk, command.name
        if command.side_effect == SIDE_EFFECT_NONE:
            continue        # 只读命令：允许什么都不写
        assert command.writes_state or command.extra_receipt_keys, command.name
    # schema 里的 run_output_required 是显式声明的（mecha 契约键）
    assert "run_output_required" not in RECEIPT_REQUIRED_KEYS
    assert {"ok", "command", "run_ids"} <= set(RECEIPT_REQUIRED_KEYS)


def test_prepare_dataset_csv_fails_closed_without_dataset_roots(app):
    """未声明 dataset_roots 时，csv 数据源必须 fail closed（AI 不得读任意路径）。"""
    csv_path = Path(app.layout.root) / "outside.csv"
    _frame().to_csv(csv_path, index=False)
    app.switch_human()
    receipt = _as(app, app.human_channel, {
        "command": "prepare_dataset",
        "source": {"kind": "csv", "path": str(csv_path), "target": "target"},
    })
    assert receipt["ok"] is False
    assert receipt["error_kind"] == "dataset_roots_required"
    assert app.engine.dataset_ids() == []       # 被拒请求不留登记


def test_prepare_dataset_csv_allowed_inside_dataset_roots():
    """显式 dataset_roots 内允许读取；根外路径仍拒绝。"""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ml-mecha-roots-") as td:
        root = Path(td)
        inside = root / "inside.csv"
        _frame().to_csv(inside, index=False)
        instance = assemble_ml_mecha(root=root, dataset_roots=[str(root)], seed=13)
        try:
            instance.switch_human()
            ok = _as(instance, instance.human_channel, {
                "command": "prepare_dataset",
                "source": {"kind": "csv", "path": str(inside), "target": "target"},
            })
            assert ok["ok"] is True, ok
            denied = _as(instance, instance.human_channel, {
                "command": "prepare_dataset",
                "source": {"kind": "csv", "path": str(root.parent / "nope.csv"),
                           "target": "target"},
            })
            assert denied["ok"] is False
            assert denied["error_kind"] == "dataset_path_denied"
        finally:
            instance.close()


def main() -> int:
    """不装 pytest 时的直跑入口（与仓内其他 test_*.py 同形）。"""
    import tempfile

    tests = [test_tool_projection_is_read_only_capable,
             test_tool_errors_normalized_to_tool_failure,
             test_authority_gate_locked_denies_writes,
             test_authority_side_switch_cannot_be_forged,
             test_prepare_dataset_csv_fails_closed_without_dataset_roots,
             test_prepare_dataset_csv_allowed_inside_dataset_roots,
             test_read_queries_are_registered_in_surface,
             test_commands_registered_in_core_registry,
             test_write_tool_failures_are_tool_failure,
             test_run_method_end_to_end_and_receipt_contract,
             test_resource_guard_rejects_without_escaping,
             test_history_resource_guard_is_effective_value,
             test_history_attribution_actor_and_call_id,
             test_monitor_reconciles_claims_against_raw_history,
             test_monitor_marks_deliberate_mismatch_as_disputed,
             test_static_ml_mecha_does_not_import_el_modules,
             test_static_ml_mecha_does_not_copy_el_command_or_state_names,
             test_static_ml_mecha_does_not_load_gui_or_opt_layers,
             test_static_ml_domain_keys_absent_from_mecha_core,
             test_static_command_surface_is_ml_owned]
    failures = []
    for fn in tests:
        if "app" in fn.__code__.co_varnames:
            with tempfile.TemporaryDirectory(prefix="ml-mecha-test-") as td:
                instance = assemble_ml_mecha(root=td, seed=13)
                try:
                    fn(instance)
                except Exception as exc:  # noqa: BLE001
                    failures.append((fn.__name__, exc))
                finally:
                    instance.close()
        else:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                failures.append((fn.__name__, exc))
        print(("  ✓ " if not failures or failures[-1][0] != fn.__name__ else "  ✗ ")
              + fn.__name__)
    for name, exc in failures:
        print(f"  FAILED {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
