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
from mecha.errors import GateDenied, MechaError     # noqa: E402
from mecha.gate import Channel                      # noqa: E402
from mecha.monitor import Claim, Monitor            # noqa: E402
from mecha.surface import ExecutionContext          # noqa: E402
from mecha.tools import CURRENT_CALL_ID, check_tool_name  # noqa: E402

from ml_mecha.assembly import READ_QUERIES, assemble_ml_mecha  # noqa: E402
from ml_mecha.engine import (RECEIPT_REQUIRED_KEYS, clear_channel,  # noqa: E402
                             use_channel)
from ml_mecha.summarizer import ml_summarizer       # noqa: E402
from ml_mecha.validator import STATE_KEYS           # noqa: E402

TOOL_NAMES = ("cancel_run", "compare_methods", "describe_dataset",
              "describe_method", "describe_methods", "describe_run",
              "prepare_dataset", "read_job", "run_method", "run_method_batch",
              "submit_run")


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


def test_scope_policy_denies_unlisted_command_scope(app):
    """AI 未被授予的 scope 在副作用前 fail closed；显式授权后放行。"""
    dataset_id = _prepared(app)
    app.scopes.revoke("ai", "run_method")
    denied = _as(app, app.ai_channel,
                 {"command": "run_method", "method": "logistic",
                  "dataset_id": dataset_id})
    assert denied["ok"] is False
    assert denied["error_kind"] == "scope_denied"
    assert app.engine.session.export_state()["records"] == []

    app.scopes.grant("ai", "run_method")
    allowed = _as(app, app.ai_channel,
                  {"command": "run_method", "method": "logistic",
                   "dataset_id": dataset_id})
    assert allowed["ok"] is True


def test_job_contract_runs_command_async(app):
    """ML 长命令经 mecha Job Contract 跑：状态、result_ref、审计归因都可查。"""
    dataset_id = _prepared(app)
    job = app.submit({"command": "run_method", "method": "logistic",
                      "dataset_id": dataset_id})
    status = app.wait_job(job, 20)
    assert status["state"] == "done", status
    assert status["command"] == "run_method"
    assert status["result_ref"]["run_ids"] == [job.result["run_id"]]
    assert job.result["ok"] is True
    audit = [e for e in app.history.events() if e.key == "command.run_method"]
    assert audit and audit[-1].actor == "ml-ai"
    assert any(j["id"] == job.id for j in app.list_jobs())


def test_job_contract_failure_has_no_fake_ref(app):
    """失败命令不伪造 result_ref；job 状态仍完成（回执 ok=False）。"""
    app.switch_ai()
    job = app.submit({"command": "run_method", "method": "logistic",
                      "dataset_id": "no-such-dataset"})
    status = app.wait_job(job, 20)
    assert status["state"] == "done", status
    assert job.result["ok"] is False
    assert status.get("result_ref") is None


def test_cancelled_command_entry_is_a_noop_failure(app):
    """取消语义（第三轮 P1-3 残余）：入口已取消 ⇒ 结构化失败，零副作用零审计。

    这是"取消真的省了工作"的可执行判据：取消**不是**把成功回执改个标签，
    而是命令根本没开始——不产 run、不写 ``ml.run``、不写 ``command.<name>``
    审计、History 零新增。最后用同一条调用（不取消）做对照，防止上面的
    "什么都没发生"断言因为别的原因（比如命令本来就跑不动）而空转。
    """
    dataset_id = _prepared(app)

    before = len(app.history.events())
    records_before = len(app.engine.session.export_state()["records"])

    ctx = ExecutionContext()
    ctx.request_cancel()
    receipt = _as(app, app.ai_channel,
                  {"command": "run_method", "method": "logistic",
                   "dataset_id": dataset_id}, context=ctx)

    # 1) 回执形状：必填键齐 + 机读 kind + 可读说明
    assert receipt["ok"] is False
    assert receipt["error_kind"] == "cancelled"
    for key in RECEIPT_REQUIRED_KEYS:
        assert key in receipt, key
    assert receipt["run_ids"] == []
    assert any("取消" in str(w) for w in receipt["warnings"]), receipt["warnings"]

    # 2) 零副作用：不产 run、不写 ml.run、不写 command.<name>、History 零新增
    assert len(app.engine.session.export_state()["records"]) == records_before
    events = app.history.events()
    assert len(events) == before, [e.key for e in events[before:]]
    assert [e for e in events if e.key == "ml.run"] == []
    assert [e for e in events if e.key == "command.run_method"] == []

    # 3) 对照片：同一调用不取消 ⇒ 成功且真的产 run / 写史（防断言空转）
    ok = _as(app, app.ai_channel,
             {"command": "run_method", "method": "logistic",
              "dataset_id": dataset_id}, context=ExecutionContext())
    assert ok["ok"] is True, ok
    assert ok["run_ids"], ok
    assert len(app.engine.session.export_state()["records"]) == records_before + 1
    after = app.history.events()
    assert [e for e in after if e.key == "ml.run"]
    assert [e for e in after if e.key == "command.run_method"]


def test_job_path_carries_call_id_into_audit(app):
    """P2-5 回归：job 路径的命令审计必须保住调用方的 ``call_id``。

    contextvars 不跨线程，所以 ``submit`` 必须在提交线程里**捕获**再在 worker
    内 ``set``。先跑一条不设 call_id 的对照（审计必须为空串），证明下面读到的
    ``job-call-1`` 不是某处写死的常量。
    """
    dataset_id = _prepared(app)

    plain = app.submit({"command": "run_method", "method": "logistic",
                        "dataset_id": dataset_id})
    assert app.wait_job(plain, 20)["state"] == "done"
    plain_audit = [e for e in app.history.events()
                   if e.key == "command.run_method"][-1]
    assert plain_audit.call_id == "", plain_audit.call_id

    CURRENT_CALL_ID.set("job-call-1")
    try:
        job = app.submit({"command": "run_method", "method": "logistic",
                          "dataset_id": dataset_id})
    finally:
        CURRENT_CALL_ID.set("")

    status = app.wait_job(job, 20)
    assert status["state"] == "done", status
    assert job.result["ok"] is True, job.result

    audit = [e for e in app.history.events() if e.key == "command.run_method"][-1]
    assert audit.actor == "ml-ai"
    assert audit.call_id == "job-call-1", audit.call_id
    # 域运行摘要与命令审计互引同一个 call_id（两处同源）
    run_event = [e for e in app.history.events() if e.key == "ml.run"][-1]
    assert run_event.call_id == "job-call-1", run_event.call_id


def test_uncancellable_command_cancel_job_is_rejected(app):
    """第三轮审查：``cancel_supported`` 从死声明变成真契约（不可取消侧）。

    ``prepare_dataset`` 的核心声明是 ``cancel_supported=False``（见
    ``ml_mecha/core_commands.py``）：装配点必须把这条声明带进 job，``cancel_job``
    必须被结构化拒（``job_not_cancellable``），且**不静默取消**——被拒之后命令
    照常跑完、数据集真的建出来。

    拒收与 job 状态无关（只读声明字段），所以这里连"完成后再取消"也断言被拒：
    判据不靠赌时序，不会有 flaky。最后用同一条命令**不取消**做对照，证明上面的
    "建成功"不是因为取消路径整体坏了。
    """
    frame_source = {"kind": "frame", "frame": _frame(), "target": "target"}
    app.switch_human()                  # prepare_dataset 走人类侧写权
    job = app.submit({"command": "prepare_dataset", "name": "nc",
                      "source": dict(frame_source)}, side="human")

    # 声明被带进 job（轮询面直接可读，不用猜默认值）
    assert job.cancel_supported is False
    assert job.to_dict()["cancel_supported"] is False

    with pytest.raises(MechaError) as exc:
        app.cancel_job(job)
    assert exc.value.kind == "job_not_cancellable", exc.value
    assert exc.value.hint.strip(), exc.value

    # 被拒 ≠ 被打断：job 照常完成，数据集真的建出来了
    status = app.wait_job(job, 30)
    assert status["state"] == "done", status
    assert job.result["ok"] is True, job.result
    assert job.result["dataset_id"] in app.engine.dataset_ids()

    # 终态之后照样拒（拒的是"这个命令不可取消"，不是某个瞬间的竞态）
    with pytest.raises(MechaError) as exc2:
        app.cancel_job(job)
    assert exc2.value.kind == "job_not_cancellable", exc2.value

    # 对照片：同一条命令不取消 ⇒ 成功且建出**另一个**数据集（防断言空转）
    ok_job = app.submit({"command": "prepare_dataset", "name": "nc-control",
                         "source": dict(frame_source)}, side="human")
    ok_status = app.wait_job(ok_job, 30)
    assert ok_status["state"] == "done", ok_status
    assert ok_job.result["ok"] is True, ok_job.result
    assert ok_job.result["dataset_id"] != job.result["dataset_id"]


def test_cancellable_command_cancel_job_is_not_rejected(app):
    """对偶：``cancel_supported=True`` 的命令，``cancel_job`` 不得报不可取消。

    取消是协作式的（检查点只在命令入口与方法边界），所以这里**不断言**终态一定
    是 ``cancelled``——那条竞态由命令体判据负责。本判据只钉"闸门没有被错误关上"：
    提交后取消不抛 ``job_not_cancellable``，且 job 最终落到一个合法终态。
    """
    dataset_id = _prepared(app)
    job = app.submit({"command": "run_method", "method": "logistic",
                      "dataset_id": dataset_id})
    assert job.cancel_supported is True
    assert job.to_dict()["cancel_supported"] is True

    try:
        app.cancel_job(job)
    except MechaError as e:                      # noqa: PERF203 - 就是要看 kind
        raise AssertionError(f"可取消的命令取消被拒：{e.kind}: {e}") from e

    status = app.wait_job(job, 30)
    assert status["state"] in ("done", "cancelled"), status
    if status["state"] == "done":
        # 取消来晚了是合法结局（请求先于/晚于终点都行）；此时回执必须完整
        assert job.result["ok"] is True, job.result


def test_submit_unknown_command_keeps_existing_failure_path(app):
    """``submit`` 不为未知命令发明新拒绝：既有失败路径原样保留。

    核心命令面查不到该命令时（``commands.spec(...) is None``），装配点**不**
    在这里抛错——既有路径是 worker 内的分派产 ``ok=False`` 的教学回执
    （``error_kind="unknown_command"``），job 照常到终态；此时没有声明可读，
    可取消性沿用 Job 默认。
    """
    app.switch_ai()
    job = app.submit({"command": "no_such_command_xyz"})
    status = app.wait_job(job, 20)
    assert status["state"] == "done", status
    assert job.result["ok"] is False, job.result
    assert job.result["error_kind"] == "unknown_command", job.result
    assert job.cancel_supported is True, "取不到声明时应沿用 Job 默认"


def test_submit_side_is_validated_and_does_not_widen_authority(app):
    """P2-2 回归：``side`` 只选**既有**通道，不扩权；非法值当场拒。
    Authority 以 ``authority_mode_mismatch`` 拒——证明 submit 只是通道选择器，
    没有第二套授权逻辑（选 human 不会换来人类写权）。
    """
    with pytest.raises(MechaError) as exc:
        app.submit({"command": "run_method", "method": "logistic",
                    "dataset_id": "whatever"}, side="root")
    assert exc.value.kind == "bad_side"
    assert app.list_jobs() == []        # 非法 side 不留半拉子 job

    dataset_id = _prepared(app)
    assert app.mode is Mode.AI

    good = app.submit({"command": "run_method", "method": "logistic",
                       "dataset_id": dataset_id}, side="ai")
    status = app.wait_job(good, 20)
    assert status["state"] == "done", status
    assert good.result["ok"] is True, good.result

    # 不扩权：mode=AI 下人类通道照样被拒（不是"选了 human 就能写"）
    human_side = app.submit({"command": "run_method", "method": "logistic",
                             "dataset_id": dataset_id}, side="human")
    human_status = app.wait_job(human_side, 20)
    assert human_status["state"] == "done", human_status
    assert human_side.result["ok"] is False, human_side.result
    assert human_side.result["error_kind"] == "authority_mode_mismatch"


def test_writes_state_declaration_matches_actual_history(app):
    """P2-7 防复发：声明的 ``writes_state`` 必须**逐键等于**实际写入的键集合。

    做法是机械对账而不是读代码：四条命令各跑一次，收集它在 History 里
    实际写下的 ``current.*`` 键（actor=ml-ai 的那些），与该命令声明的集合
    比双向相等——多声明（谎报会写）少声明（偷偷写）都判红。
    """
    from ml_mecha.commands import command_map

    app.switch_ai()                     # AI 侧持有全部 4 条 scope，四条都真跑

    def _actual(spec):
        mark = len(app.history.events())
        receipt = _as(app, app.ai_channel, spec)
        assert receipt["ok"] is True, receipt
        keys = {e.key for e in app.history.events()[mark:]
                if e.actor == "ml-ai" and e.key.startswith("current.")}
        return receipt, keys

    prep, prep_keys = _actual({
        "command": "prepare_dataset", "name": "declared",
        "source": {"kind": "frame", "frame": _frame(), "target": "target"}})
    dataset_id = prep["dataset_id"]

    observed = {"prepare_dataset": prep_keys}
    for spec in ({"command": "run_method", "method": "logistic",
                  "dataset_id": dataset_id},
                 {"command": "run_method_batch", "methods": ["logistic", "svc"],
                  "dataset_id": dataset_id},
                 {"command": "compare_methods", "methods": ["logistic", "svc"],
                  "dataset_id": dataset_id}):
        _receipt, keys = _actual(spec)
        observed[spec["command"]] = keys

    declared = {name: set(spec.writes_state)
                for name, spec in command_map().items()}
    assert set(observed) == set(declared)
    assert observed == declared, {
        name: {"declared_only": sorted(declared[name] - observed[name]),
               "written_only": sorted(observed[name] - declared[name])}
        for name in declared
        if declared[name] != observed[name]}


def test_toolhost_is_the_tool_transport_seam(app):
    """ToolHost 契约在 ML 宿主上有真实消费者：定义面同一张表、调用经宿主 seam。"""
    assert app.toolhost.registry is app.tools          # 同一对象，非第二真值
    assert app.toolhost.schemas() == app.tools.schemas()
    result = app.call_tool("describe_methods", {"family": "linear"})
    assert result["is_error"] is False
    assert result["value"]["method_count"] > 0
    bad = app.call_tool("no_such_tool", {})
    assert bad["is_error"] is True


def test_run_event_carries_artifact_locator(app):
    """Artifact 契约在 ML 宿主上有真实消费者：ml.run 事件只带不透明 locator。"""
    dataset_id = _prepared(app)
    receipt = _as(app, app.ai_channel,
                  {"command": "run_method", "method": "logistic",
                   "dataset_id": dataset_id})
    assert receipt["ok"] is True
    events = [e for e in app.history.events() if e.key == "ml.run"]
    assert events, "没有 ml.run 事件"
    locator = events[-1].value.get("artifact_locator")
    assert locator, f"事件缺 artifact_locator：{events[-1].value}"
    assert "artifact-" in locator and "/" not in locator and "\\" not in locator
    assert app.artifacts.exists(locator)
    meta = app.artifacts.describe(locator)
    assert meta["kind"] == "ml_run_summary"
    blob = json.loads(app.artifacts.retrieve(locator).decode("utf-8"))
    assert blob["run_id"] == receipt["run_id"]


def test_write_tool_failures_are_tool_failure(app):
    """写工具失败必须是 ``is_error=True`` + ToolFailure kind，不是假成功。

    ⚠ **行为变更（框架 ADR「命令面治理闸前置」）**：工具面注册的是 **ai 侧**
    （``assembly.py`` 的 ``register_ml_tools(..., side="ai")``）⇒ 在 ``human`` 模式下调它，
    现在会被框架的**前置** authority 检查当场拒（``authority_mode_mismatch``），
    **不再跑到 handler**。旧行为是"handler 先跑、报它自己的错"（本用例原先把
    ``dataset_roots_required`` 当成这条路径的期望——那是在"检查滞后于副作用"的旧次序下
    才成立）。本判据因此**两段都钉**：① 新次序（前置拒，fail closed）；
    ② 开闸后 handler 级失败仍被归一化成 ToolFailure（原意保留）。
    """
    csv_path = Path(app.layout.root) / "outside.csv"
    _frame().to_csv(csv_path, index=False)
    bad = {"source": {"kind": "csv", "path": str(csv_path), "target": "target"}}

    # ① 新次序：human 模式下调 **ai 侧**工具 ⇒ 前置闸当场拒（不是 handler 的错）
    app.switch_human()
    early = app.tools.execute("prepare_dataset", bad)
    assert early["is_error"] is True, early
    assert early["error"]["info"]["kind"] == "authority_mode_mismatch", early

    # 人类侧准备一份数据（**必须在 human 模式**：它是 human 侧通道的活）
    dataset_id = _prepare(app, app.human_channel)["dataset_id"]

    # ② 开闸后走到 handler：csv 未授权 ⇒ 归一化成 ToolFailure（原意）
    app.switch_ai()
    result = app.tools.execute("prepare_dataset", bad)
    assert result["is_error"] is True
    assert result["error"]["info"]["kind"] == "dataset_roots_required"

    # 未知方法同样归一化（不是 value.ok=False 的假成功）
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

    命令名扫的是"非 docstring 的字符串字面量**整句**"：第三轮审查 P2-1 抓到
    ``mecha/commands.py`` / ``mecha/tools.py`` 的**用户可见 hint** 里塞了宿主
    命令名当例子（旧断言只判整串相等 ⇒ 整句 hint 漏抓 = 假绿）。核心可以写
    中性示例（``do_thing`` / ``compute_summary``），但不许出现真宿主命令名——
    所以这里改成**子串**扫描：核心若真把命令名写进可见文本，判据必红。
    """
    assert MECHA_ROOT.is_dir(), (
        f"mecha 仓不在位：{MECHA_ROOT}（缺真依赖必须红，不 skip——"
        "最终验收审查 M1：skip 会让核心领域词守卫静默失效）")
    core = sorted((MECHA_ROOT / "mecha").rglob("*.py"))
    assert core, "mecha 核心源码为空？"

    # 领域状态键与领域方法名：源码里一个都不许有（含 docstring/注释）
    forbidden = set(STATE_KEYS) | {"resource_guard", "dataset_id", "pipeline_id",
                                   "random_forest", "logistic"}
    offenders: list[str] = []
    for path in core:
        text = path.read_text(encoding="utf-8")
        for token in sorted(forbidden):
            if re.search(rf"(?<![\w.]){re.escape(token)}(?![\w])", text):
                offenders.append(f"{path.relative_to(MECHA_ROOT)}: 含 {token!r}")
    assert offenders == [], offenders

    # 命令名：AST 字符串字面量（排除 docstring）按**子串**查——整句 hint、
    # 报错文本、suggest 都是用户可见面，塞进宿主命令名一律算违规。
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
                for name in sorted(command_names):
                    if name in node.value:
                        hits.append(f"{path.relative_to(MECHA_ROOT)}: "
                                    f"{node.value[:70]!r} 含 {name!r}")
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


# --------------------------- 框架 ADR「域快照纯净与失败说真话」（2026-09-26）的落地钉
def test_audit_keys_are_rejected_as_domain_state(app):
    """⭐ `command.*` **审计键**不许当域状态写（响亮拒绝 + 可教学），域键照常放行。

    依据：框架 ADR「域快照纯净与失败说真话」。审计从前**借 `gate.set` 落史** ⇒ 审计键会
    走到宿主 validator 上，本仓那时是"**只校验形状就放行**"。框架改后审计走 `Gate.record`、
    **不再经过宿主 validator**（迁移须知 2）⇒ 那个放行分支成了**写时死码**，而"审计键不是
    域状态"这件事**没人守**。⇒ 改成**显式拒绝**，本判据钉住它（**行为变更，已在 commit 申报**）。

    对偶（防把闸关死）：真的域键 `current.dataset_id` 必须照常通过。
    """
    from mecha.commands import AUDIT_KEY_PREFIX

    # 用**装配好的** validator（`assembly` 把它接到 Gate 上，见 `validate=`）——
    # 不是自己造一个，判据才落在真链路上。
    v = app.engine.validator
    # 域键：放行（对偶——否则"全拒"也能让下面那些过）。⚠ 值也要合法：
    # ML 的 validator 会**对着事实**校验（未知 dataset_id 会被拒），所以先真建一个。
    dataset_id = _prepared(app)
    v.validate_state("current.dataset_id", dataset_id)
    v.validate_state("current.pipeline_id", None)

    # 审计键：拒绝，且文案可教学（说清它是什么 + 你大概想写什么）
    with pytest.raises(GateDenied) as exc:
        v.validate_state(AUDIT_KEY_PREFIX + "prepare_dataset", {"actor": "ml-ai"})
    assert exc.value.kind == "audit_key_not_state", exc.value
    assert "审计" in str(exc.value) and "域状态" in str(exc.value), exc.value
    assert exc.value.hint.strip(), exc.value
    # 形状合法也不放行（旧行为正是"形状合法即接受"）
    with pytest.raises(GateDenied):
        v.validate_state(AUDIT_KEY_PREFIX + "run_method", {})
    # 而且**键名里的命令是否真存在**与判定无关（这是"类别"问题，不是"这条命令"问题）
    with pytest.raises(GateDenied):
        v.validate_state(AUDIT_KEY_PREFIX + "totally_made_up", {"x": 1})


def test_cancel_job_returns_core_truth_instead_of_dropping_it(app):
    """⭐ 人类侧的 ``cancel_job`` 必须**透出**核心说的真话（不再返回 `None` 把它扔掉）。

    对偶两向：① 运行中的可取消 job ⇒ `cancel_requested=True` 且 `terminal=False`；
    ② **已经跑完**的 job ⇒ `terminal=True` 且 `cancel_requested=False`
    （后者是 ADR 的要害：对终态 job 说"已请求取消"就是谎）。
    """
    dataset_id = _prepared(app)
    job = app.submit({"command": "run_method", "method": "logistic",
                      "dataset_id": dataset_id})
    got = app.cancel_job(job)
    assert isinstance(got, dict), got
    assert got.get("job_id") == job.id, got
    assert got.get("terminal") is False, got
    assert got.get("cancel_requested") is True, got
    assert got.get("state_before"), got
    app.wait_job(job, 120)                       # 等它落终态（取消是协作式的）

    late = app.cancel_job(job)
    assert late.get("terminal") is True, late
    assert late.get("cancel_requested") is False, (
        f"对已终态的 job 仍说「请求了取消」⇒ 人类侧会以为它停了：{late}")


# ------------------------------------- 命令面：required 语义翻转 + 审批闸接线
# 依据：mecha ADR `docs/v2/notes/implemented/2026-09-26-命令面治理闸前置与声明式opt-in.md`
# ① `parameters.required` **缺失或空 = 无必填**（对齐 JSON Schema）——老写法从"全必填"
#    翻转成"都不必填" ⇒ 可能**静默接受**原本会被拒的调用。② `approval_required` 真的拦；
#    审批通道由**调用方注入**（框架刻意不自动接线）。本组判据把这两条钉在**本适配层**。

def test_core_command_specs_declare_explicit_required(app):
    """⭐ 四条命令的**核心投影**都必须带**显式非空** `required`（否则语义翻转会开口子）。

    这是 ADR「迁移须知 1」对 ML 的直接检查项：ML 的命令声明走
    ``commands.py → core_commands.core_command_specs()`` 机械投影，投影**无条件**写出
    ``"required": [...]``。本判据逐条核对"核心 spec 里的 required == 宿主声明里的必填"，
    并**显式要求非空**——万一哪天有人把投影改成"空就不写"，这里立刻红。
    """
    from ml_mecha.commands import COMMANDS
    commands = app.software.commands
    checked = []
    for host in COMMANDS:
        spec = commands.spec(host.name)
        assert spec is not None, f"核心注册表里没有 {host.name}"
        params = dict(spec.parameters)
        assert "required" in params, (
            f"{host.name} 的核心参数 schema **没有写 required** ⇒ 按新语义它变成"
            "「都不必填」，会静默接受原本该被拒的调用（ADR 迁移须知 1）")
        core_required = list(params["required"])
        host_required = list(host.required_params)
        assert core_required == host_required, (host.name, core_required, host_required)
        assert core_required, (
            f"{host.name} 的 required 是空的 ⇒ 它现在「什么都不必填」；"
            "若这是有意的，请删掉这条断言并说明；否则补齐声明")
        # 必填项必须都在 properties 里（否则模型永远填不出来）
        assert set(core_required) <= set(dict(params["properties"])), host.name
        checked.append(host.name)
    # R8 非退化：真的逐条比过 4 条（不是空转）
    assert sorted(checked) == ["compare_methods", "prepare_dataset", "run_method",
                               "run_method_batch"], checked


def test_missing_required_param_is_rejected_not_silently_accepted(app):
    """⭐ 缺必填参数必须被**拒**（不是静默接受）——语义翻转的杀伤面就在这里。

    ``run_method`` 的必填是 ``method``：不给它，核心命令面必须在**干活之前**拒
    （``missing_arguments`` 一类结构化失败），而不是拿 ``None`` 一路跑下去。
    """
    app.switch_human()
    receipt = _as(app, app.human_channel, {"command": "run_method"})
    assert receipt["ok"] is False, receipt
    assert receipt.get("error_kind"), receipt          # 机读分类必须在
    # 只读命令（describe_*）不受影响：它们没有命令声明，走 Surface
    assert app.engine.query("method_catalog", family="", dataset_id="")["methods"]


def test_approval_channel_is_injected_into_invoke(app):
    """⭐ 审批通道**真的接到了 invoke 上**（R17：不是"看起来能注入"）。

    做法：包一层 ``software.commands.invoke`` 记录实参，跑一条真命令，断言
    ``approval is app.software.approval`` —— 即装配点把 ``Software.approval`` 一路传到了
    命令面（ADR §Consequences：框架**刻意不自动接线**，所以这一层必须由本仓显式做）。
    """
    seen: dict = {}
    real_invoke = app.software.commands.invoke

    def spy(name, args, **kwargs):
        seen["name"] = name
        seen["approval"] = kwargs.get("approval")
        seen["channel"] = kwargs.get("channel")
        seen["gate"] = kwargs.get("gate")
        return real_invoke(name, args, **kwargs)

    app.software.commands.invoke = spy
    try:
        app.switch_human()
        receipt = _prepare(app, app.human_channel)
    finally:
        app.software.commands.invoke = real_invoke
    assert receipt["ok"], receipt
    assert seen.get("name") == "prepare_dataset", seen
    assert seen.get("approval") is app.software.approval, (
        "审批通道没接到 invoke ⇒ 将来声明 approval_required=True 的命令会因"
        "「没接通道」被 fail closed 拒（那是设计，但会让声明者以为闸坏了）")
    assert seen.get("gate") is app.software.gate, seen


def test_no_ml_command_declares_approval_required(app):
    """今天的**事实**：四条命令都没有声明审批（这道闸不挡任何东西）。

    钉住它是为了让"加声明"成为**有意的**动作：一旦有人声明 ``approval_required=True``，
    本条会红——那时请确认审批人已接线（上一条判据），并更新
    ``assembly.py`` 的 Known Limitations（那里写着今天没有命令用到它）。
    """
    from ml_mecha.commands import COMMANDS
    declared = [c.name for c in COMMANDS]
    assert declared, "命令声明为空？"
    approvals = {name: bool(app.software.commands.spec(name).approval_required)
                 for name in declared}
    assert not any(approvals.values()), (
        f"有命令声明了 approval_required=True：{approvals} ⇒ 请确认审批通道已接线"
        "（bind_command_registry 的 approval=）并同步更新 assembly.py 的说明；"
        "否则这条闸会把该命令**全部拒绝**（fail closed）")


def main() -> int:
    """不装 pytest 时的直跑入口（与仓内其他 test_*.py 同形）。"""
    # 默认控制台可能是 GBK（本机 cp936）：✓/✗ 会让直跑入口在**第一个测试
    # 之前**就崩掉（exit 1），真实失败被编码崩溃掩盖。与 tests/run_all.py
    # 同款加固（第三轮审查 P2-8）。pytest 路径本来就走 UTF-8，不受影响。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import tempfile

    tests = [test_tool_projection_is_read_only_capable,
             test_tool_errors_normalized_to_tool_failure,
             test_authority_gate_locked_denies_writes,
             test_authority_side_switch_cannot_be_forged,
             test_prepare_dataset_csv_fails_closed_without_dataset_roots,
             test_prepare_dataset_csv_allowed_inside_dataset_roots,
             test_read_queries_are_registered_in_surface,
             test_commands_registered_in_core_registry,
             test_scope_policy_denies_unlisted_command_scope,
             test_job_contract_runs_command_async,
             test_job_contract_failure_has_no_fake_ref,
             test_cancelled_command_entry_is_a_noop_failure,
             test_job_path_carries_call_id_into_audit,
             test_submit_side_is_validated_and_does_not_widen_authority,
             test_uncancellable_command_cancel_job_is_rejected,
             test_cancellable_command_cancel_job_is_not_rejected,
             test_submit_unknown_command_keeps_existing_failure_path,
             test_writes_state_declaration_matches_actual_history,
             test_toolhost_is_the_tool_transport_seam,
             test_run_event_carries_artifact_locator,
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
             test_static_command_surface_is_ml_owned,
             test_core_command_specs_declare_explicit_required,
             test_missing_required_param_is_rejected_not_silently_accepted,
             test_approval_channel_is_injected_into_invoke,
             test_no_ml_command_declares_approval_required,
             test_audit_keys_are_rejected_as_domain_state,
             test_cancel_job_returns_core_truth_instead_of_dropping_it]
    failures = []
    for fn in tests:
        if "app" in fn.__code__.co_varnames:
            with tempfile.TemporaryDirectory(prefix="ml-mecha-test-") as td:
                instance = assemble_ml_mecha(root=td, seed=13)
                try:
                    fn(instance)
                except (KeyboardInterrupt, SystemExit):
                    raise
                # ⚠ 必须捕 **BaseException**，不能只捕 Exception：`pytest.raises` 失败时抛的
                # `Failed` **继承 BaseException** ⇒ 只捕 Exception 会让**整个直跑入口被掀翻**
                # （实测：`DID NOT RAISE` 直接把 main 打成 traceback，**没有 `✗` 行、没有汇总**）。
                # 后果不是"假绿"（exit code 仍是 1 ⇒ 门禁照样红），而是**红看不清**
                # ——"哪条判据红"这件事被 traceback 吃掉。
                except BaseException as exc:  # noqa: BLE001
                    failures.append((fn.__name__, exc))
                finally:
                    instance.close()
        else:
            try:
                fn()
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:  # noqa: BLE001
                failures.append((fn.__name__, exc))
        print(("  ✓ " if not failures or failures[-1][0] != fn.__name__ else "  ✗ ")
              + fn.__name__)
    for name, exc in failures:
        print(f"  FAILED {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
