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
框架仓（与本仓同级）。**不 vendor mecha**（两个仓各自独立演进）。
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
MECHA_ROOT = Path(os.environ.get("MECHA_ROOT") or os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "mecha")))
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

TOOL_NAMES = ("cancel_run", "checkpoint_detail", "clear_current_state",
              "compare_methods", "describe_dataset",
              "describe_method", "describe_methods", "describe_network",
              "describe_networks", "describe_run", "describe_training",
              "eval_custom", "prepare_dataset", "read_job", "run_custom",
              "run_method", "run_method_batch", "run_queries", "submit_run")


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
    """ML 6 条受控命令注册进核心 CommandRegistry（双宿主消费核心机制）。"""
    names = set(app.software.commands.names())
    assert names == {"prepare_dataset", "run_method", "run_method_batch",
                     "compare_methods", "run_custom", "clear_current_state",
                     "eval_custom"}
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
    audit = [e for e in app.history.events() if e.op == "command.run_method"]
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
    assert len(events) == before, [(e.op, e.target) for e in events[before:]]
    assert [e for e in events if e.op == "ml.run"] == []
    assert [e for e in events if e.op == "command.run_method"] == []

    # 3) 对照片：同一调用不取消 ⇒ 成功且真的产 run / 写史（防断言空转）
    ok = _as(app, app.ai_channel,
             {"command": "run_method", "method": "logistic",
              "dataset_id": dataset_id}, context=ExecutionContext())
    assert ok["ok"] is True, ok
    assert ok["run_ids"], ok
    assert len(app.engine.session.export_state()["records"]) == records_before + 1
    after = app.history.events()
    assert [e for e in after if e.op == "ml.run"]
    assert [e for e in after if e.op == "command.run_method"]


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
                   if e.op == "command.run_method"][-1]
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

    audit = [e for e in app.history.events() if e.op == "command.run_method"][-1]
    assert audit.actor == "ml-ai"
    assert audit.call_id == "job-call-1", audit.call_id
    # 域运行摘要与命令审计互引同一个 call_id（两处同源）
    run_event = [e for e in app.history.events() if e.op == "ml.run"][-1]
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
    # ⚠ **行为变更（2026-10-03，依据 mecha 指导意见 §3-2）**：此前是
    #   "提交成功 → 轮询 → 发现失败"；现在在 submit **之前**用
    #   `authority.gate(side)` 预检 ⇒ **当场**抛 GateDenied、**不建 job**
    #   （"同步拒绝 > 提交后轮询失败"）。AI 因此少白等一轮，
    #   也不会把"权限问题"误读成"实验失败"。
    jobs_before = len(app.list_jobs())
    with pytest.raises(GateDenied) as denied:
        app.submit({"command": "run_method", "method": "logistic",
                    "dataset_id": dataset_id}, side="human")
    assert denied.value.kind == "authority_mode_mismatch", denied.value
    assert len(app.list_jobs()) == jobs_before, "预检失败不得留下半拉子 job"


def test_writes_state_declaration_matches_actual_history(app):
    """P2-7 防复发：声明的 ``writes_state`` 必须**逐键等于**实际写入的键集合。

    做法是机械对账而不是读代码：四条命令各跑一次，收集它在 History 里
    实际写下的 ``current.*`` 键（actor=ml-ai 的那些），与该命令声明的集合
    比双向相等——多声明（谎报会写）少声明（偷偷写）都判红。
    """
    from ml_mecha.commands import command_map

    app.switch_ai()                     # AI 侧持有全部 4 条 scope，四条都真跑

    def _actual(spec, *, allow_fail=False):
        mark = len(app.history.events())
        receipt = _as(app, app.ai_channel, spec)
        if not allow_fail:
            assert receipt["ok"] is True, receipt
        keys = {e.target for e in app.history.events()[mark:]
                if e.actor == "ml-ai" and e.target.startswith("current.")}
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
                  "dataset_id": dataset_id},
                 {"command": "run_custom", "dataset_id": dataset_id,
                  "code": _RESIDUAL_CODE, "overrides": {"epochs": 5},
                  "device": "cpu"},
                 # ⚠ `eval_custom` 声明 `writes_state=()`（它不改配置台面）⇒ 这里
                 #   **允许它失败**（没有 checkpoint 自然会失败），但要求它
                 #   **一个状态键都不写** —— "不改台面的命令，失败也不能改台面"。
                 #   这比"跳过它"强：跳过等于不检查。
                 {"command": "eval_custom", "run_id": "custom_network-0-0",
                  "code": "def evaluate(model, ctx):\n    return {}"},
                 # 清零放**最后**：它会动前面几步写下的同一批键
                 # （声明要求"声明 == 实际写入"，而清零写的是全部状态键）。
                 {"command": "clear_current_state"}):
        _receipt, keys = _actual(spec, allow_fail=(spec["command"] == "eval_custom"))
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
    events = [e for e in app.history.events() if e.op == "ml.run"]
    assert events, "没有 ml.run 事件"
    locator = events[-1].after.get("artifact_locator")
    assert locator, f"事件缺 artifact_locator：{events[-1].after}"
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


# ---------------------------------------------------------------- 3b. 神经网络观测面
#: 训练观测判据用的固定 run_id（事件目录 = `runs/<PROBE_ID>/tb/`，用完即清）。
_NN_PROBE_ID = "nncrit-0000001-42"


def _has_torch() -> bool:
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def _clean_nn_probe():
    import shutil
    from pathlib import Path

    from ml_toolbox.core.persistence import RUNS_DIR
    shutil.rmtree(Path(RUNS_DIR) / _NN_PROBE_ID, ignore_errors=True)


def _train_nn(app, dataset_id, *, epochs=6):
    """用**固定 run_id** 真训练一次（事件文件因此落在可预期的目录里）。"""
    from ml_toolbox.core import registry, runner
    from ml_toolbox.core.contracts import RunConfig

    registry.load_builtin()
    entry = app.engine.dataset_entry(dataset_id)
    assert entry is not None
    _clean_nn_probe()
    cfg = RunConfig(overrides={"hidden": "16,8", "epochs": epochs,
                               "latent_every": 3},
                    extras={"run_id": _NN_PROBE_ID})
    rec = runner.run_one(registry.get("torch_mlp"), entry.spec, cfg)
    assert rec.result.ok, f"训练本身失败：{rec.result.error}"
    return rec


def test_device_declaration_reaches_the_method_layer(app):
    """⭐ 引擎的 `device` 声明必须**进 RunRequest.extras**（它曾是死参数）。

    实测的缺陷（2026-10-03）：`run_method(device=...)` 被 `clean_device` 清洗、
    也写进了 `current.device` 域状态与审计，却**从不进 RunRequest** ⇒ 方法层
    永远收不到它。后果是"审计说 cuda、实跑 CPU"——**比参数不存在更糟**，
    因为它留下了一条假记录。

    ⚠ 这条判据**只钉管道**（extras 里有没有那个键），不钉"实际用了哪块卡"：
    后者依赖本机有没有 CUDA，写成 `diag["device"] == "cpu"` 会在无卡机器上
    **永远绿**（`auto` 本来就落 cpu）⇒ 我第一版就是这么写错的，做红验证时
    两次都绿才发现。解析规则另由
    `test_regressions.py::test_neural_device_resolution_is_deterministic` 钉。
    """
    en = app.engine
    req, options = en._build_request(
        {"command": "run_method", "method": "logistic", "device": "cuda:1"},
        method="logistic")
    assert req.extras.get("device") == "cuda:1", req.extras
    cfg = req.config()
    assert cfg.extras.get("device") == "cuda:1", \
        "device 没进 config.extras ⇒ 方法层收不到（`config()` 是拷贝语义）"
    # 不传时是声明里的默认（auto），不是一个空串 —— 方法层要能分辨"没要求"
    req2, _ = en._build_request({"command": "run_method", "method": "logistic"},
                                method="logistic")
    assert req2.extras.get("device") == "auto", req2.extras


def test_training_observations_are_reachable_from_the_tool_face(app):
    """⭐ AI **经工具面**能拿到逐 epoch 训练观测（原先只能走 HTTP 只读路由）。

    为什么这条最要紧（2026-10-03 实测的缺口）：观测数值一直是**逐 epoch 落盘**的
    （`runs/<id>/tb/`），但只出现在 `/latest`、`/series` 两条 HTTP 路由上 ⇒
    AI 经 MCP **一条都拿不到**，"梯度爆没爆 / 哪层死了 / 要不要早停"全答不了；
    而 `describe_run` 只回**产物类别名**（`nn_history` 在列表里、数值不给）。
    ⇒ 本条钉的就是"这条链真的通了"：训练 → 事件文件 → 查询面 → 工具面。
    """
    if not _has_torch():                       # torch 是可选依赖，缺就**明确**跳过
        print("    （跳过：未装 torch）")
        return
    dataset_id = _prepared(app)
    try:
        _train_nn(app, dataset_id)

        # ① 只读查询：最新一轮的数值真给出来了
        detail = app.engine.query("training_detail", run_id=_NN_PROBE_ID)
        assert detail["available"], detail.get("reason")
        latest = detail["latest"]
        assert latest["available"]
        for tag in ("loss", "val_loss", "grad_norm", "dead_relu"):
            assert tag in latest, f"缺 {tag}（实得 {sorted(latest)}）"
        assert latest["grad_by_layer"], "缺逐层梯度（'哪层死了'要靠它）"
        assert detail["tags"], "tags 为空 ⇒ 工具不知道该读什么"

        # ② 工具面：AI 真的调得到（且默认不内联整条序列）
        got = app.tools.execute("describe_training", {"run_id": _NN_PROBE_ID})
        assert got["is_error"] is False, got
        value = got["value"]
        assert value["available"] is True
        assert value["latest"]["loss"] is not None
        assert "series" not in value, "默认不该内联整条序列（响应会很大）"

        # ③ 要趋势时显式打开：每个 tag 都带自己的 steps（tag 可以晚开始）
        with_series = app.tools.execute(
            "describe_training",
            {"run_id": _NN_PROBE_ID, "include_series": True, "limit": 50})["value"]
        assert with_series["series"], "没回序列"
        loss = with_series["series"]["loss"]
        assert len(loss["steps"]) == len(loss["values"]) == 6, loss
        assert loss["values"] == sorted(loss["values"], reverse=True), \
            "这 6 个 epoch 的 loss 应当单调下降（方向反了？）"
        assert with_series["max_points"] > 0, "没告诉调用方截断上限"

        # ④ 数据来自**磁盘**（不依赖会话内存）——这正是本工具存在的独立理由
        assert app.engine.query("training_detail", run_id=_NN_PROBE_ID,
                                include_series=True)["available"]
    finally:
        _clean_nn_probe()


def test_progress_hook_must_be_attached_before_session_run(app):
    """⭐ 机制判据：钩子必须挂进 ``request.extras``，**且要早于 ``config()`` 那次拷贝**。

    本条专门钉那个"太晚"的坑（见 `test_per_epoch_progress_reaches_read_job`）：
    ``RunRequest.config()`` 是 `extras=dict(self.extras)` 的**一次性快照**
    （`api.py:89-94`）⇒ 挂在 `session.run` 之后的钩子**永远传不到 cfg**，
    而症状只在后台 job 上显形（同步路径会假绿）。
    ⇒ 直接断言"挂上之后 `config()` 认得出它"，比只断言"心里想着要挂"强。
    """
    from mecha.surface import ExecutionContext
    from ml_mecha import engine as E

    class _Cfg:
        def __init__(self, extras):
            self.extras = extras

    class _Req:
        """照抄 `RunRequest.config()` 的**拷贝语义**（真实的那份在 `api.py:89-94`）。

        ⚠ 必须连 `method`/`seed` 一起给：`_attach_progress` 在算 `run_id` 时要用它们
        （`f"{method}-{ms}-{seed}"`，与 `runner.run_one` 同一规则）。
        这条判据此前**只在 pytest 下跑过、门禁不跑**（main 用手写列表）⇒ 桩缺字段
        导致它一直是红的却没人看见；改成内省收集后立刻暴露。
        """
        def __init__(self, method="logistic", seed=7):
            self.extras = {}
            self.method = method
            self.seed = seed

        def config(self):
            return _Cfg(dict(self.extras))

    ctx = ExecutionContext()
    req = _Req()
    E._attach_progress(req, ctx)
    assert callable(req.extras.get("on_progress")), "钩子没挂上 request.extras"
    assert callable(req.config().extras.get("on_progress")), \
        "钩子挂晚了：`config()` 的拷贝看不见它 ⇒ 方法层收不到"

    # 真调一下：分数落在 [0,1]、阶段名带 epoch 进度
    req.extras["on_progress"](2, 4, "epoch")
    frac, stage = ctx.progress()[-1]
    assert frac == 0.75, frac
    assert "3/4" in stage, stage

    # context 缺失/不合用时是 no-op（不能因为"没有执行上下文"就炸）
    bare = _Req()
    E._attach_progress(bare, None)
    assert bare.extras == {}, bare.extras


def test_training_observation_degrades_honestly(app):
    """读不到 ⇒ **说清为什么**，不是空数组假装"跑到第 0 个 epoch"。"""
    got = app.tools.execute("describe_training", {"run_id": "definitely-no-such-run"})
    assert got["is_error"] is False, "只读工具不该因为'没有数据'报错"
    value = got["value"]
    assert value["available"] is False
    assert value["reason"], "读不到却没说原因"
    assert value["tags"] == [] and value["latest"]["available"] is False


#: 残差网络的源码（**这是旋钮路线永远表达不了的拓扑**：`x + f(x)` 需要跨层引用输入，
#: 而 `nn.Sequential` 只能表达线性链 —— 见 `mlp.py::_build`）。
_RESIDUAL_CODE = '''
class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.fc1, self.fc2, self.act = nn.Linear(w, w), nn.Linear(w, w), nn.ReLU()

    def forward(self, x):
        return self.act(x + self.fc2(self.act(self.fc1(x))))


class Network:
    """输入投影 → 若干个残差块 → 输出头。"""

    def __init__(self, width=16, blocks=2):
        self.width, self.blocks = width, blocks

    def build(self, n_in, n_out, *, nn):
        layers = [nn.Linear(n_in, self.width), nn.ReLU()]
        layers += [Block(self.width) for _ in range(self.blocks)]
        layers.append(nn.Linear(self.width, n_out))
        return nn.Sequential(*layers)
'''


def test_custom_network_runs_and_keeps_the_observation_surface(app):
    """⭐ AI 写网络代码能真跑，而且**观测面与检视面照旧**（不另开一条路）。

    这是"自由度交给 AI、固定部分交给程序"那套理念的落地判据：
    - **自由**：残差网络的跳跃连接是 `nn.Sequential` 表达不了的拓扑，
      旋钮堆再多也生不出来（这一条就是"为什么需要 codeact"的实证）；
    - **固定**：训练循环/TensorBoard/逐 epoch 观测/指标/存档/检视页
      **一行都没重写**——因为跑的就是一台 `MLMethod`（适配器代写 `fit`）。

    ⚠ 判据刻意**不**断言"残差网比 MLP 好"：那是数据与调参的事，不是契约的事。
    它断言的是"能跑 + 观测面在 + 拓扑真的是残差"。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    dataset_id = _prepared(app)
    try:
        got = app.tools.execute("run_custom", {
            "device": "cpu", "code": _RESIDUAL_CODE,
            "dataset_id": dataset_id, "name": "resnet_crit",
            "diag": True, "persist": True, "seed": 4242,
            "overrides": {"epochs": 6, "lr": 0.01}})
        assert got["is_error"] is False, got
        value = got["value"]
        assert value["ok"] is True, value
        assert value["network_name"] == "resnet_crit"
        rid = value["run_id"]

        # ① 拓扑真的是残差 —— 并且**旋钮路线生不出它**（负对照）
        #    ⚠ 不能用存档里的模型验：`load_run_record` **不带 estimator**（设计如此，
        #      历史载入的运行没有已拟合模型）⇒ 这里拿沙箱编译产物验拓扑。
        import torch as _t

        from ml_mecha.sandbox import call_network_build, compile_network
        net = compile_network(_RESIDUAL_CODE).instance
        model = call_network_build(net, 6, 3, _t.nn)
        names = [type(m).__name__ for m in model.modules()]
        assert "Block" in names, f"自定义网络里没有残差块：{sorted(set(names))}"
        # 负对照：内置 torch_mlp 的结构里**永远**不会有这种块
        # （`_build` 返回 Sequential ⇒ 表达不了 `x + f(x)`）
        from ml_toolbox.core import registry as _reg
        from ml_toolbox.core.contracts import RunConfig as _RC
        _mlp = _reg.get("torch_mlp")
        p_mlp = _RC(overrides={"hidden": "8,4"}).resolve(_mlp.param_schema)
        builtin = _mlp._build(p_mlp, 6, 3, _t.nn)
        assert isinstance(builtin, _t.nn.Sequential), \
            "内置 MLP 不再是 Sequential ⇒ '拓扑写死'这个论证前提变了"
        assert not any(type(m).__name__ == "Block" for m in builtin.modules()), \
            "内置 MLP 里出现了残差块？论证前提变了"

        from pathlib import Path

        from ml_toolbox.core.persistence import RUNS_DIR, load_run_record
        try:
            record = load_run_record(rid)
            # ② 观测面白拿：TB/describe_training 必须有逐 epoch 数据
            train = app.tools.execute("describe_training",
                                      {"run_id": rid, "include_series": True})["value"]
            assert train["available"] is True, train.get("reason")
            assert len(train["series"]["loss"]["values"]) == 6, train["series"]["loss"]
            for tag in ("loss", "val_loss", "grad_norm", "dead_relu"):
                assert tag in train["tags"], f"缺 {tag}（{sorted(train['tags'])}）"
            # ③ 五页检视沿用（artifacts 与 torch_mlp 同构）
            for key in ("nn_history", "nn_weights", "nn_latent", "nn_layers"):
                assert key in record.result.artifacts, f"缺工件 {key}"
            # ④ 代码跟着运行走（跨会话取回、按名字复用的前提）
            src = record.result.artifacts.get("custom_network_source")
            assert src is not None and "class Network" in str(src[0])
            assert record.config.get("extras", {}).get("network_name") == "resnet_crit"
        finally:
            import shutil
            shutil.rmtree(Path(RUNS_DIR) / rid, ignore_errors=True)

        # ⑤ 临时注册必须撤干净：注册表里不许留下"代码可能已经不对"的空壳
        from ml_toolbox.core import registry
        assert "custom_network" not in registry.names(), \
            "run_custom 跑完没撤注册项 ⇒ 注册表里留了一个代码可能过期的壳"
    finally:
        _clean_nn_probe()


def test_custom_network_reuse_by_name_avoids_repasting(app):
    """⭐ 网络能**按名字取回**（改一版再跑，不必重贴整段代码）。

    为什么单列（用户点破的实际风险）：改个学习率就要重贴一遍网络代码的话，
    token 全烧在复制粘贴上 ⇒ "自由"换来的是低效。⇒ 本条钉"取回可得、可改、可再跑"。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    from pathlib import Path
    import shutil

    from ml_toolbox.core.persistence import RUNS_DIR

    dataset_id = _prepared(app)
    made: list[str] = []
    try:
        first = app.tools.execute("run_custom", {
            "device": "cpu", "code": _RESIDUAL_CODE,
            "dataset_id": dataset_id, "name": "reuse_crit",
            "persist": True, "overrides": {"epochs": 5}})["value"]
        made.append(first["run_id"])

        cat = app.tools.execute("describe_networks", {})["value"]
        names = [r["name"] for r in cat["networks"]]
        assert "reuse_crit" in names, names

        got = app.tools.execute("describe_network", {"name": "reuse_crit"})["value"]
        assert got["available"] is True, got.get("reason")
        assert "class Network" in got["code"]
        assert got["code"].strip() == _RESIDUAL_CODE.strip(), "取回的源码与原文不一致"

        # 改一版（只动一行）再跑：这正是"不重贴"的价值
        tweaked = got["code"].replace("blocks=2", "blocks=3")
        assert tweaked != got["code"]
        second = app.tools.execute("run_custom", {
            "device": "cpu", "code": tweaked,
            "dataset_id": dataset_id, "name": "reuse_deep",
            "persist": True, "overrides": {"epochs": 5}})["value"]
        made.append(second["run_id"])
        assert second["ok"] is True, second

        # 取不存在的名字 ⇒ 如实说（不回空源码假装成功）
        miss = app.tools.execute("describe_network", {"name": "no_such_net"})["value"]
        assert miss["available"] is False and miss["reason"], miss
    finally:
        for rid in made:
            shutil.rmtree(Path(RUNS_DIR) / rid, ignore_errors=True)


def test_custom_network_errors_are_readable_not_raw_stacks(app):
    """沙箱错误必须**可教学**：说清是哪一类 + 怎么改，而不是一句裸栈。

    为什么（用户裁决的边界）：执行走 `exec`，代码由 AI 产出 ⇒ 报错质量直接决定它
    能不能自己改对。⇒ 五种典型错法各有专门的 `kind` 与提示，且**都不许**逃逸成
    `internal_error`（那等于把"你写错了"和"框架坏了"混成一样）。
    """
    dataset_id = _prepared(app)

    cases = [
        ("空代码", "", "empty_code"),
        ("语法错", "class Network\n    pass", "code_syntax_error"),
        ("没有 Network 类", "class Foo:\n    pass\n", "missing_network_class"),
        ("缺 build", "class Network:\n    pass\n", "missing_build"),
        ("build 返回错类型",
         "class Network:\n    def build(self, n_in, n_out, *, nn):\n"
         "        return 42\n", "build_wrong_type"),
    ]
    for label, code, want_kind in cases:
        got = app.tools.execute("run_custom", {"device": "cpu", "code": code,
                                           "dataset_id": dataset_id})
        assert got["is_error"] is True, f"{label} 竟然没报错"
        blob = json.dumps(got, ensure_ascii=False)
        assert want_kind in blob or "Network" in blob or "语法" in blob, \
            f"{label} 的错误不可读：{blob[:300]}"
        assert "internal_error" not in blob, f"{label} 被归成了内部错误：{blob[:300]}"


#: 最小网络代码（判据里做"能跑就行"的提交用）。
_MIN_NET_CODE = ("class Network:\n"
                 "    def build(self, n_in, n_out, *, nn):\n"
                 "        return nn.Sequential(nn.Linear(n_in, 8), nn.ReLU(),\n"
                 "                             nn.Linear(8, n_out))\n")


def test_background_custom_network_reports_progress_and_run_id(app):
    """⭐ 后台跑自定义网络：**逐代数值 + 训练中途就拿到 run_id**（用户裁决的"发射后不管"）。

    这是把"AI 控制训练"做成**人类操作 GUI 那样**的那条路：发一个开始信号就走，
    之后自己轮询。三个必须成立的事实：

    1. `submit_run(code=...)` 能提交 —— 它以前把命令**写死**成 `run_method`
       ⇒ 自定义网络只能同步跑，而同步受客户端 10 分钟超时（`TOOL_TIMEOUT`）
       ⇒ **长训练等于跑不了**（超时后连 `run_id` 都拿不到）；
    2. 进度里带 **`loss`/`val_loss`**（有界窗口）—— 一次轮询就能判断该不该早停，
       不必再拉一次完整曲线；
    3. 进度里带 **`run_id`** —— `result_ref` 只在终态才有，不带它就没法"边跑边查曲线"。

    ⚠ 判据不 sleep 等训练跑完（那会变成时序噪音）：提交一个**足够长**的训练，
    在它还在 running 时就断言上面三件，然后取消收尾。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    dataset_id = _prepared(app)
    job_id = ""
    try:
        got = app.tools.execute("submit_run", {
            "device": "cpu", "code": _RESIDUAL_CODE, "name": "bg_crit",
            "dataset_id": dataset_id,
            "seed": 5150, "diag": True,
            "overrides": {"epochs": 400, "batch": 2, "lr": 0.05}})
        assert got["is_error"] is False, got
        value = got["value"]
        job_id = value["job_id"]
        # ① 提交的是 run_custom（不是被写死的 run_method）
        assert value["command"] == "run_custom", value

        # ②③ 轮询：训练**还在跑**的时候，就应看到逐代数值与 run_id
        import time
        seen_run_id, seen_loss, seen_epochs = "", None, 0
        for _ in range(60):
            time.sleep(0.5)
            j = app.tools.execute("read_job", {"job_id": job_id})["value"]
            latest = j.get("progress_latest") or {}
            if not seen_run_id and j.get("run_id"):
                seen_run_id = j["run_id"]
            if latest.get("loss") is not None:
                seen_loss = latest
            seen_epochs = max(seen_epochs, len(j.get("epochs") or []))
            if seen_run_id and seen_loss is not None and seen_epochs >= 2:
                break
            if j["state"] != "running":
                break
        assert seen_run_id, "训练中途拿不到 run_id ⇒ '边跑边查曲线'不可用"
        assert seen_loss is not None, "进度里没有 loss/val_loss ⇒ 轮询无法决策早停"
        assert seen_epochs >= 2, f"逐代数值窗口没在长（{seen_epochs} 条）"
        assert {"loss", "val_loss"} <= set(seen_loss), seen_loss
        # 有界：窗口不会随训练轮数无限增长
        from ml_mecha.engine import PROGRESS_EPOCH_WINDOW
        assert seen_epochs <= PROGRESS_EPOCH_WINDOW, seen_epochs

        # ④ 中途就能按 run_id 查曲线（不必等终态）
        train = app.tools.execute("describe_training", {
            "run_id": seen_run_id, "include_series": True})["value"]
        assert train["available"] is True, train.get("reason")
        assert len(train["series"]["loss"]["values"]) >= 1

        # ⑤ 取消：**真的在 epoch 边界停下**（协作式），不是等它跑满 400 轮
        c = app.tools.execute("cancel_run", {"job_id": job_id})["value"]
        assert c["ok"] is True and c["cancel_requested"] is True, c
        for _ in range(60):
            time.sleep(0.5)
            j = app.tools.execute("read_job", {"job_id": job_id})["value"]
            if j["state"] != "running":
                break
        assert j["state"] == "cancelled", j["state"]
        ran = j["progress_latest"]["epoch"]
        assert ran < 399, f"取消后还跑到第 {ran} 轮 ⇒ 训练器没有协作式取消点"
    finally:
        if job_id:
            app.tools.execute("cancel_run", {"job_id": job_id})
        from pathlib import Path

        from ml_toolbox.core.persistence import RUNS_DIR
        import shutil
        for d in Path(RUNS_DIR).glob("custom_network-*-5150"):
            shutil.rmtree(d, ignore_errors=True)


def test_internal_job_marker_never_leaks_into_command_params(app):
    """⭐ 后台标记必须被**摘掉**后再校验（它不是命令参数）。

    实测踩过：给后台请求打 `{"job_run": True}` 之后，`validate_spec` 立刻判
    "接受了未声明的参数" ⇒ **job 以 ok=False 瞬间结束**，看起来像"训练秒完"，
    而真相是它根本没跑。⇒ 引擎在 `run` 里**校验之前**把它摘成显式形参。

    本条钉两个方向：① `JOB_RUN_FLAG` 不在任何命令的声明参数里；
    ② 带标记跑一次 `run_method` 仍然成功（标记被摘掉、且真的生效为 persist）。
    """
    from ml_mecha.commands import COMMANDS
    from ml_mecha.engine import JOB_RUN_FLAG

    for command in COMMANDS:
        names = {p.name for p in command.parameters}
        assert JOB_RUN_FLAG not in names, \
            f"{command.name} 把内部标记声明成了命令参数（它不该出现在参数面）"

    dataset_id = _prepared(app)
    # 带标记跑一次：必须成功（而不是像"未声明的参数"那样瞬间失败）
    # ⚠ 走 `_as`（显式绑通道）——裸 `engine.run` 没有操作者通道，会以
    #   `missing_channel` 被拒（那是另一条闸，不是本判据要测的东西）。
    receipt = _as(app, app.ai_channel,
                  {"command": "run_method", "method": "logistic",
                   "dataset_id": dataset_id, JOB_RUN_FLAG: True})
    assert receipt.get("ok") is True, receipt
    rid = receipt["run_id"]
    try:
        # 标记的**唯一作用**是打开 persist ⇒ 应该有存档
        from pathlib import Path

        from ml_toolbox.core.persistence import RUNS_DIR
        rec = Path(RUNS_DIR) / rid / "record.json"
        assert rec.is_file(), "后台标记没有生效（persist 没被打开）"
    finally:
        from pathlib import Path

        from ml_toolbox.core.persistence import RUNS_DIR
        import shutil
        shutil.rmtree(Path(RUNS_DIR) / rid, ignore_errors=True)


def test_background_job_persists_without_asking(app):
    """⭐ 后台任务**终态自动落盘**（用户裁决：别让"意外"白训练）。

    为什么：job 状态只在**进程内存**里（注册表无淘汰、也无持久化）⇒ 权威一重启，
    `job_id` 与 `result_ref` 全没了：训练其实跑完了、数据也在 `runs/`，但**没有路标**。
    ⇒ 后台任务默认落盘，等于给结果留一个不依赖"进程活到取回那一刻"的锚。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    from pathlib import Path

    from ml_toolbox.core.persistence import RUNS_DIR
    import shutil

    dataset_id = _prepared(app)
    rid = ""
    try:
        got = app.tools.execute("submit_run", {
            "device": "cpu", "code": _MIN_NET_CODE, "name": "persist_crit",
            "dataset_id": dataset_id,
            "seed": 6160, "overrides": {"epochs": 5}})
        assert got["is_error"] is False, got
        job_id = got["value"]["job_id"]
        # ⚠ **不传 persist**：这正是本判据的要点
        for _ in range(120):
            j = app.tools.execute("read_job", {"job_id": job_id})["value"]
            rid = j.get("run_id") or rid
            if j["state"] != "running":
                break
            import time
            time.sleep(0.5)
        assert j["state"] == "done", j
        rid = rid or ((j.get("result_ref") or {}).get("run_ids") or [""])[0]
        assert rid, "拿不到 run_id"
        rec = Path(RUNS_DIR) / rid / "record.json"
        assert rec.is_file(), "后台任务跑完没落盘 ⇒ 丢了 job 就再也找不回结果"
    finally:
        if rid:
            shutil.rmtree(Path(RUNS_DIR) / rid, ignore_errors=True)


#: 自定义**预处理步骤**源码：造一个内置步骤做不到的特征（逐行比值）。
_CUSTOM_STEP_CODE = '''
class Step(StepBase):
    key, title = "ratio", "造个比值特征"

    def apply(self, df, target, diag=False):
        out = df.copy()
        cols = [c for c in out.columns if c != target]
        if len(cols) >= 2:
            out["f0_over_f1"] = out[cols[0]] / (out[cols[1]].abs() + 1e-9)
        return out, {"added": 1, "big_num": "+1 列"}
'''


def test_ai_can_assemble_the_preprocessing_chain(app):
    """⭐ AI **自己组装预处理链**：配内置步骤 + 写自定义步骤 + 指定切分。

    为什么这是"自由化的那半"（用户裁决）：默认链（中位数填 → one-hot → 标准化 →
    8:2 切分）对多数题够用，但换策略/加特征工程/改切分比例就没办法了。
    本判据钉住四件：

    1. **顺序 = 声明顺序**（那是作者意图，不做任何重排）；
    2. **自定义步骤真跑**（`class Step(StepBase)` 由沙箱编译后注入管道）；
    3. **切分口径真的按声明生效**，且 `describe_dataset` 报的是**这份数据自己的**
       口径（不是引擎默认值——AI 现在可以逐数据集指定切分）；
    4. **坏声明一律响亮**（未知步骤/未知键/坏取值/坏切分都不静默忽略）。
    """
    ml = app
    # ⚠ 工具面**绑在 ai 通道**上 ⇒ 模式必须是 ai。
    #   `_prepared` 会"人类侧准备一份数据 + 开 AI 闸"（出厂是 LOCKED）。
    _prepared(ml)
    got = ml.tools.execute("prepare_dataset", {
        "source": {"kind": "synthetic", "task": "classification", "n_samples": 150},
        "name": "assembled", "target": "target",
        "pipeline": {"steps": [
            {"key": "missing", "params": {"strategy": "mean"}},
            {"key": "ratio", "code": _CUSTOM_STEP_CODE},
            {"key": "scale", "params": {"mode": "robust"}},
        ], "split": {"test_size": 0.2, "seed": 7, "stratify": True}}})
    assert got["is_error"] is False, got
    v = got["value"]
    assert [c["step"] for c in v["pipeline_chain"]] == ["missing", "ratio", "scale"], \
        v["pipeline_chain"]
    assert v["split_policy"] == {"test_size": 0.2, "stratify": True, "seed": 7,
                                 "time_split": False}, v["split_policy"]
    # 自定义步骤真的加了一列（n_features 比输入多 1）
    assert v["n_features"] == 5, v["n_features"]        # synthetic 默认 4 列 + ratio

    # ⭐ 查询面报的是**这份数据集自己的**划分口径（不是引擎默认的 0.25/42）
    detail = app.engine.query("dataset_detail", dataset_id=v["dataset_id"])
    assert detail["split_policy"]["test_size"] == 0.2, detail["split_policy"]
    assert detail["split_policy"]["seed"] == 7, detail["split_policy"]
    by = {c["step"]: c for c in detail["pipeline_chain"]}
    assert by["missing"]["summary"].get("strategy") == "mean"
    assert by["scale"]["summary"].get("mode") == "robust"
    assert by["ratio"]["summary"].get("added") == 1, by["ratio"]

    # 这条链产出的数据**能用**（跑个方法，不是只装配了好看）
    run = ml.tools.execute("run_method", {"method": "logistic",
                                          "dataset_id": v["dataset_id"]})
    assert run["is_error"] is False, run
    assert run["value"]["metrics"]["f1"] > 0.5, run["value"]["metrics"]

    # ---- 顺序就是顺序 + enabled=false 会被跳过
    order = ml.tools.execute("prepare_dataset", {
        "source": {"kind": "synthetic", "task": "classification", "n_samples": 80},
        "name": "order", "target": "target",
        "pipeline": {"steps": [{"key": "scale"}, {"key": "encode"},
                               {"enabled": False, "key": "missing"}]}})["value"]
    chain = ml.tools.execute("describe_dataset",
                             {"dataset_id": order["dataset_id"]})["value"]["pipeline_chain"]
    assert [c["step"] for c in chain] == ["scale", "encode", "missing"], chain
    assert chain[-1]["skipped"] is True, chain

    # ---- 坏声明一律响亮（**不静默忽略**）
    bad = [
        {"steps": [{"key": "no_such_step"}]},                        # 未知步骤
        {"steps": [{"key": "scale", "bogus": 1}]},                    # 未知键
        # ⚠ 这条专门打"参数名不在该步骤的可调表里"——它与"坏取值"是**两条不同的
        #   守卫**（`unknown_step_param` vs `bad_step_param_value`）。我第一版只写了
        #   坏取值 ⇒ 那条参数名守卫**判据覆盖不到**（停掉它判据照样绿，实测发现）。
        {"steps": [{"key": "scale", "params": {"no_such_param": 1}}]},
        {"steps": [{"key": "scale", "params": {"mode": "bogus"}}]},   # 坏取值
        {"steps": [{"key": "scale"}], "split": {"test_size": 1.5}},   # 坏切分
        {"steps": [{"key": "scale"}], "split": {"ratio": 0.2}},       # 未知 split 键
        {"steps": [{"key": "x", "code": "class Step\n  pass"}]},      # 语法错
        {"steps": [{"key": "x", "code": "class Step:\n    pass\n"}]},  # 没继承基类
    ]
    for decl in bad:
        out = ml.tools.execute("prepare_dataset", {
            "source": {"kind": "synthetic", "task": "classification", "n_samples": 60},
            "name": "bad", "target": "target", "pipeline": decl})
        assert out["is_error"] is True, f"坏声明被接受了：{decl}"
        blob = json.dumps(out, ensure_ascii=False)
        assert "internal_error" not in blob, f"坏声明被归成内部错误：{blob[:200]}"


#: 一个**真的图卷积**（AI 通过工具面提交的 GNN；仓库依赖里没有 PyG/DGL，也不需要）。
#: ⚠ 张量一律用 `ctx["device"]` 建：**模型可能已被搬到显卡**（`device=auto` 有卡
#: 就上卡）⇒ 随手 `torch.arange(...)` 会撞"混合设备"。这段代码同时也是**范例**。
_GCN_TOOL_CODE = '''
class GCNLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)

    def forward(self, h, edge_index, n_nodes, edge_weight=None, device=None):
        idx = torch.arange(n_nodes, device=device)
        src = torch.cat([edge_index[0], idx])
        dst = torch.cat([edge_index[1], idx])
        w = torch.cat([edge_weight, torch.ones(n_nodes, device=device)]) \\
            if edge_weight is not None else torch.ones(src.numel(), device=device)
        A = torch.zeros(n_nodes, n_nodes, device=device)
        A.index_put_((src, dst), w, accumulate=True)
        A = A.clamp(max=1.0)
        deg = A.sum(1).clamp(min=1e-6)
        Dm = torch.diag(deg.pow(-0.5))
        return self.lin(Dm @ A @ Dm @ h)


class Net(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.g1 = GCNLayer(n_in, n_hidden)
        self.g2 = GCNLayer(n_hidden, n_out)

    def forward(self, x, edge_index, n_nodes, edge_weight=None, device=None):
        h = torch.relu(self.g1(x, edge_index, n_nodes, edge_weight, device))
        return self.g2(h, edge_index, n_nodes, edge_weight, device)


class Network:
    def build(self, n_in, n_out, *, nn):
        return Net(n_in, 16, n_out)

    def forward_graph(self, model, ctx):
        # ⚠ 设备从 ctx 取（见上）：模型可能已在显卡上
        return model(ctx["x"], ctx["edge_index"], ctx["n_nodes"],
                     ctx.get("edge_weight"), ctx.get("device"))
'''


def _sbm_graph(n_per=20, seed=3):
    """随机块模型（工具面判据用）：返回 (X 列表, y 列表, 边列表)。"""
    import numpy as np
    rng = np.random.RandomState(seed)
    labels = np.repeat([0, 1, 2], n_per)
    centers = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, -2.0]])
    X = centers[labels] + rng.normal(scale=0.6, size=(len(labels), 2))
    edges = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            p = 0.35 if labels[i] == labels[j] else 0.03
            if rng.rand() < p:
                edges.append([int(i), int(j)])
    return X.tolist(), labels.tolist(), edges


def test_gnn_runs_through_the_tool_face(app):
    """⭐ **AI 经工具面跑 GNN**：`prepare_dataset(graph=...)` + `run_custom` 的图网络。

    这条是"目标达成"的用户可见证据（不是内部能跑就行）：
    1. 工具面**收得下图**（边表 + 节点特征 + 节点标签），并把拓扑事实回报出来；
    2. 不声明支持图的方法**被明确拒绝**（不静默忽略拓扑）；
    3. AI 写的图卷积**真能跑通**，且观测面/工件与表格任务同构
       （`describe_training` 有逐 epoch 曲线；`y_pred` 按 **test 掩码**对齐）。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    ml = app
    _prepared(ml)                       # 开 AI 闸
    X, y, edges = _sbm_graph()
    n = len(y)

    got = ml.tools.execute("prepare_dataset", {
        "source": {"kind": "arrays", "X": X, "y": y},
        "name": "gnn_tool", "target": "target",
        "graph": {"edges": edges},
        "pipeline": {"steps": [{"key": "scale"}]}})
    assert got["is_error"] is False, got
    v = got["value"]
    ds = v["dataset_id"]
    # ① 工具面回报拓扑事实（无向图 ⇒ 边数翻倍）
    gf = v.get("graph") or {}
    assert gf.get("n_nodes") == n, gf
    assert gf.get("n_edges") == 2 * len(edges), (gf, len(edges))

    # ② 不声明支持图的方法 → 明确拒绝（不是静默按独立节点跑）
    bad = ml.tools.execute("run_method", {"method": "torch_mlp", "dataset_id": ds,
                                          "overrides": {"epochs": 5}})
    assert bad["is_error"] is True, "带图数据跑 torch_mlp 竟然成功了"
    assert "辅助结构" in json.dumps(bad, ensure_ascii=False)

    # ③ GNN 跑通 + 观测面同构
    out = ml.tools.execute("run_custom", {
        "device": "cpu", "code": _GCN_TOOL_CODE, "dataset_id": ds,
        "name": "gcn_tool",
        "diag": True, "seed": 8080, "overrides": {"epochs": 30, "lr": 0.05}})
    assert out["is_error"] is False, out
    val = out["value"]
    assert val["ok"] is True, val
    rid = val["run_id"]
    train = ml.tools.execute("describe_training",
                             {"run_id": rid, "include_series": True})["value"]
    assert train["available"] is True, train.get("reason")
    assert len(train["series"]["loss"]["values"]) == 30, train["series"]["loss"]
    detail = ml.tools.execute("describe_run", {"run_id": rid})["value"]
    assert "graph_n_nodes" in detail["artifact_keys"], detail["artifact_keys"]


# ---------------------------------------------------------------- AI 调用体验
def test_clear_current_state_clears_the_bench_with_per_key_evidence(app):
    """⭐ 清零入口（指导意见 §1-P1-5）：**逐键 before/readback**，半清要响亮。

    为什么必须有这个入口：配置是**跨会话残留**的。没有它，AI 是在别人（或上一轮
    自己）没收拾的台面上干活——"我用默认配置跑一下"会变成假话，而且**没有任何
    办法回到干净状态**。

    钉三件：
    1. 清零真的落到域状态（读快照确认，不是只写回执）；
    2. 回执**逐键**给 `before`（写前值）与 `readback`（写后读回）——省掉一次往返；
    3. `n_cleared` 只数**真有值**的键，第二次调它必须诚实报 0（不是每次都报 7）。
    """
    ml = app
    ds = _prepared(ml)
    ml.tools.execute("run_method", {"method": "logistic", "dataset_id": ds})
    before_snap = dict(ml.software.gate.snapshot)
    assert before_snap.get("current.method") == "logistic", before_snap

    got = ml.tools.execute("clear_current_state", {})
    assert got["is_error"] is False, got
    v = got["value"]
    assert v["ok"] is True and v["n_cleared"] >= 4, v
    assert not v["warnings"], v["warnings"]
    by = {c["key"]: c for c in v["cleared"]}
    # ① 逐键 before 说的是真话（不是被过滤成 None）
    assert by["current.method"]["before"] == "logistic", by["current.method"]
    assert by["current.dataset_id"]["before"] == ds, by["current.dataset_id"]
    # ② readback 确认写后真的空了
    assert all(c["readback"] is None for c in v["cleared"]), v["cleared"]
    # ③ 读快照确认落地（回执不是自说自话）
    after_snap = dict(ml.software.gate.snapshot)
    assert all(after_snap.get(k) is None for k in by), after_snap
    # ④ 再清一次：诚实报 0（台面本来就空了）
    again = ml.tools.execute("clear_current_state", {})["value"]
    assert again["n_cleared"] == 0, again
    assert again["n_already_clean"] >= 4, again
    # ⑤ 史必须留着（append-only）：清零是"再写一批 None"，不是删史
    events = [e for e in ml.history.events() if "clear_current_state" in str(e)]
    assert events, "清零必须留痕"


def test_pick_projection_is_honest_about_missing_keys(app):
    """⭐ `pick` 投影：只回关注量，**缺失键如实列 `_missing`**（不静默回 null）。

    依据指导意见 §1-P1-6（"扫 N 个点撑爆上下文"）与 §4-4（"缺失键如实列，绝不静默
    null"）。

    ⚠ 本仓的**边界**（实测踩过）：回执里 `output_schema.required` 声明的必填键
    **删不掉**（工具面会以 `bad_tool_output` 拒），所以 `pick` 只装在
    **大负载不是必填键**的地方（`describe_training.series`）。本判据同时钉住：
    投影后必填骨架仍在、大负载被换成点名的深层值。
    """
    ml = app
    ds = _prepared(ml)
    if not _has_torch():
        print("    （跳过：未装 torch——只有神经网络会写训练事件）")
        return
    rid = ml.tools.execute("run_method", {
        "method": "torch_mlp", "dataset_id": ds,
        "overrides": {"epochs": 8}})["value"]["run_id"]

    full = ml.tools.execute("describe_training",
                            {"run_id": rid, "include_series": True})["value"]
    assert "series" in full, full
    picked = ml.tools.execute("describe_training", {
        "run_id": rid, "include_series": True,
        "pick": ["series.loss.values", "nope.nothere"]})["value"]
    # ① 大负载被投影（整条 series 不再回）
    assert "series" not in picked, picked.keys()
    assert len(picked["series.loss.values"]) == 8, picked["series.loss.values"]
    # ② 必填骨架仍在（工具面契约）
    for k in ("ok", "run_id", "available", "latest", "tags", "error"):
        assert k in picked, (k, sorted(picked))
    # ③ 缺失键如实列（不与"值是 null"混为一谈）
    assert picked["_missing"] == ["nope.nothere"], picked.get("_missing")


def test_run_queries_orchestrates_reads_in_one_round_trip(app):
    """⭐ 编排入口（指导意见 §4）：一次往返跑一串只读查询。

    为什么值得占一个工具位：没有它时"列方法 → 看参数 → 查数据集 → 看训练"要
    4+ 次往返。依据："只读能力超过 10 条、或扫描/多步组合是常见用法时"——
    我们 7 条只读查询 + **多步组合确实是常见用法**。

    钉四件（每件都对应指导意见的一条）：
    1. `output` **挂在步里**（不是顶层位置数组 ⇒ 不必再花回合对齐）；
    2. 成功步有**短摘要**，失败步是**教学原因**；
    3. **首个失败就停**（`stopped_at`）；
    4. 参数预检**可教学且不泄露内部符号**（`banned_words` 管不到错误消息，
       所以这条得单独钉）。
    """
    ml = app
    ds = _prepared(ml)
    ml.tools.execute("run_method", {"method": "logistic", "dataset_id": ds})

    got = ml.tools.execute("run_queries", {"actions": [
        {"op": "method_catalog", "args": {"family": "neural"}},
        {"op": "method_detail", "args": {"name": "logistic"}},
        {"op": "dataset_detail", "args": {"dataset_id": ds}},
    ]})
    assert got["is_error"] is False, got
    v = got["value"]
    assert v["ok"] is True and v["stopped_at"] == -1, v
    assert [s["op"] for s in v["steps"]] == ["method_catalog", "method_detail",
                                             "dataset_detail"], v["steps"]
    for s in v["steps"]:
        assert s["output"], s                      # ① output 挂在步里
        assert s["detail"], s                      # ② 成功步也给摘要
        assert s["ok"] is True

    # ③ 失败步：教学原因 + 就地停下（后面那步不再跑）
    bad = ml.tools.execute("run_queries", {"actions": [
        {"op": "method_detail", "args": {"name": "no_such_method"}},
        {"op": "method_catalog", "args": {}},
    ]})["value"]
    assert bad["ok"] is False and bad["stopped_at"] == 0, bad
    assert bad["n_steps"] == 1, bad
    assert "no_such_method" in bad["steps"][0]["detail"], bad["steps"][0]

    # ④ 参数预检：可教学（说清该给什么），且**不泄露内部符号**
    for label, acts, kind in (
            ("错参数名", [{"op": "method_detail", "args": {"method": "x"}}],
             "unknown_action_arg"),
            ("缺必给参数", [{"op": "dataset_detail", "args": {}}],
             "missing_action_arg"),
            ("未知 op", [{"op": "no_such_op"}], "unknown_action"),
    ):
        out = ml.tools.execute("run_queries", {"actions": acts})
        assert out["is_error"] is True, (label, out)
        info = (out.get("error") or {}).get("info") or {}
        assert info.get("kind") == kind, (label, info)
        assert info.get("hint"), label
        blob = json.dumps(out, ensure_ascii=False)
        for leak in ("MLEngine", "_query_", "Surface", "codeact"):
            assert leak not in blob, f"{label} 的错误泄露了内部符号 {leak!r}：{blob[:200]}"

    # 上界：不封顶的话一段动作能把整个能力面扫一遍
    too_many = ml.tools.execute("run_queries",
                                {"actions": [{"op": "summary"}] * 30})
    assert too_many["is_error"] is True
    assert ((too_many.get("error") or {}).get("info") or {}).get("kind") \
        == "too_many_actions", too_many


def test_submit_prechecks_write_authority_and_caps_concurrency(app):
    """⭐ 提交前的两道闸（指导意见 §3-1/§3-2）。

    这两条框架**只给机制、要项目自己补**：
    - **写权预检**：不预检的话，LOCKED 下会"提交成功 → 轮询 → 发现失败"
      （AI 白等一轮，还可能把失败读成实验结论）。预检让它**同步**被拒。
    - **并发上限**：local job 是"每次 submit 新起线程、不排队、无上限"
      ⇒ 不兜住的话连续提交 N 个训练会把机器打满。超额要**响亮拒绝**，
      且名额在任务**真跑完**后释放（放在 submit 之后就等于没限）。
    """
    ml = app
    _prepared(ml)
    # ① LOCKED：同步拒绝，且**不建 job**
    ml.lock()
    out = ml.tools.execute("submit_run", {"method": "logistic"})
    assert out["is_error"] is True, out
    info = (out.get("error") or {}).get("info") or {}
    assert info.get("kind") == "authority_locked", info

    # ② 并发上限：su 满即拒（用**慢**方法占住名额）
    ml.switch_ai()
    ml.tools.execute("prepare_dataset", {
        "source": {"kind": "synthetic", "task": "classification",
                   "n_samples": 600},
        "name": "cap", "target": "target"})
    cap = ml.submit_max_jobs if hasattr(ml, "submit_max_jobs") else None
    cap = cap or type(ml)._max_jobs if hasattr(type(ml), "_max_jobs") else 4
    cap = getattr(ml, "_max_jobs", 4)
    accepted, rejected = [], None
    for _ in range(cap + 2):
        got = ml.tools.execute("submit_run", {
            "method": "random_forest",
            "overrides": {"n_estimators": 120}})
        if got["is_error"]:
            rejected = (got.get("error") or {}).get("info") or {}
            break
        accepted.append(got["value"]["job_id"])
    assert len(accepted) == cap, (len(accepted), cap)
    assert rejected.get("kind") == "too_many_jobs", rejected
    # ③ 跑完后名额释放
    for job_id in accepted:
        ml.tools.execute("read_job", {"job_id": job_id, "wait_s": 120})
    after = ml.tools.execute("submit_run", {"method": "logistic"})
    assert after["is_error"] is False, after
    # ⚠ **最后一个 job 也要等完**：不等的话实例关闭时它还在跑，
    #   临时目录清理会因句柄未释放而 `WinError 145`（实测踩过 —— 那条错误
    #   还会掀翻直跑入口的汇总，让"哪条判据红"看不见）。
    ml.tools.execute("read_job", {"job_id": after["value"]["job_id"],
                                  "wait_s": 120})


def test_tool_descriptions_carry_no_implementation_words(app):
    """⭐ 词表**真的启用**了（指导意见 §5）：描述里不许出现实现词汇。

    ⚠ 钉的是"**半铺好的纪律**"这类缺陷：管道一直在（`define_tool(banned_words=…)`
    会在构造期校验、撞词就造不出工具），但装配点传的是**空表** ⇒ 纪律铺了却没启用，
    比没铺更坏（它看起来"已经有守卫了"）。实测于 2026-10-03：`submit_run` 的描述里
    还写着"轮询"——正是这词表把那一处抓出来的。

    本判据从两侧钉：① 装配期用的是非空词表；② 线上每一条描述都不含表里的词
    （即使有人绕过装配点加了工具，这里也会红）。
    """
    from ml_mecha.tools import BANNED_WORDS
    assert BANNED_WORDS, "词表是空的 ⇒ 这条纪律等于没启用"
    schemas = app.tools.schemas()
    assert schemas, "没有工具面"
    offenders = []
    for s in schemas:
        low = s["description"].lower()
        for w in BANNED_WORDS:
            if w.lower() in low:
                offenders.append((s["name"], w))
    assert not offenders, f"描述里出现实现词汇：{offenders}"


def test_blank_run_id_finds_the_latest_run_with_observations(app):
    """`describe_training` 留空 = **最近写过训练事件的那次**（描述说了就得做到）。

    ⚠ 钉的是一个"描述与行为不符"的真缺陷（2026-10-03 实测）：工具描述写着
    "留空则按最近落盘的 run 找"，而引擎把空 `run_id` **直接透传** ⇒ 一律
    `available=false`。AI 照着描述省略参数，然后会得到"这次训练没有观测数据"
    这个**错结论**（把"我调用方式不对"读成"数据不存在"）。
    """
    ml = app
    ds = _prepared(ml)
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    rid = ml.tools.execute("run_method", {
        "method": "torch_mlp", "dataset_id": ds,
        "overrides": {"epochs": 6}})["value"]["run_id"]
    got = ml.tools.execute("describe_training", {})["value"]
    assert got["available"] is True, got
    assert got["run_id"], got
    assert got["tags"], got
    # 挑的必须是**真有事件**的那次；本次刚跑的 run 至少应能被找到
    assert got["run_id"] == rid or got["run_id"].startswith("torch_mlp"), got


def test_ai_input_errors_are_teachable(app):
    """⭐ **AI 写错输入时必须学到"怎么改"**，而不是"框架坏了"（指导意见 P0-3）。

    ## 为什么这条是"使用者实测"的固化

    指导意见的原话：*"失败必须机器可判别，否则 AI 会读成物理结论"*——
    EL 的使用者 AI **差点把"参数崩了"当成"π 脉冲无效"的实验结论**。我用同一把尺子
    实测了 10 个"AI 参数写错"的场景，当时 **4 处不可教学**：
    `overrides={"epochs":"many"}` 报 `internal_error`、
    `{"edges":[["a","b"]]}` 报 `internal_error`（连内部类名
    `ml_toolbox.core.contracts.ParamValueError` 都漏出去）。

    ## 三层修复（判据同时钉住三层，缺一层就漏）

    1. `tools._raise_on_failure` 的 **最后一公里**：原先 `hint=` 写死成万能句、
       `suggest=` 读错键名（引擎写 `error_suggest`）⇒ 引擎的 hint 全被丢掉；
    2. `ParamSpec.clean` 抛**裸 `ValueError`** ⇒ 加 `ParamValueError`（带 kind/hint）；
    3. `runner.run_one` 把 `method.fit` 的异常**压成一句 traceback 字符串** ⇒
       让 `MLResult`/`RunSummary` 带着 kind/hint/suggest 穿过它。

    ## 判据

    每一条都要求：**kind 具体**（不是 `internal_error`/`command_failed`）、
    **hint 非空且不是万能句**、**消息里不出现内部模块名**。
    """
    ml = app
    ds = _prepared(ml)

    #: (场景, 工具, 参数, 期望 kind) —— 期望 kind 写死，防止"随便给个 kind 就算过"
    cases = [
        ("覆写值不是数字", "run_method",
         {"method": "logistic", "dataset_id": ds, "overrides": {"C": "很大"}},
         "bad_param_value"),
        ("epochs 给了字符串", "run_method",
         {"method": "torch_mlp", "dataset_id": ds, "overrides": {"epochs": "many"}},
         "bad_param_value"),
        ("参数低于下限", "run_method",
         {"method": "torch_mlp", "dataset_id": ds, "overrides": {"batch": -1}},
         "bad_param_value"),
        ("覆写的键不存在", "run_method",
         {"method": "logistic", "dataset_id": ds, "overrides": {"没这个参数": 1}},
         "unknown_key"),
        ("seed 越界", "run_method",
         {"method": "logistic", "dataset_id": ds, "seed": -5}, "bad_seed"),
        ("资源声明写错", "run_method",
         {"method": "logistic", "dataset_id": ds,
          "resource_guard": {"max_kernel_mb": "big"}}, "bad_resource_guard"),
        ("device 不认识", "run_method",
         {"method": "logistic", "dataset_id": ds, "device": "tpu"}, "bad_device"),
        ("边表不是数字", "prepare_dataset",
         {"source": {"kind": "synthetic", "task": "classification",
                     "n_samples": 30},
          "name": "g", "target": "target", "graph": {"edges": [["a", "b"]]}},
         "bad_graph_edge"),
        ("方法名不存在", "run_method",
         {"method": "no_such", "dataset_id": ds}, "unknown_key"),
        ("dataset_id 不存在", "run_method",
         {"method": "logistic", "dataset_id": "ds-999"}, "unknown_dataset"),
    ]
    #: 万能句（旧版所有失败都退化成它）——出现即视为"没教会任何东西"
    GENERIC = "命令未生效；检查参数、权限与数据引用后重试"

    problems = []
    for label, tool, args, want_kind in cases:
        got = ml.tools.execute(tool, args)
        if not got.get("is_error"):
            problems.append((label, "被接受了（本该失败）", ""))
            continue
        err = got.get("error") or {}
        info = err.get("info") or {}
        kind = str(info.get("kind") or "")
        hint = str(info.get("hint") or "")
        msg = str(err.get("message") or "")
        if kind != want_kind:
            problems.append((label, f"kind={kind!r} 期望 {want_kind!r}", hint))
        elif not hint or hint == GENERIC:
            problems.append((label, "hint 缺失或是万能句", hint))
        elif "ml_toolbox." in msg or "ml_mecha." in msg:
            # 内部模块名漏进**模型可见文本**：与 banned_words 同一条纪律
            # （那条只管 description，错误消息得单独钉）
            problems.append((label, f"消息漏出内部模块名：{msg[:70]}", hint))
    assert not problems, "\n".join(
        f"{a}: {b}（hint={c!r}）" for a, b, c in problems)


# ---------------------------------------------------------------- 门禁辅助
def test_dataset_query_exposes_chain_params_and_real_split_policy(app):
    """⭐ AI 能看见**预处理链每一步的参数**与**真实的划分口径**。

    两个缺口一次钉住（2026-10-03 实测）：
    1. `dataset_detail` 原先只给 `step` + `big_num` ⇒ AI 知道"有一步叫 scale"，
       但**不知道它按什么参数跑**（standard 还是 robust、编码是 onehot 还是
       ordinal），只能猜；指纹 `pipeline_id` 又不含这些可读信息。
    2. ⚠ **`split` 步骤的 `test_size` 是误导的**：`SplitStep.apply` 写的是**步骤
       参数**（默认 0.25），而真正决定划分的是 `Pipeline.test_size`
       （`pipeline.py::_split`）⇒ 两者可以不一致。⇒ 链上那个数**剔除**，
       真值由 `split_policy`（取自 Pipeline 本体）给。
    """
    dataset_id = _prepared(app)
    detail = app.engine.query("dataset_detail", dataset_id=dataset_id)

    chain = detail["pipeline_chain"]
    keys = [c["step"] for c in chain]
    for want in ("missing", "encode", "scale", "split"):
        assert want in keys, f"链上少了 {want}（实得 {keys}）"
    for c in chain:
        assert "params" in c and isinstance(c["params"], dict), c
        assert "title" in c and c["title"], f"{c['step']} 没有人类可读标题"
        assert "summary" in c and isinstance(c["summary"], dict), c

    # 参数与摘要要**说得出做法**（不是空壳）
    by_step = {c["step"]: c for c in chain}
    assert by_step["missing"]["summary"].get("strategy"), "缺失值策略没透出"
    assert by_step["scale"]["summary"].get("mode"), "缩放方式没透出"

    # ⚠ `split` 步骤那个误导性的 test_size 必须**不在**（它不决定划分）
    assert "test_size" not in by_step["split"]["summary"], \
        "步骤级 test_size 又漏出来了（它与 Pipeline.test_size 可以不一致）"

    # 真值在 split_policy，且取自 Pipeline 本体（与工程默认一致）
    policy = detail["split_policy"]
    assert set(policy) == {"test_size", "stratify", "seed", "time_split"}, policy

    # 工具面同样看得到（AI 用的是这一层）
    got = app.tools.execute("describe_dataset", {"dataset_id": dataset_id})["value"]
    assert got["split_policy"] == policy
    assert got["pipeline_chain"][0]["summary"], "工具面把链摘要丢了"


def test_per_epoch_progress_reaches_read_job(app):
    """⭐ 后台长训练**逐 epoch 报进度** —— AI 靠它判断"跑到哪了 / 该不该早停"。

    为什么必须钉（实测的假绿，值得记下来）：钩子最初挂在 `_execute_one` 里，
    而那**太晚** —— `Session.run` 一进来就调 `request.config()`，而
    ``RunRequest.config()`` 是 `extras=dict(self.extras)` 的**一次性拷贝**
    （`api.py:89-94`）⇒ 之后往 `request.extras` 里塞东西**传不到 cfg**。
    症状极具欺骗性：**同步** `run_method` 看着一切正常，只有**后台 job**
    一条进度都没有 —— 正好是"人看得见、AI 看不见"的那个缺口。
    ⇒ 判据必须走**真的 job**，不能只验 `_attach_progress` 有没有设上键。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    dataset_id = _prepared(app)
    try:
        job = app.submit({"command": "run_method", "method": "torch_mlp",
                          "dataset_id": dataset_id,
                          "overrides": {"hidden": "16,8", "epochs": 8}},
                         side="ai")
        assert job.id
        status = app.wait_job(job, timeout=180.0)
        assert status["state"] == "done", status
        raw = status.get("progress") or []
        assert raw, ("job 一条进度都没有 ⇒ 钩子没挂上（见本判据 docstring 的"
                     "那条'太晚'教训）")
        # 进度是**逐 epoch** 的（不是只有一条批次级），且分数单调不减、以 1.0 收尾
        assert len(raw) >= 8, f"进度点太少（{len(raw)}），像是批次级而非逐 epoch"
        fracs = [float(p[0]) for p in raw]
        assert fracs == sorted(fracs), f"进度分数不单调：{fracs}"
        assert fracs[-1] == 1.0, f"最后一次进度不是 1.0：{fracs[-1]}"
        assert all(isinstance(p[1], str) and p[1] for p in raw), "进度缺少阶段名"

        # 工具面同样看得到（AI 用的是 read_job）
        read = app.tools.execute("read_job", {"job_id": job.id})["value"]
        assert len(read["progress"]) == len(raw)
        assert read["progress_latest"]["frac"] == 1.0
    finally:
        _clean_nn_probe()


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
                    if e.target == "current.resource_guard"]
    assert guard_events, "resource_guard 没进 History"
    assert guard_events[-1].after == {"max_kernel_mb": 123.0}, guard_events[-1].after


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
    audit_events = [e for e in events if e.op == "command.run_method"]
    assert audit_events, "核心命令审计没进 History"
    audit = audit_events[-1]
    assert audit.actor == "ml-ai"
    assert audit.call_id == "call-abc123"
    assert audit.after["result_ref"]["run_ids"] == [receipt["run_id"]]

    # 域运行摘要（ml.run）保留 run 引用与标量指标，供 Monitor 复述
    run_events = [e for e in events if e.op == "ml.run"]
    assert run_events, "域运行摘要没进 History"
    run_event = run_events[-1]
    assert run_event.actor == "ml-ai"
    assert run_event.call_id == "call-abc123"
    assert run_event.after["run_id"] == receipt["run_id"]

    state_events = [e for e in events if e.target.startswith("current.")]
    assert state_events, "状态键没进 History"
    for e in state_events:
        assert e.target in STATE_KEYS, e.target
        assert e.actor in ("ml-gui", "ml-ai")

    # 准备阶段是人类侧发起的 → 归因必须是人类标签
    prepare_events = [e for e in events if e.target == "current.pipeline_id"]
    assert prepare_events and prepare_events[0].actor == "ml-gui"

    # History 只存引用与标量：事件体序列化后不得出现大数组/大对象
    for e in events:
        payload = json.dumps(e.after, ensure_ascii=False, default=str)
        assert len(payload) < 2000, f"事件 {e.target} 体过大，疑似内联大对象"
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
    assert any(e.after.get("run_id") == receipt["run_id"]
               for e in app.history.events()
               if isinstance(e.after, dict))


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
#: ⚠ **本表列的是 EL 的名字，不是我们的**：`reset_config` / `run_actions` 就在里面
#: ——2026-10-03 实测撞上：我按 mecha 指导意见给新能力起的这两个名字**恰好是 EL 的**
#: ⇒ 宪章 R1 禁止复制 ⇒ 改成 ML 自己的词汇（`clear_current_state` / `run_queries`）。
#: （一次批量改名脚本曾把**本表**也改掉，导致判据自己失效；这里保留原样。）
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
                                          "run_method_batch", "compare_methods",
                                          "run_custom", "clear_current_state",
                                          "eval_custom"}
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
    """⭐ 六条命令的**核心投影**都必须带**显式非空** `required`（否则语义翻转会开口子）。

    这是 ADR「迁移须知 1」对 ML 的直接检查项：ML 的命令声明走
    ``commands.py → core_commands.core_command_specs()`` 机械投影，投影**无条件**写出
    ``"required": [...]``。本判据逐条核对"核心 spec 里的 required == 宿主声明里的必填"，
    并**显式要求非空**——万一哪天有人把投影改成"空就不写"，这里立刻红。

    ⚠ `clear_current_state` 是**唯一的例外**：它的语义就是"整张台面清零"，本来就没有
    参数（给它加 `keys` 等于自己造一个"部分清空"的语义面）。⇒ 显式豁免它，
    而不是把"required 非空"这条断言整个删掉（那条断言对其它五条命令仍有价值）。
    """
    from ml_mecha.commands import COMMANDS
    #: 语义上无参数、故 required 恒为空的命令（**显式豁免，理由见 docstring**）
    #:
    #: ⚠ `run_custom` 在 2026-10-04 之后也在这里：它的真实约束是"**`code` 与 `code_path`
    #: 二选一**"（至少给一个），而**平面 schema 表达不了"至少一个"** ⇒ 两个都声明成
    #: 可选。少给会被入口用**可教学**的 `missing_code` 拒；两个同时给会被
    #: `bad_code_source` 拒（见 pitfalls A31：同时给时静默取一个会骗人）。
    NO_PARAM_COMMANDS = {"clear_current_state", "run_custom"}
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
        if host.name not in NO_PARAM_COMMANDS:
            assert core_required, (
                f"{host.name} 的 required 是空的 ⇒ 它现在「什么都不必填」；"
                "若这是有意的，请把它加进 NO_PARAM_COMMANDS 并写清理由；"
                "否则补齐声明")
        # 必填项必须都在 properties 里（否则模型永远填不出来）
        assert set(core_required) <= set(dict(params["properties"])), host.name
        checked.append(host.name)
    # R8 非退化：真的逐条比过（不是空转）
    assert sorted(checked) == ["clear_current_state", "compare_methods",
                               "eval_custom", "prepare_dataset", "run_custom",
                               "run_method", "run_method_batch"], checked


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

    # ⭐ **内省自动收集**（2026-10-03 改）：此前这里是一张**手写**的测试名列表，
    #   而门禁用**脚本**方式跑本文件 ⇒ 新判据忘了加进来就**永远不会被跑**。
    #   实测：20 条已定义的判据（含 GNN 端到端、后台落盘、训练观测面）从未进过门禁，
    #   其中一条当时就是红的。⇒ 不再手写清单：`globals()` 里所有 `test_*` 全收，
    #   按名字排序保证顺序可复现。**漂移从此不可能**（定义即入闸）。
    tests = [fn for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    assert tests, "没收集到任何判据（main 的内省收集坏了）"



    failures = []
    for fn in tests:
        if "app" in fn.__code__.co_varnames:
            # ⚠ **临时目录的创建/清理也要纳入捕获**：Windows 上若实例关闭时还有
            #   未结束的后台 job 握着文件句柄，`TemporaryDirectory.__exit__` 会抛
            #   `OSError: [WinError 145] 目录不是空的` ⇒ 原写法里它**在 try 之外**，
            #   会直接把整个直跑入口掀翻（没有 `✗`、没有汇总 —— "红看不清"）。
            #   ⇒ 整个 with 块包进 try，让"清理失败"也变成一条可见的判据失败，
            #   而且**不假装通过**（它是一条真实的问题：有东西没关干净）。
            with tempfile.TemporaryDirectory(
                    prefix="ml-mecha-test-", ignore_cleanup_errors=True) as td:
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
                    try:
                        instance.close()
                    except BaseException as exc:  # noqa: BLE001
                        failures.append((fn.__name__, exc))
        else:
            try:
                fn()
            except (KeyboardInterrupt, SystemExit):
                raise
            # ⚠ 同上面的理由，**别改回 `Exception`**：`pytest.raises` 失败抛的 `Failed`
            # 继承 `BaseException` ⇒ 改窄了会让这个入口被掀翻成 traceback（红看不见）。
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
