# -*- coding: utf-8 -*-
"""模型可见工具投影：从 ML 命令面/查询面生成工具（宪章 §11.2 第 1—4 阶段）。

纪律（宪章 R4 + mecha ``tools.py`` 命名律）：

- 工具名是**动词_宾语** snake_case，且不带项目前缀（``mecha_`` 收编）；
  ``get_*`` / ``list_*`` 这类"视图名当工具"也收编——只读枚举并进
  ``describe_*`` 或走 ``output_schema`` 投影；
- 每个工具的 ``output_schema["required"]`` **显式声明**回执必填键；
- 错误经 ``MechaError → ToolFailure {message, info}`` 归一化（``ToolRegistry``
  的 execute 负责，不在工具体里手抄错误字典）；
- 大结果一律走**引用**：``run_id`` / ``dataset_id``，绝不把 ``y_pred``、
  ``artifacts``、完整配置内联进工具输出。

工具只是"投影"：所有写路径仍然经 :class:`ml_mecha.engine.MLEngine` 的受控
命令 + mecha ``Gate``；这里不新增第二条写通道。

## 长任务：job 类工具（P1）

``submit_run`` / ``read_job`` / ``cancel_run`` 是**调度面**，不是新的写通道：
``submit_run`` 把 ``run_method`` 提交成 mecha job（走 ``MLMecha.submit``，
由它把操作者通道与 ``call_id`` 带进后台线程），真正的域写入仍由那条受控命令
经 ``Gate`` 完成，审计落 ``command.run_method`` + ``ml.run``。因此：

- 这三个工具**不套 ``_with_channel``**（它们的写路径不在本进程的调用栈上，
  而在 job 线程里）。这同时是硬约束：``capabilities.declared_side_effect`` 用
  "闭包是否持有 ``Channel``"与"有无同名写命令"两条证据交叉判定，而
  ``submit_run`` **没有同名命令**；若这里给它绑一个 ``Channel``，两条证据就
  不一致，能力清单会当场 ``CapabilityDrift``（fail loud，方向正确）。
- 代价是能力清单把它们记为 ``side_effect=False``（**调用当刻**不写域状态），
  而"提交即产生一次运行"这个事实写在各自的 description 与
  内部记录（未随仓发布，存档在仓外） 的专门小节里，不靠那个 bool 表达。
- ``host`` 是 job 服务 seam（``MLMecha``）；省略时（如 ``capabilities.py``
  只为取名而构造工具、不执行）job 工具仍能构造，调用时才结构化报错。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mecha.errors import MechaError
from mecha.gate import Channel
from mecha.providers.codeact import Action
from mecha.surface import ExecutionContext
from mecha.tools import ToolDefinition, define_tool

from .engine import MLEngine, clear_channel, use_channel

#: 本地宿主的默认通道标签（细标签归因用；side 才定写权）。
LOCAL_ACTOR = "ml-local"

#: ``read_job`` 单次最多阻塞多久（秒）。阻塞是**可选便利**，不是"等到跑完"的
#: 承诺：MCP 侧单次调用有超时（dsh 默认 60 s），模型应轮询。
READ_JOB_MAX_WAIT_S = 30.0

#: job 工具在宿主未接 job 服务时的结构化失败（不是静默假成功）。
_NO_JOB_HOST_HINT = ("job 工具需要一个接好 Job 服务的宿主（MLMecha）；"
                     "capabilities.py 只为取名而构造工具，不执行它们")


def _raise_on_failure(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    """把引擎的 ``ok=False`` 回执转成 ``ToolRegistry`` 认识的结构化失败。

    引擎回执是领域层返回值（``{"ok": False, "error_kind": ...}``）；工具层若不
    转异常，``ToolRegistry.execute`` 会误判为 ``is_error=False``——AI 看到的
    是"成功但结果失败"，而不是可教学的 ``ToolFailure {message, info}``。

    ## ⚠ 这里是"可教学"的**最后一公里**（2026-10-03 修，实测抓到的静默丢失）

    旧版有两处把它掐断：
    - ``hint=`` 是**硬编码**的万能句（"命令未生效；检查参数…"）⇒ 引擎精心写的
      ``exc.hint``（如"seed 允许 0..2147483647"）**在到达 AI 之前被丢掉**；
    - ``suggest=`` 读的键名是 ``suggest``，而引擎写的是 **``error_suggest``**
      ⇒ **永远是空串**。

    症状极具欺骗性：``kind`` 是对的（所以"机器可判别"那一条看着没问题），
    但 AI 拿不到"**该怎么改**"。而指导意见 P0-3 的原话是"AI 读 ``info.kind``
    决定怎么改"——**光有 kind 不够**。⇒ 现在原样透传，只在**确实没有**时才退回万能句。
    """
    if receipt.get("ok"):
        return receipt
    warnings = [str(w) for w in receipt.get("warnings", []) if w]
    message = str(receipt.get("error") or "")
    if not message and warnings:
        message = "；".join(warnings)
    hint = str(receipt.get("error_hint") or "")
    suggest = str(receipt.get("error_suggest") or "")
    if not hint:
        # ⚠ 只有"引擎也没给"时才用万能句：它是**兜底**，不是**默认**。
        #   （旧版把它当默认 ⇒ 所有失败看起来都是同一句废话。）
        hint = "命令未生效；检查参数、权限与数据引用后重试"
    raise MechaError(
        message or "ML 命令失败",
        kind=str(receipt.get("error_kind") or "command_failed"),
        hint=hint,
        suggest=suggest,
    )


def _raise_on_total_failure(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    """批量/对比类回执：**全部**失败才转异常；有任一成功则原样返回。

    为什么不能对批量直接用 :func:`_raise_on_failure`：批量与对比的语义是
    "**逐方法隔离失败**"，部分成功是**正常结果**，不是工具失败。若一律转异常，
    已经成功跑出来的那些 ``run_id`` 会连同失败项一起被丢掉——工具自己的
    description 承诺了"返回每个方法的运行编号与指标，失败项只给编号与原因"，
    那样就与实现自相矛盾（模型拿不到任何一个 run_id，还得重跑）。

    "全部失败"= 什么都没达成，仍按写工具纪律 fail loud（不制造假成功）；
    而"部分失败"用 ``ok=False`` + 逐项明细如实表达，调用方读得到。
    """
    if receipt.get("ok"):
        return receipt
    items = receipt.get("runs") or receipt.get("table") or []
    if any(isinstance(item, Mapping) and item.get("ok") for item in items):
        return receipt
    return _raise_on_failure(receipt)


def _local_context() -> ExecutionContext:
    """工具路径的执行上下文。

    工具不由 ``Surface.run`` 调用，所以这里显式造一个上下文（而不是传 None）：
    命令体拿到的 contract 与框架路径一致（取消/进度是显式字段，消 D-16）。
    本地同步工具暂不暴露取消，但契约形状先对齐。
    """
    return ExecutionContext()


#: ⭐ **描述里不许出现的实现词汇**（指导意见 §5：模型可见文本只含任务概念）。
#:
#: ⚠ 为什么必须是**真词表**：这条纪律的管道早铺好了（`define_tool(banned_words=...)`
#: 会在**构造期**校验 description，撞词就 `bad_description` 造不出工具），但一直
#: 传的是**空表** ⇒ **纪律铺了却没启用**——比没铺更坏，因为它看起来"已经有守卫了"。
#: （实测于 2026-10-03：`register_ml_tools` 调用处没传 `banned_words`。）
#:
#: 词表原则：**框架/传输/内部结构的词**不该出现在模型读的文本里（那是设计者的词汇）；
#: 而"运行/指标/数据集/方法/存档"是**任务概念**，可以用。加词时先问：
#: "AI 看到这个词，会不会去猜一个它不该关心的东西？"
BANNED_WORDS: tuple[str, ...] = (
    "gate", "authority", "history", "surface", "monitor",       # 框架机制
    "channel", "contextvar", "snapshot", "快照", "写权",          # 内部状态
    "registry", "provider", "adapter",                          # 结构词汇
    "contextvar", "daemon", "线程", "轮询", "semaphore",          # 执行细节
    "contextvars", "jobregistry", "toolregistry", "engine_kwargs",
)


def _pick(obj: Any, paths: Any) -> dict:
    """按**键路径**投影，只回关注量（指导意见 §1-P1-6：扫 N 个点别撑爆上下文）。

    ## 三条纪律（照指导意见）

    1. **整键优先**：JSON 的键本身可以含点 ⇒ 先按整键取；取不到再按 `.` 拆路径。
    2. **缺失不静默**：取不到的键如实列进 `_missing`，**绝不回 null 冒充"值是空"**
       —— "没有这个键"与"这个键的值是 null"是两件不同的事。
    3. **空投影 = 不投影**：`paths` 为空/None ⇒ 原样返回（调用方没要求投影）。

    @param paths 键路径列表，如 ``["run_id", "metrics.f1", "series.loss.values"]``
    """
    if not paths:
        return dict(obj) if isinstance(obj, Mapping) else {"value": obj}
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    got: dict = {}
    missing: list = []
    for raw in paths:
        key = str(raw)
        if isinstance(obj, Mapping) and key in obj:
            got[key] = obj[key]                      # ① 整键优先
            continue
        cur: Any = obj
        ok = True
        for part in key.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            elif isinstance(cur, (list, tuple)) and part.isdigit() \
                    and int(part) < len(cur):
                cur = cur[int(part)]
            else:
                ok = False
                break
        if ok:
            got[key] = cur
        else:
            missing.append(key)
    if missing:
        got["_missing"] = missing                    # ② 缺失如实列（不静默 null）
    return got


#: 读工具共用的 `pick` 参数声明（**一处定义**，免得各写一份描述而走样）。
PICK_PARAM: dict = {
    "type": "array",
    "description": "只回这些键路径（可选），例如 ['run_id','metrics.f1']；支持点号下钻。"
                   "取不到的键会如实列进 `_missing`（不会静默回 null）。"
                   "**扫多个结果/对比时用它**，免得把整份回执灌进上下文。",
}


def build_ml_tools(engine: MLEngine, *, actor: str = LOCAL_ACTOR,
                   side: str = "human",
                   banned_words: tuple[str, ...] = (),
                   host: Any = None,
                   codeact: Any = None) -> list[ToolDefinition]:
    """构造 ML 工具面（顺序稳定；宿主注册进自己的 ToolRegistry）。

    工具按**内容**分类，不按传输形态分类：本地 Python 宿主、dsh/MCP 宿主
    都用同一批 :class:`ToolDefinition`（mecha ``tools.py`` 的白名单投影）。

    ``host`` 只给 job 类工具用（它们要 ``MLMecha.submit`` / Job 注册表）；
    省略时那三个工具照样构造得出来（能力清单只取名不执行），调用时才报
    ``no_job_host``。
    """
    channel = Channel(actor, side)

    def _with_channel(fn):
        """把操作者通道绑到执行上下文，再调受控命令（fail closed 的地基）。"""
        def wrapped(**kwargs):
            token = use_channel(channel)
            try:
                return fn(**kwargs)
            finally:
                clear_channel(token)
        return wrapped

    return [
        _describe_methods(engine, banned_words),
        _describe_method(engine, banned_words),
        _describe_dataset(engine, banned_words),
        _describe_run(engine, banned_words),
        _describe_training(engine, banned_words),
        _describe_networks(engine, banned_words),
        _describe_network(engine, banned_words),
        _prepare_dataset(engine, _with_channel, banned_words),
        _run_method(engine, _with_channel, banned_words),
        _run_method_batch(engine, _with_channel, banned_words),
        _compare_methods(engine, _with_channel, banned_words),
        _run_custom(engine, _with_channel, banned_words),
        _submit_run(host, side, banned_words),
        _read_job(host, banned_words),
        _cancel_run(host, banned_words),
        _clear_current_state(engine, _with_channel, banned_words),
        _run_queries(engine, codeact, banned_words),
        _checkpoint_detail(engine, banned_words),
        _eval_custom(engine, _with_channel, banned_words),
    ]


def _checkpoint_detail(engine: MLEngine,
                       banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    """查某个 run 的 **checkpoint 摘要**（在不在、几个张量、含不含 bias、哪块设备）。

    为什么单开一条：这是"**训练完能把模型拿回来**"的入口。`describe_run` 只给产物
    **类别名**，而权重是几十万个数 ⇒ 回执只给**定位符与结构摘要**；真要用它走
    `eval_custom`（系统装载后交给你的代码）。
    ⚠ 它**读磁盘** ⇒ 跨重启照样有效（`describe_run` 读会话内存，重启后就查不到）。
    """

    def execute(*, run_id: str = "") -> Mapping[str, Any]:
        got = engine.query("checkpoint_detail", run_id=str(run_id))
        return {"ok": True, **got, "error": ""}

    return define_tool(
        name="checkpoint_detail",
        description="查某次自定义网络运行的**模型存档**：在不在、有哪些文件、"
                    "多少张量/参数、**含不含偏置**、在哪块设备上训的、最佳轮次是第几轮。"
                    "⚠ 权重**不内联**（几十万个数不进对话）：要真用模型，"
                    "调 `eval_custom` 把它交给你的代码（采样/算能量/换尺寸评估）。",
        parameters={"run_id": {"type": "string", "description": "运行编号",
                               "default": ""}},
        output_schema={"type": "object",
                       "required": ["ok", "run_id", "available", "files", "tags",
                                    "params", "meta", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


def _eval_custom(engine: MLEngine, wrap,
                 banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    """**把训练好的模型交给调用方的代码**（NQS 需求的那个"能调外部函数的口子"）。"""

    def execute(*, run_id: str, code: str, which: str = "best",
                dataset_id: str = "",
                payload: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        spec: dict[str, Any] = {"command": "eval_custom", "run_id": str(run_id),
                                "code": str(code), "which": str(which or "best")}
        if dataset_id:
            spec["dataset_id"] = str(dataset_id)
        if payload:
            spec["payload"] = dict(payload)
        receipt = _raise_on_failure(engine.run(spec, context=_local_context()))
        return {"ok": bool(receipt["ok"]), "run_id": receipt.get("run_id", ""),
                "which": receipt.get("which", ""),
                "metrics": dict(receipt.get("metrics") or {}),
                "extra_keys": list(receipt.get("extra_keys") or []),
                # 这次评估实际拿到几份负载 —— 反馈 §2-缺口2 的教训：让"没给数据"
                # **看得见**，而不是等代码里 KeyError 才发现
                "n_payload_arrays": int(receipt.get("n_payload_arrays") or 0),
                "payload_keys": list(receipt.get("payload_keys") or []),
                "note": receipt.get("note", ""),
                "warnings": list(receipt.get("warnings") or []),
                "error": receipt.get("error", "")}

    return define_tool(
        name="eval_custom",
        description="**装载某次训练的模型，交给你的代码去评估**（采样、算能量、"
                    "换一个没见过的尺寸外推…）并回它上报的指标。"
                    "代码只需定义 `def evaluate(model, ctx) -> dict`；`ctx` 里有 "
                    "`payload`（逐行负载，**可以现场给新数据**）、`meta`"
                    "（n_in/n_out/训练设备/最佳轮次）、`torch`/`nn`/`device`。"
                    "⚠ 它在**权威进程**里执行你的代码——与 run_custom 同一口径"
                    "（受控命令，各自留痕；沙箱负责别把宿主搞崩，不是权限边界）。",
        parameters={
            "run_id": {"type": "string", "description": "要装载哪一次运行"},
            "code": {"type": "string",
                     "description": "评估代码：定义 `def evaluate(model, ctx) -> dict`"},
            "which": {"type": "string",
                      "description": "装哪个权重：best（默认，验证损失最好的那轮）/ last",
                      "default": "best"},
            "dataset_id": {"type": "string",
                           "description": "复用会话里某份数据集的负载", "default": ""},
            "payload": {"type": "object",
                        "description": "**现场给新负载**（如换一个没见过的 N）：与 "
                                       "prepare_dataset 的 payload 同形；"
                                       "这里**不校验行数**（负载不绑数据集）"},
        },
        output_schema={"type": "object",
                       "required": ["ok", "run_id", "which", "metrics",
                                    "extra_keys", "n_payload_arrays",
                                    "payload_keys", "note", "warnings", "error"]},
        execute=wrap(execute),
        banned_words=banned_words,
    )


#: 一段编排最多多少步。**项目要自己想上限**（指导意见 §3：框架给机制，
#: "上限"是项目策略）：不封顶的话一段动作可以把整个能力面扫一遍，
#: 上下文与 CPU 都会被打满，而模型自己毫无察觉。
MAX_ACTIONS = 25

#: CodeAct 的**内置动作**（框架侧，与只读查询并列）。照抄框架 `_dispatch`
#: 的实现，附上各自的参数形状——这样预检能覆盖全部 op，而不是只覆盖查询。
_BUILTIN_ACTIONS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "run": (("spec",), ("spec",)),          # 走 Surface.run（受控命令面）
    "estimate": (("spec",), ()),
    "summary": ((), ()),
    "schema": ((), ()),
}


def _action_ops(engine: MLEngine) -> set:
    """能力面里可用的 op（内置动作 + 已注册的只读查询）。"""
    from .engine import READ_QUERY_NAMES
    return set(_BUILTIN_ACTIONS) | set(READ_QUERY_NAMES)


def _action_arg_spec(engine: MLEngine,
                     op: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """某个 op 的 ``(可给参数, 必给参数)``；不在能力面 ⇒ ``None``。

    ⚠ 查询那部分**从实现签名派生**（`engine.query_arg_spec`），不手写表
    —— 手写的必与实现漂移，而漂移会以"预检放行、运行期 TypeError"的形式出现。
    """
    if op in _BUILTIN_ACTIONS:
        return _BUILTIN_ACTIONS[op]
    return engine.query_arg_spec(op)


def _run_queries(engine: MLEngine, codeact: Any,
                 banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    """⭐ **一次往返里跑一串只读查询**（指导意见 §4 的模型入口）。

    ## 为什么值得占一个工具位

    没有它时，"列方法 → 看几个方法的参数 → 查数据集 → 查最近两次训练"这种
    **常见组合**要 6+ 次往返。编排省下的回合是实打实的（每次往返都带着
    一大坨上下文）。判据（指导意见）："只读能力超过 10 条、或扫描/多步组合
    是常见用法时"——我们 7 条只读查询 + **扫描/多步组合确实是常见用法**。

    ## 回执形状（照指导思想，**逐项都有**）

    ```json
    {"ok": true, "steps": [
      {"index": 0, "op": "method_catalog", "ok": true, "detail": "…短摘要…",
       "output": {...}},
      {"index": 1, "op": "method_detail", "ok": false,
       "detail": "[unknown_method] …教学原因…", "output": null}],
     "stopped_at": 1, "n_steps": 2, "error": ""}
    ```

    - **output 直接挂在步里**（不是顶层的**位置数组**——那会逼调用方自己对齐，
      "省下的回合又花回去"）；
    - 成功步 `detail` 给**短摘要**，失败步 `detail` 是**教学原因**；
    - ⚠ **首个失败就停**（`stopped_at` 指出是哪步）：后面的步骤可能依赖前一步的
      结果，硬跑下去只会得到一串"因为前面失败而失败"的噪声。

    ## 能力边界（如实说，不假装）

    够得到的只有**只读查询**（外加 4 个内置动作 `summary`/`schema`/`estimate`/`run`）。
    **写操作不在编排里**：每条写仍要单独经受控命令面（各自过闸、各自留痕）。
    ⇒ 要用它来"多步写"是不行的，也不该行。
    """

    def execute(*, actions: Any = None, pick: Any = None) -> Mapping[str, Any]:
        if codeact is None:
            raise MechaError(
                "本实例没有接上编排器",
                kind="no_codeact",
                hint="装配点未注入 CodeAct；逐条调用只读查询工具同样有效",
                suggest="改用 describe_methods / describe_run 等单个查询",
            )
        if not isinstance(actions, (list, tuple)) or not actions:
            raise MechaError(
                "actions 需要一个非空数组",
                kind="bad_actions",
                hint='每项形如 {"op": "method_catalog", "args": {}}',
                suggest='[{"op": "method_catalog"}, {"op": "dataset_detail"}]',
            )
        if len(actions) > MAX_ACTIONS:
            raise MechaError(
                f"一段编排最多 {MAX_ACTIONS} 步，收到 {len(actions)} 步",
                kind="too_many_actions",
                hint="拆成几段；每段都返回后你可以据此决定下一段",
            )
        parsed = []
        for i, raw in enumerate(actions):
            if not isinstance(raw, Mapping) or not str(raw.get("op") or "").strip():
                raise MechaError(
                    f"第 {i + 1} 步缺 `op`",
                    kind="bad_action",
                    hint='每项形如 {"op": "method_catalog", "args": {"family": "neural"}}',
                )
            args = raw.get("args") or {}
            if not isinstance(args, Mapping):
                raise MechaError(f"第 {i + 1} 步的 args 需要对象", kind="bad_action")
            parsed.append(Action(str(raw["op"]), dict(args)))
        # ⭐ **跑之前先把每一步的参数查清**（同步拒绝 > 半路炸）。
        #   为什么必须预检（实测踩过）：不查就交给 Python，错参数会抛
        #   `TypeError: ..._query_method_detail() got an unexpected keyword argument`
        #   —— 那条消息**泄露内部符号**、也不告诉 AI 正确的参数名是什么
        #   （`banned_words` 管不到错误消息，它只管工具描述）。
        #   ⚠ 一次全查、有错就**一个都不跑**：避免"跑了前几步才发现第 5 步写错"。
        allowed_ops = _action_ops(engine)
        for i, act in enumerate(parsed):
            spec = _action_arg_spec(engine, act.op)
            if spec is None:
                raise MechaError(
                    f"第 {i + 1} 步的 op {act.op!r} 不在能力面",
                    kind="unknown_action",
                    hint=f"可用的 op：{sorted(allowed_ops)}",
                )
            allowed, required = spec
            unknown = sorted(set(act.args) - set(allowed))
            if unknown:
                raise MechaError(
                    f"第 {i + 1} 步（{act.op}）不认识参数 {unknown}",
                    kind="unknown_action_arg",
                    hint=f"{act.op!r} 可给：{list(allowed)}；必给：{list(required)}",
                )
            missing = sorted(set(required) - set(act.args))
            if missing:
                raise MechaError(
                    f"第 {i + 1} 步（{act.op}）缺必给参数 {missing}",
                    kind="missing_action_arg",
                    hint=f"{act.op!r} 必给：{list(required)}；可给：{list(allowed)}",
                )
        result = codeact.run(parsed)
        steps = []
        for log in result.steps:
            out = (result.outputs[log.index]
                   if log.ok and log.index < len(result.outputs) else None)
            step: dict = {"index": int(log.index), "op": str(log.op),
                          "ok": bool(log.ok), "detail": str(log.detail or ""),
                          "output": None}
            if log.ok:
                body = out if isinstance(out, Mapping) else {"value": out}
                step["output"] = _pick(body, pick) if pick else dict(body)
                keys = [k for k in step["output"] if not k.startswith("_")]
                step["detail"] = step["detail"] or \
                    f"{len(keys)} 个字段：" + ", ".join(sorted(keys)[:6])
            steps.append(step)
        return {"ok": result.stopped_at < 0, "steps": steps,
                "stopped_at": int(result.stopped_at),
                "n_steps": len(steps), "error": ""}

    return define_tool(
        name="run_queries",
        description="**一次往返跑一串只读查询**（列方法→看参数→查数据集→看训练…），"
                    "比一条条调省很多来回。给 `actions`，每步形如 "
                    '{"op": "method_catalog", "args": {}}；可用的 op 就是那些只读查询'
                    "的名字（见各 describe_* 工具）。回执每步都带自己的结果与短摘要；"
                    "**某步失败就停在那里**（`stopped_at` 指出是哪步），"
                    "因为后面的步骤常依赖前一步的结果。"
                    "⚠ 它**只读**：写操作（准备数据、跑方法、改配置）仍要单独调用，"
                    "各自生效、各自留痕。",
        parameters={
            "actions": {"type": "array",
                        "description": "动作序列；每项 {'op': 查询名, 'args': {...}}"},
            "pick": PICK_PARAM,
        },
        output_schema={"type": "object",
                       "required": ["ok", "steps", "stopped_at", "n_steps", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


def _clear_current_state(engine: MLEngine, wrap,
                  banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    """⭐ 把配置台面清零（指导意见 §1-P1-5）。

    为什么 AI 需要一个"打扫"入口：配置是**跨会话残留**的。上一轮留下的
    "最近用的方法/数据集/覆写"会让这一轮的"我用默认配置跑一下"变成假话，
    而**没有入口能回到干净状态**时，AI 只能在一个不知底细的台面上干活。

    ⚠ 回执**逐键给 before/readback**（写前值 / 写后读回）：调用方不必再问一次
    "清干净了没有"。若某键读回不为空 ⇒ 回执里如实 `warnings`（半清状态比不清更坏）。
    """

    def execute() -> Mapping[str, Any]:
        receipt = _raise_on_failure(engine.run(
            {"command": "clear_current_state"}, context=_local_context()))
        return {"ok": bool(receipt["ok"]),
                "n_cleared": int(receipt.get("n_cleared", 0)),
                "n_already_clean": int(receipt.get("n_already_clean", 0)),
                "cleared": list(receipt.get("cleared") or []),
                "already_clean": list(receipt.get("already_clean") or []),
                "readback": dict(receipt.get("readback") or {}),
                "warnings": list(receipt.get("warnings") or []),
                "error": receipt.get("error", "")}

    return define_tool(
        name="clear_current_state",
        description="把当前配置清零、回到默认状态（数据集引用、方法、参数覆写、"
                    "随机种子、设备等都清空）。**要在干净台面上开始时先调它**；"
                    "回执逐项给出清空前的值与清空后读回的值，所以不用再查一次。",
        parameters={},
        output_schema={"type": "object",
                       "required": ["ok", "n_cleared", "n_already_clean",
                                    "cleared", "already_clean", "readback",
                                    "warnings", "error"]},
        execute=wrap(execute),
        banned_words=banned_words,
    )


# ---------------------------------------------------------------- 只读投影
def _describe_methods(engine: MLEngine,
                      banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, family: str = "", dataset_id: str = "") -> Mapping[str, Any]:
        result = engine.query("method_catalog", family=family, dataset_id=dataset_id)
        rows = [{"name": m["name"], "family": m["family"], "task": m["task"],
                 "target_kind": m["target_kind"], "purposes": list(m["purposes"]),
                 "supports_predict": m["supports_predict"]}
                for m in result["methods"]]
        return {"ok": True, "method_count": len(rows), "methods": rows,
                "error": ""}

    return define_tool(
        name="describe_methods",
        description="列出当前可用的机器学习方法及其族、任务类型与用途；"
                    "可选按族筛选，或按已准备的数据集筛选出真正能跑的方法。",
        parameters={
            "family": {"type": "string", "description": "按方法族筛选；留空表示不过滤",
                       "default": ""},
            "dataset_id": {"type": "string",
                           "description": "已准备数据集引用；给了就只列能处理它的方法",
                           "default": ""},
        },
        output_schema={"type": "object",
                       "required": ["ok", "method_count", "methods", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


def _describe_method(engine: MLEngine,
                     banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, name: str) -> Mapping[str, Any]:
        detail = engine.query("method_detail", name=name)
        return {"ok": True, "name": detail["name"], "family": detail["family"],
                "task": detail["task"], "target_kind": detail["target_kind"],
                "purposes": list(detail["purposes"]),
                "purposes_cn": detail["purposes_cn"],
                "supports_predict": detail["supports_predict"],
                "param_schema": detail["param_schema"], "error": ""}

    return define_tool(
        name="describe_method",
        description="说明一个机器学习方法的可调参数、默认值、取值范围与建模用途，"
                    "用于在运行前确认参数名与合法取值。",
        parameters={"name": {"type": "string", "description": "方法名"}},
        output_schema={"type": "object",
                       "required": ["ok", "name", "task", "param_schema", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


def _describe_dataset(engine: MLEngine,
                      banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, dataset_id: str = "") -> Mapping[str, Any]:
        dataset_id = dataset_id or (engine.dataset_ids() or [""])[-1]
        detail = engine.query("dataset_detail", dataset_id=dataset_id)
        return {"ok": True, "dataset_id": detail["dataset_id"],
                "name": detail["name"], "target": detail["target"] or "",
                "n_rows": detail["n_rows"], "n_features": detail["n_features"],
                "target_kind": detail["target_kind"] or "",
                "pipeline_id": detail["pipeline_id"],
                "columns": list(detail["columns"]),
                "pipeline_chain": detail["pipeline_chain"],
                # ⭐ 实际的训练/测试划分口径（test_size/stratify/seed/time_split）。
                #   与链上 `split` 步骤分开给：那个步骤的参数**不决定**划分
                #   （真源是 Pipeline 本体），见 `engine._chain_summary` 的说明。
                "split_policy": detail.get("split_policy") or {},
                "schema_mismatch": detail["schema_mismatch"], "error": ""}

    return define_tool(
        name="describe_dataset",
        description="说明一份已准备数据视图的行列规模、目标列、**预处理链逐步参数**"
                    "（填充策略/编码模式/缩放方式…）、**实际划分口径**（split_policy）"
                    "与管道指纹；用于判断数据是否适合某个方法。",
        parameters={"dataset_id": {"type": "string",
                                   "description": "数据集引用；留空用最近准备的一份",
                                   "default": ""}},
        output_schema={"type": "object",
                       "required": ["ok", "dataset_id", "n_rows", "n_features",
                                    "pipeline_id", "pipeline_chain",
                                    "split_policy", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


def _describe_run(engine: MLEngine,
                  banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    """⚠ **故意不给 `pick`**：这个回执的大头（`metrics` / `artifact_keys`）都是
    **声明过的必填键** ⇒ 工具面会校验它们在不在，`pick` 删不掉它们，
    那就不该提供"看起来能省其实省不了"的旋钮（省 token 的正解是
    `compare_methods` 的对比表，或 `describe_training` 的 `pick`）。
    """

    def execute(*, run_id: str = "") -> Mapping[str, Any]:
        if not run_id:
            records = engine.session.export_state()["records"]
            run_id = records[-1]["run_id"] if records else ""
        detail = engine.query("run_detail", run_id=run_id)
        return {"ok": True, "run_id": detail["run_id"], "method": detail["method"],
                "task": detail["task"], "run_ok": detail["ok"],
                "primary_metric": detail["primary_metric"],
                "metrics": detail["metrics"], "params": detail["params"],
                "dataset": detail["dataset"], "pipeline_id": detail["pipeline_id"],
                "artifact_keys": detail["artifact_keys"],
                "error": detail["error"]}

    return define_tool(
        name="describe_run",
        description="按运行编号回看一次机器学习运行的方法、指标、生效参数与产物类别；"
                    "大规模数值产物只给类别名与运行编号，不内联。"
                    "**要比多个 run 时用 compare_methods**（它直接给对比表，"
                    "不必逐个拉再自己对齐）。",
        parameters={"run_id": {"type": "string",
                               "description": "运行编号；留空用最近一次运行",
                               "default": ""}},
        output_schema={"type": "object",
                       "required": ["ok", "run_id", "method", "primary_metric",
                                    "metrics", "artifact_keys", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


def _describe_training(engine: MLEngine,
                       banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    """⭐ 训练过程观测：**逐 epoch 的数值**（只读投影，不经写权门）。

    为什么必须单开一个工具（2026-10-03 实测的缺口）：训练观测数据一直是
    **逐 epoch 落盘**的（`runs/<id>/tb/`），但那批数值只出现在**只读 HTTP 路由**
    `/latest`、`/series` 上（`ml_mecha/monitor_http.py`）⇒ AI 经工具面**一条都拿不到**，
    而 `describe_run` 只回"产物类别名"（`nn_history` 在列表里、数值不给）。
    于是 AI 无法回答"梯度爆没爆 / 哪层死了 / 要不要早停"——那正是观测数据的用途。

    ⚠ 与 `describe_run` 的**分工**（别合并）：
    - `describe_run` = 这次运行的**元数据与最终指标**（会话内存态）；
    - 本工具 = **过程数值**（磁盘事件文件），因此对"未落盘的运行"（`persist=false`，
      AI 默认）与"跨重启的历史运行"同样有效——那两种情形 `describe_run` 都做不到。
    """

    def execute(*, run_id: str = "", include_series: bool = False,
                limit: int = 200, pick: Any = None) -> Mapping[str, Any]:
        detail = engine.query("training_detail", run_id=run_id,
                              include_series=bool(include_series),
                              limit=int(limit))
        out = {"ok": True, **detail, "error": ""}
        # ⚠ `pick` 的**边界**（实测踩过，别想当然）：回执里声明的**必填键**
        #   （`output_schema.required`）必须始终在 ⇒ 工具面会校验、缺了直接
        #   `bad_tool_output`。所以 pick 是"**补集**"：骨架照给，另外把你点名的
        #   深层路径**精确取出来**。本项目里它真正省 token 的地方是 `series`
        #   —— 那是**非必填**的大负载（`pick=["series.loss.values"]` 时不再回整条
        #   `series`，只回 loss 那几个点）。
        if pick:
            required = {k: out[k] for k in ("ok", "run_id", "available", "latest",
                                            "tags", "error") if k in out}
            body = {k: v for k, v in out.items() if k not in required}
            merged = _pick(body, pick)
            # 若点了必填键内部的路径（如 `latest.loss`），把它作为**额外**键给出
            for k, v in _pick(out, pick).items():
                merged.setdefault(k, v)
            return {**required, **merged}
        return out

    return define_tool(
        name="describe_training",
        description="读一个 run 的**训练过程观测数值**（逐 epoch）：最新一轮的 "
                    "loss / val_loss / grad_norm / lr / dead_relu / 逐层梯度范数，"
                    "可选整条序列（include_series=true）用于判断趋势"
                    "（如'梯度是否在多个数量级上快速增大'）。"
                    "序列较长时用 `pick` 只取关心的那几条（如 "
                    "['series.loss.values']），别把整条序列灌进上下文。"
                    "⚠ 数据来自 TensorBoard 事件文件：**没有事件文件就如实说读不到**"
                    "（训练时未装 tensorboard 时如此）。",
        parameters={
            "run_id": {"type": "string",
                       "description": "运行编号；留空则按最近落盘的 run 找", "default": ""},
            "include_series": {"type": "boolean",
                               "description": "是否连整条逐 epoch 序列一起回"
                                              "（默认 false 只回最新一轮；序列较大）",
                               "default": False},
            "limit": {"type": "integer",
                      "description": "每个 tag 最多回多少点（只对 include_series 生效）",
                      "default": 200},
            "pick": PICK_PARAM,
        },
        output_schema={"type": "object",
                       "required": ["ok", "run_id", "available", "latest",
                                    "tags", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


def _describe_networks(engine: MLEngine,
                       banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    """⭐ 已保存的自定义网络清单（**让 AI 不必反复重贴同一段代码**）。

    为什么必须有（用户点破的实际风险）：改个学习率就要重贴一遍网络代码的话，
    token 全烧在复制粘贴上 ⇒ "自由度"换来的是低效。有了它，第一次写、之后按名字取。

    ⚠ 名字是 `describe_networks`（不是 `list_networks`）：框架有命名律——
    `list_` 前缀被禁（"视图/枚举不是工具，是 schema 的投影"），
    与 `describe_methods` 同类。实测撞过这条守卫。
    """

    def execute(*, limit: int = 50) -> Mapping[str, Any]:
        got = engine.query("network_catalog", limit=int(limit))
        return {"ok": True, **got, "error": ""}

    return define_tool(
        name="describe_networks",
        description="列出自定义网络（`run_custom` 存下来的）：名字、主指标、"
                    "源码行数、用过几次、来自哪个 run。"
                    "**先查这里再动手写**——已经跑过的网络可以直接取回源码改一版"
                    "（用 describe_network），不必重贴。",
        parameters={"limit": {"type": "integer",
                              "description": "最多扫最近多少个 run 的存档",
                              "default": 50}},
        output_schema={"type": "object",
                       "required": ["ok", "networks", "scanned", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


def _describe_network(engine: MLEngine,
                      banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    """按名字取回已保存网络的**源码**（改一版再跑，不重贴）。"""

    def execute(*, name: str, limit: int = 200) -> Mapping[str, Any]:
        got = engine.query("network_source", name=str(name), limit=int(limit))
        return {"ok": True, **got, "error": ""}

    return define_tool(
        name="describe_network",
        description="按名字取回某个已保存自定义网络的**源码**（连同它来自哪个 run）。"
                    "拿它改一版再 `run_custom` 是最省的迭代方式。"
                    "⚠ 找不到就如实回 available=false + 原因，不回空源码。",
        parameters={"name": {"type": "string", "description": "网络名字（见 list_networks）"},
                    "limit": {"type": "integer",
                              "description": "最多扫最近多少个 run 的存档",
                              "default": 200}},
        output_schema={"type": "object",
                       "required": ["ok", "available", "name", "code", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


# ---------------------------------------------------------------- 受控命令
def _run_custom(engine: MLEngine, wrap,
                banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    """⭐ 跑**调用方给出的网络代码**（AI 写结构，系统管训练与观测）。

    为什么要有它（用户裁决的设计）：`torch_mlp` 的旋钮再多也表达不了**拓扑**——
    拓扑写死成 `nn.Sequential` ⇒ 残差/多分支/任意前向全都做不到。本工具把
    "结构"这一段交给代码，而**其余一切照旧**（切分/指标/存档/五页检视/
    `describe_training` 观测面全部白拿，因为它跑的就是一台 `MLMethod`）。
    """

    def execute(*, code: str, dataset_id: str = "",
                overrides: Mapping[str, Any] | None = None, seed: int | None = None,
                diag: bool = False, persist: bool = False, name: str = "",
                resource_guard: Mapping[str, Any] | None = None,
                device: str = ""
                ) -> Mapping[str, Any]:
        spec: dict[str, Any] = {"command": "run_custom", "code": str(code),
                                "diag": bool(diag), "persist": bool(persist)}
        if dataset_id:
            spec["dataset_id"] = dataset_id
        if overrides:
            spec["overrides"] = dict(overrides)
        if seed is not None:
            spec["seed"] = seed
        if name:
            spec["name"] = str(name)
        if resource_guard:
            spec["resource_guard"] = dict(resource_guard)
        if device:
            spec["device"] = str(device)
        receipt = _raise_on_failure(engine.run(spec, context=_local_context()))
        return {"ok": bool(receipt["ok"]), "run_id": receipt.get("run_id", ""),
                "network_name": receipt.get("network_name", ""),
                "primary_metric": receipt.get("primary_metric", ""),
                "metrics": dict(receipt.get("metrics", {})),
                "elapsed_s": receipt.get("elapsed_s", 0.0),
                "warnings": list(receipt.get("warnings", [])),
                "error": receipt.get("error", "")}

    return define_tool(
        name="run_custom",
        description="跑一段**自定义网络代码**：代码只需要定义 `class Network` 并实现 "
                    "`build(self, n_in, n_out, *, nn)`（返回 nn.Module），"
                    "可选 `loss` / `configure` / `forward_step`。"
                    "训练循环、TensorBoard、逐 epoch 观测、指标、存档、检视页"
                    "**全部由系统代管**（所以残差/多分支/任意前向都能跑，"
                    "而 describe_training 照样读得到曲线）。"
                    "⚠ 代码只写结构：**不许**自己写盘或碰 GUI（观测由系统统一负责）。",
        parameters={
            "code": {"type": "string",
                     "description": "定义 `class Network` 的 Python 代码"},
            "dataset_id": {"type": "string",
                           "description": "数据集引用；留空用最近准备的一份",
                           "default": ""},
            "overrides": {"type": "object",
                          "description": "训练超参覆写：epochs/batch/lr/"
                                         "weight_decay/val_split/latent_every/"
                                         "grad_clip/early_stop/early_stop_patience"},
            "seed": {"type": "integer", "description": "随机种子（整数）"},
            "diag": {"type": "boolean", "description": "是否保留诊断信息",
                     "default": False},
            "persist": {"type": "boolean",
                        "description": "是否写入 ML 运行存档（**代码会跟着存档走**，"
                                       "之后可按名字复用，不必重贴）",
                        "default": False},
            "name": {"type": "string",
                     "description": "给这段网络起个名字（存进存档便于复用）",
                     "default": ""},
            "resource_guard": {"type": "object",
                               "description": "资源预算声明，例如 {'max_kernel_mb': 128}"},
            "device": {"type": "string",
                       "description": "计算设备：auto（默认，有卡就上卡）/ cpu / cuda:0。"
                                      "⚠ 网络代码里建张量要用 `ctx['device']`"
                                      "（模型可能已在显卡上），否则撞'混合设备'",
                       "default": ""},
        },
        output_schema={"type": "object",
                       "required": ["ok", "run_id", "primary_metric", "metrics",
                                    "elapsed_s", "warnings", "error"]},
        execute=wrap(execute),
        banned_words=banned_words,
    )


def _prepare_dataset(engine: MLEngine, wrap,
                     banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, source: Mapping[str, Any], name: str = "",
                target: str = "", time_col: str = "",
                diag: bool = False,
                pipeline: Mapping[str, Any] | None = None,
                graph: Mapping[str, Any] | None = None,
                payload: Mapping[str, Any] | None = None,
                target_kind: str = "") -> Mapping[str, Any]:
        spec: dict[str, Any] = {"command": "prepare_dataset", "source": source,
                                "diag": bool(diag)}
        if name:
            spec["name"] = name
        if target:
            spec["target"] = target
        if time_col:
            spec["time_col"] = time_col
        if pipeline:
            spec["pipeline"] = dict(pipeline)
        if graph:
            spec["graph"] = dict(graph)
        if payload:
            spec["payload"] = dict(payload)
        if target_kind:
            spec["target_kind"] = str(target_kind)
        receipt = _raise_on_failure(
            engine.run(spec, context=_local_context()))
        dataset = receipt.get("dataset") or {}
        return {"ok": bool(receipt["ok"]),
                "dataset_id": receipt.get("dataset_id", ""),
                "pipeline_id": receipt.get("pipeline_id", ""),
                "n_rows": dataset.get("n_rows", 0),
                "n_features": dataset.get("n_features", 0),
                "target": dataset.get("target") or "",
                # 回执里带上"这条链长什么样"与**真实划分口径**：调用方不用再查一次，
                # 而且不用去猜"我配的切分到底生效了没有"。
                "pipeline_chain": list(receipt.get("pipeline_chain") or []),
                "split_policy": dict(receipt.get("split_policy") or {}),
                "graph": dict(receipt.get("graph") or {}),
                # 负载的**形状与 dtype**：这是"我的复数还在不在"的直接答案
                "payload": dict(receipt.get("payload") or {}),
                "target_kind": receipt.get("target_kind", ""),
                "warnings": list(receipt.get("warnings", [])),
                "error": receipt.get("error", "")}

    return define_tool(
        name="prepare_dataset",
        description="登记一份数据并在**训练部分**上拟合预处理链，返回数据集引用、"
                    "管道指纹、**链摘要**与**实际划分口径**。"
                    "可给 `pipeline` 声明来**自己组装**这条链（选内置步骤+参数、"
                    "换顺序、写自定义步骤、指定切分比例）；省略则用默认链"
                    "（缺失值→编码→缩放→划分）。",
        parameters={
            "source": {"type": "object",
                       "description": "数据来源声明，例如 {'kind': 'csv', 'path': 'x.csv'}、"
                                      "{'kind': 'frame', 'frame': <表格>}、"
                                      "{'kind': 'arrays', 'X': [...], 'y': [...]}、"
                                      "{'kind': 'synthetic', 'task': 'classification'}"},
            "name": {"type": "string", "description": "数据集显示名", "default": "dataset"},
            "target": {"type": "string", "description": "目标列名（监督/时序任务需要）",
                       "default": ""},
            "time_col": {"type": "string", "description": "时序任务的时间列名",
                         "default": ""},
            "diag": {"type": "boolean", "description": "是否保留预处理链的诊断中间产物",
                     "default": False},
            "pipeline": {"type": "object",
                         "description": "预处理链声明（省略=默认链）："
                                        "{'steps': [{'key': 'missing', 'params': "
                                        "{'strategy': 'median'}}, {'key': 'scale', "
                                        "'params': {'mode': 'robust'}}, {'key': "
                                        "'my_step', 'code': 'class Step(StepBase): ...'}], "
                                        "'split': {'test_size': 0.2, 'stratify': true, "
                                        "'seed': 7}}。顺序即执行顺序；未知键/未知步骤/"
                                        "坏取值都会**报错而不是静默忽略**。"},
            "graph": {"type": "object",
                      "description": "**图结构**（GNN 用，可选）："
                                     "{'edges': [[src, dst], ...]} 或 "
                                     "{'edges_path': 'edges.csv', 'src': 's', 'dst': 'd', "
                                     "'weight': 'w'}。**行号指向观测表的行**（节点=行）；"
                                     "默认按无向图自动补反向边；边指向不存在的节点会报错。"
                                     "src/dst/weight 之外的列会当作**多分量边特征**"
                                     "（`ctx['edge_attr']`）。带了图之后，只有声明支持图的"
                                     "方法（run_custom 的自定义网络）能消费它——别的方法"
                                     "会被明确拒绝，不会**静默忽略**拓扑。"},
            "payload": {"type": "object",
                        "description": "**逐行负载**（可选，给变长/复数的大数组）："
                                       "{'hdf5': {'path': 'd.h5', 'datasets': "
                                       "{'amps': '/config/amps'}}} 或 "
                                       "{'arrays': {'名字': [[...], [...]]}}。"
                                       "每个数组**第 0 维 = 行数**（切一批行 = 切前几维）；"
                                       "**保住 dtype**（复数就是复数、位串是整数、"
                                       "mask 是 bool）；变长的维要先 pad 再配 mask"
                                       "（`core.paddings.pad_rows` 是助手）。"
                                       "配 `target_kind='custom'` 就是"
                                       "「没有标准标签、指标由网络自己上报」的任务。"},
            "target_kind": {"type": "string",
                            "description": "只有 'custom' 需要写（无标准标签、"
                                           "标准指标不适用、由 Network.metrics 上报）；"
                                           "回归/分类由数据自动推断", "default": ""},
        },
        output_schema={"type": "object",
                       "required": ["ok", "dataset_id", "pipeline_id", "n_rows",
                                    "n_features", "pipeline_chain", "split_policy",
                                    "payload", "target_kind", "warnings", "error"]},
        execute=wrap(execute),
        banned_words=banned_words,
    )


def _run_method(engine: MLEngine, wrap,
                banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, method: str, dataset_id: str = "",
                overrides: Mapping[str, Any] | None = None, seed: int | None = None,
                resource_guard: Mapping[str, Any] | None = None,
                device: str = "", persist: bool = False,
                diag: bool = False) -> Mapping[str, Any]:
        # ⚠ `diag` 曾在 `parameters` 里声明、却**没进这个签名**：调用方按声明传
        #   `diag` 会撞上"unexpected keyword argument" ⇒ 该参数在工具面上**从来
        #   调不通**（声明与实现漂移）。实测于 2026-10-03（我自己的探针撞到的）。
        #   判据：`test_ml_delivery` 的逐参数探针表对 `run_method.diag` 有真值。
        spec: dict[str, Any] = {"command": "run_method", "method": method,
                                "persist": bool(persist), "diag": bool(diag)}
        if dataset_id:
            spec["dataset_id"] = dataset_id
        if overrides:
            spec["overrides"] = dict(overrides)
        if seed is not None:
            spec["seed"] = seed
        if resource_guard:
            spec["resource_guard"] = dict(resource_guard)
        if device:
            spec["device"] = device
        receipt = _raise_on_failure(
            engine.run(spec, context=_local_context()))
        return {"ok": bool(receipt["ok"]), "run_id": receipt.get("run_id", ""),
                "method": method,
                "primary_metric": receipt.get("primary_metric", ""),
                "metrics": dict(receipt.get("metrics", {})),
                "elapsed_s": receipt.get("elapsed_s", 0.0),
                "warnings": list(receipt.get("warnings", [])),
                "error": receipt.get("error", "")}

    return define_tool(
        name="run_method",
        description="在已准备的数据视图上运行一个机器学习方法并返回运行编号与指标摘要；"
                    "预测数组等大产物留在结果侧，此处只给运行编号。",
        parameters={
            "method": {"type": "string", "description": "已注册的方法名"},
            "dataset_id": {"type": "string",
                           "description": "数据集引用；留空用最近准备的一份",
                           "default": ""},
            "overrides": {"type": "object", "description": "运行级参数覆写（不改全局默认）"},
            "seed": {"type": "integer", "description": "随机种子（整数）"},
            "diag": {"type": "boolean", "description": "是否保留诊断信息", "default": False},
            "persist": {"type": "boolean", "description": "是否把本次运行写入 ML 运行存档",
                        "default": False},
            "resource_guard": {"type": "object",
                               "description": "资源预算声明，例如 {'max_kernel_mb': 128}"},
            "device": {"type": "string", "description": "计算设备声明：auto / cpu / cuda:0",
                       "default": ""},
        },
        output_schema={"type": "object",
                       "required": ["ok", "run_id", "method", "primary_metric",
                                    "metrics", "elapsed_s", "warnings", "error"]},
        execute=wrap(execute),
        banned_words=banned_words,
    )


def _run_method_batch(engine: MLEngine, wrap,
                      banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, methods: list[str], dataset_id: str = "",
                overrides: Mapping[str, Any] | None = None,
                seed: int | None = None,
                resource_guard: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        # ⚠ 这里**不能**加 `diag`：`run_method_batch` 的**命令声明**里没有它
        #   （`commands.py`），加了会被 `validate_spec` 以"未声明的参数"拒。
        #   工具面的参数集必须是命令声明参数集的**子集**——这条由
        #   `test_tool_params_stay_within_command_declaration` 守着。
        spec: dict[str, Any] = {"command": "run_method_batch",
                                "methods": list(methods)}
        if dataset_id:
            spec["dataset_id"] = dataset_id
        if overrides:
            spec["overrides"] = dict(overrides)
        if seed is not None:
            spec["seed"] = seed
        if resource_guard:
            spec["resource_guard"] = dict(resource_guard)
        receipt = _raise_on_total_failure(
            engine.run(spec, context=_local_context()))
        runs = list(receipt.get("runs", []))
        return {"ok": bool(receipt["ok"]),
                "run_ids": list(receipt.get("run_ids", [])),
                "primary_metric": receipt.get("primary_metric", ""),
                "runs": runs,
                "failed_methods": [str(r.get("method", "")) for r in runs
                                   if isinstance(r, Mapping) and not r.get("ok")],
                "warnings": list(receipt.get("warnings", [])),
                "error": receipt.get("error", "")}

    return define_tool(
        name="run_method_batch",
        description="在同一份数据视图上顺序运行多个方法，逐方法隔离失败；"
                    "返回每个方法的运行编号与指标，失败项只给编号与原因。",
        parameters={
            "methods": {"type": "array", "description": "按顺序运行的方法名列表（受数量上限约束）"},
            "dataset_id": {"type": "string",
                           "description": "数据集引用；留空用最近准备的一份",
                           "default": ""},
            "overrides": {"type": "object", "description": "所有方法共用的运行级参数覆写"},
            "seed": {"type": "integer", "description": "随机种子（整数）"},
            "resource_guard": {"type": "object",
                               "description": "资源预算声明，例如 {'max_kernel_mb': 128}"},
        },
        output_schema={"type": "object",
                       "required": ["ok", "run_ids", "primary_metric", "runs",
                                    "failed_methods", "warnings", "error"]},
        execute=wrap(execute),
        banned_words=banned_words,
    )


def _compare_methods(engine: MLEngine, wrap,
                     banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, methods: list[str], dataset_id: str = "",
                seed: int | None = None,
                overrides: Mapping[str, Any] | None = None,
                resource_guard: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        spec: dict[str, Any] = {"command": "compare_methods",
                                "methods": list(methods)}
        if dataset_id:
            spec["dataset_id"] = dataset_id
        if seed is not None:
            spec["seed"] = seed
        if overrides:
            spec["overrides"] = dict(overrides)
        if resource_guard:
            spec["resource_guard"] = dict(resource_guard)
        # 与 run_method_batch 同一规则：部分成功不是工具失败（否则成功项的
        # run_id / table 会被一起丢掉，与 description 承诺的"失败项只给编号与
        # 原因"自相矛盾）；全部失败才 fail loud。
        receipt = _raise_on_total_failure(
            engine.run(spec, context=_local_context()))
        return {"ok": bool(receipt["ok"]),
                "run_ids": list(receipt.get("run_ids", [])),
                "primary_metric": receipt.get("primary_metric", ""),
                "table": list(receipt.get("table", [])),
                "winner": dict(receipt.get("winner", {})),
                "warnings": list(receipt.get("warnings", [])),
                "error": receipt.get("error", "")}

    return define_tool(
        name="compare_methods",
        description="在同一份数据视图上对比多个方法的主指标，返回可排序对比表与最优方法；"
                    "逐方法隔离失败，失败项只给运行编号与失败原因。",
        parameters={
            "methods": {"type": "array", "description": "参与对比的方法名列表"},
            "dataset_id": {"type": "string",
                           "description": "数据集引用；留空用最近准备的一份",
                           "default": ""},
            "seed": {"type": "integer", "description": "随机种子（整数）"},
            "overrides": {"type": "object", "description": "所有方法共用的运行级参数覆写"},
            "resource_guard": {"type": "object",
                               "description": "资源预算声明，例如 {'max_kernel_mb': 128}"},
        },
        output_schema={"type": "object",
                       "required": ["ok", "run_ids", "primary_metric", "table",
                                    "winner", "warnings", "error"]},
        execute=wrap(execute),
        banned_words=banned_words,
    )


# ---------------------------------------------------------------- job 调度面
def _require_job_host(host: Any) -> Any:
    """job 工具的宿主 seam：没有就结构化拒（不静默假成功）。"""
    if host is None:
        raise MechaError(
            "job 工具没有可用的 Job 服务",
            kind="no_job_host",
            hint=_NO_JOB_HOST_HINT,
            suggest="经 assemble_ml_mecha 装配的实例才带 Job 服务",
        )
    return host


def _submit_run(host: Any, side: str,
                banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, method: str = "", dataset_id: str = "",
                overrides: Mapping[str, Any] | None = None,
                seed: int | None = None,
                device: str = "",
                persist: bool = False, diag: bool = False,
                code: str = "", name: str = "") -> Mapping[str, Any]:
        ml = _require_job_host(host)
        # ⭐ **长训练的正确形态**（用户裁决）：只发一个"开始"信号就走，
        #   之后靠 read_job 主动轮询。这是唯一可行的模式——LLM 的一次工具调用
        #   本质是 HTTP 请求，客户端必须有超时（本实例 10 分钟），所以
        #   "同步等训练跑完"对长训练天生不成立。
        #
        # `code` 给了就提交 `run_custom`（AI 写的网络），否则提交 `run_method`。
        # ⚠ 这条以前**不存在**：`submit_run` 把命令**写死**成 run_method ⇒
        #   自定义网络只能同步跑，也就等于长训练跑不了（10 分钟超时后 AI
        #   连 run_id 都拿不到）。宿主层本来就支持任意命令，缺的只是这里透传。
        if not code and not method:
            raise MechaError(
                "submit_run 需要 method（内置方法名）或 code（自定义网络代码）",
                kind="missing_arguments",
                hint="二选一：内置方法给 method=；AI 写的网络给 code=（可配 name=）",
                suggest='submit_run(method="torch_mlp", ...) 或 '
                        'submit_run(code="class Network: ...", name="resnet_v1")',
            )
        command = "run_custom" if code else "run_method"
        spec: dict[str, Any] = {"command": command,
                                "persist": bool(persist), "diag": bool(diag)}
        if code:
            spec["code"] = str(code)
            if name:
                spec["name"] = str(name)
        else:
            spec["method"] = str(method)
        if dataset_id:
            spec["dataset_id"] = dataset_id
        if overrides:
            spec["overrides"] = dict(overrides)
        if seed is not None:
            spec["seed"] = seed
        if device:
            spec["device"] = device
        # 通道与 call_id 由 MLMecha.submit 带进后台线程（contextvars 不跨线程）；
        # 这里**不**用 _with_channel —— 见模块 docstring「长任务：job 类工具」。
        job = ml.submit(spec, side=side)
        return {"ok": True, "job_id": job.id, "command": job.command,
                "cancel_supported": bool(job.cancel_supported),
                "state": job.state.value, "error": ""}

    return define_tool(
        name="submit_run",
        description="把一个运行**提交为后台任务**并立刻返回任务编号（长训练就该用这条："
                    "发一个开始信号就走，之后用 read_job 查进度）。"
                    "给 `method` 跑内置方法；给 `code` 跑自定义网络（AI 写的结构）。"
                    "⚠ 后台跑完会**自动落盘**（runs/<run_id>/record.json）——"
                    "所以就算你后来丢了 job_id，结果也还在。",
        parameters={
            "method": {"type": "string",
                       "description": "内置方法名（与 code 二选一）", "default": ""},
            "code": {"type": "string",
                     "description": "自定义网络代码（定义 `class Network`；与 method 二选一）",
                     "default": ""},
            "name": {"type": "string",
                     "description": "自定义网络的名字（存进存档便于复用）", "default": ""},
            "dataset_id": {"type": "string",
                           "description": "数据集引用；留空用最近准备的一份",
                           "default": ""},
            "overrides": {"type": "object", "description": "运行级参数覆写（不改全局默认）"},
            "seed": {"type": "integer", "description": "随机种子（整数）"},
            "device": {"type": "string", "description": "计算设备声明：auto / cpu / cuda:0",
                       "default": ""},
            "persist": {"type": "boolean",
                        "description": "是否把本次运行写入 ML 运行存档（后台任务另有一条"
                                       "**终态自动落盘**的兜底，见描述）",
                        "default": False},
            "diag": {"type": "boolean",
                     "description": "是否保留诊断信息（如神经网络的特征归因页需要它）",
                     "default": False},
        },
        output_schema={"type": "object",
                       "required": ["ok", "job_id", "command", "cancel_supported",
                                    "state", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


def _read_job(host: Any,
              banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, job_id: str, wait_s: float = 0.0) -> Mapping[str, Any]:
        ml = _require_job_host(host)
        try:
            timeout = max(0.0, min(float(wait_s or 0.0), READ_JOB_MAX_WAIT_S))
        except (TypeError, ValueError):
            raise MechaError(
                f"wait_s 必须是数字，收到 {wait_s!r}",
                kind="bad_wait_s",
                hint=f"wait_s 是可选阻塞秒数，上限 {READ_JOB_MAX_WAIT_S:g}",
            ) from None
        status = ml.jobs.wait(job_id, timeout)
        raw = status.get("progress") or []
        progress = [{"frac": float(p[0]), "stage": str(p[1])}
                    for p in raw if isinstance(p, (list, tuple)) and len(p) == 2]
        # ⭐ 富载荷（loss / val_loss / run_id）取自挂在执行上下文上的 sidecar。
        #   为什么要有它（用户裁决）：
        #   ① `result_ref`（含 run_id）**只在终态才有** ⇒ AI 训练中途拿不到句柄，
        #      "边跑边查曲线"实际不可用；带上 run_id 后第一次轮询就能去查。
        #   ② 只报 frac/stage 时，判断"该不该早停"还得再拉一次完整曲线；
        #      带上 loss/val_loss 后**一次轮询就能决策**。
        #   ⚠ 逐代列表**有界**（`PROGRESS_EPOCH_WINDOW`）：完整曲线按需从
        #     `describe_training` 拉，不在这里线性膨胀。
        extra: dict = {}
        try:
            from ml_mecha.engine import _PROGRESS_SIDECAR
            extra = dict(getattr(ml.jobs.get(job_id).context,
                                 _PROGRESS_SIDECAR, None) or {})
        except Exception:                                   # noqa: BLE001
            extra = {}
        latest = dict(progress[-1]) if progress else {}
        latest.update(extra.get("latest") or {})
        return {"ok": True, "job_id": str(status.get("id") or job_id),
                "state": str(status.get("state") or ""),
                "cancel_supported": bool(status.get("cancel_supported", False)),
                "progress": progress,
                "progress_latest": latest,
                "run_id": str(extra.get("run_id") or ""),
                "epochs": list(extra.get("epochs") or []),
                "result_ref": dict(status.get("result_ref") or {}),
                "error": str(status.get("error") or ""),
                "ref_error": str(status.get("ref_error") or "")}

    return define_tool(
        name="read_job",
        description="按任务编号查一次后台任务的进度与结果：状态、**逐代数值**"
                    "（`epochs`/`progress_latest` 带 loss/val_loss，够判断该不该早停）、"
                    "**训练中途就能拿到的 `run_id`**（拿去 describe_training 查曲线）、"
                    "结果引用、失败原因；不阻塞（除非给 wait_s）。",
        parameters={
            "job_id": {"type": "string", "description": "submit_run 返回的任务编号"},
            "wait_s": {"type": "number",
                       "description": f"可选：最多阻塞多少秒等它推进（上限 "
                                      f"{READ_JOB_MAX_WAIT_S:g}）；默认 0 立即返回",
                       "default": 0},
        },
        output_schema={"type": "object",
                       "required": ["ok", "job_id", "state", "cancel_supported",
                                    "progress", "progress_latest", "run_id",
                                    "epochs", "result_ref", "error", "ref_error"]},
        execute=execute,
        banned_words=banned_words,
    )


def _cancel_run(host: Any,
                banned_words: tuple[str, ...] = ()) -> ToolDefinition:
    def execute(*, job_id: str) -> Mapping[str, Any]:
        ml = _require_job_host(host)
        # ⚠ **行为变更（2026-09-27）**：从前这里自己探两次状态，
        # 并**硬编** `cancel_requested: True`——于是对**已经跑完**的 job 也说"已请求取消"
        # （那正是框架 ADR 点名的谎：下游会以为它停了）。
        # 框架现在把真话**写在返回值里**（`state_before` / `terminal` / `cancel_requested`；
        # 终态 ⇒ `cancel_requested=False`）⇒ 直接采用，不再自探、不再硬编。
        #
        # 关于字段：`state_before`（请求**前**的状态）与框架同名字段**是同一个事实** ⇒ 保留原名；
        # 旧的 `state`（请求**后** 0ms 再探一次）**删掉**——它不是独立事实（协作式取消下几乎
        # 恒等于 `state_before`），且与 `cancel_requested` 并列会诱导"还在跑 ⇒ 会停"的误读；
        # 换成 `terminal`（它是否**已经结束**）——这才是模型真正需要知道的那件事。
        got = ml.jobs.cancel(job_id)
        return {"ok": True, "job_id": str(got.get("job_id") or job_id),
                "cancel_requested": bool(got.get("cancel_requested")),
                "terminal": bool(got.get("terminal")),
                "state_before": str(got.get("state_before") or ""), "error": ""}

    return define_tool(
        name="cancel_run",
        description="请求取消一个后台任务。取消是协作式的：任务的取消标志被置位，"
                    "它会在下一个检查点停止；不可取消的任务会被明确拒绝。"
                    "回执里 `terminal=true` 表示**它本来就已经结束**（此时什么都没取消）。",
        parameters={
            "job_id": {"type": "string", "description": "submit_run 返回的任务编号"},
        },
        output_schema={"type": "object",
                       "required": ["ok", "job_id", "cancel_requested",
                                    "terminal", "state_before", "error"]},
        execute=execute,
        banned_words=banned_words,
    )


def register_ml_tools(registry: Any, engine: MLEngine, **kwargs) -> list[Any]:
    """把工具面注册进宿主的 ``ToolRegistry``，返回 disposer 列表（可全撤）。"""
    disposers = []
    for tool in build_ml_tools(engine, **kwargs):
        disposers.append(registry.register(tool))
    return disposers


__all__ = ["build_ml_tools", "register_ml_tools", "LOCAL_ACTOR",
           "READ_JOB_MAX_WAIT_S"]
