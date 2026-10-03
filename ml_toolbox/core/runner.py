# -*- coding: utf-8 -*-
"""运行器：把 (方法, 数据视图, 配置) 跑成 MLResult；支持批量遍历与对比。

所有异常在方法层被捕获进 MLResult.error（遍历中一个方法失败不影响其他）。
"""
from __future__ import annotations

import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .contracts import (MLResult, MLMethod, RunConfig, DataSpec,
                        is_lower_better,
                        TASK_SUPERVISED, TASK_TIMESERIES)
from . import registry
from .resources import (ResourceBudget, ResourceBudgetExceeded,
                        ResourceEstimate, check_resources)


@dataclass
class RunRecord:
    """一次运行的完整留痕（实验记录持久化的单元）。"""
    run_id: str
    method: str
    family: str
    task: str
    target_kind: str | None
    dataset: str
    pipeline_id: str
    config: dict
    result: MLResult
    started_at: float = field(default_factory=time.time)
    # wall-clock 分段耗时（秒）。legacy_elapsed 与 MLResult.elapsed 保持一致；
    # total 包含 evaluate/CV。CV 内会再次 fit，故各阶段不应简单相加。
    timings: dict = field(default_factory=dict)
    # 仅在 extras["resource_guard"] 启用且通过/触发估算时存在。
    resource_estimate: ResourceEstimate | None = None


def _run_id(method: str, seed: int) -> str:
    return f"{method}-{int(time.time() * 1000) % 10_000_000}-{seed}"


class _StageTimer:
    """极低开销 wall-clock 分段计时器。"""

    def __init__(self):
        self.values: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.values[name] = self.values.get(name, 0.0) \
                + max(0.0, time.perf_counter() - t0)

    def ensure(self, *names: str):
        for name in names:
            self.values.setdefault(name, 0.0)


def _resource_guard(method: MLMethod, spec: DataSpec,
                    cfg: RunConfig) -> ResourceEstimate | None:
    """执行 extras.resource_guard；未配置时零额外估算开销。"""
    guard = cfg.extras.get("resource_guard")
    if guard is None or guard is False:
        return None
    options = dict(guard) if isinstance(guard, dict) else {}
    budget = ResourceBudget.from_mapping(options)
    X = spec.X
    shape = getattr(X, "shape", (0, 0))
    n_samples, n_features = int(shape[0]), int(shape[1]) if len(shape) > 1 else 0
    params = method.params(cfg)
    return check_resources(
        method, n_samples, n_features, params=params,
        n_qubits=options.get("n_qubits"), budget=budget,
        safety_factor=float(options.get("safety_factor", 1.0)),
    )


def run_one(method: MLMethod, spec: DataSpec,
            cfg: RunConfig | None = None) -> RunRecord:
    """运行一个方法并返回向后兼容的 ``MLResult.elapsed`` 与分段 timings。

    ``MLResult.elapsed`` 保持旧语义：fit + predict 阶段（以及 split/guard）
    的 wall-clock；``RunRecord.timings['total']`` 额外包含 evaluate/CV。
    """
    cfg = cfg or RunConfig()
    wall0 = time.time()
    perf0 = time.perf_counter()
    timer = _StageTimer()
    resource_estimate = None
    # ⭐ **跑之前就定死 run_id**（原来在 return 处现算）。
    # 为什么必须在前面：神经网络族的 TensorBoard 事件要**逐 epoch 实时写**到
    # `runs/<run_id>/tb/`，而那个目录一路不能变（变了直播就断）。
    # 经 `cfg.extras` 往下传；末态构造 RunRecord 时**复用同一个值** ⇒ 目录名与记录名一致。
    run_id = cfg.extras.get("run_id") or _run_id(method.name, cfg.seed)
    cfg.extras["run_id"] = run_id
    #: **方法内部的进度钩子**（可选，逗号（epoch, total）之外不承诺任何形状）。
    #: 为什么走 extras 而不是新增 `fit` 形参：`fit(problem, cfg, diag)` 是方法层契约，
    #: 为了一个可选观测能力去改**每个方法**的签名不划算；而 extras 本来就是
    #: "任务级扩展"的通道（cv_folds 等已在用）。
    #: ⚠ **它不落盘、不进指纹**：运行结束前必须摘掉，否则 RunConfig 会被
    #: `record.json` 序列化时炸在"不是 JSON 的东西"上（见下面 finally）。
    on_progress = cfg.extras.get("on_progress")
    try:
        with timer.stage("guard"):
            resource_estimate = _resource_guard(method, spec, cfg)
        # ⭐ **能力闸**：数据带了辅助结构（图边表等）而方法**没声明支持** ⇒ 拒绝。
        #   为什么必须在这里拦（而不是等方法自己调 `require`）：不拦的话
        #   `torch_mlp` 这类只读 `rows("train")` 的方法会**静默忽略**掉图结构，
        #   然后按"节点互相独立"跑出一个看起来正常、其实错的分数。
        #   （老契约下"时序忽略外生 X"就是这么静默发生的。）
        present = set(spec.aux or {})
        declared = set(getattr(method, "accepts_aux", ()) or ())
        if present - declared:
            missing = sorted(present - declared)
            res = MLResult(
                method_name=method.name, task=method.task,
                error=(f"{method.name} 不支持这份数据带的辅助结构 {missing}"
                       f"（它声明支持的是 {sorted(declared)}）。"
                       "带图结构的数据请用能消费它的方法（如 run_custom 的自定义网络，"
                       "它声明了 accepts_aux=(AUX_GRAPH,)）。"))
            return RunRecord(run_id=run_id, method=method.name,
                             family=method.family, task=method.task,
                             target_kind=spec.target_kind, dataset=spec.meta.get("dataset", ""),
                             pipeline_id=spec.meta.get("pipeline_id", ""),
                             config={"overrides": dict(cfg.overrides), "diag": bool(cfg.diag),
                                     "seed": int(cfg.seed),
                                     "extras": {k: v for k, v in cfg.extras.items()
                                                if k != "on_progress"}},
                             result=res, timings={"total": 0.0})
        # ⭐ 范式：方法收到的是**一个问题**（观测 + 标签 + 关系），不是裸的 (X, y)。
        #   `problem` 承载 spec（含 `spec.aux` 里的图边表等）⇒ predict/evaluate
        #   那两段仍用 `spec.split()`（测试数据本来就是 (X, y) 形状，不必绕一圈）。
        problem = method.problem(spec)
        with timer.stage("split"):
            Xtr, Xte, ytr, yte = spec.split()
        with timer.stage("fit"):
            res = method.fit(problem, cfg, diag=cfg.diag)
        if res.ok and method.task == TASK_SUPERVISED:
            if problem.partition_mode() == "masks":
                # ⭐ **结构化数据（图）不走这条 predict**：这里只拿得到**测试行**
                #   （`Xte`），而图的前向需要**整张图的边** ⇒ 由方法自己的
                #   `fit` 在掩码语义下把 `y_pred`/`y_true` 算好（见
                #   `custom._fit_graph`）。指标评估照旧走 `evaluate`（它只读 artifacts）。
                pass
            else:
                with timer.stage("predict"):
                    try:
                        res.artifacts.update(method.fit_extra_artifacts(Xte, res))
                        pred = method.predict(Xte, res)
                        res.artifacts["y_true"] = np.asarray(yte)
                        res.artifacts["y_pred"] = np.asarray(pred)
                    except Exception:
                        res.error = "predict 失败:\n" + traceback.format_exc(limit=6)
        elif res.ok and method.task == TASK_TIMESERIES:
            pass  # fit 内部已产出 forecast/actual
    except ResourceBudgetExceeded as e:
        res = MLResult(method_name=method.name, task=method.task, error=str(e))
        resource_estimate = e.estimate
    except Exception:
        res = MLResult(method_name=method.name, task=method.task,
                       error=traceback.format_exc(limit=10))
    finally:
        # ⚠ **摘掉进度钩子**（它是运行时对象，不是配置）：
        #   留在这里会在 `record.json` 写盘时炸在"不可 JSON 序列化"上，
        #   而且它会被带进 cfg 指纹/快照类的东西里（那都是错的）。
        #   用 `pop(..., None)`：没传过就是 no-op（零开销，P3）。
        if on_progress is not None:
            cfg.extras.pop("on_progress", None)

    # 旧字段语义保持不变：评估/CV 之前结算 elapsed。
    res.target_kind = res.target_kind or spec.target_kind
    res.elapsed = time.time() - wall0
    if res.ok:
        with timer.stage("evaluate"):
            try:
                if method.task == TASK_SUPERVISED and "y_pred" in res.artifacts:
                    res.metrics.update(method.evaluate(res, spec.X, spec.y))
            except Exception:
                res.metrics = dict(res.metrics)
        # 可选交叉验证（cfg.extras['cv_folds']>0 时才跑，默认零开销 —— P3）
        cv_folds = int(cfg.extras.get("cv_folds", 0) or 0)
        if cv_folds >= 2 and method.task == TASK_SUPERVISED:
            with timer.stage("cv"):
                try:
                    # ⚠ 把 `aux` 传进去 ⇒ 带图的数据会被 `default_cv` **响亮拒绝**
                    #   （CV 按行切片会让边索引静默失真）。这里不吞掉那条拒绝：
                    #   把原因写进 diag，让"为什么没跑 CV"看得见。
                    cv = method.cross_validate(spec.X, spec.y, cfg, cv_folds,
                                               target_kind=spec.target_kind,
                                               aux=dict(spec.aux or {}))
                    for k, v in cv.items():
                        if k == "cv_scores":
                            res.diag = dict(res.diag or {})
                            res.diag["cv_scores"] = v
                        else:
                            res.metrics[k] = v
                except Exception as exc:
                    # ⚠ **不静默**：CV 没跑成要留下原因（带图的数据会被明确拒绝，
                    #   那条拒绝就是这里最有价值的信息）。
                    res.diag = dict(res.diag or {})
                    res.diag["cv_skipped"] = f"{type(exc).__name__}: {exc}"
    timer.ensure("guard", "split", "fit", "predict", "evaluate", "cv")
    timer.values["legacy_elapsed"] = res.elapsed
    timer.values["total"] = max(0.0, time.perf_counter() - perf0)
    return RunRecord(run_id=run_id,
                     method=method.name, family=method.family,
                     task=method.task,
                     # ⚠ 用 `res.target_kind`（= 方法**实际**用的 kind），**不是**
                     # `spec.target_kind`（= 数据集的 kind）。两者会不一致：例如 `ridge`
                     # 声明只能回归（`target_kind="regression"`）却可被喂分类数据，此时
                     # 指标是 rmse/mae/r2，而 `spec.target_kind` 是 classification。
                     # 早先落 spec 的那个值 ⇒ 存档自述"分类"但指标全是回归指标
                     # （实测前 400 条里 47 条如此），下游按 target_kind 挑检视页就会挑错
                     # （混淆矩阵画不出来）。上面那行 `res.target_kind = res.target_kind or
                     # spec.target_kind` 已经把"方法没声明"的情形兜住了。
                     target_kind=res.target_kind or spec.target_kind,
                     dataset=spec.meta.get("dataset", ""),
                     pipeline_id=spec.meta.get("pipeline_id", ""),
                     config={"overrides": dict(cfg.overrides),
                             "diag": cfg.diag, "seed": cfg.seed,
                             "extras": dict(cfg.extras)},
                     result=res, timings=dict(timer.values),
                     resource_estimate=resource_estimate)


def save_timed(rec: RunRecord, saver: Callable[[RunRecord], object] | None = None):
    """执行持久化并把耗时写入 ``rec.timings['serialize']``。

    默认延迟导入 ``core.persistence.save_record``，避免 runner <-> persistence
    循环导入。 saver 主要用于批处理/测试注入；失败时耗时仍会记录。
    """
    if saver is None:
        from .persistence import save_record as saver
    t0 = time.perf_counter()
    try:
        return saver(rec)
    finally:
        elapsed = max(0.0, time.perf_counter() - t0)
        rec.timings["serialize"] = elapsed
        rec.timings["total"] = rec.timings.get("total", 0.0) + elapsed


def run_batch(names: list[str], spec: DataSpec,
              cfg: RunConfig | None = None,
              progress=None) -> list[RunRecord]:
    """遍历尝试：同一数据视图下一次跑多个方法（G1）。"""
    cfg = cfg or RunConfig()
    records = []
    for i, nm in enumerate(names):
        if progress:
            progress(i, len(names), nm)
        try:
            m = registry.get(nm)
        except KeyError as e:
            records.append(RunRecord(run_id=_run_id(nm, cfg.seed), method=nm,
                                     family="?", task="?", target_kind=None,
                                     dataset="", pipeline_id="", config={},
                                     result=MLResult(method_name=nm, task="?",
                                                     error=str(e))))
            continue
        if not m.can_handle(spec):
            continue
        records.append(run_one(m, spec, cfg))
    return records


def compare_table(records: list[RunRecord]) -> pd.DataFrame:
    """横向对比表：方法 × 指标（含失败行）。"""
    rows = []
    for r in records:
        row = {"method": r.method, "family": r.family,
               "task": r.task, "ok": r.result.ok,
               "elapsed_s": round(r.result.elapsed, 2)}
        if r.result.ok:
            for k, v in r.result.metrics.items():
                if isinstance(v, (int, float, np.floating)) and k != "cv_scores":
                    row[k] = round(float(v), 4)
            row["primary"] = r.result.primary_metric
        else:
            row["error"] = (r.result.error or "").splitlines()[-1][:120]
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "primary" in df and "ok" in df:
        ok = df[df["ok"]]
        if not ok.empty:
            pm = ok["primary"].iloc[0]
            if pm in ok:
                asc = is_lower_better(pm)
                df = pd.concat([ok.sort_values(pm, ascending=asc),
                                df[~df["ok"]]], ignore_index=True)
    return df


# ---------------------------------------------------------------- 任务适配
def auto_pages(method: MLMethod, spec: DataSpec) -> list:
    """方法未声明检视页时，按 task 给出兜底页（UI 永远有东西可显示）。

    C3：函数体已下沉 methods/plots.py（它持有绘图函数），经 registry 槽
    反向注册。core 不再 import methods——依赖方向恢复为 methods→core。
    提供者未注册（未 load_builtin 的纯后端场景）时返回空表，诚实降级。
    """
    provider = registry.page_provider()
    return provider(method, spec) if provider else []
