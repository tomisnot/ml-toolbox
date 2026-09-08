# -*- coding: utf-8 -*-
"""运行器：把 (方法, 数据视图, 配置) 跑成 MLResult；支持批量遍历与对比。

所有异常在方法层被捕获进 MLResult.error（遍历中一个方法失败不影响其他）。
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .contracts import (MLResult, MLMethod, RunConfig, DataSpec,
                        TASK_SUPERVISED, TASK_CLUSTER, TASK_MANIFOLD,
                        TASK_ANOMALY, TASK_TIMESERIES)
from . import registry


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


def _run_id(method: str, seed: int) -> str:
    return f"{method}-{int(time.time() * 1000) % 10_000_000}-{seed}"


def run_one(method: MLMethod, spec: DataSpec,
            cfg: RunConfig | None = None) -> RunRecord:
    cfg = cfg or RunConfig()
    t0 = time.time()
    try:
        Xtr, Xte, ytr, yte = spec.split()
        res = method.fit(Xtr, ytr, cfg, diag=cfg.diag)
        if res.ok and method.task == TASK_SUPERVISED:
            try:
                res.artifacts.update(method.fit_extra_artifacts(Xte, res))
                pred = method.predict(Xte, res)
                res.artifacts["y_true"] = np.asarray(yte)
                res.artifacts["y_pred"] = np.asarray(pred)
            except Exception:
                res.error = "predict 失败:\n" + traceback.format_exc(limit=6)
        elif res.ok and method.task == TASK_TIMESERIES:
            pass  # fit 内部已产出 forecast/actual
    except Exception:
        res = MLResult(method_name=method.name, task=method.task,
                       error=traceback.format_exc(limit=10))
    res.target_kind = res.target_kind or spec.target_kind
    res.elapsed = time.time() - t0
    if res.ok:
        try:
            if method.task == TASK_SUPERVISED and "y_pred" in res.artifacts:
                res.metrics.update(method.evaluate(res, spec.X, spec.y))
        except Exception:
            res.metrics = dict(res.metrics)
        # 可选交叉验证（cfg.extras['cv_folds']>0 时才跑，默认零开销 —— P3）
        cv_folds = int(cfg.extras.get("cv_folds", 0) or 0)
        if cv_folds >= 2 and method.task == TASK_SUPERVISED:
            try:
                cv = method.cross_validate(spec.X, spec.y, cfg, cv_folds)
                for k, v in cv.items():
                    if k == "cv_scores":
                        res.diag = dict(res.diag or {})
                        res.diag["cv_scores"] = v
                    else:
                        res.metrics[k] = v
            except Exception:
                pass
    return RunRecord(run_id=_run_id(method.name, cfg.seed),
                     method=method.name, family=method.family,
                     task=method.task, target_kind=spec.target_kind,
                     dataset=spec.meta.get("dataset", ""),
                     pipeline_id=spec.meta.get("pipeline_id", ""),
                     config={"overrides": dict(cfg.overrides),
                             "diag": cfg.diag, "seed": cfg.seed,
                             "extras": dict(cfg.extras)},
                     result=res)


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
                asc = pm in ("rmse", "mae", "mape", "silhouette_deficit")
                df = pd.concat([ok.sort_values(pm, ascending=asc),
                                df[~df["ok"]]], ignore_index=True)
    return df


# ---------------------------------------------------------------- 任务适配
def auto_pages(method: MLMethod, spec: DataSpec) -> list:
    """方法未声明检视页时，按 task 给出兜底页（UI 永远有东西可显示）。"""
    from .contracts import PageSpec
    from ..methods import plots as _p
    t = method.task
    if t == TASK_SUPERVISED:
        if spec.target_kind == "regression":
            return [PageSpec("fit", "拟合效果", "mpl", _p.plot_fit_1d),
                    PageSpec("resid", "残差诊断", "mpl", _p.plot_residual),
                    PageSpec("imp", "特征重要性", "mpl", _p.plot_importance,
                             hint="该方法无系数/重要性输出")]
        return [PageSpec("cm", "混淆矩阵", "mpl", _p.plot_confusion),
                PageSpec("roc", "ROC / PR", "mpl", _p.plot_roc,
                         hint="该方法不输出概率（如 LinearSVC），无法画 ROC"),
                PageSpec("pr", "PR 曲线", "mpl", _p.plot_pr,
                         hint="需要概率输出"),
                PageSpec("imp", "特征重要性", "mpl", _p.plot_importance,
                         hint="该方法无系数/重要性输出")]
    if t == TASK_CLUSTER:
        return [PageSpec("emb", "聚类散点", "mpl", _p.plot_scatter_emb),
                PageSpec("sil", "轮廓系数", "mpl", _p.plot_silhouette)]
    if t == TASK_MANIFOLD:
        return [PageSpec("emb", "降维散点", "mpl", _p.plot_scatter_emb),
                PageSpec("evr", "方差解释", "mpl", _p.plot_explained_variance,
                         hint="仅 PCA/LDA 等线性方法有方差解释概念")]
    if t == TASK_ANOMALY:
        return [PageSpec("score", "异常分数", "mpl", _p.plot_anomaly_score)]
    if t == TASK_TIMESERIES:
        return [PageSpec("fc", "预测曲线", "mpl", _p.plot_forecast),
                PageSpec("res", "残差序列", "mpl", _p.plot_ts_residual)]
    return []
