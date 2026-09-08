# -*- coding: utf-8 -*-
"""优化 runner：ask/tell 循环 + 预算控制 + 逐条回调（P13 直播接入点）。

与 ML 侧 runner 的镜像关系：
    run_one(method, spec, cfg)  -> RunRecord     （批式，一次成型）
    optimize(obj, opt, budget)  -> OptRecord     （增量，逐条产出）

异常纪律（P10）：objective 抛异常 -> 记 status='failed'、分数 +inf 喂回，
循环不中断（对标 MLResult.error 不抛出）。
"""
from __future__ import annotations

import time
import traceback

import numpy as np
import pandas as pd

from .contracts import Budget, Objective, Optimizer, OptRecord


def _run_id(optimizer: str, objective: str, seed: int) -> str:
    return (f"opt-{optimizer}-{objective}"
            f"-{int(time.time() * 1000) % 10_000_000}-{seed}")


def optimize(objective: Objective, optimizer: Optimizer,
             budget: Budget | None = None, cfg: dict | None = None,
             seed: int = 42, on_eval=None, should_stop=None) -> OptRecord:
    """跑一条完整优化轨迹。

    on_eval(record, i)     —— 每次评估后回调（UI 直播 / 进度打印）。
    should_stop() -> bool  —— 外部中止（UI 的"停止"按钮）。
    """
    budget = budget or Budget()
    cfg = cfg or {}
    multi = bool(getattr(objective, "multi", False))
    rec = OptRecord(run_id=_run_id(optimizer.name, objective.name, seed),
                    optimizer=optimizer.name, objective=objective.name,
                    space_desc=objective.space.describe(),
                    budget=budget.to_dict(), seed=seed, multi=multi)
    t0 = time.time()
    rows = []
    best = np.inf
    best_params: dict | None = None
    stall = 0

    try:
        optimizer.setup(objective.space, seed, cfg, budget)
        i = 0
        while i < budget.n_evals:
            if should_stop and should_stop():
                break
            if time.time() - t0 > budget.time_limit:
                break
            if budget.stall and stall >= budget.stall and not multi:
                break
            params = optimizer.ask()
            batch = isinstance(params, list)
            plist = params if batch else [params]
            if batch:
                # 预算边界：评估前截断一代，objective 调用数严格 ≤ n_evals
                room = budget.n_evals - i
                plist = plist[:max(room, 1)]
            results = []
            for p in plist:
                try:
                    s = objective(p)
                    if multi:
                        s = np.asarray(s, float)
                        status = "ok" if np.isfinite(s).all() else "failed"
                        if status == "failed":
                            s = np.full_like(s, np.inf)
                    else:
                        s = float(s)
                        status = "ok" if np.isfinite(s) else "failed"
                        if status == "failed":
                            s = np.inf
                except Exception:
                    s = np.full(objective.n_obj, np.inf) if multi else np.inf
                    status = "failed"
                results.append((p, s, status))
            # 一代全部评完再 tell（种群式算法的语义）
            optimizer.tell(
                [r[0] for r in results] if batch else results[0][0],
                [r[1] for r in results] if batch else results[0][1],
                status=[r[2] for r in results] if batch else results[0][2])
            for p, s, status in results:
                row = dict(p)
                if multi:
                    row.update({f"f{j}": float(s[j]) for j in range(len(s))})
                    row.update(status=status, ts=time.time() - t0)
                else:
                    if s < best:
                        best, best_params, stall = s, p, 0
                    else:
                        stall += 1
                    row.update(score=s, status=status, ts=time.time() - t0,
                               best_so_far=best)
                rows.append(row)
                i += 1
                if on_eval:
                    rec.history = pd.DataFrame(rows)
                    rec.best = None if multi else _mk_best(best_params, best)
                    on_eval(rec, len(rows) - 1)
                if i >= budget.n_evals:
                    break
    except Exception:
        rec.error = traceback.format_exc(limit=10)

    rec.history = pd.DataFrame(rows)
    if multi and len(rec.history):
        rec.pareto = pareto_front(rec.history, objective.n_obj)
    if not multi:
        rec.best = _mk_best(best_params, best)
    rec.elapsed = time.time() - t0
    return rec


def dominates(a, b) -> bool:
    """a 支配 b：a 各目标 ≤ b 且至少一个严格 <（均最小化）。"""
    return bool(np.all(a <= b) and np.any(a < b))


def pareto_front(history: pd.DataFrame, n_obj: int) -> pd.DataFrame:
    """非支配解集（O(n^2) 足够评估历史规模）。"""
    ok = history[history["status"] == "ok"].reset_index(drop=True)
    if ok.empty:
        return ok
    F = ok[[f"f{j}" for j in range(n_obj)]].to_numpy(float)
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if not keep[i]:
            continue
        for j in range(len(F)):
            if i != j and keep[j] and dominates(F[j], F[i]):
                keep[i] = False
                break
    return ok[keep].copy()


def _mk_best(params, score):
    if params is None or not np.isfinite(score):
        return None
    d = dict(params)
    d["score"] = float(score)
    return d


def compare_records(records: list[OptRecord]) -> "pd.DataFrame":
    """横向对比：优化器 × 最终 best / 达到阈值的评估次数（G4 遍历的回报表）。"""
    rows = []
    for r in records:
        ok = r.history[r.history["status"] == "ok"] if len(r.history) else \
            r.history
        if getattr(r, "multi", False):
            row = {"optimizer": r.optimizer, "objective": r.objective,
                   "pareto_n": len(r.pareto) if r.pareto is not None else 0,
                   "n_evals": len(ok), "elapsed_s": round(r.elapsed, 2)}
            if r.error:
                row["error"] = r.error.splitlines()[-1][:120]
            rows.append(row)
            continue
        row = {"optimizer": r.optimizer, "objective": r.objective,
               "best": (round(float(r.best["score"]), 6) if r.best else None),
               "n_evals": len(ok), "elapsed_s": round(r.elapsed, 2)}
        if len(ok) and "best_so_far" in ok.columns:
            bs = ok["best_so_far"].to_numpy()
            # 收敛效率：走完"首次评估 -> 最终 best"进度的 90% 所需评估次数
            # （按 range 定义，正负值安全）
            span = bs[0] - bs[-1]
            row["evals_to_90pct"] = (1 if span <= 1e-12 else
                                     int(np.argmax(bs <= bs[0] - 0.9 * span)) + 1)
            row["median_score"] = round(float(ok["score"].median()), 6)
        if r.error:
            row["error"] = r.error.splitlines()[-1][:120]
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "best" in df:
        df = df.sort_values("best", na_position="last").reset_index(drop=True)
    return df
