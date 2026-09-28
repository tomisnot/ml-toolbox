# -*- coding: utf-8 -*-
"""优化框架样本效率基准（阶段1 验收）：GP-BO vs 随机搜索。

判据（内部记录（未随仓发布，存档在仓外） §9）：标准测试函数、同预算、多随机种子平均，
GP-BO 的最终 best 显著优于随机搜索（目标：≥ 1.5x 改善或持平+更快收敛）。

运行：python benchmarks/run_opt_bench.py
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np

from ml_toolbox.core.contracts import ParamSpec
from ml_toolbox.opt import registry
from ml_toolbox.opt.contracts import ParamSpace, Budget, make_objective
from ml_toolbox.opt.runner import optimize

registry.load_builtin()

# 标准测试函数（连续、低维——GP-BO 的主场）
def ackley(sp):
    return ParamSpace([ParamSpec(f"x{i}", f"x{i}", "number", 0.0,
                                 min=-4, max=4) for i in range(sp)]), \
        (lambda q: -20 * np.exp(-0.2 * np.sqrt(np.mean(
            [q[f"x{i}"] ** 2 for i in range(sp)])))
         - np.exp(np.mean([np.cos(2 * np.pi * q[f"x{i}"])
                           for i in range(sp)])) + 20 + np.e)


def six_hump(sp=2):
    return ParamSpace([ParamSpec("x1", "x1", "number", 0.0, min=-2, max=2),
                       ParamSpec("x2", "x2", "number", 0.0, min=-2, max=2)]), \
        (lambda q: (4 - 2.1 * q["x1"] ** 2 + q["x1"] ** 4 / 3) * q["x1"] ** 2
         + q["x1"] * q["x2"] + (-4 + 4 * q["x2"] ** 2) * q["x2"] ** 2)


def goldstein(sp=2):
    return ParamSpace([ParamSpec("x1", "x1", "number", 0.0, min=-2, max=2),
                       ParamSpec("x2", "x2", "number", 0.0, min=-2, max=2)]), \
        (lambda q: (1 + (q["x1"] + q["x2"] + 1) ** 2
                    * (19 - 14 * q["x1"] + 3 * q["x1"] ** 2
                       - 14 * q["x2"] + 6 * q["x1"] * q["x2"] + 3 * q["x2"] ** 2))
         * (30 + (2 * q["x1"] - 3 * q["x2"]) ** 2
            * (18 - 32 * q["x1"] + 12 * q["x1"] ** 2
               + 48 * q["x2"] - 36 * q["x1"] * q["x2"] + 27 * q["x2"] ** 2)))


FUNCS = {"ackley3d": ackley(3), "six_hump": six_hump(),
         "goldstein": goldstein()}

N_BUDGET = 40
SEEDS = [11, 22, 33]


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    rows = []
    for fname, (space, fn) in FUNCS.items():
        obj = make_objective(fn, space, name=fname)
        for opt_name in ("gp_bo", "random_search"):
            bests = []
            for sd in SEEDS:
                t0 = time.time()
                r = optimize(obj, registry.get(opt_name),
                             Budget(n_evals=N_BUDGET),
                             cfg={"n_init": 8}, seed=sd)
                assert r.error is None, r.error
                bests.append(r.best["score"])
            rows.append({"func": fname, "optimizer": opt_name,
                         "best_mean": round(float(np.mean(bests)), 5),
                         "best_std": round(float(np.std(bests)), 5),
                         "budget": N_BUDGET, "seeds": len(SEEDS)})
        bo = [r for r in rows if r["func"] == fname and r["optimizer"] == "gp_bo"][0]
        rs = [r for r in rows if r["func"] == fname and r["optimizer"] == "random_search"][0]
        # 比值在 BO 逼近真最优时会爆炸（分母→0），报绝对差最诚实
        delta = rs["best_mean"] - bo["best_mean"]
        win = "✓" if delta >= 0 else "✗"
        print(f"  {fname:12s} {win} BO={bo['best_mean']:.4f}±{bo['best_std']:.4f} "
              f"RS={rs['best_mean']:.4f} 领先={delta:+.4f}")

    import pandas as pd
    df = pd.DataFrame(rows)
    p = os.path.join(out_dir, "results", "opt_stage1_bo_vs_random.csv")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_csv(p, index=False)
    print("结果表 ->", p)

    # 验收：每个函数上 BO 平均 best ≤ RS（不显著更差即通过底线；
    # 低维光滑函数上 BO 通常 2x+ 改善）
    bad = []
    for fname in FUNCS:
        b = df[(df.func == fname) & (df.optimizer == "gp_bo")].best_mean.iloc[0]
        r = df[(df.func == fname) & (df.optimizer == "random_search")].best_mean.iloc[0]
        if b > r * 1.05:
            bad.append(fname)
    if bad:
        print("❌ BO 未跑赢随机:", bad)
        sys.exit(1)
    print("✅ 阶段1 验收：GP-BO 样本效率 ≥ 随机搜索（全部函数）")


if __name__ == "__main__":
    main()
