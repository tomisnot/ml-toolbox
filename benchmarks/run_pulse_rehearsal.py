# -*- coding: utf-8 -*-
"""实战彩排：微波脉冲参数寻优（假黑盒，模拟原子模拟软件）。

目的不是"跑出好结果"，是**在彩排期把真实工况的坑踩完**：
- 子进程编码（中文/UTF-8/GBK 管道）、科学计数法解析、浮点格式；
- 硬约束：模拟器自己拒绝（退出码 3）-> 工具箱怎么处理；
- 昂贵评估下的预算效率；偶发失败不崩（P10）。

带硬约束时三种失败处理策略的对照（实战选哪个的依据）：
  A censored：约束违反记 failed，不进任何模型（GP 只见可行点）；
  B 惩罚    ：约束违反喂一个大有限分（GP 学"这片高"）；
  C cEI     ：约束违反标 infeasible，额外 GP 分类器建模可行域，
              采集=EI×P(可行)（GP-BO 的 constrain 开关）。

运行：python benchmarks/run_pulse_rehearsal.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "pulse_fake_sim.py")

import numpy as np

from ml_toolbox.core.contracts import ParamSpec
from ml_toolbox.opt import registry as oreg
from ml_toolbox.opt.contracts import Budget, ParamSpace
from ml_toolbox.opt.process import ProcessObjective
from ml_toolbox.opt.runner import optimize

oreg.load_builtin()

SPACE = ParamSpace([
    ParamSpec("amp", "脉冲幅度", "number", 0.7, min=0.1, max=1.5,
              hint="归一化拉比频率"),
    ParamSpec("len", "脉冲时长", "number", 1.43, min=0.5, max=3.0),
    ParamSpec("phase", "相位", "number", 1.57, min=0.0, max=3.14),
])
CMD = f'python "{SIM}" --amp {{amp:.4f}} --len {{len:.4f}} --phase {{phase:.4f}}'
PARSE = r"fidelity_loss[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
# 假黑盒对约束违反的拒绝信号（stderr 里的子串）
CON_SIGNAL = "违反"


class PenaltyProcessObjective(ProcessObjective):
    """策略 B：约束违反 -> 大有限惩罚分（喂 GP）。真崩溃仍抛（failed）。"""
    PENALTY = 10.0

    def evaluate(self, params):
        try:
            return super().evaluate(params)
        except Exception as e:
            if CON_SIGNAL in str(e):
                return self.PENALTY
            raise


def _feasible(r):
    if not r.best:
        return False
    return 0.85 <= r.best["amp"] * r.best["len"] <= 1.15


def run(strategy, optimizer_name, budget_n=50, seed=3):
    t0 = time.time()
    if strategy == "A":
        obj = ProcessObjective(CMD, SPACE, parse=PARSE, name="pulse")
    elif strategy == "B":
        obj = PenaltyProcessObjective(CMD, SPACE, parse=PARSE, name="pulse")
    else:
        obj = ProcessObjective(CMD, SPACE, parse=PARSE, name="pulse",
                               constraint_signal=CON_SIGNAL)
    cfg = {}
    if optimizer_name == "gp_bo":
        cfg = {"n_init": 12, "constrain": strategy == "C"}
    r = optimize(obj, oreg.get(optimizer_name), Budget(n_evals=budget_n),
                 cfg=cfg, seed=seed)
    n_inf = int((r.history["status"] == "infeasible").sum()) if len(r.history) else 0
    n_fail = int((r.history["status"] == "failed").sum()) if len(r.history) else 0
    return r, n_inf, n_fail, time.time() - t0


def main():
    print("=" * 72)
    print("微波脉冲寻优彩排（假黑盒：中文/科学计数法/硬约束θ∈[.85,1.15]/噪声）")
    print("=" * 72)
    rows = []
    for strat, label in [("A", "A censored"), ("B", "B 惩罚"), ("C", "C cEI")]:
        for opt in ("gp_bo", "random_search"):
            r, ninf, nfail, secs = run(strat, opt)
            best = r.best["score"] if r.best else float("nan")
            feas = _feasible(r)
            p = r.best or {}
            rows.append({"strat": strat, "opt": opt, "best": best, "feas": feas,
                         "rec": r})
            print(f"  {label:11s} {opt:14s} best={best:.4f} 可行={feas} "
                  f"不可行={ninf} 失败={nfail} 用时={secs:.0f}s  "
                  f"θ={p.get('amp',0)*p.get('len',0):.3f} "
                  f"phase={p.get('phase',0):.3f} amp={p.get('amp',0):.3f}")

    b_bo = next(x for x in rows if x["strat"] == "B" and x["opt"] == "gp_bo")
    b_rs = next(x for x in rows if x["strat"] == "B" and x["opt"] == "random_search")
    c_bo = next(x for x in rows if x["strat"] == "C" and x["opt"] == "gp_bo")
    c_rs = next(x for x in rows if x["strat"] == "C" and x["opt"] == "random_search")
    a_bo = next(x for x in rows if x["strat"] == "A" and x["opt"] == "gp_bo")
    print("-" * 72)
    print("  真最优参考：θ=1 phase=1.571 amp=0.7 len=1.43，损失≈0")
    ok = True
    # 薄可行带（θ∈[.85,1.15]，~10% 空间）的实战结论：
    #  A censored 会让 GP-BO 退化成随机（可行点太稀，目标 GP 学不动）；
    #  B 自适应惩罚最强（全量点喂 GP，地形信息利用充分）——薄带默认推荐；
    #  C cEI 修复死锁后优于随机，但薄带下不及惩罚（目标 GP 仍只见稀疏可行点）。
    if not b_bo["feas"] or b_bo["best"] > 0.05:
        print(f"❌ 惩罚+GP-BO 未达近优可行解 best={b_bo['best']:.4f}"); ok = False
    if b_bo["best"] > b_rs["best"]:
        print("❌ 惩罚+GP-BO 未优于随机"); ok = False
    if c_bo["best"] > c_rs["best"] + 1e-9:
        print("❌ cEI+GP-BO 未优于随机（死锁修复未生效？）"); ok = False
    if ok:
        print("✅ 彩排通过。带硬约束黑盒的策略结论（薄可行带）：")
        print(f"   惩罚 best={b_bo['best']:.4f} < cEI best={c_bo['best']:.4f}"
              f" < censored best={a_bo['best']:.4f}（censored≈随机=退化）")
        print("   → 薄可行带默认用 on_infeasible='penalize'；可行占比中等时用 cEI。")
        from ml_toolbox.opt import persistence
        persistence.save_record(b_bo["rec"])
        print("  最优轨迹已存 runs/", b_bo["rec"].run_id)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
