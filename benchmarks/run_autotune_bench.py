# -*- coding: utf-8 -*-
"""阶段4 数模验收基准：AutoTuner 实战（接缝2）。

场景（数模标准流程）：给定真实数据集，对多个候选方法自动调参，
对比"默认参数 CV 分"与"GP-BO 调参后 CV 分"。

验收判据（docs/优化定位.md §9）：调参结果 ≥ 手工默认（每个方法都不显著劣化，
且至少一个方法显著提升）。

运行：python benchmarks/run_autotune_bench.py
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ml_toolbox.core import registry as ml
from ml_toolbox.core import runner
from ml_toolbox.core.dataset import Dataset
from ml_toolbox.core.pipeline import Pipeline
from ml_toolbox.core.contracts import RunConfig
from ml_toolbox.opt import registry as oreg
from ml_toolbox.opt.contracts import Budget
from ml_toolbox.opt.bridges import autotune

ml.load_builtin()
oreg.load_builtin()

METHODS = ["svc", "knn", "random_forest", "logistic", "gaussian_nb"]
CV = 3
N_EVALS = 20


def load_data():
    # digits：多分类且默认参数明显次优（SVC 默认 C=1 欠拟合，调参有真实增益）
    from sklearn.datasets import load_digits
    d = load_digits(as_frame=True)
    return Dataset(d.frame.rename(columns={"target": "target"}),
                   name="digits", target="target")


def main():
    ds = load_data()
    spec = Pipeline.default().run(ds)
    rows = []
    for name in METHODS:
        # 基线：默认参数 CV
        base = runner.run_one(ml.get(name), spec,
                              RunConfig(extras={"cv_folds": CV}))
        base_f1 = base.result.metrics.get("cv_f1_mean",
                                          base.result.metrics.get("f1", np.nan))
        # 调参：GP-BO 搜超参
        t0 = time.time()
        rec, best, val = autotune(name, spec.X, spec.y,
                                  optimizer=oreg.get("gp_bo"),
                                  budget=Budget(n_evals=N_EVALS),
                                  cv_folds=CV, seed=7)
        if rec.error:
            print(f"  {name:14s} 调参失败: {rec.error.splitlines()[-1][:80]}")
            continue
        rows.append({"method": name, "default_cv_f1": round(float(base_f1), 4),
                     "tuned_cv_f1": round(float(val), 4),
                     "gain": round(float(val) - float(base_f1), 4),
                     "best_params": {k: v for k, v in best.items()
                                     if k != "score"},
                     "secs": round(time.time() - t0, 1)})
        print(f"  {name:14s} default={base_f1:.4f}  tuned={val:.4f}  "
              f"gain={val - base_f1:+.4f}")

    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "opt_stage4_autotune.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print("结果表 ->", out)

    # 验收：无显著劣化（-0.01 容差）且至少一个提升 >0.005
    bad = df[df.gain < -0.01]
    wins = (df.gain > 0.005).sum()
    if len(bad):
        print("❌ 调参显著劣化的方法:", list(bad.method))
        sys.exit(1)
    if wins == 0:
        print("❌ 无任何方法从调参获益")
        sys.exit(1)
    print(f"✅ 阶段4 验收：{wins}/{len(df)} 方法调参提升，无显著劣化")


if __name__ == "__main__":
    main()
