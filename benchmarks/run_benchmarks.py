# -*- coding: utf-8 -*-
"""数学建模基准实战：用经典建模赛题数据检验工具箱"遍历好不好使"。

场景（对应数模常见题型）：
  B1 分类     Iris（小样本多类）+ Wine（中等维数）
  B2 回归     California Housing（多特征连续预测）/ 或合成
  B3 聚类     Mallards-like blobs（无监督分群）
  B4 异常     信用卡式偏置异常
  B5 时序     Airline 客运量（强季节性，statsmodels 内置）
  B6 降维可视化 高维数据 2D 嵌入对比

判据：最优方法是否落在已知答案附近（如 Iris 应 >0.95、时序应显著优于朴素法）。
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ml_toolbox.core import registry, runner
from ml_toolbox.core.dataset import Dataset
from ml_toolbox.core.pipeline import Pipeline
from ml_toolbox.core.contracts import RunConfig

registry.load_builtin()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def bench(name, ds, task, cfg=None, pipe=None):
    cfg = cfg or RunConfig()
    spec = (pipe or Pipeline.default()).run(ds)
    methods = [m for m in registry.all_methods()
               if m.task == task and m.can_handle(spec)]
    print(f"\n### {name}  (n={len(ds.frame)}, task={task}, {len(methods)} 方法)")
    t0 = time.time()
    recs = runner.run_batch([m.name for m in methods], spec, cfg)
    table = runner.compare_table(recs)
    table.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    fails = [r.method for r in recs if not r.result.ok]
    ok = table[table["ok"]] if "ok" in table else table
    if not ok.empty:
        pm = ok["primary"].iloc[0] if "primary" in ok else None
        if pm and pm in ok:
            best = ok.iloc[0]
            worst = ok.iloc[-1]
            print(f"  最优: {best['method']:18s} {pm}={best[pm]}")
            print(f"  最差: {worst['method']:18s} {pm}={worst[pm]}")
    if fails:
        print(f"  失败 {len(fails)}: {fails}")
    print(f"  遍历耗时 {time.time()-t0:.1f}s")
    return table, fails


def main():
    all_fails = {}

    # B1a Iris：sklearn 内置，小样本 3 类
    from sklearn.datasets import load_iris
    d = load_iris(as_frame=True)
    ds = Dataset(d.frame.rename(columns={"target": "target"}), name="iris",
                 target="target")
    all_fails["iris"] = bench("B1a_iris", ds, "supervised")[1]

    # B1b Wine
    from sklearn.datasets import load_wine
    d = load_wine(as_frame=True)
    ds = Dataset(d.frame.rename(columns={"target": "target"}), name="wine",
                 target="target")
    all_fails["wine"] = bench("B1b_wine", ds, "supervised")[1]

    # B2 回归：California housing（真实数模回归题；抽样 3000 行控时，
    # 且 GPR 等 O(n^3) 方法可承受）
    try:
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing()
        df = pd.DataFrame(d.data, columns=d.feature_names)
        df["target"] = d.target
        df = df.sample(3000, random_state=0).reset_index(drop=True)
        all_fails["housing"] = bench("B2_housing",
                                     Dataset(df, name="housing", target="target"),
                                     "supervised")[1]
    except Exception as e:
        print("B2 跳过（数据下载失败）:", e)

    # B3 聚类：blobs
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=5, random_state=3)
    all_fails["blobs"] = bench("B3_blobs", Dataset.from_arrays(X, None, "blobs"),
                               "cluster")[1]

    # B4 异常：合成偏置异常
    rng = np.random.RandomState(7)
    X = rng.randn(400, 5)
    X[:30] += 4
    lab = np.zeros(400); lab[:30] = 1
    all_fails["anom"] = bench("B4_anomaly",
                              Dataset.from_arrays(X, lab, "anom"), "anomaly")[1]

    # B5 时序：AirPassengers（强季节性）。优先 statsmodels 内置，失败用合成季节性序列
    vals = None
    try:
        import statsmodels.api as sm
        r = sm.datasets.get_rdataset("AirPassengers")
        vals = pd.to_numeric(r.data.iloc[:, -1], errors="coerce").dropna().to_numpy(float)
    except Exception:
        vals = None
    if vals is None or len(vals) < 24:
        t = np.arange(96, dtype=float)
        vals = 200 + 2.0 * t + 40 * np.sin(2 * np.pi * t / 12) \
            + np.random.RandomState(2).randn(96) * 5
    ts_ds = Dataset.from_arrays(np.asarray(vals).reshape(-1, 1),
                                vals, "ts")
    ts_ds.frame = ts_ds.frame.rename(columns={"f0": "t"})
    ts_ds.time_col = "t"          # 时间列不参与特征，仅按序切分
    pipe_ts = Pipeline(steps=[], time_split=True, test_size=0.125)
    all_fails["ts"] = bench("B5_timeseries", ts_ds, "timeseries",
                            cfg=RunConfig(extras={"horizon": 12}),
                            pipe=pipe_ts)[1]

    # B6 降维：digits 高维
    from sklearn.datasets import load_digits
    d = load_digits(as_frame=True)
    ds = Dataset(d.frame.rename(columns={"target": "target"}), name="digits",
                 target="target")
    all_fails["digits"] = bench("B6_digits_manifold", ds, "manifold")[1]

    print("\n================ 基准汇总 ================")
    total = sum(len(v) for v in all_fails.values())
    for k, v in all_fails.items():
        print(f"  {k}: {len(v)} 失败 {v if v else ''}")
    print(f"结果表已写入 {OUT}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
