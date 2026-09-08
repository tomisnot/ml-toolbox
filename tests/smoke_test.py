# -*- coding: utf-8 -*-
"""端到端冒烟测试：数据 -> 管道 -> 遍历全部注册方法 -> 对比表 -> 持久化。

不依赖 Qt，纯核心链路。运行：python tests/smoke_test.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ml_toolbox.core import registry, runner, persistence
from ml_toolbox.core.dataset import Dataset
from ml_toolbox.core.pipeline import Pipeline
from ml_toolbox.core.contracts import RunConfig

registry.load_builtin()


def make_regression():
    from sklearn.datasets import make_regression as _mr
    X, y = _mr(n_samples=200, n_features=8, noise=15, random_state=0)
    return Dataset.from_arrays(X, y, name="synth_reg")


def make_classification():
    from sklearn.datasets import make_classification as _mc
    X, y = _mc(n_samples=220, n_features=10, n_informative=5,
               n_classes=3, random_state=0)
    return Dataset.from_arrays(X, y, name="synth_cls")


def make_cluster():
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=200, centers=4, random_state=0)
    return Dataset.from_arrays(X, None, name="synth_cluster")


def make_anomaly():
    rng = np.random.RandomState(0)
    X = rng.randn(180, 4)
    X[:15] += 5  # 注入异常
    lab = np.zeros(180); lab[:15] = 1
    return Dataset.from_arrays(X, lab, name="synth_anom")


def make_timeseries():
    t = np.arange(120, dtype=float)
    y = 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.RandomState(1).randn(120)
    return Dataset.from_arrays(y.reshape(-1, 1), y, name="synth_ts")


def check(name, ds, task_filter, cfg=None):
    cfg = cfg or RunConfig()
    pipe = Pipeline.default()
    spec = pipe.run(ds)
    methods = [m for m in registry.all_methods() if m.task == task_filter
               and m.can_handle(spec)]
    print(f"\n=== {name} | {len(methods)} 方法 | task={task_filter} ===")
    recs = runner.run_batch([m.name for m in methods], spec, cfg)
    table = runner.compare_table(recs)
    fails = [r.method for r in recs if not r.result.ok]
    for r in recs:
        flag = "OK " if r.result.ok else "ERR"
        pm = r.result.primary_metric
        val = r.result.metrics.get(pm, float("nan"))
        print(f"  [{flag}] {r.method:18s} {pm}={val if isinstance(val,float) else '-'}")
        if not r.result.ok:
            print("      ->", (r.result.error or "").strip().splitlines()[-1][:160])
    # 检视页可构建性检查
    for r in recs:
        if r.result.ok:
            m = registry.get(r.method)
            pages = m.inspect_pages(cfg) or runner.auto_pages(m, spec)
            assert pages, f"{r.method} 无检视页"
    return recs, fails


def main():
    all_fails = {}
    all_fails["reg"] = check("回归", make_regression(), "supervised")[1]
    all_fails["cls"] = check("分类", make_classification(), "supervised")[1]
    all_fails["clu"] = check("聚类", make_cluster(), "cluster")[1]
    all_fails["ano"] = check("异常", make_anomaly(), "anomaly")[1]
    all_fails["ts"] = check("时序", make_timeseries(), "timeseries")[1]

    # 降维单独测（用带标签数据，LDA 投影需要 y）
    all_fails["man"] = check("降维", make_classification(), "manifold")[1]

    # 持久化 round-trip
    recs, _ = check("持久化(回归子集)", make_regression(), "supervised")
    ok = [r for r in recs if r.result.ok][:2]
    for r in ok:
        d = persistence.save_record(r)
        loaded = persistence.load_record(r.run_id)
        assert loaded["run_id"] == r.run_id
        print(f"  持久化 OK -> {os.path.basename(d)}")

    total_fail = sum(len(v) for v in all_fails.values())
    print("\n================ 汇总 ================")
    for k, v in all_fails.items():
        print(f"  {k}: {len(v)} 失败 {v if v else ''}")
    print(f"注册方法总数: {len(registry.names())}")
    print(f" families: {registry.families()}")
    if total_fail:
        print(f"\n❌ {total_fail} 个方法失败")
        sys.exit(1)
    print("\n✅ 全部方法通过")


if __name__ == "__main__":
    main()
