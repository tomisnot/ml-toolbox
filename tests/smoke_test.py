# -*- coding: utf-8 -*-
"""端到端冒烟测试：数据 -> 管道 -> 遍历全部注册方法 -> 对比表 -> 持久化。

不依赖 Qt，纯核心链路。运行：python tests/smoke_test.py

快/慢分层（用户要求门 <1min）：默认 **fast**——小数据集 + 跳过 JIT 冷启动大户
（umap/tsne，numba 首次编译 30s+，其"能否出图"由 test_regressions 的
pages_declared_or_auto 覆盖）。设 `MLTB_SLOW=1` 跑全量（含慢降维 + 大数据集）。
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

SLOW = os.environ.get("MLTB_SLOW") == "1"
# fast：小样本够验证"跑通 + 出指标 + 有检视页"；slow：更接近真实规模
_N = 240 if SLOW else 60
_SLOW_METHODS = ("umap", "tsne")        # JIT 冷启动大户，仅 slow 层跑
# fast：统一压迭代预算（resolve() 只认各方法 schema 里的键，未知键自动忽略）。
# members 换成轻量双任务方法：stacking/voting 的基学习器不吃外层 overrides，
# 默认 3×300 棵树 ×4 折是最大耗时户。
_FAST_CFG = RunConfig(overrides={"n_estimators": 30, "max_iter": 100,
                                 "epochs": 15, "members": "knn,svc",
                                 "voting": "hard"})   # svc 无概率，soft 会炸


def make_regression():
    from sklearn.datasets import make_regression as _mr
    X, y = _mr(n_samples=_N, n_features=6, noise=15, random_state=0)
    return Dataset.from_arrays(X, y, name="synth_reg")


def make_classification():
    from sklearn.datasets import make_classification as _mc
    X, y = _mc(n_samples=_N, n_features=8, n_informative=5,
               n_classes=3, random_state=0)
    return Dataset.from_arrays(X, y, name="synth_cls")


def make_cluster():
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=_N, centers=4, random_state=0)
    return Dataset.from_arrays(X, None, name="synth_cluster")


def make_anomaly():
    rng = np.random.RandomState(0)
    n = max(_N, 60)
    X = rng.randn(n, 4)
    X[:15] += 5  # 注入异常
    lab = np.zeros(n); lab[:15] = 1
    return Dataset.from_arrays(X, lab, name="synth_anom")


def make_timeseries():
    n = 240 if SLOW else 80
    t = np.arange(n, dtype=float)
    y = 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.RandomState(1).randn(n)
    return Dataset.from_arrays(y.reshape(-1, 1), y, name="synth_ts")


def check(name, ds, task_filter, cfg=None):
    cfg = cfg or RunConfig()
    pipe = Pipeline.default()
    spec = pipe.run(ds)
    methods = [m for m in registry.all_methods() if m.task == task_filter
               and m.can_handle(spec)]
    # 环境豁免（C5）：受限环境（沙箱/CI）某些方法因并行核探测不可用，
    # MLTB_SKIP_METHODS=umap,lightgbm 把它们移出并计 SKIP（不记失败）。
    from ml_toolbox.core.parallel import skip_methods
    skip = set(skip_methods())
    if not SLOW:
        skip |= set(_SLOW_METHODS)       # fast 层：JIT 冷启动大户移入 slow
    skipped = [m.name for m in methods if m.name in skip]
    methods = [m for m in methods if m.name not in skip]
    print(f"\n=== {name} | {len(methods)} 方法 | task={task_filter} ===")
    if skipped:
        print(f"  SKIP：{', '.join(skipped)}")
    recs = runner.run_batch([m.name for m in methods], spec, cfg)
    table = runner.compare_table(recs)
    fails = [r.method for r in recs if not r.result.ok]
    for r in recs:
        flag = "OK " if r.result.ok else "ERR"
        pm = r.result.primary_metric
        val = r.result.metrics.get(pm, float("nan"))
        print(f"  [{flag}] {r.method:18s} {pm}="
              f"{val if isinstance(val,float) else '-'}"
              f"  ({r.result.elapsed:.1f}s)")
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
    cfg = None if SLOW else _FAST_CFG
    all_fails = {}
    reg_recs, all_fails["reg"] = check("回归", make_regression(), "supervised", cfg)
    all_fails["cls"] = check("分类", make_classification(), "supervised", cfg)[1]
    all_fails["clu"] = check("聚类", make_cluster(), "cluster", cfg)[1]
    all_fails["ano"] = check("异常", make_anomaly(), "anomaly", cfg)[1]
    all_fails["ts"] = check("时序", make_timeseries(), "timeseries", cfg)[1]

    # 降维单独测（用带标签数据，LDA 投影需要 y）
    all_fails["man"] = check("降维", make_classification(), "manifold", cfg)[1]

    # 持久化 round-trip：复用回归块的运行记录（此前重跑 22 方法 = 纯浪费）
    ok = [r for r in reg_recs if r.result.ok][:2]
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
