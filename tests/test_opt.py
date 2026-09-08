# -*- coding: utf-8 -*-
"""优化框架回归测试（阶段 0+）：契约 + runner + 持久化 + 引擎。

运行：python tests/test_opt.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ml_toolbox.core.contracts import ParamSpec
from ml_toolbox.opt import registry
from ml_toolbox.opt.contracts import ParamSpace, Budget, make_objective
from ml_toolbox.opt.runner import optimize, compare_records

registry.load_builtin()

PASS = 0
FAIL = []


def check(name, fn):
    global PASS
    try:
        fn()
        PASS += 1
        print(f"  ✓ {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  ✗ {name}: {type(e).__name__}: {str(e)[:200]}")


def _mixed_space():
    return ParamSpace([
        ParamSpec("x", "x", "number", 0.0, min=-5, max=5),
        ParamSpec("lr", "lr", "number", 1e-2, min=1e-5, max=1.0, log=True),
        ParamSpec("k", "k", "int", 5, min=1, max=20),
        ParamSpec("ker", "kernel", "select", "rbf",
                  choices=["rbf", "poly", "lin"]),
        ParamSpec("sw", "weight", "bool", False),
        ParamSpec("hidden", "hidden", "text", "64,32"),   # 固定透传
    ])


def test_space_vector_roundtrip():
    """向量化往返：dict -> [0,1]^d -> dict 语义等价（含 log/类别/bool）。"""
    sp = _mixed_space()
    assert sp.dim == 5 and sp.keys == ["x", "lr", "k", "ker", "sw"]
    rng = np.random.RandomState(0)
    for _ in range(30):
        p = sp.sample(rng)
        v = sp.to_vector(p)
        assert v.shape == (5,) and (v >= 0).all() and (v <= 1).all()
        p2 = sp.from_vector(v)
        assert p2["hidden"] == "64,32"          # 固定项透传
        assert p2["ker"] == p["ker"]
        assert p2["sw"] == p["sw"]
        assert abs(p2["x"] - p["x"]) < 1e-6
        assert abs(p2["lr"] - p["lr"]) < 1e-6   # log 往返（from_vector 已 clip）
        assert abs(p2["k"] - p["k"]) <= 1      # int 量化误差


def test_space_clip_out_of_bounds():
    """越界向量自动投影回界内（优化器给出界候选不崩）。"""
    sp = _mixed_space()
    p = sp.from_vector([9.0, -3.0, 0.5, 0.5, 0.5])
    assert -5.0 <= p["x"] <= 5.0 and 1e-5 <= p["lr"] <= 1.0
    assert p["ker"] in ("rbf", "poly", "lin")


def test_grid_points_coverage():
    sp = ParamSpace([ParamSpec("a", "a", "number", 0.0, min=0, max=1),
                     ParamSpec("b", "b", "select", "x", choices=["x", "y"])])
    pts = sp.grid_points(3)
    assert len(pts) == 6, "3 格点 × 2 类别 = 6"
    assert all(set(q) == {"a", "b"} for q in pts)
    assert {q["b"] for q in pts} == {"x", "y"}     # 类别全覆盖


def test_baseline_runs():
    """随机/网格在二次函数上跑通，best 逼近最优。"""
    sp = ParamSpace([ParamSpec("x", "x", "number", 0.0, min=-5, max=5)])
    obj = make_objective(lambda q: (q["x"] - 2.3) ** 2, sp, name="quad")
    for nm in ("random_search", "grid_search"):
        r = optimize(obj, registry.get(nm), Budget(n_evals=40), seed=1)
        assert r.error is None, r.error
        assert r.best and r.best["score"] < 1.0, f"{nm} best={r.best}"
        assert len(r.history) == 40
        assert (r.history["best_so_far"].diff().dropna() <= 1e-12).all(), \
            "best_so_far 必须单调不增"


def test_failure_isolation():
    """P10：objective 抛异常 -> status=failed + inf，循环不中断。"""
    sp = _mixed_space()

    def bad(q):
        if q["x"] > 0:
            raise RuntimeError("sim fail")
        return q["x"] ** 2
    r = optimize(make_objective(bad, sp, name="bad"),
                 registry.get("random_search"), Budget(n_evals=30), seed=2)
    assert r.error is None
    n_fail = int((r.history["status"] == "failed").sum())
    assert 0 < n_fail < 30, "应有失败也有成功"
    assert np.isinf(r.history.loc[r.history["status"] == "failed",
                                 "score"]).all()
    assert r.best["score"] < 25   # 成功评估的 x^2 正常参与 best


def test_budget_stall_and_time():
    sp = ParamSpace([ParamSpec("x", "x", "number", 0.0, min=-5, max=5)])
    obj = make_objective(lambda q: q["x"] ** 2, sp, name="q")
    r = optimize(obj, registry.get("random_search"),
                 Budget(n_evals=500, stall=8), seed=3)
    assert len(r.history) < 500, "stall 应提前停"
    # 时间限：目标函数带 sleep（模拟昂贵评估），否则微秒级评估跑满 500 次
    import time as _t
    slow = make_objective(lambda q: (_t.sleep(0.005), q["x"] ** 2)[1],
                          sp, name="slow")
    r2 = optimize(slow, registry.get("random_search"),
                  Budget(n_evals=500, time_limit=0.1), seed=4)
    assert len(r2.history) < 500, "时间限应提前停"


def test_should_stop_callback():
    """外部中止（UI 停止按钮的钩子）。"""
    sp = ParamSpace([ParamSpec("x", "x", "number", 0.0, min=-5, max=5)])
    obj = make_objective(lambda q: q["x"] ** 2, sp, name="q")
    n = [0]
    r = optimize(obj, registry.get("random_search"), Budget(n_evals=100),
                 seed=5, on_eval=lambda rec, i: n.__setitem__(0, i + 1),
                 should_stop=lambda: n[0] >= 7)
    assert 7 <= len(r.history) <= 9, f"应在 7 次后尽快停，实际 {len(r.history)}"


def test_on_eval_stream():
    """P13 直播：on_eval 逐条回调，record 携带截至当前的历史。"""
    sp = ParamSpace([ParamSpec("x", "x", "number", 0.0, min=-5, max=5)])
    obj = make_objective(lambda q: q["x"] ** 2, sp, name="q")
    lens = []
    optimize(obj, registry.get("random_search"), Budget(n_evals=12), seed=6,
             on_eval=lambda rec, i: lens.append(len(rec.history)))
    assert lens == list(range(1, 13)), "回调次数与历史长度须逐条递增"


def test_maximize_direction():
    """minimize=False 的目标：runner 统一化后 best 语义仍是"最好"。"""
    sp = ParamSpace([ParamSpec("x", "x", "number", 0.0, min=-5, max=5)])
    obj = make_objective(lambda q: -(q["x"] - 1.0) ** 2, sp, name="neg",
                         minimize=False)
    r = optimize(obj, registry.get("random_search"), Budget(n_evals=60),
                 seed=7)
    assert r.best["x"] is not None
    assert abs(r.best["x"] - 1.0) < 2.0   # 最大化峰值附近


def test_compare_records():
    sp = ParamSpace([ParamSpec("x", "x", "number", 0.0, min=-5, max=5)])
    obj = make_objective(lambda q: (q["x"] + 1.0) ** 2, sp, name="quad")
    recs = [optimize(obj, registry.get(nm), Budget(n_evals=30), seed=8)
            for nm in ("random_search", "grid_search")]
    df = compare_records(recs)
    assert df["best"].is_monotonic_increasing, "compare 表按 best 升序"
    assert {"best", "n_evals", "evals_to_90pct"} <= set(df.columns)


def test_persistence_roundtrip():
    from ml_toolbox.opt import persistence
    sp = ParamSpace([ParamSpec("x", "x", "number", 0.0, min=-5, max=5),
                     ParamSpec("ker", "k", "select", "a", choices=["a", "b"])])
    obj = make_objective(lambda q: q["x"] ** 2, sp, name="pers")
    r = optimize(obj, registry.get("random_search"), Budget(n_evals=20),
                 seed=9)
    d = persistence.save_record(r)
    assert os.path.exists(os.path.join(d, "opt_record.json"))
    assert os.path.exists(os.path.join(d, "history.csv"))
    back = persistence.load_record(r.run_id)
    assert len(back.history) == 20
    assert abs(back.best["score"] - r.best["score"]) < 1e-12
    assert back.space_desc == r.space_desc
    # 与 ML 侧持久化互不混淆
    ids = {x["run_id"] for x in persistence.list_records()}
    assert r.run_id in ids and r.run_id.startswith("opt-")


def test_history_to_dataset():
    """接缝3 物理载体：评估历史 -> Dataset（参数为 X，分数为 y）。"""
    sp = ParamSpace([ParamSpec("x", "x", "number", 0.0, min=-5, max=5),
                     ParamSpec("hidden", "h", "text", "64")])   # 固定项不进 X
    obj = make_objective(lambda q: (q["x"] - 1) ** 2, sp, name="ds")
    r = optimize(obj, registry.get("random_search"), Budget(n_evals=25),
                 seed=10)
    ds = r.to_dataset()
    assert ds is not None and ds.target == "target"
    assert list(ds.frame.columns) == ["x", "target"]
    assert len(ds.frame) == 25


def main():
    tests = [(k[5:], v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"优化框架测试 {len(tests)} 项")
    for name, fn in tests:
        check(name, fn)
    print(f"\n{PASS} passed, {len(FAIL)} failed")
    if FAIL:
        for n, e in FAIL:
            print(f"  FAILED {n}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
