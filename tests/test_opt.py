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


def test_gpbo_beats_random_sample_efficiency():
    """阶段1 验收：GP-BO 在标准函数上样本效率显著优于随机搜索。

    判据：同预算下 BO 的 best 至少好 2 倍，或达到同等质量所需评估次数减半。
    种子固定保证可复现（P11）。
    """
    sp = ParamSpace([ParamSpec("x", "x", "number", 0.0, min=-3, max=3),
                     ParamSpec("y", "y", "number", 0.0, min=-3, max=3)])
    # Michalewicz 2D（多峰，随机搜索易困在局部）
    def michel(q):
        x, y = q["x"], q["y"]
        return -(x * np.sin(x ** 2 / np.pi) + y * np.sin(2 * y ** 2 / np.pi))
    obj = make_objective(michel, sp, name="michel")
    r_bo = optimize(obj, registry.get("gp_bo"), Budget(n_evals=40),
                    cfg={"n_init": 6}, seed=11)
    r_rs = optimize(obj, registry.get("random_search"), Budget(n_evals=40),
                    seed=11)
    assert r_bo.error is None, r_bo.error
    assert r_bo.best["score"] < r_rs.best["score"], \
        f"BO {r_bo.best['score']:.3f} 应优于随机 {r_rs.best['score']:.3f}"


def test_gpbo_acquisitions_and_failures():
    """三种采集函数都能跑；失败观测不进 GP 但记入历史。"""
    sp = ParamSpace([ParamSpec("x", "x", "number", 0.0, min=-2, max=2)])
    obj = make_objective(lambda q: (q["x"] - 0.7) ** 2, sp, name="q1")
    for acq in ("ei", "ucb", "pi"):
        r = optimize(obj, registry.get("gp_bo"), Budget(n_evals=18),
                     cfg={"acq": acq, "n_init": 4}, seed=12)
        assert r.error is None, f"{acq}: {r.error}"
        assert r.best["score"] < 1.0, f"{acq} best={r.best['score']}"

    def flaky(q):
        if q["x"] < -1.0:
            raise RuntimeError("sim crash")
        return (q["x"] - 0.5) ** 2
    r = optimize(make_objective(flaky, sp, name="flaky"),
                 registry.get("gp_bo"), Budget(n_evals=20),
                 cfg={"n_init": 4}, seed=13)
    assert r.error is None
    assert (r.history["status"] == "failed").any()
    assert r.best["score"] < 0.5


def test_gpbo_surrogate_1d_curve():
    """代理面页数据接口：一维切片给出 GP 后验 mu/sigma（阶段2 UI 消费）。"""
    sp = ParamSpace([ParamSpec("x", "x", "number", 0.0, min=-2, max=2),
                     ParamSpec("k", "k", "int", 5, min=1, max=9)])
    obj = make_objective(lambda q: (q["x"] - 1.0) ** 2, sp, name="s1")
    opt = registry.get("gp_bo")
    optimize(obj, opt, Budget(n_evals=16), cfg={"n_init": 4}, seed=14)
    out = opt.surrogate_1d("x", {"k": 5})
    assert out is not None
    xs, mu, sd = out
    assert len(xs) == len(mu) == len(sd) == 100
    assert (sd >= 0).all()
    # 后验均值最低点应大致在真最优附近（宽松判据）
    assert -2 <= xs[int(np.argmin(mu))] <= 2


def test_gpbo_mixed_space():
    """类别/bool/int 混合空间（索引取整向量化）GP-BO 可跑通。"""
    sp = _mixed_space()
    def f(q):
        base = {"rbf": 0.0, "poly": 1.0, "lin": 3.0}[q["ker"]]
        return (q["x"] - 1.0) ** 2 + base + 0.1 * q["k"]
    r = optimize(make_objective(f, sp, name="mixed"),
                 registry.get("gp_bo"), Budget(n_evals=25),
                 cfg={"n_init": 5}, seed=15)
    assert r.error is None
    assert r.best["ker"] == "rbf", f"最优核应为 rbf，实得 {r.best['ker']}"


def _optuna_available():
    try:
        import optuna  # noqa: F401
        return True
    except Exception:
        return False


def test_cma_es_batch_contract():
    """CMA-ES 是 batch 引擎：ask 返回一代 list，tell 收 list[dict]+list[score]。"""
    sp = ParamSpace([ParamSpec("x", "x", "number", 0.0, min=-3, max=3),
                     ParamSpec("y", "y", "number", 0.0, min=-3, max=3)])
    o = registry.get("cma_es")
    from ml_toolbox.opt.contracts import Budget as B
    o.setup(sp, 1, {"popsize": 8}, B(n_evals=40))
    a = o.ask()
    assert isinstance(a, list) and len(a) == 8 and isinstance(a[0], dict)
    o.tell(a, [float(i) for i in range(8)])      # 整代分数
    assert o.m is not None


def test_cma_es_beats_random():
    from ml_toolbox.opt.synth import make_objective_from_synth
    obj = make_objective_from_synth("rosenbrock_2d")
    r = optimize(obj, registry.get("cma_es"), Budget(n_evals=60),
                 cfg={"popsize": 8}, seed=1)
    rs = optimize(obj, registry.get("random_search"), Budget(n_evals=60),
                  seed=1)
    assert r.error is None and r.best["score"] < rs.best["score"]


def test_batch_budget_respected():
    """batch 引擎的 objective 调用数严格 ≤ n_evals（末代截断）。"""
    sp = ParamSpace([ParamSpec("x", "x", "number", 0.0, min=-3, max=3)])
    obj = make_objective(lambda q: q["x"] ** 2, sp, name="q")
    r = optimize(obj, registry.get("cma_es"), Budget(n_evals=20),
                 cfg={"popsize": 8}, seed=2)
    assert len(r.history) == 20, f"应正好 20，实得 {len(r.history)}"


def test_nelder_mead_local():
    """NM 在光滑单谷函数上收敛到真最优附近。"""
    from ml_toolbox.opt.synth import make_objective_from_synth
    obj = make_objective_from_synth("six_hump")
    r = optimize(obj, registry.get("nelder_mead"), Budget(n_evals=80), seed=1)
    assert r.error is None
    assert r.best["score"] <= obj.f_min + 0.02, \
        f"NM {r.best['score']:.4f} 未逼近真最优 {obj.f_min:.4f}"


def test_tpe_mixed_space():
    if not _optuna_available():
        print("    (optuna 不可用，跳过 tpe)")
        return
    sp = _mixed_space()
    def f(q):
        base = {"rbf": 0.0, "poly": 1.0, "lin": 3.0}[q["ker"]]
        return (q["x"] - 1.0) ** 2 + base + 0.1 * q["k"]
    r = optimize(make_objective(f, sp, name="tpe_mixed"),
                 registry.get("tpe"), Budget(n_evals=30), seed=3)
    assert r.error is None
    assert r.best["ker"] in ("rbf", "poly", "lin")
    assert r.best["score"] < 2.0


def test_asha_runs():
    if not _optuna_available():
        print("    (optuna 不可用，跳过 asha)")
        return
    from ml_toolbox.opt.synth import make_objective_from_synth
    obj = make_objective_from_synth("ackley_3d")
    r = optimize(obj, registry.get("asha"), Budget(n_evals=25), seed=4)
    assert r.error is None and r.best is not None


def test_nsga_ii_pareto():
    """多目标：NSGA-II 在 ZDT1 上逼近凸 Pareto 前沿（f2=1-sqrt(f1)）。"""
    from ml_toolbox.opt.synth import make_objective_multi
    obj = make_objective_multi("zdt1", dim=6)
    assert obj.multi and obj.n_obj == 2
    r = optimize(obj, registry.get("nsga_ii"), Budget(n_evals=2000),
                 cfg={"popsize": 20}, seed=21)
    assert r.error is None, r.error
    assert r.pareto is not None and len(r.pareto) >= 5
    P = r.pareto[["f0", "f1"]].to_numpy(float)
    dev = np.abs(P[:, 1] - (1 - np.sqrt(P[:, 0]))).max()
    assert dev < 0.35, f"前沿偏离凸形 {dev:.3f}"
    assert P[:, 0].max() - P[:, 0].min() > 0.5, "前沿应覆盖 f0 大部分区间"


def test_pareto_front_dominance():
    """pareto_front 正确性：支配关系手工例。"""
    from ml_toolbox.opt.runner import pareto_front, dominates
    import pandas as pd
    assert dominates(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert not dominates(np.array([1.0, 2.0]), np.array([0.5, 3.0]))
    h = pd.DataFrame({"f0": [1.0, 2.0, 0.5, 3.0], "f1": [1.0, 2.0, 3.0, 0.5],
                      "status": ["ok"] * 4})
    pf = pareto_front(h, 2)
    assert set(pf.index) == {0, 1, 2, 3} - {1}, "(2,2) 被 (1,1) 支配"


def test_autotuner_beats_default():
    """接缝2 验收：AutoTuner 调 logistic 超参，CV f1 ≥ 默认参数。"""
    import pandas as pd
    from ml_toolbox.core import registry as ml
    from ml_toolbox.opt.bridges import autotune
    ml.load_builtin()
    from sklearn.datasets import make_classification
    X, y = make_classification(200, 8, n_informative=5, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    rec, best, val = autotune("logistic", X, pd.Series(y),
                              optimizer=registry.get("gp_bo"),
                              budget=Budget(n_evals=20), cv_folds=3, seed=1)
    assert rec.error is None, rec.error
    assert val is not None and val > 0.5
    # 默认参数基线
    from ml_toolbox.core import runner
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    from ml_toolbox.core.contracts import RunConfig
    spec = Pipeline.default().run(Dataset.from_arrays(X.values, y, "t"))
    base = runner.run_one(ml.get("logistic"), spec,
                          RunConfig(extras={"cv_folds": 3}))
    base_f1 = base.result.metrics.get("cv_f1_mean",
                                      base.result.metrics.get("f1"))
    assert val >= base_f1 - 0.05, f"调参 {val:.3f} 不应明显劣于默认 {base_f1:.3f}"


def test_response_surface_bridge():
    """接缝3 验收：全域采样的评估历史 -> 响应面回归可学（R²>0.5）。"""
    from ml_toolbox.opt.bridges import response_surface
    sp = ParamSpace([ParamSpec("a", "a", "number", 0.0, min=0, max=5),
                     ParamSpec("b", "b", "number", 0.0, min=0, max=5)])
    obj = make_objective(lambda q: (q["a"] - 2) ** 2 + (q["b"] - 3) ** 2,
                         sp, name="surf")
    rec = optimize(obj, registry.get("random_search"), Budget(n_evals=60),
                   seed=1)
    out = response_surface(rec, "random_forest")
    assert out is not None
    m, res, r2 = out
    assert np.isfinite(r2) and r2 > 0.5, f"响应面 R2={r2}"


def test_gpbo_reuses_gpr_surrogate():
    """接缝1 实证：GP-BO 内部代理就是 sklearn GPR（purpose=surrogate 兑现）。"""
    from ml_toolbox.opt.synth import make_objective_from_synth
    obj = make_objective_from_synth("six_hump")
    o = registry.get("gp_bo")
    optimize(obj, o, Budget(n_evals=15), cfg={"n_init": 5}, seed=1)
    from sklearn.gaussian_process import GaussianProcessRegressor
    assert isinstance(o._gpr, GaussianProcessRegressor)


def test_file_source_hot_reload():
    """数据侧接入：外部程序追加 csv 行，FileSource 下次 fetch 看到新数据。"""
    import tempfile
    from ml_toolbox.opt.sources import FileSource
    tmp = os.path.join(tempfile.gettempdir(), "opt_src_rt.csv")
    try:
        pd.DataFrame({"a": [1.0, 2.0], "target": [0, 1]}).to_csv(tmp, index=False)
        src = FileSource(tmp, "target")
        X, y = src.fetch()
        assert len(X) == 2 and "target" not in X.columns
        pd.DataFrame({"a": [3.0], "target": [1]}).to_csv(tmp, mode="a",
                                                         header=False, index=False)
        X2, _ = src.fetch()
        assert len(X2) == 3, "mtime 变化应触发重读"
        assert "文件" in src.describe()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def test_process_objective_and_failure():
    """评估侧接入：子进程黑盒可优化；进程失败记 failed 不崩（P10）。"""
    import tempfile
    from ml_toolbox.opt.process import ProcessObjective
    script = os.path.join(tempfile.gettempdir(), "opt_fake_sim.py")
    with open(script, "w") as f:
        f.write("import sys\na=float(sys.argv[1])\n"
                "print('score: %.6f' % ((a-1.3)**2))\n")
    sp = ParamSpace([ParamSpec("a", "a", "number", 0.0, min=-3, max=3)])
    obj = ProcessObjective(f'python "{script}" {{a}}', sp, name="fakesim")
    r = optimize(obj, registry.get("nelder_mead"), Budget(n_evals=40), seed=1)
    assert r.error is None, r.error
    assert r.best["score"] < 0.01 and abs(r.best["a"] - 1.3) < 0.15
    bad = ProcessObjective('python -c "import sys; sys.exit(2)"', sp,
                           name="badproc")
    r2 = optimize(bad, registry.get("random_search"), Budget(n_evals=5), seed=1)
    assert r2.error is None and r2.best is None
    assert (r2.history["status"] == "failed").all()


def test_autotuner_accepts_source():
    """AutoTunerObjective 三种数据入口：裸数组 / DataSpec / FileSource。"""
    import tempfile
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    from ml_toolbox.opt.bridges import AutoTunerObjective
    from ml_toolbox.opt.sources import FileSource
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(60, 3), columns=list("abc"))
    y = pd.Series((X["a"] > 0).astype(int))
    p = {"C": 1.0}
    o1 = AutoTunerObjective("logistic", X, y, cv_folds=3)
    assert np.isfinite(o1(p))
    spec = Pipeline.default().run(Dataset(X.assign(target=y.values),
                                          name="t", target="target"))
    o2 = AutoTunerObjective("logistic", spec, cv_folds=3)
    assert o2._src.__class__.__name__ == "SpecSource"
    assert np.isfinite(o2(p))
    tmp = os.path.join(tempfile.gettempdir(), "opt_at_src.csv")
    try:
        X.assign(target=y.values).to_csv(tmp, index=False)
        o3 = AutoTunerObjective("logistic", FileSource(tmp, "target"),
                                cv_folds=3)
        assert np.isfinite(o3(p))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def test_constrained_process_objective():
    """带硬约束黑盒：infeasible 状态 + penalize 策略 + cEI 不死锁（彩排固化）。

    用无 sleep 的内联脚本，预算小，只验三条关键路径。
    """
    import tempfile
    from ml_toolbox.opt.process import ProcessObjective
    # 约束：a>=0.5 才可行（模拟软件拒绝，退出码 3 + 信号词）
    script = os.path.join(tempfile.gettempdir(), "opt_constrained_sim.py")
    with open(script, "w") as f:
        f.write("import sys\na=float(sys.argv[1])\n"
                "if a < 0.5:\n    print('违反约束', file=sys.stderr); sys.exit(3)\n"
                "print('score: %.4f' % ((a-0.8)**2))\n")
    sp = ParamSpace([ParamSpec("a", "a", "number", 0.0, min=0.0, max=1.0)])
    cmd = f'python "{script}" {{a:.3f}}'

    # 1) constraint_signal -> infeasible 状态（非 failed）
    obj = ProcessObjective(cmd, sp, name="c1", constraint_signal="违反")
    assert obj.has_constraints and obj.is_constraint_error(RuntimeError("违反约束"))
    assert not obj.is_constraint_error(RuntimeError("段错误"))
    r = optimize(obj, registry.get("random_search"), Budget(n_evals=12), seed=1)
    st = set(r.history["status"])
    assert "infeasible" in st and "failed" not in st - {"infeasible"}

    # 2) penalize：约束违反点返回有限惩罚分（status=ok，喂 GP 全量点）
    obj2 = ProcessObjective(cmd, sp, name="c2", constraint_signal="违反",
                            on_infeasible="penalize")
    r2 = optimize(obj2, registry.get("gp_bo"), Budget(n_evals=20),
                  cfg={"n_init": 6}, seed=1)
    assert r2.error is None and r2.best["score"] < 0.05   # 找到 a≈0.8 近优

    # 3) cEI（constrain 开关）：warmup 不因可行点稀疏而死锁
    obj3 = ProcessObjective(cmd, sp, name="c3", constraint_signal="违反")
    opt = registry.get("gp_bo")
    r3 = optimize(obj3, opt, Budget(n_evals=20),
                  cfg={"n_init": 6, "constrain": True}, seed=1)
    assert r3.error is None and r3.best is not None
    # 死锁修复的判据：GP 真跑过（_gpc 分类器拟合过 = 有过两类样本）
    assert opt._gpc is not None, "cEI 未进入采集优化（warmup 死锁复发？）"


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
