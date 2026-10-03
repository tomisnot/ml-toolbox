# -*- coding: utf-8 -*-
"""回归测试（外部资产 `checklist.md` 的 C 关）：核心契约 + 关键坑的永久断言。

运行：python tests/test_regressions.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SLOW = os.environ.get("MLTB_SLOW") == "1"
_EPOCHS = 60 if SLOW else 30      # fast：够验证四页工件契约；slow：真实训练

PASS = 0
FAIL = []


def check(name, fn):
    global PASS
    import time
    t0 = time.time()
    try:
        fn()
        PASS += 1
        print(f"  ✓ {name} ({time.time() - t0:.1f}s)")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  ✗ {name}: {type(e).__name__}: {str(e)[:200]}")


# ---------------------------------------------------------------- 契约
def test_registry_loaded():
    from ml_toolbox.core import registry
    registry.load_builtin()
    fams = registry.families()
    for fam in ("linear", "ensemble", "svm", "knn", "bayes",
                "cluster", "manifold", "anomaly", "timeseries"):
        assert fam in fams and fams[fam], f"缺方法族 {fam}"
    assert len(registry.names()) >= 40


def test_param_clean():
    from ml_toolbox.core.contracts import ParamSpec, RunConfig
    p = ParamSpec("alpha", label="α", kind="number", default=1.0, min=0, max=10)
    assert p.clean("2.5") == 2.5
    assert p.clean(None) == 1.0
    assert p.clean("") == 1.0
    try:
        p.clean("abc"); assert False
    except ValueError:
        pass
    try:
        p.clean("99"); assert False
    except ValueError:
        pass
    b = ParamSpec("flag", label="f", kind="bool", default=False)
    assert b.clean(True) is True
    assert b.clean("true") is True
    assert b.clean(None) is False


def test_pipeline_chain():
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.randn(80, 4), columns=list("abcd"))
    df["cat"] = rng.choice(["x", "y", "z"], 80)
    df["target"] = df["a"] * 2 + rng.randn(80) * 0.1
    spec = Pipeline.default().run(Dataset(df, name="t", target="target"))
    chain = spec.meta["chain"]
    assert len(chain) == 4
    assert "big_num" in chain[0]["summary"]
    assert spec.target_kind == "regression"
    assert len(spec.train_idx) + len(spec.test_idx) == 80
    # 指纹稳定
    spec2 = Pipeline.default().run(Dataset(df, name="t", target="target"))
    assert spec.meta["pipeline_id"] == spec2.meta["pipeline_id"]


def test_time_split_order():
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    s = pd.Series(np.arange(40.0))
    ds = Dataset(pd.DataFrame({"t": np.arange(40), "y": s}), name="t",
                 target="y", time_col="t")
    spec = Pipeline(steps=[], time_split=True, test_size=0.25).run(ds)
    assert spec.train_idx.max() < spec.test_idx.min()   # 未来不泄漏进训练


def test_run_batch_isolation():
    """一个方法失败不拖垮遍历：error 进 MLResult，不抛出。"""
    from ml_toolbox.core import registry, runner
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    from ml_toolbox.core.contracts import RunConfig
    registry.load_builtin()
    from sklearn.datasets import make_classification
    X, y = make_classification(120, 6, random_state=0)
    spec = Pipeline.default().run(Dataset.from_arrays(X, y, "t"))
    recs = runner.run_batch(["logistic", "no_such_method", "knn"],
                            spec, RunConfig())
    by = {r.method: r for r in recs}
    assert by["logistic"].result.ok
    assert by["knn"].result.ok
    assert not by["no_such_method"].result.ok
    assert by["no_such_method"].result.error


def test_lightgbm_import_order():
    """坑 L1 永久断言：lightgbm 必须先于 PyQt5 import。"""
    import ml_toolbox.methods.ensemble as ens
    assert ens._lgb is not None, "lightgbm 应在 ensemble 顶层导入"
    mods = list(sys.modules)
    li = next(i for i, m in enumerate(mods) if m == "lightgbm")
    qi = [i for i, m in enumerate(mods) if m.startswith("PyQt5")]
    if qi:
        assert li < min(qi), "lightgbm 必须早于任何 PyQt5 模块被 import"


def _problem(X, y=None, *, kind=None):
    """造一个"全量即训练"的 `Problem`（判据里直接调 `fit` 时用）。

    ⚠ 范式已扩张为 `fit(problem, cfg, diag)`（见 `docs/契约.md` §1）⇒ 判据不该再
    按 `fit(X, y, ...)` 调。这个助手让"全量训练"的语义与 `runner` 一致
    （`train_idx/test_idx` 都指全量 ⇒ 方法里 `rows("train")` 拿到全量）。
    """
    from ml_toolbox.core.contracts import DataSpec, Problem
    import numpy as _np
    idx = _np.arange(len(X))
    spec = DataSpec(X=X, y=y, train_idx=idx, test_idx=idx, target_kind=kind)
    return Problem(spec)


def test_anomaly_direction():
    """坑 L8：分数越大越异常、label 1=异常（在全量数据上直接 fit 验证）。"""
    from ml_toolbox.core import registry
    from ml_toolbox.core.contracts import RunConfig
    registry.load_builtin()
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(200, 3), columns=list("abc"))
    X.iloc[:10] += 6
    y = pd.Series(np.where(np.arange(200) < 10, 1, 0))
    m = registry.get("iforest")
    r = m.fit(_problem(X, y), RunConfig())
    assert r.ok
    a = r.artifacts
    assert a["labels"][:10].mean() > 0.5, "注入的异常应被标 1"
    assert a["scores"][a["labels"] == 1].mean() > a["scores"][a["labels"] == 0].mean()
    assert r.metrics.get("auc", 0) > 0.9


def test_cv_runs():
    from ml_toolbox.core import registry
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    from ml_toolbox.core.contracts import RunConfig
    registry.load_builtin()
    from sklearn.datasets import make_regression
    X, y = make_regression(100, 5, noise=10, random_state=0)
    spec = Pipeline.default().run(Dataset.from_arrays(X, y, "r"))
    m = registry.get("ridge")
    cv = m.cross_validate(spec.X, spec.y, RunConfig(), cv_folds=3)
    assert "cv_rmse_mean" in cv and np.isfinite(cv["cv_rmse_mean"])


def test_cv_integration_zero_cost():
    """CV 默认关（零开销）；extras.cv_folds>=2 才产出 cv_* 指标（P3）。"""
    from ml_toolbox.core import registry, runner
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    from ml_toolbox.core.contracts import RunConfig
    registry.load_builtin()
    from sklearn.datasets import make_classification
    X, y = make_classification(120, 6, n_informative=4, random_state=0)
    spec = Pipeline.default().run(Dataset.from_arrays(X, y, "c"))
    off = runner.run_one(registry.get("logistic"), spec, RunConfig())
    assert not any(k.startswith("cv_") for k in off.result.metrics)
    on = runner.run_one(registry.get("logistic"), spec,
                        RunConfig(extras={"cv_folds": 4}))
    assert "cv_f1_mean" in on.result.metrics
    assert len(on.result.diag.get("cv_scores", [])) == 4


def test_persistence_roundtrip():
    from ml_toolbox.core import registry, runner, persistence
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    from ml_toolbox.core.contracts import RunConfig
    registry.load_builtin()
    from sklearn.datasets import make_regression
    X, y = make_regression(80, 4, random_state=0)
    spec = Pipeline.default().run(Dataset.from_arrays(X, y, "r"))
    rec = runner.run_one(registry.get("ridge"), spec, RunConfig())
    d = persistence.save_record(rec)
    meta = persistence.load_record(rec.run_id)
    assert meta["method"] == "ridge"
    assert os.path.exists(os.path.join(d, "record.json"))
    assert os.path.exists(os.path.join(d, "artifacts.npz"))


def test_history_artifact_roundtrip_core():
    """核心层：DataFrame 工件拆双数组存盘再读回（不依赖 UI）。"""
    from ml_toolbox.core import registry, runner, persistence
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    from ml_toolbox.core.contracts import RunConfig
    registry.load_builtin()
    from sklearn.datasets import make_regression
    X, y = make_regression(80, 4, random_state=0)
    spec = Pipeline.default().run(Dataset.from_arrays(X, y, "r"))
    rec = runner.run_one(registry.get("ridge"), spec, RunConfig())
    d = persistence.save_record(rec)
    z = np.load(os.path.join(d, "artifacts.npz"))
    assert "feature_importance_values" in z.files
    assert "feature_importance_names" in z.files
    assert "y_pred" in z.files


def test_transform_new_predict_unseen():
    """数模关键链路：训练集跑管道 -> 新数据同变换 -> 预测（未见类别/缺失值鲁棒）。"""
    from ml_toolbox.core import registry, runner
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    from ml_toolbox.core.contracts import RunConfig
    registry.load_builtin()
    rng = np.random.RandomState(0)
    n = 120
    train = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n),
                          "c": rng.choice(["x", "y", "z"], n),
                          "target": rng.randn(n)})
    train.loc[:5, "a"] = np.nan
    test = pd.DataFrame({"a": rng.randn(30), "b": rng.randn(30),
                         "c": rng.choice(["x", "y", "w"], 30)})  # 未见类别 w
    test.loc[:2, "a"] = np.nan
    pipe = Pipeline.default()
    spec = pipe.run(Dataset(train, name="t", target="target"))
    Xnew = pipe.transform_new(test, target="target")
    assert list(Xnew.columns) == list(spec.X.columns), "新数据列须与训练对齐"
    assert Xnew.shape[0] == 30 and int(Xnew.isna().sum().sum()) == 0
    m = registry.get("ridge")
    rec = runner.run_one(m, spec, RunConfig())
    pred = m.predict(Xnew, rec.result)
    assert len(pred) == 30 and np.isfinite(pred).all()


def test_purposes_derivation():
    """建模目的维度：task+tags+family 推导正确，且无方法漏标。"""
    from ml_toolbox.core import registry
    registry.load_builtin()
    for m in registry.all_methods():
        ps = m.purposes_of()
        assert ps, f"{m.name} 无用途标注"
    g = lambda nm: registry.get(nm).purposes_of()
    from ml_toolbox.core.contracts import (
        PURPOSE_PREDICT, PURPOSE_DISCRIMINATE, PURPOSE_EXPLAIN,
        PURPOSE_CLUSTER, PURPOSE_REDUCE, PURPOSE_EVALUATE,
        PURPOSE_ANOMALY, PURPOSE_SURROGATE, PURPOSE_BASELINE)
    assert PURPOSE_EXPLAIN in g("lasso")            # 稀疏归因
    assert PURPOSE_EXPLAIN not in g("xgboost")
    assert PURPOSE_SURROGATE in g("gpr")            # 代理模型
    assert PURPOSE_BASELINE in g("dummy")
    # tags 里的 'baseline'（族默认起点）不得误标为"参照基线"目的
    assert PURPOSE_BASELINE not in g("pca")
    assert PURPOSE_BASELINE not in g("kmeans")
    assert PURPOSE_REDUCE in g("pca") and PURPOSE_EVALUATE in g("pca")
    assert PURPOSE_CLUSTER in g("kmeans")
    assert PURPOSE_ANOMALY in g("iforest")


def test_predict_with_fresh_instance():
    """独立引用：fit 与 predict 用不同方法实例，标签编码不丢。"""
    from ml_toolbox.core import registry
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    from ml_toolbox.core.contracts import RunConfig
    registry.load_builtin()
    from sklearn.datasets import make_classification
    X, y = make_classification(120, 6, n_classes=3, n_clusters_per_class=1,
                               n_informative=4, n_redundant=1, random_state=0)
    y = np.array([f"cls_{v}" for v in y])          # 字符串标签
    spec = Pipeline.default().run(Dataset.from_arrays(X, y, "t"))
    r = registry.get("logistic").fit(_problem(spec.X, spec.y), RunConfig())
    fresh = registry.get("logistic")               # 全新实例
    pred = fresh.predict(spec.X.iloc[:10], r)
    assert set(np.unique(pred)) <= {"cls_0", "cls_1", "cls_2"}, \
        "新实例 predict 丢了标签编码"


def test_pages_declared_or_auto():
    """每个方法要么声明检视页，要么有兜底页，且 mpl 页必须绑绘图函数。"""
    from ml_toolbox.core import registry
    from ml_toolbox.core.runner import auto_pages
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    registry.load_builtin()
    from sklearn.datasets import make_classification
    X, y = make_classification(100, 5, random_state=0)
    spec = Pipeline.default().run(Dataset.from_arrays(X, y, "c"))
    for m in registry.all_methods():
        pages = m.inspect_pages(None) or auto_pages(m, spec)
        assert pages, f"{m.name} 无检视页"
        for ps in pages:
            if ps.kind == "mpl":
                assert ps.plot is not None or ps.hint, \
                    f"{m.name}/{ps.key} mpl 页既无 plot 也无 hint"


def _neural_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def test_neural_fit_and_artifacts():
    """方案C 核心：torch_mlp 训练产出四页所需全部契约工件。"""
    if not _neural_available():
        print("    (torch 不可用，跳过)")
        return
    from ml_toolbox.core import registry, runner
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    from ml_toolbox.core.contracts import RunConfig
    registry.load_builtin()
    from sklearn.datasets import make_classification
    X, y = make_classification(150, 8, n_informative=5, random_state=0)
    spec = Pipeline.default().run(Dataset.from_arrays(X, y, "nn"))
    m = registry.get("torch_mlp")
    r = runner.run_one(m, spec, RunConfig(diag=True,
                                          overrides={"epochs": _EPOCHS,
                                                     "lr": 0.01,
                                                     "latent_every": 4})).result
    assert r.ok, r.error
    a = r.artifacts
    assert a["nn_history"]["loss"].shape[0] == _EPOCHS
    # 初始权重快照必须在（构造时采集，非 epoch 后）——否则 ΔW 视图恒空
    assert all(v is not None for v in a["nn_weights"]["first"].values())
    assert set(a["nn_weights"]["last"]) == set(a["nn_weights"]["first"])
    assert a["nn_latent"].ndim == 3 and a["nn_latent"].shape[2] == 2
    assert len(a["nn_latent_epochs"]) == a["nn_latent"].shape[0]
    assert a["labels"] is not None and len(a["labels"]) == a["nn_latent"].shape[1]
    assert "死ReLU比例" in a["nn_layers"].columns
    assert a["nn_attr"].shape[1] == X.shape[1]          # 归因列对齐
    assert r.metrics["f1"] > 0.7


def test_neural_structure_knobs_are_really_effective():
    """⭐ 结构旋钮**真的生效**：每个激活/优化器/调度都跑一遍，且拒绝静默忽略。

    为什么必须有（本仓的判据口径）：`param_schema` 里多一个 `select` 只说明
    "界面/工具收得下这个值"，**不说明它被用上了**。而"参数形同生效"正是本仓
    反复在防的假绿（多个坑都记在 `docs/pitfalls.md`）。
    ⇒ 三件一起钉：
    1. 每个激活、每个优化器都**真能跑出结果**（不是抛在构造里）；
    2. 上了调度器之后 **TB 里的 lr 曲线必须真的在变**——若仍写参数里那个常量，
       曲线是一条**平线**，图就在骗人；
    3. **不适用就该拒绝**：给 adam 传 `momentum` 必须报错，未知取值必须响亮
       （不是静默退回 relu）。
    """
    if not _neural_available():
        print("    (torch 不可用，跳过)")
        return
    import numpy as np
    from sklearn.datasets import make_classification

    from ml_toolbox.core import registry, runner
    from ml_toolbox.core.contracts import RunConfig
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    from ml_toolbox.methods.neural.mlp import ACTIVATIONS, OPTIMIZERS, SCHEDULES

    registry.load_builtin()
    X, y = make_classification(120, 6, n_informative=4, random_state=0)
    spec = Pipeline.default().run(Dataset.from_arrays(X, y, "knobs"))
    m = registry.get("torch_mlp")

    def _run(**ov):
        # ⚠ `diag` 是 RunConfig 的**顶层字段**（不是 overrides 里的一项）：
        #   塞进 overrides 会被 `resolve()` 丢掉、`fit` 也就拿不到 ⇒ diag 恒空。
        cfg = RunConfig(diag=True,
                        overrides={"hidden": "8,4", "epochs": _EPOCHS, **ov})
        return runner.run_one(m, spec, cfg)

    # ① 每个激活都能跑（且 diag 报出的就是它——参数与事实对得上）
    for act in ACTIVATIONS:
        rec = _run(activation=act, diag=True)
        assert rec.result.ok, f"activation={act} 跑失败：{rec.result.error}"
        assert rec.result.diag["activation"] == act, rec.result.diag
    # ② 每个优化器都能跑
    for o in OPTIMIZERS:
        rec = _run(optimizer=o, lr=0.05, diag=True)
        assert rec.result.ok, f"optimizer={o} 跑失败：{rec.result.error}"
        assert rec.result.diag["optimizer"] == o, rec.result.diag
    # ③ ⭐ 调度器：必须能让 lr **真的衰减**（否则图是平线 = 骗人）
    for sched in ("step", "cosine"):
        rec = _run(lr=0.1, epochs=max(8, _EPOCHS), lr_schedule=sched,
                   lr_step=4, lr_gamma=0.5)
        assert rec.result.ok, rec.result.error
        series = np.asarray(rec.result.artifacts["nn_history"]["lr"], float)
        assert len(np.unique(np.round(series, 6))) > 1, \
            f"lr_schedule={sched} 的 lr 曲线是平的 ⇒ 调度没生效（图在骗人）"
        assert series[-1] < series[0], f"{sched} 应当让 lr 下降：{series}"
    # ④ ⭐ 不适用就**拒绝**，不静默忽略。
    #   ⚠ 这里**必须直测构造器**，不能靠"跑一次看有没有炸"：静默忽略时
    #   `momentum` 根本不会被传给 torch（只有 sgd 分支用它）⇒ 整轮训练**照样成功**，
    #   任何"看最终结果"的断言都发现不了它。实测踩过这条（判据当时是哑的）。
    torch = __import__("torch")
    probe = torch.nn.Linear(2, 2)
    # 用**同一个契约**造参数（`RunConfig.resolve` 是"未覆写取默认"的单一真源），
    # 不手抄默认值 —— 手抄的那份会悄悄漂。
    def _params(**ov):
        return RunConfig(overrides=ov).resolve(m.param_schema)

    for name in ("adam", "adamw", "rmsprop"):
        try:
            m._optimizer(_params(optimizer=name, momentum=0.9),
                         probe.parameters(), torch)
            raise AssertionError(
                f"optimizer={name} + momentum=0.9 被静默忽略了（参数在骗人）")
        except ValueError as exc:
            assert "momentum" in str(exc), exc
    # 默认 momentum=0 时任何优化器都不该报错
    for name in OPTIMIZERS:
        m._optimizer(_params(optimizer=name, momentum=0.0),
                     probe.parameters(), torch)
    # sgd **真的**把 momentum 用上了（不是收下就扔）
    opt = m._optimizer(_params(optimizer="sgd", momentum=0.9),
                       probe.parameters(), torch)
    assert float(opt.param_groups[0].get("momentum") or 0.0) == 0.9, \
        "sgd 没拿到 momentum ⇒ '收下就扔'"
    # ⑤ 未知取值响亮（不退回默认——那会让参数形同生效）
    for bad in ({"activation": "no_such_act"}, {"optimizer": "no_such_opt"},
                {"lr_schedule": "no_such_sched"}):
        assert not _run(**bad).result.ok, f"{bad} 没有被拒（静默退回默认了？）"


def test_neural_device_resolution_is_deterministic():
    """⭐ `device` 的解析规则（**不依赖本机有没有 CUDA**）。

    ⚠ 为什么单列这一条：把"设备透传"写成一整轮真训练 + 断言 `diag["device"]=="cpu"`
    是**假绿**——本机 `torch.cuda.is_available()` 为 False 时，`auto` 本来就会
    落到 cpu ⇒ 透传**断了也照样绿**（我在停掉守卫做红验证时发现的，两次都绿）。
    ⇒ 这里直测解析函数，并把 `cuda` 伪装成可用，让判定与硬件**无关**：
    1. 引擎透传的声明赢过"本机自动"；
    2. 方法自己的 `device` 参数**不会**被引擎默认的 `auto` 吃掉（否则就是
       "参数形同生效"）；
    3. 两边都 `auto` ⇒ 才回落到"有卡用卡、没卡用 CPU"。
    """
    from ml_toolbox.methods.neural.mlp import TorchMLP

    class _FakeDev:
        def __init__(self, s):
            self.s = str(s)

        def __repr__(self):
            return f"dev({self.s})"

    class _FakeTorch:
        """只提供 `_device` 用到的那点表面（**不导入 torch**，本判据无依赖）。"""
        cuda = type("c", (), {"is_available": staticmethod(lambda: True)})
        device = staticmethod(_FakeDev)

    t = _FakeTorch
    # ① 引擎声明赢
    assert repr(TorchMLP._device({"device": "auto"}, t,
                                 type("C", (), {"extras": {"device": "cpu"}})())) \
        == "dev(cpu)"
    assert repr(TorchMLP._device({"device": "auto"}, t,
                                 type("C", (), {"extras": {"device": "cuda:1"}})())) \
        == "dev(cuda:1)"
    # ② 具体的方法参数不被默认 auto 吃掉（有卡也不许改道）
    assert repr(TorchMLP._device({"device": "cpu"}, t,
                                 type("C", (), {"extras": {"device": "auto"}})())) \
        == "dev(cpu)"
    # ③ 两边 auto ⇒ 有卡用卡
    assert repr(TorchMLP._device({"device": "auto"}, t,
                                 type("C", (), {"extras": {"device": "auto"}})())) \
        == "dev(cuda)"
    # 没有 cfg（裸调用）也要能工作
    assert repr(TorchMLP._device({"device": "cpu"}, t)) == "dev(cpu)"


def test_neural_loss_init_clip_early_stop_and_device():
    """⭐ 第 1 批旋钮：损失 / 初始化 / 梯度裁剪 / 早停 / device 端到端。

    为什么每条都要真跑而不是"设上了就算"：
    1. `loss`——**与任务不匹配必须拒**（给回归配 celoss 会在反向传播时炸在形状上，
       而"悄悄换回 mse"会让人以为交叉熵生效了）；
    2. `init`——`xavier`/`kaiming` 要真的改到权重（不是收下就扔）；
    3. `grad_clip`——有它时训练仍收敛、且 diag 报出真值；
    4. `early_stop`——真停（不是跑满 epochs 却假装早停），**且没验证集时必须拒**
       （val_split=0 时没有任何"改善"可判断，静默忽略＝旋钮形同生效）；
    5. `device`——端到端走一遍 `RunConfig.extras["device"]`（解析规则本身由
       `test_neural_device_resolution_is_deterministic` 钉，那条与硬件无关）。
    """
    if not _neural_available():
        print("    (torch 不可用，跳过)")
        return
    import numpy as np
    from sklearn.datasets import make_classification, make_regression

    from ml_toolbox.core import registry, runner
    from ml_toolbox.core.contracts import RunConfig
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    from ml_toolbox.methods.neural.mlp import INITS, LOSSES

    registry.load_builtin()
    X, y = make_classification(120, 6, n_informative=4, random_state=0)
    spec_c = Pipeline.default().run(Dataset.from_arrays(X, y, "nn-c"))
    Xr, yr = make_regression(90, 5, noise=0.1, random_state=1)
    spec_r = Pipeline.default().run(Dataset.from_arrays(Xr, yr, "nn-r"))
    assert spec_r.target_kind == "regression", spec_r.target_kind
    m = registry.get("torch_mlp")

    def _run(spec, **ov):
        return runner.run_one(m, spec, RunConfig(
            diag=True, overrides={"hidden": "8,4", "epochs": _EPOCHS, **ov}))

    # ① 损失：自动 / 分类交叉熵 / 回归 MSE 都能跑；不匹配的组合必须拒
    assert _run(spec_c, loss="auto").result.ok
    assert _run(spec_c, loss="celoss").result.ok
    assert _run(spec_r, loss="mse").result.ok
    for spec, loss in ((spec_c, "mse"), (spec_r, "celoss"), (spec_c, "no_such")):
        assert not _run(spec, loss=loss).result.ok, \
            f"{loss} 用在 kind={spec.target_kind} 上竟然没被拒"
    assert _run(spec_c, loss="auto").result.diag["loss"] == "auto"

    # ② 初始化：三种都能跑，且真的改到了权重（xavier 与默认必须不同）
    import torch
    p_default = dict(RunConfig(overrides={"hidden": "8,4", "init": "default"})
                     .resolve(m.param_schema))
    p_xavier = dict(RunConfig(overrides={"hidden": "8,4", "init": "xavier"})
                    .resolve(m.param_schema))
    torch.manual_seed(0)
    w_def = [l.weight.detach().clone() for l in
             m._build(p_default, 6, 3, torch.nn).modules()
             if isinstance(l, torch.nn.Linear)]
    torch.manual_seed(0)
    mdl = m._build(p_xavier, 6, 3, torch.nn)
    m._init_weights(mdl, p_xavier, torch.nn)
    w_xav = [l.weight.detach().clone() for l in mdl.modules()
             if isinstance(l, torch.nn.Linear)]
    assert any(not torch.allclose(a, b) for a, b in zip(w_def, w_xav)), \
        "xavier 与 default 的权重完全一样 ⇒ init 没生效"
    for init in INITS:                       # 每个候选都真能跑（且 diag 对得上）
        rec = _run(spec_c, init=init)
        assert rec.result.ok, f"init={init}: {rec.result.error}"
        assert rec.result.diag["init"] == init

    # ③ 梯度裁剪：能跑，且 diag 报真值
    rec = _run(spec_c, grad_clip=0.5)
    assert rec.result.ok, rec.result.error
    assert rec.result.diag["grad_clip"] == 0.5

    # ④ 早停：真停（用一个必然很快过拟合的配置逼它停）
    rec = _run(spec_c, early_stop="on", early_stop_patience=3,
               epochs=200, lr=0.5, batch=8, hidden="64,64")
    d = rec.result.diag
    assert rec.result.ok, rec.result.error
    assert d["epochs_run"] < 200, f"早停没生效：跑满了 {d['epochs_run']} 轮"
    assert d["early_stopped_at"] is not None
    # 没验证集 ⇒ 必须拒（不静默忽略）
    assert not _run(spec_c, early_stop="on", val_split=0.0).result.ok, \
        "early_stop=on 且 val_split=0 时没有拒绝（静默忽略了）"

    # ⑤ device 透传：RunConfig.extras["device"] 必须决定实际设备
    cfg = RunConfig(diag=True, overrides={"hidden": "8", "epochs": _EPOCHS},
                    extras={"device": "cpu"})
    assert runner.run_one(m, spec_c, cfg).result.diag["device"] == "cpu"
    # 方法自己的 device 参数也不能被"引擎默认的 auto"吃掉
    cfg2 = RunConfig(diag=True,
                     overrides={"hidden": "8", "epochs": _EPOCHS, "device": "cpu"})
    assert runner.run_one(m, spec_c, cfg2).result.diag["device"] == "cpu"


def test_neural_param_choices_are_deterministic():
    """`select` 的 `choices` 顺序必须**稳定**（两次取到同一顺序）。

    为什么：`choices` 是接口的一部分——面板下拉框按它排、`describe_method`
    把它原样回给 AI。用 `frozenset` 当来源时遍历顺序不稳定 ⇒ AI 两次问到的
    候选顺序不一样（"同一事实两个说法"，且无法复现）。
    """
    from ml_toolbox.methods.neural.mlp import (ACTIVATIONS, OPTIMIZERS,
                                               SCHEDULES, TorchMLP)

    assert isinstance(OPTIMIZERS, tuple) and isinstance(SCHEDULES, tuple), \
        "应当是有序元组，不是 set/frozenset"
    for p in TorchMLP.param_schema:
        if p.choices:
            assert list(p.choices) == list(p.choices), p.key
            assert len(set(p.choices)) == len(list(p.choices)), \
                f"{p.key} 的候选有重复：{p.choices}"
    for spec in (("activation", list(ACTIVATIONS)),
                 ("optimizer", list(OPTIMIZERS)),
                 ("lr_schedule", list(SCHEDULES))):
        key, want = spec
        got = next(p for p in TorchMLP.param_schema if p.key == key)
        assert list(got.choices) == want, f"{key} 候选顺序漂了：{got.choices}"
        assert got.default in want, f"{key} 的默认值不在候选里：{got.default}"


def test_neural_fresh_predict_string_labels():
    """独立引用：新实例 + 字符串标签 predict 不丢编码（同 logistic 约定）。"""
    if not _neural_available():
        print("    (torch 不可用，跳过)")
        return
    from ml_toolbox.core import registry
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    from ml_toolbox.core.contracts import RunConfig
    registry.load_builtin()
    from sklearn.datasets import make_classification
    X, y = make_classification(120, 6, n_classes=3, n_clusters_per_class=1,
                               n_informative=4, random_state=0)
    y = np.array([f"c{v}" for v in y])
    spec = Pipeline.default().run(Dataset.from_arrays(X, y, "t"))
    r = registry.get("torch_mlp").fit(_problem(spec.X, spec.y),
                                      RunConfig(overrides={"epochs": 8}))
    pred = registry.get("torch_mlp").predict(spec.X.iloc[:10], r)   # 全新实例
    assert set(np.unique(pred)) <= {"c0", "c1", "c2"}


def test_neural_persistence_nested():
    """嵌套 dict 工件（nn_weights/nn_history）经 "__" 展平存盘可完整重组。"""
    if not _neural_available():
        print("    (torch 不可用，跳过)")
        return
    from ml_toolbox.core import registry, runner, persistence
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    from ml_toolbox.core.contracts import RunConfig
    registry.load_builtin()
    from sklearn.datasets import make_classification
    X, y = make_classification(120, 6, random_state=0)
    spec = Pipeline.default().run(Dataset.from_arrays(X, y, "n"))
    rec = runner.run_one(registry.get("torch_mlp"), spec,
                         RunConfig(overrides={"epochs": 6, "latent_every": 3}))
    persistence.save_record(rec)
    z = np.load(os.path.join(persistence.RUNS_DIR, rec.run_id, "artifacts.npz"))
    assert "nn_weights__last__0.weight" in z.files
    assert "nn_layers_values" in z.files and "nn_layers_cols" in z.files
    # UI 层重组（不启动窗口，纯函数）
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from ml_toolbox.ui.history import load_run_record
    r2 = load_run_record(rec.run_id)
    a = r2.result.artifacts
    assert isinstance(a["nn_weights"], dict)
    assert set(a["nn_weights"]["last"]) == set(a["nn_weights"]["first"])
    assert a["nn_latent"].shape == rec.result.artifacts["nn_latent"].shape
    assert list(a["nn_layers"].columns) == \
        list(rec.result.artifacts["nn_layers"].columns)


# ---------------------------------------------------------------- C1：方向/任务判定单一来源
def test_metric_direction_single_source():
    """M1：指标方向收编为 core.contracts.is_lower_better，三处消费者一致。"""
    from ml_toolbox.core.contracts import is_lower_better, LOWER_IS_BETTER
    # silhouette_deficit 曾被 gallery 漏掉（分叉点），现必须为越低越好
    assert is_lower_better("silhouette_deficit")
    for m in ("rmse", "mae", "mape"):
        assert is_lower_better(m), m
    for m in ("f1", "r2", "accuracy", "score"):
        assert not is_lower_better(m), m
    assert isinstance(LOWER_IS_BETTER, frozenset)
    # 三处消费者确实改用同一函数（读源码文本，不 import UI 以保持核心门无 Qt）
    import os as _os
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

    def _src(*p):
        with open(_os.path.join(root, *p), encoding="utf-8") as f:
            return f.read()

    assert "is_lower_better" in _src("ml_toolbox", "core", "runner.py")
    assert "is_lower_better" in _src("ml_toolbox", "ui", "gallery.py")
    br = _src("ml_toolbox", "opt", "bridges.py")
    assert "_LOWER_BETTER" not in br and "is_lower_better" in br


def test_infer_kind_single_source():
    """M10：任务判定收编为 core.contracts.infer_kind，dataset/pipeline 共用。"""
    from ml_toolbox.core.contracts import (infer_kind, REGRESSION_CARDINALITY)
    assert infer_kind(None) is None
    # 基数 > 阈值 的数值列 -> 回归
    yr = pd.Series(np.arange(REGRESSION_CARDINALITY + 5, dtype=float))
    assert infer_kind(yr) == "regression"
    # 低基数数值列 / 字符串列 -> 分类
    yc = pd.Series([0, 1, 0, 1] * 3)
    assert infer_kind(yc) == "classification"
    ys = pd.Series(["a", "b", "c"] * 10)
    assert infer_kind(ys) == "classification"
    # 外部调用者可直接传 ndarray/list，不应因 ndarray 没有 nunique 崩溃
    assert infer_kind(np.arange(REGRESSION_CARDINALITY + 5, dtype=float)) == "regression"
    assert infer_kind(np.array([0, 1, 0, 1])) == "classification"
    assert infer_kind(["a", "b", "c"]) == "classification"
    # pipeline 的 _infer_kind 是同一对象（别名，非第二份实现）
    from ml_toolbox.core import pipeline
    assert pipeline._infer_kind is infer_kind


def test_autotune_restore_inverse():
    """M2：AutoTunerObjective.restore 是 evaluate 方向的精确逆（替 to_maximize）。"""
    from ml_toolbox.opt.bridges import AutoTunerObjective
    from sklearn.datasets import make_classification
    X, y = make_classification(60, 5, n_informative=3, n_redundant=1,
                               random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    obj = AutoTunerObjective("logistic", X, pd.Series(y), cv_folds=2)
    # f1 越大越好 -> evaluate 取负（最小化方向），restore 应还原为正
    raw = 0.83
    assert abs(obj.restore(-raw) - raw) < 1e-12
    # 越低越好的指标（rmse）方向不翻
    obj._metric = "rmse"
    assert abs(obj.restore(0.42) - 0.42) < 1e-12


# ---------------------------------------------------------------- C5：并行度契约化
def test_parallel_njobs_contract():
    """M5：MLTB_NJOBS 三态统一控制并行度（成员/元学习器语义分离）。"""
    import os as _os
    from ml_toolbox.core import parallel
    saved = _os.environ.get("MLTB_NJOBS")
    try:
        _os.environ.pop("MLTB_NJOBS", None)
        assert parallel.nj() == -1 and parallel.nj(1) == 1   # 未设 -> 调用点默认
        assert parallel.meta_nj() == 1                        # 外层默认 1（L18）
        assert not parallel.restricted()
        _os.environ["MLTB_NJOBS"] = "off"
        assert parallel.nj() == 1 and parallel.meta_nj() == 1
        assert parallel.restricted()
        _os.environ["MLTB_NJOBS"] = "auto"
        assert parallel.nj() == -1
        assert parallel.meta_nj() == 1                        # auto 不放行外层
        _os.environ["MLTB_NJOBS"] = "4"
        assert parallel.nj() == 4 and parallel.meta_nj() == 4
        _os.environ["MLTB_NJOBS"] = "-1"                     # 显式 -1 才放行外层
        assert parallel.meta_nj() == -1
    finally:
        if saved is None:
            _os.environ.pop("MLTB_NJOBS", None)
        else:
            _os.environ["MLTB_NJOBS"] = saved


def test_core_does_not_import_methods():
    """C3/M4 永久断言：core 不得 import methods（依赖方向 methods→core）。

    曾经的违规点：core.runner.auto_pages 直接 from ..methods import plots。
    现 auto_pages 函数体下沉 methods/plots.py，经 registry 页提供者槽反向
    注册；本断言读源码防回潮（不 import UI，保持核心门无 Qt）。
    """
    import os as _os
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    with open(_os.path.join(root, "ml_toolbox", "core", "runner.py"),
              encoding="utf-8") as f:
        src = f.read()
    assert "from ..methods" not in src and "from .methods" not in src, \
        "core/runner 重新 import methods——C3 依赖倒置回潮"
    # 反向注册确实生效：load_builtin 后提供者已就位，旧代家族兜底页不断
    from ml_toolbox.core import registry, runner
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    registry.load_builtin()
    assert registry.page_provider() is not None, "页提供者未注册"
    from sklearn.datasets import make_regression
    X, y = make_regression(60, 4, random_state=0)
    spec = Pipeline.default().run(Dataset.from_arrays(X, y, "r"))
    m = registry.get("ridge")                       # 旧代：无 inspect_pages
    pages = m.inspect_pages(None) or runner.auto_pages(m, spec)
    assert pages, "ridge 兜底页丢失（auto_pages 下沉后断链？）"


# ---------------------------------------------------------------- 量子启发方法
def _qspec_reg(n=60, d=4, seed=0, cls=False):
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = (X[:, 0] > 0).astype(int) if cls else X[:, 0] * 2 + np.sin(X[:, 1])
    return Pipeline.default().run(Dataset.from_arrays(X, y, name="q"))


def test_quantum_methods_registered_in_families():
    """量子启发方法“归到该归的族”：不单独成族，tags 带 quantum。"""
    from ml_toolbox.core import registry
    registry.load_builtin()
    expect = {"qsvc": "svm", "qlssvc": "svm", "qkrr": "linear",
              "qkmeans": "cluster", "qpca": "manifold",
              "qreservoir": "timeseries"}
    fams = registry.families()
    for nm, fam in expect.items():
        assert nm in fams.get(fam, []), f"{nm} 应在族 {fam}，实归 {fams.get(fam)}"
        m = registry.get(nm)
        assert "quantum" in m.tags, f"{nm} 缺 quantum 标签"
    # 没有新增名为 quantum 的 ML 族（量子启发不单独成族）
    assert "quantum" not in fams, "ML 侧不应出现 quantum 独立族"


def test_quantum_feature_map_analytic_kernel():
    """n=1 保真度核必须等于解析式 cos²((x−z)/2)（书式 3.79）。

    编码约定（qubit 位序 / Ry 角度 / 归一）一旦漂移，核值会错但方法不报错——
    这是只有断言能抓的静默 bug。
    """
    from ml_toolbox.methods import qsim
    xs = np.linspace(0, np.pi, 6)
    S = qsim.feature_map_states(xs.reshape(-1, 1))
    K = qsim.fidelity_kernel(S)
    ana = np.cos((xs[:, None] - xs[None, :]) / 2) ** 2
    assert np.allclose(K, ana, atol=1e-9), "n=1 量子核偏离解析 cos²((x−z)/2)"
    # 保真度核必须 PSD（式 3.70-3.72）
    S4 = qsim.feature_map_states(np.random.RandomState(1).rand(12, 3) * np.pi)
    ev = np.linalg.eigvalsh(qsim.fidelity_kernel(S4))
    assert ev.min() > -1e-9, "量子核非半正定"


def test_quantum_shots_reproducible():
    """同 seed+shots 的核矩阵逐位一致（反演测试采样可复现）。"""
    from ml_toolbox.core import registry, runner
    from ml_toolbox.core.contracts import RunConfig
    registry.load_builtin()
    spec = _qspec_reg(50, cls=True)
    a = runner.run_one(registry.get("qsvc"), spec,
                       RunConfig(seed=7, overrides={"shots": 200}))
    b = runner.run_one(registry.get("qsvc"), spec,
                       RunConfig(seed=7, overrides={"shots": 200}))
    assert np.allclose(a.result.artifacts["q_kernel"],
                       b.result.artifacts["q_kernel"])


def test_quantum_supervised_fresh_instance_string_labels():
    """qsvc：换新实例仅凭 result 预测，仍返回原始字符串标签（L19 同类）。"""
    from ml_toolbox.core import registry, runner
    from ml_toolbox.core.contracts import RunConfig
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    registry.load_builtin()
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(30, 2) + 1.5, rng.randn(30, 2) - 1.5])
    y = np.array(["cat"] * 30 + ["dog"] * 30)
    spec = Pipeline.default().run(Dataset.from_arrays(X, y, name="s"))
    rec = runner.run_one(registry.get("qsvc"), spec, RunConfig(seed=42))
    assert rec.result.ok, rec.result.error
    fresh = registry.get("qsvc")                 # 全新未拟合实例
    pred = fresh.predict(spec.X, rec.result)
    assert set(np.unique(pred)) <= {"cat", "dog"}, \
        f"跨实例预测未还原原始标签：{set(np.unique(pred))}"


def test_qpca_matches_classical_pca():
    """DME 幂迭代首方向与经典 PCA 第一主成分对齐（子空间正确性）。"""
    from ml_toolbox.core import registry, runner
    from ml_toolbox.core.contracts import RunConfig
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    registry.load_builtin()
    rng = np.random.RandomState(3)
    t = rng.randn(120)
    X = np.column_stack([t * 3 + 0.1 * rng.randn(120), t * 3 + 0.1 * rng.randn(120),
                         0.2 * rng.randn(120)])
    spec = Pipeline.default().run(Dataset.from_arrays(X, None, name="p"))
    rec = runner.run_one(registry.get("qpca"), spec, RunConfig(seed=42))
    assert rec.result.ok, rec.result.error
    from sklearn.decomposition import PCA
    Xtr = spec.X.to_numpy(float)[spec.train_idx]      # qpca 在训练段拟合
    pc1 = PCA(n_components=1, random_state=0).fit(Xtr).components_[0]
    emb = rec.result.artifacts["embedding"][:, 0]
    # 嵌入投影方向应与 PC1 高度相关（|corr| 接近 1）
    proj = Xtr @ pc1
    assert len(emb) == len(proj), "嵌入行数与训练段不一致"
    assert abs(np.corrcoef(emb, proj)[0, 1]) > 0.95, "DME 首方向偏离经典 PCA"


def test_qreservoir_memory_and_forecast():
    """量子储层：产出有限 rmse + 记忆容量谱（读出/滚动外推契约）。"""
    from ml_toolbox.core import registry, runner
    from ml_toolbox.core.contracts import RunConfig
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import Pipeline
    registry.load_builtin()
    t = np.arange(160.0)
    s = 0.3 * t + 8 * np.sin(2 * np.pi * t / 12) + np.random.RandomState(1).randn(160) * 0.5
    spec = Pipeline.default().run(Dataset.from_arrays(s.reshape(-1, 1), s, name="ts"))
    rec = runner.run_one(registry.get("qreservoir"), spec, RunConfig(seed=42))
    assert rec.result.ok, rec.result.error
    assert np.isfinite(rec.result.metrics["rmse"]), "rmse 非有限"
    mc = rec.result.artifacts["qrc_mc"]
    assert len(mc) > 0 and np.nanmax(mc) > 0, "记忆容量谱为空/全零"


def test_sqa_engine_registered_family_quantum():
    """opt 侧：量子退火引擎注册在 family=quantum，单点 ask/tell 能收敛。"""
    from ml_toolbox.opt import registry as oreg
    from ml_toolbox.opt.contracts import ParamSpace, Budget, make_objective
    from ml_toolbox.core.contracts import ParamSpec
    from ml_toolbox.opt.runner import optimize
    oreg.load_builtin()
    assert "quantum_anneal" in oreg.families().get("quantum", []), \
        "量子退火应注册在 opt family=quantum"
    space = ParamSpace([ParamSpec(f"x{i}", "", "number", 0.0, min=-2, max=2)
                        for i in range(3)])
    obj = make_objective(lambda p: sum(p[k] ** 2 for k in ["x0", "x1", "x2"]),
                         space, name="sphere")
    r = optimize(obj, oreg.get("quantum_anneal"), Budget(n_evals=80), seed=42)
    assert r.error is None and r.best is not None
    assert float(r.best["score"]) < 1.0, "SQA 未把球函数降到 1.0 以下"


# ---------------------------------------------------------------- 开源前卫生（夹具脱敏）
def test_cases_fixtures_have_no_foreign_or_local_absolute_paths():
    """`cases/**` 里不许出现**本机/他人**的绝对路径（开源前脱敏的永久钉子）。

    来历（2026-09-27/28）：5 个 `freeze_autotune*.json` 夹具原先都把 `proc.cwd` 写成
    **另一个项目**的本机绝对路径 ⇒ 两个后果：① 随仓发布等于暴露他人本地路径；
    ② 那条 UI 判据**依赖别人机器上的目录存在**（那个目录一被移走，判据就红——实测发生过）。
    ⇒ 夹具只放中性占位 `C:\\path\\to\\blackbox`，判据在装载前换成真临时目录。

    口径（如实）：查的是**已知的他人/本机字样**（项目名、机器名、数据盘盘符、用户目录），
    不是"任何盘符路径"——`C:\\path\\to\\blackbox` 这种中性占位是允许的。
    """
    import glob as _glob

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = [p for p in _glob.glob(os.path.join(root, "cases", "**", "*"), recursive=True)
             if os.path.isfile(p)]
    # R8 自证：真的扫到了夹具（否则"没命中"只是因为没读到东西）
    assert len(files) >= 5, f"cases/ 下只扫到 {len(files)} 个文件 ⇒ 判据可能没生效"

    bad_words = ("Energy " + "Level", "Baidu" + "Syncdisk", "LEN" + "OVO", "/Users/", "D:" + chr(92))
    hits = []
    for path in sorted(files):
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for w in bad_words:
            if w in text:
                hits.append(f"{os.path.relpath(path, root)}: {w!r}")
    assert not hits, ("夹具里出现本机/他人绝对路径字样（开源前必须脱敏）：\n  "
                      + "\n  ".join(hits))


def test_archived_target_kind_is_the_methods_not_the_datasets():
    """⭐ 归档的 ``target_kind`` 必须是**方法实际用的**那个，不是**数据集的**那个。

    ## 为什么（这是一个真 bug 的永久断言）

    `ridge` 声明 `target_kind="regression"`（只能回归），但**可以被喂分类数据**——
    此时指标是 `rmse/mae/r2/mape`。早先 `run_one` 落盘用的是 `spec.target_kind`
    （数据集的 kind）⇒ 存档自述"分类"而指标全是回归指标（实测前 400 条里 **47 条**如此）。
    后果不是"数字难看"，而是**下游按 `target_kind` 挑检视页会挑错**：
    面板/`auto_pages` 会给它配混淆矩阵与 ROC，而画混淆矩阵需要**离散**预测——
    回归预测是连续值 ⇒ 直接 `ValueError: Classification metrics can't handle a mix of
    multiclass and continuous targets`，那一页在面板里就炸给你看。

    ⇒ 不变量：`RunRecord.target_kind == res.target_kind or spec.target_kind`。
    """
    from ml_toolbox.core.contracts import DataSpec, RunConfig
    from ml_toolbox.core.runner import run_one
    from ml_toolbox.core.registry import get as _get
    from ml_toolbox.core import registry
    registry.load_builtin()

    rng = np.random.RandomState(0)
    n = 80
    frame = pd.DataFrame(rng.randn(n, 4), columns=[f"f{i}" for i in range(4)])
    y = pd.Series((frame["f0"] > 0).astype(int))          # 分类数据的形态

    ridge = _get("ridge")
    assert getattr(ridge, "target_kind", None) == "regression", \
        "前提变了：ridge 现在不是纯回归方法 ⇒ 本判据要跟着改"
    spec = DataSpec(X=frame, y=y, target_kind="classification", meta={})
    rec = run_one(ridge, spec, RunConfig(seed=42))

    assert rec.target_kind == "regression", (
        f"落盘用了数据集的 kind（{rec.target_kind!r}）而不是方法实际用的"
        f"（{rec.result.target_kind!r}）⇒ 存档自述与指标会互相矛盾")
    # 指标确实是回归的（这正是"自述分类"会显得荒谬的地方）
    got = set(rec.result.metrics)
    assert got & {"rmse", "mae", "r2"}, f"这批指标不像回归：{sorted(got)}"


def main():
    # 默认控制台可能是 GBK（本机 cp936）：✓/✗ 会让直跑入口在**失败分支**打印时
    # 崩成 UnicodeEncodeError，把真实的失败详情一起吞掉（第三轮审查 P2-9）。
    # 与 tests/run_all.py / tests/test_ml_mecha.py::main() 同款加固。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tests = [(k[5:], v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"回归测试 {len(tests)} 项")
    for name, fn in tests:
        check(name, fn)
    print(f"\n{PASS} passed, {len(FAIL)} failed")
    if FAIL:
        for n, e in FAIL:
            print(f"  FAILED {n}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
