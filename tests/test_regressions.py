# -*- coding: utf-8 -*-
"""回归测试（checklist.md C 关）：核心契约 + 关键坑的永久断言。

运行：python tests/test_regressions.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

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
    r = m.fit(X, y, RunConfig())
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
    r = registry.get("logistic").fit(spec.X, spec.y, RunConfig())
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
                                          overrides={"epochs": 60,
                                                     "lr": 0.01,
                                                     "latent_every": 4})).result
    assert r.ok, r.error
    a = r.artifacts
    assert a["nn_history"]["loss"].shape[0] == 60
    # 初始权重快照必须在（构造时采集，非 epoch 后）——否则 ΔW 视图恒空
    assert all(v is not None for v in a["nn_weights"]["first"].values())
    assert set(a["nn_weights"]["last"]) == set(a["nn_weights"]["first"])
    assert a["nn_latent"].ndim == 3 and a["nn_latent"].shape[2] == 2
    assert len(a["nn_latent_epochs"]) == a["nn_latent"].shape[0]
    assert a["labels"] is not None and len(a["labels"]) == a["nn_latent"].shape[1]
    assert "死ReLU比例" in a["nn_layers"].columns
    assert a["nn_attr"].shape[1] == X.shape[1]          # 归因列对齐
    assert r.metrics["f1"] > 0.7


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
    r = registry.get("torch_mlp").fit(spec.X, spec.y,
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


def main():
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
