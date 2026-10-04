# -*- coding: utf-8 -*-
"""`Problem` 范式与图任务（GNN）的判据。

## 判什么

1. **范式等价**：`problem.rows("train")` **逐字等于**老接口传给 `fit` 的 `(Xtr, ytr)`
   —— 这是"扩张而不是破坏"的可验证定义（59 个方法的行为不该变）；
2. **能力声明**：数据带了辅助结构而方法**没声明支持** ⇒ **响亮拒绝**，
   不许静默忽略（老契约下"时序忽略外生 X"就是这么静默发生的）；
3. **掩码 vs 切行**：图任务用布尔掩码划分（索引指向**全量**行）——
   切行会让边索引失效，这是"行不独立"那一类问题的正解；
4. **GNN 端到端**：AI 写的图卷积代码能跑通，且观测面/工件与表格任务同构。

运行：``python tests/test_problem.py``（自带 main，与仓内其它 ``test_*.py`` 同形）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_toolbox.core.contracts import (AUX_GRAPH, DataSpec, Problem,  # noqa: E402
                                       RunConfig)


def _spec(n=60, d=4, kind="classification", seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, d), columns=[f"f{i}" for i in range(d)])
    y = pd.Series((X["f0"] + rng.randn(n) * 0.3 > 0).astype(int)) if kind else None
    # 前 75% 训练、后 25% 测试（**索引指向全量行**，与 pipeline 同构）
    k = int(n * 0.75)
    return DataSpec(X=X, y=y, train_idx=np.arange(k), test_idx=np.arange(k, n),
                    target_kind=kind)


# ============================================================ 1. 范式等价
def test_rows_train_equals_the_old_fit_arguments():
    """⭐ `rows("train")` == 老 `spec.split()` 传给 `fit` 的 `(Xtr, ytr)`。

    这条是"范式扩张**没有**改变既有方法行为"的可验证定义：
    老 runner 是 `Xtr, Xte, ytr, yte = spec.split()` 后 `fit(Xtr, ytr)`。
    """
    spec = _spec()
    old_Xtr, old_Xte, old_ytr, old_yte = spec.split()
    p = Problem(spec)
    Xtr, ytr = p.rows("train")
    Xte, yte = p.rows("test")
    pd.testing.assert_frame_equal(Xtr, old_Xtr)
    pd.testing.assert_frame_equal(Xte, old_Xte)
    pd.testing.assert_series_equal(ytr, old_ytr)
    pd.testing.assert_series_equal(yte, old_yte)
    # "all" 拿全量
    Xall, yall = p.rows("all")
    assert len(Xall) == len(spec.X) and len(yall) == len(spec.y)
    # 直通属性
    assert len(p) == len(spec.X) and p.target_kind == spec.target_kind


def test_masks_point_at_full_rows_not_slices():
    """⭐ `masks()` 是**布尔掩码**（长度 = 全量行数），不是切片索引。

    图任务的正解：同一张图、只对指定节点算损失。切行会让边索引失效，
    而且训练节点会沿边"看到"验证节点（消息传递）⇒ 分数乐观化。
    """
    spec = _spec(n=40)
    p = Problem(spec)
    m_tr, m_te = p.masks()
    assert m_tr.dtype == bool and m_te.dtype == bool
    assert len(m_tr) == len(m_te) == 40, "掩码长度必须等于全量行数"
    assert int(m_tr.sum()) == 30 and int(m_te.sum()) == 10
    assert not (m_tr & m_te).any(), "训练/测试掩码不该重叠"
    assert (m_tr | m_te).all(), "这份数据里两者应覆盖全部行"
    # 与 rows() 对齐：掩码选出的行 == rows("train") 的行
    assert list(np.where(m_tr)[0]) == list(p.rows("train")[0].index + 0), \
        "掩码与 rows() 必须指向同一批行"


# ============================================================ 2. 能力声明
def test_aux_requires_declared_support_and_refuses_honestly():
    """⭐ 没声明支持却想取辅助结构 ⇒ **拒绝**（两个方向分别给可读原因）。

    为什么必须拒（不是返回 None）：静默返回 None 会让方法以为"没有边"而按独立
    样本跑，得到一个**看起来正常但其实错**的结果。老契约下 `time_split` 之外的
    "时序忽略外生 X"就是这类静默。
    """
    spec = _spec()
    spec.aux[AUX_GRAPH] = pd.DataFrame({"src": [0], "dst": [1], "weight": [1.0]})
    # ① 方法**没声明**支持 ⇒ 拒绝，且原因指向"这个方法吃不了"
    p_no = Problem(spec, allowed=())
    try:
        p_no.require(AUX_GRAPH)
        raise AssertionError("没声明支持却拿到了辅助结构")
    except ValueError as exc:
        assert "没有声明支持" in str(exc), exc
    # ② 声明了但**数据没带** ⇒ 拒绝，原因指向"数据缺这个结构"
    plain = _spec()
    p_yes = Problem(plain, allowed=(AUX_GRAPH,))
    try:
        p_yes.require(AUX_GRAPH)
        raise AssertionError("数据没带却拿到了辅助结构")
    except ValueError as exc:
        assert "没有带辅助结构" in str(exc), exc
    # ③ 声明了且数据有 ⇒ 正常拿到，且能读出 edge_index
    p_ok = Problem(spec, allowed=(AUX_GRAPH,))
    ei, ew = p_ok.graph()
    assert ei.shape == (2, 1) and ei[0].tolist() == [0] and ei[1].tolist() == [1]
    assert ew.tolist() == [1.0]
    assert p_ok.partition_mode() == "masks"
    assert Problem(plain).partition_mode() == "rows"


def test_aux_capability_gate_blocks_methods_that_do_not_declare_it():
    """⭐ 运行入口的能力闸：带图数据跑 `torch_mlp` 必须**被拒绝**（不静默忽略）。

    闸在 `runner.run_one`（**不是**等方法自己调 `require`）：只读 `rows("train")`
    的方法永远不会去 `require`，所以只在 require 拦是拦不住的。
    """
    from ml_toolbox.core import registry, runner
    registry.load_builtin()
    spec = _spec()
    spec.aux[AUX_GRAPH] = pd.DataFrame({"src": [0, 1], "dst": [1, 2],
                                        "weight": [1.0, 1.0]})
    rec = runner.run_one(registry.get("logistic"), spec, RunConfig())
    assert rec.result.ok is False, "带图数据跑 logistic 竟然成功了（静默忽略拓扑）"
    msg = rec.result.error or ""
    assert "辅助结构" in msg and "graph" in msg, msg
    # 对照：没有图的数据照常跑
    ok = runner.run_one(registry.get("logistic"), _spec(), RunConfig())
    assert ok.result.ok is True, ok.result.error


# ============================================================ 3. GNN 端到端
#: 一个**真的图卷积**（自己实现消息传递；仓库依赖里没有 PyG/DGL，也不需要）。
_GCN_CODE = '''
class GCNLayer(nn.Module):
    """H' = act(D^-1/2 A_hat D^-1/2 H W)（稠密邻接，小图够用）。"""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)

    def forward(self, h, edge_index, n_nodes, edge_weight=None):
        idx = torch.arange(n_nodes)
        src = torch.cat([edge_index[0], idx])
        dst = torch.cat([edge_index[1], idx])
        w = torch.ones(src.numel())
        if edge_weight is not None:
            w = torch.cat([edge_weight, torch.ones(n_nodes)])
        A = torch.zeros(n_nodes, n_nodes)
        A.index_put_((src, dst), w, accumulate=True)
        A = A.clamp(max=1.0)
        deg = A.sum(1).clamp(min=1e-6)
        Dm = torch.diag(deg.pow(-0.5))
        return self.lin(Dm @ A @ Dm @ h)


class Net(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.g1 = GCNLayer(n_in, n_hidden)
        self.g2 = GCNLayer(n_hidden, n_out)

    def forward(self, x, edge_index, n_nodes, edge_weight=None):
        h = torch.relu(self.g1(x, edge_index, n_nodes, edge_weight))
        return self.g2(h, edge_index, n_nodes, edge_weight)


class Network:
    def __init__(self, hidden=16):
        self.hidden = hidden

    def build(self, n_in, n_out, *, nn):
        return Net(n_in, self.hidden, n_out)

    def forward_graph(self, model, ctx):
        return model(ctx["x"], ctx["edge_index"], ctx["n_nodes"],
                     ctx.get("edge_weight"))
'''


def _sbm(n_per=25, seed=0):
    """随机块模型：3 个社区；节点特征带社区信号、边块内密块间疏。"""
    rng = np.random.RandomState(seed)
    labels = np.repeat([0, 1, 2], n_per)
    centers = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, -2.0]])
    X = centers[labels] + rng.normal(scale=0.6, size=(len(labels), 2))
    edges = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            p = 0.35 if labels[i] == labels[j] else 0.03
            if rng.rand() < p:
                edges.append([i, j])
    return X, labels, edges


def test_gnn_runs_end_to_end_with_node_masks():
    """⭐ GNN 端到端：图卷积代码能跑通，且**观测面/工件与表格任务同构**。

    这条同时证明三件事：
    1. **残差/任意拓扑之外，图拓扑也表达得了**（`nn.Sequential` 表达不了消息传递）；
    2. 划分走**节点掩码**（`y_pred` 长度 == 测试节点数，而不是切行）；
    3. 观测面白拿：`nn_history` 有逐 epoch 曲线、`graph_n_*` 有拓扑事实。
    """
    try:
        import torch  # noqa: F401
    except Exception:
        print("    (torch 不可用，跳过)")
        return
    from ml_toolbox.core import runner
    from ml_toolbox.methods.neural import custom as C

    X, y, edges = _sbm()
    n = len(y)
    k = int(n * 0.75)
    spec = DataSpec(X=pd.DataFrame(X, columns=["a", "b"]),
                    y=pd.Series(y),
                    train_idx=np.arange(k), test_idx=np.arange(k, n),
                    target_kind="classification")
    spec.aux[AUX_GRAPH] = pd.DataFrame(edges, columns=["src", "dst"])
    spec.aux[AUX_GRAPH]["weight"] = 1.0
    # 无向图：补反向边（引擎在 prepare_dataset 里做；这里手工等价）
    rev = spec.aux[AUX_GRAPH].rename(columns={"src": "dst", "dst": "src"})
    spec.aux[AUX_GRAPH] = pd.concat([spec.aux[AUX_GRAPH], rev], ignore_index=True)

    m = C.make_custom_instance(_GCN_CODE)
    assert AUX_GRAPH in m.accepts_aux, "自定义网络必须声明支持图"
    p = m.problem(spec)
    assert p.partition_mode() == "masks"

    # ⚠ 走 `runner.run_one`（不是直接 `fit`）：指标是 `runner` 在 fit 之后调
    #   `evaluate` 算出来的 —— 直接 `fit` 拿不到 metrics（我第一版就这么写错了，
    #   断言 `metrics.get("accuracy", 0) > 0.6` 报的是 `{}`）。
    cfg = RunConfig(overrides={"epochs": 40, "lr": 0.05}, diag=True)
    rec = runner.run_one(m, spec, cfg)
    res = rec.result
    assert res.ok, res.error
    a = res.artifacts
    # ① 节点掩码语义：预测长度 = 测试节点数（不是全量、也不是切行）
    assert len(a["y_pred"]) == n - k, (len(a["y_pred"]), n - k)
    assert len(a["y_true"]) == n - k
    # ② 观测面白拿：逐 epoch 曲线 + 拓扑事实
    assert len(a["nn_history"]["loss"]) == 40, len(a["nn_history"]["loss"])
    assert a["nn_epochs_run"][0] == 40
    assert int(a["graph_n_nodes"][0]) == n
    assert int(a["graph_n_edges"][0]) == len(spec.aux[AUX_GRAPH])
    # ③ diag 说清这是图任务
    assert res.diag["task_mode"] == "graph"
    assert res.diag["n_test_nodes"] == n - k
    # ④ 指标真的算出来了（不是 -1/缺失）
    assert res.metrics.get("accuracy", 0) > 0.6, res.metrics


def test_graph_predict_refuses_new_rows_with_a_readable_reason():
    """图模型对"一批新行"预测**不成立** ⇒ 给可读理由，不是崩在 KeyError 上。

    图前向需要**整张图的边**，而 `predict(X)` 只给新行 ⇒ 语义上就没有答案。
    （测试节点的预测已随运行产出，按 test 掩码对齐。）
    """
    try:
        import torch  # noqa: F401
    except Exception:
        print("    (torch 不可用，跳过)")
        return
    from ml_toolbox.core import runner
    from ml_toolbox.methods.neural import custom as C

    X, y, edges = _sbm(n_per=12)
    n = len(y)
    spec = DataSpec(X=pd.DataFrame(X, columns=["a", "b"]), y=pd.Series(y),
                    train_idx=np.arange(int(n * 0.75)),
                    test_idx=np.arange(int(n * 0.75), n),
                    target_kind="classification")
    spec.aux[AUX_GRAPH] = pd.DataFrame(edges, columns=["src", "dst"])
    m = C.make_custom_instance(_GCN_CODE)
    res = m.fit(m.problem(spec), RunConfig(overrides={"epochs": 3, "lr": 0.05}))
    assert res.ok, res.error
    try:
        m.predict(spec.X.iloc[:5], res)
        raise AssertionError("图模型对新行预测竟然成功了")
    except NotImplementedError as exc:
        assert "整张图" in str(exc), exc


# ============================================================ 4. 时序外生列
def test_timeseries_discloses_or_uses_exogenous_columns():
    """时序：**ARIMA 真消费**外生列；不支持外生列的方法**如实披露**。

    老契约下"时序忽略外生 X"是**静默**的（README 记的已知限制）。现在
    `supports_exog` 显式声明：ARIMA=True（SARIMAX 带 exog），其余 7 个为 False
    ⇒ 带了外生列时写进 `diag["unused_features"]`，**不静默**。
    """
    from ml_toolbox.core import registry
    registry.load_builtin()
    n = 80
    t = np.arange(n)
    rng = np.random.RandomState(0)
    y = 10 + 0.5 * t + np.sin(t / 3.0) + rng.randn(n) * 0.2
    exog = pd.DataFrame({"exo": np.cos(t / 5.0)})
    spec = DataSpec(X=exog, y=pd.Series(y), train_idx=np.arange(n),
                    test_idx=np.arange(n), target_kind="regression")
    # ① 不支持外生列的方法：披露，而不是静默
    ses = registry.get("ses")
    assert ses.supports_exog is False
    r1 = ses.fit(ses.problem(spec), RunConfig())
    assert r1.ok, r1.error
    assert "unused_features" in (r1.diag or {}), r1.diag
    assert r1.diag["unused_features"] == ["exo"]
    # ② ARIMA：声明支持并真的用上。
    #    ⚠ `diag` 是 **fit 的形参**（`runner` 是 `fit(problem, cfg, diag=cfg.diag)`），
    #    不是 `RunConfig.diag` —— 写错地方会得到"开了诊断却没诊断"，我第一版就踩了。
    ar = registry.get("arima")
    assert ar.supports_exog is True
    r2 = ar.fit(ar.problem(spec),
                RunConfig(overrides={"seasonal_period": 0}), diag=True)
    assert r2.ok, r2.error
    assert (r2.diag or {}).get("exog_used") == 1, r2.diag
    assert "unused_features" not in (r2.diag or {}), "ARIMA 用上了外生列，不该报未使用"


# ============================================================ 直跑入口
#: ⭐ **内省收集**（2026-10-03 改）：此前这里是**手写**的判据清单，而门禁用
#: **脚本**方式跑本文件 ⇒ 新判据忘了加进来就**永远不会进闸**。实锤：
#: `test_ml_delivery.py::test_tool_params_stay_within_command_declaration`
#: （"工具声明与实现漂移"那条）从未在门禁里跑过。
#: ⇒ 定义即入闸，漂移不可能。（前提：本文件的 `test_*` 都是无参函数。）
_TESTS = [fn for name, fn in sorted(globals().items())
          if name.startswith("test_") and callable(fn)]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    failures = []
    for fn in _TESTS:
        try:
            fn()
            print(f"  \u2713 {fn.__name__}")
        except Exception as exc:                       # noqa: BLE001
            failures.append((fn.__name__, exc))
            print(f"  \u2717 {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(_TESTS) - len(failures)}/{len(_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
