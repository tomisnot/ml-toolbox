# -*- coding: utf-8 -*-
"""Torch recorder 性能契约测试：一个 probe forward + 冻结 latent PCA。

运行：python tests/test_nn_perf.py
只验证确定性计数/工件结构，不用脆弱的墙钟阈值。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:                                   # optional dependency
    TORCH_AVAILABLE = False


def test_one_probe_forward_and_frozen_pca():
    assert TORCH_AVAILABLE, "torch 不可用"
    from ml_toolbox.methods.neural.recorder import TrainingRecorder

    torch.manual_seed(7)
    # 两个隐藏层且启用 Dropout：同时覆盖 Linear→ReLU 的模块映射。
    model = nn.Sequential(
        nn.Linear(4, 6), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(6, 5), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(5, 3),
    )
    epochs, probe_n = 5, 20
    rec = TrainingRecorder(model, n_samples=probe_n, epochs=epochs,
                           latent_every=2, max_latent_samples=10)
    probe = torch.randn(probe_n, 4)
    labels = np.arange(probe_n) % 2
    rows = np.arange(probe_n) + 100
    rec.set_probe(probe, labels=labels, probe_idx=rows)

    calls = {"n": 0}

    def count_forward(_module, _inputs, _output):
        calls["n"] += 1
    handle = model.register_forward_hook(count_forward)
    try:
        for ep in range(epochs):
            rec.epoch(ep, 1.0 / (ep + 1), None, 1e-3)
    finally:
        handle.remove()
        rec.close()

    # 旧实现 latent_every=2 时应有 8 次 probe forward（5 dead + 3 latent）；
    # 合并后严格每 epoch 一次。
    assert calls["n"] == epochs
    assert rec.probe_forward_count == epochs
    assert rec.pca_fit_count == 1
    assert rec._latent_pca is not None
    assert rec._latent_pca.n_components == 2

    art = rec.finalize()
    assert art["nn_history"]["loss"].shape == (epochs,)
    assert art["nn_history"]["dead_relu"].shape == (epochs,)
    # epochs=0/2/4 三帧，统一 max_latent=10、PCA 两维。
    assert art["nn_latent"].shape == (3, 10, 2)
    assert np.array_equal(art["nn_latent_epochs"], [0, 2, 4])
    assert rec.last_labels.shape == (10,)
    assert rec.last_idx.shape == (10,)
    assert np.array_equal(rec.last_idx, rows[rec._probe_selection])

    # Dropout 下旧 idx//2 会错位；新映射中两个 Linear 都有值，输出层为 NaN。
    health = art["nn_layers"]["死ReLU比例"]
    assert np.isfinite(health.loc["0.weight"])
    assert np.isfinite(health.loc["3.weight"])
    assert np.isnan(health.loc["6.weight"])


def test_torch_mlp_artifacts_and_metrics_survive():
    assert TORCH_AVAILABLE, "torch 不可用"
    import pandas as pd
    from sklearn.datasets import make_classification
    from ml_toolbox.core.contracts import RunConfig
    from ml_toolbox.core.registry import get, load_builtin
    from ml_toolbox.methods.neural.mlp import TorchMLP

    load_builtin()
    X, y = make_classification(n_samples=96, n_features=6, n_informative=4,
                               n_redundant=0, random_state=3)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y)
    method = TorchMLP()
    result = method.fit(X, y, RunConfig(seed=11, overrides={
        "hidden": "8,4", "epochs": 6, "batch": 32, "val_split": 0,
        "latent_every": 2, "device": "cpu",
    }))

    assert result.ok, result.error
    pred = method.predict(X, result)
    result.artifacts.update(method.fit_extra_artifacts(X, result))
    result.artifacts["y_true"] = np.asarray(y)
    result.artifacts["y_pred"] = np.asarray(pred)
    metrics = method.evaluate(result, X, y)

    assert pred.shape == (len(y),)
    assert np.isfinite(metrics["f1"]) and 0 <= metrics["f1"] <= 1
    for key in ("nn_history", "nn_grads", "nn_weights", "nn_latent",
                "nn_latent_epochs", "nn_layers"):
        assert key in result.artifacts
    assert result.artifacts["nn_latent"].ndim == 3
    assert result.artifacts["nn_latent"].shape[2] == 2
    assert result.artifacts["nn_history"]["loss"].shape == (6,)
    assert result.artifacts["nn_history"]["dead_relu"].shape == (6,)


def main():
    tests = [test_one_probe_forward_and_frozen_pca,
             test_torch_mlp_artifacts_and_metrics_survive]
    if not TORCH_AVAILABLE:
        print("SKIP torch 不可用：2 项未执行（不计通过）")
        return 0
    passed = 0
    for fn in tests:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"{passed}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
