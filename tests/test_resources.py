# -*- coding: utf-8 -*-
"""core.resources 与 runner 分段计时的轻量回归测试。

运行：python tests/test_resources.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ml_toolbox.core import runner
from ml_toolbox.core.contracts import DataSpec, MLMethod, RunConfig
from ml_toolbox.core.resources import (ResourceBudget, ResourceBudgetExceeded,
                                       check_resources, estimate_resources)


class _TimedProbe(MLMethod):
    name = "timed_probe"
    display_name = "计时探针"
    family = "linear"
    task = "supervised"
    target_kind = "regression"
    param_schema = []

    def fit(self, X, y, cfg, diag=False):
        time.sleep(0.001)
        return self._new_result(params={}, target_kind="regression")

    def predict(self, X, result):
        time.sleep(0.001)
        return np.zeros(len(X), dtype=float)

    def evaluate(self, result, X, y):
        time.sleep(0.001)
        return {"rmse": 0.0}

    def cross_validate(self, X, y, cfg, cv_folds=5):
        time.sleep(0.001)
        return {"cv_rmse_mean": 0.0, "cv_scores": [0.0] * cv_folds}


class _KernelProbe(MLMethod):
    name = "gpr"
    display_name = "资源守卫探针"
    family = "linear"
    task = "supervised"
    target_kind = "regression"
    param_schema = []
    fitted = False

    def fit(self, X, y, cfg, diag=False):
        type(self).fitted = True
        return self._new_result(params={}, target_kind="regression")


def _spec(n=20):
    X = pd.DataFrame(np.arange(n * 2, dtype=float).reshape(n, 2))
    y = pd.Series(np.arange(n, dtype=float))
    return DataSpec(X=X, y=y, train_idx=np.arange(0, n - 4),
                    test_idx=np.arange(n - 4, n), target_kind="regression",
                    meta={"dataset": "probe"})


def test_resource_estimates():
    est = estimate_resources("gpr", 1000, 8)
    assert est.complexity == "O(n^3)"
    assert est.kernel_elements == 1_000_000
    assert est.kernel_mb > 7
    assert est.operation_units == 1e9

    q = estimate_resources(
        "qreservoir", 200, 6,
        params={"n_qubits": 6, "virtual": 3},
    )
    assert q.quantum and q.n_qubits == 6
    assert q.density_matrix_bytes == 16 * (2 ** 6) ** 2
    assert q.operation_units > 0

    p = estimate_resources("qpca", 200, 12,
                           params={"n_iter": 10})
    assert p.density_matrix_bytes > 0
    assert p.complexity == "O(n_iter*d^3)"

    qk = estimate_resources("qkmeans", 10000, 12,
                            params={"n_clusters": 3})
    assert qk.n_qubits == 4  # ceil(log2(max(12, 2)))
    assert qk.kernel_elements == 30000
    try:
        check_resources("qkmeans", 10000, 12,
                        params={"n_clusters": 3},
                        budget=ResourceBudget(max_kernel_mb=0.0001))
    except ResourceBudgetExceeded:
        pass
    else:
        raise AssertionError("qkmeans 超限资源未触发 guard")

    for method, n, d, params, budget in (
        ("qsvc", 5000, 4, {}, ResourceBudget(max_kernel_mb=0.1)),
        ("qreservoir", 100, 6, {"n_qubits": 6, "virtual": 3},
         ResourceBudget(max_quantum_state_mb=0.00001)),
    ):
        try:
            check_resources(method, n, d, params=params, budget=budget)
        except ResourceBudgetExceeded:
            pass
        else:
            raise AssertionError(f"{method} 超限资源未触发 guard")

    try:
        check_resources("gpr", 2000, 4,
                        budget=ResourceBudget(max_kernel_mb=0.001,
                                              max_operation_units=1e12))
    except ResourceBudgetExceeded as e:
        assert e.estimate.method == "gpr"
        assert e.estimate.violations
    else:
        raise AssertionError("超限资源未触发 guard")


def test_runner_stage_timings_and_serialize():
    rec = runner.run_one(
        _TimedProbe(), _spec(), RunConfig(extras={"cv_folds": 2}),
    )
    assert rec.result.ok
    assert rec.resource_estimate is None
    for key in ("guard", "split", "fit", "predict", "evaluate", "cv",
                "legacy_elapsed", "total"):
        assert key in rec.timings, key
    assert rec.timings["fit"] > 0
    assert rec.timings["predict"] > 0
    assert rec.timings["evaluate"] > 0
    assert rec.timings["cv"] > 0
    assert rec.timings["total"] >= rec.result.elapsed
    assert rec.timings["legacy_elapsed"] == rec.result.elapsed

    marker = object()
    result = runner.save_timed(rec, saver=lambda _rec: marker)
    assert result is marker
    assert rec.timings["serialize"] >= 0


def test_optional_runner_resource_guard():
    _KernelProbe.fitted = False
    cfg = RunConfig(extras={"resource_guard": {
        "max_kernel_mb": 0.001,
        "max_operation_units": 1e12,
    }})
    rec = runner.run_one(_KernelProbe(), _spec(), cfg)
    assert not rec.result.ok
    assert "资源预算超限" in (rec.result.error or "")
    assert not _KernelProbe.fitted
    assert rec.resource_estimate is not None
    assert rec.resource_estimate.violations


def main():
    tests = [test_resource_estimates,
             test_runner_stage_timings_and_serialize,
             test_optional_runner_resource_guard]
    for fn in tests:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"✅ resources/timing tests passed ({len(tests)})")


if __name__ == "__main__":
    main()
