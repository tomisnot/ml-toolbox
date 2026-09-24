# -*- coding: utf-8 -*-
"""Stable Session facade tests (headless, no Qt, no AI integration)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_toolbox.api import (API_VERSION, RunRequest, RunState, Session,
                            SessionError, TerminalReason)
from ml_toolbox.core.contracts import RunConfig
from ml_toolbox.core.dataset import Dataset
from ml_toolbox.core.pipeline import Pipeline


def _dataset():
    rng = np.random.RandomState(3)
    x = rng.randn(80, 4)
    y = (x[:, 0] + 0.2 * x[:, 1] > 0).astype(int)
    frame = pd.DataFrame(x, columns=list("abcd"))
    frame["target"] = y
    return Dataset(frame, name="api_test", target="target")


def test_session_methods_prepare_and_run():
    s = Session(seed=17)
    methods = s.list_methods()
    assert methods and any(m["name"] == "logistic" for m in methods)
    assert s.describe()["api_version"] == API_VERSION

    spec = s.prepare(_dataset(), pipeline=Pipeline.default())
    req = RunRequest(method="logistic", overrides={"C": 1.0}, seed=17)
    rec = s.run(spec, req)
    assert rec.result.ok, rec.result.error
    snap = s.state(req.request_id)
    assert snap.state is RunState.SUCCEEDED
    assert snap.terminal_reason is TerminalReason.COMPLETED
    assert snap.run_id == rec.run_id
    assert s.record(rec.run_id) is rec

    pred = s.predict(rec, spec.X.head(5))
    assert len(pred) == 5
    assert s.last_state().state is RunState.SUCCEEDED


def test_session_method_failure_and_resource_guard():
    s = Session()
    spec = s.prepare(_dataset(), pipeline=Pipeline.default())
    req = RunRequest(method="svc", extras={"resource_guard": {
        "max_kernel_mb": 0.000001,
    }})
    rec = s.run(spec, req)
    assert not rec.result.ok
    snap = s.state(req.request_id)
    assert snap.state is RunState.FAILED
    assert snap.terminal_reason is TerminalReason.RESOURCE_LIMIT

    bad = RunRequest(method="not_registered")
    try:
        s.run(spec, bad)
    except SessionError:
        pass
    else:
        raise AssertionError("unknown method should raise SessionError")
    assert s.state(bad.request_id).terminal_reason is TerminalReason.CONFIG_ERROR


def test_session_batch_and_explicit_save():
    s = Session()
    spec = s.prepare(_dataset(), pipeline=Pipeline.default())
    progress = []
    records = s.run_batch(spec, ["logistic", "svc", "not_registered"],
                           request=RunRequest(overrides={"max_iter": 50}),
                           progress=lambda i, n, name: progress.append((i, n, name)),
                           saver=lambda rec: "saved")
    assert [r.method for r in records] == ["logistic", "svc"]
    assert progress == [(0, 3, "logistic"), (1, 3, "svc"), (2, 3, "not_registered")]
    assert all(r.timings.get("serialize", 0) >= 0 for r in records)


def test_run_config_conversion_isolated():
    overrides = {"C": 2.0}
    extras = {"cv_folds": 2}
    req = RunRequest(method="logistic", overrides=overrides, extras=extras)
    overrides["C"] = 99.0
    extras["cv_folds"] = 99
    cfg = req.config()
    assert cfg.overrides["C"] == 2.0
    assert cfg.extras["cv_folds"] == 2


def main():
    tests = [test_session_methods_prepare_and_run,
             test_session_method_failure_and_resource_guard,
             test_session_batch_and_explicit_save,
             test_run_config_conversion_isolated]
    for fn in tests:
        fn()
        print("PASS", fn.__name__)
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
