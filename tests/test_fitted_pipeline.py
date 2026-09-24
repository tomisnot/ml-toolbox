# -*- coding: utf-8 -*-
"""FittedPipeline state-isolation tests."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_toolbox.core import Dataset, Pipeline
from ml_toolbox.core.pipeline import FittedPipeline


def _dataset(mean: float, name: str) -> Dataset:
    rng = np.random.RandomState(1)
    frame = pd.DataFrame({"a": rng.randn(40) + mean,
                          "b": rng.randn(40) + mean,
                          "target": rng.randn(40)})
    return Dataset(frame, name=name, target="target")


def test_fitted_pipeline_isolated_between_fits():
    p = Pipeline.default()
    fp1 = p.fit(_dataset(0.0, "a"), diag=False)
    fp2 = p.fit(_dataset(100.0, "b"), diag=False)
    assert isinstance(fp1, FittedPipeline)
    assert fp1.pipeline_id and fp2.pipeline_id
    # The original builder was not fitted/mutated by the convenience API.
    assert all(not getattr(step, "state_", {}) for step in p.steps)
    x1 = fp1.transform_new(pd.DataFrame({"a": [0.0], "b": [0.0]}))
    x2 = fp2.transform_new(pd.DataFrame({"a": [100.0], "b": [100.0]}))
    assert abs(float(x1.iloc[0].mean())) < 0.2
    assert abs(float(x2.iloc[0].mean())) < 0.2
    # Both handles remain usable after the other fit.
    assert fp1.transform_new(pd.DataFrame({"a": [0.0], "b": [0.0]})).shape[1] == 2
    assert fp2.transform_new(pd.DataFrame({"a": [100.0], "b": [100.0]})).shape[1] == 2


def test_fitted_pipeline_exposes_stable_id():
    p = Pipeline.default()
    fp = p.fit(_dataset(3.0, "c"))
    assert fp.fingerprint() == fp.pipeline_id
    assert fp.spec.meta["pipeline_id"] == fp.pipeline_id


def main():
    tests = [test_fitted_pipeline_isolated_between_fits,
             test_fitted_pipeline_exposes_stable_id]
    for fn in tests:
        fn()
        print("PASS", fn.__name__)
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
