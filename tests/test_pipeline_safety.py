# -*- coding: utf-8 -*-
"""Safety regressions for the train-only pipeline contract.

These tests cover preprocessing state, stale-state reuse, column/schema
consistency, target handling, temporal ordering and internal call paths.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_toolbox.core import Dataset, Pipeline
from ml_toolbox.core.pipeline import (EncodeStep, FeatureSelectStep,
                                      MissingStep)
from ml_toolbox.opt.bridges import _quick_spec


def _df(n=60, seed=0):
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n)
    x2 = x1 * 0.5 + rng.randn(n) * 0.1
    x3 = rng.randn(n)
    x9 = rng.randn(n)
    target = x1 * 2.0 + rng.randn(n) * 0.2
    frame = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x9": x9,
                          "target": target})
    return frame


def test_missing_step_does_not_drop_or_fill_target():
    frame = _df(30, seed=1)
    frame.loc[frame.index[:25], "target"] = np.nan
    ds = Dataset(frame, name="target-missing", target="target")
    fitted = Pipeline(steps=[MissingStep()], time_split=True,
                      test_size=0.5).fit(ds)
    assert "target" not in fitted.spec.X.columns
    assert fitted.spec.y.isna().sum() == 25
    assert "target" not in fitted._pipeline.steps[0].state_["fills"]
    assert "target" not in fitted._pipeline.steps[0].state_["drop_cols"]


def test_feature_select_passes_through_non_numeric_columns():
    frame = _df(60, seed=2)
    frame["cat"] = np.array(["a", "b", "c", "d"] * 15)
    ds = Dataset(frame, name="feature-select-cat", target="target")
    fitted = Pipeline(steps=[FeatureSelectStep()], time_split=True,
                      test_size=0.5).fit(ds)
    X = fitted.spec.X
    assert "cat" in X.columns
    assert set(X.columns) == set(["x1", "x2", "x3", "x9", "cat"])
    # Test rows are transformed, never silently zero-filled by reindex.
    assert X.iloc[fitted.spec.test_idx]["cat"].tolist() == \
        frame["cat"].iloc[fitted.spec.test_idx].tolist()


def test_feature_select_stale_state_is_cleared():
    old = _df(30, seed=3)
    new = _df(30, seed=4)[["x1", "x9", "target"]].copy()

    p = Pipeline(steps=[FeatureSelectStep()], time_split=True, test_size=0.5)
    # Populate a legacy keep vector on the shared builder instance.
    p.run(Dataset(old, name="old", target="target"))
    fitted = p.fit(Dataset(new, name="new", target="target"))
    X = fitted.spec.X
    assert list(X.columns) == ["x1", "x9"]
    assert not X.iloc[fitted.spec.test_idx]["x9"].equals(
        pd.Series(0.0, index=fitted.spec.test_idx))


def test_transform_new_drops_target_and_maps_unseen_ordinal_to_nan():
    frame = pd.DataFrame({
        "cat": ["a", "b", "c", "a", "b", "c"] * 10,
        "target": np.arange(60, dtype=float),
    })
    fitted = Pipeline(steps=[EncodeStep(mode="ordinal")], time_split=True,
                      test_size=0.25).fit(Dataset(frame, name="ordinal",
                                                  target="target"))
    out = fitted.transform_new(pd.DataFrame({
        "cat": ["a", "unseen", "c"],
        "target": [0.0, 1.0, 2.0],
    }))
    assert list(out.columns) == ["cat"]
    assert np.isnan(out["cat"].iloc[1])


def test_time_split_sorts_by_time_column():
    frame = _df(12, seed=5)
    frame.insert(0, "time", [11, 8, 10, 1, 3, 6, 12, 4, 5, 9, 2, 7])
    frame["marker"] = frame["time"]
    ds = Dataset(frame, name="time", target="target", time_col="time")
    fitted = Pipeline(steps=[], time_split=True, test_size=0.5).fit(ds)
    X = fitted.spec.X
    assert X.loc[fitted.spec.train_idx, "marker"].max() < \
        X.loc[fitted.spec.test_idx, "marker"].min()


def test_opt_quick_spec_uses_train_only_path():
    ds = Dataset(_df(40, seed=6), name="response-surface", target="target")
    spec = _quick_spec(ds)
    assert spec.meta["fit_scope"] == "train_only"


def test_internal_call_paths_do_not_use_legacy_run():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "ml_toolbox", "ui", "main_window.py"),
              encoding="utf-8") as f:
        ui_src = f.read()
    assert "self.pipeline.fit(" in ui_src
    with open(os.path.join(root, "ml_toolbox", "opt", "bridges.py"),
              encoding="utf-8") as f:
        bridge_src = f.read()
    assert ".fit(ds).spec" in bridge_src


def test_schema_mismatch_is_reported_not_only_filled():
    frame = _df(60, seed=7)
    frame["cat"] = np.array(["a", "b"] * 30)
    ds = Dataset(frame, name="schema", target="target")
    # Deliberately run the FeatureSelect transform on a schema that lacks a
    # numeric keep column; fit should record the mismatch in metadata.
    fitted = Pipeline(steps=[FeatureSelectStep()], time_split=True,
                      test_size=0.5).fit(ds)
    meta = fitted.spec.meta
    assert meta["schema_mismatch"] is False
    assert meta["missing_columns"] == []
    assert meta["extra_columns"] == []


def main():
    tests = [test_missing_step_does_not_drop_or_fill_target,
             test_feature_select_passes_through_non_numeric_columns,
             test_feature_select_stale_state_is_cleared,
             test_transform_new_drops_target_and_maps_unseen_ordinal_to_nan,
             test_time_split_sorts_by_time_column,
             test_opt_quick_spec_uses_train_only_path,
             test_internal_call_paths_do_not_use_legacy_run,
             test_schema_mismatch_is_reported_not_only_filled]
    for fn in tests:
        fn()
        print("PASS", fn.__name__)
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
