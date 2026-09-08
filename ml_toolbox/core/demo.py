# -*- coding: utf-8 -*-
"""内置演示数据集（离线可用，覆盖数模常见题型）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import Dataset


def demo_names() -> list[str]:
    return ["iris", "wine", "digits", "breast_cancer", "housing_small",
            "blobs", "anomaly_synth", "timeseries_synth"]


def load_demo(name: str) -> Dataset:
    if name == "breast_cancer":
        # 真实医学分类数据（数模常用），30 特征二分类
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer(as_frame=True)
        return Dataset(d.frame.rename(columns={"target": "target"}),
                       name="breast_cancer", target="target",
                       source="demo:breast_cancer")
    if name == "iris":
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        return Dataset(d.frame.rename(columns={"target": "target"}),
                       name="iris", target="target", source="demo:iris")
    if name == "wine":
        from sklearn.datasets import load_wine
        d = load_wine(as_frame=True)
        return Dataset(d.frame.rename(columns={"target": "target"}),
                       name="wine", target="target", source="demo:wine")
    if name == "digits":
        from sklearn.datasets import load_digits
        d = load_digits(as_frame=True)
        return Dataset(d.frame.rename(columns={"target": "target"}),
                       name="digits", target="target", source="demo:digits")
    if name == "housing_small":
        # 真实回归题抽样 2000 行，保证 UI 秒级响应
        try:
            from sklearn.datasets import fetch_california_housing
            d = fetch_california_housing()
        except Exception:
            d = None
        if d is not None:
            df = pd.DataFrame(d.data, columns=d.feature_names)
            df["target"] = d.target
            df = df.sample(2000, random_state=0).reset_index(drop=True)
            return Dataset(df, name="housing", target="target",
                           source="demo:housing")
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=2000, n_features=8, noise=12,
                               random_state=0)
        return Dataset.from_arrays(X, y, name="reg_synth")
    if name == "blobs":
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=300, centers=4, random_state=3)
        return Dataset.from_arrays(X, None, name="blobs")
    if name == "anomaly_synth":
        rng = np.random.RandomState(7)
        X = rng.randn(400, 5)
        X[:30] += 4
        lab = np.zeros(400); lab[:30] = 1
        return Dataset.from_arrays(X, lab, name="anomaly")
    if name == "timeseries_synth":
        t = np.arange(120, dtype=float)
        y = 200 + 2.0 * t + 40 * np.sin(2 * np.pi * t / 12) \
            + np.random.RandomState(2).randn(120) * 5
        return Dataset.from_arrays(y.reshape(-1, 1), y, name="ts")
    raise ValueError(f"未知演示数据集: {name}")
