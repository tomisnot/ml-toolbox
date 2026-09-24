# -*- coding: utf-8 -*-
"""预处理管道：有向步骤链，每步产出摘要（处理链页大卡片来源）。

- 步骤无状态，fit 统计量在 transform 时冻结并记入摘要（可复现、可检视）；
- 覆写参数走 step.params 运行级生效，不改全局默认（pitfalls #17）；
- diag=True 时每步附带中间产物（如缺失率表、缩放前后分布）。
"""
from __future__ import annotations

import abc
import copy
import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .contracts import DataSpec, infer_kind
from .dataset import Dataset

# 任务判定的唯一实现在 contracts.infer_kind（M10）；此处保留别名供
# methods/* 既有 import 路径使用，避免每个方法族各存一份判定逻辑。
_infer_kind = infer_kind


# ---------------------------------------------------------------- 步骤基类
class Step(abc.ABC):
    key: str = ""
    title: str = ""

    def __init__(self, enabled: bool = True, **params):
        self.enabled = enabled
        self.params = params          # 运行级覆写（None/缺省 = 用默认）
        self.state_: dict = {}        # fit 冻结的统计量

    @abc.abstractmethod
    def apply(self, df: pd.DataFrame, target: str | None,
              diag: bool = False) -> tuple[pd.DataFrame, dict]:
        """返回 (新 df, 摘要 dict)。摘要含大卡片关键数字。"""

    def transform(self, df: pd.DataFrame, target: str | None) -> pd.DataFrame:
        """对新数据套用 apply 阶段冻结的统计量（state_）；默认不支持。"""
        raise NotImplementedError(f"步骤 {self.key} 不支持对新数据 transform")

    def describe(self) -> dict:
        return {"key": self.key, "title": self.title,
                "enabled": self.enabled, "params": dict(self.params)}


# ---------------------------------------------------------------- 内置步骤
class MissingStep(Step):
    key, title = "missing", "缺失值处理"

    def apply(self, df, target, diag=False):
        before = int(df.isna().sum().sum())
        miss_rate = (df.isna().mean() * 100)
        drop_cols = [c for c in df.columns
                     if miss_rate[c] > self.params.get("drop_threshold", 50)]
        df = df.drop(columns=drop_cols)
        num = df.select_dtypes(include=[np.number]).columns
        cat = df.select_dtypes(exclude=[np.number]).columns
        strategy = self.params.get("strategy", "median")
        fills = {}
        for c in num:
            v = df[c].median() if strategy == "median" else df[c].mean()
            df[c] = df[c].fillna(v)
            fills[c] = float(v)
        for c in cat:
            v = df[c].mode().iloc[0] if not df[c].mode().empty else "unknown"
            df[c] = df[c].fillna(v)
            fills[c] = str(v)
        self.state_ = {"drop_cols": drop_cols, "fills": fills}
        summary = {"input_missing": before, "dropped_cols": len(drop_cols),
                   "filled_cols": len(fills), "strategy": strategy,
                   "big_num": f"缺失 {before} → 0"}
        if diag:
            summary["diag"] = {"missing_rate": miss_rate.to_dict(),
                               "fills": fills, "dropped": drop_cols}
        return df, summary

    def transform(self, df, target):
        df = df.drop(columns=[c for c in self.state_["drop_cols"]
                              if c in df.columns])
        for c, v in self.state_["fills"].items():
            if c in df.columns:
                df[c] = df[c].fillna(v)
        return df


class EncodeStep(Step):
    key, title = "encode", "类别编码"

    def apply(self, df, target, diag=False):
        cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
               and c != target]
        mode = self.params.get("mode", "onehot")
        if not cat:
            self.state_ = {"cat_cols": [], "mode": mode,
                           "columns": [c for c in df.columns if c != target]}
            return df, {"encoded_cols": 0, "mode": mode, "big_num": "无类别列"}
        out = pd.DataFrame(index=df.index)
        cats_map = {}
        for c in df.columns:
            if c in cat:
                if mode == "ordinal":
                    cats = df[c].astype("category").cat.categories
                    cats_map[c] = cats
                    codes = df[c].astype("category").cat.codes
                    out[c] = codes.astype(float)
                else:
                    d = pd.get_dummies(df[c], prefix=c, dtype=float)
                    out = pd.concat([out, d], axis=1)
            else:
                out[c] = df[c]
        n_new = int(out.shape[1] - df.shape[1] + len(cat))
        self.state_ = {"cat_cols": cat, "mode": mode,
                       "columns": [c for c in out.columns if c != target]}
        if mode == "ordinal":
            self.state_["cats"] = cats_map
        return out, {"encoded_cols": len(cat), "mode": mode,
                     "big_num": f"{len(cat)} 列 → {n_new} 列"}

    def transform(self, df, target):
        """对新数据按训练期冻结的列集合编码并对齐（缺列补 0，多列丢弃）。"""
        mode = self.state_.get("mode", "onehot")
        cat = self.state_.get("cat_cols", [])
        if not cat:
            return df.reindex(columns=self.state_["columns"], fill_value=0.0)
        out = pd.DataFrame(index=df.index)
        for c in df.columns:
            if c in cat:
                if mode == "ordinal":
                    cats = self.state_.get("cats", {}).get(c)
                    codes = pd.Categorical(df[c], categories=cats).codes
                    out[c] = np.asarray(codes, float)
                else:
                    out = pd.concat(
                        [out, pd.get_dummies(df[c], prefix=c, dtype=float)],
                        axis=1)
            else:
                out[c] = df[c]
        return out.reindex(columns=self.state_["columns"], fill_value=0.0)


class ScaleStep(Step):
    key, title = "scale", "数值缩放"

    def apply(self, df, target, diag=False):
        from sklearn.preprocessing import RobustScaler, StandardScaler
        num = [c for c in df.select_dtypes(include=[np.number]).columns
               if c != target]
        mode = self.params.get("mode", "standard")
        scaler = (StandardScaler() if mode == "standard" else RobustScaler())
        vals = df[num].to_numpy(float)
        scaled = scaler.fit_transform(vals)
        out = df.copy()
        out[num] = scaled
        self.state_ = {"mode": mode, "n_scaled": len(num),
                       "scaler": scaler, "cols": num}
        summary = {"mode": mode, "n_scaled": len(num),
                   "big_num": f"{len(num)} 列 {mode}"}
        if diag:
            summary["diag"] = {
                "mean_before": vals.mean(axis=0)[:20].tolist(),
                "mean_after": scaled.mean(axis=0)[:20].tolist()}
        return out, summary

    def transform(self, df, target):
        sc = self.state_.get("scaler")
        cols = [c for c in self.state_.get("cols", []) if c in df.columns]
        if sc is None or not cols:
            return df
        out = df.copy()
        out[cols] = sc.transform(df[cols].to_numpy(float))
        return out


class FeatureSelectStep(Step):
    key, title = "feature_select", "特征筛选"

    def apply(self, df, target, diag=False):
        if target is None or target not in df.columns:
            return df, {"big_num": "无目标列，跳过"}
        from sklearn.feature_selection import (mutual_info_classif,
                                               mutual_info_regression,
                                               f_regression,
                                               SelectKBest)
        num = [c for c in df.select_dtypes(include=[np.number]).columns
               if c != target]
        if len(num) < 3:
            return df, {"big_num": "特征过少，跳过"}
        k = int(self.params.get("top_k", 0)) or len(num)
        kind = self.params.get("score", "mi")
        X = df[num]
        y = df[target]
        is_reg = infer_kind(y) == "regression"
        if kind == "mi":
            fn = mutual_info_regression if is_reg else mutual_info_classif
            scores = fn(X.values, y.values,
                        random_state=int(self.params.get("seed", 42)))
        else:
            if not is_reg:
                from sklearn.preprocessing import LabelEncoder
                y = LabelEncoder().fit_transform(y)
            scores, _ = f_regression(X.values, y)
        scores = np.nan_to_num(np.asarray(scores, float))
        keep = [c for c, s in sorted(zip(num, scores),
                                     key=lambda t: -t[1])[:k]]
        dropped = [c for c in num if c not in keep]
        out = df.drop(columns=dropped)
        self.state_ = {"keep": keep, "scores": dict(zip(num, scores.tolist()))}
        summary = {"kept": len(keep), "dropped": len(dropped),
                   "big_num": f"保留 {len(keep)} 特征"}
        if diag:
            summary["diag"] = {"scores": self.state_["scores"]}
        return out, summary

    def transform(self, df, target):
        keep = self.state_.get("keep")
        if not keep:
            return df
        cols = [c for c in keep if c in df.columns]
        if target and target in df.columns:
            cols.append(target)
        return df[cols]


class SplitStep(Step):
    key, title = "split", "训练/测试划分"

    def apply(self, df, target, diag=False):
        # 划分不动 df 本身，摘要在此记录；实际索引由 Pipeline.run 计算
        frac = float(self.params.get("test_size", 0.25))
        return df, {"test_size": frac, "big_num": f"test = {frac:.0%}"}

    def transform(self, df, target):
        return df      # 划分不改变数据本身，对新数据是 no-op


BUILTIN_STEPS = {s.key: s for s in
                 (MissingStep, EncodeStep, ScaleStep, FeatureSelectStep, SplitStep)}


# ---------------------------------------------------------------- 管道
@dataclass
class Pipeline:
    steps: list = field(default_factory=list)
    seed: int = 42
    test_size: float = 0.25
    stratify: bool = True
    time_split: bool = False              # 时序：按时间顺序切分，不打乱

    @classmethod
    def default(cls) -> "Pipeline":
        return cls(steps=[MissingStep(), EncodeStep(), ScaleStep(), SplitStep()])

    def fit(self, ds: Dataset, diag: bool = False) -> "FittedPipeline":
        """Fit an isolated pipeline copy and return a reusable handle.

        The legacy :meth:`run` method remains mutable for compatibility, while
        this entry point prevents a later ``fit`` on the same Pipeline object
        from overwriting the step state used by ``transform_new``.
        """
        fitted = copy.deepcopy(self)
        spec = fitted.run(ds, diag=diag)
        return FittedPipeline(fitted, spec)

    def run(self, ds: Dataset, diag: bool = False) -> DataSpec:
        """Dataset -> DataSpec（X, y, 划分索引, 处理链摘要）。"""
        df = ds.frame.copy()
        chain = []
        for st in self.steps:
            if not st.enabled:
                chain.append({"step": st.describe(), "skipped": True})
                continue
            df, summary = st.apply(df, ds.target, diag=diag)
            entry = {"step": st.describe(), "summary": summary}
            chain.append(entry)

        y = df[ds.target] if ds.target and ds.target in df.columns else None
        X = df.drop(columns=[c for c in (ds.target,) if c and c in df.columns])
        if ds.time_col and ds.time_col in X.columns:
            X = X.drop(columns=[ds.time_col])

        train_idx, test_idx = self._split(df, y)
        target_kind = _infer_kind(y)
        spec = DataSpec(
            X=X.reset_index(drop=True),
            y=y.reset_index(drop=True) if y is not None else None,
            train_idx=train_idx, test_idx=test_idx,
            target_kind=target_kind,
            n_classes=int(y.nunique()) if y is not None and target_kind == "classification" else 0,
            meta={"chain": chain, "pipeline_id": self.fingerprint(),
                  "dataset": ds.name, "source": ds.source,
                  "n_features": int(X.shape[1]), "n_samples": int(len(df))},
        )
        return spec

    def transform_new(self, frame: pd.DataFrame,
                      target: str | None = None) -> pd.DataFrame:
        """对新数据套用 run() 时冻结的统计量，输出与 spec.X 同构的特征矩阵。
        （数模场景：训练集跑管道 -> 测试集同变换 -> 模型预测 -> 交表）"""
        df = frame.copy()
        for st in self.steps:
            if not st.enabled:
                continue
            df = st.transform(df, target)
        # 与 run() 相同的列剔除
        if target and target in df.columns:
            df = df.drop(columns=[target])
        return df

    def _split(self, df, y):
        n = len(df)
        if self.time_split:
            k = int(n * (1 - self.test_size))
            return np.arange(k), np.arange(k, n)
        from sklearn.model_selection import train_test_split
        idx = np.arange(n)
        strat = y if (self.stratify and y is not None
                      and _infer_kind(y) == "classification") else None
        tr, te = train_test_split(idx, test_size=self.test_size,
                                  random_state=self.seed, stratify=strat)
        return tr, te

    def fingerprint(self) -> str:
        blob = json.dumps([s.describe() for s in self.steps]
                          + [self.test_size, self.stratify, self.time_split],
                          sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class FittedPipeline:
    """Immutable-by-ownership handle for one prepared pipeline state.

    The contained Pipeline is private and deep-copied by :meth:`Pipeline.fit`.
    Callers should use this handle for ``transform_new`` rather than reaching
    into the original mutable Pipeline's step objects.
    """

    _pipeline: Pipeline = field(repr=False)
    spec: DataSpec

    @property
    def pipeline_id(self) -> str:
        return str(self.spec.meta.get("pipeline_id", ""))

    def transform_new(self, frame: pd.DataFrame,
                      target: str | None = None) -> pd.DataFrame:
        return self._pipeline.transform_new(frame, target)

    def fingerprint(self) -> str:
        return self.pipeline_id or self._pipeline.fingerprint()
