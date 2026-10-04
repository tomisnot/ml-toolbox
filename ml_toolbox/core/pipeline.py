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
        # Target is excluded from drop/fill: preprocessing must not delete or
        # impute the label column.
        target_col = df[target].copy() if target is not None and target in df.columns else None
        candidate = df.drop(columns=[target] if target is not None and target in df.columns else [])
        before = int(candidate.isna().sum().sum())
        miss_rate = (candidate.isna().mean() * 100)
        drop_cols = [c for c in candidate.columns
                     if miss_rate[c] > self.params.get("drop_threshold", 50)]
        num = candidate.select_dtypes(include=[np.number]).columns
        cat = candidate.select_dtypes(exclude=[np.number]).columns
        strategy = self.params.get("strategy", "median")
        fills = {}
        for c in num:
            v = candidate[c].median() if strategy == "median" else candidate[c].mean()
            candidate[c] = candidate[c].fillna(v)
            fills[c] = float(v)
        for c in cat:
            v = candidate[c].mode().iloc[0] if not candidate[c].mode().empty else "unknown"
            candidate[c] = candidate[c].fillna(v)
            fills[c] = str(v)
        df = candidate
        if target_col is not None:
            df[target] = target_col
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
            if c in df.columns and c != target:
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
                    codes = np.asarray(pd.Categorical(df[c], categories=cats).codes, float)
                    codes[codes < 0] = np.nan
                    out[c] = codes
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
        # Clear state first: a skipped fit must never expose a previous keep
        # vector to transform (stale state caused train/test column mismatch).
        self.state_ = {}
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
        passthrough = [c for c in df.columns if c != target and c not in num]
        out = df.drop(columns=dropped)
        self.state_ = {"keep": keep, "passthrough": passthrough,
                       "scores": dict(zip(num, scores.tolist()))}
        summary = {"kept": len(keep), "dropped": len(dropped),
                   "big_num": f"保留 {len(keep)} 特征"}
        if diag:
            summary["diag"] = {"scores": self.state_["scores"]}
        return out, summary

    def transform(self, df, target):
        keep = self.state_.get("keep")
        if not keep:
            return df
        passthrough = self.state_.get("passthrough", [])
        cols = [c for c in df.columns if c in keep or c in passthrough]
        if target and target in df.columns and target not in cols:
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
    #: **按列分组留出**：`group_by` 是列名，`holdout` 是要留出的取值列表。
    #: 为什么需要它（NQS 需求）：要"训练集内 vs **尺寸外推**分开看"——随机切分
    #: 会把外推尺寸的一部分也放进训练，那个分数就不是外推了。
    #: ⚠ 它也是**堵同源泄漏**的正规手段（同组不许跨训练/测试）。
    group_by: str = ""
    holdout: tuple = ()

    @classmethod
    def default(cls) -> "Pipeline":
        return cls(steps=[MissingStep(), EncodeStep(), ScaleStep(), SplitStep()])

    def fit(self, ds: Dataset, diag: bool = False) -> "FittedPipeline":
        """Fit an isolated, train-only pipeline state and return its handle.

        Unlike legacy :meth:`run`, stateful steps learn only from the training
        partition; the test partition is transformed with the frozen state.
        """
        fitted = copy.deepcopy(self)
        spec = fitted._fit_split_aware(ds, diag=diag)
        return FittedPipeline(fitted, spec)

    def _fit_split_aware(self, ds: Dataset, diag: bool = False) -> DataSpec:
        """Internal split-before-fit implementation used by :meth:`fit`."""
        raw = ds.frame.copy()
        if self.time_split and ds.time_col and ds.time_col in raw.columns:
            # A time split is only valid when rows are chronological.
            raw = raw.sort_values(ds.time_col, kind="mergesort")
        raw = raw.reset_index(drop=True)
        y_raw = raw[ds.target] if ds.target and ds.target in raw.columns else None
        train_idx, test_idx = self._split(raw, y_raw)
        train = raw.iloc[train_idx].copy()
        test = raw.iloc[test_idx].copy()
        if ds.target and ds.target in test.columns:
            test = test.drop(columns=[ds.target])
        chain = []
        for st in self.steps:
            if not st.enabled:
                chain.append({"step": st.describe(), "skipped": True})
                continue
            train, summary = st.apply(train, ds.target, diag=diag)
            test = st.transform(test, None)
            chain.append({"step": st.describe(), "summary": summary})
        target_cols = [ds.target] if ds.target else []
        if ds.time_col:
            target_cols.append(ds.time_col)
        X_train = train.drop(columns=[c for c in target_cols if c and c in train.columns])
        X_test = test.drop(columns=[c for c in target_cols if c and c in test.columns])
        missing_cols = [c for c in X_train.columns if c not in X_test.columns]
        extra_cols = [c for c in X_test.columns if c not in X_train.columns]
        schema_mismatch = bool(missing_cols or extra_cols)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
        X = pd.concat([X_train, X_test]).sort_index()
        y_parts = []
        if y_raw is not None:
            y_parts = [y_raw.iloc[train_idx], y_raw.iloc[test_idx]]
            y = pd.concat(y_parts).sort_index()
        else:
            y = None
        target_kind = _infer_kind(y)
        return DataSpec(
            X=X.reset_index(drop=True), y=y.reset_index(drop=True) if y is not None else None,
            train_idx=train_idx, test_idx=test_idx, target_kind=target_kind,
            n_classes=int(y.nunique()) if y is not None and target_kind == "classification" else 0,
            meta={"chain": chain, "pipeline_id": self.fingerprint(),
                  "dataset": ds.name, "source": ds.source,
                  "n_features": int(X.shape[1]), "n_samples": int(len(raw)),
                  "fit_scope": "train_only",
                  # ⭐ 记下**实际生效**的划分口径（取自 Pipeline 本体，不是步骤参数）。
                  #   为什么落进 meta：AI 现在可以**逐数据集**指定切分（`split` 声明），
                  #   而 `describe_dataset` 要报的是"这一份数据当时怎么切的"——
                  #   只报引擎默认值会说错（见 pitfalls A2）。
                  "split_policy": self._split_policy(),
                  "schema_mismatch": schema_mismatch,
                  "missing_columns": missing_cols,
                  "extra_columns": extra_cols},
        )

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
                  "n_features": int(X.shape[1]), "n_samples": int(len(df)),
                  "split_policy": self._split_policy()},
        )
        return spec

    def _split_policy(self) -> dict:
        """**实际生效**的划分口径（单一真源 = Pipeline 本体的字段）。

        为什么要单独一个方法：`SplitStep.apply` 摘要里那个 `test_size` 取自
        **步骤参数**，与真实划分**解耦**（pitfalls A2）⇒ 展示层必须有一个
        权威来源。这个方法就是它，两个入口（`run` / `_fit_split_aware`）共用。
        """
        out = {"test_size": float(self.test_size), "stratify": bool(self.stratify),
               "seed": int(self.seed), "time_split": bool(self.time_split)}
        if self.group_by:
            # ⭐ **分组留出**（NQS：按 N 留出 14/16/18 看"尺寸外推"）
            out["group_by"] = str(self.group_by)
            out["holdout"] = list(self.holdout)
        return out

    def transform_new(self, frame: pd.DataFrame,
                      target: str | None = None) -> pd.DataFrame:
        """对新数据套用 run() 时冻结的统计量，输出与 spec.X 同构的特征矩阵。
        （数模场景：训练集跑管道 -> 测试集同变换 -> 模型预测 -> 交表）"""
        df = frame.copy()
        # The output schema is feature-only; never impute or pass through the
        # target column, even if the caller supplies one.
        if target and target in df.columns:
            df = df.drop(columns=[target])
        for st in self.steps:
            if not st.enabled:
                continue
            df = st.transform(df, None)
        return df

    def _split(self, df, y):
        n = len(df)
        # ⭐ **按列分组留出**（`group_by` + `holdout`）——优先于其它口径。
        #
        # 为什么需要它（NQS 需求原文）："**训练集内 vs 尺寸外推分开看**：N≤12 参与
        # 训练、N≥14 没见过，我要盯外推有没有崩"。随机切分做不到这件事——它会把
        # N=14 的一部分也放进训练 ⇒ 那个"外推分数"根本不是外推。
        # ⚠ 它顺手堵住一整族**同源泄漏**：同一组（同一物理尺寸/同一来源/同一病人）
        # 被切开训练与测试，与时序的"未来泄漏"、图的"邻居泄漏"是同一类问题。
        if self.group_by:
            col = str(self.group_by)
            cols = list(getattr(df, "columns", []))
            if col not in cols:
                raise ValueError(
                    f"split.group_by={col!r} 不在数据表的列里（现有列：{cols[:12]}）")
            hold = {v if not isinstance(v, str) else v for v in (self.holdout or ())}
            # 容错：JSON 传进来的数字/字符串都可能，按"字符串形式相等"匹配一次
            raw = list(df[col])
            want = {str(v) for v in hold}
            te = np.asarray([i for i, v in enumerate(raw) if str(v) in want], dtype=int)
            tr = np.asarray([i for i, v in enumerate(raw) if str(v) not in want],
                            dtype=int)
            if len(te) == 0 or len(tr) == 0:
                raise ValueError(
                    f"分组留出把数据切空了：留出值 {sorted(want)} 命中 {len(te)} 行、"
                    f"剩 {len(tr)} 行训练。该列实际取值："
                    f"{sorted({str(v) for v in raw})[:12]}")
            return tr, te
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


# ================================================================ 管道装配（AI 组装预处理链）
#: "内置步骤 × 可调参数"的**单一真源**（工具 schema 的 `choices` 从这里取）。
#: ⚠ 为什么手写而不是反射：`Step.apply` 里的默认值是**读参数时内联写死的**
#: （`self.params.get("strategy", "median")`），反射不出来 ⇒ 与其猜，不如**明列**，
#: 并由 `tests/test_regressions.py` 判据核对"列出来的键真的被该步骤读到"。
KNOWN_STEP_PARAMS: dict[str, tuple[str, ...]] = {
    "missing": ("strategy", "drop_threshold"),
    "encode": ("mode",),
    "scale": ("mode",),
    "feature_select": ("top_k", "score", "seed"),
    "split": ("test_size",),
}

#: 每个内置步骤参数的**取值域**（`None` = 自由取值）。同样由判据核对。
STEP_PARAM_CHOICES: dict[str, dict[str, tuple | None]] = {
    "missing": {"strategy": ("median", "mean"), "drop_threshold": None},
    "encode": {"mode": ("onehot", "ordinal")},
    "scale": {"mode": ("standard", "robust")},
    "feature_select": {"top_k": None, "score": ("mi", "f"), "seed": None},
    "split": {"test_size": None},
}

#: 声明里允许出现的顶层键（未知键硬失败）。
_STEP_KEYS = frozenset({"key", "params", "enabled", "code"})
_DECL_KEYS = frozenset({"steps", "split"})
_SPLIT_KEYS = frozenset({"test_size", "stratify", "seed", "time_split",
                         "group_by", "holdout"})

#: 自定义步骤代码里必须定义的类名（与 `Network` 同一约定）。
REQUIRED_STEP_CLASS = "Step"


class StepDeclError(ValueError):
    """声明不合法。`kind` 机读、`detail` 给人/AI 看（可教学）。"""

    def __init__(self, message: str, *, kind: str = "bad_step_declaration",
                 detail: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.detail = detail

    def teaching_text(self) -> str:
        out = f"[{self.kind}] {self}"
        if self.detail:
            out += f"\n{self.detail}"
        return out


def builtin_step_keys() -> list[str]:
    """内置步骤的 key（稳定排序）——工具 schema 与文档共用这一处。"""
    return sorted(BUILTIN_STEPS)


def is_builtin_step(key: str) -> bool:
    return str(key) in BUILTIN_STEPS


def check_step_class(cls: Any) -> None:
    """自定义步骤类必须**真的是 Step 的子类**且有 `apply`。

    ⚠ 为什么要求继承 `Step` 而不是"鸭子类型"：`Pipeline._fit_split_aware` 会
    把测试集喂给 `st.transform(...)`（默认 `Step.transform` 抛
    `NotImplementedError`），而 `describe()`/`params`/`state_` 也都在基类上。
    ⇒ 不继承就会出现"能 fit、到 transform 才炸"的**延迟失败**，
    那种失败比当场拒绝难查得多。
    """
    if not isinstance(cls, type):
        raise StepDeclError(
            f"`{REQUIRED_STEP_CLASS}` 不是类（是 {type(cls).__name__}）",
            kind="step_not_a_class")
    if not issubclass(cls, Step):
        raise StepDeclError(
            f"`{REQUIRED_STEP_CLASS}` 必须继承 `ml_toolbox.core.pipeline.Step`",
            kind="step_not_subclass",
            detail="基类提供 apply/transform/describe/params/state_ 五项契约；"
                   "不继承会在 transform（测试集那一步）才炸，属延迟失败")
    if not callable(getattr(cls, "apply", None)):
        raise StepDeclError(f"`{REQUIRED_STEP_CLASS}` 缺 `apply(self, df, target, diag=False)`",
                            kind="step_missing_apply")
    if cls.apply is getattr(Step, "apply", None):
        raise StepDeclError(
            f"`{REQUIRED_STEP_CLASS}` 没有实现自己的 `apply`（还是基类那个抽象方法）",
            kind="step_missing_apply")


def make_compiled_step(code: str, *, index: int,
                       compile_fn: Callable[[str], Any]) -> type[Step]:
    """把一段代码编译成步骤类（**编译动作由调用方注入**）。

    ⚠ 为什么 `compile_fn` 是注入的：沙箱住在边界层（`ml_mecha/sandbox.py`），
    而 core **不许**反向依赖它（依赖方向单向：ui/methods → core）。
    ⇒ core 提供"声明 → 管道"的机械层，编译这件"接触执行"的事由调用方传进来。

    ⚠ **没写 `transform` 就给一个兜底**（实测缺它必炸）：
    `Pipeline._fit_split_aware` 会把**测试集**也喂给每一步的 `transform`
    （`st.transform(test, None)`），而基类默认抛 `NotImplementedError`
    ⇒ 作者只写了 `apply` 的话，管道在**切分那一步**才炸，报错还叫
    "不支持对新数据 transform"，离真正原因很远。
    兜底 = 对测试集再调一次 `apply`，**仅当作者没实现 transform 时**。
    ⚠ 这个兜底**只对"逐行、无拟合统计量"的纯变换成立**（如造比值/取对数）；
    要按训练集统计量变换（均值/分位数）的作者**必须自己写 transform**，
    否则新数据会被按它自己的统计量变换——那是**静默的错**。
    """
    compiled = compile_fn(str(code))
    cls = getattr(compiled, "instance", None) or getattr(compiled, "cls", None)
    if cls is None:
        cls = getattr(compiled, REQUIRED_STEP_CLASS, None)
    check_step_class(cls)
    if not getattr(cls, "key", ""):
        cls.key = f"custom_{int(index)}"          # type: ignore[attr-defined]
    if cls.transform is getattr(Step, "transform", None):
        # 这一支**必须自己声明**：它是"作者没写 transform"的显式落点，
        # 而不是悄悄继承基类那个会抛的实现。
        cls.transform = _passthrough_transform     # type: ignore[assignment]
        cls.params = dict(getattr(cls, "params", {}) or {})
    return cls


def _passthrough_transform(self, df: pd.DataFrame,
                           target: str | None) -> pd.DataFrame:
    """兜底的 `transform`：对测试集再走一遍作者的 `apply`。

    ⚠ 只对**纯变换**（逐行、不依赖训练集统计量）正确。作者若按训练集统计量
    变换，必须自己实现 `transform`（见 `make_compiled_step` 的说明）。
    """
    out, _summary = self.apply(df, target, diag=False)
    return out


def build_steps(declaration: Any, *,
                compile_fn: Callable[[str], Any] | None = None) -> list[Step]:
    """按声明造步骤链（**顺序 = 声明顺序**）。"""
    if declaration is None:
        return list(Pipeline.default().steps)
    if not isinstance(declaration, (list, tuple)):
        raise StepDeclError(
            f"steps 需要数组，收到 {type(declaration).__name__}",
            kind="bad_steps_type",
            detail='示例：[{"key": "missing"}, {"key": "scale", "params": {"mode": "robust"}}]')
    out: list[Step] = []
    for i, raw in enumerate(declaration):
        if not isinstance(raw, dict):
            raise StepDeclError(f"第 {i + 1} 个步骤需要对象，收到 {type(raw).__name__}",
                                kind="bad_step_type")
        unknown = sorted(set(raw) - _STEP_KEYS)
        if unknown:
            raise StepDeclError(
                f"第 {i + 1} 个步骤有未知键 {unknown}",
                kind="unknown_step_key",
                detail=f"允许的键：{sorted(_STEP_KEYS)}（未知键不静默忽略）")
        key = str(raw.get("key") or "").strip()
        code = raw.get("code")
        if code and not key:
            key = f"custom_{i}"
        if not key:
            raise StepDeclError(f"第 {i + 1} 个步骤缺 `key`", kind="missing_step_key")
        params = raw.get("params") or {}
        if not isinstance(params, dict):
            raise StepDeclError(f"第 {i + 1} 步的 params 需要对象",
                                kind="bad_step_params")
        enabled = bool(raw.get("enabled", True))

        if code:
            if compile_fn is None:
                raise StepDeclError(
                    f"第 {i + 1} 步带了 code，但没有可用的编译器",
                    kind="compile_unavailable",
                    detail="自定义步骤代码需要沙箱编译；裸 core 用法请直接继承 Step 并传实例")
            cls = make_compiled_step(str(code), index=i, compile_fn=compile_fn)
            inst = cls(enabled=enabled, **params)
            if getattr(inst, "key", "") in (None, ""):
                inst.key = key
            out.append(inst)
            continue

        if key not in BUILTIN_STEPS:
            raise StepDeclError(
                f"第 {i + 1} 步的 key {key!r} 不是内置步骤，也没有 code",
                kind="unknown_step_key_name",
                detail=f"内置步骤：{builtin_step_keys()}；"
                       f"自定义步骤要写 `code`（定义 `class Step`）")
        _check_params(key, params, index=i)
        out.append(BUILTIN_STEPS[key](enabled=enabled, **params))
    return out


def _check_params(key: str, params: dict, *, index: int) -> None:
    """内置步骤的参数名必须被该步骤真的读（防"设了不生效"）。

    ⚠ 这条守卫的必要性：`Step.apply` 用 `self.params.get("mode", "onehot")` 读参数
    ⇒ 键名打错**不会有任何报错**，只是那一步按默认值跑。作者以为设上了、
    实际没设 —— 正是本仓反复在防的那类假绿。
    """
    known = KNOWN_STEP_PARAMS.get(key, ())
    unknown = sorted(set(params) - set(known))
    if unknown:
        raise StepDeclError(
            f"第 {index + 1} 步 {key!r} 不认识参数 {unknown}",
            kind="unknown_step_param",
            detail=f"{key!r} 可调：{list(known)}")
    choices = STEP_PARAM_CHOICES.get(key, {})
    for pname, value in params.items():
        allowed = choices.get(pname)
        if allowed and value not in allowed:
            raise StepDeclError(
                f"第 {index + 1} 步 {key!r} 的 {pname}={value!r} 不在候选里",
                kind="bad_step_param_value",
                detail=f"候选：{list(allowed)}")


def build_pipeline(declaration: Any, *,
                   compile_fn: Callable[[str], Any] | None = None) -> Pipeline:
    """按声明造一条完整 `Pipeline`（步骤链 + 划分口径）。

    @param declaration ``{"steps": [...], "split": {...}}``；
        省略整份 ⇒ 返回 `Pipeline.default()`（**零行为变化**）。
    """
    if declaration is None or declaration == {}:
        return Pipeline.default()
    if not isinstance(declaration, dict):
        raise StepDeclError(
            f"pipeline 声明需要对象，收到 {type(declaration).__name__}",
            kind="bad_pipeline_declaration")
    unknown = sorted(set(declaration) - _DECL_KEYS)
    if unknown:
        raise StepDeclError(
            f"pipeline 声明有未知键 {unknown}",
            kind="unknown_pipeline_key",
            detail=f"允许的键：{sorted(_DECL_KEYS)}")
    steps = build_steps(declaration.get("steps"), compile_fn=compile_fn)
    split = declaration.get("split")
    if split is None:
        base = Pipeline.default()
        return Pipeline(steps=steps, seed=base.seed, test_size=base.test_size,
                        stratify=base.stratify, time_split=base.time_split)
    if not isinstance(split, dict):
        raise StepDeclError("split 需要对象", kind="bad_split_declaration")
    unknown = sorted(set(split) - _SPLIT_KEYS)
    if unknown:
        raise StepDeclError(
            f"split 有未知键 {unknown}", kind="unknown_split_key",
            detail=f"允许的键：{sorted(_SPLIT_KEYS)}")
    base = Pipeline.default()
    test_size = float(split.get("test_size", base.test_size))
    if not 0.0 < test_size < 1.0:
        raise StepDeclError(
            f"split.test_size={test_size} 必须在 (0, 1) 之间",
            kind="bad_test_size",
            detail="它是**测试集占比**（0.2 = 8:2 切分）")
    seed = int(split.get("seed", base.seed))
    if seed < 0:
        raise StepDeclError("split.seed 需要非负整数", kind="bad_split_seed")
    group_by = str(split.get("group_by") or "")
    holdout = tuple(split.get("holdout") or ())
    if bool(group_by) != bool(holdout):
        raise StepDeclError(
            "split.group_by 与 split.holdout 必须**成对**给",
            kind="bad_group_split",
            detail='例如 {"group_by": "N", "holdout": [14, 16, 18]}'
                   "（把 N=14/16/18 整组留出，用来量**尺寸外推**）")
    return Pipeline(steps=steps, seed=seed, test_size=test_size,
                    stratify=bool(split.get("stratify", base.stratify)),
                    time_split=bool(split.get("time_split", base.time_split)),
                    group_by=group_by, holdout=holdout)


def describe_declaration(declaration: Any) -> list[dict]:
    """声明 → 人类可读摘要（**不含代码**；用于回执与日志）。

    为什么不含代码：回执要紧凑，代码可能很长；要取代码看 `record.json`。
    """
    if not isinstance(declaration, dict):
        return []
    out = []
    for i, raw in enumerate(declaration.get("steps") or []):
        if not isinstance(raw, dict):
            continue
        entry = {"index": i, "key": str(raw.get("key") or f"custom_{i}"),
                 "enabled": bool(raw.get("enabled", True)),
                 "params": dict(raw.get("params") or {})}
        if raw.get("code"):
            entry["custom"] = True
            entry["code_lines"] = len(str(raw["code"]).splitlines())
        out.append(entry)
    return out


#: `apply` 的签名（判据核对自定义步骤实现了它，而不是只继承）。
APPLY_SIGNATURE = "(self, df, target, diag=False)"


def apply_signature_ok(cls: type) -> bool:
    """自定义步骤的 `apply` 签名是否可用（位置参数至少两个）。"""
    try:
        params = list(inspect.signature(cls.apply).parameters.values())
    except (TypeError, ValueError):
        return False
    # self + df + target = 至少 3 个位置参数
    return len([p for p in params
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) >= 3


__all__ = [
    "KNOWN_STEP_PARAMS", "STEP_PARAM_CHOICES", "REQUIRED_STEP_CLASS",
    "StepDeclError", "builtin_step_keys", "is_builtin_step", "check_step_class",
    "make_compiled_step", "build_steps", "build_pipeline",
    "describe_declaration", "apply_signature_ok", "APPLY_SIGNATURE",
]
