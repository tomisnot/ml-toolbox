# -*- coding: utf-8 -*-
"""核心契约：所有方法族共享的基类 / 协议 / 数据结构。

设计原则（见 docs/项目定位.md）：
- P1 核心逻辑与界面分离：本模块零 Qt 依赖；
- P2 契约优先，注册扩展：方法只通过本模块定义的结构与框架交互；
- P3 零侵入默认：diag=False 时数值路径不变、不产出诊断。

== 方法契约一览 ==

MLMethod 子类必须声明 ClassVar：
  name / family / task / tags / param_schema
并实现：
  fit(X, y, cfg, diag)            -> MLResult
可选覆写：
  predict(X, result)              -> np.ndarray
  evaluate(result, X, y)          -> dict   （默认按 task 自动计算）
  cross_validate(X, y, cfg, cv)   -> dict   （默认按 task 自动交叉验证）

MLResult.artifacts 约定键（UI 按此消费，缺失即显示语义化占位）：
  supervised : y_true, y_pred, y_prob, residual, feature_importance
  cluster    : labels, score, silhouette_per_sample
  manifold   : embedding, explained_variance_ratio
  anomaly    : scores, labels, contamination
  timeseries : forecast, actual, residual
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 任务常量
TASK_SUPERVISED = "supervised"      # 回归/分类统一：y 为 Series，回归 float / 分类 label
TASK_CLUSTER = "cluster"
TASK_MANIFOLD = "manifold"          # 降维
TASK_ANOMALY = "anomaly"
TASK_TIMESERIES = "timeseries"

ALL_TASKS = (TASK_SUPERVISED, TASK_CLUSTER, TASK_MANIFOLD,
             TASK_ANOMALY, TASK_TIMESERIES)

# ---------------------------------------------------------------- 建模目的
# task 是"学习机制"轴，purpose 是"建模目的"轴（数模视角：预测/判别/归因/…）。
# 一个方法可服务多个目的；UI 按此筛选。
PURPOSE_PREDICT = "predict"            # 预测/预报（回归、时序外推）
PURPOSE_DISCRIMINATE = "discriminate"  # 判别/分类
PURPOSE_EXPLAIN = "explain"            # 归因解释（系数、特征重要性、稀疏选择）
PURPOSE_CLUSTER = "cluster"            # 分群/类型分析
PURPOSE_REDUCE = "reduce"              # 降维/特征提取
PURPOSE_EVALUATE = "evaluate"          # 综合评价（方差权重、主成分打分）
PURPOSE_ANOMALY = "anomaly"            # 异常检测
PURPOSE_SURROGATE = "surrogate"        # 代理模型/响应面（供优化算法调用）
PURPOSE_BASELINE = "baseline"          # 参照基线（任何模型都应赢它）

PURPOSE_LABELS = {
    PURPOSE_PREDICT: "预测", PURPOSE_DISCRIMINATE: "判别",
    PURPOSE_EXPLAIN: "归因解释", PURPOSE_CLUSTER: "分群",
    PURPOSE_REDUCE: "降维", PURPOSE_EVALUATE: "综合评价",
    PURPOSE_ANOMALY: "异常检测", PURPOSE_SURROGATE: "代理模型",
    PURPOSE_BASELINE: "参照基线",
}


def derive_purposes(task: str, target_kind: Optional[str],
                    tags=(), family: str = "") -> tuple:
    """按"学习机制 + 已沉淀 tags"推导建模目的（新方法零成本继承）。

    方法可用 ClassVar purposes 显式覆盖/补充。
    注意："参照基线"目的只认 family=='baseline'——tags 里的 'baseline'
    是"该族默认起点"的意思，与数模的"必赢基线"是两回事（曾撞名误标）。
    """
    out: list = []
    if task == TASK_SUPERVISED:
        out.append(PURPOSE_PREDICT)
        if target_kind in (None, "classification"):
            out.append(PURPOSE_DISCRIMINATE)
    elif task == TASK_CLUSTER:
        out.append(PURPOSE_CLUSTER)
    elif task == TASK_MANIFOLD:
        out.append(PURPOSE_REDUCE)
    elif task == TASK_ANOMALY:
        out.append(PURPOSE_ANOMALY)
    elif task == TASK_TIMESERIES:
        out.append(PURPOSE_PREDICT)
    t = set(tags)
    if t & {"interpretable", "sparse", "importance", "feature-selection",
            "regularized", "statistical", "bayesian"}:
        out.append(PURPOSE_EXPLAIN)
    if t & {"surrogate", "uncertainty"}:
        out.append(PURPOSE_SURROGATE)
    if family == "baseline":
        out.append(PURPOSE_BASELINE)
    if task == TASK_MANIFOLD and t & {"linear", "supervised"}:
        out.append(PURPOSE_EVALUATE)     # PCA/LDA 的方差权重可做综合评价
    seen, ordered = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return tuple(ordered)


# ---------------------------------------------------------------- 指标方向
# "越小越好"的指标集——单一事实来源（M1：曾三处各写一份且已分叉）。
# 新增越低越好的指标只改这里，core.runner / ui.gallery / opt.bridges 全部跟随。
LOWER_IS_BETTER = frozenset({"rmse", "mae", "mape", "silhouette_deficit"})


def is_lower_better(metric: str) -> bool:
    """该指标是否越小越好（决定对比表/画廊的排序方向、调参的统一最小化）。"""
    return metric in LOWER_IS_BETTER


# ---------------------------------------------------------------- 任务判定
# 数值且唯一值 > 该阈值 -> 回归，否则分类（M10：曾三处独立实现 + UI 抄文案）。
REGRESSION_CARDINALITY = 20


def infer_kind(y) -> Optional[str]:
    """从目标列推断回归/分类（唯一的任务判定入口）。

    None -> None；非数值 -> classification；数值且基数 > REGRESSION_CARDINALITY
    -> regression。pipeline / dataset / methods 一律调用此函数。
    """
    if y is None:
        return None
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > REGRESSION_CARDINALITY:
        return "regression"
    return "classification"


# ---------------------------------------------------------------- 数据视图
@dataclass
class DataSpec:
    """数据集的一个视图：管道输出 + 划分方案。"""
    X: pd.DataFrame
    y: Optional[pd.Series] = None
    train_idx: Optional[np.ndarray] = None
    test_idx: Optional[np.ndarray] = None
    target_kind: Optional[str] = None    # 'regression' | 'classification' | None
    n_classes: int = 0
    meta: dict = field(default_factory=dict)

    def split(self):
        """返回 (Xtr, Xte, ytr, yte)；无划分时 train==test 全量。"""
        tr = self.train_idx if self.train_idx is not None else np.arange(len(self.X))
        te = self.test_idx if self.test_idx is not None else np.arange(len(self.X))
        ytr = self.y.iloc[tr] if self.y is not None else None
        yte = self.y.iloc[te] if self.y is not None else None
        return self.X.iloc[tr], self.X.iloc[te], ytr, yte


# ---------------------------------------------------------------- 参数 schema
@dataclass
class ParamSpec:
    """旋钮声明：UI 据此生成控件，并校验/清洗运行级覆写。

    共享内核：ML 侧用它做参数面板，优化侧（ml_toolbox.opt）用它构成
    ParamSpace（采样/投影/向量化）。log=True 表示对数尺度（lr/正则强度类）。
    """
    key: str
    label: str = ""
    kind: str = "number"        # number | int | select | bool | text
    default: object = None
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[list] = None
    hint: str = ""
    log: bool = False           # 对数尺度（需 min>0；优化侧采样用，ML 侧忽略）

    def clean(self, raw):
        """把 UI 原始值转成类型正确的值；非法输入抛 ValueError。"""
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            return self.default
        if self.kind == "bool":
            if isinstance(raw, bool):
                return raw
            return str(raw).strip().lower() in ("1", "true", "yes", "on")
        if self.kind in ("number", "int"):
            try:
                v = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"参数 {self.key} 需要数字，收到 {raw!r}")
            if self.kind == "int":
                v = int(round(v))
            if self.min is not None and v < self.min:
                raise ValueError(f"参数 {self.key}={v} 小于下限 {self.min}")
            if self.max is not None and v > self.max:
                raise ValueError(f"参数 {self.key}={v} 大于上限 {self.max}")
            return v
        if self.kind == "select":
            v = str(raw)
            if self.choices and v not in [str(c) for c in self.choices]:
                raise ValueError(f"参数 {self.key}={v!r} 不在候选 {self.choices}")
            return v
        return str(raw)


# ---------------------------------------------------------------- 运行配置
@dataclass
class RunConfig:
    """一次运行的配置（运行级，不写回全局 —— pitfalls #17）。"""
    overrides: dict = field(default_factory=dict)   # {param_key: raw_value}
    diag: bool = False
    seed: int = 42
    extras: dict = field(default_factory=dict)      # 任务级扩展（horizon/cv_folds/...）

    def resolve(self, schema: list[ParamSpec]) -> dict:
        """按 schema 清洗覆写；未覆写的取默认。返回干净参数字典。"""
        out = {p.key: p.default for p in schema}
        for p in schema:
            if p.key in self.overrides:
                out[p.key] = p.clean(self.overrides[p.key])
        return out


# ---------------------------------------------------------------- 运行结果
@dataclass
class MLResult:
    method_name: str
    task: str
    target_kind: Optional[str] = None
    metrics: dict = field(default_factory=dict)         # {指标名: float}
    primary_metric: str = ""                            # 对比视图排序依据
    artifacts: dict = field(default_factory=dict)       # 可视化契约数据
    params: dict = field(default_factory=dict)          # 实际生效参数
    diag: Optional[dict] = None                         # cfg.diag=True 时产出
    elapsed: float = 0.0
    error: Optional[str] = None                         # 失败时记录，不抛出

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------- 检视页声明
@dataclass
class PageSpec:
    """方法声明自己检视页：UI 基类只负责装配，不预设图长什么样（G2）。

    - kind="mpl" + plot(ax, result)：纯 matplotlib 绘制函数（方法层零 Qt）；
    - kind="pg"：交给 UI 的通用自适应组件（热图/散点缩放）；
    - kind="table"/"cards"：消费 result.artifacts 中的表格/摘要；
    - data：table 页的替代数据源 fn(record)->DataFrame（优化侧 record 无
      artifacts，C4 用它在声明时绑定计算；ML 侧留空走 artifacts 老路）；
    - plot=None 且无 artifacts：UI 显示 hint 语义化占位（模式 7）。
    """
    key: str
    title: str
    kind: str = "mpl"               # mpl | pg | table | cards
    plot: Optional[Callable] = None    # (fig_ax, result) -> None
    builder: Optional[Callable] = None  # (result, parent) -> QWidget（UI 层扩展用）
    hint: str = ""                  # 无数据时的语义化占位提示
    data: Optional[Callable] = None    # (record) -> DataFrame（table 页数据源）


# ---------------------------------------------------------------- 方法基类
class MLMethod(abc.ABC):
    """所有机器学习方法的统一基类（工具箱资源共享的根本契约）。"""

    name: ClassVar[str] = ""
    display_name: ClassVar[str] = ""
    family: ClassVar[str] = ""          # linear | ensemble | svm | knn | bayes | cluster | manifold | anomaly | timeseries
    task: ClassVar[str] = TASK_SUPERVISED
    purposes: ClassVar[tuple] = ()      # 建模目的（见 PURPOSE_*，可多值）
    target_kind: ClassVar[Optional[str]] = None   # supervised: 'regression'|'classification'|None(皆可)
    tags: ClassVar[tuple] = ()
    param_schema: ClassVar[list] = []
    supports_sample_weight: ClassVar[bool] = False
    requires_numeric: ClassVar[bool] = True       # False = 内部自行编码（如原始 RF）

    # ------------------------------------------------ 子类必须实现
    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y, cfg: RunConfig, diag: bool = False) -> MLResult:
        ...

    # ------------------------------------------------ 可选覆写
    def predict(self, X: pd.DataFrame, result: MLResult):
        raise NotImplementedError

    def fit_extra_artifacts(self, X: pd.DataFrame, result: MLResult) -> dict:
        """run_one 在 predict 前调用：概率等附加产物（默认无）。"""
        return {}

    def evaluate(self, result: MLResult, X: pd.DataFrame, y) -> dict:
        """默认评估：按 task/target_kind 自动计算标准指标。"""
        a = result.artifacts
        if result.task != TASK_SUPERVISED:
            return dict(result.metrics)
        if result.target_kind == "regression":
            yt, yp = np.asarray(a["y_true"], float), np.asarray(a["y_pred"], float)
            return regression_metrics(yt, yp)
        yt, yp = np.asarray(a["y_true"]), np.asarray(a["y_pred"])
        m = classification_metrics(yt, yp)
        if a.get("y_prob") is not None:
            m.update(roc_metrics(yt, a["y_prob"], list(np.unique(yt))))
        return m

    def cross_validate(self, X: pd.DataFrame, y, cfg: RunConfig,
                       cv_folds: int = 5) -> dict:
        """默认交叉验证：k 折重跑 fit，统计主指标 mean±std。"""
        return default_cv(self, X, y, cfg, cv_folds)

    # ------------------------------------------------ 检视页声明（默认空 = UI 兜底页）
    def inspect_pages(self, cfg: RunConfig) -> list[PageSpec]:
        return []

    @classmethod
    def summary_pages(cls) -> list[PageSpec]:
        """跨方法共享的检视页（UI 装配时自动追加，方法无需声明）。"""
        return [PageSpec("metrics", "指标", "table"),
                PageSpec("params", "参数", "table"),
                PageSpec("diag", "诊断", "text",
                         hint="开启「诊断」开关后重跑，此处显示中间产物")]

    # ------------------------------------------------ 工具
    def can_handle(self, spec: DataSpec) -> bool:
        if self.task == TASK_SUPERVISED:
            if spec.y is None:
                return False
            if self.target_kind and spec.target_kind != self.target_kind:
                return False
            return True
        if self.task == TASK_TIMESERIES:
            return spec.y is not None
        return True

    def params(self, cfg: RunConfig) -> dict:
        return cfg.resolve(self.param_schema)

    def purposes_of(self) -> tuple:
        """显式声明优先；未声明则按 task+tags+family 推导。"""
        return tuple(self.purposes) or derive_purposes(
            self.task, self.target_kind, self.tags, self.family)

    def purposes_cn(self) -> str:
        return "/".join(PURPOSE_LABELS[p] for p in self.purposes_of())

    def _new_result(self, **kw) -> MLResult:
        kw.setdefault("method_name", self.name)
        kw.setdefault("task", self.task)
        kw.setdefault("primary_metric", default_primary(self.task, kw.get("target_kind")))
        return MLResult(**kw)


# ---------------------------------------------------------------- 指标
def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": _mape(y_true, y_pred),
    }


def _mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - np.asarray(y_pred, float)[mask])
                                / y_true[mask])) * 100)


def classification_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    average = "binary" if len(np.unique(y_true)) <= 2 else "macro"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def roc_metrics(y_true, y_prob, classes) -> dict:
    from sklearn.metrics import roc_auc_score
    try:
        if len(classes) <= 2:
            p = np.asarray(y_prob)
            p = p[:, 1] if p.ndim == 2 else p
            return {"auc": float(roc_auc_score(y_true, p))}
        return {"auc": float(roc_auc_score(y_true, y_prob, multi_class="ovr",
                                           average="macro"))}
    except Exception:
        return {"auc": float("nan")}


def default_primary(task: str, target_kind: Optional[str]) -> str:
    if task == TASK_SUPERVISED:
        return "rmse" if target_kind == "regression" else "f1"
    if task == TASK_CLUSTER:
        return "silhouette"
    if task == TASK_MANIFOLD:
        return "explained_variance"
    if task == TASK_ANOMALY:
        return "auc"
    if task == TASK_TIMESERIES:
        return "rmse"
    return ""


# ---------------------------------------------------------------- 默认 CV
def default_cv(method: MLMethod, X, y, cfg: RunConfig, cv_folds: int = 5) -> dict:
    """k 折交叉验证（按 task 选折法），返回 {主指标: mean/std/scores}。"""
    from sklearn.model_selection import KFold, StratifiedKFold
    n = len(X)
    cv_folds = max(2, min(cv_folds, n))
    if method.task == TASK_SUPERVISED and method.target_kind != "regression" \
            and y is not None and pd.Series(y).nunique() > 1:
        try:
            splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                       random_state=cfg.seed)
            folds = list(splitter.split(X, y))
        except Exception:
            folds = list(KFold(cv_folds, shuffle=True,
                               random_state=cfg.seed).split(X))
    else:
        folds = list(KFold(cv_folds, shuffle=True,
                           random_state=cfg.seed).split(X))

    scores = []
    primary = ""
    for tr, te in folds:
        Xtr = (X.iloc[tr] if hasattr(X, "iloc") else X[tr])
        Xte = (X.iloc[te] if hasattr(X, "iloc") else X[te])
        ytr = (y.iloc[tr] if hasattr(y, "iloc") else y[tr])
        yte = (y.iloc[te] if hasattr(y, "iloc") else y[te])
        try:
            res = method.fit(Xtr, ytr, cfg, diag=False)
            if not res.ok:
                continue
            pred = method.predict(Xte, res)
            res.artifacts["y_true"] = np.asarray(yte)
            res.artifacts["y_pred"] = np.asarray(pred)
            m = method.evaluate(res, Xte, yte if hasattr(yte, "__len__") else yte)
            res.metrics.update(m)
            primary = primary or res.primary_metric
            if primary and primary in m:
                scores.append(float(m[primary]))
        except Exception:
            continue
    if not scores:
        return {"cv_" + (primary or "score"): float("nan")}
    arr = np.asarray(scores, float)
    return {f"cv_{primary}_mean": float(arr.mean()),
            f"cv_{primary}_std": float(arr.std()),
            "cv_scores": arr.tolist()}
