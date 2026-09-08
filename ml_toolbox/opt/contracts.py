# -*- coding: utf-8 -*-
"""优化框架核心契约（docs/优化定位.md §5）。

与 ML 侧的关系（兄弟框架 + 共享内核）：
- 共享：core.ParamSpec（参数空间词汇表）、core.registry 的注册模式、
  persistence 的"运行留痕"思想；
- 独立：批式契约（MLMethod.fit）与增量契约（Optimizer.ask/tell）互不污染。

问题类 = 序贯黑盒优化（SMBO）：每评估一次得一条信息、调一次参数。
循环 = ask-and-tell；核心约束 = 评估昂贵（样本效率为王）。

本模块零 Qt（P8）、零 scipy/sklearn 顶层依赖（引擎各自惰性 import）。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Optional

import numpy as np
import pandas as pd

from ..core.contracts import ParamSpec


# ================================================================ 参数空间
class ParamSpace:
    """统一参数空间：升级自 ParamSpec 列表（log 尺度 / 向量化 / 投影）。

    维度定义：连续/整数参数各占 1 维（归一化到 [0,1] 供优化器内部使用）；
    类别参数占 1 维（索引取整）；bool 占 1 维；text 不参与寻优
    （按固定值透传，如 torch_mlp 的 hidden="64,32"）。
    """

    def __init__(self, params: list[ParamSpec]):
        self.params = list(params)
        # 参与寻优的参数：text 与"有界缺失的 number/int"固定透传
        # （优化器只能在有界域工作；半无界旋钮如 leaf_size 保持默认即可）
        self._opt: list[ParamSpec] = []
        self._fixed: dict[str, object] = {}
        for p in self.params:
            unbounded = (p.kind in ("number", "int")
                         and (p.min is None or p.max is None))
            if p.kind == "text" or unbounded:
                self._fixed[p.key] = p.default
            else:
                self._opt.append(p)

    # ------------------------------------------------ 元信息
    @property
    def keys(self) -> list[str]:
        return [p.key for p in self._opt]

    @property
    def dim(self) -> int:
        return len(self._opt)

    def spec(self, key: str) -> Optional[ParamSpec]:
        for p in self._opt:
            if p.key == key:
                return p
        return None

    def describe(self) -> list[dict]:
        """给 UI / 日志的维度说明表。"""
        out = []
        for p in self._opt:
            kind = ("log-uniform" if p.kind == "number" and p.log else
                    "uniform" if p.kind == "number" else
                    "randint" if p.kind == "int" else
                    "categorical" if p.kind == "select" else "bool")
            out.append({"key": p.key, "kind": kind,
                        "low": p.min, "high": p.max,
                        "choices": p.choices, "label": p.label or p.key})
        return out

    # ------------------------------------------------ 向量化（优化器内部坐标）
    def to_vector(self, params: dict) -> np.ndarray:
        """参数字典 -> [0,1]^dim 向量（类别取索引/连续做线性或 log 归一）。"""
        v = np.zeros(self.dim)
        for i, p in enumerate(self._opt):
            x = params.get(p.key, p.default)
            if p.kind == "select":
                ch = [str(c) for c in (p.choices or [])]
                try:
                    v[i] = ch.index(str(x)) / max(len(ch) - 1, 1)
                except ValueError:
                    v[i] = 0.0
            elif p.kind == "bool":
                v[i] = 1.0 if _as_bool(x) else 0.0
            else:
                lo, hi = float(p.min), float(p.max)
                x = float(x)
                if p.log:
                    x = (np.log(max(x, lo, 1e-300)) - np.log(max(lo, 1e-300))) \
                        / max(np.log(max(hi, 1e-300)) - np.log(max(lo, 1e-300)), 1e-300)
                else:
                    x = (x - lo) / max(hi - lo, 1e-300)
                v[i] = float(np.clip(x, 0.0, 1.0))
        return v

    def from_vector(self, v) -> dict:
        """[0,1]^dim 向量 -> 参数字典（含固定透传项），自动 clip 回界内。"""
        v = np.asarray(v, float).reshape(-1)
        out = dict(self._fixed)
        for i, p in enumerate(self._opt):
            t = float(np.clip(v[i] if i < len(v) else 0.0, 0.0, 1.0))
            if p.kind == "select":
                ch = [str(c) for c in (p.choices or [""])]
                out[p.key] = ch[int(round(t * (len(ch) - 1)))]
            elif p.kind == "bool":
                out[p.key] = t >= 0.5
            elif p.kind == "int":
                lo, hi = int(p.min), int(p.max)
                out[p.key] = int(round(lo + t * (hi - lo)))
            else:
                lo, hi = float(p.min), float(p.max)
                out[p.key] = float(np.clip(
                    np.exp(np.log(max(lo, 1e-300))
                           + t * (np.log(max(hi, 1e-300))
                                  - np.log(max(lo, 1e-300))))
                    if p.log else lo + t * (hi - lo), lo, hi))
        return out

    # ------------------------------------------------ 采样
    def sample(self, rng: np.random.RandomState) -> dict:
        """先验均匀采样（log 参数在对数域均匀）。"""
        v = rng.rand(self.dim)
        return self.from_vector(v)

    def grid_points(self, n_per_dim: int = 5) -> list[dict]:
        """逐维网格（基线用；维度多时组合爆炸，runner 会截断并警告）。"""
        axes = []
        for p in self._opt:
            if p.kind == "select":
                axes.append([str(c) for c in (p.choices or [])])
            elif p.kind == "bool":
                axes.append([True, False])
            elif p.kind == "int":
                lo, hi = int(p.min), int(p.max)
                k = min(n_per_dim, hi - lo + 1)
                axes.append([int(x) for x in
                             np.unique(np.linspace(lo, hi, k).round())])
            else:
                lo, hi = float(p.min), float(p.max)
                if p.log:
                    axes.append(list(np.exp(np.linspace(np.log(max(lo, 1e-300)),
                                                        np.log(max(hi, 1e-300)),
                                                        n_per_dim))))
                else:
                    axes.append(list(np.linspace(lo, hi, n_per_dim)))
        import itertools
        return [dict(zip(self.keys, combo)) | dict(self._fixed)
                for combo in itertools.product(*axes)] if axes else [dict(self._fixed)]


def _as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("1", "true", "yes", "on")


# ================================================================ 目标函数
class Objective(abc.ABC):
    """黑盒目标：一次评估 = 一次运行。统一"最小化"语义。

    实现约定（P10）：评估失败不要抛异常逃到框架——返回 float('inf')
    或 raise 均可，runner 会捕获并记 status='failed'。
    """
    name: str = "objective"
    space: ParamSpace
    multi: bool = False          # True = 返回向量（多目标，阶段3）
    n_obj: int = 1               # 目标数（multi=True 时 >1）
    noise: bool = False          # 含噪（数模仿真标定）→ 提示需要重复评估
    minimize: bool = True        # maximize 的目标取负实现，runner 统一化

    def __init__(self, name="objective", space: ParamSpace = None,
                 noise: bool = False, minimize: bool = True,
                 multi: bool = False, n_obj: int = 1):
        self.name = name
        self.space = space
        self.noise = noise
        self.minimize = minimize
        self.multi = multi
        self.n_obj = n_obj if multi else 1

    @abc.abstractmethod
    def evaluate(self, params: dict) -> float:
        """原始方向的一次评估。multi=True 时返回可迭代（各目标原始方向）。"""
        ...

    def __call__(self, params: dict):
        v = self.evaluate(params)
        if self.multi:
            a = np.asarray(v, float)
            return a if self.minimize else -a
        return float(v) if self.minimize else -float(v)


class CallableObjective(Objective):
    """把 lambda/函数包装成 Objective（数模标定、合成函数、AutoTuner 都用它）。"""

    def __init__(self, fn: Callable[[dict], float], space: ParamSpace,
                 name="callable", noise=False, minimize=True,
                 multi=False, n_obj=1):
        super().__init__(name=name, space=space, noise=noise, minimize=minimize,
                         multi=multi, n_obj=n_obj)
        self._fn = fn

    def evaluate(self, params):
        return self._fn(params)


# ================================================================ 优化器契约
class Optimizer(abc.ABC):
    """增量契约：ask 出候选，tell 收结果。内部信念（GP/分布/种群）自持。

    单点序贯（GP-BO/TPE）ask 返回一个 dict；种群式（CMA-ES/GA）返回 list[dict]
    （一代），tell 相应接收列表——runner 按 ask 的返回类型分发。
    """

    name: ClassVar[str] = ""
    display_name: ClassVar[str] = ""
    family: ClassVar[str] = ""           # baseline | bo | evo | schedule | local
    tags: ClassVar[tuple] = ()
    batch: ClassVar[bool] = False        # True = 一代多点（并行评估友好）
    multi_objective: ClassVar[bool] = False
    param_schema: ClassVar[list] = []    # 优化器自身超参——复用 ParamSpec！

    def __init__(self):
        self.space: ParamSpace | None = None
        self.cfg: dict = {}
        self._rng = np.random.RandomState(0)

    def setup(self, space: ParamSpace, seed: int, cfg: dict,
              budget: "Budget"):
        self.space = space
        self.cfg = dict(cfg)
        self._rng = np.random.RandomState(seed)
        self.budget = budget

    # ------------------------------------------------ 子类必须实现
    @abc.abstractmethod
    def ask(self):
        """-> dict 或 list[dict]（batch=True 时）。"""
        ...

    @abc.abstractmethod
    def tell(self, params, score, status="ok"):
        """params 与 ask 返回形态一致；score 已统一为最小化方向。"""
        ...

    # ------------------------------------------------ 可选覆写
    def inspect_pages(self, record: "OptRecord"):
        """声明式检视页（复用 core.PageSpec 契约，UI 层同一装配器）。"""
        return []


# ================================================================ 预算与记录
@dataclass
class Budget:
    """交付约束（G6）：次数 / 时间 / 停滞三者先到先停。"""
    n_evals: int = 50
    time_limit: float = float("inf")     # 秒
    stall: int = 0                       # 连续多少次评估无改善即停（0=不启用）

    def to_dict(self):
        return {"n_evals": self.n_evals,
                "time_limit": (None if np.isinf(self.time_limit)
                               else self.time_limit),
                "stall": self.stall}


@dataclass
class OptRecord:
    """一次优化运行 = 完整评估历史。历史才是数据资产（接缝3：可回流 ML）。"""
    run_id: str
    optimizer: str
    objective: str
    space_desc: list = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    seed: int = 42
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    best: Optional[dict] = None          # {params..., score}
    multi: bool = False                  # 多目标轨迹（history 含 f0..fk 列）
    pareto: Optional[pd.DataFrame] = None    # 非支配解集（multi 时）
    elapsed: float = 0.0
    error: Optional[str] = None

    @property
    def n_evals(self) -> int:
        return int((self.history.get("status", pd.Series(dtype=str))
                    == "ok").sum()) if "status" in self.history else 0

    def to_dataset(self):
        """评估历史 -> ML 可消费 Dataset（参数为 X，分数为 y；接缝3）。"""
        from ..core.dataset import Dataset
        h = self.history[self.history["status"] == "ok"]
        if h.empty:
            return None
        pcols = self.space_desc and [d["key"] for d in self.space_desc] or []
        X = h[pcols] if pcols else h.drop(columns=["score", "status", "ts",
                                                   "best_so_far"])
        return Dataset(X.assign(target=h["score"].to_numpy()),
                       name=f"opt_{self.objective}", target="target",
                       source=f"opt_record:{self.run_id}")

    def curves(self) -> pd.DataFrame:
        """收敛曲线数据（best-so-far 已在 tell 时维护，直接取）。"""
        return self.history.copy()


# ================================================================ 便捷入口
def make_objective(fn, space: ParamSpace, name="callable", noise=False,
                   minimize=True) -> Objective:
    return CallableObjective(fn, space, name=name, noise=noise,
                             minimize=minimize)
