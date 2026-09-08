# -*- coding: utf-8 -*-
"""optuna 适配器（family=bo/schedule，引擎=适配）。

判据（docs/优化定位.md §6）：只消费评估轨迹、内部状态无需深度检视的算法
交给 optuna，藏在我们的 ask/tell 契约后面——换库不动框架。

- tpe：树状密度估计采样器（混合空间/类别参数强项，单目标）
- asha：Successive Halving 剪枝器（调 ML 超参：每次评估有中间分数可早停）

optuna 未安装 -> 本模块 import 失败 -> engines/__init__ 跳过，其余优化器不受影响
（torch 惰性依赖先例照搬）。

失败观测：runner 传 status='failed' -> study.tell(state=FAIL)（optuna 语义）。
"""
from __future__ import annotations

import numpy as np

from ..contracts import Optimizer, ParamSpec
from ..registry import register

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS = True
except Exception:                                    # pragma: no cover
    optuna = None
    _HAS = False


def _suggest(trial, p: ParamSpec):
    """把 ParamSpec 映射到 optuna suggest_*。"""
    if p.kind == "int":
        return trial.suggest_int(p.key, int(p.min), int(p.max))
    if p.kind == "number":
        return trial.suggest_float(p.key, float(p.min), float(p.max),
                                   log=bool(p.log))
    if p.kind == "bool":
        return trial.suggest_categorical(p.key, [True, False])
    if p.kind == "select":
        return trial.suggest_categorical(p.key, [str(c) for c in p.choices])
    return p.default


class _OptunaBase(Optimizer):
    """study.ask/tell 与我们的 ask/tell 的桥。"""

    def setup(self, space, seed, cfg, budget):
        super().setup(space, seed, cfg, budget)
        if not _HAS:
            raise RuntimeError("未安装 optuna（pip install optuna）")
        self._trial = None
        self._study = self._make_study(seed)

    def _make_study(self, seed):                          # 子类覆写
        raise NotImplementedError

    def ask(self):
        self._trial = self._study.ask()
        params = dict(self.space._fixed)
        for p in self.space._opt:
            params[p.key] = _suggest(self._trial, p)
        return params

    def tell(self, params, score, status="ok"):
        if self._trial is None:
            return
        if status == "ok" and np.isfinite(score):
            self._study.tell(self._trial, float(score))
        else:
            self._study.tell(self._trial, state=optuna.trial.TrialState.FAIL)
        self._trial = None


@register
class TPE(_OptunaBase):
    name = "tpe"
    display_name = "贝叶斯优化 TPE"
    family = "bo"
    tags = ("sample-efficient", "mixed-space", "categorical")
    param_schema = [
        ParamSpec("n_startup", "随机预热次数", "int", 10, min=1, max=100,
                  hint="TPE 在积累够样本前的纯随机评估数"),
    ]

    def _make_study(self, seed):
        from optuna.samplers import TPESampler
        return optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=seed,
                               n_startup_trials=int(self.cfg.get(
                                   "n_startup", 10))))


@register
class ASHA(_OptunaBase):
    name = "asha"
    display_name = "ASHA 早停调度"
    family = "schedule"
    batch = False
    tags = ("hyperparam", "early-stopping", "expensive")
    param_schema = [
        ParamSpec("reduction", "折减因子 η", "int", 3, min=2, max=8,
                  hint="每轮保留 1/η 的候选"),
        ParamSpec("min_resource", "最小资源", "int", 1, min=1),
    ]

    def _make_study(self, seed):
        from optuna.pruners import SuccessiveHalvingPruner
        # ASHA 需要 objective 中途 trial.report；本适配退化为
        # SuccessiveHalving（无中间报告时等价于带剪枝的 TPE 随机）。
        return optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=SuccessiveHalvingPruner(
                reduction_factor=int(self.cfg.get("reduction", 3)),
                min_resource=int(self.cfg.get("min_resource", 1))))
