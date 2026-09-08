# -*- coding: utf-8 -*-
"""基线优化器：随机搜索 / 网格（family=baseline）。

存在意义 = 参照物（G4 遍历哲学）：GP-BO/CMA-ES 的收敛曲线必须显著
优于随机搜索，否则"智能"是假的。数模论文里这也叫"对照实验"。
"""
from __future__ import annotations

import numpy as np

from ..contracts import Optimizer, ParamSpec
from ..registry import register


@register
class RandomSearch(Optimizer):
    name = "random_search"
    display_name = "随机搜索"
    family = "baseline"
    tags = ("baseline", "reference")
    param_schema = []            # 无自身超参

    def ask(self):
        return self.space.sample(self._rng)

    def tell(self, params, score, status="ok"):
        pass                     # 无信念可更新——基线的诚实


@register
class GridSearch(Optimizer):
    name = "grid_search"
    display_name = "网格搜索"
    family = "baseline"
    tags = ("baseline", "reference")
    param_schema = [
        ParamSpec("n_per_dim", "每维格点", "int", 5, min=2, max=15,
                  hint="网格 = n^dim 组合，维度多时自动截断"),
    ]

    def setup(self, space, seed, cfg, budget):
        super().setup(space, seed, cfg, budget)
        n = int(self.cfg.get("n_per_dim", 5))
        pts = space.grid_points(n)
        # 组合爆炸保护：预算内均匀抽稀（保持覆盖性）
        if len(pts) > budget.n_evals:
            sel = np.unique(np.linspace(0, len(pts) - 1,
                                        budget.n_evals).astype(int))
            pts = [pts[i] for i in sel]
        self._pts = pts
        self._i = 0

    def ask(self):
        if self._i >= len(self._pts):
            return self.space.sample(self._rng)   # 网格耗尽退化为随机
        p = self._pts[self._i]
        self._i += 1
        return p

    def tell(self, params, score, status="ok"):
        pass
