# -*- coding: utf-8 -*-
"""Nelder-Mead 单纯形法（自研，零依赖，family=local）。

赌注：便宜光滑问题的局部精修兜底。经典 NM 每步可能要评估反射/扩张/
收缩点，这里用"待评队列"拆成 ask/tell：ask 从队列取一个顶点，
tell 按经典规则决定下一步往队列放什么（扩张/收缩/缩向）。
在 [0,1]^d 归一化坐标上运行；失败观测按 +inf 参与排序（自然被替换）。
"""
from __future__ import annotations

import numpy as np

from ..contracts import Optimizer, ParamSpec
from ..registry import register


@register
class NelderMead(Optimizer):
    name = "nelder_mead"
    display_name = "Nelder-Mead 单纯形"
    family = "local"
    tags = ("local", "derivative-free", "smooth")
    param_schema = [
        ParamSpec("init_size", "初始单纯形尺寸", "number", 0.25, min=0.01,
                  max=0.9, hint="[0,1] 归一化域"),
    ]

    def setup(self, space, seed, cfg, budget):
        super().setup(space, seed, cfg, budget)
        d = max(space.dim, 1)
        self.d = d
        s = float(self.cfg.get("init_size", 0.25))
        x0 = self._rng.rand(d) * 0.6 + 0.2
        self.V = np.clip(np.vstack(
            [x0] + [x0 + s * np.eye(d)[i] for i in range(d)]), 0.0, 1.0)
        self.F = np.full(d + 1, np.nan)
        # 队列元素：('vertex', i) 评估顶点 i；
        #          ('op', kind, x, ref_score) 评估临时操作点
        self._q = [('vertex', i) for i in range(d + 1)]
        self._pending = None

    # ------------------------------------------------ 主接口
    def ask(self):
        self._pending = self._q.pop(0)
        x = (self.V[self._pending[1]] if self._pending[0] == 'vertex'
             else self._pending[2])
        return self.space.from_vector(x)

    def tell(self, params, score, status="ok"):
        s = float(score) if status == "ok" and np.isfinite(score) else np.inf
        item = self._pending
        if item[0] == 'vertex':
            self.F[item[1]] = s
            if np.isnan(self.F).any():
                return                          # 初始化未完成
            self._reflect()
        else:
            _, kind, x, ref = item
            getattr(self, "_after_" + kind)(x, s, ref)

    # ------------------------------------------------ 经典 NM 规则
    def _sort(self):
        o = np.argsort(self.F)
        self.V, self.F = self.V[o], self.F[o]

    def _reflect(self):
        self._sort()
        xg = self.V[:-1].mean(0)
        xr = np.clip(2 * xg - self.V[-1], 0.0, 1.0)
        self._q.append(('op', 'reflect', xr, np.nan))

    def _after_reflect(self, x, s, ref):
        if s < self.F[0]:                       # 优于最好 -> 扩张
            xg = self.V[:-1].mean(0)
            xe = np.clip(2 * x - xg, 0.0, 1.0)  # xg + 2*(x-xg) = 2x - xg
            self._xr = x                        # 记住反射点（扩张失败时回退）
            self._q.append(('op', 'expand', xe, s))
        elif s < self.F[-2]:                    # 好于次差 -> 接受反射点
            self.V[-1], self.F[-1] = x, s
            self._reflect()
        else:                                   # 收缩（向重心）
            xg = self.V[:-1].mean(0)
            xc = np.clip(0.5 * xg + 0.5 * self.V[-1], 0.0, 1.0)
            self._q.append(('op', 'contract', xc, s))

    def _after_expand(self, x, s, ref):
        if s < ref:                             # 扩张优于反射 -> 用扩张点
            self.V[-1], self.F[-1] = x, s
        else:                                   # 否则接受反射点
            self.V[-1], self.F[-1] = self._xr, ref
        self._reflect()

    def _after_contract(self, x, s, ref):
        if s < self.F[-1]:                      # 收缩成功 -> 接受
            self.V[-1], self.F[-1] = x, s
            self._reflect()
        else:                                   # 缩向最优（全体收缩）
            b = self.V[0]
            self.V[1:] = np.clip(b + 0.5 * (self.V[1:] - b), 0.0, 1.0)
            self.F[1:] = np.nan
            self._q = [('vertex', i) for i in range(1, self.d + 1)]
