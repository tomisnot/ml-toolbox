# -*- coding: utf-8 -*-
"""采集函数（acquisition）：给定 GP 后验 (mu, sigma)，算"该去哪看"的分数。

统一约定：**越大越值得评估**（runner 已把目标统一为最小化方向）。
EI 用闭式解（Jones 1998 数值稳定式），PI/UCB 同样闭式，零 scipy 依赖
（正态 cdf/pdf 用 math.erf 自实现，避免为两个函数拉 scipy.optimize）。
"""
from __future__ import annotations

import math

import numpy as np

_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x):
    x = np.asarray(x, float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / _SQRT2))


def _norm_pdf(x):
    x = np.asarray(x, float)
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def ei(mu, sigma, best, xi=0.01):
    """期望改进（Expected Improvement），最小化目标的数值稳定式。

    best = 当前最优分数；xi = 探索偏置（越大越激进）。
    """
    sigma = np.maximum(sigma, 1e-12)
    imp = best - xi - mu                      # >0 表示预测有改进
    z = imp / sigma
    return imp * _norm_cdf(z) + sigma * _norm_pdf(z)


def ucb(mu, sigma, best=None, beta=2.0):
    """置信界下界（最小化）：mu - beta*sigma，转成"越大越好"取负。

    best 参数为统一签名而设（UCB 不消费）。
    """
    return -(mu - beta * np.maximum(sigma, 0.0))


def pi(mu, sigma, best, xi=0.01):
    """改进概率（Probability of Improvement）。"""
    sigma = np.maximum(sigma, 1e-12)
    return _norm_cdf((best - xi - mu) / sigma)


ACQUISITIONS = {"ei": ei, "ucb": ucb, "pi": pi}
