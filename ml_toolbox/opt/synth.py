# -*- coding: utf-8 -*-
"""合成测试函数库（零 Qt，core 层）：UI 目标选择器 + 数模验收的靶子。

标准优化测试函数（最小化方向）。每个返回 (ParamSpace, fn, f_min, x_min)：
    f_min/x_min 供基准判"离真最优多近"（数模里=标定参数离已知真值多远）。
"""
from __future__ import annotations

import numpy as np

from ..core.contracts import ParamSpec
from .contracts import ParamSpace


def _box(dim, lo, hi):
    return ParamSpace([ParamSpec(f"x{i}", f"x{i}", "number", 0.0,
                                 min=lo, max=hi) for i in range(dim)])


def ackley(dim=3):
    sp = _box(dim, -5, 5)
    def f(q):
        v = np.array([q[f"x{i}"] for i in range(dim)])
        return (-20 * np.exp(-0.2 * np.sqrt(np.mean(v ** 2)))
                - np.exp(np.mean(np.cos(2 * np.pi * v))) + 20 + np.e)
    return sp, f, 0.0, np.zeros(dim)


def rosenbrock(dim=2):
    sp = _box(dim, -3, 3)
    def f(q):
        v = np.array([q[f"x{i}"] for i in range(dim)])
        return float(np.sum(100 * (v[1:] - v[:-1] ** 2) ** 2 + (v[:-1] - 1) ** 2))
    return sp, f, 0.0, np.ones(dim)


def six_hump():
    sp = ParamSpace([ParamSpec("x1", "x1", "number", 0.0, min=-2, max=2),
                     ParamSpec("x2", "x2", "number", 0.0, min=-2, max=2)])
    def f(q):
        x, y = q["x1"], q["x2"]
        return ((4 - 2.1 * x ** 2 + x ** 4 / 3) * x ** 2 + x * y
                + (-4 + 4 * y ** 2) * y ** 2)
    return sp, f, -1.0316, np.array([0.0898, -0.7126])


def goldstein_price():
    sp = ParamSpace([ParamSpec("x1", "x1", "number", 0.0, min=-2, max=2),
                     ParamSpec("x2", "x2", "number", 0.0, min=-2, max=2)])
    def f(q):
        x, y = q["x1"], q["x2"]
        a = (1 + (x + y + 1) ** 2
             * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2))
        b = (30 + (2 * x - 3 * y) ** 2
             * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
        return a * b
    return sp, f, 3.0, np.array([0.0, -1.0])


def hartmann(dim=3):
    sp = _box(dim, 0, 1)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    P = 10 ** -4 * np.array([[3689, 1170, 2673, 469, 1713, 1796],
                             [193, 110, 22147, 1530, 31687, 38547],
                             [101, 113, 16719, 4847, 24393, 27424],
                             [26, 161, 6250, 2820, 16517, 28792]])[:, :dim]
    def f(q):
        x = np.array([q[f"x{i}"] for i in range(dim)])
        return -float(np.sum(alpha * np.exp(
            -np.sum(A * (x - P) ** 2, axis=1))))
    return sp, f, -3.86278, np.array([0.114614, 0.555649, 0.852547])[:dim]


SYNTH = {
    "ackley_3d": ("Ackley 3D（多峰）", ackley, dict(dim=3)),
    "rosenbrock_2d": ("Rosenbrock 2D（香蕉谷）", rosenbrock, dict(dim=2)),
    "six_hump": ("六驼峰（全局最优已知）", six_hump, {}),
    "goldstein": ("Goldstein-Price", goldstein_price, {}),
    "hartmann_3d": ("Hartmann 3D", hartmann, dict(dim=3)),
}


def synth_names() -> list[str]:
    return list(SYNTH)


def synth_label(key: str) -> str:
    return SYNTH[key][0]


def make_synth(key: str):
    """-> (ParamSpace, fn, f_min, x_min)。"""
    label, builder, kw = SYNTH[key]
    return builder(**kw)


def make_objective_from_synth(key: str):
    """-> opt.Objective（UI/基准直接用）。"""
    from .contracts import make_objective
    sp, fn, f_min, x_min = make_synth(key)
    obj = make_objective(fn, sp, name=key)
    obj.f_min = f_min          # type: ignore[attr-defined]  基准判近真值用
    obj.x_min = x_min          # type: ignore[attr-defined]
    return obj
