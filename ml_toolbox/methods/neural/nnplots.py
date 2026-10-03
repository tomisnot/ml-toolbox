# -*- coding: utf-8 -*-
"""神经网络可视化函数（纯 matplotlib，零 Qt）。

供 PageSpec(kind='mpl', plot=fn) 消费，对应四张检视页中的两张：
  plot_nn_dynamics        —— 训练动态（loss/val/lr/grad/dead ReLU）
  plot_feature_attribution —— 积分梯度归因条形图

另外两张（权重热图/表示演化回放）是自定义 widget，见 ui/neural_pages.py。
"""
from __future__ import annotations

from ...core import plot_mpl
from .. import specs as _specs


def plot_nn_dynamics(ax, result):
    """训练动态：loss/val_loss 主轴 + lr/grad_norm 副轴（log）。

    ⚠ **已迁移**：数由 `methods/specs.spec_nn_dynamics` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    plot_mpl.draw(_specs.spec_nn_dynamics(result), ax)


def plot_feature_attribution(ax, result):
    """积分梯度归因：Top-20 横向条形，正负逐色。

    ⚠ **已迁移**：数由 `methods/specs.spec_feature_attribution` 取，画法归 `core.plot_mpl` 渲染器。
    保留本函数名是因为 GUI 的 `PageSpec.plot` 指向它（调用点一行不改）。
    """
    plot_mpl.draw(_specs.spec_feature_attribution(result), ax)


