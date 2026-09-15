# -*- coding: utf-8 -*-
"""神经网络可视化函数（纯 matplotlib，零 Qt）。

供 PageSpec(kind='mpl', plot=fn) 消费，对应四张检视页中的两张：
  plot_nn_dynamics        —— 训练动态（loss/val/lr/grad/dead ReLU）
  plot_feature_attribution —— 积分梯度归因条形图

另外两张（权重热图/表示演化回放）是自定义 widget，见 ui/neural_pages.py。
"""
from __future__ import annotations

import numpy as np


def plot_nn_dynamics(ax, result):
    """训练动态：loss/val/lr/grad_norm/dead_relu 五联曲线。"""
    h = result.artifacts.get("nn_history")
    if h is None:
        ax.text(0.5, 0.5, "缺少 nn_history 工件", ha="center", va="center",
                color="#888")
        return
    epochs = h["epoch"]
    ax.plot(epochs, h["loss"], "b-", label="train loss", lw=1.4)
    if np.isfinite(h["val_loss"]).any():
        ax.plot(epochs, h["val_loss"], "r--", label="val loss", lw=1.4)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(fontsize=14, loc="upper right")
    ax.grid(alpha=0.25)
    # 副轴：lr + grad_norm
    ax2 = ax.twinx()
    ax2.plot(epochs, h["lr"], "g:", label="lr", lw=1)
    ax2.plot(epochs, h["grad_norm"], "m:", label="grad norm", lw=1)
    ax2.set_yscale("log")
    ax2.set_ylabel("lr / grad norm (log)")
    ax2.legend(fontsize=14, loc="lower right")
    ax.set_title("训练动态")


def plot_feature_attribution(ax, result):
    """积分梯度归因条形图：跨探针样本平均，正负着色，Top-N 绝对值。"""
    attr = result.artifacts.get("nn_attr")
    if attr is None:
        ax.text(0.5, 0.5, "归因未计算（勾选工具栏「诊断」后重跑）",
                ha="center", va="center", color="#888")
        ax.set_xticks([]); ax.set_yticks([])
        return
    attr = np.atleast_2d(np.asarray(attr, float))
    mean = attr.mean(axis=0)                 # 跨样本平均归因（带符号）
    names = result.artifacts.get("nn_feat_names")
    idx = np.argsort(np.abs(mean))[-20:]
    colors = ["tab:red" if mean[i] >= 0 else "tab:blue" for i in idx]
    ax.barh(range(len(idx)), mean[idx], color=colors)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([str(names[i])[:22] if names is not None
                        else f"f{i}" for i in idx], fontsize=14)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("平均积分梯度归因（红=推高预测 蓝=压低）")
    ax.set_title("特征归因（Top 20，%d 个探针样本平均）" % len(attr))
    ax.grid(alpha=0.25, axis="x")
