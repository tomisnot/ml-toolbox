# -*- coding: utf-8 -*-
"""**自定义网络契约（Network）**：AI 写"结构"，系统管"训练与观测"。

## 为什么需要它（用户裁决的设计）

`torch_mlp` 只有一层旋钮（层宽/激活/优化器…），而**拓扑**是写死的
`nn.Sequential` ⇒ 残差、多输入、任意前向都表达不了。把旋钮堆到 20 个也生不出
残差网络——那只是**把 AI 的自由度换成人手搓的旋钮**，方向是反的。

⇒ 本模块立一条**作者面契约**：只写下面这个 `Network`，
其余（切分、训练循环、进度、TensorBoard、artifacts、指标、存档、检视页）
全部由 `CustomNNMethod` 适配器代管。

```
AI 写的是 Network（结构 + 前向）        ← 作者面：自由、短、无样板
        ↓  CustomNNMethod 适配（custom.py）
跑的是 MLMethod（fit/predict/PageSpec）  ← 契约面：既有设施全部照用
```

**"神经网络是一种方法"这条外部语义不变**：注册表/遍历/对比/存档都当它是方法，
只是它的 `fit` 由适配器代写（所以 AI 不必写 `fit` 样板）。

## 契约（除 `build` 外全部可选）

```python
class Network:
    def build(self, n_in, n_out, *, nn):      # 必需：返回 nn.Module
        return nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, n_out))

    def loss(self, *, nn, n_out):             # 可选：默认按任务 CE / MSE
        return nn.CrossEntropyLoss()

    def configure(self, model, *, torch, lr, weight_decay):   # 可选：自定优化器
        return torch.optim.Adam(model.parameters(), lr=lr)

    def forward_step(self, model, batch, ctx):  # 可选：多输入/特殊批次形状
        x, y = batch
        return model(x), y
```

## ⚠ 三条硬纪律

1. **只经返回值交付效果**：`Network` 的代码**不许**自己写盘、不许碰 GUI、
   不许改 `runs/`——观测与存档由适配器统一负责。这样"AI 写的代码"与
   "AI 调 `run_method`"在记账上是同一类，审计面不会因此多一个缺口。
2. **执行本身是受控命令**（走 Gate 与审计）；沙箱（`ml_mecha/sandbox.py`）
   负责的是"别把宿主搞崩 + 错误可读"，**不是**权限边界。
3. **不静默**：代码抛错 ⇒ 结构化错误原样回给调用方（含可读归因），
   **不是**空结果，也不是"跑到第 0 个 epoch"。
"""
from __future__ import annotations

#: 适配器会按这个顺序探测的可选钩子（写在这里当单一真源，别处只引用）。
NETWORK_HOOKS = ("build", "loss", "configure", "forward_step")
