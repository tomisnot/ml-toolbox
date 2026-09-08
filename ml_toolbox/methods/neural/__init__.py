# -*- coding: utf-8 -*-
"""神经网络方法族（方案 C：自研采集 + 成熟算法思想 + PageSpec 契约渲染）。

结构：
    recorder.py  TrainingRecorder —— 训练循环的"黑匣子"（零 Qt）
    mlp.py       TorchMLP —— 统一契约下的 MLP 分类/回归方法
    nnplots.py   纯 matplotlib 的神经可视化函数（喂 mpl 页）

设计对齐：
- 方法层零 Qt：所有中间量装进 MLResult.artifacts / diag（契约 §3）；
- UI 新页类型经 inspector.PAGE_BUILDERS 注册（kind="nn_weights"/"nn_replay"），
  方法层只声明 kind + 数据放哪；
- torch 惰性 import：未装 torch 时本模块可导入，fit 才报错（不拖累其余 52 方法）。
"""
from . import mlp  # noqa: F401  （@register 生效点）
