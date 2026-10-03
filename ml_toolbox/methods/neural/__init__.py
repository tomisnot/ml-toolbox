# -*- coding: utf-8 -*-
"""神经网络方法族（方案 C：自研采集 + 成熟算法思想 + PageSpec 契约渲染）。

结构：
    recorder.py  TrainingRecorder —— 训练循环的"黑匣子"（零 Qt）
    mlp.py       TorchMLP —— 统一契约下的 MLP 分类/回归方法
    network.py   Network 契约 —— **AI 写结构**的作者面（见下）
    trainer.py   训练循环托管 —— 把 TB/进度/artifacts 从"作者的责任"变成"系统的保证"
    custom.py    CustomNNMethod —— 把 Network 适配成一台真正的 MLMethod
    nnplots.py   纯 matplotlib 的神经可视化函数（喂 mpl 页）

设计对齐：
- 方法层零 Qt：所有中间量装进 MLResult.artifacts / diag（契约 §3）；
- UI 新页类型经 inspector.PAGE_BUILDERS 注册（kind="nn_weights"/"nn_replay"），
  方法层只声明 kind + 数据放哪；
- torch 惰性 import：未装 torch 时本模块可导入，fit 才报错（不拖累其余 52 方法）。

⚠ `custom.py` 也在包加载时 import，但**它有双重理由**：
- 它把"自定义网络适配器"登记进边界层的槽（`ml_mecha.engine.register_custom_network`），
  不 import 就等于 `run_custom` 命令永远拿不到适配器；
- 它**顶层不 import torch**（`nn`/`torch` 都在函数体内取）⇒ 未装 torch 时照样导入得动，
  不拖累其余方法族的注册。
⚠ 它**不 `@register`**：名字进注册表走"注册→跑→撤"（引擎负责），
否则注册表里会恒留一个代码不对的空壳。
"""
from . import mlp      # noqa: F401  （@register 生效点）
from . import custom   # noqa: F401  （登记进边界层的槽；见上）
