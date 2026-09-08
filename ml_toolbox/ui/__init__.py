# -*- coding: utf-8 -*-
"""UI 层：PyQt5 界面（只消费 core 契约，核心逻辑不依赖本包 —— P1）。"""
# 导入即注册神经检视页 builder（nn_weights / nn_replay）
from . import neural_pages  # noqa: F401
