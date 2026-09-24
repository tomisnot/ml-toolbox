# -*- coding: utf-8 -*-
"""ml_mecha —— ML Toolbox 的 mecha host adapter（薄接入层）。

归属：本包住在 **ML 仓**，不属于 mecha 框架仓。设计依据见
``docs/mecha/00-衔接基调与通用化宪章.md``：

- ML 不做 EL 化适配：命令名、状态键、工具面全部是 ML 自己的词汇；
- mecha 不做 ML 化特化：本包只使用 mecha 的通用机制（Gate / History /
  Surface / Monitor / Tools），不要求核心认识任何 ML 概念；
- 唯一装配点是 :func:`ml_mecha.assembly.assemble_ml_mecha`。

模块分工（各层都可单独 import，避免循环依赖）：

- :mod:`ml_mecha.engine`    —— ``MLEngine(Engine)``：受控命令分派 + 回执瘦身；
- :mod:`ml_mecha.validator` —— ML 状态键声明与校验（教学化错误）；
- :mod:`ml_mecha.summarizer`—— 人类可读摘要 + 可对账 ``Claim``；
- :mod:`ml_mecha.tools`     —— 从命令面投影出的模型可见工具；
- :mod:`ml_mecha.assembly`  —— ``assemble(...)`` 唯一装配点。

本模块**不**在 import 时急切导入子模块：``ml_mecha`` 的 import 不应依赖
mecha 是否已装（静态检查/文档工具只 import 包时也应能工作）。
"""

from __future__ import annotations

__all__ = ["engine", "validator", "summarizer", "tools", "assembly"]

#: 本适配器写入 mecha History 的**命令事件键前缀**（ML 词汇，不进 mecha 核心）。
COMMAND_EVENT_PREFIX = "ml.command."

#: 本适配器声明的**状态键前缀**（ML 词汇，不进 mecha 核心）。
STATE_KEY_PREFIX = "current."
