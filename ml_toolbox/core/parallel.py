# -*- coding: utf-8 -*-
"""并行度单一控制点（M5/C5：并行度是一等配置，不是散落的魔法数）。

环境变量 `MLTB_NJOBS` 三态：
    off / 0 / serial  -> 1        受限环境（沙箱/CI/禁子进程）必用：
                                   joblib/loky 不再探测物理核（WinError 5 根因），
                                   并钉 NUMBA_NUM_THREADS=1（umap 卡死根因）
    auto / -1          -> -1       全核（12 核工作站调优用）
    正整数 N           -> N        指定核数
    （未设置）          -> 各调用点的 default

⚠ 默认值语义（保持历史行为）：
- 成员模型（RF/ExtraTrees/HistGB/IsolationForest/XGBoost/LightGBM）：-1；
- 元学习器外层（Voting/Stacking）：1——成员已并行，外层再并行嵌套超订
  （L18/pitfalls：Windows OpenMP 线程池互相挤兑曾致 access violation）。
"""
from __future__ import annotations

import os

_OFF = ("off", "0", "serial", "none")


def _mode() -> str | None:
    v = os.environ.get("MLTB_NJOBS", "").strip().lower()
    return v or None


# import 即生效：受限环境下尽早钉 numba 线程数（numba 在首次 import 时读取
# NUMBA_NUM_THREADS，之后改无效）。umap 的并行核函数随之单线程化，避免
# 沙箱/CI 里核探测 PermissionError 或 loky worker 反复重启卡死（§7 / L20）；
# joblib 系（RF/IF 等）由 nj() 返回 1 走串行路径，不触发核探测。
if _mode() in _OFF:
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")


def restricted() -> bool:
    """当前是否受限模式（测试豁免、numba 钉核都看这个）。"""
    v = _mode()
    return v in _OFF


def nj(default: int = -1) -> int:
    """本次运行的 n_jobs 值。default = 未设环境变量时的调用点默认。"""
    v = _mode()
    if v is None:
        return default
    if v in _OFF:
        # 受限环境：串行 + 钉死 numba 线程（必须在 numba 首次 import 前生效，
        # 故在进程启动早期设置；umap/joblib 的核探测随之跳过）
        os.environ.setdefault("NUMBA_NUM_THREADS", "1")
        return 1
    if v in ("auto", "-1"):
        return -1
    try:
        return max(1, int(v))
    except ValueError:
        return default


def meta_nj() -> int:
    """元学习器（Voting/Stacking）外层并行度：默认 1（L18 嵌套超订防线）。

    与 nj() 的差异：`auto` 不放行外层（成员已 -1，外层再 -1 = 线程池互踩）；
    仅当显式给正整数或 -1 时才按显式值。
    """
    v = _mode()
    if v is None or v in _OFF or v == "auto":
        return 1
    try:
        return max(1, int(v)) if v != "-1" else -1
    except ValueError:
        return 1


def skip_methods() -> set:
    """`MLTB_SKIP_METHODS=umap,lightgbm` -> 测试豁免集合（环境能力差异）。"""
    raw = os.environ.get("MLTB_SKIP_METHODS", "")
    return {s.strip() for s in raw.split(",") if s.strip()}
