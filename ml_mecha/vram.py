# -*- coding: utf-8 -*-
"""**显存真相**：提交前"够不够"、跑完"用了多少"。

## 为什么需要（用户实测 §3.3，代价：一整晚）

6 GB 卡上他的配置峰值 **5.30 GB**，而运行时只剩 **3.6 GB** 可用 ⇒ OOM，
而且 CUDA 层把它报成 `device-side assert`（栈指向别处）。他的原话：
"**我事前完全不知道会不够。**" 以及"顺带：**自动分块应当是框架职责**"。

## 本模块只做两件**诚实**的事

1. **事前**：报出"现在卡上还剩多少"，并在**用户声明了预期峰值**时（
   `resource_guard.max_peak_mb`）与它比较 ⇒ 明显不够就**可读地拒**
   （`cuda_oom_predicted`，带三个数字与三条出路）。
   ⚠ **我们不去猜你网络的峰值**：任意 PyTorch 网络的内存占用取决于结构、批次、
   是否分块、优化器状态……启发式猜出来的数字会**假装精确**（那比不猜更坏）。
   所以：估算由**你**声明（或由你的网络声明），框架负责**对照实测可用量**执行。
2. **事后**：给出这次运行的**真实峰值**（`torch.cuda.max_memory_allocated`）
   ⇒ 你下一次就有数可依（"事前可估、事后可归因"）。
   没有 CUDA / 跑在 CPU ⇒ 回空表，**不假装有数字**。

## 纪律

- **无 CUDA 环境一律回空**（`{}`）：判据/CI 上不该因为它而红，也不该编造
  "0 MB 可用"这种会误导决策的值；
- 探针**只读**（`mem_get_info` / `max_memory_allocated`），不改分配器状态；
  跑完读峰值前**先 reset** 的是"统计量"而不是显存本身（见 `reset_peak`）。
"""
from __future__ import annotations

from typing import Any

_MIB = 1024.0 * 1024.0


def _torch():
    """惰性取 torch（没装/半初始化 ⇒ 回 ``None``，绝不让它把调用方炸掉）。"""
    try:
        import torch                                          # noqa: PLC0415
        return torch
    except Exception:                                          # noqa: BLE001
        return None


def probe(device: Any = None) -> dict:
    """当前显存概况：``{total_mb, free_mb, used_mb, device}``；无 CUDA ⇒ ``{}``。"""
    torch = _torch()
    if torch is None:
        return {}
    try:
        if not torch.cuda.is_available():
            return {}
        index = 0
        text = str(device or "")
        if ":" in text:
            index = int(text.split(":", 1)[1] or 0)
        free, total = torch.cuda.mem_get_info(index)
        return {"total_mb": round(total / _MIB, 1), "free_mb": round(free / _MIB, 1),
                "used_mb": round((total - free) / _MIB, 1),
                "device": f"cuda:{index}"}
    except Exception:                                          # noqa: BLE001
        return {}


def peak_mb(device: Any = None) -> float | None:
    """**本次进程**在 this device 上的实测峰值（MiB）；无 CUDA ⇒ ``None``。"""
    torch = _torch()
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        index = 0
        text = str(device or "")
        if ":" in text:
            index = int(text.split(":", 1)[1] or 0)
        return round(float(torch.cuda.max_memory_allocated(index)) / _MIB, 1)
    except Exception:                                          # noqa: BLE001
        return None


def reset_peak(device: Any = None) -> None:
    """把**峰值统计**清零（下一个窗口才量得准）。⚠ 不动已分配的张量。"""
    torch = _torch()
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            index = 0
            text = str(device or "")
            if ":" in text:
                index = int(text.split(":", 1)[1] or 0)
            torch.cuda.reset_peak_memory_stats(index)
    except Exception:                                          # noqa: BLE001
        pass


def precheck(spec: Any) -> dict:
    """按**用户声明的**预期峰值做显存预检（不够 ⇒ 可读地拒）。

    ## 为什么只认"声明"而不猜

    任意 PyTorch 网络的峰值取决于结构/批次/优化器状态/是否分块……启发式猜出来的
    数字会**假装精确**，而"假装精确的预算"比没有预算更坏（它会让人信任一个错的数）。
    ⇒ 本仓的立场：**估算由声明给出**（`resource_guard.max_peak_mb`，MiB），
    框架负责**对照实测可用量**执行它，并把三个数字都摆出来。

    ## 四种情形（都不静默）

    - 没声明 ⇒ **不判**（但我们仍会在别处报可用量，用户迟早需要它）；
    - 声明不是数字 ⇒ `bad_peak_mb`（可教学拒，不静默忽略）；
    - 明确 `device='cpu'` ⇒ 不判（跑 CPU 不涉及显存）；
    - 有 CUDA 且 `声明 > 可用 × 1.05` ⇒ `cuda_oom_predicted`（留 5% 余量：
      "刚好等于"不该被拒；用户原话是"明显不足就立刻拒"）。
    """
    from mecha.errors import MechaError

    if not isinstance(spec, dict):
        return {}
    guard = spec.get("resource_guard")
    if not isinstance(guard, dict):
        return {}
    want = guard.get("max_peak_mb")
    if want in (None, "", 0):
        return {}
    try:
        want_mb = float(want)
    except (TypeError, ValueError):
        raise MechaError(
            f"resource_guard.max_peak_mb 需要数字（MiB），收到 {want!r}",
            kind="bad_peak_mb",
            hint="它声明的是「这次运行预计的显存峰值」，框架据此做提交前预检",
        ) from None
    device = str(spec.get("device") or "")
    if device == "cpu":
        return {}
    info = probe(device or None)
    if not info:
        return {}                       # 没有 CUDA（或探不到）⇒ 不判，**不编数字**
    free = float(info.get("free_mb") or 0.0)
    if want_mb > free * 1.05:
        raise MechaError(
            f"预计显存峰值 {want_mb:.0f} MiB，而卡上只剩 {free:.0f} MiB"
            f"（总 {info.get('total_mb')} MiB）⇒ **大概会 OOM，先别跑**",
            kind="cuda_oom_predicted",
            hint="这是**提交前**的可读拒绝（不是训练失败）：省掉一次必然失败的运行，"
                 "也避免 OOM 被 CUDA 报成 device-side assert（那会指向假位置）。"
                 "⚠ 数字来自你声明的 `resource_guard.max_peak_mb` 与实测可用显存；"
                 "要更准就用上一次运行的 `peak_mb`（回执/存档里有）",
            suggest=(f"减 `overrides.batch`（例如减半再试）、给网络开分块"
                     f"（`ctx[\"chunks\"]`，`resource_guard.chunk_budget_mb=…`）、"
                     f"或换 `device='cpu'`；等卡上腾出 ≥{want_mb:.0f} MiB 再投"),
        )
    return info


__all__ = ["probe", "peak_mb", "reset_peak", "precheck"]
