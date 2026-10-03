# -*- coding: utf-8 -*-
"""TensorBoard 桥件：把 ``TrainingRecorder`` 的中间量逐 epoch 送进事件文件。

## 为什么单独立一个文件

``TrainingRecorder`` 是**纯采集器**（docstring：零 Qt、产出契约化 artifacts）；它不该知道
TensorBoard 的存在。本模块只做一件事：**把已采集的量翻译成 TensorBoard 的轴**。
⇒ 采集与呈现分开，任一侧换实现都不牵动另一侧。

## ⚠ 它必须是【可选】的（三处各自 fail-safe）

1. **包不在**：``tensorboard`` 是 ``neural`` extra 的一部分，未装时**训练照常**，本模块全部空转；
2. **写失败**：事件写失败**绝不能拖垮训练**（与 ``finalize`` 里那句"采集失败不拖垮主结果"同一条纪律）；
3. **关掉**：``tb_enabled`` 为假时只做一次属性查询，不 import。
⇒ 三处都返回"什么都不做"，而不是抛错 —— 但**写失败要留一条痕**（见 ``note_error``），
不静默吞掉（否则"图是空的"会变成没人能解释的现象）。

## 轴映射（对应 TensorBoard 的原生入口）

| 采集到的 | 送去哪 | 入口 |
|---|---|---|
| ``history.loss`` / ``val_loss`` / ``lr`` / ``grad_norm`` / ``dead_relu`` | SCALARS | ``add_scalar`` |
| ``grads[层名]`` 逐 epoch 每层梯度范数 | SCALARS（每层一条 tag） | ``add_scalar('grad/层名')`` |
| ``weights.last[层名]`` 权重矩阵 | HISTOGRAMS / DISTRIBUTIONS | ``add_histogram`` |
| ``latent`` 帧（同坐标系） | IMAGES | ``add_image`` |
| ``nn_layers`` 逐层健康表 | TEXT（Markdown） | ``add_text`` |
| 权重的起始/末态/更新量 | SCALARS（范数随时间） | ``add_scalar`` |

⚠ **本条不"自己画图"**：图的形状由 TensorBoard 决定，我们只保证**数据按轴正确送达**。
"""

from __future__ import annotations

import numpy as np

#: 事件目录名（相对 ``runs/<run_id>/``）。与 `record.json`/`artifacts.npz` 并列。
TB_DIRNAME = "tb"

#: 五个核心标量（同时是最重要的人类可读信号）。
CORE_SCALARS = ("loss", "val_loss", "lr", "grad_norm", "dead_relu")


class TensorBoardLog:
    """逐 epoch 写 TensorBoard 的单向桥。**永不抛**（写失败只记账）。"""

    def __init__(self, run_id: str, root, enabled: bool = True) -> None:
        self.run_id = str(run_id or "")
        self.root = root
        self.enabled = bool(enabled) and bool(self.run_id)
        self._writer = None
        self._tried = False
        self._stack: list[str] = []

    # ------------------------------------------------------------ 内部
    def _ensure(self):
        """惰性建 writer（只试一次）。**不吞异常**：失败记进 ``_stack`` 并返回 None。"""
        if not self.enabled or self._tried:
            return self._writer
        self._tried = True
        try:
            from torch.utils.tensorboard import SummaryWriter
            import os
            logdir = os.path.join(str(self.root), self.run_id, TB_DIRNAME)
            os.makedirs(logdir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=logdir)
        except Exception as exc:                    # 包不在 / 建目录失败 / 写权限
            self.note_error("init", exc)
            self._writer = None
        return self._writer

    def note_error(self, where: str, exc: BaseException) -> None:
        """记一条写失败（**留痕，不静默**）。"""
        self._stack.append(f"{where}: {type(exc).__name__}: {exc}")

    def _safe(self, where: str, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            self.note_error(where, exc)

    # ------------------------------------------------------------ 对外
    @property
    def errors(self) -> tuple[str, ...]:
        """写失败记录（空 = 一路顺利）。给判据/诊断用。"""
        return tuple(self._stack)

    @property
    def active(self) -> bool:
        """writer 是否真的建起来了（判据用；不建 writer 也能过）。"""
        return self._writer is not None

    def log_epoch(self, ep: int, history: dict, grads: dict) -> None:
        """一个 epoch 的标量：5 条核心 + 每层梯度范数。"""
        w = self._ensure()
        if w is None:
            return
        step = int(ep)
        for key in CORE_SCALARS:
            seq = history.get(key) or []
            if len(seq) > step:
                v = float(seq[step])
                if np.isfinite(v):                  # ⚠ val_loss 可能是 NaN（无验证集）
                    self._safe(f"scalar/{key}", w.add_scalar, key, v, step)
        for name, seq in (grads or {}).items():
            if len(seq) > step:
                v = float(seq[step])
                if np.isfinite(v):
                    self._safe(f"scalar/grad/{name}", w.add_scalar,
                               f"grad/{name}", v, step)

    def log_weights(self, weights: dict | None, step: int) -> None:
        """权重直方图 + 权重的起始/末态/更新量范数（随时间各一条标量）。"""
        w = self._ensure()
        if w is None or not isinstance(weights, dict):
            return
        for name, mat in (weights.get("last") or {}).items():
            if mat is None:
                continue
            arr = np.asarray(mat, float).ravel()
            if arr.size:
                self._safe(f"hist/{name}", w.add_histogram, f"weights/{name}", arr, int(step))
        for tag, key in (("weights_norm/first", "first"),
                         ("weights_norm/last", "last"),
                         ("weights_norm/update", "update")):
            block = weights.get(key) or {}
            total = 0.0
            seen = False
            for mat in block.values():
                if mat is None:
                    continue
                total += float(np.linalg.norm(np.asarray(mat, float)))
                seen = True
            if seen:
                self._safe(f"scalar/{tag}", w.add_scalar, tag, total, int(step))

    def log_latent_frames(self, frames, epochs, *, size: int = 256) -> None:
        """把 PCA 投影帧当图送进 IMAGES（逐 epoch 一帧，同一坐标系 ⇒ 位移可解释）。

        ⚠ 需要 matplotlib；不在时**静默跳过**（这是"锦上添花"那一档，不该拖垮别的）。
        """
        w = self._ensure()
        if w is None or frames is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        arr = np.asarray(frames, float)
        if arr.ndim != 3:
            return
        eps = list(epochs) if epochs is not None else list(range(arr.shape[0]))
        for i in range(arr.shape[0]):
            try:
                fig, ax = plt.subplots(figsize=(2.6, 2.6), dpi=100)
                ax.scatter(arr[i, :, 0], arr[i, :, 1], s=5, alpha=0.7)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"epoch {eps[i]}", fontsize=7)
                fig.tight_layout(pad=0.2)
                fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
                plt.close(fig)
                self._safe(f"image/latent/{i}", w.add_image,
                           "latent/frame", buf, int(eps[i]), dataformats="HWC")
            except Exception as exc:
                self.note_error(f"image/latent/{i}", exc)
                try:
                    plt.close("all")
                except Exception:
                    pass

    def log_layers_table(self, table, step: int = 0) -> None:
        """逐层健康表 → TEXT（Markdown；TensorBoard 的 TEXT 面板支持表格）。"""
        w = self._ensure()
        if w is None or table is None:
            return
        try:
            md = table.to_markdown()
        except Exception:
            try:
                md = "```\n" + str(table) + "\n```"
            except Exception:
                return
        self._safe("text/layers", w.add_text, "layers/health", md, int(step))

    def close(self) -> None:
        """收尾：flush + close。**关闭日志本身可读**（TensorBoard 读文件，不读进程）。"""
        w = self._writer
        self._writer = None
        if w is None:
            return
        try:
            w.flush()
        except Exception as exc:
            self.note_error("flush", exc)
        try:
            w.close()
        except Exception as exc:
            self.note_error("close", exc)
