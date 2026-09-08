# -*- coding: utf-8 -*-
"""TrainingRecorder：训练循环的黑匣子（零 Qt，产出契约化 artifacts）。

记录四类中间量（对应四张检视页）：
  history   —— 每 epoch 的 loss/val_loss/lr/梯度全局范数/死 ReLU 比例
  grads     —— 每 epoch 每层梯度范数（TensorBoard 直方图的轻量等价）
  weights   —— 首/末 epoch 各层权重矩阵 + 末次更新幅度（W - lr*g）
  latent    —— 每 epoch 末最后一隐层输出经 PCA 降到 2D 的样本坐标（带 y 标签）

latent 是"表示演化回放"的数据基础：能看见类簇在第几个 epoch 开始分开。
"""
from __future__ import annotations

import numpy as np


class TrainingRecorder:
    """挂在 torch 训练循环上的采集器。

    用法（mlp.py 内部）::

        rec = TrainingRecorder(model, X, y, epochs=..., latent_every=5)
        for ep in range(epochs):
            ... train ...
            rec.epoch(ep, loss, val_loss, lr)
        out = rec.finalize(X, y)     # -> dict 装进 MLResult.artifacts
    """

    def __init__(self, model, n_samples: int, epochs: int,
                 latent_every: int = 5, max_latent_samples: int = 600,
                 dead_threshold: float = 1e-6):
        self.model = model
        self.epochs = epochs
        self.latent_every = max(1, latent_every)
        self.max_latent = max_latent_samples
        self.dead_thr = dead_threshold
        self.history = {"epoch": [], "loss": [], "val_loss": [], "lr": [],
                        "grad_norm": [], "dead_relu": []}
        self._layer_names = [n for n, p in model.named_parameters()
                             if p.requires_grad and p.dim() >= 2]
        self.grads = {n: [] for n in self._layer_names}
        # 初始权重：构造时（训练开始前）快照，否则"初始权重/ΔW"视图恒空
        self._weights_first = {n: p.detach().cpu().numpy().copy()
                               for n, p in model.named_parameters() if p.dim() >= 2}
        self._snapshots = []          # (epoch, latent Nx2)
        self._dead_per_relu = []      # 每个 ReLU 层的死神经元比例（最近一次）

    # ------------------------------------------------ 每 epoch 调用
    def epoch(self, ep: int, loss: float, val_loss: float | None,
              lr: float):
        import torch
        self.history["epoch"].append(ep)
        self.history["loss"].append(float(loss))
        self.history["val_loss"].append(
            float(val_loss) if val_loss is not None else float("nan"))
        self.history["lr"].append(float(lr))
        # 梯度范数（本步 backward 后、step 前记录的 .grad 还在线）
        total = 0.0
        for n, p in self.model.named_parameters():
            if p.dim() >= 2 and p.grad is not None:
                g = float(p.grad.norm())
                self.grads[n].append(g)
                total += g * g
            elif p.dim() >= 2:
                self.grads[n].append(0.0)
        self.history["grad_norm"].append(float(np.sqrt(total)))
        self.history["dead_relu"].append(self._dead_relu_fraction())
        if ep % self.latent_every == 0 or ep == self.epochs - 1:
            self._snapshots.append((ep, self._latent_2d()))

    def _dead_relu_fraction(self) -> float:
        """激活恒为 0 的 ReLU 神经元占比（容量浪费指标）。

        同时记录逐层值（self._dead_per_relu），逐层健康表按层消费；
        无 ReLU 的层（如末层 Linear 对应位置）填 NaN。
        """
        import torch
        import torch.nn as nn
        vals = []

        def hook(mod, inp, out):
            if isinstance(mod, nn.ReLU):
                with torch.no_grad():
                    dead = (out.abs().sum(dim=0) < self.dead_thr).float().mean()
                    vals.append(float(dead))

        hs = [m.register_forward_hook(hook) for m in self.model.modules()]
        try:
            self.model.eval()
            with torch.no_grad():
                self.model(self._probe)
        finally:
            for h in hs:
                h.remove()
            self.model.train()
        # hook 顺序 = 模块遍历顺序 = 层顺序；权重表行是 0,2,4...（Linear.weight）
        self._dead_per_relu = vals
        return float(np.mean(vals)) if vals else float("nan")

    def set_probe(self, X_tensor, labels=None, probe_idx=None):
        """给 hook 探测用的一小批输入（+ 对齐的标签/原始行号，回放页着色用）。"""
        self._probe = X_tensor
        self._probe_labels = labels
        self._probe_idx = probe_idx
        self.last_labels = None
        self.last_idx = None

    def _latent_2d(self):
        """最后一隐层输出 -> PCA 2D（子采样防大 n 卡住）。"""
        import torch
        from sklearn.decomposition import PCA
        feats = {}

        def hook(mod, inp, out):
            feats["f"] = out.detach()

        lin = [m for m in self.model.modules()
               if m.__class__.__name__.startswith("Linear")]
        h = lin[-2].register_forward_hook(hook) if len(lin) >= 2 else None
        try:
            self.model.eval()
            with torch.no_grad():
                self.model(self._probe)
        finally:
            if h:
                h.remove()
            self.model.train()
        f = feats.get("f")
        if f is None:
            return None
        f = f.cpu().numpy()
        sel = np.arange(f.shape[0])
        if f.shape[0] > self.max_latent:
            sel = np.linspace(0, f.shape[0] - 1, self.max_latent).astype(int)
            f = f[sel]
        # 记录与快照行对齐的标签/原始行号（回放页着色 + 归因选行）
        if self._probe_labels is not None:
            self.last_labels = np.asarray(self._probe_labels)[sel]
        if self._probe_idx is not None:
            self.last_idx = np.asarray(self._probe_idx)[sel]
        try:
            return PCA(n_components=2, random_state=0).fit_transform(f)
        except Exception:
            return None

    # ------------------------------------------------ 收尾
    def finalize(self) -> dict:
        """产出契约化 artifacts 字典（键约定见 docs/契约.md §3 扩展）。"""
        import torch
        out = {}
        out["nn_history"] = {k: np.asarray(v, float)
                             for k, v in self.history.items()}
        out["nn_grads"] = {n: np.asarray(v, float)
                           for n, v in self.grads.items()}
        last = {n: p.detach().cpu().numpy()
                for n, p in self.model.named_parameters() if p.dim() >= 2}
        out["nn_weights"] = {
            "first": {n: self._weights_first.get(n) for n in last},
            "last": last,
            "update": {n: (last[n] - self._weights_first[n])
                       if self._weights_first.get(n) is not None else None
                       for n in last},
        }
        snaps = [(ep, s) for ep, s in self._snapshots if s is not None]
        if snaps:
            out["nn_latent"] = np.stack([s for _, s in snaps])   # (T, n, 2)
            out["nn_latent_epochs"] = np.asarray([ep for ep, _ in snaps])
        # 逐层健康表（table 页直接消费）。
        # Sequential 里 Linear.weight 的下标是 0,2,4...，ReLU 死神经元比例
        # 按 hook 顺序（= 层顺序）对齐；末层无 ReLU 填 NaN。
        import pandas as pd
        dead = self._dead_per_relu
        rows = list(last)
        dead_col = []
        for n in rows:
            try:
                idx = int(n.split(".")[0])
            except (ValueError, IndexError):
                dead_col.append(float("nan"))
                continue
            dead_col.append(dead[idx // 2] if idx % 2 == 0 and idx // 2 < len(dead)
                            else float("nan"))
        out["nn_layers"] = pd.DataFrame({
            "参数量": [int(np.prod(last[n].shape)) for n in rows],
            "梯度范数(均值)": [float(np.asarray(self.grads[n]).mean())
                               if n in self.grads else float("nan")
                               for n in rows],
            "权重范数": [float(np.linalg.norm(last[n])) for n in rows],
            "死ReLU比例": dead_col,
        }, index=rows)
        return out
