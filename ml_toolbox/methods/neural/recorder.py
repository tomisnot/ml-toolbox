# -*- coding: utf-8 -*-
"""TrainingRecorder：训练循环的黑匣子（零 Qt，产出契约化 artifacts）。

记录四类中间量（对应四张检视页）：
  history   —— 每 epoch 的 loss/val_loss/lr/梯度全局范数/死 ReLU 比例
  grads     —— 每 epoch 每层梯度范数（TensorBoard 直方图的轻量等价）
  weights   —— 首/末 epoch 各层权重矩阵 + 末次更新幅度
  latent    —— 冻结 PCA 坐标系中的最后一隐层表示演化（带 y 标签）

性能约定：每个 epoch 对 probe **最多做一次 forward**，同一次 forward 同时
采集 ReLU 激活和 latent；latent PCA 只在首个可投影快照拟合一次，后续帧只
transform。这样回放帧处于同一坐标系，动画位移才有可解释性。
"""
from __future__ import annotations

import numpy as np


class TrainingRecorder:
    """挂在 torch 训练循环上的采集器（torch 惰性 import）。"""

    def __init__(self, model, n_samples: int, epochs: int,
                 latent_every: int = 5, max_latent_samples: int = 600,
                 dead_threshold: float = 1e-6):
        self.model = model
        self.epochs = epochs
        self.latent_every = max(1, latent_every)
        self.max_latent = max(1, max_latent_samples)
        self.dead_thr = dead_threshold
        self.history = {"epoch": [], "loss": [], "val_loss": [], "lr": [],
                        "grad_norm": [], "dead_relu": []}
        self._layer_names = [n for n, p in model.named_parameters()
                             if p.requires_grad and p.dim() >= 2]
        self.grads = {n: [] for n in self._layer_names}
        self._weights_first = {n: p.detach().cpu().numpy().copy()
                               for n, p in model.named_parameters()
                               if p.dim() >= 2}
        self._snapshots = []
        self._dead_per_relu = []
        self._dead_by_linear = {}
        self._latent_pca = None
        self._probe = None
        self._probe_labels = None
        self._probe_idx = None
        self._probe_selection = None
        self.last_labels = None
        self.last_idx = None
        self._closed = False
        # 轻量运行计数：供性能契约测试与诊断，不改变公开 artifacts。
        self.probe_forward_count = 0
        self.pca_fit_count = 0
        self._install_hooks()

    # ------------------------------------------------ hook 管理
    def _install_hooks(self):
        """一次注册、整段训练复用；关闭 capture 时不保存任何激活。"""
        import torch.nn as nn
        modules = list(self.model.named_modules())
        linear_names = {n for n, m in modules if isinstance(m, nn.Linear)}
        self._relu_linear = {}
        self._relu_handles = []
        self._relu_activations = []
        last_linear = None
        for name, module in modules:
            if isinstance(module, nn.Linear):
                last_linear = name
            elif isinstance(module, nn.ReLU):
                # Sequential 的 Linear→ReLU（可夹 Dropout）映射；比旧版
                # idx//2 在启用 Dropout 后仍保持正确。
                self._relu_linear[name] = last_linear
                self._relu_handles.append(module.register_forward_hook(
                    self._relu_hook))

        self._latent_features = None
        linears = [(n, m) for n, m in modules
                   if m.__class__.__name__.startswith("Linear")]
        self._latent_handle = None
        if len(linears) >= 2:
            _, latent_module = linears[-2]       # 最后一隐层，排除输出层
            self._latent_handle = latent_module.register_forward_hook(
                self._latent_hook)
        self._linear_names = linear_names
        self._capture = False

    def _relu_hook(self, module, inputs, output):
        if self._capture:
            self._relu_activations.append(output.detach())

    def _latent_hook(self, module, inputs, output):
        if self._capture:
            self._latent_features = output.detach()

    def close(self):
        """移除 hook；可重复调用。finalize 后不再需要继续采集。"""
        if self._closed:
            return
        for h in self._relu_handles:
            h.remove()
        if self._latent_handle is not None:
            self._latent_handle.remove()
        self._closed = True
        self._capture = False
        self._relu_activations = []
        self._latent_features = None

    # ------------------------------------------------ 每 epoch 调用
    def epoch(self, ep: int, loss: float, val_loss: float | None,
              lr: float):
        self.history["epoch"].append(ep)
        self.history["loss"].append(float(loss))
        self.history["val_loss"].append(
            float(val_loss) if val_loss is not None else float("nan"))
        self.history["lr"].append(float(lr))
        # 梯度范数保留既有口径：调用 epoch 时最后 batch 的 .grad 仍在线。
        total = 0.0
        for n, p in self.model.named_parameters():
            if p.dim() >= 2 and p.grad is not None:
                g = float(p.grad.norm())
                self.grads[n].append(g)
                total += g * g
            elif p.dim() >= 2:
                self.grads[n].append(0.0)
        self.history["grad_norm"].append(float(np.sqrt(total)))

        latent = None
        if self._probe is not None:
            # 关键性能路径：原来 dead-ReLU forward + latent forward 合并为一次。
            latent = self._collect_probe()
        self.history["dead_relu"].append(self._dead_fraction())

        if latent is not None and (ep % self.latent_every == 0
                                   or ep == self.epochs - 1):
            self._snapshots.append((ep, self._project_latent(latent)))

    def set_probe(self, X_tensor, labels=None, probe_idx=None):
        """设置 probe；只计算一次行选择，标签/原行号与 latent 帧永久对齐。"""
        self._probe = X_tensor
        self._probe_labels = labels
        self._probe_idx = probe_idx
        n = int(X_tensor.shape[0])
        sel = np.arange(n)
        if n > self.max_latent:
            sel = np.linspace(0, n - 1, self.max_latent).astype(int)
        self._probe_selection = sel
        self.last_labels = (np.asarray(labels)[sel]
                            if labels is not None else None)
        self.last_idx = (np.asarray(probe_idx)[sel]
                         if probe_idx is not None else None)

    def _collect_probe(self):
        """一次 eval forward 同时返回 latent，并计算 dead-ReLU。"""
        import torch
        self._relu_activations = []
        self._latent_features = None
        self._capture = True
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
                self.model(self._probe)
            self.probe_forward_count += 1
            vals = []
            for act in self._relu_activations:
                dead = (act.abs().sum(dim=0) < self.dead_thr).float().mean()
                vals.append(float(dead))
            self._dead_per_relu = vals
            self._dead_by_linear = {}
            for relu_name, value in zip(self._relu_linear, vals):
                linear_name = self._relu_linear[relu_name]
                if linear_name is not None:
                    self._dead_by_linear[linear_name] = value
            latent = self._latent_features
            if latent is None:
                return None
            arr = latent.cpu().numpy()
            return arr[self._probe_selection]
        finally:
            self._capture = False
            self._relu_activations = []
            self._latent_features = None
            if was_training:
                self.model.train()

    def _dead_fraction(self):
        return (float(np.mean(self._dead_per_relu))
                if self._dead_per_relu else float("nan"))

    def _project_latent(self, features):
        """首个快照 fit PCA，后续只 transform；所有帧共享同一二维基。"""
        from sklearn.decomposition import PCA
        arr = np.asarray(features, float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return None
        try:
            if self._latent_pca is None:
                self._latent_pca = PCA(n_components=2, random_state=0).fit(arr)
                self.pca_fit_count += 1
            return np.asarray(self._latent_pca.transform(arr), float)
        except Exception:
            # PCA 失败不拖垮主结果；后续快照仍可尝试（通常同形状也会失败）。
            return None

    # ------------------------------------------------ 收尾
    def finalize(self) -> dict:
        """产出契约化 artifacts 字典（键/shape 保持兼容）。"""
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
            out["nn_latent"] = np.stack([s for _, s in snaps])
            out["nn_latent_epochs"] = np.asarray([ep for ep, _ in snaps])
        # 逐层健康表：dead 值按实际 Linear→ReLU 模块关系映射；输出层无 ReLU
        # 自然为 NaN，Dropout 不再让旧 idx//2 假设错位。
        import pandas as pd
        rows = list(last)
        dead_col = [float(self._dead_by_linear.get(n.rsplit(".", 1)[0], np.nan))
                    for n in rows]
        out["nn_layers"] = pd.DataFrame({
            "参数量": [int(np.prod(last[n].shape)) for n in rows],
            "梯度范数(均值)": [float(np.asarray(self.grads[n]).mean())
                               if n in self.grads else float("nan")
                               for n in rows],
            "权重范数": [float(np.linalg.norm(last[n])) for n in rows],
            "死ReLU比例": dead_col,
        }, index=rows)
        return out
