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

try:                                    # ⚠ 可选：TensorBoard 不在时训练照常
    from .tb_log import TensorBoardLog
except Exception:                       # pragma: no cover - 包结构异常时的兜底
    TensorBoardLog = None               # type: ignore[assignment]


class TrainingRecorder:
    """挂在 torch 训练循环上的采集器（torch 惰性 import）。

    ## TensorBoard（可选、逐 epoch 直播）

    传 ``run_id`` + ``tb_root`` 时，每个 epoch 会把已采集的量**实时**写进
    ``<tb_root>/<run_id>/tb/``（见 ``tb_log.py`` 的轴映射）。⇒ TensorBoard 进程
    指着 ``<tb_root>/<run_id>`` 就能看到曲线**逐 epoch 长出来**（人看 Web，
    AI 用 ``EventAccumulator`` 读同一份）。
    ⚠ **不传就是纯采集器**，行为与从前逐字一致（老调用方零改动）。
    """

    def __init__(self, model, n_samples: int, epochs: int,
                 latent_every: int = 5, max_latent_samples: int = 600,
                 dead_threshold: float = 1e-6,
                 run_id: str = "", tb_root=None, tb_enabled: bool = True):
        self.model = model
        self.epochs = epochs
        self.latent_every = max(1, latent_every)
        self.max_latent = max(1, max_latent_samples)
        self.dead_thr = dead_threshold
        self.tb = (TensorBoardLog(run_id, tb_root, enabled=tb_enabled)
                   if (TensorBoardLog is not None and run_id and tb_root is not None)
                   else None)
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
        #: 可选探针前向（图/多输入模型用；见 `set_probe`）。
        self._probe_fn = None
        self._probe_selection = None
        self.last_labels = None
        self.last_idx = None
        #: ⭐ 最新一次的**每个参数组 lr**（Q7b）：`custom.py` 收尾时写进 artifacts。
        self.lr_by_group: dict = {}
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
        # 按模块名收集；一次探针可以由多个 micro-batch 组成。
        self._relu_activations = {}
        last_linear = None
        for name, module in modules:
            if isinstance(module, nn.Linear):
                last_linear = name
            elif isinstance(module, nn.ReLU):
                # Sequential 的 Linear→ReLU（可夹 Dropout）映射；比旧版
                # idx//2 在启用 Dropout 后仍保持正确。
                self._relu_linear[name] = last_linear
                self._relu_handles.append(module.register_forward_hook(
                    lambda mod, inp, out, name=name:
                    self._relu_hook(name, mod, inp, out)))

        self._latent_features = None
        self._latent_name = None
        linears = [(n, m) for n, m in modules
                   if m.__class__.__name__.startswith("Linear")]
        self._latent_handle = None
        if len(linears) >= 2:
            self._latent_name, latent_module = linears[-2]       # 最后一隐层，排除输出层
            self._latent_handle = latent_module.register_forward_hook(
                self._latent_hook)
        self._linear_names = linear_names
        self._capture = False

    def _relu_hook(self, name, module, inputs, output):
        if self._capture:
            self._relu_activations.setdefault(name, []).append(
                output.detach().cpu())

    def _latent_hook(self, module, inputs, output):
        if self._capture:
            part = output.detach().cpu()
            if self._latent_features is None:
                self._latent_features = [part]
            else:
                self._latent_features.append(part)

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
        self._relu_activations = {}
        self._latent_features = None

    # ------------------------------------------------ 每 epoch 调用
    def epoch(self, ep: int, loss: float, val_loss: float | None,
              lr: float, extra: dict | None = None,
              lr_by_group: dict | None = None):
        """记一轮。

        ``extra``：**方法层自己上报的标量**（如 `{"fid/ch0": 0.96, "nll/phase": 1.2}`）。
        ⚠ 它们只往 TB 写（见 `tb_log.log_epoch`），**不进 `history`**——
        `history` 的形状有产物契约（`nn_history` 的键集）钉着，加键属行为变更。
        而"AI 看得见"这条已经由 TB 那条路满足了（`describe_training` 读**全部** tag）。

        ``lr_by_group``：**每个参数组的 lr**（Q7b，`{组名: lr}`）⇒ TB 写 `lr/<组名>`，
        而 `on_step`/`metrics` 也能从 `ctx["lr_by_group"]` 读到。
        用途：NQS 的**不对称步长**（幅度组/相位组不同 lr）"改了却无法确认生效"
        必须能查。⚠ 与 `extra` 同一条纪律：**不进 `history`**（那是产物契约）。
        """
        if lr_by_group:
            #: 最新一份留在实例上：`custom.py` 训练收尾时把它写进 artifacts，
            #: 供 `describe_run` / 存档看（不只是 TB）。
            self.lr_by_group = dict(lr_by_group)
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

        #: ⭐ 逐 epoch 直播（本 epoch 的量已在上面全部采集完）——写失败只记账，不拖垮训练。
        if self.tb is not None:
            self.tb.log_epoch(ep, self.history, self.grads, extra=extra,
                              lr_by_group=lr_by_group)

    def set_probe(self, X_tensor, labels=None, probe_idx=None, probe_fn=None):
        """设置 probe；只计算一次行选择，标签/原行号与 latent 帧永久对齐。

        ``probe_fn``：**可选的探针前向**（无参可调用，返回该次前向的输出）。
        为什么不总是 `self.model(self._probe)`：图任务（GNN）的前向需要
        `edge_index`，喂一个裸张量会直接抛——而且多输入模型同理。
        ⇒ 给了 `probe_fn` 就用它，否则沿用"喂张量"的老路（**零行为变化**）。
        """
        self._probe = X_tensor
        self._probe_labels = labels
        self._probe_idx = probe_idx
        self._probe_fn = probe_fn
        n = int(X_tensor.shape[0]) if X_tensor is not None else 0
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
        self._relu_activations = {}
        self._latent_features = None
        self._capture = True
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
                if self._probe_fn is not None:
                    self._probe_fn()          # 图/多输入：由调用方给出前向
                else:
                    self.model(self._probe)
            self.probe_forward_count += 1
            vals = []
            by_name = {}
            for relu_name in self._relu_linear:
                pieces = self._relu_activations.get(relu_name, [])
                if not pieces:
                    continue
                # 同一个 ReLU 实例可能被网络复用在不同宽度的层之间；这时
                # 不能强行 cat。逐段计算再取均值，既支持复用，也支持探针分块。
                groups = {}
                for part in pieces:
                    groups.setdefault(tuple(part.shape[1:]), []).append(part)
                per = []
                for same in groups.values():
                    act = (same[0] if len(same) == 1
                           else __import__("torch").cat(same, dim=0))
                    per.append(float((act.abs().sum(dim=0) < self.dead_thr)
                                     .float().mean()))
                value = float(np.mean(per)) if per else float("nan")
                vals.append(value)
                by_name[relu_name] = value
            self._dead_per_relu = vals
            self._dead_by_linear = {}
            for relu_name, value in by_name.items():
                linear_name = self._relu_linear[relu_name]
                if linear_name is not None:
                    self._dead_by_linear[linear_name] = value
            latent = self._latent_features
            if latent is None:
                return None
            if isinstance(latent, list):
                groups = {}
                for part in latent:
                    groups.setdefault(tuple(part.shape[1:]), []).append(part)
                # latent 通常只有一个形状；若网络复用线性层导致多形状，
                # 选行数最多的组，避免观测失败拖垮训练。
                latent = max(groups.values(), key=lambda xs: sum(len(x) for x in xs))
                latent = (latent[0] if len(latent) == 1
                          else __import__("torch").cat(latent, dim=0))
            arr = latent.numpy()
            sel = self._probe_selection
            sel = sel[sel < len(arr)]
            return arr[sel]
        finally:
            self._capture = False
            self._relu_activations = {}
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

        # ⭐ TensorBoard 收尾：权重直方图 / 范数、latent 帧、逐层健康表。
        # 标量已在每个 epoch 实时写过；这里补"只在收尾才完整"的那几轴。
        if self.tb is not None:
            last_ep = int(self.history["epoch"][-1]) if self.history["epoch"] else 0
            self.tb.log_weights(out["nn_weights"], last_ep)
            if "nn_latent" in out:
                self.tb.log_latent_frames(out["nn_latent"],
                                          out.get("nn_latent_epochs"))
            self.tb.log_layers_table(out.get("nn_layers"), last_ep)
            # ⭐ 最后写布局：把 ~20 个标量排成「训练动态 / 逐层梯度 / 权重范数」三组。
            #   不排的话 TensorBoard 按 tag 字典序铺，loss 与 grad/0.weight 混在一起。
            self.tb.write_layout(list(self.grads))
            self.tb.close()
        return out

    @property
    def tb_errors(self) -> tuple[str, ...]:
        """TensorBoard 写失败记录（诊断用；空 = 一路顺利）。"""
        return self.tb.errors if self.tb is not None else ()
