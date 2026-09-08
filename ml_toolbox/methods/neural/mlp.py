# -*- coding: utf-8 -*-
"""TorchMLP：统一契约下的神经网络方法（分类/回归双任务）。

fit 时 TrainingRecorder 全程采集中间量 -> artifacts 契约化 ->
UI 按 PageSpec 渲染（训练动态 / 权重热图 / 表示演化回放 / 特征归因）。

torch 惰性 import（不装 torch 不影响其余方法注册）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.contracts import (MLMethod, RunConfig, PageSpec, ParamSpec,
                               TASK_SUPERVISED, regression_metrics,
                               classification_metrics)
from ...core.registry import register
from ...core.pipeline import _infer_kind
from . import nnplots


@register
class TorchMLP(MLMethod):
    name = "torch_mlp"
    display_name = "神经网络 MLP"
    family = "neural"
    task = TASK_SUPERVISED
    target_kind = None
    tags = ("deep", "nonlinear", "gpu")
    param_schema = [
        ParamSpec("hidden", "隐藏层", "text", "64,32",
                  hint="逗号分隔的每层宽度，如 128,64,32"),
        ParamSpec("epochs", "训练轮数", "int", 60, min=5, max=2000),
        ParamSpec("lr", "学习率", "number", 1e-3, min=1e-6, max=1),
        ParamSpec("batch", "批大小", "int", 64, min=8, max=4096),
        ParamSpec("weight_decay", "权重衰减", "number", 0.0, min=0),
        ParamSpec("dropout", "Dropout", "number", 0.0, min=0, max=0.9),
        ParamSpec("val_split", "验证集比例", "number", 0.2, min=0, max=0.5,
                  hint="0 = 不切验证集"),
        ParamSpec("latent_every", "表示快照间隔", "int", 5, min=1,
                  hint="每 N epoch 存一帧隐层 2D 投影（回放页用）"),
        ParamSpec("device", "设备", "select", "auto",
                  choices=["auto", "cpu", "cuda"]),
    ]

    # ------------------------------------------------ 训练
    def fit(self, X: pd.DataFrame, y, cfg: RunConfig, diag: bool = False):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from .recorder import TrainingRecorder

        p = self.params(cfg)
        dev = self._device(p, torch)
        torch.manual_seed(cfg.seed)
        kind = self.target_kind or _infer_kind(y)

        Xv = X.to_numpy(float)
        self._mu, self._sd = Xv.mean(0), Xv.std(0) + 1e-9
        Xs = (Xv - self._mu) / self._sd
        if kind == "classification":
            from sklearn.preprocessing import LabelEncoder
            self._le = LabelEncoder()
            yv = self._le.fit_transform(np.asarray(y))
            n_out = len(self._le.classes_)
        else:
            self._le = None
            yv = np.asarray(y, float)
            n_out = 1

        # 验证集切分（时序无关的随机切分即可；数模表格场景通用）
        n = len(Xs)
        vs = float(p["val_split"])
        perm = np.random.RandomState(cfg.seed).permutation(n)
        if 0 < vs < 1:
            k = int(n * (1 - vs))
            tr_idx, va_idx = perm[:k], perm[k:]
        else:
            tr_idx, va_idx = perm, np.empty(0, int)

        model = self._build(p, Xs.shape[1], n_out, nn).to(dev)
        hidden = self._hidden_list(p)
        rec = TrainingRecorder(model, len(tr_idx), int(p["epochs"]),
                               latent_every=int(p["latent_every"]))
        probe_n = min(256, len(tr_idx))
        probe_rows = tr_idx[:probe_n]
        probe = torch.tensor(Xs[probe_rows], dtype=torch.float32, device=dev)
        rec.set_probe(probe, labels=yv[probe_rows], probe_idx=probe_rows)
        self._probe_idx = probe_rows

        Xt = torch.tensor(Xs[tr_idx], dtype=torch.float32, device=dev)
        yt = torch.tensor(yv[tr_idx], dtype=torch.float32 if kind != "classification"
                          else torch.long, device=dev)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=int(p["batch"]),
                            shuffle=True)
        Xva = (torch.tensor(Xs[va_idx], dtype=torch.float32, device=dev)
               if len(va_idx) else None)
        yva = (torch.tensor(yv[va_idx], dtype=torch.float32 if kind != "classification"
               else torch.long, device=dev) if len(va_idx) else None)

        opt = torch.optim.Adam(model.parameters(), lr=float(p["lr"]),
                               weight_decay=float(p["weight_decay"]))
        lossf = (nn.CrossEntropyLoss() if kind == "classification"
                 else nn.MSELoss())

        model.train()
        for ep in range(int(p["epochs"])):
            tot = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                out = model(xb)
                loss = lossf(out.squeeze(-1) if kind != "classification" else out, yb)
                loss.backward()
                opt.step()
                tot += float(loss) * len(xb)
            vl = None
            if Xva is not None:
                model.eval()
                with torch.no_grad():
                    o = model(Xva)
                    vl = float(lossf(o.squeeze(-1) if kind != "classification"
                                     else o, yva))
                model.train()
            rec.epoch(ep, tot / max(len(tr_idx), 1), vl, float(p["lr"]))

        res = self._new_result(target_kind=kind, params=p)
        res.est = model
        res.classes_ = (self._le.classes_ if kind == "classification"
                        and self._le is not None else None)
        res._dev, res._mu, res._sd = dev, self._mu, self._sd
        res._torch = torch
        try:
            res.artifacts = rec.finalize()
        except Exception:
            pass                       # 采集失败不拖垮主结果
        # 回放页着色标签：与 latent 快照同一子采样索引（recorder 已对齐）
        if "nn_latent" in res.artifacts:
            lab = rec.last_labels
            if lab is not None:
                res.artifacts["labels"] = lab
        if diag and hasattr(self, "_probe_idx") and len(self._probe_idx):
            try:
                rows = X.to_numpy(float)[self._probe_idx]
                res.artifacts["nn_attr"] = np.stack(
                    [self.feature_attribution(r, res) for r in rows])
                res.artifacts["nn_feat_names"] = np.array(
                    [str(c) for c in X.columns])
            except Exception:
                pass
        if diag:
            res.diag = {"epochs": int(p["epochs"]), "n_train": len(tr_idx),
                        "device": str(dev), "hidden": hidden}
        return res

    # ------------------------------------------------ 预测
    def predict(self, X: pd.DataFrame, result):
        torch = result._torch
        model = result.est
        Xs = (X.to_numpy(float) - result._mu) / result._sd
        t = torch.tensor(Xs, dtype=torch.float32, device=result._dev)
        model.eval()
        with torch.no_grad():
            out = model(t)
        if result.target_kind == "classification":
            pred = out.argmax(1).cpu().numpy()
            le = getattr(self, "_le", None)
            if le is None:
                # 换新实例调用：从 result 携带的 classes_ 重建编码器（同 base.py 约定）
                from sklearn.preprocessing import LabelEncoder
                classes = getattr(result, "classes_", None)
                if classes is None:
                    return pred
                le = LabelEncoder()
                le.classes_ = np.asarray(classes)
            return le.inverse_transform(pred)
        return out.squeeze(-1).cpu().numpy()

    def fit_extra_artifacts(self, X, result) -> dict:
        torch = result._torch
        model = result.est
        Xs = (X.to_numpy(float) - result._mu) / result._sd
        t = torch.tensor(Xs, dtype=torch.float32, device=result._dev)
        model.eval()
        if result.target_kind == "classification":
            with torch.no_grad():
                prob = torch.softmax(model(t), 1).cpu().numpy()
            return {"y_prob": prob}
        return {}

    # ------------------------------------------------ 归因（积分梯度，自实现）
    def feature_attribution(self, X_row: np.ndarray, result, steps: int = 32):
        """单样本特征归因：积分梯度（Captum 同源思想，零新依赖）。

        返回与 X 列等长的归因数组（对预测输出/所选类的贡献，正=推高）。
        """
        torch = result._torch
        model = result.est
        x0 = torch.tensor((X_row - result._mu) / result._sd,
                          dtype=torch.float32, device=result._dev).unsqueeze(0)
        base = torch.zeros_like(x0)
        path = base + (x0 - base) * torch.linspace(
            0, 1, steps, device=result._dev).view(-1, 1)
        path.requires_grad_(True)
        out = model(path)
        if result.target_kind == "classification":
            probs = torch.softmax(out, 1)
            tgt = probs.argmax(1)
            score = probs.gather(1, tgt.view(-1, 1)).sum()
        else:
            score = out.sum()
        grad, = torch.autograd.grad(score, path)
        avg = grad.mean(0)
        attr = ((x0 - base) * avg).squeeze(0).detach().cpu().numpy()
        return attr

    # ------------------------------------------------ 工具
    @staticmethod
    def _device(p, torch):
        if p["device"] == "cuda":
            return torch.device("cuda")
        if p["device"] == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _hidden_list(p):
        return [int(s) for s in str(p["hidden"]).replace(" ", "").split(",")
                if s.strip() and int(s) > 0]

    def _build(self, p, n_in, n_out, nn):
        layers, prev = [], n_in
        for h in self._hidden_list(p):
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if float(p["dropout"]) > 0:
                layers.append(nn.Dropout(float(p["dropout"])))
            prev = h
        layers.append(nn.Linear(prev, n_out))
        import torch.nn as nnm
        return nnm.Sequential(*layers)

    # ------------------------------------------------ 检视页声明（方案 C 四页）
    def inspect_pages(self, cfg):
        return [
            PageSpec("nn_dyn", "训练动态", "mpl", nnplots.plot_nn_dynamics),
            PageSpec("nn_w", "权重热图", "nn_weights",
                     hint="缺少权重快照工件"),
            PageSpec("nn_replay", "表示演化", "nn_replay",
                     hint="缺少隐层投影快照（检查 latent_every 参数）"),
            PageSpec("nn_attr", "特征归因", "mpl",
                     nnplots.plot_feature_attribution,
                     hint="归因在诊断开启时计算（积分梯度，32 步）"),
            PageSpec("nn_layers", "逐层健康", "table",
                     hint="缺少逐层表"),
        ]
