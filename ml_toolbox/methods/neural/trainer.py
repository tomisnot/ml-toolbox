# -*- coding: utf-8 -*-
"""**神经网络训练器**：一个训练循环，托管全部观测面。

## 为什么要有它（用户裁决的"档 A"）

自定义网络最大的**静默风险**是：AI 自己写训练循环时**忘了**上报 ⇒ TensorBoard
与 `describe_training` **一起变空**，而训练本身"成功"了——正是本仓反复在防的
那种假绿。⇒ 循环由本模块托管，网络作者只写结构：

```
Network.build(...)  →  trainer.run(...)  →  recorder/TB/进度/artifacts 全部齐
```

于是"AI 写的网络"与 `torch_mlp` 在观测面上**完全同构**：loss / val_loss / lr /
grad_norm / dead_relu / 逐层梯度 / 权重直方图 / latent 帧，一个不少。

## 与 `mlp.py` 的关系（如实说明）

`TorchMLP.fit` 里也有一段等价的循环（先有它、后有本模块）。目前**两处并存**：
`TorchMLP` 走自己那段（保持既有行为与判据零风险），自定义网络走这里。
两边的**协议一致**（同样的 recorder 调用序列、同样的早停/裁剪/调度顺序），
由 `test_regressions.py` 的自定义网络判据与既有神经判据共同守着。
⇒ 合并成一处是后续清理项，不是本次的赌注。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PrepResult:
    """数据准备结果（两个消费者共用：`TorchMLP` 与 `CustomNNMethod`）。"""

    Xs: np.ndarray
    yv: np.ndarray
    tr_idx: np.ndarray
    va_idx: np.ndarray
    kind: str
    n_out: int
    mu: np.ndarray
    sd: np.ndarray
    label_encoder: object = None


def prepare_supervised(X: pd.DataFrame, y, *, kind: str, seed: int,
                       val_split: float) -> PrepResult:
    """标准化 + 标签编码 + 训练/验证切分（一次做全，避免两处漂移）。"""
    from sklearn.preprocessing import LabelEncoder

    Xv = X.to_numpy(float)
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-9
    Xs = (Xv - mu) / sd
    le = None
    if kind == "classification":
        le = LabelEncoder()
        yv = le.fit_transform(np.asarray(y))
        n_out = len(le.classes_)
    else:
        yv = np.asarray(y, float)
        n_out = 1
    n = len(Xs)
    perm = np.random.RandomState(seed).permutation(n)
    vs = float(val_split)
    if 0 < vs < 1:
        k = int(n * (1 - vs))
        tr, va = perm[:k], perm[k:]
    else:
        tr, va = perm, np.empty(0, int)
    return PrepResult(Xs=Xs, yv=yv, tr_idx=tr, va_idx=va, kind=kind,
                      n_out=n_out, mu=mu, sd=sd, label_encoder=le)


def train(recorder, *, model, prep: PrepResult, torch, nn,
          epochs: int, batch: int, lr: float, loss_fn,
          optimizer=None, scheduler=None, grad_clip: float = 0.0,
          early_stop: bool = False, patience: int = 10,
          on_progress=None, forward_step=None) -> dict:
    """跑训练循环，**托管全部观测**。

    @returns ``{"epochs_run": int, "stopped_at": int|None, "best_val": float}``

    ⚠ 上报顺序是有讲究的（与 `mlp.py` 一致，别改）：
    ``recorder.epoch(...)`` 记的是**本轮实际用的 lr** ⇒ 必须在 `scheduler.step()`
    **之前**调用；反了会把"下一轮的 lr"记成"本轮的 lr"，差一轮。
    """
    from torch.utils.data import DataLoader, TensorDataset

    tr, va = prep.tr_idx, prep.va_idx
    Xt = torch.tensor(prep.Xs[tr], dtype=torch.float32)
    yt = torch.tensor(prep.yv[tr],
                      dtype=torch.long if prep.kind == "classification"
                      else torch.float32)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=int(batch), shuffle=True)
    Xva = (torch.tensor(prep.Xs[va], dtype=torch.float32) if len(va) else None)
    yva = (torch.tensor(prep.yv[va],
                        dtype=torch.long if prep.kind == "classification"
                        else torch.float32) if len(va) else None)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    model.train()
    total = int(epochs)
    stopped_at, best, bad = None, float("inf"), 0
    cancelled = False
    best_epoch = None
    best_watch = float("inf")
    try:
        for ep in range(total):
            tot = torch.zeros((), dtype=torch.float32)
            for batch_items in loader:
                optimizer.zero_grad()
                if forward_step is not None:
                    out, target = forward_step(model, batch_items, None)
                    loss = loss_fn(out, target)
                else:
                    xb, yb = batch_items
                    out = model(xb)
                    loss = loss_fn(out.squeeze(-1)
                                   if prep.kind != "classification" else out, yb)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                tot = tot + loss.detach() * len(batch_items[0])
            vl = None
            if Xva is not None:
                model.eval()
                with torch.no_grad():
                    o = model(Xva)
                    vl = float(loss_fn(o.squeeze(-1)
                                       if prep.kind != "classification" else o, yva))
                model.train()
            train_loss = float(tot) / max(len(tr), 1)
            cur_lr = float(optimizer.param_groups[0]["lr"])
            recorder.epoch(ep, train_loss, vl, cur_lr)
            if scheduler is not None:
                scheduler.step()
            if on_progress is not None:
                # ⭐ 钩子**带数值**：AI 一次轮询就能判断"该不该早停"，不必再拉一次曲线。
                #   返回**真值** = 调用方请求停止（协作式取消的唯一落点）。
                try:
                    if on_progress(ep, total, "epoch", train_loss, vl):
                        stopped_at, cancelled = ep, True
                        break
                except TypeError:
                    # 老签名（3 参）也认：方法层不该假设调用方一定给新形状
                    try:
                        if on_progress(ep, total, "epoch"):
                            stopped_at, cancelled = ep, True
                            break
                    except Exception:                          # noqa: BLE001
                        on_progress = None
                except Exception:                              # noqa: BLE001
                    on_progress = None
            # ⭐ 记"验证损失最好的那一轮"：用户点破的洞见是**末代没有特殊意义**
            #   ⇒ 把"从哪一轮开始就没再好过"变成可查的事实（今天只存最后一轮的权重，
            #   所以至少要让 AI 知道它看的不是最佳模型）。没有验证集时退化成训练损失。
            watch = vl if vl is not None else train_loss
            if best_epoch is None or watch < best_watch - 1e-9:
                best_watch, best_epoch = watch, ep
            if early_stop and vl is not None:
                if vl < best - 1e-6:
                    best, bad = vl, 0
                else:
                    bad += 1
                    if bad >= max(1, int(patience)):
                        stopped_at = ep
                        break
    finally:
        recorder.close()
    return {"epochs_run": int(stopped_at + 1) if stopped_at is not None else total,
            "stopped_at": stopped_at, "best_val": best, "cancelled": cancelled,
            "best_epoch": best_epoch, "best_watch": best_watch}
