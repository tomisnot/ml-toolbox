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

#: 可选的激活函数（**名字 → 构造函数**；`nn` 是 torch.nn，惰性传进来）。
#: 为什么用"名字 → 构造"而不是直接放模块：本模块顶层**不 import torch**
#: （torch 是可选依赖，未装时其余 58 个方法照常注册）。
ACTIVATIONS = {
    "relu": lambda nn: nn.ReLU(),
    "tanh": lambda nn: nn.Tanh(),
    "sigmoid": lambda nn: nn.Sigmoid(),
    "gelu": lambda nn: nn.GELU(),
    "leaky_relu": lambda nn: nn.LeakyReLU(negative_slope=0.01),
    "elu": lambda nn: nn.ELU(),
}

#: 可选优化器（**有序**：`frozenset` 的遍历顺序不稳定 ⇒ 面板下拉框与
#: `describe_method` 回给 AI 的 `choices` 会每次不一样。显示顺序是接口的一部分）。
OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop")

#: 可选学习率调度（同样有序）。
SCHEDULES = ("none", "step", "cosine")

#: 可选损失（**只放不改变输出形状的**）。
#: ⚠ 为什么没有 bce / focal：那类损失要 `n_out=1` + sigmoid，会连带改变
#: `y_prob` 的形状与 `predict` 的取类方式 ⇒ 属于"自定义代码"那一档
#: （见 `docs/pitfalls.md` 的 A4：宁可少一个旋钮，不要一个"看起来生效"的假象）。
LOSSES = ("auto", "mse", "celoss")

#: 可选权重初始化。`default` = torch 出厂（Linear 的 kaiming-uniform + bias）。
INITS = ("default", "xavier", "kaiming")


@register
class TorchMLP(MLMethod):
    name = "torch_mlp"
    display_name = "神经网络 MLP"
    family = "neural"
    task = TASK_SUPERVISED
    target_kind = None
    tags = ("deep", "nonlinear", "gpu")

    @staticmethod
    def _tb_root():
        """TensorBoard 事件目录的根（= 落盘同一个 ``RUNS_DIR``）。

        ⚠ **惰性 import**：``core.persistence`` 会 import 本族的注册面，模块顶层 import 它会成环。
        取不到就回 ``None`` ⇒ 调用方（recorder）退化成"不写 TensorBoard"，**训练照常**。
        """
        try:
            from ...core.persistence import RUNS_DIR
            return RUNS_DIR
        except Exception:
            return None

    param_schema = [
        ParamSpec("hidden", "隐藏层", "text", "64,32",
                  hint="逗号分隔的每层宽度，如 128,64,32"),
        ParamSpec("activation", "激活函数", "select", "relu",
                  choices=list(ACTIVATIONS),
                  hint="每层 Linear 之后接哪个非线性（默认 relu）"),
        ParamSpec("epochs", "训练轮数", "int", 60, min=5, max=2000),
        ParamSpec("optimizer", "优化器", "select", "adam",
                  choices=list(OPTIMIZERS),
                  hint="sgd 才吃 momentum；给 adam 传 momentum 会被拒绝（不静默忽略）"),
        ParamSpec("lr", "学习率", "number", 1e-3, min=1e-6, max=1),
        ParamSpec("momentum", "动量（仅 sgd）", "number", 0.0, min=0.0, max=0.99,
                  hint="非 sgd 优化器传非 0 值 ⇒ 明确报错，不假装生效"),
        ParamSpec("batch", "批大小", "int", 64, min=8, max=4096),
        ParamSpec("weight_decay", "权重衰减", "number", 0.0, min=0),
        ParamSpec("dropout", "Dropout", "number", 0.0, min=0, max=0.9),
        ParamSpec("lr_schedule", "学习率调度", "select", "none",
                  choices=list(SCHEDULES),
                  hint="step 每 lr_step 轮 ×lr_gamma；cosine 余弦退火到 0"),
        ParamSpec("lr_step", "调度步长（仅 step）", "int", 30, min=1,
                  hint="每多少轮衰减一次（仅 lr_schedule=step 生效）"),
        ParamSpec("lr_gamma", "衰减系数（仅 step）", "number", 0.5, min=0.01, max=1.0,
                  hint="每次衰减乘这个系数（仅 lr_schedule=step 生效）"),
        ParamSpec("loss", "损失函数", "select", "auto",
                  choices=list(LOSSES),
                  hint="auto=分类用交叉熵/回归用 MSE；bce 这类会改变输出形状的"
                       "损失不在此列（它影响 y_prob 轮廓），要它请走自定义代码"),
        ParamSpec("init", "权重初始化", "select", "default",
                  choices=list(INITS),
                  hint="xavier 适合 tanh/sigmoid，kaiming 适合 relu 家族"),
        ParamSpec("grad_clip", "梯度裁剪", "number", 0.0, min=0.0,
                  hint="0 = 不裁剪；>0 时按全局范数裁剪（治梯度爆炸的常用手段）"),
        ParamSpec("early_stop", "早停", "select", "off",
                  choices=["off", "on"],
                  hint="on ⇒ 验证损失连续 early_stop_patience 轮不改善就停"
                       "（需要 val_split > 0，否则拒绝）"),
        ParamSpec("early_stop_patience", "早停耐心", "int", 10, min=1,
                  hint="连续多少轮没改善就停（仅 early_stop=on 生效）"),
        ParamSpec("val_split", "验证集比例", "number", 0.2, min=0, max=0.5,
                  hint="0 = 不切验证集"),
        ParamSpec("latent_every", "表示快照间隔", "int", 5, min=1,
                  hint="每 N epoch 存一帧隐层 2D 投影（回放页用）"),
        ParamSpec("device", "设备", "select", "auto",
                  choices=["auto", "cpu", "cuda"]),
    ]

    # ------------------------------------------------ 训练
    def fit(self, problem, cfg: RunConfig, diag: bool = False):
        """内置 MLP（`Problem` 范式）：只吃训练段，就是原来的 (Xtr, ytr)。

        ⚠ 结构化数据（图）不走这里：`torch_mlp` 的 `accepts_aux=()`（不声明支持），
        带辅助结构的数据应由 `run_custom` 的 Network 代码消费 —— 否则会被
        `Problem.require` 结构化拒绝（**不静默忽略**）。
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from .recorder import TrainingRecorder

        X, y = problem.rows("train")
        p = self.params(cfg)
        dev = self._device(p, torch, cfg)
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
                               latent_every=int(p["latent_every"]),
                               # ⭐ 直播：runner 在跑之前就定好 run_id（见 core/runner.py），
                               #   经 cfg.extras 传下来 ⇒ 事件目录 runs/<run_id>/tb/ 全程不变，
                               #   TensorBoard 指着它就能看曲线逐 epoch 长出来。
                               run_id=str(getattr(cfg, "extras", {}).get("run_id") or ""),
                               tb_root=self._tb_root())
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

        opt = self._optimizer(p, model.parameters(), torch)
        sched = self._scheduler(p, opt, torch)
        lossf = self._loss(p, kind, nn)
        self._init_weights(model, p, nn)
        clip = float(p.get("grad_clip") or 0.0)
        # 早停：**声明要它就必须真给验证集**——val_split=0 时没有任何"改善"可判断，
        # 静默忽略等于让这个旋钮形同生效（本仓纪律：参数不许骗人）。
        want_early = str(p.get("early_stop") or "off").lower() == "on"
        if want_early and Xva is None:
            raise ValueError(
                "early_stop='on' 需要验证集，但 val_split=0 ⇒ 无法判断'有没有改善'。"
                "请设 val_split>0，或把 early_stop 关掉（不静默忽略）")
        patience = max(1, int(p.get("early_stop_patience") or 10))
        stopped_at, best_vl, bad = None, float("inf"), 0

        model.train()
        #: 方法层可选的**进度钩子**（`runner.run_one` 从 `cfg.extras["on_progress"]`
        #: 认它，引擎经 `ExecutionContext.report_progress` 注入）。
        #: 没有就是 None ⇒ 一行都不多跑（P3 零侵入默认）。
        on_progress = cfg.extras.get("on_progress")
        total_epochs = int(p["epochs"])
        try:
            for ep in range(total_epochs):
                # 在 epoch 内保持 loss 为 device tensor，避免 CUDA 每个 batch
                # 通过 float(loss) 同步；只在 recorder 需要标量时同步一次。
                tot = torch.zeros((), dtype=torch.float32, device=dev)
                for xb, yb in loader:
                    opt.zero_grad()
                    out = model(xb)
                    loss = lossf(out.squeeze(-1) if kind != "classification"
                                 else out, yb)
                    loss.backward()
                    if clip > 0:
                        # 全局范数裁剪：治梯度爆炸的常规手段（对每个 batch 生效）
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    opt.step()
                    tot += loss.detach() * len(xb)
                vl = None
                if Xva is not None:
                    model.eval()
                    with torch.no_grad():
                        o = model(Xva)
                        vl = float(lossf(o.squeeze(-1)
                                         if kind != "classification" else o, yva))
                    model.train()
                # ⭐ 报**当前实际**学习率（不是参数里那个常量）：上了调度器之后
                #   两者会分叉。若仍写 `p["lr"]`，TB 的 lr 曲线会是**一条平的直线**
                #   而真实 lr 在衰减 —— 那是"图在骗人"，本仓反复在防的那类假绿。
                cur_lr = float(opt.param_groups[0]["lr"])
                rec.epoch(ep, float(tot) / max(len(tr_idx), 1), vl, cur_lr)
                if sched is not None:
                    # ⚠ 放在 rec.epoch 之后：本轮的 lr 已经记下了，调度影响的是下一轮。
                    #   顺序反了会把"衰减后的 lr"记成"本轮用的 lr"，差一轮。
                    sched.step()
                if on_progress is not None:
                    # ⚠ 钩子失败**不许**拖垮训练（引擎侧已吞异常，这里再兜一层：
                    #   方法层不该假设调用方守纪律）——观测是附加的，主结果优先。
                    try:
                        on_progress(ep, total_epochs, "epoch")
                    except Exception:
                        on_progress = None
                if want_early and vl is not None:
                    # 早停判据用**验证损失**（不是训练损失——后者总在降，停不下来）
                    if vl < best_vl - 1e-6:
                        best_vl, bad = vl, 0
                    else:
                        bad += 1
                        if bad >= patience:
                            stopped_at = ep
                            break
        finally:
            # hook 只服务 recorder probe；训练结束立即移除，避免每个训练
            # batch 都经过一次无效 Python hook。
            rec.close()

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
            # ⭐ 报**生效的结构与训练配置**（不只是层宽）：AI 想核对自己"设了什么"
            #   时，`params` 里那串原始覆写不足以说明"到底跑的是什么"——
            #   例如 lr 上了调度会衰减、val_split=0 时 val_loss 整条不存在、
            #   早停会让实际轮数**少于** epochs。这些都必须说出来。
            res.diag = {"epochs": int(p["epochs"]),
                        "epochs_run": int(stopped_at + 1) if stopped_at is not None
                        else total_epochs,
                        "early_stopped_at": stopped_at,
                        "loss": str(p.get("loss") or "auto"),
                        "init": str(p.get("init") or "default"),
                        "grad_clip": float(p.get("grad_clip") or 0.0),
                        "n_train": len(tr_idx), "device": str(dev), "hidden": hidden,
                        "activation": str(p.get("activation") or "relu"),
                        "optimizer": str(p.get("optimizer") or "adam"),
                        "lr_schedule": str(p.get("lr_schedule") or "none"),
                        "lr_final": float(opt.param_groups[0]["lr"]),
                        "n_val": len(va_idx), "n_test_probe": int(probe_n)}
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
    def _device(p, torch, cfg=None):
        """设备：**引擎透传的声明优先**，其次方法自己的 `device` 参数。

        ⚠ 为什么是这个优先级（2026-10-03 定，之前引擎那个字段是死参数）：
        - 引擎侧 `device`（`run_method(device=...)`）是**运行级声明**，也写进了
          `current.device` 域状态与审计 ⇒ 它必须是"实际用的那个"，否则
          审计里写的设备与真跑的设备会不一致（那比没有更糟）；
        - `device='auto'` 是"我不关心"的意思 ⇒ 不覆盖方法参数，让
          `torch_mlp` 自己的 `device`（也是 auto/cpu/cuda）说话。
        ⇒ 于是 `run_method(device="cpu")` 与 `overrides={"device": "cpu"}`
        两条路都能生效，且**不会互相打架**。
        """
        declared = ""
        if cfg is not None:
            declared = str((getattr(cfg, "extras", {}) or {}).get("device") or "")
        own = str(p.get("device") or "auto")
        # ⚠ 优先级：**具体 > auto**，两边都具体时引擎的运行级声明赢。
        #   为什么不是"引擎一律赢"：引擎的 `device` 默认就是 `auto`（= 「我不关心」），
        #   若让它无条件覆盖，`overrides={"device": "cpu"}` 会被静默吃掉 ⇒
        #   又一个"参数形同生效"（正是本仓在防的那类假绿）。
        if own not in ("", "auto"):
            chosen = own
        elif declared.split(":")[0].strip().lower() not in ("", "auto"):
            chosen = declared
        else:
            chosen = "auto"
        kind, _, index = chosen.partition(":")
        kind = kind.strip().lower()
        index = index.strip()
        if kind == "cuda":
            return torch.device(f"cuda:{index}" if index else "cuda")
        if kind == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _hidden_list(p):
        return [int(s) for s in str(p["hidden"]).replace(" ", "").split(",")
                if s.strip() and int(s) > 0]

    # ------------------------------------------------ 结构装配（AI 可配的那几个旋钮）
    def _activation(self, name, nn):
        """激活函数模块。**未知取值抛**（不静默退回 relu——那会让参数形同生效）。"""
        key = str(name or "relu").lower()
        fn = ACTIVATIONS.get(key)
        if fn is None:
            raise ValueError(f"未知 activation {name!r}；可选 {sorted(ACTIVATIONS)}")
        return fn(nn)

    def _optimizer(self, p, params, torch):
        """优化器。**与优化器无关的旋钮给了非默认值 ⇒ 明确报错**。

        为什么报错而不是忽略（本仓纪律：参数不许形同生效）：
        ``momentum`` 只对 sgd 有意义；若给 adam 传 ``momentum=0.9`` 而静默忽略，
        AI/人会以为"我设了动量"，而实际训练里根本没有——那就是**参数在骗人**。
        ⇒ 宁可让他当场看见一条可读错误，也不要一个"看起来生效了"的假象。
        """
        name = str(p.get("optimizer") or "adam").lower()
        lr = float(p["lr"])
        wd = float(p["weight_decay"])
        momentum = float(p.get("momentum") or 0.0)
        if name != "sgd" and momentum:
            raise ValueError(
                f"momentum={momentum} 只对 optimizer='sgd' 有意义，"
                f"当前 optimizer={name!r} ⇒ 拒绝静默忽略（请去掉它或改用 sgd）")
        if name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=momentum,
                                   weight_decay=wd)
        if name == "rmsprop":
            return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
        raise ValueError(f"未知 optimizer {name!r}；可选 {sorted(OPTIMIZERS)}")

    def _scheduler(self, p, opt, torch):
        """学习率调度器；``none`` ⇒ ``None``（**零开销，不多走一步**）。"""
        name = str(p.get("lr_schedule") or "none").lower()
        if name == "none":
            return None
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                opt, step_size=int(p.get("lr_step") or 30),
                gamma=float(p.get("lr_gamma") or 0.5))
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, int(p["epochs"])))
        raise ValueError(f"未知 lr_schedule {name!r}；可选 {sorted(SCHEDULES)}")

    def _loss(self, p, kind, nn):
        """损失函数。``auto`` = 今天的出厂行为（分类交叉熵 / 回归 MSE）。

        ⚠ **与任务不匹配的组合直接拒**（不静默换回默认）：给回归配 ``celoss``
        会在反向传播时炸在形状上，而"悄悄换成 mse"会让人以为交叉熵生效了。
        这也是损失函数**没有** `bce` 的原因——它要改输出形状（见 `LOSSES` 说明）。
        """
        name = str(p.get("loss") or "auto").lower()
        if name == "auto":
            return (nn.CrossEntropyLoss() if kind == "classification"
                    else nn.MSELoss())
        if name == "mse":
            if kind == "classification":
                raise ValueError(
                    "loss='mse' 与分类任务不匹配（用自动的交叉熵，或换回归数据）")
            return nn.MSELoss()
        if name == "celoss":
            if kind != "classification":
                raise ValueError(
                    "loss='celoss' 只用于分类任务（回归请用 loss='mse' 或 auto）")
            return nn.CrossEntropyLoss()
        raise ValueError(f"未知 loss {name!r}；可选 {sorted(LOSSES)}")

    def _init_weights(self, model, p, nn):
        """按声明做权重初始化。``default`` ⇒ 原样返回（**零动作**）。

        ⚠ 只初始化隐藏层（最后一层 `Linear` 是输出头，重置它没有理论依据）；
        遍历用 `named_modules` 而不是直接 `children`，这样将来换成嵌套结构
        （残差块等）时不用改这里。
        """
        name = str(p.get("init") or "default").lower()
        if name == "default":
            return
        kind = str(p.get("activation") or "relu").lower()
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if not linears:
            return
        for layer in linears[:-1]:                     # 留输出头不动
            if name == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif name == "kaiming":
                # relu 家族用 fan_out + relu 模式；tanh/sigmoid 退回 xavier 更稳
                if kind in ("relu", "leaky_relu", "elu", "gelu"):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(layer.weight)
            else:
                raise ValueError(f"未知 init {name!r}；可选 {sorted(INITS)}")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _build(self, p, n_in, n_out, nn):
        act = self._activation(p.get("activation"), nn)
        layers, prev = [], n_in
        for h in self._hidden_list(p):
            layers += [nn.Linear(prev, h), act]
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
