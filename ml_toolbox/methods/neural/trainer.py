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

import contextlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ...core.chunks import RowChunks

#: 从 `resource_guard` 里读分块预算（MiB）；没给就用 `core.chunks` 的默认。
#: ⚠ 为什么走 `resource_guard` 而不是新开一个参数：它本来就是"这次运行的资源声明"
#: （`max_*_mb` 都在里面）⇒ 分块预算是同类事实，不该分裂成第二个旋钮。
def chunk_budget_from(extras: Any) -> float | None:
    try:
        guard = (extras or {}).get("resource_guard") or {}
        raw = guard.get("chunk_budget_mb")
        return float(raw) if raw not in (None, "") else None
    except Exception:                                          # noqa: BLE001
        return None


@contextlib.contextmanager
def hook_device(torch: Any, dev: Any):
    """在**调用用户钩子**期间把 torch 默认设备设成训练设备，退出时复原。

    ## 为什么必须有它（实测：反馈 §3.1 烧了十几次往返）

    用户代码里随手建的张量**落在 CPU**，而模型在显卡上 ⇒ 报出来的**不是**
    "设备不一致"，而是 `Sizes of tensors must match except in dimension 2`、
    `not enough values to unpack`、`mat1 and mat2 shapes cannot be multiplied`
    —— **全部伪装成维度错**，把人往完全错误的方向引（他的最小复现是一个
    `torch.tensor([0., 1.])` 常数张量）。

    ⇒ 与其要求每处都写 `device=…`（漏一处就中招），不如在**用户的钩子执行期间**
    把默认设备摆正：`torch.tensor([0., 1.])` 自动落到训练设备。
    - 作用域**只在用户代码那一次调用内**（`with` 包住），我们自己的代码不受影响；
    - 想要 CPU 张量照样可以：`torch.tensor(..., device="cpu")`。

    ⚠ `set_default_device` 是**进程级**设置 ⇒ 必须 try/finally 复原，
    否则会渗到我们自己的代码与下一个方法（那才是真的难查）。
    """
    setter = getattr(torch, "set_default_device", None)
    if setter is None or dev is None or str(dev) == "cpu":
        yield                                    # 老 torch 或 CPU：静默退化（不假装做到）
        return
    prev = None
    try:
        prev = torch.get_default_device()
    except Exception:                                      # noqa: BLE001
        prev = None
    ok = False
    try:
        setter(dev)
        ok = True
    except Exception:                                      # noqa: BLE001
        ok = False
    try:
        yield
    finally:
        if ok:
            try:
                setter(prev if prev is not None else "cpu")
            except Exception:                              # noqa: BLE001
                pass


@dataclass
class PrepResult:
    """数据准备结果（两个消费者共用：`TorchMLP` 与 `CustomNNMethod`）。"""

    Xs: np.ndarray
    #: 标准标签；**自定义任务（`kind="custom"`）时为 `None`**
    #: —— 那时标签住在逐行负载里，损失由网络自己算。
    yv: object
    tr_idx: np.ndarray
    va_idx: np.ndarray
    kind: str
    n_out: int
    mu: np.ndarray
    sd: np.ndarray
    label_encoder: object = None


def prepare_supervised(X: pd.DataFrame, y, *, kind: str, seed: int,
                       val_split: float) -> PrepResult:
    """标准化 + 标签编码 + 训练/验证切分（一次做全，避免两处漂移）。

    ``kind="custom"``（或 ``y is None``）：**不做标签处理** —— 特征照样标准化与切分，
    但"标签"由逐行负载携带、损失由网络自己算（见 `train` 的 `prep.yv is None` 分支）。
    ⚠ 这时 `n_out=0`：真实输出形状由网络自己决定（我们**不猜**）。
    """
    from sklearn.preprocessing import LabelEncoder

    Xv = X.to_numpy(float)
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-9
    Xs = (Xv - mu) / sd
    le = None
    if y is None or kind == "custom":
        yv = None
        n_out = 0
    elif kind == "classification":
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


def graph_ctx(problem, torch, *, masks=True, device=None) -> dict:
    """**图前向上下文的单一构造点**（`forward_graph(model, ctx)` 的那个 ctx）。

    ⚠ 为什么要收成一个函数：这个 ctx 里有 7 个键（x/edge_index/edge_weight/
    n_nodes/masks/kind/torch/nn），而调用点有三处（训练循环、推理、探针）
    —— 我一开始在三个地方各拼一遍，结果漏了 `x`，报出来的是
    `KeyError: 'x'`（离真正原因很远）。⇒ 一处构造、处处引用。
    """
    import numpy as np
    dev = resolve_device(device, torch) if device is not None else torch.device("cpu")
    x_all = torch.tensor(problem.X.to_numpy(float), dtype=torch.float32).to(dev)
    ei, ew = problem.graph()
    ctx = {"x": x_all,
           "edge_index": torch.tensor(ei, dtype=torch.long).to(dev),
           "edge_weight": torch.tensor(ew, dtype=torch.float32).to(dev),
           "n_nodes": int(len(x_all)),
           "kind": problem.target_kind, "torch": torch, "nn": torch.nn,
           "device": str(dev)}
    # ⭐ **多分量边特征**（NQS：耦合有 6 个物理分量，塞不进一个标量权重）
    ea = problem.edge_attr()
    if ea is not None:
        ctx["edge_attr"] = torch.tensor(ea, dtype=torch.float32).to(dev)
        ctx["edge_attr_names"] = problem.edge_attr_names()
    if masks:
        m_tr, m_te = problem.masks()
        ctx["train_mask"] = torch.tensor(m_tr, dtype=torch.bool).to(dev)
        ctx["test_mask"] = torch.tensor(m_te, dtype=torch.bool).to(dev)
    return ctx


def train_graph(recorder, *, model, problem, torch, nn,
                epochs: int, lr: float, loss_fn, optimizer=None,
                scheduler=None, grad_clip: float = 0.0,
                early_stop: bool = False, patience: int = 10,
                on_progress=None, forward_graph=None, device=None,
                metrics=None, keep_best: bool = False) -> dict:
    """**图任务的训练循环**：一张图、节点掩码划分、全量前向。

    与 `train`（表格版）的三处本质差别（都源于"行不独立"）：
    1. **不切行**：全图喂进模型 ⇒ 边索引始终有效、不需要节点重编号；
       训练/验证的区分靠 `problem.masks()` 的**布尔掩码**（只对指定节点算损失）；
    2. **不 shuffle**：整图前向，没有"批次"这回事；
    3. 损失按掩码取子集 ⇒ 验证分数**不泄漏**训练节点（这是图任务最容易搞错的地方）。

    `forward_graph(model, ctx)` 是 `Network` 的可选钩子（签名见
    `ml_toolbox/methods/neural/network.py`）；不写就用默认
    `model(x) → logits`。
    """
    import numpy as np

    dev = resolve_device(device, torch)
    model = model.to(dev)
    x_all = torch.tensor(problem.X.to_numpy(float), dtype=torch.float32).to(dev)
    y_np = np.asarray(problem.y)
    kind = problem.target_kind
    y_all = torch.tensor(y_np, dtype=torch.long if kind == "classification"
                         else torch.float32).to(dev)
    m_tr, m_te = problem.masks()
    tr_mask = torch.tensor(m_tr, dtype=torch.bool).to(dev)
    va_mask = torch.tensor(m_te, dtype=torch.bool).to(dev)
    # ctx 由**单一构造点**产出（别在这里内联拼：漏一个键就报 KeyError 'x'）
    ctx = graph_ctx(problem, torch, device=str(dev))

    def _forward():
        if forward_graph is not None:
            return forward_graph(model, ctx)
        return model(x_all)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    model.train()
    total, stopped_at, best, bad = int(epochs), None, float("inf"), 0
    cancelled, best_epoch, best_watch = False, None, float("inf")
    best_state = None

    def _loss_on(mask):
        logits = _forward()
        sel = logits[tr_mask] if mask is tr_mask else logits[va_mask]
        tgt = y_all[tr_mask] if mask is tr_mask else y_all[va_mask]
        if kind != "classification":
            sel = sel.squeeze(-1) if sel.dim() > 1 else sel
        return loss_fn(sel, tgt)

    try:
        for ep in range(total):
            optimizer.zero_grad()
            loss = _loss_on(tr_mask)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss = float(loss.detach())
            vl = None
            if bool(va_mask.any()):
                model.eval()
                with torch.no_grad():
                    vl = float(_loss_on(va_mask).detach())
                model.train()
            cur_lr = float(optimizer.param_groups[0]["lr"])
            extra = None
            if metrics is not None:
                try:
                    got = metrics(model, dict(ctx, step=int(ep), kind="metrics"))
                    if isinstance(got, dict) and got:
                        extra = got
                except Exception:                              # noqa: BLE001
                    metrics = None
            recorder.epoch(ep, train_loss, vl, cur_lr, extra=extra)
            if scheduler is not None:
                scheduler.step()
            if on_progress is not None:
                try:
                    if on_progress(ep, total, "epoch", train_loss, vl):
                        stopped_at, cancelled = ep, True
                        break
                except TypeError:
                    try:
                        if on_progress(ep, total, "epoch"):
                            stopped_at, cancelled = ep, True
                            break
                    except Exception:                          # noqa: BLE001
                        on_progress = None
                except Exception:                              # noqa: BLE001
                    on_progress = None
            watch = vl if vl is not None else train_loss
            if best_epoch is None or watch < best_watch - 1e-9:
                best_watch, best_epoch = watch, ep
                if keep_best:
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
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
    #: ⭐ **实测显存峰值**（事后归因）：只在真跑在 CUDA 上时才有值
    #: —— CPU 回 `None`，**不假装有数字**（用户实测 §3.3：他事前不知道会 OOM，
    #: 事后也没有任何地方告诉他"这次其实用了多少"）。
    peak_vram_mb = None
    try:
        if getattr(dev, "type", "") == "cuda":
            peak_vram_mb = round(
                float(torch.cuda.max_memory_allocated(dev)) / (1024.0 * 1024.0), 1)
    except Exception:                                          # noqa: BLE001
        peak_vram_mb = None
    return {"epochs_run": int(stopped_at + 1) if stopped_at is not None else total,
            "stopped_at": stopped_at, "best_val": best, "cancelled": cancelled,
            "best_epoch": best_epoch, "best_watch": best_watch,
            "device": str(dev),
            "peak_vram_mb": peak_vram_mb,
            # ⚠ **图任务不给 `chunks`**：分块的单位是"行"，而图任务的行**不是独立
            #   样本**（整图前向 + 节点掩码）⇒ 按行切会破坏拓扑。这里如实不给，
            #   也不编一个没人算过的数字。
            "last_state": {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()},
            "best_state": best_state if keep_best else None,
            "graph": problem.graph_facts()}


def predict_graph(model, problem, torch, *, forward_graph=None,
                  classes=None, device=None):
    """图任务的推理：全图前向，**只回测试节点的预测**（与 `y_true[te]` 对齐）。

    ⚠ 返回**定型的数组**，不是 `dtype=object`：object 数组存进 `artifacts.npz`
    会在读回时炸 `Object arrays cannot be loaded when allow_pickle=False`
    （实测踩过 —— 我第一版返回 object 数组，训练成功但存档读不回来）。

    @returns ``(y_pred, y_prob_or_None)``，两者都只含 **test 掩码**那些节点。
    """
    import numpy as np

    x_all = torch.tensor(problem.X.to_numpy(float), dtype=torch.float32)
    # 推理不需要掩码（只取 test 那些节点），但 ctx 仍从同一处来
    dev = resolve_device(device, torch) if device is not None else torch.device("cpu")
    model = model.to(dev)
    ctx = graph_ctx(problem, torch, masks=False, device=str(dev))
    model.eval()
    with torch.no_grad():
        logits = forward_graph(model, ctx) if forward_graph is not None \
            else model(x_all)
    te = np.asarray(problem.test_idx if problem.test_idx is not None
                    else np.arange(len(x_all)), dtype=int)
    n = len(x_all)
    if len(te) and (te.min() < 0 or te.max() >= n):
        raise ValueError("test_idx 越界：图任务的索引必须指向全量节点")
    k = problem.target_kind
    prob = None
    if k == "classification":
        prob_all = torch.softmax(logits, 1).cpu().numpy()
        pred_all = logits.argmax(1).cpu().numpy()
        if classes is not None:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.classes_ = np.asarray(classes)
            pred_all = le.inverse_transform(pred_all)
        # 定标：预测的 dtype 要跟标签一致（int64 / 字符串对象由 LabelEncoder 给出）
        pred = np.asarray(pred_all)[te]
        prob = np.asarray(prob_all, dtype=float)[te]
        return pred, prob
    pred = np.asarray(logits.squeeze(-1).cpu().numpy(), dtype=float)[te]
    return pred, None


def resolve_device(raw: Any, torch: Any) -> Any:
    """把 `device` 声明解析成 `torch.device`（**单一真源**，别处都用它）。

    规则（与 `mlp._device` 一致，且**可复现、不依赖硬件猜测**）：
    - `"cpu"` ⇒ CPU；`"cuda"` / `"cuda:N"` ⇒ 指定卡（无卡时**报错而不是静默回落**
      —— 静默回落等于"审计说 cuda、实跑 CPU"，那是假记录）；
    - `"auto"` / 空 ⇒ 有 CUDA 就用 `cuda:0`，否则 CPU。
    """
    s = str(raw or "auto").strip().lower()
    if s in ("", "auto"):
        try:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        except Exception:                                      # noqa: BLE001
            return torch.device("cpu")
    if s.startswith("cuda"):
        if not getattr(torch.cuda, "is_available", lambda: False)():
            raise RuntimeError(
                "要求了 CUDA 但本机 torch.cuda.is_available() 为假"
                "（没有卡 / 驱动不匹配）⇒ 拒绝静默回落 CPU："
                "那会留下'审计说 cuda、实跑 CPU'的假记录。"
                "要跑 CPU 请显式写 device='cpu'")
        return torch.device(s)
    return torch.device(s)


def train(recorder, *, model, prep: PrepResult, torch, nn,
          epochs: int, batch: int, lr: float, loss_fn,
          optimizer=None, scheduler=None, grad_clip: float = 0.0,
          early_stop: bool = False, patience: int = 10,
          on_progress=None, forward_step=None, device=None,
          payload=None, on_step=None, metrics=None, metrics_every: int = 1,
          keep_best: bool = False, chunk_budget_mb: float | None = None) -> dict:
    """跑训练循环，**托管全部观测**。

    @returns ``{"epochs_run": int, "stopped_at": int|None, "best_val": float,
               "best_epoch": int|None, "last_state": dict, "best_state": dict|None}``

    ⚠ 上报顺序是有讲究的（与 `mlp.py` 一致，别改）：
    ``recorder.epoch(...)`` 记的是**本轮实际用的 lr** ⇒ 必须在 `scheduler.step()`
    **之前**调用；反了会把"下一轮的 lr"记成"本轮的 lr"，差一轮。

    ## 这一版新增的五件事（NQS 需求）

    | 参数 | 解决什么 |
    |---|---|
    | `device` | **训练真的上卡**（此前这条路上硬编码 CPU ⇒ 46 万参数跑不动） |
    | `payload` | **逐行负载**：变长/复数数组按行切开，逐批放进 `ctx` 给 `forward_step` |
    | `on_step` | **步**的概念（预热"前 N 步关掉某块损失"） |
    | `metrics` | **方法自己上报指标**（分项损失/保真度/按通道分组）⇒ 写进 TB，AI 白拿 |
    | `keep_best` | **留住最佳轮权重**（末代没有特殊意义；此前只存末态） |
    """
    from torch.utils.data import DataLoader, TensorDataset

    dev = resolve_device(device, torch)
    tr, va = prep.tr_idx, prep.va_idx
    Xt = torch.tensor(prep.Xs[tr], dtype=torch.float32, device=dev)
    # ⭐ **自定义任务**（`prep.yv is None`）：没有标准标签 ⇒ 批次里不带 y，
    #   训练/验证的损失都由网络的 `forward_step` 给出（(out, target) 那个约定）。
    #   ⚠ 那时**必须**有 `forward_step`，否则我们没有任何可算的东西 ⇒ 早失败。
    has_y = prep.yv is not None
    if not has_y and forward_step is None:
        raise ValueError(
            "自定义任务（target_kind='custom'）没有标准标签 ⇒ 必须由网络提供 "
            "`forward_step(model, batch, ctx) -> (out, target)`；"
            "只写 `forward` 时我们没有可算的损失。")
    yt = None
    if has_y:
        yt = torch.tensor(prep.yv[tr],
                          dtype=torch.long if prep.kind == "classification"
                          else torch.float32, device=dev)
    idx_t = torch.tensor(np.asarray(tr, dtype=np.int64), dtype=torch.long)
    items = (Xt, yt, idx_t) if has_y else (Xt, idx_t)
    loader = DataLoader(TensorDataset(*items), batch_size=int(batch), shuffle=True)
    Xva = (torch.tensor(prep.Xs[va], dtype=torch.float32, device=dev)
           if len(va) else None)
    yva = (torch.tensor(prep.yv[va],
                        dtype=torch.long if prep.kind == "classification"
                        else torch.float32, device=dev)
           if (has_y and len(va)) else None)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    model = model.to(dev)

    #: 逐行负载的**张量版**：整份放在设备上，按批索引 ⇒ 每批只切一次（省拷贝）。
    pay_t = None
    if payload:
        pay_t = {str(k): _as_tensor(v, torch, dev) for k, v in payload.items()}

    #: ⭐ **整场训练复用的分块器**（`ctx["chunks"]`）：预算可来自
    #: `resource_guard.chunk_budget_mb`；复用是为了让"这次切了几块"留得下来。
    chunker = RowChunks(
        budget_mb=chunk_budget_mb,
        x_row_bytes=int(getattr(prep.Xs, "nbytes", 0) or 0)
        // max(1, int(len(prep.Xs))))

    def _ctx(row_idx, step, kind="train"):
        """给 `forward_step` / `metrics` 的上下文（**单一构造点**）。

        ⚠ 一处构造、处处引用：上一轮我在三个地方各拼一遍 ctx，漏了一个键，
        报出来是 `KeyError: 'x'`（离根因很远）。
        """
        out = {"torch": torch, "nn": nn, "device": str(dev), "step": int(step),
               "kind": kind, "n_rows": int(len(prep.Xs)),
               "train_idx": np.asarray(tr, dtype=np.int64),
               "val_idx": np.asarray(va, dtype=np.int64),
               # ⭐ **框架级分块**（2026-10-05，用户实测 §3.3）：用户不必再手写
               #   "按 R 切块 + 按块加权累加"。`reduce(..., mode="mean")` 里
               #   权重的算法由框架负责（见 `core.chunks` 的模块说明）。
               #   ⚠ **同一个 chunker 复用整场训练**（不是每次 `_ctx` 新建一个）：
               #   那样 `last_plan` 才留得住"这次到底切了几块"，回执才有话可说。
               "chunks": chunker}
        if pay_t is not None:
            if row_idx is None:
                out["payload"] = dict(pay_t)                    # 全量（metrics 用）
            else:
                out["payload"] = {k: v[row_idx.to(v.device)]
                                  for k, v in pay_t.items()}
        return out

    model.train()
    total = int(epochs)
    stopped_at, best, bad = None, float("inf"), 0
    cancelled = False
    best_epoch = None
    best_watch = float("inf")
    gstep = 0
    best_state = None
    last_extra: dict = {}
    #: 最近一次 `on_step` 上报的**步级**指标（进回执：让"步"这一级也有数字可看）
    last_step_metrics: dict = {}
    try:
        for ep in range(total):
            tot = torch.zeros((), dtype=torch.float32, device=dev)
            for batch_items in loader:
                if has_y:
                    xb, yb, ib = batch_items
                else:
                    xb, ib = batch_items
                    yb = None
                optimizer.zero_grad()
                ctx = _ctx(ib, gstep)
                if forward_step is not None:
                    # ⚠ 用户钩子期间默认设备 = 训练设备（否则随手建的张量落 CPU，
                    #   报出来的是"维度错"，见 `hook_device` 的说明）
                    with hook_device(torch, dev):
                        out, target = forward_step(model, (xb, yb), ctx)
                    loss = loss_fn(out, target)
                else:
                    out = model(xb)
                    loss = loss_fn(out.squeeze(-1)
                                   if prep.kind != "classification" else out, yb)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                tot = tot + loss.detach() * len(xb)
                # ⭐ **步钩子**：预热（"前 N 步先只训模长"）需要一个"步"的概念。
                #   ⭐ 从本轮起它还**能上报步级指标**：返回 dict 就写进 TB ⇒
                #   `describe_training` 看得到"步"这一级的曲线。
                #   为什么需要（实测）：NLL 在前 ~100 步内就崩到 0，而 epoch 级观测
                #   **看不出它什么时候崩的** —— 最好的诊断信息死在"看不到"上。
                if on_step is not None:
                    try:
                        with hook_device(torch, dev):
                            said = on_step(model, gstep, ctx)
                        if isinstance(said, dict) and said:
                            if recorder.tb is not None:
                                recorder.tb.log_step(said, gstep)
                            last_step_metrics.clear()
                            last_step_metrics.update({
                                str(k): float(v) for k, v in said.items()
                                if isinstance(v, (int, float))
                                and not isinstance(v, bool)})
                    except Exception:                          # noqa: BLE001
                        on_step = None        # 观测/调度失败不拖垮训练（与进度钩子同纪律）
                gstep += 1
            vl = None
            if Xva is not None:
                model.eval()
                with torch.no_grad():
                    if forward_step is not None:
                        # ⭐ **有这个钩子就全程走它**（训练/验证/探针同一条入口）：
                        #   反馈 §2-缺口3 的教训是"别停在中间"—— 验证走 `model(Xva)`
                        #   会逼作者写一个**与损失无关的假 forward**（就为了让验证别炸）。
                        vi = torch.tensor(np.asarray(va, dtype=np.int64),
                                          dtype=torch.long)
                        with hook_device(torch, dev):
                            vo, vt = forward_step(model, (Xva, yva if has_y else None),
                                                  _ctx(vi, ep, kind="val"))
                        vl = float(loss_fn(vo, vt))
                    elif has_y:
                        o = model(Xva)
                        vl = float(loss_fn(o.squeeze(-1)
                                           if prep.kind != "classification" else o,
                                           yva))
                model.train()
            train_loss = float(tot) / max(len(tr), 1)
            cur_lr = float(optimizer.param_groups[0]["lr"])
            # ⭐ **方法自己上报的指标**：写进 TB ⇒ AI 通过 `describe_training` 直接读到。
            #
            # ⚠ 三条实测出来的纪律（反馈 §2-缺口5）：
            #   1. **必须在 `model.eval()` + `no_grad()` 下调用**：观测没有梯度语义，
            #      否则作者算一次保真度就建一整张图（吃显存）；我为此得自己写 no_grad。
            #   2. **它在每轮之后被调用一次、拿全量负载** ⇒ 别在里面做全量重活。
            #      贵的指标用 `Network.metrics_every = N` 降频（末轮一定算，保证终值不假）。
            #   3. 上报失败只关掉它自己，**不拖垮训练**（观测是附加的）。
            extra = None
            if metrics is not None and (
                    int(ep) % max(1, metrics_every) == 0 or ep == total - 1):
                try:
                    was_training = bool(model.training)
                    model.eval()
                    with torch.no_grad(), hook_device(torch, dev):
                        got = metrics(model, _ctx(None, ep, kind="metrics"))
                    if was_training:
                        model.train()
                    if isinstance(got, dict) and got:
                        extra = got
                        last_extra = dict(got)
                except Exception:                              # noqa: BLE001
                    metrics = None            # 上报失败不拖垮训练（观测是附加的）
                    try:
                        model.train()
                    except Exception:                          # noqa: BLE001
                        pass
            recorder.epoch(ep, train_loss, vl, cur_lr, extra=extra)
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
            #   ⇒ 把"从哪一轮开始就没再好过"变成可查的事实。
            watch = vl if vl is not None else train_loss
            if best_epoch is None or watch < best_watch - 1e-9:
                best_watch, best_epoch = watch, ep
                if keep_best:
                    # 快照到 **CPU**：卡上显存要留给训练，且存档不该占显存。
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
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
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    #: ⭐ **实测显存峰值**（事后归因，用户实测 §3.3）：只在真跑在 CUDA 上时才有值
    #: ⇒ CPU 回 `None`，**不假装有数字**。
    peak_vram_mb = None
    try:
        if getattr(dev, "type", "") == "cuda":
            peak_vram_mb = round(
                float(torch.cuda.max_memory_allocated(dev)) / (1024.0 * 1024.0), 1)
    except Exception:                                          # noqa: BLE001
        peak_vram_mb = None
    return {"epochs_run": int(stopped_at + 1) if stopped_at is not None else total,
            "stopped_at": stopped_at, "best_val": best, "cancelled": cancelled,
            "best_epoch": best_epoch, "best_watch": best_watch,
            "device": str(dev), "steps": int(gstep),
            "peak_vram_mb": peak_vram_mb,
            # ⭐ **框架分块的实际方案**（网络用了 `ctx["chunks"]` 才有）：回执要能说
            #   "已分 N 块" —— 否则"分块了没有"只能靠猜（用户实测要的就是这句话）。
            "chunks": chunker.facts(),
            # 最后一轮上报的自定义指标：让它**同时进回执**（TB 那条路给"看曲线"，
            # 这条给 `describe_run`/`compare_methods` 的"看数字"）
            "last_extra": dict(last_extra),
            # 步级指标（`on_step` 的返回）：TB 那条给**曲线**，这条给**数字**，
            # 两者同一批 tag ⇒ 回执与曲线不会各说各话。
            "last_step_metrics": dict(last_step_metrics),
            "last_state": last_state, "best_state": best_state}


def _as_tensor(arr, torch, dev):
    """numpy 数组 → 设备上的张量（**复数/整数都保住 dtype**）。

    ⚠ 这里**不做** `float32` 强转：负载里可能有 `complex64`（波函数振幅）、
    `int8`（位串 0/1）、`bool`（mask）⇒ 一律强转就把物理量毁了。
    """
    import numpy as np
    a = np.asarray(arr)
    if np.iscomplexobj(a):
        t = torch.tensor(a.astype(np.complex64))
    elif a.dtype == bool:
        t = torch.tensor(a)
    elif np.issubdtype(a.dtype, np.integer):
        t = torch.tensor(a.astype(np.int64))
    else:
        t = torch.tensor(a.astype(np.float32))
    return t.to(dev)
