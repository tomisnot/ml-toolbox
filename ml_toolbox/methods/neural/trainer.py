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
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ...core.chunks import RowChunks

#: 中途存档的**默认关闭**（0 = 只在终态写一次，与从前逐位一致）。
#: ⚠ 为什么不默认开：本仓纪律是"默认不改行为"（P3 零开销）。要让长训练默认有存档，
#: 应该由**调用方**（`run_custom` 的参数/回执提示）显式要求，而不是在这里偷偷改语义。
DEFAULT_CHECKPOINT_EVERY = 0

#: `metrics` 钩子**连续失败多少次**后开始**降级重试**（不再每轮都试）。
#: ⚠ 为什么不是"第一次失败就永久关掉"（用户实测的**静默黑洞**，2026-10-07）：
#: 他的 `metrics` 在**全量负载**上一次 256-token 解码就 **OOM**，而旧实现
#: `except Exception: metrics = None` ⇒ **第一次失败后整场再也不调用** ⇒
#: **TB 里一个 epoch 级 tag 都没有**，而训练照常、loss 照降
#: ⇒ 表面看像"作者没写 metrics"，其实他写了。三种病（**没写 / 每轮失败 / 名字写错**）
#: 修法完全不同 ⇒ 必须能区分，而且**不能永久关闭**。
METRICS_FAIL_TOLERANCE = 3

#: 降级之后**每多少轮再试一次**（外加**末轮一定试**）——
#: "临时性失败"（显存一时紧张）还有机会自己恢复，而代价有界。
METRICS_RETRY_EVERY = 10


def _metrics_due(ep: int, total: int, health: dict, every: int) -> bool:
    """这一轮该不该调 `metrics`（失败后**降级**而不是永久关闭）。

    - 正常：按 `metrics_every` 的节奏 + **末轮一定算**（原有语义一字不改）；
    - 连续失败超过 `METRICS_FAIL_TOLERANCE` 之后：**每 `METRICS_RETRY_EVERY` 轮再试一次
      + 末轮必试** ⇒ 既不放弃（不会"整场一条都没有"），也不再每轮付失败代价
      （他的失败是 OOM，每轮重试很贵）。
    """
    due = (int(ep) % max(1, int(every)) == 0) or int(ep) == int(total) - 1
    if not due:
        return False
    if not health.get("throttled"):
        return True
    return int(ep) == int(total) - 1 or int(ep) % METRICS_RETRY_EVERY == 0


def _metrics_failed(health: dict, ep: int, exc: BaseException) -> None:
    """记一次 `metrics` 失败（**留痕**：轮次 + 类型 + 消息；最多留 3 条）。"""
    health["failures"] = int(health.get("failures", 0)) + 1
    health["consecutive"] = int(health.get("consecutive", 0)) + 1
    errs = health.setdefault("errors", [])
    if len(errs) < 3:
        errs.append(f"epoch {int(ep)}：{type(exc).__name__}: {str(exc)[:200]}")
    if int(health["consecutive"]) >= METRICS_FAIL_TOLERANCE \
            and not health.get("throttled"):
        health["throttled"] = True
        health["throttled_from_epoch"] = int(ep)
        health["note"] = (
            f"`metrics` 连续失败 {int(health['consecutive'])} 次（见 `errors`）⇒ "
            f"**降级**为每 {METRICS_RETRY_EVERY} 轮再试一次 + 末轮必试"
            "（**没有**永久关闭）。⚠ 最常见的原因是**在 `metrics` 里对全量负载做重活**"
            "——它每轮拿到的是**全量负载**（实测：7344 行、一次 256-token 解码就 OOM）"
            "⇒ 请在 `metrics` 内部**分批**（`ctx[\"chunks\"]` 或自己切片）")


def _metrics_ok(health: dict) -> None:
    """一次成功：清掉连续失败计数（**降级是可恢复的**）。"""
    health["consecutive"] = 0
    if health.get("throttled"):
        health["recovered_after_epoch"] = health.get("throttled_from_epoch")


def _metrics_health(hook, /, **extra) -> dict:
    """`metrics` 钩子的健康事实（进回执/诊断/checkpoint —— **不静默**）。"""
    out = {"hook": callable(hook), "calls": 0, "failures": 0, "consecutive": 0,
           "errors": [], "throttled": False}
    out.update(extra)
    return out


def _group_lrs(optimizer) -> dict:
    """优化器**每个参数组**的当前 lr（`{组名: lr}`）—— Q7b。

    ## 为什么必须有（使用者的实测需求）

    他的 NQS 修复要用**不对称步长**（幅度组/相位组不同 lr，文献里的
    `m = δτ_φ/δτ_A ≥ 1`），而观测面原来**只有一个标量 `lr`**
    ⇒ **改了却无法确认它真的生效**（"改了个东西但无法确认它生效"是他这一路最大的坑之一）。

    ## 组名从哪来（约定）

    `Network.configure` 建优化器时可以给每个组起名：
    `torch.optim.Adam([{"params": amp_params, "lr": 1e-3, "name": "amp"},
                       {"params": ph_params,  "lr": 3e-3, "name": "phase"}])`
    ⇒ 没起名的组按 `g0`/`g1`…（**如实编号，不假装有名字**）。

    ⚠ PyTorch 允许 `param_groups` 里带任意额外键（`name` 就是其中一个）⇒
    这条**不需要**框架改契约，只是把它**读出来**。
    """
    out: dict = {}
    try:
        for i, g in enumerate(getattr(optimizer, "param_groups", []) or []):
            name = str(g.get("name") or f"g{i}")
            try:
                out[name] = float(g.get("lr"))
            except (TypeError, ValueError):
                out[name] = None
    except Exception:                                          # noqa: BLE001
        return {}
    return out


def _try_checkpoint(on_checkpoint, model, *, step: int, epoch: int,
                    facts: dict, best_state=None, best_epoch=None,
                    best_value=None, best_step=None) -> None:
    """调一次**中途存档**；失败**不拖垮训练**，但必须**留痕**（可教学）。

    ## 为什么不能抛

    与 TensorBoard 那条路同一纪律（`tb_log` 的三处 fail-safe）：观测/存档是**附加的**，
    主结果（训练）优先。磁盘满/权限/被杀都可能在 5 小时训练里出现一次，
    为了它整场训练报错是**错的**处置。

    ## 但为什么也不能静默

    反过来，"我给了 `checkpoint_every` 却没有存档"若不留痕，就是本仓反复在防的
    假绿（使用者会以为"有存档"，直到死机那一刻才发现没有）。⇒ 收集进 `facts["errors"]`，
    由适配器写进 artifacts / diag / 回执。

    ## `best_state` / `best_epoch`（Q4b）

    周期回调**不只看当前权重**：还要把"**当前的最佳快照**"一并交给它，
    这样 `best.pt` 也能在**中途**落盘（否则一次被中断的长训练只有 `last.pt`，
    拿不到"按我的指标最好的那一份"）。⚠ 给的是**快照引用**（不是 model）——
    中途那一刻的 model 权重早已不是 best 那一轮的了。
    """
    if on_checkpoint is None:
        return
    kw = {"model": model, "step": int(step), "epoch": int(epoch)}
    if best_state is not None or best_epoch is not None:
        kw["best_state"] = best_state
        kw["best_epoch"] = best_epoch
        kw["best_value"] = best_value
        kw["best_step"] = best_step
    try:
        said = on_checkpoint(**kw)
    except TypeError:
        #: 第三方的 `on_checkpoint` 可能只认三个参数 ⇒ **退回老签名**（不因新能力打红）
        try:
            said = on_checkpoint(model=model, step=int(step), epoch=int(epoch))
        except Exception as exc:                               # noqa: BLE001
            facts.setdefault("errors", []).append(
                f"中途存档失败（step={int(step)}）：{type(exc).__name__}: {exc}")
            return
    except Exception as exc:                                   # noqa: BLE001
        facts.setdefault("errors", []).append(
            f"中途存档失败（step={int(step)}）：{type(exc).__name__}: {exc}")
        return
    facts["n_ok"] = int(facts.get("n_ok", 0)) + 1
    facts["last_step"] = int(step)
    facts["last_epoch"] = int(epoch)
    if isinstance(said, Mapping):
        facts["last"] = dict(said)


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
                metrics=None, keep_best: bool = False,
                checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
                on_checkpoint=None,
                best_metric: str = "", best_direction: str = "") -> dict:
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
    #: 中途存档的事实（与表格路径同形；见 `train` 里同名变量）。
    ck_facts: dict = {"errors": [], "n_ok": 0}
    #: `metrics` 健康（2026-10-07：失败**留痕 + 降级**，不永久关闭）
    m_health: dict = _metrics_health(metrics)
    #: `best.pt` 的两套口径（Q4；与 `train` 同形）
    vl_best_epoch, vl_best_watch = None, float("inf")
    metric_best_epoch, metric_best_value, metric_best_step = None, None, None
    metric_skipped = 0
    fallback_state = None
    keep_fallback = keep_best

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
            # ⭐ **中途存档（图任务）**：图任务**没有 batch**（整图前向，一步 = 一轮）
            #   ⇒ `checkpoint_every` 在这里的单位是 **epoch**（与表格路径的"优化步"
            #   不同，如实写在参数说明里，不让使用者自己猜）。
            #   ⚠ 调用点**只有一个**：在下面"best 刷新之后"（见那里的说明）——
            #     在这里再放一个会让每个周期**写两次**（且第二次的 best 才是新的）。
            train_loss = float(loss.detach())
            vl = None
            if bool(va_mask.any()):
                model.eval()
                with torch.no_grad():
                    vl = float(_loss_on(va_mask).detach())
                model.train()
            cur_lr = float(optimizer.param_groups[0]["lr"])
            extra = None
            #: ⚠ 与表格路径同一条纪律（2026-10-07）：**失败留痕 + 降级，不永久关闭**。
            #:   图任务里 metrics 也是每轮一次（没有 `metrics_every` 的降频语义）。
            if metrics is not None and _metrics_due(ep, total, m_health, 1):
                m_health["calls"] = int(m_health.get("calls", 0)) + 1
                try:
                    got = metrics(model, dict(ctx, step=int(ep), kind="metrics"))
                    _metrics_ok(m_health)
                    if isinstance(got, dict) and got:
                        extra = got
                except Exception as exc:                       # noqa: BLE001
                    _metrics_failed(m_health, ep, exc)
            recorder.epoch(ep, train_loss, vl, cur_lr, extra=extra,
                           lr_by_group=_group_lrs(optimizer))
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
            #: ① val_loss 口径**永远算**（未声明 `best_metric` 时的行为，也是对照）
            if vl_best_epoch is None or watch < vl_best_watch - 1e-9:
                vl_best_watch, vl_best_epoch = watch, ep
                if keep_fallback:
                    fallback_state = {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()}
            #: ② 声明了 `best_metric` 就按那个 tag 选（Q4；图任务同样支持 ——
            #:   **不许**"声明了却悄悄走 val_loss"）。
            #:   图任务**没有 batch** ⇒ `step` 就是轮次（如实这么记）。
            if best_metric:
                cur = None
                if isinstance(extra, dict) and best_metric in (extra or {}):
                    try:
                        cur = float(extra[best_metric])
                    except (TypeError, ValueError):
                        cur = None
                if cur is None:
                    metric_skipped += 1
                else:
                    better = (metric_best_epoch is None
                              or (cur > metric_best_value + 1e-12
                                  if best_direction == "higher"
                                  else cur < metric_best_value - 1e-12))
                    if better:
                        metric_best_value, metric_best_epoch = cur, ep
                        metric_best_step = int(ep + 1)
                        best_watch, best_epoch = cur, ep
                        if keep_best:
                            best_state = {k: v.detach().cpu().clone()
                                          for k, v in model.state_dict().items()}
                        keep_fallback = False
                        fallback_state = None
            elif best_epoch is None or watch < best_watch - 1e-9:
                best_watch, best_epoch = watch, ep
                if keep_best:
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
            # ⭐ 中途存档：放在 **best 刷新之后** —— 这样 `best.pt` 落的是**最新**那一份
            #   （Q4b：被中断的长训练也要能拿到"按我的指标最好的那一份"）。
            if (on_checkpoint is not None and int(checkpoint_every or 0) > 0
                    and (ep + 1) % max(1, int(checkpoint_every)) == 0):
                _try_checkpoint(on_checkpoint, model, step=ep + 1, epoch=ep,
                                facts=ck_facts, best_state=best_state,
                                best_epoch=best_epoch, best_value=best_watch,
                                best_step=(metric_best_step if best_metric
                                           else None))
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
    #: `best.pt` 的选择口径（Q4）——与 `train` 同形（图任务里 step == 轮次）。
    if best_metric and metric_best_epoch is None:
        best_epoch, best_watch = vl_best_epoch, vl_best_watch
        if keep_best and best_state is None:
            best_state = fallback_state
        best_by = {"tag": "val_loss", "direction": "lower",
                   "value": (None if vl_best_watch == float("inf") else vl_best_watch),
                   "epoch": vl_best_epoch, "step": None,
                   "requested_tag": str(best_metric),
                   "note": (f"声明的 `best_metric={best_metric!r}` 在整场训练里"
                            f"**一次都没上报**（跳过了 {metric_skipped} 轮）"
                            f"⇒ **退回 val_loss** 选 best。检查 tag 名拼写")}
    elif best_metric:
        best_by = {"tag": str(best_metric), "direction": str(best_direction or "lower"),
                   "value": metric_best_value, "epoch": metric_best_epoch,
                   "step": metric_best_step, "skipped_epochs": int(metric_skipped),
                   "note": (f"按 `best_metric={best_metric!r}`（{best_direction}）选；"
                            f"⚠ val_loss 最佳在第 {vl_best_epoch} 轮"
                            f"（{vl_best_watch:.6g}）—— 两者常常不是同一轮")}
    else:
        best_by = {"tag": "val_loss", "direction": "lower",
                   "value": (None if best_watch == float("inf") else best_watch),
                   "epoch": best_epoch, "step": None,
                   "note": "未声明 `best_metric` ⇒ 仍按 **val_loss** 选（与从前一致）"}
    val_loss_best = {"epoch": vl_best_epoch,
                     "value": (None if vl_best_watch == float("inf")
                               else vl_best_watch)}
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
            # ⭐ 中途存档的事实 + `best.pt` 的选择口径（与表格路径同形）
            "checkpoint": dict(ck_facts),
            "best_by": dict(best_by),
            "val_loss_best": dict(val_loss_best),
            "best_metric_skipped_n": int(metric_skipped),
            #: ⭐ `metrics` 健康（图任务同样报：**不静默**）
            "metrics_health": dict(m_health),
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
          keep_best: bool = False, chunk_budget_mb: float | None = None,
          checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
          on_checkpoint=None,
          best_metric: str = "", best_direction: str = "") -> dict:
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
               #: ⭐ **每个参数组的 lr**（Q7b）：`on_step`/`metrics` 里能直接读到它
               #:   ⇒ 作者可以在自己的 tag 里**分组上报**（`lr/amp`、`lr/phase`），
               #:   而不必靠"我记得 configure 里设了多少"。
               "lr_by_group": _group_lrs(optimizer),
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
    #: ⭐ **`metrics` 钩子的健康事实**（2026-10-07）：它为什么没产出 tag，
    #: 必须能区分"没写钩子 / 每轮失败 / 名字写错"——三种病修法完全不同。
    m_health: dict = _metrics_health(metrics)
    #: 中途存档的事实（`_try_checkpoint` 往里写）：成功了 N 次 / 失败原因 / 最后一次的步号。
    #: ⚠ **即使 `checkpoint_every=0` 也要在**（回执形状一致 ⇒ 消费者不用判键在不在）。
    ck_facts: dict = {"errors": [], "n_ok": 0}
    #: ⭐ **`best.pt` 的选择口径**（Q4）：val_loss 口径永远算（`vl_*`），
    #: 声明了 `best_metric` 时另按那个 tag 选（`metric_*`）。
    #: `fallback_state` 只在"声明的指标一次都还没上报"期间维护 —— 一旦指标可用就释放，
    #: 这样声明路径的额外内存峰值是**一份快照**（而不是两份）。
    vl_best_epoch, vl_best_watch = None, float("inf")
    metric_best_epoch, metric_best_value, metric_best_step = None, None, None
    metric_skipped = 0
    fallback_state = None
    keep_fallback = keep_best
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
                # ⭐⭐ **中途存档**（2026-10-07，用户实测的 A1："5 小时训练中途死机＝全丢"）。
                #   ⚠ 语义：`checkpoint_every` 数的是**优化步**（= 一个 batch 一次
                #   `optimizer.step()`），**不是 epoch** —— 用户的原话是"每 N 步落一份"，
                #   而长训练里 epoch 太粗（一轮可能几十分钟）。
                #   ⚠ 落盘的是**当前 `last`**；`best` 仍由 `keep_best` 在验证改善时更新
                #   （语义一行没动）。
                #   ⚠ 放在 `on_step` **之后**：这样"存档里记的步号"与"on_step 看到的
                #   步号"是同一个数（同一批的指标与权重**同源**，不是差一步）。
                if (on_checkpoint is not None and int(checkpoint_every or 0) > 0
                        and gstep % max(1, int(checkpoint_every)) == 0):
                    _try_checkpoint(on_checkpoint, model, step=gstep, epoch=ep,
                                    facts=ck_facts, best_state=best_state,
                                    best_epoch=best_epoch, best_value=best_watch,
                                    best_step=metric_best_step)
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
            #   3. ⚠⚠ **失败只降级、不永久关闭，而且必须留痕**（2026-10-07 用户实测的静默黑洞）：
            #      他的 `metrics` 在全量负载上 OOM ⇒ 旧实现第一次失败后 `metrics = None`
            #      ⇒ 整场再没调用过 ⇒ **TB 里一个 epoch 级 tag 都没有**，而训练完全正常
            #      ⇒ 看起来像"作者没写 metrics"。现在：留痕（轮次/类型/消息）+
            #      连续失败 N 次后**降级重试**（每 10 轮 + 末轮），**永不永久关闭**。
            extra = None
            if metrics is not None and _metrics_due(ep, total, m_health, metrics_every):
                m_health["calls"] = int(m_health.get("calls", 0)) + 1
                try:
                    was_training = bool(model.training)
                    model.eval()
                    with torch.no_grad(), hook_device(torch, dev):
                        got = metrics(model, _ctx(None, ep, kind="metrics"))
                    if was_training:
                        model.train()
                    _metrics_ok(m_health)
                    if isinstance(got, dict) and got:
                        extra = got
                        last_extra = dict(got)
                except Exception as exc:                       # noqa: BLE001
                    #: ⚠ **留痕 + 降级**（不再 `metrics = None`）：见上面的第 3 条纪律
                    _metrics_failed(m_health, ep, exc)
                    try:
                        model.train()
                    except Exception:                          # noqa: BLE001
                        pass
            recorder.epoch(ep, train_loss, vl, cur_lr, extra=extra,
                           lr_by_group=_group_lrs(optimizer))
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
            #: ① **val_loss 口径**永远算（它是"未声明 `best_metric`"时的行为，
            #:   也是声明后要如实报出的对照：两者常常**不是同一轮**）。
            if vl_best_epoch is None or watch < vl_best_watch - 1e-9:
                vl_best_watch, vl_best_epoch = watch, ep
                if keep_fallback:
                    fallback_state = {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()}
            #: ② ⭐ **决定 `best.pt` 的口径**（Q4）：声明了就按声明的 tag 选
            #:   （`best_metric` 在为空的默认路径上逐位等价于旧行为）。
            if best_metric:
                cur = None
                if isinstance(extra, dict) and best_metric in (extra or {}):
                    try:
                        cur = float(extra[best_metric])
                    except (TypeError, ValueError):
                        cur = None
                if cur is None:
                    metric_skipped += 1
                else:
                    better = (metric_best_epoch is None
                              or (cur > metric_best_value + 1e-12
                                  if best_direction == "higher"
                                  else cur < metric_best_value - 1e-12))
                    if better:
                        metric_best_value, metric_best_epoch = cur, ep
                        metric_best_step = int(gstep)
                        best_watch, best_epoch = cur, ep
                        if keep_best:
                            best_state = {k: v.detach().cpu().clone()
                                          for k, v in model.state_dict().items()}
                        #: 指标已经能用了 ⇒ 不再维护 val_loss 兜底快照（省一份内存）
                        if metric_best_epoch is not None:
                            keep_fallback = False
                            fallback_state = None
            elif best_epoch is None or watch < best_watch - 1e-9:
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
    #: ⭐ **`best.pt` 到底按什么选的**（Q4）：说清 tag / 方向 / 值 / 轮次 / 步号。
    #: ⚠ 两个口径**都报**：声明了 `best_metric` 时 `best_by` 是那个 tag，
    #: `val_loss_best` 是**同一场训练的 val_loss 口径对照**（实测两者常常不是同一轮）。
    if best_metric and metric_best_epoch is None:
        #: 声明的指标**一次都没上报**（名字写错 / `metrics_every` 下从不出现）⇒
        #: **退回 val_loss 并如实说**（不静默、也不让 best.pt 缺席）。
        best_epoch, best_watch = vl_best_epoch, vl_best_watch
        if keep_best and best_state is None:
            best_state = fallback_state
        best_by = {"tag": "val_loss", "direction": "lower",
                   "value": (None if vl_best_watch == float("inf") else vl_best_watch),
                   "epoch": vl_best_epoch, "step": None,
                   "requested_tag": str(best_metric),
                   "note": (f"声明的 `best_metric={best_metric!r}` 在整场训练里"
                            f"**一次都没上报**（跳过了 {metric_skipped} 轮）"
                            f"⇒ **退回 val_loss** 选 best。检查 tag 名拼写，"
                            f"或让 `metrics` 一定返回它")}
    elif best_metric:
        best_by = {"tag": str(best_metric),
                   "direction": str(best_direction or "lower"),
                   "value": metric_best_value, "epoch": metric_best_epoch,
                   "step": metric_best_step,
                   "skipped_epochs": int(metric_skipped),
                   "note": (f"按 `best_metric={best_metric!r}`"
                            f"（{best_direction}）选；"
                            f"⚠ 与 val_loss 的最佳轮**可能不是同一轮**："
                            f"val_loss 最佳在第 {vl_best_epoch} 轮"
                            f"（{vl_best_watch:.6g}）")}
    else:
        best_by = {"tag": "val_loss", "direction": "lower",
                   "value": (None if best_watch == float("inf") else best_watch),
                   "epoch": best_epoch, "step": None,
                   "note": "未声明 `best_metric` ⇒ 仍按 **val_loss** 选（与从前一致）"}
    val_loss_best = {"epoch": vl_best_epoch,
                     "value": (None if vl_best_watch == float("inf")
                               else vl_best_watch)}
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
            # ⭐ **中途存档的事实**（`checkpoint_every>0` 时才有 `n_ok>0`）：
            #   `n_ok` 是"真的成功落盘了几次"、`last_step` 是最后那次的步号、
            #   `errors` 是**失败原因**（可教学：不静默，也不拖垮训练）。
            "checkpoint": dict(ck_facts),
            # ⭐ **best.pt 的选择口径**（Q4）：`best_by`（决定 best.pt 的那个）+ 
            #   `val_loss_best`（对照）。未声明时 `best_by.tag == "val_loss"`、
            #   值与从前的 `best_watch`/`best_epoch` 一致 ⇒ 逐位兼容。
            "best_by": dict(best_by),
            "val_loss_best": dict(val_loss_best),
            "best_metric_skipped_n": int(metric_skipped),
            # ⭐ **`metrics` 健康**（**不静默**）：作者没写钩子 / 每轮失败 / 名字写错
            #   是三种病 ⇒ `hook`/`failures`/`errors` 让它们分得开。
            "metrics_health": dict(m_health),
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
