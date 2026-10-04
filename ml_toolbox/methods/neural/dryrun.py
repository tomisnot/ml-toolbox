# -*- coding: utf-8 -*-
"""**试跑（dry run）**：把"形状类错误"从"一次一个往返"压成"一次全看到"。

## 为什么需要它（实测痛点，来自 `docs/建议-AI往返成本与训练观测面.md`）

反馈者训一个 NQS 网络时，**形状类错误犯了约 15 次，每次花掉一整个工具往返**
（提交 → 等 → 读错 → 改 → 再提交）。他的原话：

> 本地跑一次 python 是秒级，所以我**被迫**采用"本地磨对形状、再提交一次"的策略。

其中最贵的一类是**本地测不出来的**：他本地 B=3 全绿，而真实负载的 batch 是 2
⇒ 一提交就错。**只有"拿真实负载做一次前向"才能发现那一类**——那正是本模块的定义。

## 它做什么（只做三件）

1. **编译 + 建模型**（`build` 拿得到负载形状 ⇒ 不靠猜维度）；
2. 用**真实负载**跑**若干个批大小**的一次前向（**不反传、不落盘、不计入实验记录**）；
3. 回**逐模块的输入/输出形状轨迹** + **首个异常的结构化归因**
   （尽量指出"最后一个执行到的模块收到了什么形状"）。

## 两条纪律

- **不静默**：跑不成要说清"在哪一步、为什么"，而不是回一个 `ok=false` 了事；
- **不走样**：复用与真跑**同一套** `prepare_supervised` / `_build_ctx` /
  `forward_step` 调用约定 ⇒ 试跑通过**真的意味着**真跑的形状对得上
  （否则试跑只是另一种"看着对"）。
"""
from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

#: 默认试的批大小。为什么是这几个：
#: - **1**：单样本（最能暴露"沿 batch 维做错"的实现）；
#: - **2**：反馈者真实踩到的那一档（本地 B=3、真实 B=2 ⇒ 只有真数据能抓）；
#: 再加上**训练批大小**（真跑会用的那个），去重后跑，样本不够就退到"有多少用多少"。
DEFAULT_BATCH_PROBES = (1, 2)


def _shape_of(obj) -> Any:
    """把一个张量/元组/列表的形状写成**可 JSON 序列化**的形式。"""
    if hasattr(obj, "shape"):
        return list(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [_shape_of(x) for x in obj]
    return str(type(obj).__name__)


@contextlib.contextmanager
def trace_modules(model):
    """记录**每个被执行到的子模块**的输入/输出形状（按执行顺序）。

    ## 为什么要 **pre-hook**（不只是 forward-hook）

    forward-hook 只在模块**跑完之后**触发 ⇒ **炸在模块内部时它永远不触发**，
    轨迹就是空的（实测：错在第一个 `Linear` 上时，"形状轨迹"回了个空数组 ——
    而这恰恰是最需要它的场合）。

    ⇒ 用 `forward_pre_hook` 在**进模块之前**先记一条，`forward_hook` 再补上输出。
    于是每个条目有三种状态，而它们正好构成诊断：
    - `out` 有值：它正常跑完了；
    - `out is None`：**它进去了、没出来** ⇒ 失败就在它里面（报告要指这个）。

    ⚠ 用一个**栈**：嵌套调用时 pre/post 是配对的，post 时栈顶才是自己那一条。
    """
    trace: list[dict] = []
    stack: list[dict] = []

    def _pre(name: str):
        def _hook(_mod, inputs):
            entry = {"module": str(name), "in": _shape_of(inputs), "out": None}
            trace.append(entry)
            stack.append(entry)
        return _hook

    def _post(_name: str):
        def _hook(_mod, _inputs, output):
            if stack:
                stack.pop()["out"] = [_shape_of(output)]
        return _hook

    handles = []
    for name, mod in model.named_modules():
        if name:                       # 跳过根模块：它会把整份输入再记一遍，没信息量
            handles.append(mod.register_forward_pre_hook(_pre(name)))
            handles.append(mod.register_forward_hook(_post(name)))
    try:
        yield trace
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:                                  # noqa: BLE001
                pass


def classify_shape_error(exc: BaseException, trace: list[dict]) -> dict:
    """把异常归成**可教学**的形态，并尽量指出失败位置。

    为什么值得单列：这些报错**互相伪装** —— 设备不一致会说成"维度不匹配"，
    gather 的秩错会说成"维度数不同"，reshape 会说成"size 不对"
    ⇒ 不做归因的话，使用者只能一个个试（反馈者正是这么烧掉 15 个往返的）。
    """
    msg = f"{type(exc).__name__}: {exc}"
    low = msg.lower()
    where = ""
    if trace:
        # ⭐ 优先指**进去了没出来的那个**（失败就在它里面）；
        #   它们都在 pre-hook 里留下了"收到什么"，所以哪怕炸在第一层也有话说。
        stuck = [e for e in trace if e.get("out") is None]
        if stuck:
            where = (f"失败在 `{stuck[-1]['module']}`（进去了、没出来）："
                     f"它收到 {stuck[-1]['in']}")
        else:
            last = trace[-1]
            where = (f"最后一个执行完的模块是 `{last['module']}`："
                     f"收到 {last['in']}、给出 {last['out']}")
    kind, hint = "runtime_error", ""
    if "same device" in low:
        kind = "device_mismatch"
        hint = ("张量分散在不同设备（模型在显卡、某个张量在 CPU）。"
                "⚠ 这类错**常伪装成维度错**：先查 `torch.tensor(...)` 有没有带 device"
                "（框架已在你的钩子期间把默认设备摆正，但跨设备运算仍要显式写）")
    elif "cannot be multiplied" in low or "mat1 and mat2" in low:
        kind = "shape_mismatch"
        hint = ("线性层的输入特征数与权重对不上。核对 `build` 里的维度来源："
                "应当用 `ctx[\"payload_shapes\"]`，别写死。")
    elif "must match except in dimension" in low or "sizes of tensors must match" in low:
        kind = "shape_mismatch"
        hint = ("拼接/逐元素运算两侧形状不一致。常见根因是"
                "**把 K（每行采几个样本）当成了序列长度 L**。")
    elif "same number of dimensions" in low:
        kind = "rank_mismatch"
        hint = "索引张量的**秩**与被索引张量不一致；用 `unsqueeze`/`squeeze` 对齐。"
    elif "invalid for input of size" in low:
        kind = "reshape_mismatch"
        hint = "reshape 的目标形状与元素总数不符（常见：把 `n*K` 写成了 `n*L`）。"
    elif "not enough values to unpack" in low:
        kind = "unpack_mismatch"
        hint = "解包的个数与实际返回的个数不符。"
    elif 'missing the required "forward"' in low:
        kind = "missing_forward"
        hint = ("模型没有可用的 `forward`。若你写的是 `forward_step`，"
                "**必须**有它（训练/验证/探针/预测都走它）。")
    elif "out of memory" in low:
        kind = "cuda_oom"
        hint = "显存不够：减小批大小或模型规模。"
    if where:
        hint = (hint + "；" if hint else "") + where
    return {"kind": kind, "hint": hint}


def dry_run_network(*, net, problem, params: dict, torch, device,
                    payload: dict | None = None,
                    batch_probes: Any = None,
                    max_trace: int = 80) -> dict:
    """对一段网络代码**试跑**：编译 → 建模型 → 真实负载上跑几次前向。

    @returns ``{"ok", "batch_sizes_checked", "runs", "shapes", "n_params",
                "modules", "error", "error_kind", "error_hint", ...}``
    """
    from . import trainer as _trainer
    from .custom import _build_ctx, _hook

    dev = _trainer.resolve_device(device, torch)
    kind = ("custom" if (problem is not None and problem.target_kind == "custom")
            else (problem.target_kind if problem is not None else "regression"))
    # 与真跑**同一套**准备 ⇒ 试跑的形状就是真跑的形状
    prep = _trainer.prepare_supervised(
        problem.X, (None if problem is None else problem.y), kind=kind,
        seed=int(params.get("seed", 42) or 42),
        val_split=float(params.get("val_split", 0.2)))
    n_in, n_out = int(prep.Xs.shape[1]), int(prep.n_out)

    out: dict[str, Any] = {"ok": False, "batch_sizes_checked": [], "runs": [],
                           "shapes": [], "n_params": 0, "modules": [],
                           "error": "", "error_kind": "", "error_hint": "",
                           "device": str(dev), "n_in": n_in, "n_out": n_out,
                           "note": "试跑：不反传、不落盘、不计入实验记录"}
    from ml_mecha.sandbox import call_network_build
    try:
        model = call_network_build(
            net, n_in, n_out, torch.nn,
            ctx=_build_ctx(payload, n_rows=int(len(prep.Xs)), n_features=n_in,
                           n_out=n_out, target_kind=kind, device=str(dev)))
    except Exception as exc:                                   # noqa: BLE001
        info = classify_shape_error(exc, [])
        out.update(error=f"{type(exc).__name__}: {exc}",
                   error_kind=getattr(exc, "kind", "") or info["kind"],
                   error_hint=getattr(exc, "detail", "") or info["hint"])
        return out
    model = model.to(dev)
    out["n_params"] = int(sum(int(p.numel()) for p in model.parameters()))
    out["modules"] = [n for n, _m in model.named_modules() if n][:max_trace]

    lossf = None
    try:
        if callable(getattr(net, "loss", None)):
            lossf = net.loss(nn=torch.nn, n_out=n_out)
        elif kind == "classification":
            lossf = torch.nn.CrossEntropyLoss()
        elif kind != "custom":
            lossf = torch.nn.MSELoss()
    except Exception:                                          # noqa: BLE001
        lossf = None

    tr = np.asarray(prep.tr_idx, dtype=np.int64)
    fs = _hook(net, "forward_step")
    base = int(params.get("batch", 64) or 64)
    probes: list[int] = []
    for b in (list(batch_probes) if batch_probes else [*DEFAULT_BATCH_PROBES, base]):
        b = max(1, min(int(b), len(tr)))
        if b not in probes:
            probes.append(b)

    for b in probes:
        rows = tr[:b]
        ctx = {"torch": torch, "nn": torch.nn, "device": str(dev), "step": 0,
               "kind": "dry_run", "n_rows": int(len(prep.Xs)),
               "row_idx": rows, "train_idx": tr,
               "val_idx": np.asarray(prep.va_idx, dtype=np.int64)}
        if payload:
            ctx["payload"] = {k: _trainer._as_tensor(np.asarray(v)[rows], torch, dev)
                              for k, v in payload.items()}
        xb = torch.tensor(prep.Xs[rows], dtype=torch.float32, device=dev)
        yb = (torch.tensor(prep.yv[rows],
                           dtype=torch.long if kind == "classification"
                           else torch.float32, device=dev)
              if prep.yv is not None else None)
        rec = {"batch_size": b, "ok": False, "loss": None, "error": "",
               "error_kind": "", "error_hint": "", "n_shapes": 0}
        trace: list[dict] = []
        try:
            model.eval()
            with torch.no_grad(), _trainer.hook_device(torch, dev):
                with trace_modules(model) as trace:
                    if fs is not None:
                        o, t = fs(model, (xb, yb), ctx)
                    else:
                        o = model(xb)
                        t = yb
                    if lossf is not None:
                        try:
                            rec["loss"] = float(lossf(
                                o.squeeze(-1) if (kind != "classification"
                                                  and hasattr(o, "squeeze")) else o, t))
                        except Exception as exc:               # noqa: BLE001
                            info = classify_shape_error(exc, list(trace))
                            rec.update(error=f"（损失算不出来）{type(exc).__name__}: {exc}",
                                       error_kind=info["kind"], error_hint=info["hint"])
            rec["ok"] = not rec["error"]
        except Exception as exc:                               # noqa: BLE001
            info = classify_shape_error(exc, list(trace))
            rec.update(ok=False, error=f"{type(exc).__name__}: {exc}",
                       error_kind=info["kind"], error_hint=info["hint"])
        rec["n_shapes"] = len(trace)
        rec["shapes"] = trace[:max_trace]
        out["runs"].append(rec)
        out["batch_sizes_checked"].append(b)
        if trace and not out["shapes"]:
            out["shapes"] = trace[:max_trace]
        if not rec["ok"]:
            out.update(ok=False, error=rec["error"], error_kind=rec["error_kind"],
                       error_hint=rec["error_hint"], failed_batch_size=b)
            return out                                  # 第一处失败就停（省时间）
    out["ok"] = True
    losses = [r["loss"] for r in out["runs"] if r["loss"] is not None]
    if losses:
        out["loss_first_batch"] = losses[0]
    return out
