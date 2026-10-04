# -*- coding: utf-8 -*-
"""**自定义网络适配器**：把 AI 写的 `Network` 变成一台真正的 `MLMethod`。

## 它做什么（这就是"收编"的全部含义）

`CustomNNMethod.fit` 由**系统代写**，AI 只提供结构与（可选的）损失/优化器/前向：

```
① 从 cfg.extras 取 AI 的代码 → 沙箱编译（ml_mecha.sandbox）
② 数据准备      → trainer.prepare_supervised（标准化/编码/切分）
③ build 模型    → Network.build(n_in, n_out, nn=nn)
④ 损失/优化器   → Network.loss / Network.configure（可选，有默认）
⑤ 训练          → trainer.train（**托管 TB/进度/artifacts**）
⑥ 评估/预测     → 与 torch_mlp 同一套（metrics / y_pred / y_prob）
⑦ 检视页        → **沿用 torch_mlp 的五页**（训练动态/权重热图/回放/归因/逐层健康）
```

⇒ 残差网络、多分支、任意前向都能跑，而**观测面一行不用重写**：
`describe_training`、TensorBoard、`describe_run`、存档、GUI 检视页全部照旧。

## 为什么代码走 `cfg.extras`（而不是构造参数）

`runner.run_one` 的签名是固定的（`run_one(method, spec, cfg)`），构造参数传不进去；
而 `cfg.extras` 本来就是"任务级扩展"的通道（`run_id` / `device` / `on_progress`
都走它）。代码是**字符串**，所以它还是 JSON 可序列化的 ⇒ 能原样进
`runs/<id>/record.json` 的 `config.extras`，**复用与审计都白拿**。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from ...core.contracts import (AUX_GRAPH, AUX_PAYLOAD, MLMethod, PageSpec,
                               ParamSpec,
                               TASK_SUPERVISED,
                               classification_metrics, regression_metrics)
from ...core.pipeline import _infer_kind
from . import nnplots
from . import trainer as _trainer
from .network import NETWORK_HOOKS                              # noqa: F401 - 单一真源
from .recorder import TrainingRecorder

#: 这份自定义网络**在 extras 里放代码的键名**（引擎与工具面共用同一常量）。
#: ⚠ 单一真源：`ml_mecha` 侧从 `ml_toolbox` 读它，不各写一份字面量。
CUSTOM_CODE_KEY = "custom_network_code"

#: 名字前缀：注册表里的临时方法名（`run_custom` 每次跑完就撤，见 engine）。
CUSTOM_METHOD_NAME = "custom_network"


class CustomNNMethod(MLMethod):
    """AI 写的 `Network` 的宿主方法（`fit` 由本类代写）。

    ⚠ 它**按请求临时注册**（名字固定 `custom_network`，跑完即撤）：这样
    `registry.get(name)` 的既有语义（无参构造）不用改，也不给注册表留垃圾。
    """

    name = CUSTOM_METHOD_NAME
    display_name = "自定义网络"
    family = "neural"
    task = TASK_SUPERVISED
    target_kind = None
    tags = ("deep", "nonlinear", "custom", "gpu")
    #: ⭐ **声明支持图与逐行负载**：数据带了 `aux["graph"]` 时走 `_fit_graph`
    #: （整图前向 + 节点掩码）；带了 `aux["payload"]` 时**仍走表格路径**，
    #: 但逐批把负载切给 `forward_step`（逐行负载不改变"行=样本"的语义）。
    #: 没有这条声明的话 `Problem.require` / 入口的能力闸会**结构化拒绝**——
    #: 那正是"能力要声明，不许静默丢弃"那条纪律。
    accepts_aux = (AUX_GRAPH, AUX_PAYLOAD)

    #: 训练超参：**只列自定义网络真正会消费的那些**（循环由 trainer 托管）。
    #: ⚠ 这张表是"哪些覆写被接受"的单一真源（引擎的 `clean_overrides` 按它校验）；
    #: `activation`/`optimizer`/`init` 之类**故意不在**这里——那些是 `Network`
    #: 代码自己的事，放进来会变成"收下就扔"的假旋钮。
    param_schema = [
        ParamSpec("epochs", "训练轮数", "int", 60, min=1, max=2000),
        ParamSpec("batch", "批大小", "int", 64, min=1, max=4096),
        ParamSpec("lr", "学习率", "number", 1e-3, min=1e-6, max=1),
        ParamSpec("weight_decay", "权重衰减", "number", 0.0, min=0),
        ParamSpec("val_split", "验证集比例", "number", 0.2, min=0, max=0.5,
                  hint="0 = 不切验证集"),
        ParamSpec("latent_every", "表示快照间隔", "int", 5, min=1,
                  hint="每 N epoch 存一帧隐层 2D 投影（回放页用）"),
        ParamSpec("grad_clip", "梯度裁剪", "number", 0.0, min=0.0,
                  hint="0 = 不裁剪；>0 按全局范数裁剪"),
        ParamSpec("early_stop", "早停", "select", "off", choices=["off", "on"],
                  hint="on ⇒ 验证损失连续 early_stop_patience 轮不改善就停"),
        ParamSpec("early_stop_patience", "早停耐心", "int", 10, min=1),
    ]

    #: 由引擎在派发前注入（``make_custom_instance``）；**不进 param_schema**
    #: ——它不是"方法的参数"，是"这一段代码"。
    custom_code: str = ""

    @staticmethod
    def _tb_root():
        try:
            from ...core.persistence import RUNS_DIR
            return RUNS_DIR
        except Exception:
            return None

    # ------------------------------------------------ 代码 → 网络实例
    def _network(self):
        """编译 `self.custom_code`（沙箱），返回 `CompiledNetwork`。"""
        from ml_mecha.sandbox import SandboxError, compile_network

        code = str(getattr(self, "custom_code", "") or "")
        if not code.strip():
            raise SandboxError(
                "没有拿到网络代码",
                kind="missing_custom_code",
                detail=f"引擎应把代码放进 `cfg.extras[{CUSTOM_CODE_KEY!r}]`",
            )
        return compile_network(code)

    # ------------------------------------------------ 训练
    def fit(self, problem, cfg, diag: bool = False):
        """按**数据形态**分派：普通表格走 `_fit_table`，带结构（图）走 `_fit_graph`。

        ⚠ 分派依据是 `problem.partition_mode()`（**由数据决定，不由方法决定**）：
        带了辅助结构就意味着"行不独立"（图=按边相连）⇒ 必须用**节点掩码**划分、
        整图前向，不能按行切（切了边索引失效、且训练节点会沿边看到验证节点）。
        """
        if problem.partition_mode() == "masks":
            return self._fit_graph(problem, cfg, diag=diag)
        return self._fit_table(problem, cfg, diag=diag)

    def _fit_table(self, problem, cfg, diag: bool = False):
        import torch

        X, y = problem.rows("train")

        # ⚠ **训练超参走 `self.params(cfg)`**（= 声明默认 + 运行级覆写），
        #   不是 `cfg.extras`：覆写是经 `cfg.overrides` 进来的，而 extras 只承载
        #   run_id / device / on_progress / 代码这些**非参数**的运行级东西。
        #   实测踩过：从 extras 读 ⇒ 用户给的 `epochs=6` 被静默忽略、跑了默认 60 轮
        #   （判据里表现为"6 epoch 的训练有 60 个 loss 点"）。
        p = self.params(cfg)
        run_extras = cfg.extras or {}
        # ⭐ **设备**：`device` 由引擎放进 extras（`run_custom` 已声明该参数）。
        #   ⚠ 此前这条路上硬编码 CPU（`res._dev = torch.device("cpu")`）⇒ 46 万参数的
        #   模型在 CPU 上根本跑不完，而回执看着"一切正常"。现在真解析、真搬。
        dev = _trainer.resolve_device(run_extras.get("device", "auto"), torch)
        # ⭐ **逐行负载**（NQS 需求）：变长/复数数组按行切开，逐批放进 `forward_step` 的 ctx
        payload = _payload_of(problem)
        # 训练负载的**声明**（引擎经 extras 下传）⇒ 写进 checkpoint，供 eval_custom 继承
        payload_decl = dict(run_extras.get("payload_decl") or {})
        # ⚠ `SandboxError` 走**惰性 import**：本模块顶层 import ml_mecha 会成环
        #   （ml_mecha → ml_toolbox.api → methods）。别提到模块顶部。
        from ml_mecha.sandbox import SandboxError
        try:
            compiled = self._network()
        except SandboxError as exc:
            # ⭐ **AI 写的代码出错 ⇒ 必须归成"你的代码"，不是"框架坏了"**。
            #   ⚠ 这段原先**没有** try：异常一路逃到 `runner.run_one` 的兜底，
            #   被压成一句 traceback 字符串 ⇒ 到 AI 那里成了 `internal_error` /
            #   `command_failed` + 万能 hint（实测；判据
            #   `test_custom_network_errors_are_readable_not_raw_stacks` 抓到）。
            #   沙箱错误的 `kind`/`detail` 本来就是写给人/AI 看的 ⇒ 原样带出去。
            #   （表格与图两条路同形，**同一个修法**。）
            return self._new_result(target_kind=self.target_kind).fail(
                str(exc), kind=exc.kind, hint=exc.detail,
                suggest="改代码后重跑；`describe_network` 可取回上一次的源码对照")
        net = compiled.instance
        # ⭐ **自定义任务**（`target_kind="custom"`）：没有标准标签 ⇒
        #   标签在逐行负载里，损失与指标都由网络自己给。
        #   ⚠ `kind` 必须是 "custom" 而不是靠 `_infer_kind(y)` 猜（y 是 None）。
        kind = ("custom" if (problem.target_kind == "custom" or y is None)
                else (self.target_kind or _infer_kind(y)))
        if kind == "custom" and not callable(getattr(net, "loss", None)):
            return self._new_result(target_kind="custom").fail(
                "自定义任务（没有标准标签）必须提供 `Network.loss`",
                kind="missing_custom_loss",
                hint="没有标准标签时，框架不知道拿什么算损失 ⇒ 只能由你的网络给"
                     "（`def loss(self, *, nn, n_out): return …`）",
                suggest="要么加 `loss` 钩子，要么给数据一个标准 `target` 列")
        prep = _trainer.prepare_supervised(
            X, y, kind=kind, seed=int(cfg.seed),
            val_split=float(p.get("val_split", 0.2)))
        n_in = int(prep.Xs.shape[1])
        n_out = int(prep.n_out)

        # 结构由 AI 写（沙箱编译已保证 build 存在且返回 nn.Module）
        from ml_mecha.sandbox import call_network_build
        # ⭐ **`build` 拿得到负载形状**（反馈 §2-缺口1）：只在 `build` 里"猜维度"，
        #   换个 N / 加一维特征就会**静默跑偏**（那是本仓最忌的假绿）。
        #   ⚠ 这份 ctx 还要**进 checkpoint**（`build_ctx`）：否则 `eval_custom`
        #   重建模型时给不出它，`build` 里读 `payload_shapes` 的网络会直接炸。
        build_ctx = _build_ctx(payload, n_rows=int(len(prep.Xs)), n_features=n_in,
                               n_out=n_out, target_kind=kind, device=str(dev))
        model = call_network_build(net, n_in, n_out, torch.nn, ctx=build_ctx)
        # 损失/优化器：有钩子用钩子，否则给与 torch_mlp 一致的默认
        if callable(getattr(net, "loss", None)):
            lossf = net.loss(nn=torch.nn, n_out=n_out)
        else:
            lossf = (torch.nn.CrossEntropyLoss() if kind == "classification"
                     else torch.nn.MSELoss())
        lr = float(p.get("lr", 1e-3))
        wd = float(p.get("weight_decay", 0.0))
        if callable(getattr(net, "configure", None)):
            opt = net.configure(model, torch=torch, lr=lr, weight_decay=wd)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        run_id = str(run_extras.get("run_id") or "")
        rec = TrainingRecorder(model, len(prep.tr_idx), int(p.get("epochs", 60)),
                               latent_every=int(p.get("latent_every", 5)),
                               run_id=run_id, tb_root=self._tb_root())
        probe_rows = prep.tr_idx[:min(256, len(prep.tr_idx))]
        # ⚠ `prep.yv` 在**自定义任务**里是 None（没有标准标签）⇒ 不能直接下标。
        #   探针的 labels 只用于 latent 帧的着色，没有标签就传 None（如实）。
        probe_labels = None if prep.yv is None else prep.yv[probe_rows]
        # ⭐⭐ **探针前向也必须走同一条入口**（反馈 §2-缺口3 的**真正现场**）。
        #   探针是"跑一次前向、用 hook 收集激活"（见 recorder._collect_probe），
        #   它默认调 `model(self._probe)` —— 而"只有 forward_step、没有可用 forward"
        #   的模型（NQS 的自回归头就是）会炸
        #   `NotImplementedError: Module [Net] is missing the required "forward" function`。
        #   ⚠ 反馈把它当成"验证轮调了 model(Xva)"；验证轮其实是对的（走 forward_step），
        #   **炸的是探针** —— 这也是为什么红验证要"看红的理由对不对"（A13）。
        probe_fn = None
        if callable(getattr(net, "forward_step", None)):
            probe_payload = {k: v[probe_rows] for k, v in (payload or {}).items()}

            def probe_fn():                               # noqa: F811
                pctx = {"torch": torch, "nn": torch.nn, "device": str(dev),
                        "step": 0, "kind": "probe", "n_rows": int(len(prep.Xs)),
                        "payload": {k: _trainer._as_tensor(v, torch, dev)
                                    for k, v in probe_payload.items()}}
                with _trainer.hook_device(torch, dev):
                    out, _tgt = net.forward_step(
                        model, (torch.tensor(prep.Xs[probe_rows],
                                             dtype=torch.float32, device=dev), None),
                        pctx)
                return out

        rec.set_probe(torch.tensor(prep.Xs[probe_rows], dtype=torch.float32,
                                   device=dev),
                      labels=probe_labels, probe_idx=probe_rows, probe_fn=probe_fn)
        self._probe_idx = probe_rows

        res = self._new_result(target_kind=kind, params={"lr": lr})
        res._torch = torch
        res._net = net                       # `predict` 要用它探测 `forward_step`
        res._dev = dev                      # ⭐ 如实记：审计说 cuda 就得真是 cuda
        res._mu, res._sd = prep.mu, prep.sd
        res.classes_ = (prep.label_encoder.classes_
                        if kind == "classification" and prep.label_encoder is not None
                        else None)
        try:
            stats = _trainer.train(
                rec, model=model, prep=prep, torch=torch, nn=torch.nn,
                epochs=int(p.get("epochs", 60)), batch=int(p.get("batch", 64)),
                lr=lr, loss_fn=lossf, optimizer=opt,
                grad_clip=float(p.get("grad_clip", 0.0)),
                early_stop=str(p.get("early_stop", "off")) == "on",
                patience=int(p.get("early_stop_patience", 10)),
                on_progress=run_extras.get("on_progress"),
                device=str(dev), payload=payload,
                # ⚠ **`forward_step` 此前从没被传进来**（表格路径上那个钩子是死的！
                #   用户说"我还没走到 forward_step"—— 一半原因是数据进不来，
                #   另一半原因是**我们根本没调用它**）。这条与 payload 是一对：
                #   负载给了数据，forward_step 才有地方消费。
                forward_step=_hook(net, "forward_step"),
                on_step=_hook(net, "on_step"), metrics=_hook(net, "metrics"),
                # 贵的指标（如在全量支撑上算保真度）用 `metrics_every` 降频；
                # 末轮一定算（保证回执/回看的终值不是旧的 —— 否则又是一个假事实）
                metrics_every=int(getattr(net, "metrics_every", 1) or 1),
                keep_best=True)
        except Exception as exc:                             # noqa: BLE001
            import traceback
            res = self._new_result(target_kind=kind, params={"lr": lr}).fail(
                f"自定义网络训练失败：{type(exc).__name__}: {exc}",
                kind="network_runtime_error",
                hint=_device_error_hint(exc) or
                     "是你那段网络代码在训练期抛的（不是框架）；栈留在宿主日志",
                suggest="改代码后重跑；`describe_network` 可取回上一次的源码对照")
            return res
        res.est = model
        # ⭐ **网络自己上报的指标也要进回执**（不只进 TB）：
        #   TB 那条路给"看曲线"，这条给 `describe_run` / `compare_methods` 的"看数字"。
        #   自定义任务（没有标准指标）**全靠这一条**——否则 `describe_run` 里
        #   什么都看不到，AI 只能去翻事件文件。
        learned = dict(stats.get("last_extra") or {})
        if learned:
            res.metrics.update({str(k): float(v) for k, v in learned.items()
                                if isinstance(v, (int, float))})
        # ⭐ **步级指标**（`on_step` 的返回）也进回执：与 TB 曲线同一批 tag，
        #   所以"回执里的数字"和"曲线上的点"不会各说各话。
        step_m = dict(stats.get("last_step_metrics") or {})
        if step_m:
            res.metrics.update(step_m)
        if kind == "custom" or learned or step_m:
            # 主指标：网络可显式声明（`Network.primary_metric = "fid/mean"`）；
            # 否则取**字典序第一个**（确定、可复现；不猜哪个"更重要"）。
            # ⚠ 自定义任务**必须覆盖** `_new_result` 给的默认（那是按 task/target_kind
            #   推的 "f1"/"rmse"）—— 否则回执会说主指标是 `f1`，而这次运行
            #   **根本没有 f1 这个数**（假指标比空指标更坏）。
            declared = str(getattr(net, "primary_metric", "") or "")
            if declared and declared in res.metrics:
                res.primary_metric = declared
            elif res.metrics:
                res.primary_metric = sorted(res.metrics)[0]
            else:
                res.primary_metric = ""
        # ⭐ 取消/早停/最佳轮次：**写在最显眼的地方**（`res.diag` 由 diag 开关控制，
        #   但"这次是被取消的"不该依赖诊断开关）⇒ 进 artifacts 的标量里，
        #   `describe_run` / 存档 / GUI 都看得到。
        res.artifacts["nn_epochs_run"] = np.array([int(stats["epochs_run"])])
        res.artifacts["nn_cancelled"] = np.array([1 if stats.get("cancelled") else 0])
        if stats.get("best_epoch") is not None:
            res.artifacts["nn_best_epoch"] = np.array([int(stats["best_epoch"])])
            res.artifacts["nn_best_val"] = np.array([float(stats["best_watch"])])
        if stats.get("cancelled"):
            # 协作式取消：如实标失败，理由写清楚（**不假装跑完了**）
            res.error = ("训练被协作式取消（在 epoch 边界停下）："
                         f"已跑 {stats['epochs_run']} 轮。"
                         "⚠ 今天只在命令入口/方法边界与**训练器的 epoch 边界**生效。")
        try:
            res.artifacts = rec.finalize()
        except Exception:
            pass
        if "nn_latent" in res.artifacts:
            lab = rec.last_labels
            if lab is not None:
                res.artifacts["labels"] = lab
        # ⭐ 代码跟着运行走：跨会话能取回、能按名字复用（**别让 AI 反复重贴**）
        res.artifacts["custom_network_source"] = np.array([str(compiled.source)])
        # ⭐⭐ **真 checkpoint**（NQS 需求的最高优先项："训练完能把模型拿回来"）
        #   ⚠ 为什么不能靠 `nn_weights`：那个是**诊断产物** —— 只存 `dim()>=2` 的参数
        #   （`recorder.py:246`）⇒ **偏置/LayerNorm 根本没存**，装不回来；
        #   而且只有末态（末代没有特殊意义，最佳轮才是要用的那个）。
        #   ⇒ 这里另存**完整 state_dict**（含全部参数与 buffer），best 与 last 各一份。
        ckpt = _save_checkpoints(
            run_id=run_id, res=res, stats=stats, source=str(compiled.source),
            params={k: v for k, v in p.items() if k != "custom_network_code"},
            target_kind=kind, n_in=n_in, n_out=n_out, device=str(dev),
            meta={"n_train": int(len(prep.tr_idx)), "n_val": int(len(prep.va_idx))},
            payload_decl=payload_decl,
            build_ctx=build_ctx, inline_payload=_inline_payload_of(payload_decl, payload))
        if ckpt:
            res.artifacts["checkpoint_dir"] = np.array([ckpt["dir"]])
            # ⚠ 不能存 object 数组（写进 npz 读不回，实测踩过 `allow_pickle=False`）
            #   ⇒ 文件名**拼成一个字符串**（`|` 分隔，读侧 split）。
            res.artifacts["checkpoint_files"] = np.array(["|".join(ckpt["files"])])
        if diag:
            res.diag = {"epochs": int(p.get("epochs", 60)),
                        "epochs_run": stats["epochs_run"],
                        "early_stopped_at": stats["stopped_at"],
                        "cancelled": bool(stats.get("cancelled")),
                        "best_epoch": stats.get("best_epoch"),
                        "best_val": stats.get("best_watch"),
                        "n_train": int(len(prep.tr_idx)), "n_val": int(len(prep.va_idx)),
                        "n_in": n_in, "n_out": n_out,
                        "device": str(dev), "steps": int(stats.get("steps", 0)),
                        "payload_keys": sorted(payload or {}),
                        "checkpoint": (ckpt or {}).get("files", []),
                        "hooks": sorted(h for h in NETWORK_HOOKS
                                        if callable(getattr(net, h, None)))}
        return res

    # ------------------------------------------------ 图任务（GNN）
    def _fit_graph(self, problem, cfg, diag: bool = False):
        """**图任务**：一张图、节点掩码划分、整图前向。

        与 `_fit_table` 的四处本质差别（全部源于"行不独立"）：
        1. `n_in` 从**全量**节点特征取（不切行 ⇒ 边索引始终有效）；
        2. 训练/验证靠 `problem.masks()` 的**布尔掩码**，不是切行；
        3. 前向走 `Network.forward_graph(model, ctx)`（不写就用 `model(x)` 全图）；
        4. **在 fit 里就把测试节点的预测算好**（`run_one` 的 `predict(Xte, ...)` 那条
           路对图不成立——它只给测试行，而图前向需要整张图）。
        """
        import torch

        p = self.params(cfg)
        run_extras = cfg.extras or {}
        payload_decl = dict(run_extras.get("payload_decl") or {})
        dev = _trainer.resolve_device(run_extras.get("device", "auto"), torch)
        # ⚠ `SandboxError` 走**惰性 import**：本模块顶层 import ml_mecha 会成环
        #   （ml_mecha → ml_toolbox.api → methods）。别提到模块顶部。
        from ml_mecha.sandbox import SandboxError
        try:
            compiled = self._network()
        except SandboxError as exc:
            # ⭐ **AI 写的代码出错 ⇒ 必须归成"你的代码"，不是"框架坏了"**。
            #   ⚠ 这段原先**没有** try：异常一路逃到 `runner.run_one` 的兜底，
            #   被压成一句 traceback 字符串 ⇒ 到 AI 那里成了 `internal_error` /
            #   `command_failed` + 万能 hint（实测；判据
            #   `test_custom_network_errors_are_readable_not_raw_stacks` 抓到）。
            #   沙箱错误的 `kind`/`detail` 本来就是写给人/AI 看的 ⇒ 原样带出去。
            #   （表格与图两条路同形，**同一个修法**。）
            return self._new_result(target_kind=self.target_kind).fail(
                str(exc), kind=exc.kind, hint=exc.detail,
                suggest="改代码后重跑；`describe_network` 可取回上一次的源码对照")
        net = compiled.instance
        n_in = int(problem.X.shape[1])
        # 标签：图任务按"节点分类"语义（一行一个节点一个标签）
        yv = np.asarray(problem.y)
        kind = problem.target_kind or _infer_kind(problem.y)
        self._le = None
        if kind == "classification":
            from sklearn.preprocessing import LabelEncoder
            self._le = LabelEncoder()
            yv = self._le.fit_transform(yv)
            n_out = len(self._le.classes_)
        else:
            yv = yv.astype(float)
            n_out = 1
        # 把编码后的标签写回一个**仅本次用**的 spec（trainer.train_graph 读 problem.y）
        problem.spec.y = pd.Series(yv)

        from ml_mecha.sandbox import call_network_build
        build_ctx = _build_ctx(None, n_rows=int(len(problem.X)), n_features=n_in,
                               n_out=n_out, target_kind=kind, device=str(dev))
        model = call_network_build(net, n_in, n_out, torch.nn, ctx=build_ctx)

        if callable(getattr(net, "loss", None)):
            lossf = net.loss(nn=torch.nn, n_out=n_out)
        else:
            lossf = (torch.nn.CrossEntropyLoss() if kind == "classification"
                     else torch.nn.MSELoss())
        lr = float(p.get("lr", 1e-3))
        wd = float(p.get("weight_decay", 0.0))
        if callable(getattr(net, "configure", None)):
            opt = net.configure(model, torch=torch, lr=lr, weight_decay=wd)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        run_id = str(run_extras.get("run_id") or "")
        n_nodes = int(len(problem.X))
        m_tr, _m_te = problem.masks()
        rec = TrainingRecorder(model, int(m_tr.sum()), int(p.get("epochs", 60)),
                               latent_every=int(p.get("latent_every", 5)),
                               run_id=run_id, tb_root=self._tb_root())
        # ⭐ 探针前向：图模型不能喂裸张量（需要 edge_index）⇒ 交给 recorder 一个
        #   无参可调用（它只用来采 dead-ReLU / latent 帧，不参与训练）。
        #   ctx 走 `trainer.graph_ctx`（**单一构造点**——内联拼会漏键）。
        fg = getattr(net, "forward_graph", None)

        def _probe_forward():
            c = _trainer.graph_ctx(problem, torch, device=str(dev))
            return (fg(model, c) if callable(fg) else model(c["x"]))

        probe_nodes = np.where(m_tr)[0][:min(256, int(m_tr.sum()))]
        rec.set_probe(torch.zeros((len(probe_nodes), n_in)), labels=yv[probe_nodes],
                      probe_idx=probe_nodes, probe_fn=_probe_forward)
        self._probe_idx = probe_nodes

        res = self._new_result(target_kind=kind, params={"lr": lr})
        res._torch = torch
        res._dev = dev
        res._mu = np.zeros(n_in)
        res._sd = np.ones(n_in)
        res.classes_ = (self._le.classes_ if self._le is not None else None)
        try:
            stats = _trainer.train_graph(
                rec, model=model, problem=problem, torch=torch, nn=torch.nn,
                epochs=int(p.get("epochs", 60)), lr=lr, loss_fn=lossf, optimizer=opt,
                grad_clip=float(p.get("grad_clip", 0.0)),
                early_stop=str(p.get("early_stop", "off")) == "on",
                patience=int(p.get("early_stop_patience", 10)),
                on_progress=run_extras.get("on_progress"),
                forward_graph=fg if callable(fg) else None,
                device=str(dev), metrics=_hook(net, "metrics"), keep_best=True)
        except Exception as exc:                             # noqa: BLE001
            res = self._new_result(target_kind=kind, params={"lr": lr}).fail(
                f"图任务训练失败：{type(exc).__name__}: {exc}",
                kind="network_runtime_error",
                hint=_device_error_hint(exc) or
                     "是你那段网络代码在训练期抛的（不是框架）；栈留在宿主日志")
            return res
        res.est = model
        res._is_graph = True
        # ⭐ 测试节点的预测在 fit 里算（`run_one` 的 predict(Xte) 对图不成立：
        #   图前向需要整张图的边，而那条路只给测试行）。
        #   `predict_graph` 直接回**只含 test 节点**的定型数组
        #   （**不是 object 数组**——object 数组写进 npz 会读不回来，实测踩过）。
        try:
            pred, prob = _trainer.predict_graph(
                model, problem, torch,
                forward_graph=fg if callable(fg) else None,
                classes=(self._le.classes_ if self._le is not None else None),
                device=str(dev))
            te = np.asarray(problem.test_idx, dtype=int)
            y_all = np.asarray(problem.y)
            # `y_true` 要**原始标签**（不是编码后的 0..k-1）——指标与检视页都按原始
            # 标签比对（与 `torch_mlp` 同口径）。
            if self._le is not None:
                y_true_te = self._le.inverse_transform(y_all[te].astype(int))
            else:
                y_true_te = y_all[te]
            res.artifacts["y_true"] = np.asarray(y_true_te)
            res.artifacts["y_pred"] = np.asarray(pred)
            if prob is not None:
                res.artifacts["y_prob"] = np.asarray(prob, dtype=float)
        except Exception as exc:                             # noqa: BLE001
            import traceback
            res.error = (f"图任务预测阶段失败：{type(exc).__name__}: {exc}\n"
                         + traceback.format_exc(limit=6))
        res.artifacts["nn_epochs_run"] = np.array([int(stats["epochs_run"])])
        res.artifacts["nn_cancelled"] = np.array([1 if stats.get("cancelled") else 0])
        if stats.get("best_epoch") is not None:
            res.artifacts["nn_best_epoch"] = np.array([int(stats["best_epoch"])])
            res.artifacts["nn_best_val"] = np.array([float(stats["best_watch"])])
        try:
            arts = rec.finalize()
            arts.update({k: v for k, v in res.artifacts.items()})
            res.artifacts = arts
        except Exception:
            pass
        res.artifacts["custom_network_source"] = np.array([str(compiled.source)])
        # 图任务同样要**完整 checkpoint**（与表格路径同一份实现）
        ckpt = _save_checkpoints(
            run_id=run_id, res=res, stats=stats, source=str(compiled.source),
            params={k: v for k, v in p.items()}, target_kind=kind,
            n_in=n_in, n_out=n_out, device=str(dev),
            meta={"task_mode": "graph", "n_nodes": n_nodes},
            payload_decl=payload_decl,
            # ⚠ 图路径**没有 `payload` 局部变量**（负载是表格路径的东西）⇒ 显式给 None，
            #   否则 `NameError`（实测：这一处漏了会连带把 GNN 判据打红）。
            build_ctx=build_ctx, inline_payload=_inline_payload_of(payload_decl, None))
        if ckpt:
            res.artifacts["checkpoint_dir"] = np.array([ckpt["dir"]])
            res.artifacts["checkpoint_files"] = np.array(["|".join(ckpt["files"])])
        # 图事实（节点/边/度分布/isolation）——AI 与人都该看得到
        facts = problem.graph_facts()
        if facts:
            res.artifacts["graph_n_nodes"] = np.array([int(facts["n_nodes"])])
            res.artifacts["graph_n_edges"] = np.array([int(facts["n_edges"])])
        if stats.get("cancelled"):
            res.error = ("训练被协作式取消（在 epoch 边界停下）："
                         f"已跑 {stats['epochs_run']} 轮。")
        if diag:
            # ⚠ `problem.test_idx` 是 **numpy 数组** ⇒ 不能写 `x or []`
            #   （会触发"数组真值不明确"的 ValueError，实测踩过）。
            n_test = int(len(problem.test_idx)) if problem.test_idx is not None else n_nodes
            res.diag = {"task_mode": "graph", "n_nodes": n_nodes,
                        "n_train_nodes": int(m_tr.sum()),
                        "n_test_nodes": n_test,
                        "epochs": int(p.get("epochs", 60)),
                        "epochs_run": stats["epochs_run"],
                        "early_stopped_at": stats["stopped_at"],
                        "best_epoch": stats.get("best_epoch"),
                        "best_val": stats.get("best_watch"),
                        "n_in": n_in, "n_out": n_out,
                        "graph": facts,
                        "hooks": sorted(h for h in ("build", "loss", "configure",
                                                    "forward_step", "forward_graph")
                                        if callable(getattr(net, h, None)))}
        return res

    # ------------------------------------------------ 预测（与 TorchMLP 同构）
    def predict(self, X, result):
        if getattr(result, "_is_graph", False):
            raise NotImplementedError(
                "图模型的预测需要**整张图的边**，而这里只拿到一批新行（X）"
                "⇒ 对新数据预测在图任务上不成立。测试节点的预测已随运行产出"
                "（`y_pred` 按 test 掩码对齐）；要预测新节点，请把新节点并进图后重跑。")
        torch = result._torch
        Xs = (X.to_numpy(float) - result._mu) / result._sd
        # ⚠ 张量必须放到**模型所在的那块设备**（`result._dev`）：模型在 CUDA 而
        #   输入在 CPU ⇒ `Expected all tensors to be on the same device`（实测踩过，
        #   而且只有真上卡时才暴露 —— 这是"设备真透传"这条路上容易漏的尾巴）。
        t = torch.tensor(Xs, dtype=torch.float32, device=result._dev)
        result.est.eval()
        # ⭐ **有 `forward_step` 就用它**（与训练/验证同一条入口）：只实现
        #   `forward_step` 的模型，`model(t)` 会直接炸
        #   `missing the required "forward" function`。这里没有负载（预测的是新行），
        #   所以给一个空负载 + 明确说明：真要"带负载预测"，走 `eval_custom`。
        fs = _hook(getattr(result, "_net", None), "forward_step")
        with torch.no_grad():
            if fs is not None:
                ctx = {"torch": torch, "nn": torch.nn, "device": str(result._dev),
                       "step": 0, "kind": "predict", "n_rows": int(len(Xs)),
                       "payload": {},
                       "hint": "预测新行时没有逐行负载；要带负载评估请用 eval_custom"}
                with _trainer.hook_device(torch, result._dev):
                    out, _tgt = fs(result.est, (t, None), ctx)
            else:
                out = result.est(t)
        if result.target_kind == "classification":
            pred = out.argmax(1).cpu().numpy()
            classes = getattr(result, "classes_", None)
            if classes is None:
                return pred
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.classes_ = np.asarray(classes)
            return le.inverse_transform(pred)
        return out.squeeze(-1).cpu().numpy()

    def fit_extra_artifacts(self, X, result) -> dict:
        torch = result._torch
        Xs = (X.to_numpy(float) - result._mu) / result._sd
        t = torch.tensor(Xs, dtype=torch.float32, device=result._dev)
        result.est.eval()
        if result.target_kind == "classification":
            with torch.no_grad():
                prob = torch.softmax(result.est(t), 1).cpu().numpy()
            return {"y_prob": prob}
        return {}

    # ------------------------------------------------ 检视页（沿用 torch_mlp 五页）
    def inspect_pages(self, cfg):
        """⭐ **一行不改地沿用**：自定义网络与 `torch_mlp` 的 artifacts 同构，
        所以五页检视直接可用（权重热图/表示演化/逐层健康消费同一批键）。"""
        return [
            PageSpec("nn_dyn", "训练动态", "mpl", nnplots.plot_nn_dynamics),
            PageSpec("nn_w", "权重热图", "nn_weights", hint="缺少权重快照工件"),
            PageSpec("nn_replay", "表示演化", "nn_replay",
                     hint="缺少隐层投影快照（检查 latent_every 参数）"),
            PageSpec("nn_attr", "特征归因", "mpl",
                     nnplots.plot_feature_attribution,
                     hint="归因在诊断开启时计算（积分梯度，32 步）"),
            PageSpec("nn_layers", "逐层健康", "table", hint="缺少逐层表"),
        ]


def make_custom_instance(code: str) -> CustomNNMethod:
    """造一个带着某段代码的自定义网络方法实例（引擎每跑一次就造一个）。"""
    m = CustomNNMethod()
    m.custom_code = str(code or "")
    return m


# ---------------------------------------------------------------- 网络钩子 / 负载 / 存档
def load_checkpoint_meta(run_id: str, *, which: str = "best") -> dict:
    """只读 checkpoint 的**元信息**（不装载模型、不内联权重）。

    用途：`eval_custom` 要判断"训练时用的是什么负载"（反馈建议 4：默认继承）。
    ⚠ 返回的是**声明**（`payload_decl`：HDF5 路径 + 数据集名，或小的内联数组），
    不是大数组本身 —— 大数组留在原处按需读。
    """
    import torch
    from ...core.persistence import RUNS_DIR

    w = "last" if str(which or "best").lower() in ("last", "final") else "best"
    path = Path(RUNS_DIR) / str(run_id) / "checkpoint" / f"{w}.pt"
    if not path.is_file():
        return {}
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:                                          # noqa: BLE001
        return {}
    return {k: v for k, v in ck.items() if k != "state_dict"}


def _network_of(code: str):
    """编译一段代码并回**网络实例**（`Network` 的对象；试跑用）。

    ⚠ 与 `CustomNNMethod._network()` **同一条编译路径**：试跑与真跑若各编一遍，
    迟早会出现"试跑过的代码"与"真跑的代码"不是同一份 —— 那正是最坏的假绿。
    """
    from ml_mecha.sandbox import compile_network
    return compile_network(str(code or "")).instance
def _device_error_hint(exc: BaseException) -> str:
    """识别"设备混用"这类错并给**可教学**的说法（认不出就回空串）。

    ⚠ 为什么必须转（这正是本轮引入的新风险）：`run_custom` 以前**永远在 CPU**，
    任何网络代码随手 `torch.arange(...)` 都能跑；现在 `device=auto` 会在有卡时
    真上 CUDA ⇒ **旧代码里那些"没写设备"的张量**就会撞
    `Expected all tensors to be on the same device`。
    这条报错本身对使用者毫无指向性（它不会告诉你"该用 ctx['device']"），
    而它恰恰是**本轮改动带来**的，所以必须由我们翻译。
    """
    msg = f"{type(exc).__name__}: {exc}"
    if "same device" not in msg and "device" not in msg.lower():
        return ""
    return ("网络代码里建张量时**没带设备**，而模型已经搬到显卡上 ⇒ 混用报错。"
            "修法：在代码里用 `ctx[\"device\"]` 建张量，例如 "
            "`torch.arange(n, device=ctx[\"device\"])`；"
            "或者这次显式 `device='cpu'` 跑。")


def load_checkpoint(run_id: str, *, which: str = "best"):
    """按 `run_id` 装载 checkpoint → ``(model, meta)``。

    ⚠ 用**源码重建 + `load_state_dict`**：checkpoint 里同时存了那段网络代码
    （`source`）⇒ "重建模型"不依赖你还记着代码。这也是它必须与
    `custom_network_source` 同源的原因（两处都来自沙箱编译的那份源码）。
    """
    import torch
    from ml_mecha.sandbox import call_network_build, compile_network
    from ...core.persistence import RUNS_DIR

    which = "best" if str(which or "best").lower() not in ("last", "final") else "last"
    path = Path(RUNS_DIR) / str(run_id) / "checkpoint" / f"{which}.pt"
    if not path.is_file():
        # 明确区分"没有这次运行"与"这次运行没存 checkpoint"
        alt = list((Path(RUNS_DIR) / str(run_id) / "checkpoint").glob("*.pt")) \
            if (Path(RUNS_DIR) / str(run_id)).is_dir() else []
        raise FileNotFoundError(
            f"{path} 不存在" + (f"（该目录下现有：{[p.name for p in alt]}）" if alt
                             else "（该 run 目录都不存在）"))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    compiled = compile_network(str(ck.get("source") or ""))
    # ⚠ **重建模型必须把它 `build` 时需要的东西也给它**：既然 `build` 可以收
    #   `ctx`（负载形状），checkpoint 就得存下那份 ctx —— 否则"能重建"是假的
    #   （实测：网络在 build 里读 `ctx["payload_shapes"]` 时，`eval_custom` 直接炸）。
    build_ctx = dict(ck.get("build_ctx") or {})
    if not build_ctx:
        build_ctx = {"n_in": int(ck.get("n_in") or 0),
                     "n_out": int(ck.get("n_out") or 0),
                     "payload_shapes": {}, "payload_dtypes": {},
                     "target_kind": str(ck.get("target_kind") or ""),
                     "device": "cpu", "n_rows": 0,
                     "note": "旧存档没存 build_ctx ⇒ 只给了最基本的形状事实"}
    model = call_network_build(compiled.instance, int(ck.get("n_in") or 0),
                              int(ck.get("n_out") or 0), torch.nn, ctx=build_ctx)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def describe_checkpoint(run_id: str) -> dict:
    """**只读投影**：checkpoint 在不在、有哪些文件、结构摘要（**不内联权重**）。"""
    import torch
    from ...core.persistence import RUNS_DIR

    outdir = Path(RUNS_DIR) / str(run_id) / "checkpoint"
    if not outdir.is_dir():
        return {"run_id": str(run_id), "available": False,
                "reason": "这个 run 没有 checkpoint 目录"
                          "（只有 run_custom 会写；或来自更早的版本）",
                "files": [], "tags": []}
    files = sorted(p.name for p in outdir.iterdir() if p.is_file())
    tags, params, meta = [], None, {}
    for tag in ("best", "last"):
        p = outdir / f"{tag}.pt"
        if not p.is_file():
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        tags.append(tag)
        if params is None:
            sd = ck.get("state_dict") or {}
            params = {"n_tensors": len(sd),
                      "n_params": int(sum(int(t.numel()) for t in sd.values())),
                      "has_bias": sorted(k for k, t in sd.items() if t.dim() == 1)[:8]}
            meta = {k: ck.get(k) for k in
                    ("target_kind", "n_in", "n_out", "trained_on", "tensors_on",
                     "best_epoch", "epochs_run", "source_lines")}
            meta["source_lines"] = len(str(ck.get("source") or "").splitlines())
    return {"run_id": str(run_id), "available": bool(tags),
            "reason": "" if tags else "目录在但没有 .pt 文件",
            "dir": str(outdir), "files": files, "tags": tags,
            "params": params or {}, "meta": meta,
            "note": "权重**不内联**（几十万个数不进对话）。要真用它："
                    "`eval_custom(run_id=…, code=…)` 由系统装载后交给你的代码。"}


class _PayloadMap(dict):
    """评估用的负载映射：**取不到键时给可教学的话**（不甩一个裸 `KeyError: 'xr'`）。

    反馈原文："技能里'换一个没见过的 N 评估'的示例写法会直接 `KeyError: 'xr'`。"
    —— 裸 KeyError 只说"没有这个键"，不说"你该给数据"。这里把**怎么办**写进消息，
    也把**取数的顺序**（显式 ⇒ 继承训练那份）说清楚。
    """

    def __missing__(self, key):
        raise KeyError(
            f"评估负载里没有 {key!r}（本次可用：{sorted(self) or '（空）'}）。"
            "评估用的数据按这个顺序取：① 本次显式给的 `payload` / `dataset_id`；"
            "② **默认继承训练时那份负载**（从 checkpoint 里的声明重建）。"
            "这次是空的 ⇒ 多半是**那次训练本身没给负载**，或声明里的 HDF5 已读不到。"
            "⇒ 请显式给 `dataset_id=…`（复用会话里某份数据集）或 `payload=…`"
            "（现场给新数据，例如换个没见过的 N）。")


def run_eval_code(run_id: str, code: str, *, which: str = "best",
                  payload: Any = None, dataset_id: str = "") -> dict:
    """**把训练好的模型交给调用方的代码**（NQS 需求：一个"能调用外部函数的口子"）。

    ## 为什么需要它（而不是只让 AI 取权重自己算）

    需求的原话：训练完要做三件事——**采样 4096 个构型比速度**、**算变分能量**、
    **换个没见过的 N 评估**。这三件事都要（a）**完整的模型**、（b）**他们自己的
    物理代码**。只给权重文件解决不了（a 和 b 都要一个"装载 + 调用"的地方）。

    ## 调用约定（就一条）

    代码里定义 ``def evaluate(model, ctx) -> dict``：``ctx`` 里有

    - ``payload``：逐行负载（**可以来自新数据** —— 这正是"换 N 外推评估"的用法）
    - ``meta``：checkpoint 的元信息（`n_in`/`n_out`/`target_kind`/`trained_on`…）
    - ``torch`` / ``nn`` / ``device`` / ``run_id`` / ``which``

    返回一个 ``dict``（数值会被收进回执；非数值原样保留但**不保证可序列化**）。

    ⚠ **`ctx["payload"]` 与训练路径同一个约定**：给的是**张量**（且 dtype 保住：
    复数就是复数、位串是整数、mask 是 bool），不是 numpy —— 两条路给两种东西
    是最容易咬人的不一致（实测：评估代码里写 `.float()` 会炸，因为拿到的是 ndarray）。

    ⚠ **它在权威进程里跑调用方的代码**——与 `run_custom` 同一口径：走受控命令面、
    受审计与 Gate 管辖；沙箱负责"别把宿主搞崩 + 错误可读"，**不是权限边界**
    （要 import 他们自己的物理引擎，设 `MLTB_EXTRA_PATHS` 把路径加进 sys.path）。
    """
    import torch
    from ml_mecha.sandbox import SandboxError

    try:
        model, ck = load_checkpoint(run_id, which=which)
    except FileNotFoundError as exc:
        return {"ok": False, "run_id": str(run_id), "metrics": {}, "error": str(exc),
                "error_kind": "no_checkpoint",
                "hint": "先跑一次 `run_custom`（它会写 checkpoint）；"
                        "只有自定义网络有 checkpoint"}
    ctx = {"payload": _PayloadMap(
               {str(k): _trainer._as_tensor(v, torch, torch.device("cpu"))
                for k, v in dict(payload or {}).items()}),
           "torch": torch, "nn": torch.nn,
           "device": "cpu", "run_id": str(run_id), "which": str(which),
           "meta": {k: ck.get(k) for k in
                    ("target_kind", "n_in", "n_out", "trained_on", "params",
                     "best_epoch", "epochs_run")}}
    try:
        fn = _compile_eval_code(code)
        got = fn(model, ctx)
    except SandboxError as exc:
        return {"ok": False, "run_id": str(run_id), "metrics": {},
                "error": str(exc), "error_kind": exc.kind, "hint": exc.detail}
    except Exception as exc:                                   # noqa: BLE001
        import traceback
        return {"ok": False, "run_id": str(run_id), "metrics": {},
                "error": f"{type(exc).__name__}: {exc}",
                "error_kind": "eval_code_error",
                "hint": _device_error_hint(exc) or "是你那段评估代码抛的（不是框架）",
                "traceback_tail": traceback.format_exc(limit=6)[-800:]}
    metrics = {}
    extra = {}
    if isinstance(got, dict):
        for k, v in got.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metrics[str(k)] = float(v)
            else:
                extra[str(k)] = v
    return {"ok": True, "run_id": str(run_id), "which": str(which),
            "metrics": metrics, "extra": extra, "error": "",
            "note": "这段代码在**权威进程**里跑；模型来自 checkpoint"
                    f"（{'best' if which != 'last' else 'last'}）"}


def _compile_eval_code(code: str):
    """编译评估代码并要求它定义 ``evaluate(model, ctx)``（与 `Network` 同一约定）。"""
    from ml_mecha.sandbox import SandboxError
    ns: dict = {}
    try:
        exec(compile(str(code or ""), "<ai-eval>", "exec"), ns)   # noqa: S102
    except SyntaxError as exc:
        raise SandboxError(f"评估代码有语法错误：{exc.msg}", kind="code_syntax_error",
                           detail=f"第 {exc.lineno} 行") from exc
    except Exception as exc:                                   # noqa: BLE001
        raise SandboxError(f"评估代码在定义期抛了：{type(exc).__name__}: {exc}",
                           kind="code_definition_error") from exc
    fn = ns.get("evaluate")
    if not callable(fn):
        raise SandboxError(
            "评估代码里没有 `evaluate(model, ctx)` 函数",
            kind="missing_eval_function",
            detail="契约：`def evaluate(model, ctx) -> dict`；ctx 里有 "
                   "payload / meta / torch / nn / device / run_id")
    return fn


def _hook(net, name: str):
    """取 `Network` 上的一个可选钩子；没有就回 `None`（**不假装**有）。"""
    fn = getattr(net, name, None)
    return fn if callable(fn) else None


def _build_ctx(payload, *, n_rows: int, n_features: int, n_out: int,
               target_kind: str, device: str) -> dict:
    """交给 `Network.build(..., ctx=…)` 的**形状事实**（反馈 §2-缺口1）。

    为什么需要（他的原话）："`build` 只拿得到**特征表列数**，而 NQS 的输入维度在
    **负载**里（节点特征几维、每条配置几个原子、几个采样构型）……于是只能写死维度。
    后果**不是报错，而是换一份数据时静默跑偏**。"

    ⇒ 这里给的是**事实**（形状/dtype/行数），**不含数据**（build 不该需要数据）。
    形状的含义（缺口4 的同一条）：`(R, L, d)` 里 **`R` 是行数，`(L, d)` 是"一行之内"的
    完整结构** —— 所以 `shape[1:]` 才是作者要的那一维。
    """
    shapes, dtypes = {}, {}
    for k, v in (payload or {}).items():
        try:
            import numpy as _np
            a = _np.asarray(v)
            shapes[str(k)] = list(a.shape)
            dtypes[str(k)] = str(a.dtype)
        except Exception:                                      # noqa: BLE001
            shapes[str(k)] = None
            dtypes[str(k)] = None
    return {"payload_shapes": shapes, "payload_dtypes": dtypes,
            "payload_keys": sorted(shapes),
            "n_rows": int(n_rows), "n_features": int(n_features),
            "n_out": int(n_out), "target_kind": str(target_kind),
            "device": str(device),
            "hint": "数组形状是 (R, …)：**R 是行数，… 是「一行之内」的完整结构**；"
                    "若一行是长度为 L 的向量序列，要给 (R, L, d) 而不是 (R, d)"}


def _payload_of(problem):
    """从 `Problem` 取**逐行负载**（`aux["payload"]`）⇒ 交给 trainer 逐批切给网络。

    ## 形状约定（单一真源，`prepare_dataset` 的 `payload` 声明按这个校验）

    ``aux["payload"]`` = ``{名字: 数组}``，每个数组**第 0 维 = 观测表的行数**。
    - 变长部分由调用方**先 pad 再配 mask**（我们给 `paddings.pad_rows` 那个助手）；
    - **复数/整数/bool 都保留 dtype**（波函数振幅是复数、位串是 0/1、mask 是 bool）
      —— 一律强转 float 就把物理量毁了。

    ⚠ 为什么第 0 维必须是行数：这样"切一批行"就等于"切每个数组的前几维"，
    不需要为负载再造一套索引；切分/指标/存档的既有语义**一行都不用改**。
    """
    if problem is None or not getattr(problem, "aux", None):
        return None
    pay = problem.aux.get("payload")
    if not pay:
        return None
    return {str(k): v for k, v in dict(pay).items()}


def _inline_payload_of(payload_decl, payload):
    """内联负载的**真值**（只有内联的才给；HDF5 那种靠路径重读，不进存档）。

    ⚠ 判据是**声明里有没有 `inline`**：`_jsonable_payload_decl` 会把内联 `arrays`
    转成形状/dtype 摘要并打上 `inline` 键 ⇒ 有它才说明"这份数据的真值只在内存里"。
    """
    if not isinstance(payload_decl, dict) or "inline" not in payload_decl:
        return None
    return {str(k): v for k, v in dict(payload or {}).items()}


def _save_checkpoints(*, run_id: str, res, stats: dict, source: str, params: dict,
                      target_kind: str, n_in: int, n_out: int, device: str,
                      meta: dict, payload_decl: dict | None = None,
                      build_ctx: dict | None = None,
                      inline_payload: dict | None = None) -> dict | None:
    """把**完整** `state_dict` 落盘（`best` 与 `last` 各一份）+ 结构源码 + 元信息。

    ## 为什么必须另存（不能靠 `nn_weights`）

    `nn_weights` 是**诊断产物**：只收 `dim()>=2` 的参数（`recorder.py:246`）
    ⇒ **偏置 / LayerNorm 权重根本没存**，装不回来。而需求的原话是
    "训练完我要拿模型去采样、算能量"——那个模型必须能**逐参数装回**。

    ⚠ `torch.save` 的对象是 `{"state_dict", "source", "meta"}`：源码一起存，
    所以"重建模型"这件事**不依赖你还记着那段代码**（与 `custom_network_source`
    同一思路，但这里是为了**装回权重**）。

    ⚠ 设备说明写进 meta：**存档里的张量一律在 CPU**（卡上显存要留给训练，
    且存档不该与设备绑定）⇒ 装载方自己 `.to(device)`。
    @returns ``{"dir": …, "files": [...]}``；写盘失败回 `None`（**不拖垮训练**）。
    """
    if not run_id or res is None:
        return None
    try:
        import torch
        from ...core.persistence import RUNS_DIR
        outdir = Path(RUNS_DIR) / run_id / "checkpoint"
        outdir.mkdir(parents=True, exist_ok=True)
        files = []
        payload_common = {"source": source, "params": dict(params or {}),
                          "target_kind": target_kind, "n_in": int(n_in),
                          "n_out": int(n_out), "tensors_on": "cpu",
                          "trained_on": str(device),
                          # ⭐ **训练负载的"声明"跟着 checkpoint 走**（不是数组）：
                          #   这样 `eval_custom` 默认就能**继承训练时的数据**
                          #   （反馈建议 4），跨重启也成立（声明里有 HDF5 路径就重读）。
                          #   ⚠ **大数组不进 .pt**：HDF5 那种靠**路径**重读。
                          "payload_decl": dict(payload_decl or {}),
                          # ⚠ 但**内联**负载（`payload={"arrays": …}`）的真值必须随存档走：
                          #   声明里只剩形状/dtype（声明必须 JSON 安全，见
                          #   `_jsonable_payload_decl`）⇒ 否则 `eval_custom` 的
                          #   "默认继承"对内联数据会**静默失效**。`torch.save` 能存 numpy。
                          "inline_payload": inline_payload,
                          # ⚠ **重建模型的充分条件**：`build` 可能依赖这份 ctx
                          #   （负载形状）⇒ 不存它，"能重建"就是假的（实测踩过：
                          #   网络在 build 里读 `payload_shapes` 时 eval_custom 直接炸）。
                          "build_ctx": dict(build_ctx or {}),
                          **dict(meta or {})}
        for tag, key in (("best", "best_state"), ("last", "last_state")):
            state = stats.get(key)
            if not state:
                continue
            path = outdir / f"{tag}.pt"
            torch.save({"state_dict": state, **payload_common,
                        "tag": tag, "best_epoch": stats.get("best_epoch"),
                        "epochs_run": stats.get("epochs_run")}, path)
            files.append(path.name)
        if not files:
            return None
        (outdir / "README.txt").write_text(
            "这是 run_custom 的**完整** checkpoint（含 bias/LayerNorm）。\n"
            "装载：`run_checkpoint(run_id=…)` 拿定位符；"
            "`eval_custom(from_run=…, code=…)` 由系统装载后交给你的代码。\n"
            "张量在 CPU（自己 .to(device)）；`source` 里是那段网络代码。\n",
            encoding="utf-8")
        files.append("README.txt")
        return {"dir": str(outdir), "files": files}
    except Exception:                                        # noqa: BLE001
        return None


def _register_into_host() -> None:
    """把自己登记进边界层的槽（`ml_mecha.engine`）。

    ⚠ 方向刻意是"方法层 → 边界层"，不是反过来：`ml_mecha` 顶层 import 上层会
    违反结构性纪律（`ast.walk` 静态守着）。本仓既有做法就是**注册槽**
    （同 `core.persistence` 的 opt 槽）。宿主不在场（纯库用法）⇒ 静默跳过，
    因为那时也没有 `run_custom` 命令要服务。
    """
    try:
        from ml_mecha.engine import register_custom_network
    except Exception:                                  # pragma: no cover - 纯库用法
        return
    register_custom_network(method_name=CUSTOM_METHOD_NAME, code_key=CUSTOM_CODE_KEY,
                            cls=CustomNNMethod, param_schema=CustomNNMethod.param_schema)


_register_into_host()


# 让 `evaluate` 用上与 torch_mlp 同一套指标（基类按 task 分派，这里显式写清）
def _metrics(kind: str, y_true, y_pred, y_prob=None):
    if kind == "classification":
        return classification_metrics(np.asarray(y_true), np.asarray(y_pred),
                                      y_prob=y_prob)
    return regression_metrics(np.asarray(y_true, float), np.asarray(y_pred, float))
