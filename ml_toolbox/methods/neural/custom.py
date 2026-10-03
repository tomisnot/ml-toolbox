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

from ...core.contracts import (MLMethod, PageSpec, ParamSpec, TASK_SUPERVISED,
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
    def fit(self, X, y, cfg, diag: bool = False):
        import torch

        from . import tb_log as _tb

        # ⚠ **训练超参走 `self.params(cfg)`**（= 声明默认 + 运行级覆写），
        #   不是 `cfg.extras`：覆写是经 `cfg.overrides` 进来的，而 extras 只承载
        #   run_id / device / on_progress / 代码这些**非参数**的运行级东西。
        #   实测踩过：从 extras 读 ⇒ 用户给的 `epochs=6` 被静默忽略、跑了默认 60 轮
        #   （判据里表现为"6 epoch 的训练有 60 个 loss 点"）。
        p = self.params(cfg)
        run_extras = cfg.extras or {}
        compiled = self._network()
        net = compiled.instance
        kind = self.target_kind or _infer_kind(y)
        prep = _trainer.prepare_supervised(
            X, y, kind=kind, seed=int(cfg.seed),
            val_split=float(p.get("val_split", 0.2)))
        n_in = int(prep.Xs.shape[1])
        n_out = int(prep.n_out)

        # 结构由 AI 写（沙箱编译已保证 build 存在且返回 nn.Module）
        from ml_mecha.sandbox import call_network_build
        model = call_network_build(net, n_in, n_out, torch.nn)

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
        rec.set_probe(torch.tensor(prep.Xs[probe_rows], dtype=torch.float32),
                      labels=prep.yv[probe_rows], probe_idx=probe_rows)
        self._probe_idx = probe_rows

        res = self._new_result(target_kind=kind, params={"lr": lr})
        res._torch = torch
        res._dev = torch.device("cpu")
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
                on_progress=run_extras.get("on_progress"))
        except Exception as exc:                             # noqa: BLE001
            import traceback
            res.error = ("自定义网络训练失败："
                         f"{type(exc).__name__}: {exc}\n"
                         + traceback.format_exc(limit=6))
            return res
        res.est = model
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
        if diag:
            res.diag = {"epochs": int(p.get("epochs", 60)),
                        "epochs_run": stats["epochs_run"],
                        "early_stopped_at": stats["stopped_at"],
                        "cancelled": bool(stats.get("cancelled")),
                        "best_epoch": stats.get("best_epoch"),
                        "best_val": stats.get("best_watch"),
                        "n_train": int(len(prep.tr_idx)), "n_val": int(len(prep.va_idx)),
                        "n_in": n_in, "n_out": n_out,
                        "hooks": sorted(h for h in NETWORK_HOOKS
                                        if callable(getattr(net, h, None)))}
        return res

    # ------------------------------------------------ 预测（与 TorchMLP 同构）
    def predict(self, X, result):
        torch = result._torch
        Xs = (X.to_numpy(float) - result._mu) / result._sd
        t = torch.tensor(Xs, dtype=torch.float32)
        result.est.eval()
        with torch.no_grad():
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
        t = torch.tensor(Xs, dtype=torch.float32)
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
