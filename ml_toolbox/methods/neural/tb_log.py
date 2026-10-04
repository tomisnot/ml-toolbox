# -*- coding: utf-8 -*-
"""TensorBoard 桥件：把 ``TrainingRecorder`` 的中间量逐 epoch 送进事件文件。

## 为什么单独立一个文件

``TrainingRecorder`` 是**纯采集器**（docstring：零 Qt、产出契约化 artifacts）；它不该知道
TensorBoard 的存在。本模块只做一件事：**把已采集的量翻译成 TensorBoard 的轴**。
⇒ 采集与呈现分开，任一侧换实现都不牵动另一侧。

## ⚠ 它必须是【可选】的（三处各自 fail-safe）

1. **包不在**：``tensorboard`` 是 ``neural`` extra 的一部分，未装时**训练照常**，本模块全部空转；
2. **写失败**：事件写失败**绝不能拖垮训练**（与 ``finalize`` 里那句"采集失败不拖垮主结果"同一条纪律）；
3. **关掉**：``tb_enabled`` 为假时只做一次属性查询，不 import。
⇒ 三处都返回"什么都不做"，而不是抛错 —— 但**写失败要留一条痕**（见 ``note_error``），
不静默吞掉（否则"图是空的"会变成没人能解释的现象）。

## 轴映射（对应 TensorBoard 的原生入口）

| 采集到的 | 送去哪 | 入口 |
|---|---|---|
| ``history.loss`` / ``val_loss`` / ``lr`` / ``grad_norm`` / ``dead_relu`` | SCALARS | ``add_scalar`` |
| ``grads[层名]`` 逐 epoch 每层梯度范数 | SCALARS（每层一条 tag） | ``add_scalar('grad/层名')`` |
| ``weights.last[层名]`` 权重矩阵 | HISTOGRAMS / DISTRIBUTIONS | ``add_histogram`` |
| ``latent`` 帧（同坐标系） | IMAGES | ``add_image`` |
| ``nn_layers`` 逐层健康表 | TEXT（Markdown） | ``add_text`` |
| 权重的起始/末态/更新量 | SCALARS（范数随时间） | ``add_scalar`` |

⚠ **本条不"自己画图"**：图的形状由 TensorBoard 决定，我们只保证**数据按轴正确送达**。
"""

from __future__ import annotations

import numpy as np

#: 事件目录名（相对 ``runs/<run_id>/``）。与 `record.json`/`artifacts.npz` 并列。
TB_DIRNAME = "tb"

#: 五个核心标量（同时是最重要的人类可读信号）。
CORE_SCALARS = ("loss", "val_loss", "lr", "grad_norm", "dead_relu")


def layout_groups(layer_names) -> dict:
    """标量分组的**纯函数**：``{类别: {图名: ["Multiline", [tag…]]}}``。

    ⚠ 这个形状就是 **torch** 的 `SummaryWriter.add_custom_scalars` 的入参。
    **不要走 protobuf 那条路**：`tensorboard.summary.v1.custom_scalar` 在新版里
    `import tensorflow.compat.v1`（本机只有 tensorboard、没有 tensorflow ⇒ 实测直接
    `ModuleNotFoundError`）；而 torch 这个封装**不需要 tensorflow**。

    ⚠ 分组依据是"**读的人要判什么**"，不是"数据长什么样"：
    - **训练动态** —— 一次训练是否健康（收敛 / 稳定 / 调度）；
    - **逐层梯度** —— 梯度是**逐层衰减**还是**爆炸**（这必须一层层看）；
    - **权重范数** —— 权重在**长**还是在**僵**（first/last/update 同图才看得出）。

    ⚠ **不按 tag 字典序排**：那会把 `loss` 和 `grad/0.weight` 排在一起，读的人得自己找。
    这是"帮人判"与"列清单"的区别。
    """
    names = [str(n) for n in (layer_names or ())]
    layout: dict = {
        "训练动态": {
            "损失": ["Multiline", ["loss", "val_loss"]],
            "稳定性": ["Multiline", ["grad_norm", "dead_relu"]],
            "学习率": ["Multiline", ["lr"]],
        },
    }
    if names:
        layout["逐层梯度范数（逐层衰减 / 爆炸）"] = {
            n: ["Multiline", [f"grad/{n}"]] for n in names}
    layout["权重范数（在长还是在僵）"] = {
        "首 / 末 / 更新": ["Multiline", ["weights_norm/first",
                                        "weights_norm/last",
                                        "weights_norm/update"]],
    }
    return layout


CONFIG_SUMMARY_TAG = "custom_scalars__config__"
PLUGIN_NAME = "custom_scalars"


def _layout_summary(layout: dict):
    """把 layout 打成 TB 能**真正读到**的那条 summary。

    ## 为什么不直接用现成入口（两条路都实测不通）

    | 路 | 为什么不行 |
    |---|---|
    | `torch` 的 `SummaryWriter.add_custom_scalars` | ⚠ 它把 layout 塞进 **`tensor.string_val`**，而 TB 2.19 读的是 **`plugin_data.content`** ⇒ **布局被静默丢掉**（实测：写进去了、读回 0 个类别） |
    | `tensorboard.summary.v1.custom_scalar` | ⚠ 内部 `import tensorflow.compat.v1` ⇒ 本机只有 tensorboard、没有 tensorflow ⇒ `ModuleNotFoundError` |

    ⇒ 所以按 TB 2.19 的**权威读法**自己造：`plugin_data.plugin_name="custom_scalars"`
    + **`plugin_data.content = Layout.SerializeToString()`**（纯 proto，**不需要 tensorflow**）。
    读法依据：`tensorboard/backend/event_processing/data_provider.py` 与
    `event_accumulator.py` 都取 `summary_metadata.plugin_data.content`。
    """
    from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
    from tensorboard.plugins.custom_scalar import layout_pb2

    categories = []
    for cat_title, charts in layout.items():
        pb_charts = []
        for chart_title, meta in charts.items():
            kind, tags = meta[0], list(meta[1])
            if kind == "Margin":
                content = layout_pb2.MarginChartContent(series=[
                    layout_pb2.MarginChartContent.Series(
                        value=tags[0], lower=tags[1], upper=tags[2])])
                pb_charts.append(layout_pb2.Chart(title=chart_title,
                                                  margin=content))
            else:
                content = layout_pb2.MultilineChartContent(tag=tags)
                pb_charts.append(layout_pb2.Chart(title=chart_title,
                                                  multiline=content))
        categories.append(layout_pb2.Category(title=cat_title, chart=pb_charts))

    blob = layout_pb2.Layout(category=categories).SerializeToString()
    return Summary(value=[Summary.Value(
        tag=CONFIG_SUMMARY_TAG,
        metadata=SummaryMetadata(plugin_data=SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=blob)),
    )])


class TensorBoardLog:
    """逐 epoch 写 TensorBoard 的单向桥。**永不抛**（写失败只记账）。"""

    def __init__(self, run_id: str, root, enabled: bool = True) -> None:
        self.run_id = str(run_id or "")
        self.root = root
        self.enabled = bool(enabled) and bool(self.run_id)
        self._writer = None
        self._tried = False
        self._stack: list[str] = []

    # ------------------------------------------------------------ 内部
    def _ensure(self):
        """惰性建 writer（只试一次）。**不吞异常**：失败记进 ``_stack`` 并返回 None。"""
        if not self.enabled or self._tried:
            return self._writer
        self._tried = True
        try:
            from torch.utils.tensorboard import SummaryWriter
            import os
            logdir = os.path.join(str(self.root), self.run_id, TB_DIRNAME)
            os.makedirs(logdir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=logdir)
        except Exception as exc:                    # 包不在 / 建目录失败 / 写权限
            self.note_error("init", exc)
            self._writer = None
        return self._writer

    def note_error(self, where: str, exc: BaseException) -> None:
        """记一条写失败（**留痕，不静默**）。"""
        self._stack.append(f"{where}: {type(exc).__name__}: {exc}")

    def _safe(self, where: str, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            self.note_error(where, exc)

    # ------------------------------------------------------------ 对外
    @property
    def errors(self) -> tuple[str, ...]:
        """写失败记录（空 = 一路顺利）。给判据/诊断用。"""
        return tuple(self._stack)

    @property
    def active(self) -> bool:
        """writer 是否真的建起来了（判据用；不建 writer 也能过）。"""
        return self._writer is not None

    def log_epoch(self, ep: int, history: dict, grads: dict,
                  extra: dict | None = None) -> None:
        """一个 epoch 的标量：5 条核心 + 每层梯度范数 + **调用方自定义 tag**。

        ## `extra`：让"方法自己上报想看的指标"直接落到观测面

        需求来源（NQS 用户）："训练时只有一个总 loss，我分不清是哪块在拖后腿"——
        他要 NLL/相位/保真度**各一条线**，还要**按物理通道、按分布内/外推分组**。

        ⚠ **写进来就白拿**：`describe_training`（AI 侧）读的是事件文件里**全部 scalar
        tag** ⇒ 这里多写一条，AI 那边立刻能看见，**查询面一行都不用改**。
        分组的表达方式就是 **tag 命名约定**：`fid/ch0`、`fid/ood`、`nll/phase`…
        （TensorBoard 的 tag 本来就分层，不必另造结构。）
        """
        w = self._ensure()
        if w is None:
            return
        step = int(ep)
        for key in CORE_SCALARS:
            seq = history.get(key) or []
            if len(seq) > step:
                v = float(seq[step])
                if np.isfinite(v):                  # ⚠ val_loss 可能是 NaN（无验证集）
                    self._safe(f"scalar/{key}", w.add_scalar, key, v, step)
        for name, seq in (grads or {}).items():
            if len(seq) > step:
                v = float(seq[step])
                if np.isfinite(v):
                    self._safe(f"scalar/grad/{name}", w.add_scalar,
                               f"grad/{name}", v, step)
        # ⭐ 自定义 tag（方法层每轮上报）。**非有限值不入盘**：与上面同一条纪律
        #   —— 不在图里制造 NaN 假点（"看着有这条线"比"没有"更容易骗人）。
        for tag, value in (extra or {}).items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                self._safe(f"scalar/{tag}", w.add_scalar, str(tag), v, step)

    def log_weights(self, weights: dict | None, step: int) -> None:
        """权重直方图 + 权重的起始/末态/更新量范数（随时间各一条标量）。"""
        w = self._ensure()
        if w is None or not isinstance(weights, dict):
            return
        for name, mat in (weights.get("last") or {}).items():
            if mat is None:
                continue
            arr = np.asarray(mat, float).ravel()
            if arr.size:
                self._safe(f"hist/{name}", w.add_histogram, f"weights/{name}", arr, int(step))
        for tag, key in (("weights_norm/first", "first"),
                         ("weights_norm/last", "last"),
                         ("weights_norm/update", "update")):
            block = weights.get(key) or {}
            total = 0.0
            seen = False
            for mat in block.values():
                if mat is None:
                    continue
                total += float(np.linalg.norm(np.asarray(mat, float)))
                seen = True
            if seen:
                self._safe(f"scalar/{tag}", w.add_scalar, tag, total, int(step))

    def log_latent_frames(self, frames, epochs, *, size: int = 256) -> None:
        """把 PCA 投影帧当图送进 IMAGES（逐 epoch 一帧，同一坐标系 ⇒ 位移可解释）。

        ⚠ 需要 matplotlib；不在时**静默跳过**（这是"锦上添花"那一档，不该拖垮别的）。
        """
        w = self._ensure()
        if w is None or frames is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        arr = np.asarray(frames, float)
        if arr.ndim != 3:
            return
        eps = list(epochs) if epochs is not None else list(range(arr.shape[0]))
        for i in range(arr.shape[0]):
            try:
                fig, ax = plt.subplots(figsize=(2.6, 2.6), dpi=100)
                ax.scatter(arr[i, :, 0], arr[i, :, 1], s=5, alpha=0.7)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"epoch {eps[i]}", fontsize=7)
                fig.tight_layout(pad=0.2)
                fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
                plt.close(fig)
                self._safe(f"image/latent/{i}", w.add_image,
                           "latent/frame", buf, int(eps[i]), dataformats="HWC")
            except Exception as exc:
                self.note_error(f"image/latent/{i}", exc)
                try:
                    plt.close("all")
                except Exception:
                    pass

    def log_layers_table(self, table, step: int = 0) -> None:
        """逐层健康表 → TEXT（Markdown；TensorBoard 的 TEXT 面板支持表格）。"""
        w = self._ensure()
        if w is None or table is None:
            return
        try:
            md = table.to_markdown()
        except Exception:
            try:
                md = "```\n" + str(table) + "\n```"
            except Exception:
                return
        self._safe("text/layers", w.add_text, "layers/health", md, int(step))

    def write_layout(self, layer_names) -> None:
        """把标量**排好组**写进事件文件（TensorBoard 的 custom scalars 布局 API）。

        为什么值得写这一下：不排的话，TensorBoard 按 tag 字典序铺 20 张图，
        `loss` 和 `grad/0.weight` 混在一起，读的人要自己找。
        排完打开就是「训练动态 / 逐层梯度 / 权重范数」三组，**顺序即优先级**。

        ⚠ 布局是**可选增强**：`custom_scalar` 不可用时静默跳过（不拖垮，也不假装）。
        """
        w = self._ensure()
        if w is None:
            return
        try:
            # ⚠ **不要用 `w.add_custom_scalars(...)`**：它把 layout 写进 tensor.string_val，
            #   而 TB 读 plugin_data.content ⇒ 布局会被静默丢掉（实测）。
            #   这里走自己造的那条 summary（见 `_layout_summary` 的取证）。
            fw = w._get_file_writer()
            fw.add_summary(_layout_summary(layout_groups(layer_names)), 0)
            fw.flush()
        except Exception as exc:
            self.note_error("layout", exc)

    def close(self) -> None:
        """收尾：flush + close。**关闭日志本身可读**（TensorBoard 读文件，不读进程）。"""
        w = self._writer
        self._writer = None
        if w is None:
            return
        try:
            w.flush()
        except Exception as exc:
            self.note_error("flush", exc)
        try:
            w.close()
        except Exception as exc:
            self.note_error("close", exc)
