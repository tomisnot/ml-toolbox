# -*- coding: utf-8 -*-
"""**AI 代码沙箱**：把"文本形式的 Network 定义"编译成一个可调用的对象。

## 边界声明（**诚实版**，别把它当安全沙箱读）

- 本沙箱**不是**安全边界：它在权威进程内 `exec`，代码由**用户自己所指挥的 AI**
  产出，跑在用户自己的机器上——与"用户自己写个脚本跑"的信任级别一致。
- 它**是**这三件事的边界：
  1. **归因**：编译/运行失败 ⇒ 结构化错误 + 可读原因，**不逃逸成裸栈**；
  2. **形状**：只认一个契约（`Network`），不认别的花样 ⇒ 出错点集中；
  3. **不留副作用**：代码**只经返回值交付效果**（不许自己写盘/碰 GUI/改 runs）——
     这条由 `docs/契约.md` 规定，执行侧由宿主（受控命令 + Gate）保证留痕。

⚠ **超时只覆盖"定义期"（import/编译/类体），不覆盖训练**：训练是长任务，
归 Job 面管（`submit_run` 那套）。这一点如实写在这里，不假装它能掐断训练。
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

#: 编译 AI 代码时预置的名字（`import` 是显式写的，这里只放最常用的）。
_PRELUDE = ("import math\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n")

#: 代码里必须出现的类名（约定，不是猜：契约要求一个叫 `Network` 的类）。
REQUIRED_CLASS = "Network"

#: 自定义**预处理步骤**代码里必须出现的类名（与 `Pipeline`/`Step` 契约对齐）。
#: ⚠ 与 `Network` 同一约定：名字固定 ⇒ 作者不用猜、判据也好写。
REQUIRED_STEP_CLASS = "Step"


class SandboxError(RuntimeError):
    """沙箱失败。`kind` 机读、`detail` 给人/AI 看（含可读归因）。"""

    def __init__(self, message: str, *, kind: str = "sandbox_error",
                 detail: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.detail = detail

    def teaching_text(self) -> str:
        out = f"[{self.kind}] {self}"
        if self.detail:
            out += f"\n{self.detail}"
        return out


@dataclass
class CompiledNetwork:
    """编译结果：`factory()` 造一个网络实例；`source` 原样留着（要进存档）。"""

    instance: Any
    source: str = ""
    namespace: dict = field(default_factory=dict)

    @property
    def cls(self) -> type:
        return type(self.instance)


def compile_network(code: str, *, extra_names: dict | None = None) -> CompiledNetwork:
    """把 AI 的代码编译成 `CompiledNetwork`（**只做定义期**，不训练）。

    失败一律转成 :class:`SandboxError`（带 `kind` 与可读归因）——调用方据此给出
    结构化回执，不把裸栈丢给模型。
    """
    src = str(code or "")
    if not src.strip():
        raise SandboxError("网络代码是空的", kind="empty_code",
                           detail="至少要给一个 `class Network` 定义")
    ns: dict[str, Any] = {}
    try:
        exec(_PRELUDE, ns)                                    # noqa: S102 - 见模块 docstring
        if extra_names:
            ns.update(extra_names)
        exec(compile(src, "<ai-network>", "exec"), ns)         # noqa: S102
    except SyntaxError as exc:
        raise SandboxError(
            f"网络代码有语法错误：{exc.msg}",
            kind="code_syntax_error",
            detail=f"第 {exc.lineno} 行：{(exc.text or '').rstrip()}",
        ) from exc
    except Exception as exc:                                   # noqa: BLE001
        raise SandboxError(
            f"网络代码在**定义期**就抛了：{type(exc).__name__}: {exc}",
            kind="code_definition_error",
            detail="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        ) from exc

    cls = ns.get(REQUIRED_CLASS)
    if cls is None:
        raise SandboxError(
            f"代码里没有 `{REQUIRED_CLASS}` 类",
            kind="missing_network_class",
            detail=f"契约要求定义一个名为 `{REQUIRED_CLASS}` 的类；"
                   f"当前代码里定义了：{sorted(k for k, v in ns.items() if isinstance(v, type))}")
    if not isinstance(cls, type):
        raise SandboxError(f"`{REQUIRED_CLASS}` 不是类（是 {type(cls).__name__}）",
                           kind="network_not_a_class")
    if not callable(getattr(cls, "build", None)):
        raise SandboxError(
            f"`{REQUIRED_CLASS}` 缺 `build(self, n_in, n_out, *, nn)` 方法",
            kind="missing_build",
            detail=f"实际有的是：{sorted(a for a in dir(cls) if not a.startswith('_'))}")
    try:
        instance = cls()
    except Exception as exc:                                   # noqa: BLE001
        raise SandboxError(
            f"`{REQUIRED_CLASS}()` 构造失败：{type(exc).__name__}: {exc}",
            kind="network_init_error",
            detail="构造函数不要参数、也不要在 `__init__` 里做重活（模型在 build 里建）",
        ) from exc
    return CompiledNetwork(instance=instance, source=src, namespace=ns)


def call_network_build(net: Any, n_in: int, n_out: int, nn: Any,
                       ctx: Any = None):
    """调 `Network.build`，失败归一化成 :class:`SandboxError`。

    ## `ctx`：让 `build` 拿得到**负载的形状**（不再靠猜维度）

    实测缺口（NQS 使用者的反馈 §2-缺口1）：`build(n_in, n_out)` 只给得出**特征表列数**，
    而真实输入维度常在**逐行负载**里（几个原子、每原子几维、几个采样）⇒ 作者只能
    **写死维度**，后果是"换一份数据（换个 N、加一维特征）时**静默跑偏**"，
    而那正是本仓最忌的形态（假绿）。

    ⇒ `build` 可以多收一个关键字参数 `ctx`，里面是**形状事实**（不含数据）：

    ```python
    def build(self, n_in, n_out, *, nn, ctx=None):
        d_node = ctx["payload_shapes"]["xr"][-1]      # (R, L, d) → d
        return GNN(d_node=d_node)
    ```

    ⚠ **老签名照样能跑**：只写 `(n_in, n_out, *, nn)` 时 `TypeError` 落到旧调用
    —— 与 `on_progress` 同一个做法（"新签名优先、老签名也认"，不逼既有代码改）。
    """
    def _bad(exc: BaseException):
        return SandboxError(
            f"`Network.build` 抛了：{type(exc).__name__}: {exc}",
            kind="build_error",
            detail="build 必须返回一个 `nn.Module`；"
                   "参数是 (n_in, n_out, *, nn, ctx=None)",
        )

    try:
        model = net.build(int(n_in), int(n_out), nn=nn, ctx=ctx)
    except TypeError as exc:
        if "ctx" not in str(exc):
            raise _bad(exc) from exc
        try:                                        # 老签名：不认 ctx
            model = net.build(int(n_in), int(n_out), nn=nn)
        except Exception as exc2:                   # noqa: BLE001
            raise _bad(exc2) from exc2
    except Exception as exc:                                   # noqa: BLE001
        raise _bad(exc) from exc
    if not isinstance(model, nn.Module):
        raise SandboxError(
            f"`Network.build` 返回的不是 nn.Module（是 {type(model).__name__}）",
            kind="build_wrong_type",
            detail="要 `return nn.Sequential(...)` 或自定义 nn.Module 的实例")
    return model


# ---------------------------------------------------------------- 预处理步骤
def compile_step(code: str) -> CompiledNetwork:
    """把 AI 的代码编译成**预处理步骤类**（供 `core.pipeline` 装配）。

    与 `compile_network` 同一形状——分工也一样：core 负责"声明 → 管道"的机械层，
    本函数负责"接触执行"的那一步。

    @returns :class:`CompiledNetwork`，但 **`instance` 这里放的是类本身**（不是实例）：
        因为步骤的实例由 core 在装配时按声明造（要传 params/enabled）。
    """
    src = str(code or "")
    if not src.strip():
        raise SandboxError("步骤代码是空的", kind="empty_code",
                           detail=f"至少要给一个 `class {REQUIRED_STEP_CLASS}` 定义")
    ns: dict[str, Any] = {}
    try:
        exec(_PRELUDE, ns)                                    # noqa: S102 - 见模块 docstring
        # ⚠ 把基类递进命名空间：步骤作者要写 `class Step(StepBase)`。
        #   不给的话"必须继承 Step"这条纪律就**不可满足**（作者得自己猜 import 路径），
        #   而 core 又要靠继承关系做校验 ⇒ 这里注入是最短的路。
        from ml_toolbox.core.pipeline import Step as _StepBase
        ns["StepBase"] = _StepBase
        exec(compile(src, "<ai-step>", "exec"), ns)            # noqa: S102
    except SyntaxError as exc:
        raise SandboxError(
            f"步骤代码有语法错误：{exc.msg}",
            kind="code_syntax_error",
            detail=f"第 {exc.lineno} 行：{(exc.text or '').rstrip()}",
        ) from exc
    except Exception as exc:                                   # noqa: BLE001
        raise SandboxError(
            f"步骤代码在**定义期**就抛了：{type(exc).__name__}: {exc}",
            kind="code_definition_error",
            detail="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        ) from exc

    cls = ns.get(REQUIRED_STEP_CLASS)
    if cls is None:
        defined = sorted(k for k, v in ns.items() if isinstance(v, type))
        raise SandboxError(
            f"代码里没有 `{REQUIRED_STEP_CLASS}` 类",
            kind="missing_step_class",
            detail=f"契约要求一个名为 `{REQUIRED_STEP_CLASS}` 的类，且要继承 `StepBase`"
                   f"（已注入）：`class {REQUIRED_STEP_CLASS}(StepBase): ...`。"
                   f"当前定义的类型：{defined}",
        )
    return CompiledNetwork(instance=cls, source=src, namespace=ns)
