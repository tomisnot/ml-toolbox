# -*- coding: utf-8 -*-
"""P0 补充：`checkpoint_every` 的越界/类型错**到底怎么被拒的**。

v2 探针看到 `-5` / `"abc"` 既不抛 `MechaError`、`value` 也是空 dict
⇒ 必须把**原样回执**打出来，判断这是"探针取错了字段"还是"真的没有可教学内容"。
本探针走**三条路**，逐条打印原样（`repr`）：
  ① `ParamSpec.clean`（单一真源，看它给不给可教学三件套）；
  ② `engine.run({...})`（命令层回执，不经工具层转换）；
  ③ `ml.tools.execute(...)`（AI 真正看到的那一层，含异常对象的 info）。
"""
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(r"D:\code-nosync\Mechine learning")
TD = tempfile.mkdtemp(prefix="p0-reject-")
os.environ["ML_TOOLBOX_RUNS"] = str(Path(TD) / "runs")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / "mecha"))

import numpy as np                                                        # noqa: E402
import pandas as pd                                                       # noqa: E402

from ml_mecha.assembly import assemble_ml_mecha                           # noqa: E402
from ml_toolbox.methods.neural.custom import CustomNNMethod               # noqa: E402

NET = '''
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.f = nn.Linear(n_in, 4)
        self.h = nn.Linear(4, 1)

    def forward(self, x):
        return self.h(torch.tanh(self.f(x)))


class Network:
    def build(self, n_in, n_out, *, nn, ctx=None):
        return Net(n_in)

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def forward_step(self, model, batch, ctx):
        x, y = batch
        return model(x).squeeze(-1), y
'''


def show(title, payload):
    print(f"\n===== {title} =====")
    print(payload if isinstance(payload, str)
          else json.dumps(payload, ensure_ascii=False, default=repr, indent=2))


def main():
    spec = {p.key: p for p in CustomNNMethod.param_schema}
    ps = spec["checkpoint_every"]
    print(f"checkpoint_every: kind={ps.kind!r} default={ps.default!r} "
          f"min={ps.min!r} max={ps.max!r}")
    show("① ParamSpec.clean 逐个取值（单一真源）", {
        str(v): (lambda: (_clean(ps, v)))() for v in (-5, "abc", 0, 2, 1e9, None)
    })

    ml = assemble_ml_mecha(root=TD, seed=7)
    ml.switch_ai()
    try:
        rng = np.random.RandomState(1)
        frame = pd.DataFrame(rng.randn(40, 3), columns=list("abc"))
        frame["y"] = frame["a"] - frame["b"]
        ds = ml.tools.execute("prepare_dataset", {
            "source": {"kind": "frame", "frame": frame}, "name": "rej",
            "target": "y", "pipeline": {"steps": [{"key": "scale"}]}
        })["value"]["dataset_id"]

        for bad in (-5, "abc"):
            base = {"command": "run_custom", "code": NET, "dataset_id": ds,
                    "name": "rej", "checkpoint_every": bad,
                    "overrides": {"epochs": 1, "batch": 128, "val_split": 0.0}}
            # ② 命令层原样回执
            try:
                got = ml.engine.run(dict(base), context=None)
                show(f"② engine.run(checkpoint_every={bad!r}) 回执原样",
                     {k: got.get(k) for k in sorted(got)} if isinstance(got, dict)
                     else repr(got))
            except BaseException as exc:                              # noqa: BLE001
                show(f"② engine.run(checkpoint_every={bad!r}) 抛异常", {
                    "type": type(exc).__name__, "msg": str(exc)[:500],
                    "kind": getattr(exc, "kind", None),
                    "hint": getattr(exc, "hint", None),
                    "suggest": getattr(exc, "suggest", None)})
            # ③ 工具层（⚠ 不能带 `command`：那是给 `engine.run` 的键，
            #   工具参数面没有它 ⇒ 会红在 `missing_arguments`，**与本次要验的事无关**）
            try:
                got = ml.tools.execute("run_custom", {
                    k: v for k, v in base.items() if k != "command"})
                show(f"③ tools.execute(checkpoint_every={bad!r}) 返回", repr(got)[:1500])
            except BaseException as exc:                              # noqa: BLE001
                show(f"③ tools.execute(checkpoint_every={bad!r}) 抛异常", {
                    "type": type(exc).__name__, "msg": str(exc)[:600],
                    "kind": getattr(exc, "kind", None),
                    "hint": getattr(exc, "hint", None),
                    "suggest": getattr(exc, "suggest", None),
                    "info": repr(getattr(exc, "info", None))[:600]})
    finally:
        ml.close()
    print("\n[临时根]", TD)


def _clean(ps, v):
    try:
        return {"ok": True, "value": ps.clean(v)}
    except BaseException as exc:                                      # noqa: BLE001
        return {"ok": False, "type": type(exc).__name__, "msg": str(exc)[:300],
                "kind": getattr(exc, "kind", None),
                "hint": getattr(exc, "hint", None),
                "suggest": getattr(exc, "suggest", None)}


if __name__ == "__main__":
    main()
