# -*- coding: utf-8 -*-
"""P2 判据：`tags=[…]` 跨族对齐必须**看得见警告**；同族请求**不受影响**（回归）。

走 `MLEngine._query_training_detail`（只读、不需要绑通道），直接用**盘上真 run**
（用户那条 `custom_network-2334125-0`：既有 epoch 级 `loss`，又有 `on_step` 的 `fid/step`）。
"""
import json
import sys
from pathlib import Path

REPO = Path(r"D:\code-nosync\Mechine learning")
sys.path.insert(0, str(REPO))

from ml_mecha.engine import MLEngine                                        # noqa: E402

RID = sys.argv[1] if len(sys.argv) > 1 else "custom_network-2334125-0"
verdicts = []


def q(**kw):
    return MLEngine._query_training_detail(None, run_id=RID, **kw)


def show(t, p):
    print(f"\n===== {t} =====")
    print(p if isinstance(p, str)
          else json.dumps(p, ensure_ascii=False, default=str, indent=2))


base = q()
show("① 基线：latest_steps / step_units / axes（不选 tags）", {
    "run_id": base["run_id"], "available": base["available"],
    "n_tags": len(base["tags"]),
    "latest_steps 头几条": dict(list(base["latest_steps"].items())[:6]),
    "step_units": base["step_units"],
    "axes": base["axes"],
})
verdicts.append(("A step_units 里 batch 级 tag 被点名（非空）",
                 bool(base["step_units"])))
verdicts.append(("B axes.per_epoch_estimate 有值（≈ 每轮批数）",
                 bool(base["axes"].get("per_epoch_estimate"))))

# ---------------------------------------------------------------- 跨族
cross = q(tags=["loss", "val_loss", "fid/step"], limit=200)
cmp_ = cross["compare"]
show("② 跨族 tags=['loss','val_loss','fid/step'] ⇒ compare 原样（截取首尾）", {
    "axis_warning": cmp_.get("axis_warning"),
    "aligned_at_step": cmp_.get("aligned_at_step"),
    "step_units": cmp_.get("step_units"),
    "n_points": cmp_.get("n_points"),
    "轴首尾": (cmp_.get("steps") or [])[:3] + ["..."] + (cmp_.get("steps") or [])[-3:],
    "loss 列前 6 个": (cmp_.get("tags", {}).get("loss") or [])[:6],
    "fid/step 列前 6 个": (cmp_.get("tags", {}).get("fid/step") or [])[:6],
})
verdicts.append(("C 跨族请求**带** axis_warning", bool(cmp_.get("axis_warning"))))
verdicts.append(("C2 警告文本同时点名 batch 与 epoch 两族",
                 all(s in str(cmp_.get("axis_warning"))
                     for s in ("batch", "epoch"))))
verdicts.append(("C3 每 tag 的 unit 也在 compare.step_units 里",
                 cmp_.get("step_units", {}).get("loss") == "epoch"
                 and cmp_.get("step_units", {}).get("fid/step") == "batch"))

# ---------------------------------------------------------------- 同族（回归）
same = q(tags=["loss", "val_loss"], limit=200)
cmp2 = same["compare"]
show("③ 同族 tags=['loss','val_loss'] ⇒ 不应被打扰（回归）", {
    "axis_warning": cmp2.get("axis_warning"),
    "aligned_at_step": cmp2.get("aligned_at_step"),
    "n_points": cmp2.get("n_points"),
    "aligned": cmp2.get("aligned"),
})
verdicts.append(("D 同族请求**没有** axis_warning（回归）",
                 not cmp2.get("axis_warning")))
verdicts.append(("D2 同族 aligned_at_step 非 None（有公共步）",
                 cmp2.get("aligned_at_step") is not None))

# ---------------------------------------------------------------- 没有任何公共步
alone = q(tags=["loss", "my/step_missing_xyz"] if "my/step_missing_xyz" in base["tags"]
          else ["loss", base["step_units"] and list(base["step_units"])[0] or "loss"],
          limit=50)
print("\n===== ④ 同族里步长完全错开的两个 tag（找一对来验'无公共步'的警告）=====")
print(json.dumps(alone["compare"].get("axis_warning"), ensure_ascii=False))

print("\n================ P2 判定汇总 ================")
for name, ok in verdicts:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"  共 {sum(1 for _, o in verdicts if o)}/{len(verdicts)} PASS")
