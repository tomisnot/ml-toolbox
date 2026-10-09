# -*- coding: utf-8 -*-
"""P0 端到端真跑 v2：`checkpoint_every` 全链路 + 中途可见性 + 越界/类型错的可教学拒绝。

v1 的两处探针 bug（**记录在案**）：
  1. 把 `ml.tools.execute(...)` 的回执当成 `{"value": {...}}`——对**成功**路径是对的，
     但**失败**路径工具层会抛 `MechaError`（`tools._raise_on_failure`，设计如此：
     "工具声明了参数却静默 ok=False" 是它要防的病）⇒ `r.get("value")` 是 None，
     于是三条 C 断言报 FAIL——**红的是探针，不是被测对象**（pitfalls A13）。
  2. `diag.steps` 要从 `value.diag` 取（v1 取到了 None）⇒ A2b 也跟着假红。

隔离：`ML_TOOLBOX_RUNS` 指向临时根（必须在 import ml_toolbox 之前设）。
"""
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(r"D:\code-nosync\Mechine learning")
_TMPROOT = tempfile.mkdtemp(prefix="p0-e2e-v2-")
os.environ["ML_TOOLBOX_RUNS"] = str(Path(_TMPROOT) / "runs")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / "mecha"))

import numpy as np                                                        # noqa: E402
import pandas as pd                                                       # noqa: E402
from mecha.errors import MechaError                                       # noqa: E402

from ml_mecha.assembly import assemble_ml_mecha                           # noqa: E402
from ml_toolbox.core.persistence import RUNS_DIR                          # noqa: E402

NET = '''
import time

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.f = nn.Linear(n_in, 12)
        self.h = nn.Linear(12, 1)

    def forward(self, x):
        return self.h(torch.tanh(self.f(x)))


class Network:
    primary_metric = "loss/step"

    def build(self, n_in, n_out, *, nn, ctx=None):
        return Net(n_in)

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def forward_step(self, model, batch, ctx):
        x, y = batch
        if ctx.get("kind") == "train":
            time.sleep(0.18)
        return model(x).squeeze(-1), y

    def on_step(self, model, step, ctx):
        return {"loss/step": float(step)}

    def metrics(self, model, ctx):
        return {"my/epoch_mark": 1.0}
'''


def show(title, payload):
    print(f"\n===== {title} =====")
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, default=str, indent=2))


def call_tool(ml, name, args):
    """调一次工具；**失败也把可教学三件套带出来**（不吞）。"""
    try:
        r = ml.tools.execute(name, args)
        base = dict(r) if isinstance(r, dict) else {"raw": r}
        return {"raised": None, "receipt": base,
                "value": (r or {}).get("value") if isinstance(r, dict) else None}
    except MechaError as exc:
        return {"raised": "MechaError",
                "kind": getattr(exc, "kind", None),
                "hint": getattr(exc, "hint", None),
                "suggest": getattr(exc, "suggest", None),
                "message": str(exc),
                "value": None}
    except BaseException as exc:                                  # noqa: BLE001
        return {"raised": type(exc).__name__, "message": str(exc), "value": None}


def main():
    print(f"隔离 runs 根 = {RUNS_DIR}")
    print(f"临时根        = {_TMPROOT}")
    verdicts = []
    ml = assemble_ml_mecha(root=_TMPROOT, seed=7)
    ml.switch_ai()
    try:
        rng = np.random.RandomState(0)
        frame = pd.DataFrame(rng.randn(260, 6), columns=[f"f{i}" for i in range(6)])
        frame["y"] = (frame["f0"] * 2 - frame["f1"] + 0.3 * frame["f2"]
                      + 0.1 * rng.randn(260))
        got = call_tool(ml, "prepare_dataset", {
            "source": {"kind": "frame", "frame": frame}, "name": "p0probe",
            "target": "y", "pipeline": {"steps": [{"key": "scale"}]}})
        ds = got["value"]["dataset_id"]
        print(f"dataset_id   = {ds}")

        # ---------------------------------------------------- ① 真跑（后台线程）
        box = {}
        t0 = time.time()

        def worker():
            box["r"] = call_tool(ml, "run_custom", {
                "device": "cpu", "code": NET, "dataset_id": ds,
                "name": "p0probe", "persist": True, "checkpoint_every": 2,
                "overrides": {"epochs": 6, "batch": 4, "val_split": 0.25},
                "diag": True})
            box["t_done"] = time.time()

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        events, ckdir = [], None
        first_seen, first_bytes = None, None
        while time.time() < t0 + 300:
            if ckdir is None:
                cands = sorted(
                    (p for p in Path(RUNS_DIR).glob("custom_network-*")
                     if (p / "checkpoint").is_dir()),
                    key=lambda p: p.stat().st_mtime)
                if cands:
                    ckdir = cands[-1] / "checkpoint"
                    events.append({"at_s": round(time.time() - t0, 3),
                                   "event": f"发现 checkpoint 目录 {ckdir}"})
            if ckdir is not None and (ckdir / "last.pt").is_file() and first_seen is None:
                first_seen = time.time()
                first_bytes = (ckdir / "last.pt").stat().st_size
                events.append({"at_s": round(first_seen - t0, 3),
                               "event": f"**训练仍在跑**时 last.pt 已在盘上"
                                        f"（{first_bytes} B）",
                               "训练是否已结束": "r" in box})
            if "r" in box:
                break
            time.sleep(0.02)
        th.join(timeout=180)

        res = box["r"]
        val = res["value"] or {}
        rid = str(val.get("run_id") or "")
        t_done = box.get("t_done", time.time())

        show("① 训练进行中盘上就有 last.pt（时序证据）", {
            "run_id": rid,
            "训练总墙钟_s": round(t_done - t0, 3),
            "首次看见 last.pt 的时刻_s": round(first_seen - t0, 3) if first_seen else None,
            "该时刻训练是否还在跑": bool(first_seen and first_seen < t_done),
            "首次看见时的字节数": first_bytes,
            "事件": events,
        })
        ok_mid = bool(first_seen and first_seen < t_done)
        verdicts.append(("A1 训练**进行中**盘上就有 last.pt", ok_mid))

        # ---------------------------------------------------- ② run_custom 回执
        show("② run_custom 回执（checkpoint 相关键，成功路径取 `.value`）", {
            "raised": res["raised"],
            "ok": val.get("ok"),
            "run_id": rid,
            "checkpoint_every": val.get("checkpoint_every"),
            "metrics": val.get("metrics"),
            "nn_checkpoint_every": val.get("nn_checkpoint_every"),
            "nn_checkpoint_n": val.get("nn_checkpoint_n"),
            "nn_checkpoint_last_step": val.get("nn_checkpoint_last_step"),
            "nn_checkpoint_errors": val.get("nn_checkpoint_errors"),
            "diag": {k: v for k, v in (val.get("diag") or {}).items()
                     if k in ("steps", "epochs_run", "batch", "checkpoint_every",
                              "checkpoint_periodic", "device")},
        })
        diag = val.get("diag") or {}
        show("② b 盘上 record.json 里 checkpoint 相关产物（存档自己说的话）", {
            "nn_checkpoint_every": val.get("nn_checkpoint_every"),
            "nn_checkpoint_n": val.get("nn_checkpoint_n"),
            "nn_checkpoint_last_step": val.get("nn_checkpoint_last_step"),
            "diag.checkpoint_periodic": diag.get("checkpoint_periodic"),
            "diag.steps": diag.get("steps"),
        })

        # ---------------------------------------------------- ③ checkpoint_detail
        ck = call_tool(ml, "checkpoint_detail", {"run_id": rid})["value"] or {}
        show("③ checkpoint_detail（原样；权重不内联）", ck)
        last = ck.get("last_saved") or {}
        fm = ck.get("files_meta") or {}
        verdicts.append(("A2 checkpoint_detail 报出 saved_at_step（非 None）",
                         last.get("saved_at_step") is not None))
        n_steps = int(diag.get("steps") or 0)
        verdicts.append(
            (f"A2b 终态 last.pt 的 saved_at_step == 循环总步数 {n_steps}",
             (fm.get("last") or {}).get("saved_at_step") == n_steps))

        # ---------------------------------------------------- ④ 存档 meta
        import torch
        ckdir = Path(ck["dir"])
        m_last = torch.load(ckdir / "last.pt", map_location="cpu", weights_only=False)
        m_best = torch.load(ckdir / "best.pt", map_location="cpu", weights_only=False)
        keys = ("tag", "save_reason", "checkpoint_every", "saved_at_step",
                "saved_at_epoch", "epochs_run", "best_epoch", "trained_on",
                "n_in", "n_out", "tensors_on")
        show("④ 存档 meta 原样", {
            "last.pt": {k: m_last.get(k) for k in keys},
            "best.pt": {k: m_best.get(k) for k in keys},
        })
        verdicts.append(("A3 meta.checkpoint_every == 入参 2",
                         m_last.get("checkpoint_every") == 2))
        verdicts.append(("A4 last.pt 的 state_dict 含 bias（完整存档）",
                         any(str(k).endswith("bias")
                             for k in (m_last.get("state_dict") or {}))))

        # ---------------------------------------------------- ⑤ README 原文
        text = (ckdir / "README.txt").read_text(encoding="utf-8")
        show("⑤ 生成的 checkpoint/README.txt 原文", text)
        verdicts.append(("B1 README 里没有不存在的 `run_checkpoint`",
                         "run_checkpoint" not in text))
        verdicts.append(("B1b README 用真名 checkpoint_detail / eval_custom",
                         "checkpoint_detail" in text and "eval_custom" in text))
        verdicts.append(("B1c README 指明 load 用 ck['state_dict']",
                         "state_dict" in text))
        print(">>> 残留 .tmp-* =", [p.name for p in ckdir.glob("*.tmp-*")])
        verdicts.append(("B2 原子写没留临时文件",
                         not list(ckdir.glob("*.tmp-*"))))

        # ---------------------------------------------------- ⑥ 越界/类型错
        for bad_v, why, want_reject in ((-5, "负数（越界）", True),
                                        ("abc", "字符串（类型错）", True),
                                        (0, "0（合法：= 关）", False)):
            got = call_tool(ml, "run_custom", {
                "device": "cpu", "code": NET, "dataset_id": ds,
                "name": "p0probe", "checkpoint_every": bad_v,
                "overrides": {"epochs": 1, "batch": 128, "val_split": 0.0}})
            row = {"输入": bad_v, "场景": why,
                   "raised": got.get("raised"), "kind": got.get("kind"),
                   "hint": got.get("hint"), "suggest": got.get("suggest"),
                   "message": (got.get("message") or "")[:400],
                   "回执 ok": (got.get("value") or {}).get("ok"),
                   "回执 error": (got.get("value") or {}).get("error"),
                   "回执 run_id": (got.get("value") or {}).get("run_id")}
            show(f"⑥ checkpoint_every={bad_v!r}（{why}）", row)
            if want_reject:
                taught = (got.get("raised") == "MechaError"
                          and bool(got.get("kind"))
                          and got.get("kind") not in ("internal_error", "command_failed")
                          and bool(got.get("hint")))
                verdicts.append((f"C checkpoint_every={bad_v!r} 被**可教学地**拒"
                                 f"（kind 具体 + hint 有话说）", taught))
            else:
                verdicts.append(("C0 checkpoint_every=0 合法（关掉中途存档仍跑完）",
                                 bool((got.get("value") or {}).get("ok"))))
    finally:
        try:
            ml.close()
        except Exception:                                          # noqa: BLE001
            pass

    print("\n================ P0 判定汇总 ================")
    for name, ok in verdicts:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  共 {sum(1 for _, o in verdicts if o)}/{len(verdicts)} PASS")
    print("[临时根]", _TMPROOT)


if __name__ == "__main__":
    main()
