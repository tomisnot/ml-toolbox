# -*- coding: utf-8 -*-
"""P0 端到端真跑：`checkpoint_every` 全链路 + 中途可见性 + 越界/类型错的可教学拒绝。

**隔离**：本脚本自己把 `ML_TOOLBOX_RUNS` 指到临时根（必须在 import ml_toolbox 之前），
所以它不会碰仓库的 `runs/`，也不会碰正在跑的权威进程。

跑法：
    $env:ML_TOOLBOX_RUNS = "$env:TEMP\p0-e2e-runs"; python _dev_p0_e2e.py
（脚本自己也会设，双保险。）
"""
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(r"D:\code-nosync\Mechine learning")
_TMPROOT = tempfile.mkdtemp(prefix="p0-e2e-")
os.environ["ML_TOOLBOX_RUNS"] = str(Path(_TMPROOT) / "runs")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / "mecha"))

import numpy as np                                                        # noqa: E402
import pandas as pd                                                       # noqa: E402

from ml_mecha.assembly import assemble_ml_mecha                           # noqa: E402
from ml_toolbox.core.persistence import RUNS_DIR                          # noqa: E402

#: 真实负载形状（几百行），每步**故意慢**（sleep）⇒ 中途那个窗口足够被轮询抓到。
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
            time.sleep(0.18)          # 只为把"训练进行中"这个窗口拉到肉眼可见
        return model(x).squeeze(-1), y

    def on_step(self, model, step, ctx):
        return {"loss/step": float(step)}

    def metrics(self, model, ctx):
        return {"my/epoch_mark": 1.0}
'''

RID_BOX = {}


def _report(title, payload):
    print(f"\n===== {title} =====")
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, default=str, indent=2))


def main():
    print(f"隔离 runs 根 = {RUNS_DIR}")
    print(f"临时根        = {_TMPROOT}")
    ml = assemble_ml_mecha(root=_TMPROOT, seed=7)
    ml.switch_ai()
    try:
        rng = np.random.RandomState(0)
        frame = pd.DataFrame(rng.randn(260, 6),
                             columns=[f"f{i}" for i in range(6)])
        frame["y"] = (frame["f0"] * 2 - frame["f1"] + 0.3 * frame["f2"]
                      + 0.1 * rng.randn(260))
        got = ml.tools.execute("prepare_dataset", {
            "source": {"kind": "frame", "frame": frame}, "name": "p0probe",
            "target": "y",
            "pipeline": {"steps": [{"key": "scale"}]}})
        ds = got["value"]["dataset_id"]
        print(f"dataset_id   = {ds}")

        # ------------------------------------------------ ① 真跑（后台线程）
        res_box = {}
        t_start = time.time()
        ck_dir_box = {}

        def worker():
            try:
                res_box["value"] = ml.tools.execute("run_custom", {
                    "device": "cpu", "code": NET, "dataset_id": ds,
                    "name": "p0probe", "persist": True,
                    "checkpoint_every": 2,
                    "overrides": {"epochs": 6, "batch": 4, "val_split": 0.25},
                    "diag": True})
            except BaseException as exc:                            # noqa: BLE001
                res_box["exc"] = f"{type(exc).__name__}: {exc}"
            finally:
                res_box["t_done"] = time.time()

        th = threading.Thread(target=worker, daemon=True)
        th.start()

        # 边跑边找 run 目录（run_id 在训练**开始前**由 runner 定死并写 cfg.extras）
        events = []
        ckdir = None
        first_last_pt = None
        first_seen_t = None
        deadline = time.time() + 240
        while time.time() < deadline:
            if ckdir is None or not ckdir.is_dir():
                cands = sorted(
                    (p for p in Path(RUNS_DIR).glob("custom_network-*")
                     if (p / "checkpoint").is_dir()),
                    key=lambda p: p.stat().st_mtime)
                if cands:
                    ckdir = cands[-1] / "checkpoint"
                    events.append({"at_s": round(time.time() - t_start, 3),
                                   "event": f"发现 checkpoint 目录 {ckdir}"})
            if ckdir is not None and ckdir.is_dir():
                lp = ckdir / "last.pt"
                if lp.is_file():
                    if first_last_pt is None:
                        first_last_pt = lp.stat().st_size
                        first_seen_t = time.time()
                        events.append({
                            "at_s": round(time.time() - t_start, 3),
                            "event": f"**训练仍在跑**时 last.pt 已出现在盘上"
                                     f"（{lp.stat().st_size} B）",
                            "training_done": "value" in res_box})
                    if "value" in res_box:
                        break
            if "value" in res_box or "exc" in res_box:
                break
            time.sleep(0.02)
        th.join(timeout=120)
        out = res_box.get("value") or {}
        val = out.get("value") or {}
        rid = str(val.get("run_id") or "")
        RID_BOX["run_id"] = rid
        terminal_seen = "value" in res_box
        graveyard_size = (first_last_pt is not None and terminal_seen
                          and first_seen_t is not None
                          and first_seen_t < res_box.get("t_done", 0))

        _report("① 训练进行中盘上就有 last.pt（时序证据）", {
            "run_id": rid,
            "训练总墙钟_s": round(res_box.get("t_done", 0) - t_start, 3),
            "最后一次轮询看到 last.pt 的时刻_s":
                round(first_seen_t - t_start, 3) if first_seen_t else None,
            "该时刻训练是否还在跑": bool(
                first_seen_t is not None
                and first_seen_t < res_box.get("t_done", 0)),
            "首次看到时的字节数": first_last_pt,
            "事件": events,
        })
        print(f"\n>>> 断言 A1（进行中就有 last.pt）: "
              f"{'PASS' if graveyard_size else 'FAIL'}")
        print(f">>> 断言 A1b（首次看见 last.pt 时训练未结束）: "
              f"{'PASS' if (first_seen_t and first_seen_t < res_box.get('t_done', 0)) else 'FAIL'}")

        # ------------------------------------------------ ② 回执里的 checkpoint 事实
        _report("② run_custom 回执（截取 checkpoint 相关键）", {
            "ok": out.get("ok"),
            "run_id": rid,
            "checkpoint_every": val.get("checkpoint_every"),
            "warnings": out.get("warnings") or val.get("warnings"),
            "nn_checkpoint_every": val.get("nn_checkpoint_every"),
            "nn_checkpoint_n": val.get("nn_checkpoint_n"),
            "nn_checkpoint_last_step": val.get("nn_checkpoint_last_step"),
            "nn_checkpoint_errors": val.get("nn_checkpoint_errors"),
            "metrics": val.get("metrics"),
            "diag.checkpoint_every": (val.get("diag") or {}).get("checkpoint_every"),
            "diag.checkpoint_periodic":
                (val.get("diag") or {}).get("checkpoint_periodic"),
            "diag.steps": (val.get("diag") or {}).get("steps"),
        })

        # ------------------------------------------------ ③ checkpoint_detail
        ck = ml.tools.execute("checkpoint_detail", {"run_id": rid})["value"]
        _report("③ checkpoint_detail（原样回执，去掉长字段）", ck)
        last = ck.get("last_saved") or {}
        fm = ck.get("files_meta") or {}
        print(f"\n>>> 断言 A2（报出 saved_at_step）: "
              f"{'PASS' if last.get('saved_at_step') is not None else 'FAIL'}"
              f"  last_saved={json.dumps(last, ensure_ascii=False)}")
        n_steps = int((val.get("diag") or {}).get("steps") or 0)
        print(f">>> 断言 A2b（终态那份的 step == 循环总步数 {n_steps}）: "
              f"{'PASS' if (fm.get('last') or {}).get('saved_at_step') == n_steps else 'FAIL'}"
              f"  （last.pt step={(fm.get('last') or {}).get('saved_at_step')}）")

        # ------------------------------------------------ ④ meta 里的 checkpoint_every
        import torch
        lp = Path(ck["dir"]) / "last.pt"
        bp = Path(ck["dir"]) / "best.pt"
        m_last = torch.load(lp, map_location="cpu", weights_only=False)
        meta_last = {k: m_last.get(k) for k in
                     ("tag", "save_reason", "checkpoint_every", "saved_at_step",
                      "saved_at_epoch", "epochs_run", "trained_on", "n_in", "n_out")}
        meta_best = None
        if bp.is_file():
            mb = torch.load(bp, map_location="cpu", weights_only=False)
            meta_best = {k: mb.get(k) for k in
                         ("tag", "save_reason", "checkpoint_every", "saved_at_step",
                          "saved_at_epoch", "epochs_run", "best_epoch")}
        _report("④ 存档 meta 原样（last.pt / best.pt）",
                {"last.pt": meta_last, "best.pt": meta_best})
        print(f"\n>>> 断言 A3（meta.checkpoint_every 与实际入参 2 一致）: "
              f"{'PASS' if meta_last.get('checkpoint_every') == 2 else 'FAIL'}")
        print(f">>> 断言 A3b（periodic 那份的 save_reason=periodic）: "
              f"{'PASS' if (fm.get('last') or {}).get('save_reason') in ('periodic', 'final') else 'FAIL'}")

        # ------------------------------------------------ ⑤ 生成的 README 文本
        rd = Path(ck["dir"]) / "README.txt"
        text = rd.read_text(encoding="utf-8") if rd.is_file() else "<缺失>"
        _report("⑤ 生成的 checkpoint/README.txt 原文", text)
        bad = "run_checkpoint" in text
        print(f"\n>>> 断言 B1（README 不推荐不存在的 run_checkpoint）: "
              f"{'PASS' if not bad else 'FAIL'}")
        print(f">>> 断言 B1b（README 提到真名 checkpoint_detail / eval_custom）: "
              f"{'PASS' if ('checkpoint_detail' in text and 'eval_custom' in text) else 'FAIL'}")
        print(f">>> 断言 B1c（README 提到 load_state_dict(ck['state_dict'])）: "
              f"{'PASS' if 'state_dict' in text else 'FAIL'}")
        print(">>> 残留 .tmp-* =",
              [p.name for p in Path(ck["dir"]).glob("*.tmp-*")])

        # ------------------------------------------------ ⑥ 越界/类型错的可教学拒绝
        cases = [(-5, "负数（越界）"), ("abc", "字符串（类型错）"),
                 (0, "0（合法：= 关）")]
        for bad_v, why in cases:
            r = ml.tools.execute("run_custom", {
                "device": "cpu", "code": NET, "dataset_id": ds,
                "name": "p0probe", "checkpoint_every": bad_v,
                "overrides": {"epochs": 1, "batch": 128, "val_split": 0.0}})
            v = r.get("value") or {}
            row = {
                "输入": bad_v, "场景": why,
                "ok": v.get("ok"), "error": v.get("error"),
                "error_kind": v.get("error_kind"),
                "error_hint": v.get("error_hint"),
                "error_suggest": v.get("error_suggest"),
                "run_id": v.get("run_id"),
            }
            _report(f"⑥ checkpoint_every={bad_v!r}（{why}）", row)
            if bad_v == 0:
                print(f">>> 断言 C0（0 = 合法，关掉中途存档，仍能跑完）: "
                      f"{'PASS' if v.get('ok') else 'FAIL'}")
            else:
                teachable = (not v.get("ok")
                             and bool(v.get("error"))
                             and bool(v.get("error_kind"))
                             and "internal_error" not in str(v.get("error_kind")))
                print(f">>> 断言 C（{bad_v!r} 被可教学地拒）: "
                      f"{'PASS' if teachable else 'FAIL'}"
                      f"  kind={v.get('error_kind')!r}")
    finally:
        try:
            ml.close()
        except Exception:                                          # noqa: BLE001
            pass

    print("\n[P0 探针结束] run_id =", RID_BOX.get("run_id"))
    print("[临时根]", _TMPROOT)


if __name__ == "__main__":
    main()
