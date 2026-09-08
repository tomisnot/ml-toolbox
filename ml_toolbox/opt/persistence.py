# -*- coding: utf-8 -*-
"""优化运行持久化：runs/<run_id>/{opt_record.json, history.csv}。

与 ML 侧的差异（docs/优化定位.md §5）：ML 存数值工件 npz；优化存
**追加式评估历史 CSV**——历史本身就是表格资产（回流 ML 做响应面，接缝3），
csv 比 npz 更通用（Excel/pandas 直读）。
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from .contracts import OptRecord

_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
RUNS_DIR = os.environ.get("ML_TOOLBOX_OPT_RUNS", os.path.join(_root, "runs"))


def save_record(rec: OptRecord) -> str:
    d = os.path.join(RUNS_DIR, rec.run_id)
    os.makedirs(d, exist_ok=True)
    meta = {
        "run_id": rec.run_id, "optimizer": rec.optimizer,
        "objective": rec.objective, "space": rec.space_desc,
        "budget": rec.budget, "seed": rec.seed,
        "best": rec.best, "n_evals": len(rec.history),
        "elapsed": round(rec.elapsed, 3), "error": rec.error,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "kind": "opt",
    }
    with open(os.path.join(d, "opt_record.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    if len(rec.history):
        rec.history.to_csv(os.path.join(d, "history.csv"), index=False)
    return d


def load_record(run_id: str) -> OptRecord:
    import pandas as pd
    with open(os.path.join(RUNS_DIR, run_id, "opt_record.json"),
              encoding="utf-8") as f:
        meta = json.load(f)
    hp = os.path.join(RUNS_DIR, run_id, "history.csv")
    hist = (pd.read_csv(hp) if os.path.exists(hp) else pd.DataFrame())
    return OptRecord(run_id=meta["run_id"], optimizer=meta["optimizer"],
                     objective=meta["objective"],
                     space_desc=meta.get("space", []),
                     budget=meta.get("budget", {}), seed=meta.get("seed", 42),
                     history=hist, best=meta.get("best"),
                     elapsed=meta.get("elapsed", 0.0), error=meta.get("error"))


def list_records() -> list[dict]:
    """只列优化运行（record.json 与 opt_record.json 互不混淆）。"""
    if not os.path.isdir(RUNS_DIR):
        return []
    out = []
    for rid in sorted(os.listdir(RUNS_DIR), reverse=True):
        p = os.path.join(RUNS_DIR, rid, "opt_record.json")
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    out.append(json.load(f))
            except Exception:
                continue
    return out
