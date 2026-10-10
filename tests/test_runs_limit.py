# -*- coding: utf-8 -*-
"""``core.runs.list_meta(limit=…)`` 的判据：**有限额的结果必须等于全量结果的前 N 条**。

## 为什么需要它（实测现场，2026-10-10）

监控面板那一页（``/runs``）**只要最新几十条**，而旧实现把 ``runs/`` 下每一条 ``record.json``
都读一遍再切片：真实仓（数千个目录）实测 **16.6 秒** ⇒ 页面等于打不开。修法是"按目录 mtime
预筛 + 只读够用的那些"（0.5 秒级）。

⚠ 但"按 mtime 预筛"是一个**近似**：目录 mtime 与记录里的 ``saved_at`` 同源却不保证同序
（同一秒内多次落盘、或有人拷/碰过文件）。⇒ 这条判据钉住**结果等价**（留了余量），
而不是钉住实现细节；余量不足时它会红。

## 三组

- **A 等价**：随机 `saved_at` + **故意打乱 mtime**（让两种顺序不一致）⇒ 各种 limit 下
  `list_meta(limit=k) == list_meta()[:k]`。
- **B 边界**：`limit=0` ⇒ 空；`limit > 总数` ⇒ 全量；目录不存在 ⇒ 空；坏 `record.json` 跳过。
- **C 不退化（红证）**：把候选余量压成 1（模拟"只看 mtime 最新的一条"）⇒ 在打乱 mtime 的
  存档上**必须**与全量结果不一致（证明 A 不是空转）。
"""
from __future__ import annotations

import json
import os
import random
import time

from ml_toolbox.core import runs


def _mk(root, n: int, *, shuffle_mtime: bool = False) -> None:
    """造 n 条存档：``saved_at`` 与目录 mtime 顺序**故意不一致**。"""
    os.makedirs(root, exist_ok=True)
    for i in range(n):
        rid = f"run-{i:03d}"
        d = os.path.join(root, rid)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "record.json"), "w", encoding="utf-8") as f:
            json.dump({"saved_at": f"2026-01-01T00:{i:02d}:00", "method": f"m{i}"}, f)
    if shuffle_mtime:
        #: 把 mtime 打乱（越新的 saved_at 越可能拿到更旧的 mtime）
        for name in os.listdir(root):
            p = os.path.join(root, name)
            stamp = random.random() * 1_000_000
            os.utime(p, (stamp, stamp))


def test_limited_equals_full_prefix(tmp_path):
    """A ⭐ 各种 limit 下都与"全量前 N 条"逐条相同（即使 mtime 顺序被打乱）。"""
    root = str(tmp_path / "runs")
    _mk(root, 60, shuffle_mtime=True)
    full = runs.list_meta(root, "record.json")
    assert len(full) == 60
    for k in (1, 5, 17, 59, 60, 61, 200):
        got = runs.list_meta(root, "record.json", k)
        assert [m["run_id"] for m in got] == [m["run_id"] for m in full[:k]], (
            f"limit={k} 的有限额结果与全量前 {k} 条不一致（预筛余量不够）")
    #: 未给 limit ⇒ 与旧行为完全相同（全量、新→旧）
    assert [m["run_id"] for m in runs.list_meta(root, "record.json")] == \
           [m["run_id"] for m in full]


def test_limit_boundaries_and_broken_records(tmp_path):
    """B 边界：0 / 超总数 / 目录不存在 / 坏记录。"""
    root = str(tmp_path / "runs")
    _mk(root, 8)
    assert runs.list_meta(root, "record.json", 0) == []
    assert len(runs.list_meta(root, "record.json", 8)) == 8
    assert len(runs.list_meta(root, "record.json", 999)) == 8
    assert runs.list_meta(str(tmp_path / "不存在"), "record.json", 5) == []
    #: 坏记录**跳过**（旧行为），不是整表失败
    bad = os.path.join(root, "run-bad")
    os.makedirs(bad, exist_ok=True)
    with open(os.path.join(bad, "record.json"), "w", encoding="utf-8") as f:
        f.write("{ 不是 JSON")
    got = runs.list_meta(root, "record.json", 999)
    assert "run-bad" not in [m["run_id"] for m in got]
    assert len(got) == 8


def test_red_proof_margin_matters(tmp_path, monkeypatch):
    """C ⛔ 红证：余量压成 1（只看 mtime 最新那一条）时，**必须**与全量不一致。"""
    root = str(tmp_path / "runs")
    random.seed(7)
    _mk(root, 40, shuffle_mtime=True)
    full = [m["run_id"] for m in runs.list_meta(root, "record.json")]
    monkeypatch.setattr(runs, "_LIST_META_MARGIN", 1)
    monkeypatch.setattr(runs, "_LIST_META_FLOOR", 1)
    naive = [m["run_id"] for m in runs.list_meta(root, "record.json", 5)]
    assert naive != full[:5], (
        "余量压到 1 竟然还一致 ⇒ 这条判据没有在测「预筛余量」（存档的 mtime 没被打乱？）")
    #: 恢复真余量 ⇒ 又一致（证明差别只来自余量）
    monkeypatch.undo()
    assert [m["run_id"] for m in runs.list_meta(root, "record.json", 5)] == full[:5]
