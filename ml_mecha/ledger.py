# -*- coding: utf-8 -*-
"""**身份台账**：让"哪份数据 / 哪个作业"在**重启之后**仍然对得上号。

## 它解决什么（用户实测，出现频率最高的一条）

使用者原话（《长任务编排与失败可读性》§3.4）："重启后 `dataset_id` 全失效，**必须
重新 `prepare_dataset`**。本轮我因此**重复登记了七八次**。更麻烦的是：每次登记
**返回新 id**，于是'上一轮用的哪份数据'在会话里丢失，只能靠**我的记忆**对齐。"

⇒ 两个毛病，两处修：

1. **每次返回新 id** ⇒ 现在 id 是**内容寻址**的：`ds-<fp12>`，`fp` 由
   （来源声明 ⊕ 管道声明 ⊕ 目标列 ⊕ 目标类型 ⊕ 图/逐行负载声明 ⊕ 行数）算出。
   **同样的数据必然得到同一个 id**（幂等），于是"上一轮用的哪份数据"永远对得上。
2. **重启即清零** ⇒ 台账**落盘**（`<工作区>/.ml-mecha/ml_ledger/datasets.jsonl`，
   追加式 + 一行一条），装配时读回；可复原的来源（csv / synthetic）**懒复原**
   ——重启后第一次引用那个 id 时按声明重建，**不必重新登记**。

## 为什么 id 里**不带**用户起的名字（而用 `label` 承担可读性）

用户想要"id 能读出含义"（例如 `ds-001-口径对照-N8-随机留出150`）。但**名字进 id
就破坏幂等**：同一份数据换个名字登记，就会裂成两个 id——那正是本条要消灭的病。
⇒ id 只承载**内容身份**（`ds-<fp12>`，可机器对齐、可跨会话比对），
可读含义由 `label`（用户给的名字）与 `aliases`（同一份数据用过的所有名字）承担，
两者都随回执与 `describe_dataset` 一起给。**机器对齐靠 id，人读靠 label。**

## 为什么是 jsonl 而不是数据库

规模是几十~几千条（一次会话里几次 prepare）⇒ 追加式 jsonl + 内存折叠足够，
且**没有新依赖**。读侧只认"最后一条同 fp 的记录"（追加式 ⇒ 天然是
last-write-wins，不需要回改历史行）。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: 台账文件名（一个文件一类实体；今天只有数据集，作业/链随后续批次进来）。
DATASETS_FILE = "datasets.jsonl"

#: 写入台账的**单个**声明最大序列化体积（字节）。超了就不存声明（只存事实），
#: 并如实标 `restorable: false`——**宁可说不能复原，也不要存半份**。
MAX_DECL_BYTES = 65536


def canonical(obj: Any) -> str:
    """规范 JSON：键排序 + 紧凑分隔 ⇒ **同一份内容必然同一串**（指纹的前提）。

    ⚠ `default=str` 是**兜底**而不是常规路径：指纹要让"同一份数据"稳定，
    所以调用方应先把声明整理成 JSON 原生类型（见 `engine._source_decl`）。
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=str)


def fingerprint(decl: Any) -> str:
    """内容指纹 = sha1(规范 JSON) 前 12 位（与 `Pipeline.fingerprint()` 同一口径）。"""
    return hashlib.sha1(canonical(decl).encode("utf-8")).hexdigest()[:12]


class Ledger:
    """工作区台账（追加式 jsonl）。**写失败不吞**：身份错了比没有身份更贵。"""

    def __init__(self, root: str | os.PathLike | None = None) -> None:
        self.root = Path(root) if root else None
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        self._loaded = False

    # ---------------------------------------------------------------- 读
    @property
    def path(self) -> Path | None:
        return (self.root / DATASETS_FILE) if self.root is not None else None

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        path = self.path
        if path is None or not path.is_file():
            return
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue          # 半行（进程被杀）⇒ 跳过，不让它毒化整个台账
                    key = str(rec.get("fp_decl") or "")
                    if not key:
                        continue
                    prev = self._cache.get(key)
                    if prev is not None:
                        # 追加式 ⇒ 后写覆盖；`aliases` / `registrations` 要**累加**
                        merged = dict(prev)
                        merged.update(rec)
                        merged["aliases"] = sorted(set(prev.get("aliases") or [])
                                                   | set(rec.get("aliases") or []))
                        rec = merged
                    self._cache[key] = rec
        except OSError:
            return

    def lookup(self, fp_decl: str) -> dict | None:
        """按**声明指纹**查一条台账记录（``None`` = 没见过这份数据）。"""
        with self._lock:
            self._load()
            rec = self._cache.get(str(fp_decl))
            return dict(rec) if rec else None

    def all(self) -> list[dict]:
        with self._lock:
            self._load()
            return [dict(r) for r in self._cache.values()]

    # ---------------------------------------------------------------- 写
    def remember(self, *, fp_decl: str, dataset_id: str, label: str,
                 facts: Mapping[str, Any], restore: Mapping[str, Any] | None,
                 at: str) -> dict:
        """记一条登记（**追加**）。同 fp 再登记 ⇒ 只累加 `aliases`/`registrations`。"""
        with self._lock:
            self._load()
            key = str(fp_decl)
            prev = self._cache.get(key) or {}
            aliases = sorted(set(prev.get("aliases") or []) | {str(label)})
            rec = {
                "at": str(at),
                "fp_decl": key,
                "dataset_id": str(dataset_id),
                "label": str(label),
                "aliases": aliases,
                "first_seen": str(prev.get("first_seen") or at),
                "registrations": int(prev.get("registrations") or 0) + 1,
                "n_rows": int(facts.get("n_rows") or 0),
                "n_features": int(facts.get("n_features") or 0),
                "target": facts.get("target") or "",
                "target_kind": facts.get("target_kind") or "",
                "pipeline_id": facts.get("pipeline_id") or "",
                "source": facts.get("source") or "",
                "split_policy": dict(facts.get("split_policy") or {}),
                "chain": list(facts.get("chain") or []),
                #: 复原所需的一切（缺失/过大 ⇒ `restorable: false`，**如实**）
                "restore": dict(restore or {}),
                "restorable": bool((restore or {}).get("restorable")),
                "restore_note": str((restore or {}).get("note") or ""),
            }
            self._cache[key] = rec
            path = self.path
            if path is None:
                return dict(rec)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return dict(rec)


def decl_fits(decl: Any) -> bool:
    """声明是否小到可以存进台账（超了就只存事实 + `restorable: false`）。"""
    try:
        return len(canonical(decl).encode("utf-8")) <= MAX_DECL_BYTES
    except (TypeError, ValueError):
        return False


__all__ = ["Ledger", "canonical", "fingerprint", "decl_fits",
           "DATASETS_FILE", "MAX_DECL_BYTES"]
