# -*- coding: utf-8 -*-
"""数据源抽象（接缝2 的修正）：AutoTuner 的数据不必来自 ML 工作区。

真实主场景是"接入正在运行的程序"：采集系统/仿真器持续往文件里写数据，
调参循环每次评估前拿最新的一份。文件是进程间集成的通用语。

DataSource.fetch() -> (X: DataFrame, y: Series|None)
- SpecSource：ML 工作区当前 DataSpec（内部数据集，快照式，向后兼容）；
- FileSource：csv/parquet 路径 + 目标列，mtime 变化才重读（外部程序只管写）。

零 Qt（P8）。
"""
from __future__ import annotations

import abc
import os

import pandas as pd


class DataSource(abc.ABC):
    """一次 fetch 得到评估用的 (X, y)。实现须可反复调用。"""

    @abc.abstractmethod
    def fetch(self):
        ...

    @abc.abstractmethod
    def describe(self) -> str:
        """UI 状态栏一行文案（含"是否新鲜"信息）。"""
        ...


class SpecSource(DataSource):
    """ML 工作区当前数据视图（DataSpec）的薄包装。"""

    def __init__(self, spec):
        self._spec = spec

    @property
    def spec(self):
        return self._spec

    def fetch(self):
        return self._spec.X, self._spec.y

    def describe(self) -> str:
        s = self._spec
        name = s.meta.get("dataset", "?")
        return f"内部数据集 [{name}] n={len(s.X)} 目标列={s.y.name if s.y is not None else '无'}"


class ArraySource(DataSource):
    """内存里的 (X, y) 包成 DataSource（AutoTuner 裸数组入口的兼容层）。"""

    def __init__(self, X: pd.DataFrame, y):
        self._X, self._y = X, y

    def fetch(self):
        return self._X, self._y

    def describe(self) -> str:
        return f"内存数据 n={len(self._X)}"


class FileSource(DataSource):
    """外部程序写的 csv/parquet；mtime 变了才重读（廉价轮询语义）。"""

    def __init__(self, path: str, target: str, reload_each: bool = True):
        self.path = path
        self.target = target
        self.reload_each = reload_each
        self._cache: tuple | None = None
        self._mtime: float = -1.0

    def fetch(self):
        mt = os.path.getmtime(self.path)          # 文件不存在 -> OSError，上层记 failed
        if self._cache is None or (self.reload_each and mt != self._mtime):
            df = (pd.read_parquet(self.path) if self.path.lower().endswith(".parquet")
                  else pd.read_csv(self.path))
            if self.target not in df.columns:
                raise ValueError(f"数据源缺目标列 {self.target!r}（现有：{list(df.columns)[:12]}）")
            y = df[self.target]
            X = df.drop(columns=[self.target])
            self._cache = (X, y)
            self._mtime = mt
        return self._cache

    def describe(self) -> str:
        n = len(self._cache[0]) if self._cache else "?"
        fresh = "每次评估重读" if self.reload_each else "快照"
        return f"文件 {os.path.basename(self.path)} n={n} 目标列={self.target}（{fresh}）"
