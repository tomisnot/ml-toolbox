# -*- coding: utf-8 -*-
"""数据集：单一数据入口（G3）。所有方法消费同一份 Dataset，方法层不读文件。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """原始数据 + 目标列声明。清洗/特征/划分全部交给 Pipeline。"""
    frame: pd.DataFrame
    name: str = "dataset"
    target: Optional[str] = None
    time_col: Optional[str] = None          # 时序任务的时间列
    source: str = ""                        # 来源描述（文件路径 / 生成器名）
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------ 构造
    @classmethod
    def load(cls, path: str, target: Optional[str] = None,
             time_col: Optional[str] = None, **read_kw) -> "Dataset":
        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv", ".txt", ".tsv"):
            sep = read_kw.pop("sep", "," if ext != ".tsv" else "\t")
            df = pd.read_csv(path, sep=sep, **read_kw)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path, **read_kw)
        elif ext in (".parquet", ".pq"):
            df = pd.read_parquet(path, **read_kw)
        elif ext == ".json":
            df = pd.read_json(path, **read_kw)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        return cls(frame=df, name=os.path.splitext(os.path.basename(path))[0],
                   target=target, time_col=time_col, source=path)

    @classmethod
    def from_arrays(cls, X, y=None, name="dataset",
                    time=None) -> "Dataset":
        df = pd.DataFrame(np.asarray(X),
                          columns=[f"f{i}" for i in range(np.asarray(X).shape[1])])
        frame = df
        if y is not None:
            frame["target"] = np.asarray(y)
        ds = cls(frame=frame, name=name,
                 target="target" if y is not None else None)
        if time is not None:
            ds.time_col = "time"
            ds.frame.insert(0, "time", np.asarray(time))
        return ds

    # ------------------------------------------------ 基本检视
    def columns(self) -> list:
        return list(self.frame.columns)

    def feature_columns(self) -> list:
        drop = {self.target, self.time_col} - {None}
        return [c for c in self.frame.columns if c not in drop]

    def profile(self) -> dict:
        """数据画像（处理链第一张卡片的摘要来源）。"""
        df = self.frame
        num = df.select_dtypes(include=[np.number])
        cat = df.select_dtypes(exclude=[np.number])
        return {
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "n_numeric": int(num.shape[1]),
            "n_categorical": int(cat.shape[1]),
            "n_missing": int(df.isna().sum().sum()),
            "target": self.target,
            "target_kind": self._target_kind(),
        }

    def _target_kind(self) -> Optional[str]:
        if not self.target or self.target not in df_cols(self.frame):
            return None
        from .contracts import infer_kind    # 任务判定单一来源（M10）
        return infer_kind(self.frame[self.target])


def df_cols(frame) -> list:
    return list(frame.columns)
