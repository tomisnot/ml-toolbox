# -*- coding: utf-8 -*-
"""曲面体检（N5，opt 层纯函数）：从评估历史读数，回答"该用哪个优化器"。

动机：GP-BO 的平滑假设 vs TPE 的密度比假设，各有擅长域；选错 = 白烧预算
（freeze 5D 实测：GP 提议点均值持续变差，RF CV R²=0.16 证明面不可学）。
与其凭经验猜，不如对已有历史做 4 项廉价诊断（零新评估）：

  1. 可学习性  —— RandomForest 5 折 CV R²：高 -> 光滑 -> GP 友好
  2. 局部平滑  —— 归一化域最近邻的平均 |Δy|：小 -> GP 假设成立
  3. 维度重要性—— permutation importance：定位该收窄/该搜的维度
  4. 好点边缘  —— top-k 好点在各维的取值区间：TPE 能读出的先验窗口

输出一个 tidy DataFrame（指标 / 值 / 解读），直接喂 PageSpec(kind="table")。
sklearn 惰性 import（纯后端无 sklearn 时降级为提示，不崩）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pcols(record) -> list:
    return [d["key"] for d in record.space_desc]


# 节流缓存：直播每点刷新一次，RF CV/permutation 是秒级开销，不可每点重算。
# 同 run 内 ok 点数增长 <25% 复用上次结果（跨过阈值才重算）；force 手动重算。
_CACHE: dict = {}


def surface_health(record, force: bool = False) -> pd.DataFrame:
    """评估历史 -> 体检表（列：指标 / 值 / 解读）。ok 点 <10 时只报样本不足。"""
    h = getattr(record, "history", None)
    if h is None or h.empty or "score" not in h.columns:
        return pd.DataFrame([{"指标": "样本", "值": "—", "解读": "尚无评估历史"}])
    n = int((h["status"] == "ok").sum())
    key = getattr(record, "run_id", "")
    cached = _CACHE.get(key)
    if not force and cached and n < cached[0] * 1.25:
        return cached[1]
    df = _compute(record, n)
    _CACHE[key] = (n, df)
    return df


def _compute(record, n) -> pd.DataFrame:
    rows = []
    h = record.history
    ok = h[h["status"] == "ok"]
    cols = [c for c in _pcols(record) if c in ok.columns]
    rows.append({"指标": "有效评估数", "值": n,
                 "解读": "体检需 ≥20 个 ok 点" if n < 20 else "样本充足"})
    if n < 20 or not cols:                      # 小样本：RF CV 是噪声，只给基础统计
        if n >= 1:
            rows.append({"指标": "分数范围", "值":
                         f"[{ok['score'].min():.4g}, {ok['score'].max():.4g}]",
                         "解读": "样本不足 20，昂贵诊断（可学习性/平滑性）未跑"})
        return pd.DataFrame(rows)
    X = ok[cols].to_numpy(float)
    y = ok["score"].to_numpy(float)
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=1)
        r2 = cross_val_score(rf, X, y, cv=min(5, max(2, n // 5)),
                             scoring="r2", n_jobs=1)
        m = float(np.nanmean(r2))
        rows.append({"指标": "可学习性 (RF CV R²)", "值": round(m, 3),
                     "解读": ("光滑可学 -> GP-BO 友好" if m > 0.5 else
                              "粗糙/多峰 -> GP 假设弱，宜 TPE/排序法")})
        rf.fit(X, y)
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(rf, X, y, n_repeats=10, random_state=0,
                                    n_jobs=1)
        order = np.argsort(pi.importances_mean)[::-1]
        top = cols[int(order[0])]
        rows.append({"指标": "主导维度", "值": top,
                     "解读": "敏感性最高，优先收窄其搜索区间"})
        for c, v in zip(cols, pi.importances_mean):
            rows.append({"指标": f"重要性·{c}", "值": round(float(v), 4),
                         "解读": ""})
    except Exception as e:                          # sklearn 缺失/奇异
        rows.append({"指标": "可学习性", "值": "—",
                     "解读": f"sklearn 诊断跳过：{type(e).__name__}"})
    # 局部平滑：归一化域最近邻 |Δy|
    try:
        lo, hi = X.min(0), X.max(0)
        span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
        Z = (X - lo) / span
        d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn = d2.argmin(1)
        dmin = np.sqrt(d2.min(1))
        dy = np.abs(y - y[nn])
        close = dmin < 0.1
        rows.append({"指标": "局部平滑 (近邻|Δy|)",
                     "值": round(float(dy[close].mean()) if close.any()
                                 else float(dy.mean()), 4),
                     "解读": ("近邻一致 -> GP 插值可靠"
                              if (close.any() and dy[close].mean() < 0.1)
                              else "近邻跳变 -> 光滑核失效，GP 会退化成乱撒")})
    except Exception:
        pass
    # 好点边缘：top-k 各维区间（TPE 能读的先验窗口）
    k = max(3, int(round(0.2 * n)))
    best = ok.nsmallest(k, "score")
    for c in cols:
        v = best[c].to_numpy(float)
        rows.append({"指标": f"好点区间·{c}",
                     "值": f"[{v.min():.4g}, {v.max():.4g}]",
                     "解读": f"top-{k} 好点分布（收窄候选域的依据）"})
    return pd.DataFrame(rows)
