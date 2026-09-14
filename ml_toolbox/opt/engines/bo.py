# -*- coding: utf-8 -*-
"""GP-BO：高斯过程贝叶斯优化（自研，复用 sklearn GPR——接缝1 的实证）。

流程（经典 SMBO）：
    1. 前 n_init 次随机预热（GP 需要初始观测）；
    2. 每次 ask：用已观测点拟合 GP（Matern 3/2 + 白噪声）→
       在 [0,1]^d 上最大化采集函数（EI/UCB/PI）选下一候选；
    3. tell：记录观测，更新信念。

采集函数内部优化 = 两段式（低成本够用）：
    粗：先验均匀采 n_candidates 点算 acq；
    细：取 Top-k 邻域撒小扰动局部精修。
高维（>6）时该策略会退化，属已知边界（数模标定通常 ≤5 维）。

失败观测（status=failed，score=inf）不进 GP——censored，只留在历史里。
"""
from __future__ import annotations

import numpy as np

from ..contracts import Optimizer, ParamSpec
from ..registry import register
from .acquisition import ACQUISITIONS


@register
class GPBO(Optimizer):
    name = "gp_bo"
    display_name = "贝叶斯优化 GP"
    family = "bo"
    tags = ("sample-efficient", "surrogate", "expensive")
    supports_surrogate = True        # UI 代理切片页据此显示（不问私有字段）
    param_schema = [
        ParamSpec("acq", "采集函数", "select", "ei",
                  choices=["ei", "ucb", "pi"],
                  hint="EI 均衡 / UCB 偏探索(beta) / PI 偏利用"),
        ParamSpec("xi", "探索偏置 ξ", "number", 0.01, min=0.0, max=0.5,
                  hint="EI/PI 的改进余量，越大越激进"),
        ParamSpec("beta", "UCB 系数", "number", 2.0, min=0.0, max=10.0),
        ParamSpec("n_init", "随机预热次数", "int", 8, min=2, max=50,
                  hint="GP 拟合前的纯随机评估数"),
        ParamSpec("n_candidates", "候选池", "int", 2000, min=200, max=20000),
        ParamSpec("constrain", "约束感知(cEI)", "bool", False,
                  hint="黑盒有硬约束时开启：额外 GP 分类器建模可行域，"
                       "采集函数=EI×P(可行)，避免在不可行区浪费评估"),
    ]

    def setup(self, space, seed, cfg, budget):
        super().setup(space, seed, cfg, budget)
        self._X: list[np.ndarray] = []      # [0,1]^d 观测坐标
        self._y: list[float] = []
        self._Xc: list[np.ndarray] = []     # 约束分类样本（可行=1/不可行=0）
        self._yc: list[int] = []
        self._gpr = None
        self._gpc = None                     # 可行域分类器（constrain 开启时）
        self._fallback_n = 0                 # 观测不足时随机退化的计数

    # ------------------------------------------------ 主接口
    def ask(self):
        v = self._next_points(1)
        return v[0] if v else self.space.sample(self._rng)

    def ask_batch(self, n: int):
        """并行候选：预热期 = n 个随机点；之后 = 采集函数 top-k 贪心多样化。

        不做多样化会让 n 个并行点全挤在采集函数同一个峰上（GP 视角下
        它们几乎等价），信息量 ≈ 1。贪心策略：按 acq 降序排队，逐个入选，
        但拒绝与已入选点距离 < min_dist 的候选（在 [0,1]^d 归一化坐标下）。
        """
        n = max(int(n), 1)
        pts = self._next_points(n)
        while len(pts) < n:                     # 采集失败兜底：随机补齐
            pts.append(self.space.sample(self._rng))
        return pts

    def warm_tell(self, params, score, status="ok"):
        # GP 的信念 = 纯观测集合，不依赖 ask 状态 -> 历史点直接走 tell 摄入
        # （ok 进 GP、infeasible 进约束分类器、failed censored）。
        self.tell(params, score, status)

    def _next_points(self, n: int):
        """-> list[dict]，长度 ≤ n（不足由调用方随机兜底）。"""
        constrain = bool(self.cfg.get("constrain", False))
        if constrain:
            # 约束模式：可行点可能极稀疏，不能等"凑够 n_init 个可行点"（会永远
            # 停在随机预热）。只要 目标GP 有≥2可行点 + 分类器两类都见过，就开跑。
            ready = (len(self._X) >= 2 and len(self._Xc) >= 4
                     and len(set(self._yc)) >= 2)
            if not ready:
                return [self.space.sample(self._rng) for _ in range(n)]
        elif len(self._X) < max(int(self.cfg.get("n_init", 8)), 2):
            return [self.space.sample(self._rng) for _ in range(n)]
        vs = self._optimize_acq(k=n)
        if vs is None:                          # GP 拟合失败 -> 交给调用方随机兜底
            self._fallback_n += 1
            return []
        return [self.space.from_vector(v) for v in vs]

    def tell(self, params, score, status="ok"):
        # 并行批次（ask_batch）与种群式会传 list；单点 ask 传 dict
        if isinstance(params, list):
            for p, s, st in zip(params, score,
                                status if isinstance(status, list)
                                else [status] * len(params)):
                self.tell(p, s, st)
            return
        v = self.space.to_vector(params)
        if status == "ok" and np.isfinite(score):
            self._X.append(v)
            self._y.append(float(score))
        # 约束感知：可行/不可行都进分类器（infeasible 是有用信息）
        if bool(self.cfg.get("constrain", False)):
            if status == "ok":
                self._Xc.append(v); self._yc.append(1)
            elif status == "infeasible":
                self._Xc.append(v); self._yc.append(0)
        # failed（崩溃/超时）不进任何模型（censored），历史由 runner 记录

    # ------------------------------------------------ GP 与采集
    def _fit(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (Matern, WhiteKernel,
                                                      ConstantKernel)
        X = np.asarray(self._X)
        y = np.asarray(self._y)
        # 去重防奇异（重复点加微扰）
        X = X + self._rng.normal(0, 1e-9, X.shape)
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=0.3, length_scale_bounds=(0.02, 5.0),
                           nu=2.5)
                  + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e1)))
        g = GaussianProcessRegressor(kernel=kernel, alpha=1e-8,
                                     normalize_y=True,
                                     n_restarts_optimizer=2, random_state=0)
        g.fit(X, y)
        return g

    def _optimize_acq(self, k: int = 1):
        """采集函数 top-k（k=1 即经典单点 ask）。-> list[[0,1]^d] 或 None。

        k>1 时贪心多样化：按 acq 降序逐个入选，拒绝与已入选点欧氏距离
        < min_dist（默认 0.05·sqrt(d)，即"不同盆地"）的候选。
        """
        try:
            gpr = self._fit()
        except Exception:
            return None
        self._gpr = gpr
        best = float(np.min(self._y))
        name = str(self.cfg.get("acq", "ei"))
        acq = ACQUISITIONS[name]
        kw = {"ei": dict(xi=float(self.cfg.get("xi", 0.01))),
              "pi": dict(xi=float(self.cfg.get("xi", 0.01))),
              "ucb": dict(beta=float(self.cfg.get("beta", 2.0)))}[name]
        nc = int(self.cfg.get("n_candidates", 2000))
        C = self._rng.rand(nc, self.space.dim)
        s = self._acq_at(gpr, acq, best, C, kw)
        # 局部精修：Top-5 邻域撒扰动
        top = np.argsort(s)[-5:]
        for j in top:
            for _ in range(60):
                cand = np.clip(C[j] + self._rng.normal(0, 0.03, C.shape[1]),
                               0.0, 1.0)
                C = np.vstack([C, cand])
        s2 = self._acq_at(gpr, acq, best, C, kw)
        if bool(self.cfg.get("constrain", False)):
            s2 = s2 * self._pfits(C)          # cEI = EI × P(可行)
        if k <= 1:
            return [C[int(np.argmax(s2))]]
        order = np.argsort(s2)[::-1]
        min_dist = float(self.cfg.get("min_dist", 0.05)) * np.sqrt(self.space.dim)
        chosen = []
        for idx in order:
            v = C[idx]
            if all(np.linalg.norm(v - u) >= min_dist for u in chosen):
                chosen.append(v)
                if len(chosen) >= k:
                    break
        return chosen

    def _acq_at(self, gpr, acq, best, V, kw):
        mu, sd = gpr.predict(V, return_std=True)
        return acq(mu, sd, best, **kw)

    def _pfits(self, V) -> np.ndarray:
        """可行概率 P(可行|x)：对 0/1 标签做 GP 分类（latent + logistic）。

        样本不足或两类缺失时返回全 1（不干预采集）。分类器缓存在 self._gpc，
        每次 ask 重拟合（评估昂贵，分类开销可忽略）。
        """
        if len(self._yc) < 4 or len(set(self._yc)) < 2:
            return np.ones(len(V))
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        try:
            gpc = GaussianProcessClassifier(
                kernel=RBF(length_scale=0.5, length_scale_bounds=(0.05, 5.0)),
                random_state=0)
            gpc.fit(np.asarray(self._Xc), np.asarray(self._yc))
            self._gpc = gpc
            return gpc.predict_proba(V)[:, 1]
        except Exception:
            return np.ones(len(V))

    # ------------------------------------------------ 检视支持（阶段2 UI 消费）
    def surrogate_1d(self, key: str, other: dict, n: int = 100):
        """固定其余参数，沿某一维画 GP 后验均值/标准差（代理面页数据）。

        -> (xs 原始坐标, mu, sigma)；未拟合时返回 None。
        """
        if self._gpr is None or self.space.spec(key) is None:
            return None
        base = self.space.to_vector(other)   # 缺的参数走默认值
        ts = np.linspace(0, 1, n)
        V = np.tile(base, (n, 1))
        j = self.space.keys.index(key)
        V[:, j] = ts
        mu, sd = self._gpr.predict(V, return_std=True)
        xs = np.array([float(self.space.from_vector(v)[key]) for v in V])
        return xs, mu, sd
