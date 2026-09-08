# -*- coding: utf-8 -*-
"""CMA-ES（协方差矩阵自适应进化策略，自研，零新依赖）。

family=evo，batch=True：每 ask 一代 λ 个候选（种群式序贯），tell 整代更新。
赌注：中等预算、连续空间、可画搜索分布演化（协方差椭圆，阶段2 检视页可扩展）。

实现在 [0,1]^d 归一化坐标上进行（ParamSpace.to_vector），采样后 clip 回界内。
公式取 Hansen CMA-ES tutorial 的标准 (μ/μI, λ)-CMA-ES。
失败观测（score=inf）排序自然落到底部，被 elite 选择丢弃。
"""
from __future__ import annotations

import numpy as np

from ..contracts import Optimizer, ParamSpec
from ..registry import register


@register
class CMAES(Optimizer):
    name = "cma_es"
    display_name = "CMA-ES"
    family = "evo"
    batch = True
    tags = ("evolution-strategy", "continuous", "medium-budget")
    param_schema = [
        ParamSpec("sigma0", "初始步长", "number", 0.3, min=0.01, max=1.0,
                  hint="[0,1] 归一化域内的初始标准差"),
        ParamSpec("popsize", "种群大小 λ", "int", 0, min=0, max=200,
                  hint="0 = 自动 4+3·ln(d)"),
    ]

    def setup(self, space, seed, cfg, budget):
        super().setup(space, seed, cfg, budget)
        d = max(space.dim, 1)
        lam = int(self.cfg.get("popsize", 0)) or int(4 + 3 * np.log(d))
        lam = max(lam, 4)
        mu = max(lam // 2, 1)
        w = np.log(mu + 0.5) - np.arange(1, mu + 1)
        w = np.maximum(w, 0); w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cs = (mueff + 2) / (d + mueff + 5)
        self._c = dict(
            d=d, lam=lam, mu=mu, w=w, mueff=mueff,
            cs=cs,
            ds=1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs,
            cc=(4 + cs / d) / (d + 4 + 2 * cs / d),
            c1=2 / ((d + 1.3) ** 2 + mueff),
            cmu=min(1 - 2 / ((d + 1.3) ** 2 + mueff),
                    2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff)),
        )
        self.m = np.full(d, 0.5)
        self.sigma = float(self.cfg.get("sigma0", 0.3))
        self.C = np.eye(d)
        self.pc = np.zeros(d)
        self.psc = np.zeros(d)
        self._invsqrtC = np.eye(d)
        self._asked = None
        self._g = 0

    # ------------------------------------------------ ask：采一代
    def ask(self):
        c, d = self._c, self._c["d"]
        # 协方差特征分解（每代做，d 小可接受）
        try:
            evals, evecs = np.linalg.eigh(self.C)
            evals = np.maximum(evals, 1e-20)
        except np.linalg.LinAlgError:
            evals, evecs = np.ones(d), np.eye(d)
        self._B = evecs
        self._D = np.sqrt(evals)
        self._invsqrtC = evecs @ np.diag(1 / self._D) @ evecs.T
        Z = self._rng.standard_normal((c["lam"], d))
        Y = (Z * self._D) @ self._B.T
        X = self.m + self.sigma * Y
        X = np.clip(X, 0.0, 1.0)
        self._asked = X
        self._m_old = self.m.copy()
        return [self.space.from_vector(x) for x in X]

    # ------------------------------------------------ tell：整代更新
    def tell(self, params, score, status="ok"):
        c = self._c
        scores = np.asarray(score, float)
        X = self._asked
        order = np.argsort(scores)                 # 升序：越小越好
        sel = order[:c["mu"]]
        if len(sel) == 0:
            return
        w = c["w"][:len(sel)]
        w = w / w.sum()
        Xw = w @ X[sel]                          # (mu,) @ (mu,d) -> (d,)
        y_w = Xw - self._m_old
        self.m = self._m_old + y_w                 # cm=1
        # 演化路径
        z_mean = self._invsqrtC @ (self.m - self._m_old) / self.sigma
        hsig = (np.linalg.norm(z_mean)
                / np.sqrt(2 * (1 - (1 - c["cs"]) ** (2 * (self._g + 1)))
                          ) < c["d"] + 2 / (c["d"] + 1))
        self.pc = ((1 - c["cc"]) * self.pc
                   + hsig * np.sqrt(c["cc"] * (2 - c["cc"]) * c["mueff"]) * z_mean)
        self.psc = (1 - c["cs"]) * self.psc + np.sqrt(
            c["cs"] * (2 - c["cs"]) * c["mueff"]) * z_mean
        # 协方差更新
        artmp = (X[sel] - self._m_old) / self.sigma
        delta_h = (1 - hsig) * c["cc"] * (2 - c["cc"])
        self.C = ((1 - c["c1"] - c["cmu"]) * self.C
                  + c["c1"] * (np.outer(self.pc, self.pc)
                               + delta_h * self.C)
                  + c["cmu"] * (artmp * w[:, None]).T @ artmp)
        self.C = (self.C + self.C.T) / 2           # 数值对称
        # 步长自适应
        self.sigma *= np.exp(min(0.6, (c["cs"] / c["ds"])
                                 * (np.linalg.norm(self.psc) / np.sqrt(c["d"]) - 1)))
        self._g += 1

    # ------------------------------------------------ 检视支持
    def distribution(self):
        """当前搜索分布（mean, sigma, C）——分布演化动画页消费。"""
        return self.m.copy(), self.sigma, self.C.copy()


# ================================================================ NSGA-II
@register
class NSGAII(Optimizer):
    """多目标遗传算法 NSGA-II（自研，batch，非支配排序+拥挤距离）。

    赌注：数模双/多目标（成本-精度、强度-重量）交付 Pareto 前沿。
    objective 需 multi=True、n_obj>=2；tell 收到的是分数向量列表。
    SBX 交叉 + 多项式变异，标准 Deb 2002。
    """
    name = "nsga_ii"
    display_name = "NSGA-II 多目标"
    family = "evo"
    batch = True
    multi_objective = True
    tags = ("multi-objective", "pareto", "evolutionary")
    param_schema = [
        ParamSpec("popsize", "种群大小", "int", 40, min=8, max=500),
        ParamSpec("eta_c", "交叉分布指数", "number", 20.0, min=1, max=100),
        ParamSpec("eta_m", "变异分布指数", "number", 20.0, min=1, max=100),
        ParamSpec("p_mut", "变异概率", "number", -1.0, min=-1, max=1.0,
                  hint="-1 = 自动 1/dim"),
    ]

    def setup(self, space, seed, cfg, budget):
        super().setup(space, seed, cfg, budget)
        self.d = max(space.dim, 1)
        self.N = int(self.cfg.get("popsize", 40))
        self.eta_c = float(self.cfg.get("eta_c", 20.0))
        self.eta_m = float(self.cfg.get("eta_m", 20.0))
        pm = float(self.cfg.get("p_mut", -1.0))
        self.p_mut = (1.0 / self.d) if pm < 0 else pm
        self.P = np.clip(self._rng.rand(self.N, self.d), 0.0, 1.0)
        self.F = None
        self._to_eval = self.P          # 初代：评估父种群本身

    def ask(self):
        # 返回"待评估"的一代（初代=父种群，之后=子代 Q）
        return [self.space.from_vector(x) for x in self._to_eval]

    def tell(self, params, score, status="ok"):
        F = np.atleast_2d(np.asarray(score, float))
        if F.shape[0] != len(self._to_eval):        # 截断代 → 补齐
            F = self._pad(F, len(self._to_eval))
        F = F.copy()
        F[~np.isfinite(F)] = np.inf
        if self.F is None:                          # 初代父种群分数到手
            self.F = F
            self._to_eval = self._breed()           # 生成子代 Q 去评估
            return
        # Q 的分数到手：合并父+子，环境选择出新一代 P，再繁殖 Q
        R = np.vstack([self.P, self._to_eval])
        RF = np.vstack([self.F, F])
        self.P, self.F = self._envi_select(R, RF, self.N)
        self._to_eval = self._breed()

    # ------------------------------------------------ 遗传算子
    def _breed(self):
        Q = []
        idx = self._rng.permutation(len(self.P))
        for a, b in zip(idx[0::2], idx[1::2]):
            c1, c2 = self._sbx(self.P[a], self.P[b])
            Q += [self._mutate(c1), self._mutate(c2)]
        return np.clip(np.array(Q[:len(self.P)]), 0.0, 1.0)

    def _sbx(self, p1, p2):
        u = self._rng.rand(len(p1))
        beta = np.where(u <= 0.5,
                        (2 * u) ** (1 / (self.eta_c + 1)),
                        (1 / np.maximum(2 * (1 - u), 1e-12)) ** (1 / (self.eta_c + 1)))
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        swap = self._rng.rand(len(p1)) < 0.5
        c1, c2 = np.where(swap, c2, c1), np.where(swap, c1, c2)
        return c1, c2

    def _mutate(self, x):
        m = self._rng.rand(len(x)) < self.p_mut
        u = self._rng.rand(len(x))
        delta = np.where(u < 0.5,
                         (2 * u) ** (1 / (self.eta_m + 1)) - 1,      # ∈[-1,0) 向左
                         1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1)))  # ∈(0,1] 向右
        # 步长按移动方向的边界距离缩放：向左最多到 0，向右最多到 1
        dist = np.where(delta < 0, x, 1.0 - x)
        return np.where(m, x + delta * dist, x)

    def _pad(self, F, n):
        pad = np.full((n - F.shape[0], F.shape[1]), np.inf)
        return np.vstack([F, pad]) if n > F.shape[0] else F[:n]

    # ------------------------------------------------ 环境选择
    def _envi_select(self, R, RF, N):
        fronts = self._fast_nondom_sort(RF)
        sel, i = [], 0
        for fr in fronts:
            if len(sel) + len(fr) <= N:
                sel.extend(fr)
            else:
                fr = np.array(fr)
                cd = self._crowding(RF[fr])
                order = fr[np.argsort(-cd)]
                sel.extend(order[:N - len(sel)])
                break
            i += 1
        sel = np.array(sel[:N])
        return R[sel], RF[sel]

    def _fast_nondom_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]          # 被 i 支配的
        nd = np.zeros(n, int)
        fronts = [[]]
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self._dom(F[p], F[q]):
                    S[p].append(q)
                elif self._dom(F[q], F[p]):
                    nd[p] += 1
            if nd[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        nxt.append(q)
            fronts.append(nxt)
            i += 1
        return [f for f in fronts if f]

    @staticmethod
    def _dom(a, b):
        return np.all(a <= b) and np.any(a < b)

    def _crowding(self, Fm):
        n = len(Fm)
        d = np.zeros(n)
        for j in range(Fm.shape[1]):
            o = np.argsort(Fm[:, j])
            d[o[0]] = d[o[-1]] = np.inf
            rng = Fm[o[-1], j] - Fm[o[0], j]
            if rng > 1e-12:
                d[o[1:-1]] += (Fm[o[2:], j] - Fm[o[:-2], j]) / rng
        return d
