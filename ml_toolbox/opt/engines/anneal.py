# -*- coding: utf-8 -*-
"""横场量子退火（SQA）启发式引擎（自研，零依赖，family=quantum）。

对应教材第四章 §4.2「量子退火」：H(s)=A(s)·H_D+B(s)·H_P（式 4.61），
横场 H_D=-ΣX_i 早期主导（量子涨落产生比特翻转与叠加），问题项 H_P
后期主导（测量得低能自旋构型）。§4.2.3 明确点出量子退火与模拟退火的
**机制区别**：量子以**隧穿**穿垒（多自旋联合翻转），热退火以**热激活**
翻垒（单自旋越过）。本引擎在归一化 [0,1]^d 坐标上做单链 ask/tell，用
这两条各自的可观测量模拟该机制，不做真实含时薛定谔演化——
「量子行为 ≠ 量子加速」（书 §4.2.4 反复强调，公平比较是验收的一部分）。

三个量子启发的具体落点：
1. 退火调度 schedule(s)：s=t/T 从 0→1，隧穿幅度 Γ(s)、温度 T(s) 同步
   衰减（式 4.61 的 A(s)↓、B(s)↑；§4.1.3 最小能隙决定困难区间）。
2. 隧穿式簇翻转（tunnel）：早期以较大簇宽联合重采若干坐标 = 穿越宽势垒；
   后期退回单坐标局部下降 = 经典退火。这是与纯 SA 唯一但本质的差别。
3. Metropolis 接受（式 4.61 尾注「低能采样」语义）：以退火后的 T(s) 决定
   接受上 uphill 的概率，收尾趋近贪心。

诚实边界：黑盒昂贵评估下用**单链**（不做 Trotter 多副本 PIMC，副本会成倍
消耗评估预算）；能否胜过随机/模拟退火由 compare_records 的收敛曲线裁决，
本文件不宣称加速。失败观测（status≠ok）拒绝并保留当前态。
"""
from __future__ import annotations

import numpy as np

from ..contracts import Optimizer, ParamSpec
from ..registry import register


def _schedule(shape: str, s: float) -> float:
    """归一化进度 s∈[0,1] → 剩余「量子性」比例 g(s)∈[0,1]（g 大=早期）。"""
    s = float(np.clip(s, 0.0, 1.0))
    if shape == "exponential":
        return np.exp(-3.0 * s)
    if shape == "logarithmic":          # 慢冷：困难区间多留时间（§4.1 局域调度）
        return 1.0 / (1.0 + 3.0 * s)
    return 1.0 - s                       # linear（默认，式 4.61 线性插值）


@register
class SimulatedQuantumAnnealing(Optimizer):
    name = "quantum_anneal"
    display_name = "量子退火（SQA）"
    family = "quantum"
    tags = ("annealing", "tunneling", "combinatorial", "derivative-free")
    batch = False
    param_schema = [
        ParamSpec("tunnel", "隧穿簇宽", "int", 2, min=1, max=8,
                  hint="每步联合重采的坐标数上限：>1 体现量子隧穿（§4.2.3），"
                       "=1 退化为逐坐标热激活"),
        ParamSpec("gamma", "初始隧穿幅度", "number", 0.5, min=0.02, max=1.0,
                  hint="早期坐标扰动/重采的尺度（式 4.61 A(s) 的强度）"),
        ParamSpec("temp", "初温 T0", "number", 0.3, min=0.01, max=2.0,
                  hint="Metropolis 温度，随调度衰减（热涨落项）"),
        ParamSpec("schedule", "退火调度", "select", "linear",
                  choices=["linear", "exponential", "logarithmic"],
                  hint="Γ(s)/T(s) 的衰减形状（§4.1.3：最小能隙处宜放慢）"),
    ]

    def setup(self, space, seed, cfg, budget):
        super().setup(space, seed, cfg, budget)
        self.d = max(space.dim, 1)
        self._x = self._rng.rand(self.d)
        self._f = np.inf
        self._best_x = self._x.copy()
        self._best_f = np.inf
        self._step = 0
        self._total = max(int(budget.n_evals), 1)
        self._pending = None

    # ------------------------------------------------ 主接口
    def ask(self):
        g = _schedule(str(self.cfg.get("schedule", "linear")),
                      self._step / self._total)         # 剩余量子性
        x = self._x.copy()
        # 隧穿簇翻转：簇宽随 Γ(s) 增大（早期穿宽垒，后期局部）
        maxw = max(1, int(round(self.cfg.get("tunnel", 2) * (0.4 + 0.6 * g)
                                + (1 if g > 0.5 else 0))))
        maxw = min(maxw, self.d)
        kk = self._rng.randint(1, maxw + 1)
        cols = self._rng.choice(self.d, size=kk, replace=False)
        amp = float(self.cfg.get("gamma", 0.5)) * g
        if self._step % 3 == 0 or amp > 0.15:
            # 重采（弹到新的本征构型）：隧穿式长跳
            x[cols] = self._rng.rand(kk)
        else:
            # 局域微扰（热激活式近邻）
            x[cols] = np.clip(x[cols] + self._rng.randn(kk) * 0.1, 0.0, 1.0)
        self._pending = np.clip(x, 0.0, 1.0)
        return self.space.from_vector(self._pending)

    def tell(self, params, score, status="ok"):
        g = _schedule(str(self.cfg.get("schedule", "linear")),
                      self._step / self._total)
        s = float(score) if status == "ok" and np.isfinite(score) else np.inf
        self._step += 1
        if s < self._best_f:                             # 更新全局最优
            self._best_f, self._best_x = s, self._pending.copy()
        # Metropolis：温度随调度衰减 → 收尾趋近贪心（式 4.61 低能采样语义）
        T = max(float(self.cfg.get("temp", 0.3)) * g, 1e-3)
        dE = s - self._f
        if dE < 0 or self._rng.rand() < np.exp(-max(dE, 0.0) / T):
            self._x, self._f = self._pending.copy(), s
        else:                                            # 拒绝 → 回退到当前态
            self._pending = self._x.copy()
