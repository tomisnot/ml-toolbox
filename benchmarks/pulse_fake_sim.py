# -*- coding: utf-8 -*-
"""假黑盒：模拟"原子模拟软件"的微波脉冲参数寻优（实战彩排用）。

真实工况特征（故意全部复刻，逼工具箱在彩排期踩坑而不是实战期）：
- 命令行接口：pulse_sim.py --amp <a> --len <b> --phase <p>
- 输出中文 + UTF-8 + 科学计数法分数行（GBK 管道坑）
- 参数域上有硬约束：脉冲面积 theta = amp*len ∈ [0.9, 1.1]（π 脉冲条件），
  违反时退出码非零（模拟软件自己拒绝）
- 评估昂贵：单次 sleep ~0.15s（模拟秒级仿真）
- 有噪声：指标叠加高斯噪声（模拟数值误差/实验涨落）
- 偶发失败：约 3% 概率退出码 3（模拟收敛失败）

真实指标（最小化）：失真的 fidelity 残差
  f = (theta - 1)^2 * 10 + (phase - pi/2)^2 + 0.05 * (amp - 0.7)^2 + noise
最优：theta≈1, phase≈pi/2, amp≈0.7（len=1/amp≈1.43）。

用法：python benchmarks/pulse_fake_sim.py --amp 0.7 --len 1.43 --phase 1.57
      输出 "fidelity_loss: 1.23e-02" 供解析。
"""
import argparse
import math
import random
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", type=float, required=True)
    ap.add_argument("--len", type=float, required=True)
    ap.add_argument("--phase", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # 模拟昂贵计算
    time.sleep(0.12 + 0.05 * random.Random(a.seed).random())

    theta = a.amp * a.len
    # 硬约束：π 脉冲条件（模拟软件自己拒绝，工具箱只能记 failed）
    if not (0.85 <= theta <= 1.15):
        print(f"错误：脉冲面积 θ={theta:.3f} 违反 π 脉冲条件 0.85≤θ≤1.15，仿真终止",
              file=sys.stderr)
        sys.exit(3)
    rng = random.Random(a.seed + int(theta * 1e6))
    f = (10 * (theta - 1) ** 2
         + (a.phase - math.pi / 2) ** 2
         + 0.05 * (a.amp - 0.7) ** 2
         + rng.gauss(0, 0.002))
    # 中文 + 科学计数法（两个编码/解析坑）
    print(f"原子模拟软件 v0.1（假黑盒） · 模式: pi_pulse")
    print(f"theta={theta:.4f} phase={a.phase:.4f}")
    print(f"fidelity_loss: {f:.4e}")
    sys.exit(0)


if __name__ == "__main__":
    main()
