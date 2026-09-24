# -*- coding: utf-8 -*-
"""小规模量子态模拟器（numpy 态矢量 / 密度矩阵）——量子启发方法的共享机构。

定位（对齐教材《量子机器学习》第三章 §3.1、第七章 §7.1、第九章 §9.1）：
不是量子计算机接口，而是把书里的编码 / 线路 / 协议在态矢量层面**逐步复现**：
- 角度编码 / 二阶量子特征映射（式 3.6、3.13，Havlíček 式 ZZ 纠缠层）；
- 保真度量子核 k(x,z)=|⟨φ(x)|φ(z)⟩|²（式 3.69），反演测试的有限 shots
  采样语义按式 3.73–3.78（Bernoulli 计数 + Hoeffding 统计涨落）；
- 振幅编码（式 3.9，保真度/交换测试距离的输入形态，式 7.13–7.18）；
- 密度矩阵的 Pauli 可观测量期望（式 9.21 的储层读出特征）。

这里复现的是**算法结构与统计行为**，不是量子加速——资源用 n_qubits 上限
（≤8 比特，2^8=256 振幅）显式钉死，超出即 PCA 降维或报错（诚实降级）。
零 Qt、零新依赖（仅 numpy）；方法层约定同 P1。
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- 单比特门
# 基态编号约定与教材一致：n 比特基矢 |b_0 b_1 … b_{n-1}⟩ 的指标
# k = Σ_j b_j · 2^{n-1-j}（qubit 0 为最高位，与 kron 直积序一致）。

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)


def zero_states(m: int, n: int) -> np.ndarray:
    """m 行、每行是 |0…0⟩ 的批量态矢量 (m, 2^n)。"""
    psi = np.zeros((m, 1 << n), dtype=complex)
    psi[:, 0] = 1.0
    return psi


def _bits(dim: int, q: int, n: int) -> np.ndarray:
    """每个基态在 qubit q 上的比特 (0/1)，长度 dim=2^n。"""
    idx = np.arange(dim)
    return (idx >> (n - 1 - q)) & 1


def apply_1q(psi: np.ndarray, U: np.ndarray, q: int) -> np.ndarray:
    """批量态矢 (m, 2^n) 上对 qubit q 作用单比特门 U（2×2，可对角元变体
    传入逐行系数矩阵的写法见 apply_ry_batch）。"""
    m, dim = psi.shape
    n = int(round(np.log2(dim)))
    left, right = 1 << q, dim >> q >> 1
    v = psi.reshape(m, left, 2, right)
    a0 = v[:, :, 0, :].copy()
    a1 = v[:, :, 1, :].copy()
    v[:, :, 0, :] = U[0, 0] * a0 + U[0, 1] * a1
    v[:, :, 1, :] = U[1, 0] * a0 + U[1, 1] * a1
    return v.reshape(m, dim)


def apply_ry_batch(psi: np.ndarray, theta: np.ndarray, q: int) -> np.ndarray:
    """逐样本角度 θ[m] 的 Ry(θ) 旋转（式 3.6 的角度编码本体）。"""
    m = psi.shape[0]
    c = np.cos(theta / 2.0)[:, None, None]
    s = np.sin(theta / 2.0)[:, None, None]
    dim = psi.shape[1]
    n = int(round(np.log2(dim)))
    left, right = 1 << q, dim >> q >> 1
    v = psi.reshape(m, left, 2, right)
    a0 = v[:, :, 0, :].copy()
    a1 = v[:, :, 1, :].copy()
    v[:, :, 0, :] = c * a0 - s * a1
    v[:, :, 1, :] = s * a0 + c * a1
    return v.reshape(m, dim)


def apply_diag_phase(psi: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """对角酉：逐基态乘相位 (m, 2^n) 或 (2^n,)。"""
    return psi * phases


def rz_phases(theta: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rz(θ)=exp(-iθZ/2) 的逐基态相位；b 为该量子比特的 0/1（本征值 s=1-2b）。"""
    s = 1.0 - 2.0 * b
    return np.exp(-0.5j * np.asarray(theta)[..., None] * s)


# ---------------------------------------------------------------- 特征映射
def feature_map_states(Xs: np.ndarray, gamma: float = 1.0,
                       reps: int = 1) -> np.ndarray:
    """二阶角度编码量子特征映射 |φ(x)⟩ = U_φ(x)|0…0⟩（式 3.13/3.6）。

    Xs 已缩放到 [0, π]（m, n）。每层结构：
      ⊗Ry(γ·x_j) → 环形最近邻对 Rzz((π-x_j)(π-x_k)/4)；
    reps>1 时层间插 H^⊗n（数据重上传思想，式 3.6 段落；深度换表达力）。
    n=1 时退化为式 (3.6) 的纯角度编码，核对解析核 ∏cos²((x_j-z_j)/2)；
    纠缠门使 n≥2 的态不可分解——核值不再是任何经典低维特征的内积
    （§3.3.2 例的反面），这正是量子核的算法内容。
    """
    Xs = np.asarray(Xs, float)
    m, n = Xs.shape
    if m == 0 or n == 0 or n > 8:
        raise ValueError(f"特征映射需 1≤n≤8 量子比特，收到 n={n}, m={m}")
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    psi = zero_states(m, n)
    dim = 1 << n
    angle = gamma * Xs                              # (m, n)
    bcols = [_bits(dim, q, n) for q in range(n)]
    for rep in range(max(1, int(reps))):
        if rep > 0:
            for q in range(n):
                psi = apply_1q(psi, H, q)
        for q in range(n):
            psi = apply_ry_batch(psi, angle[:, q], q)
        if n >= 2:
            for q in range(n):
                k = (q + 1) % n
                if k == q:
                    continue
                zz = (1.0 - 2.0 * bcols[q]) * (1.0 - 2.0 * bcols[k])
                phi = (np.pi - Xs[:, q]) * (np.pi - Xs[:, k]) / 4.0
                psi = apply_diag_phase(psi, np.exp(-0.5j * phi[:, None] * zz))
    return psi


def fidelity_kernel(S: np.ndarray, S2: np.ndarray | None = None,
                    shots: int = 0, rng=None) -> np.ndarray:
    """保真度核 k(x,z)=|⟨φ(x)|φ(z)⟩|²（式 3.69；反演测试的解析值，式 3.74）。

    shots>0 时按式 3.73 的 Bernoulli 计数模拟有限测量：k̂ = Binom(S,k)/S，
    对称化后返回——对角线是自身重叠、恒为 1，不受采样噪声影响。
    """
    B = S if S2 is None else S2
    K = np.abs(S @ B.conj().T) ** 2
    K = np.clip(K, 0.0, 1.0)
    if shots and shots > 0:
        rng = rng or np.random.RandomState(42)
        K = rng.binomial(int(shots), K) / float(shots)
        if S2 is None:
            K = (K + K.T) / 2.0
    return K


# ---------------------------------------------------------------- 振幅编码
def amplitude_encode(X: np.ndarray):
    """振幅编码（式 3.9）：行补零到 2^n 并除范数；返回 (states, norms)。

    书 §3.1.2 强调：归一化丢失 ‖x‖——范数另行返回，距离重建（式 7.7）
    必须在经典侧乘回。零行无法编码，states 给零向量、距离恒为 1。
    """
    X = np.asarray(X, float)
    m, d = X.shape
    n = max(1, int(np.ceil(np.log2(max(d, 2)))))
    dim = 1 << n
    P = np.zeros((m, dim))
    P[:, :d] = X
    norms = np.linalg.norm(P, axis=1)
    unit = np.zeros_like(P)
    nz = norms > 0
    unit[nz] = P[nz] / norms[nz][:, None]
    return unit, norms


def fidelity_distance(S: np.ndarray, C: np.ndarray,
                      shots: int = 0, rng=None) -> np.ndarray:
    """量子可区分度距离 d(x,c)=1-|⟨x|c⟩|²（式 7.18 的 SWAP 测试语义）。"""
    F = np.abs(S @ C.conj().T) ** 2
    if shots and shots > 0:
        rng = rng or np.random.RandomState(42)
        F = rng.binomial(int(shots), np.clip(F, 0, 1)) / float(shots)
    return 1.0 - np.clip(F, 0.0, 1.0)


# ---------------------------------------------------------------- 密度矩阵
def embed_1q(op: np.ndarray, q: int, n: int) -> np.ndarray:
    """单比特算符嵌入 n 比特空间（I⊗…⊗op⊗…⊗I，qubit 0 在左）。"""
    M = np.array([[1.0]], dtype=complex)
    for j in range(n):
        M = np.kron(M, op if j == q else IDENTITY)
    return M


def kron_chain(ops: list) -> np.ndarray:
    M = np.array([[1.0]], dtype=complex)
    for o in ops:
        M = np.kron(M, o)
    return M


def ptrace_q0(rho: np.ndarray, n: int) -> np.ndarray:
    """对 qubit 0 取偏迹（储层的「丢弃旧第一比特」步骤，式 9.23）。"""
    d2 = 1 << (n - 1)
    v = rho.reshape(2, d2, 2, d2)
    return np.einsum("iaib->ab", v)


def prep_q0(sigma: np.ndarray, rho_rest: np.ndarray) -> np.ndarray:
    """把输入态 σ(2×2) 张量回 qubit 0（式 9.23 的替换步）。"""
    return np.kron(sigma, rho_rest)


def evolve(rho: np.ndarray, U: np.ndarray) -> np.ndarray:
    return U @ rho @ U.conj().T


def expectation(rho: np.ndarray, O: np.ndarray) -> float:
    """可观测量期望 x_m(t)=Tr[O ρ]（式 9.21）。厄米 ρ/O 下为实数。"""
    return float(np.real(np.trace(O @ rho)))


def expm_diagonal(H: np.ndarray) -> tuple:
    """厄米矩阵谱分解（e^{tH} 类演化的经典模拟后端）。"""
    return np.linalg.eigh(H)


# ---------------------------------------------------------------- 数据编码
class QuantumEncoder:
    """经典表 → 特征映射态 的拟合式变换器（训练期冻结统计量）。

    流程：d 维特征 →（d>n_qubits 时 PCA 到 n_qubits 维）→ min-max 到 [0,1]
    → 角度域 θ = u·π。常数列钉 0.5（避免除零）。预测/交叉验证必须复用
    fit 阶段的 scaler/pca——统计量泄漏防线，同 core.pipeline 约定。
    """

    def __init__(self, n_qubits: int = 5, gamma: float = 1.0,
                 reps: int = 1, seed: int = 42):
        self.n_qubits = int(n_qubits)
        self.gamma = float(gamma)
        self.reps = int(reps)
        self.seed = seed

    def fit(self, X: np.ndarray) -> "QuantumEncoder":
        X = np.asarray(X, float)
        d = X.shape[1]
        self.n_q = max(1, min(self.n_qubits, d))
        self.pca_ = None
        if d > self.n_q:
            from sklearn.decomposition import PCA
            self.pca_ = PCA(n_components=self.n_q,
                            random_state=self.seed).fit(X)
            Z = self.pca_.transform(X)
        else:
            Z = X
        lo = Z.min(axis=0)
        hi = Z.max(axis=0)
        rng_ = hi - lo
        const = rng_ <= 1e-12
        rng_[const] = 1.0
        self.lo_, self.span_, self.const_ = lo, rng_, const
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        Z = self.pca_.transform(X) if self.pca_ is not None else X
        u = (Z - self.lo_) / self.span_
        u = np.clip(u, 0.0, 1.0)
        u[:, self.const_] = 0.5
        return feature_map_states(u * np.pi, self.gamma, self.reps)
