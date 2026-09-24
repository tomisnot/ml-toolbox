# -*- coding: utf-8 -*-
"""轻量资源预算估算与运行守卫。

本模块只做保守的 *事前估算*，不分配大矩阵、不读取系统指标，也不替代
操作系统资源监控。重点覆盖显式核方法、量子态/密度矩阵模拟等 O(n²/³)
路径，让调用方在 fit 前得到可解释的预算结论。

典型用法::

    budget = ResourceBudget(max_kernel_mb=128, max_operation_units=1e11)
    est = check_resources("gpr", n_samples=4000, n_features=20, budget=budget)
    print(est.complexity, est.kernel_mb)

runner 的可选守卫配置::

    RunConfig(extras={"resource_guard": {
        "max_kernel_mb": 128,
        "max_operation_units": 1e11,
        "safety_factor": 1.25,
    }})
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

_MIB = 1024.0 * 1024.0
_FLOAT8 = 8
_FLOAT16 = 16


@dataclass(frozen=True)
class ResourceBudget:
    """一次运行允许的轻量预算；任一上限为 ``None`` 表示该项不限制。"""

    max_kernel_mb: float | None = 256.0
    max_working_set_mb: float | None = 1024.0
    max_quantum_state_mb: float | None = 64.0
    max_operation_units: float | None = 1e12
    max_qubits: int | None = 8

    def __post_init__(self):
        for name in ("max_kernel_mb", "max_working_set_mb",
                     "max_quantum_state_mb", "max_operation_units"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value))
                                      or float(value) < 0):
                raise ValueError(f"{name} 必须是非负有限数或 None")
        if self.max_qubits is not None and int(self.max_qubits) < 0:
            raise ValueError("max_qubits 必须是非负整数或 None")

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> "ResourceBudget":
        """从配置字典构造；未知控制字段（如 safety_factor）由调用方处理。"""
        values = values or {}
        if not isinstance(values, Mapping):
            raise TypeError("resource_guard 必须是 bool 或 dict")
        fields = {
            "max_kernel_mb", "max_working_set_mb",
            "max_quantum_state_mb", "max_operation_units", "max_qubits",
        }
        return cls(**{k: values[k] for k in fields if k in values})


@dataclass(frozen=True)
class ResourceEstimate:
    """资源估算结果。所有 byte 数已包含调用方传入的 safety_factor。"""

    method: str
    n_samples: int
    n_features: int
    complexity: str
    kernel_elements: int = 0
    kernel_bytes: int = 0
    state_vector_bytes: int = 0
    density_matrix_bytes: int = 0
    working_set_bytes: int = 0
    operation_units: float = 0.0
    quantum: bool = False
    n_qubits: int | None = None
    warnings: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()

    @property
    def within_budget(self) -> bool:
        return not self.violations

    @property
    def kernel_mb(self) -> float:
        return self.kernel_bytes / _MIB

    @property
    def working_set_mb(self) -> float:
        return self.working_set_bytes / _MIB

    @property
    def quantum_state_mb(self) -> float:
        return (self.state_vector_bytes + self.density_matrix_bytes) / _MIB


class ResourceBudgetExceeded(RuntimeError):
    """估算结果超过显式预算；消息包含所有违规项。"""

    def __init__(self, estimate: ResourceEstimate):
        self.estimate = estimate
        detail = "; ".join(estimate.violations)
        super().__init__(f"{estimate.method} 资源预算超限: {detail}")


_CUBIC_EXACT = {"gpr", "kernel_ridge", "qsvc", "qkrr"}
_LINEAR_KERNEL_EXCLUSIONS = {"linear_svc"}
_DEFAULT_KERNEL_METHODS = {
    "svc", "svr", "nusvc", "ocsvm", "kernel_pca",
    "gpr", "kernel_ridge", "qsvc", "qkrr", "qlssvc", "qkmeans",
}


def _method_info(method) -> tuple[str, set[str]]:
    if isinstance(method, str):
        return method, set()
    name = str(getattr(method, "name", "") or type(method).__name__)
    return name, set(getattr(method, "tags", ()) or ())


def _effective_qubits(name: str, n_features: int,
                      params: Mapping | None, explicit: int | None) -> int | None:
    if explicit is not None:
        return max(0, int(explicit))
    p = params or {}
    if "n_qubits" in p:
        nq = int(p["n_qubits"])
        if nq > 0:
            return nq
        if name in {"qsvc", "qkrr", "qlssvc"}:
            return min(max(1, n_features), 5)
    if name == "qkmeans":
        # qkmeans 使用振幅编码，维度是 2^ceil(log2(max(d, 2)))。
        return max(1, int(math.ceil(math.log2(max(2, n_features)))))
    if name in {"qsvc", "qkrr", "qlssvc"}:
        return min(max(1, n_features), 5)
    if name == "qreservoir":
        return 4
    return None


def _scaled_int(value: float, safety_factor: float) -> int:
    return int(math.ceil(max(0.0, float(value)) * safety_factor))


def _scaled_float(value: float, safety_factor: float) -> float:
    return max(0.0, float(value)) * safety_factor


def estimate_resources(method, n_samples: int, n_features: int = 1, *,
                       params: Mapping | None = None,
                       n_qubits: int | None = None,
                       budget: ResourceBudget | None = None,
                       safety_factor: float = 1.0) -> ResourceEstimate:
    """估算方法的主要内存/操作量，不执行方法本身。

    复杂度是用于 guard 的保守量级而非严格 FLOP 证明：核方法按稠密矩阵
    路径估算，SVM 取 ``n³`` 上界；量子方法按 complex128 态矢量/密度矩阵
    与矩阵乘法估算。
    """
    name, tags = _method_info(method)
    p = dict(params or {})
    n = max(0, int(n_samples))
    d = max(0, int(n_features))
    sf = float(safety_factor)
    if not math.isfinite(sf) or sf <= 0:
        raise ValueError("safety_factor 必须是正有限数")
    budget = budget or ResourceBudget()

    quantum = name.startswith("q") or "quantum" in tags or name == "qreservoir"
    nq = _effective_qubits(name, d, p, n_qubits) if quantum else None
    is_kernel = name in _DEFAULT_KERNEL_METHODS or (
        "kernel" in tags and name not in _LINEAR_KERNEL_EXCLUSIONS)
    kernel_elements = 0
    state_bytes = 0
    density_bytes = 0
    ops = float(max(0, n) * max(0, d))

    if is_kernel:
        if name == "qkmeans":
            k = max(1, int(p.get("n_clusters", 3)))
            kernel_elements = n * k
            max_iter = max(1, int(p.get("max_iter", 40)))
            ops = float(n * k * max_iter)
        else:
            kernel_elements = n * n
            if name in _CUBIC_EXACT:
                ops = float(n) ** 3
            elif name == "qlssvc":
                support = min(n, max(1, int(p.get("max_support", 300))))
                ops = float(n * n + support ** 3)
            else:
                ops = float(n * n)
        shots = max(0, int(p.get("shots", 0) or 0))
        if name in {"qsvc", "qkrr", "qlssvc"} and shots:
            # 有限 shots 的 Bernoulli 采样仍需生成约 n²*shots 个随机数。
            ops += float(n * n * shots)

    if quantum and nq is not None:
        dim = 1 << nq
        if name in {"qsvc", "qkrr", "qlssvc", "qkmeans"}:
            state_bytes = _scaled_int(n * dim * _FLOAT16, sf)
            reps = max(1, int(p.get("reps", 1)))
            ops += float(n * dim * max(1, nq) * reps)
        if name == "qreservoir":
            density_bytes = _scaled_int(dim * dim * _FLOAT16, sf)
            virtual = max(1, int(p.get("virtual", 3)))
            # 每次演化包含两个稠密 (2^n x 2^n) 乘法，约 O(8^n)。
            ops += float(max(0, n) * virtual * 2 * (dim ** 3))

    if name == "qpca":
        # 当前实现是经典 d x d 协方差/密度矩阵，不按 2^n 量子比特估算。
        density_bytes = _scaled_int(d * d * _FLOAT8, sf)
        n_iter = max(1, int(p.get("n_iter", 60)))
        ops = float(n_iter * max(1, d) ** 3)

    kernel_bytes = _scaled_int(kernel_elements * _FLOAT8, sf)
    # 原始特征表 + 一份常见 transform 副本；大矩阵另计。
    base_working = _scaled_int(2 * n * d * _FLOAT8, sf)
    working = base_working + kernel_bytes + state_bytes + density_bytes

    warnings: list[str] = []
    if is_kernel and n >= 3000:
        warnings.append(f"稠密核矩阵 n={n}，建议先抽样或降维")
    elif is_kernel:
        kernel_mb = kernel_bytes / _MIB
        if kernel_mb >= 64:
            warnings.append(f"核矩阵约 {kernel_mb:.1f} MiB")
    if quantum and nq is not None and nq >= 7:
        warnings.append(f"经典量子模拟 n_qubits={nq} 成本快速增长")
    if ops >= 1e10:
        warnings.append(f"保守操作量约 {ops:.3g}")

    violations: list[str] = []
    if budget.max_kernel_mb is not None and kernel_bytes > budget.max_kernel_mb * _MIB:
        violations.append(f"kernel {kernel_bytes / _MIB:.1f} MiB > "
                          f"{budget.max_kernel_mb:g} MiB")
    if budget.max_working_set_mb is not None and working > budget.max_working_set_mb * _MIB:
        violations.append(f"working set {working / _MIB:.1f} MiB > "
                          f"{budget.max_working_set_mb:g} MiB")
    if budget.max_quantum_state_mb is not None and state_bytes + density_bytes > budget.max_quantum_state_mb * _MIB:
        violations.append(f"quantum state {(state_bytes + density_bytes) / _MIB:.1f} MiB > "
                          f"{budget.max_quantum_state_mb:g} MiB")
    if budget.max_operation_units is not None and ops > budget.max_operation_units:
        violations.append(f"operations {ops:.3g} > {budget.max_operation_units:.3g}")
    if budget.max_qubits is not None and nq is not None and nq > budget.max_qubits:
        violations.append(f"n_qubits {nq} > {budget.max_qubits}")

    complexity = "O(n)"
    if name == "qpca":
        complexity = "O(n_iter*d^3)"
    elif is_kernel:
        complexity = "O(n^3)" if name in _CUBIC_EXACT or name == "qlssvc" else "O(n^2)"
    if name == "qreservoir":
        complexity = "O(n*virtual*8^n_qubits)"
    elif name in {"qsvc", "qkrr", "qlssvc", "qkmeans"}:
        complexity += " + state simulation"

    return ResourceEstimate(
        method=name, n_samples=n, n_features=d, complexity=complexity,
        kernel_elements=kernel_elements, kernel_bytes=kernel_bytes,
        state_vector_bytes=state_bytes, density_matrix_bytes=density_bytes,
        working_set_bytes=working, operation_units=ops, quantum=quantum,
        n_qubits=nq, warnings=tuple(warnings), violations=tuple(violations),
    )


def check_resources(method, n_samples: int, n_features: int = 1, *,
                    params: Mapping | None = None,
                    n_qubits: int | None = None,
                    budget: ResourceBudget | None = None,
                    safety_factor: float = 1.0) -> ResourceEstimate:
    """返回估算；超限时抛 :class:`ResourceBudgetExceeded`。"""
    estimate = estimate_resources(
        method, n_samples, n_features, params=params, n_qubits=n_qubits,
        budget=budget, safety_factor=safety_factor,
    )
    if estimate.violations:
        raise ResourceBudgetExceeded(estimate)
    return estimate
