# -*- coding: utf-8 -*-
"""优化器注册表：与 core.registry 同一模式（P9 契约先行，注册扩展）。"""
from __future__ import annotations

from .contracts import Optimizer

_REGISTRY: dict[str, type[Optimizer]] = {}


def register(cls: type[Optimizer]) -> type[Optimizer]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} 缺少 name")
    if cls.name in _REGISTRY:
        raise ValueError(f"优化器名重复: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> Optimizer:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"未注册的优化器: {name}（可用: {sorted(_REGISTRY)}）")
    return cls()


def names() -> list[str]:
    return sorted(_REGISTRY)


def all_optimizers() -> list[Optimizer]:
    return [c() for c in _REGISTRY.values()]


def families() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in _REGISTRY.values():
        out.setdefault(c.family, []).append(c.name)
    return {k: sorted(v) for k, v in sorted(out.items())}


def load_builtin():
    """导入内置优化器族（幂等）。"""
    from . import engines  # noqa: F401
