# -*- coding: utf-8 -*-
"""方法注册表：新增方法 = 实现 MLMethod 子类 + @register，不改框架（P2）。"""
from __future__ import annotations

from .contracts import MLMethod

_REGISTRY: dict[str, type[MLMethod]] = {}


def register(cls: type[MLMethod]) -> type[MLMethod]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} 缺少 name")
    if cls.name in _REGISTRY:
        raise ValueError(f"方法名重复: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> MLMethod:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"未注册的方法: {name}（可用: {sorted(_REGISTRY)}）")
    return cls()


def all_methods() -> list[MLMethod]:
    return [c() for c in _REGISTRY.values()]


def names() -> list[str]:
    return sorted(_REGISTRY)


def by_family(family: str) -> list[MLMethod]:
    return [c() for c in _REGISTRY.values() if c.family == family]


def for_spec(spec) -> list[MLMethod]:
    """能处理该数据视图的全部方法（遍历系统的候选集）。"""
    return [m for m in all_methods() if m.can_handle(spec)]


def families() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in _REGISTRY.values():
        out.setdefault(c.family, []).append(c.name)
    return {k: sorted(v) for k, v in sorted(out.items())}


def load_builtin():
    """导入内置方法族（幂等）。"""
    from .. import methods  # noqa: F401
