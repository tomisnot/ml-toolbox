# -*- coding: utf-8 -*-
"""ML 命令面到 mecha 核心 ``CommandRegistry`` 的声明桥。

本模块是"第二宿主逼出通用机制"的落点：ML 的命令声明（``ml_mecha.commands``）
保持域特有字段（risk / writes_state / side_effect 字符串），这里把它们投影成
核心 ``CommandSpec``——核心只认识机制字段：参数 schema / 回执 schema /
side_effect bool / scope / estimate / cancel / approval。

纪律：

- 不把 ML 的 risk / writes_state 塞进核心；它们在宿主声明里，核心不需要懂；
- 核心参数 schema 由宿主参数声明机械生成（properties + required），不手抄；
- 有副作用命令必须在装配点注册，经核心的 ``command.<name>`` 审计留痕。
"""
from __future__ import annotations

from mecha.commands import CommandSpec, define_command

from .commands import COMMANDS, SIDE_EFFECT_NONE
from .engine import DEFAULT_EST_SEC, RECEIPT_REQUIRED_KEYS


def core_command_specs() -> list[CommandSpec]:
    """把 ML 命令声明投影为核心 ``CommandSpec`` 列表（顺序稳定）。"""
    out: list[CommandSpec] = []
    for host in COMMANDS:
        properties = {p.name: p.projection() for p in host.parameters}
        required = list(host.required_params)
        out.append(define_command(
            name=host.name,
            description=host.summary,
            parameters={"type": "object", "properties": properties,
                        "required": required},
            output_schema={"type": "object",
                           "required": list(RECEIPT_REQUIRED_KEYS)},
            side_effect=host.side_effect != SIDE_EFFECT_NONE,
            scope=(host.side_effect,),
            estimate_sec=DEFAULT_EST_SEC,
            cancel_supported=host.name != "prepare_dataset",
            approval_required=False,
        ))
    return out


__all__ = ["core_command_specs"]
