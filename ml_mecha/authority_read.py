# -*- coding: utf-8 -*-
"""把**写权自省**注册进本软件的**能力面**（AI 那条腿）。

## 为什么能力面上也要有它（设计 §7.1.1 的"三处"）

`read_write_authority` 要在**三处**出现：① **软件自己的能力面**（本模块）
② 控制端点（`app_entry.py::run_mcp`）③ 聚合端点（框架）。
⚠ 设计专门警告：**最容易漏的是①** —— 施工时只想着"面板要读"，
就漏掉"**AI 自己也要能自查**"。漏了的现象是：AI 写被拒（`authority_locked` /
`authority_mode_mismatch`）之后**无法自查**，只能盲撞（框架的
`_with_read_hint` 会把 hint 指向这个工具名 —— 而工具不在表里，等于白指）。

⛔ **只注册这一个**：`set_write_authority` / `open_manual_surface` 是**改**写权的，
**绝不给 AI**（§7.1.1：不许 AI 给自己解锁）。它们只在**控制端点**上。

⚠ **不自己写实现**：机制在框架 `mecha.authority.register_authority_read`，
本模块只做"在哪注册、用什么文案"这两件项目自己的事。
"""
from __future__ import annotations

from typing import Any

from mecha.authority import register_authority_read

#: 给 AI 看的"怎么切写权"（框架**不写死**这句：那是**项目知识**）。
#: ⚠ **别写端口/凭据** —— 这句话 AI 会读到。
#: ⚠ 与 `ml_mecha/mecha_site.py::Service.how_to_open` / `app_entry.py` 的文案
#:   **同一事实**：三处都在回答"怎么切"。改一处就一起改（措辞可变，语义不许漂）。
HOW_TO_OPEN = (
    "写权由**人**在启动面板（常驻控制台）上切：点本软件的「切到手动」即归人"
    "（本机可信、无口令），再点「交还给 AI」交还模型。"
    "从 mecha 启动的形态带人手面（`--with-gui`）⇒ 同一页的「开手动面」会**真的把 "
    "GUI 窗口开出来**（开在软件进程内，与 MCP 面共用同一份门/账）。"
)


def register_authority_read_tool(registry: Any, authority: Any) -> None:
    """把只读自省 `read_write_authority` 注册进**能力面**。

    ⚠ 传的是**声明文案**，不是"读当前值"：工具**每次调用时**才去问 authority
    （框架的 `execute` 闭包持有对象引用）⇒ 切完写权立刻能读到新值。
    """
    register_authority_read(registry, authority, how_to_open=HOW_TO_OPEN)


__all__ = ["HOW_TO_OPEN", "register_authority_read_tool"]
