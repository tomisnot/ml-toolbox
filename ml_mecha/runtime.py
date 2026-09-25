# -*- coding: utf-8 -*-
"""运行期描述符与端口文件的**单一真源**（交付形态 P2）。

为什么要单独一个模块：端口/状态文件名在四个地方被读写（权威写、看门人读与写、
监控页读、故障排查的人读），散落各处的字面量一旦漂移就是"起得来但看不见"
（``07`` §3.3 点名的假绿温床）。本模块把**文件名、语义、清理规则**收在一处。

三个文件（都在数据根，默认 = 仓根）：

============================  ==========================  ==============================
文件                           谁写                         内容
============================  ==========================  ==============================
``.mcp-port``                  权威                        裸端口（MCP 服务）
``.ml-monitor-port``           权威                        裸端口（只读监控端点）
``.ml-mecha-runtime.json``     看门人（模式/端口）、权威补 pid  运行期描述符（单一真源）
``.mode-request``              任意子进程                  切换意图（gui / ai）
``.mode-state``                看门人                      当前模式
============================  ==========================  ==============================

**清理规则同一条**（针对 Windows 的现实）：``terminate()`` / ``taskkill /F``
**不跑** Python 的 ``finally`` ⇒ "起写止删"不能只靠收尾。因此：

- :func:`clear_port_file` 在**启动前**调用时（``expected=None``）无条件删陈旧文件
  ——安全性论证：调用方此时已装配成功（手握 ``.ml-mecha/.writer.lock`` 写租约），
  同根不可能还有别的活宿主，所以这个文件必属死进程；
- 收尾时调用（``expected=<自己写的端口>``）则**只在内容仍是自己的**才删，
  免得误删下一台权威刚写的新端口。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

#: 端口文件（裸端口文本；EL 同名的发现约定，人/工具都认）。
MCP_PORT_FILE = ".mcp-port"
MONITOR_PORT_FILE = ".ml-monitor-port"
#: 运行期描述符（模式 / pid / 端口）——端口发现的单一真源。
RUNTIME_JSON = ".ml-mecha-runtime.json"
#: 看门人两模式的状态文件。
MODE_REQUEST = ".mode-request"
MODE_STATE = ".mode-state"

_RUNTIME_FIELDS = ("mode", "pid", "authority_pid", "mcp_port", "monitor_port",
                   "dsh_port", "root", "started_at", "updated_at")


def resolve(root: str | Path, name: str) -> Path:
    """数据根下的运行期文件路径（``root`` 通常就是仓根）。"""
    return Path(root) / name


# ---------------------------------------------------------------- 端口文件
def read_port(path: str | Path | None) -> int | None:
    """读裸端口；不存在 / 空 / 非数字都返回 ``None``（不猜）。"""
    if path is None:
        return None
    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return int(raw) if raw.isdigit() else None


def write_port_file(path: str | Path | None, port: int) -> None:
    """写裸端口。``path`` 为 ``None`` 时什么都不做（调用方可能不要发现文件）。"""
    if path is None:
        return
    try:
        Path(path).write_text(str(int(port)), encoding="utf-8")
    except OSError:
        pass


def clear_port_file(path: str | Path | None, expected: int | None = None) -> None:
    """删端口文件。

    ``expected=None`` → 无条件删（**启动前**清陈旧文件，见模块 docstring 的论证）；
    ``expected=<端口>`` → 只在内容仍是该端口时删（**收尾**时不误删新主人写的）。
    """
    if path is None:
        return
    target = Path(path)
    try:
        if not target.exists():
            return
        if expected is not None:
            raw = target.read_text(encoding="utf-8").strip()
            if raw and raw != str(int(expected)):
                return
        target.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------- 描述符
def read_runtime(root: str | Path) -> dict:
    """读运行期描述符；缺失/损坏返回 ``{}``（调用方按"未知"处理，不假装知道）。"""
    try:
        payload = json.loads(resolve(root, RUNTIME_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_runtime(root: str | Path, **fields) -> dict:
    """合并写运行期描述符（保留未知/未提到的旧字段，只覆盖显式给的）。

    保留旧字段是**有意**的：看门人写 ``mode``、权威随后补 ``pid``/``mcp_port``，
    两边各写一半而不互相抹掉。
    """
    path = resolve(root, RUNTIME_JSON)
    payload = read_runtime(root)
    for key, value in fields.items():
        if key in _RUNTIME_FIELDS:
            payload[key] = value
    payload.setdefault("root", str(root))
    payload.setdefault("pid", os.getpid())
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    except OSError:
        pass
    return payload


def clear_runtime(root: str | Path) -> None:
    try:
        target = resolve(root, RUNTIME_JSON)
        if target.exists():
            target.unlink()
    except OSError:
        pass


__all__ = ["MCP_PORT_FILE", "MONITOR_PORT_FILE", "RUNTIME_JSON", "MODE_REQUEST",
           "MODE_STATE", "resolve", "read_port", "write_port_file",
           "clear_port_file", "read_runtime", "write_runtime", "clear_runtime"]
