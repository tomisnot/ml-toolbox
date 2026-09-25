# -*- coding: utf-8 -*-
"""ML Toolbox 的只读监控端点（**薄壳**）：端点、路由与四个核心投影全在 ``mecha.cockpit``。

归属：本文件住在 ML 仓，只保留**ML 的东西**：

- 运行期事实（``/status`` 的 ``pid`` / ``uptime_s`` / ``mcp_port`` / ``monitor_port``）；
- 两条**附加**只读路由（``/summary``、``/runs``，经框架 ``add_route`` 注册）与其取数；
- 数据源：``ml.software``（框架按 ``history`` / ``gate`` / ``authority`` 三个成员识别）。

四个核心路由（``/status`` ``/activity`` ``/history`` ``/config``）的形状是**跨项目契约**
（EL 与 ML 逐字段同形，面板不改渲染），现在由框架统一实现与守卫，本文件不再自己写一遍。

## 迁移记录（P3：两层传输上提框架，2026-09-25）

迁移前本文件 333 行，自持 handler 类、路由表、四个投影、线程生命周期与端口文件。
上提依据见 ``_mecha-extraction/cockpit-contract.md``（EL 为事实规范源，ML 逐字段对齐）。

## 迁移带来的三处行为变更（都由主代理裁决，均已落实）

1. **附加路由失败响亮化**（D6）：``/runs`` 的取数**不再吞异常**。此前 ``except: return []``
   会把"读不到"变成"没有数据"——两者不可区分，正是两家判据纪律反复在防的假绿。
   现在 handler 抛异常 ⇒ 框架回 **500 + 结构化错误体**（``{message, info:{kind,…}}``），
   面板侧据此显示可读错误，而不是空白表。
   ⚠ 注意区分：``runs/`` 目录**不存在**时 ``ml_toolbox.core.runs.list_meta`` 本来就**合法**
   返回空表（"确实没跑过"），那仍然是 200 + 空表——不是错误。
2. **``/config`` 的 ``n_snapshot_keys`` 由核心提供**：本文件不再自己加（框架的
   ``config_tree`` 恒回它）。ML 无逐键 schema ⇒ 走"无 schema"分支（``has_schema=false``、
   ``groups={}``、快照键全进 ``orphans``）——这条分支是可观测契约，不是降级。
3. **``/status`` 的附加键经 ``status_extra`` 注入**：核心只回 ``{ok, mode}``，
   宿主生命周期事实（pid/uptime/端口）由本文件提供，键名与迁移前逐字相同。
"""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any, Callable

from mecha.cockpit import MonitorEndpoint as _MonitorEndpoint
from mecha.cockpit import activity_records as _fw_activity
from mecha.cockpit import config_tree as _fw_config
from mecha.cockpit import history_records as _fw_history
from mecha.cockpit import monitor_summary_payload

from .runtime import MONITOR_PORT_FILE  # noqa: F401 - 历史公开名（消费方在 runtime）


def _source(ml: Any):
    """数据源：``ml.software``（有 history/gate/authority 三成员，框架自动适配）。"""
    software = getattr(ml, "software", None)
    return software if software is not None else ml


def history_records(ml: Any, since_seq: int = 0) -> list[dict]:
    """飞行记录仪形状的事件（9 键；``before`` 由全史 fold 重建）——框架投影。"""
    return _fw_history(_source(ml), since_seq)


def activity_records(ml: Any, since_seq: int = 0) -> list[dict]:
    """轻量增量事件流（6 键，无 before/after/reason）——框架投影。"""
    return _fw_activity(_source(ml), since_seq)


def config_tree(ml: Any) -> dict:
    """配置态结构树（含核心提供的 ``n_snapshot_keys``）——框架投影。"""
    return _fw_config(_source(ml))


def summary_view(ml: Any) -> dict:
    """监控概括（``Monitor.read()``）：人话摘要 + **可对账**（``disputed``）。

    ``MonitorView.summary`` 里的值是 ``Claim``（派生层复述），必须投影成
    ``{"value", "target"}`` 再上线——否则 JSON 编码会把 dataclass 变成
    ``"Claim(value=...)"`` 字符串，人就再也读不出"它复述的是哪个原始键"。
    投影与 ``disputed`` 原样透出的规则在框架 ``monitor_summary_payload`` 里（单一实现）。
    """
    return monitor_summary_payload(ml.monitor.read())


def archived_runs(limit: int = 50) -> list[dict]:
    """磁盘上的实验存档（``runs/<run_id>/record.json``）。

    **与 History 是两件事**：History 记"谁在什么时候做了什么"（进程内 + journal），
    ``runs/`` 是 ML 自己的实验真值（跨进程、跨重启、跨模式都能读）。面板在"AI 模式
    重启之后"仍要列得出跑过什么，所以读后者。

    落盘是**有条件**的：``run_method`` 的 ``persist`` 默认 ``False``，GUI 批量路径也只存
    成功项 ⇒ 空表是正常结果（面板把它与 History 并列显示，空表 ≠ 没跑过）。

    ⚠ **不吞异常**（迁移变更 1）：``runs/`` 不存在时 ``list_meta`` 已合法返回空表，
    因此这里再包一层 ``except`` 只会把真正的读失败伪装成"没有数据"。异常向上抛，
    由框架归一化成 500 + 结构化错误体。
    """
    from ml_toolbox.core import persistence
    records = persistence.list_records()
    return [dict(r) for r in records[:max(0, int(limit))]]


def _limit_of(query: Mapping[str, list[str]], default: int = 50) -> int:
    """附加路由的 ``limit`` 参数：非纯数字 ⇒ 默认值（与核心的 ``since_seq`` 同容错）。"""
    raw = (query.get("limit") or [str(default)])[0]
    return int(raw) if raw.isdigit() else default


class MonitorEndpoint:
    """只读监控端点（薄壳：委托框架 ``MonitorEndpoint``）+ ML 的两条附加路由。

    构造参数与迁移前逐字相同（``app_entry.py`` 与判据都按这些关键字调用）。
    """

    def __init__(self, ml: Any, *, host: str = "127.0.0.1", port: int = 0,
                 port_file: Any = None, mcp_port: int | None = None,
                 log: Callable[[str], None] | None = None) -> None:
        self._ml = ml
        self._started_at = time.time()
        self._mcp_port = mcp_port
        # 运行期事实：闭包持有 self，故 status_extra 每次请求都读到**当前**端口
        # （monitor_port 只有 bind 之后才知道，构造期填不了常量）。
        self._endpoint = _MonitorEndpoint(
            _source(ml), host=host, port=port, port_file=port_file,
            status_extra=self._status_extra, log=log)
        # 两条 ML 附加路由（框架统一包 {ok: true, **payload} 并归一化失败为 500）
        self._endpoint.add_route("/summary", lambda _query: summary_view(self._ml))
        self._endpoint.add_route(
            "/runs", lambda query: {"runs": archived_runs(_limit_of(query)),
                                    "source": "runs/"})

    def _status_extra(self) -> dict:
        """``/status`` 的 ML 附加键（键名与迁移前逐字相同）。"""
        return {"pid": os.getpid(),
                "uptime_s": round(max(0.0, time.time() - self._started_at), 1),
                "mcp_port": self._mcp_port,
                "monitor_port": self.port}

    # ------------------------------------------------------------ 事实面
    @property
    def port(self) -> int | None:
        return self._endpoint.port

    @property
    def url(self) -> str:
        return self._endpoint.url

    @property
    def source(self):
        """被投影的数据源（只读用途）。"""
        return self._endpoint.source

    def routes(self) -> list[str]:
        """已注册的附加路由（``/runs`` / ``/summary``）。"""
        return self._endpoint.routes()

    # ------------------------------------------------------------ 生命周期
    def start(self) -> int:
        return self._endpoint.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._endpoint.stop(timeout)

    def __enter__(self) -> "MonitorEndpoint":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.stop()
        return False


def start_monitor_endpoint(ml: Any, *, host: str = "127.0.0.1", port: int = 0,
                           port_file: Any = None, mcp_port: int | None = None,
                           log: Callable[[str], None] | None = None,
                           ) -> MonitorEndpoint:
    """便利函数：起一个已 start 的 :class:`MonitorEndpoint`（签名对齐迁移前）。"""
    endpoint = MonitorEndpoint(ml, host=host, port=port, port_file=port_file,
                               mcp_port=mcp_port, log=log)
    endpoint.start()
    return endpoint


#: 公开面：端点壳 + ML 的取数函数 + 历史公开名 ``MONITOR_PORT_FILE``。
#: 框架的 ``CORE_ROUTES`` / ``MonitorSource`` / ``start_monitor_endpoint`` **不再**
#: 从这里转发——本仓没有它们的消费者（判据要用就直接 ``from mecha.cockpit import …``）。
__all__ = ["MonitorEndpoint", "start_monitor_endpoint", "history_records",
           "activity_records", "config_tree", "summary_view", "archived_runs",
           "MONITOR_PORT_FILE"]
