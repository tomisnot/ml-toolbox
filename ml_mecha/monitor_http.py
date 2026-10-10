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

    ⚠ 共享面板的**通用**读面**不渲染** ``/summary`` 的信封（``disputed`` 那句没有标准
    渲染位）⇒ 本函数额外把 ``recent_runs`` 摆成 ``rows``（面板附加页的表格行契约），
    对账结论则由项目附加页的 ``summaryLine`` 承担。信封本身**形状不变**（只多一个键）。
    """
    payload = monitor_summary_payload(ml.monitor.read())
    summary = payload.get("summary") or {}
    runs = summary.get("recent_runs") if isinstance(summary, Mapping) else None
    payload["rows"] = list(runs) if isinstance(runs, list) else []
    return payload


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

    ⚠⭐ **``limit`` 必须传下去**（2026-10-10 实测）：旧写法是"全量读 + 切片" ⇒ 真实仓
    （``runs/`` 下数千个 ``record.json``）这一页要 **16.6 秒**（等于打不开），而它只要最新几十条。
    现在由机械层"按目录 mtime 预筛 + 只读够用的那些"（实测 0.5 秒级，见 ``core.runs.list_meta``）。
    """
    from ml_toolbox.core import persistence
    return [dict(r) for r in persistence.list_records(max(0, int(limit)))]


def _limit_of(query: Mapping[str, list[str]], default: int = 50) -> int:
    """附加路由的 ``limit`` 参数：非纯数字 ⇒ 默认值（与核心的 ``since_seq`` 同容错）。"""
    raw = (query.get("limit") or [str(default)])[0]
    return int(raw) if raw.isdigit() else default


def _run_id_of(query: Mapping[str, list[str]]) -> str:
    """``/latest`` 与 ``/series`` 的 ``run_id`` 参数（缺省空串）。"""
    return str((query.get("run_id") or [""])[0] or "")


def tb_latest(ml: Any, run_id: str = "") -> dict:
    """**最新一轮**的训练数值（AI 与面板都读这一条）。

    ``run_id`` 省略时取**最近落盘的 run**（`runs/` 新→旧的第一条）——这正是
    "AI 刚提交完训练、想知道跑到哪了"的那一问。

    ⚠ **读不到就说读不到**（``available: false`` + ``reason``），**绝不回空数组假装**。
    """
    from . import tb_series
    rid = run_id
    if not rid:
        try:
            recs = archived_runs(1)
            rid = str(recs[0].get("run_id") or "") if recs else ""
        except Exception:
            rid = ""
    if not rid:
        return {"available": False, "reason": "还没有落盘的 run（先跑一次）"}
    return tb_series.latest_of(tb_series.read_run(rid))


def tb_run_series(run_id: str, limit: int = 500) -> dict:
    """**整条序列**（逐 epoch 曲线）+ 表格行（供共享面板的附加页直接渲染）。"""
    from . import tb_series
    if not run_id:
        return {"available": False, "reason": "需要 run_id"}
    payload = tb_series.read_run(run_id, limit=max(1, int(limit)))
    if payload.get("available"):
        payload["rows"] = tb_series.rows_of(payload)
    else:
        payload["rows"] = []
    return payload



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
        self._log = log
        #: TensorBoard 辅进程（**惰性**：只有真被请求时才起，训练路径零开销）
        self._tb: Any = None
        # 运行期事实：闭包持有 self，故 status_extra 每次请求都读到**当前**端口
        # （monitor_port 只有 bind 之后才知道，构造期填不了常量）。
        self._endpoint = _MonitorEndpoint(
            _source(ml), host=host, port=port, port_file=port_file,
            status_extra=self._status_extra, log=log)
        # 两条 ML 附加路由（框架统一包 {ok: true, **payload} 并归一化失败为 500）。
        # ⚠ 两条都按**共享面板的附加页数据契约**回 `rows`（表格行取 `payload.rows`）：
        #   * `/summary` 的 `rows` = 操作史里的 `recent_runs`（信封形状不变，只多这个键）；
        #   * `/runs` 的 `rows` = 磁盘存档记录（**行为变更**：旧键名是 `runs`）。
        self._endpoint.add_route("/summary", lambda _query: summary_view(self._ml))
        self._endpoint.add_route(
            "/runs", lambda query: {"rows": archived_runs(_limit_of(query)),
                                    "source": "runs/"})
        # ---- TensorBoard：数据（读事件文件）与界面（辅进程）各一条 ----
        # 数据：`/latest` 给"最新一轮"（AI 研判用），`/series` 给整条曲线（面板折线用）。
        self._endpoint.add_route("/latest", lambda query: tb_latest(
            self._ml, _run_id_of(query)))
        self._endpoint.add_route("/series", lambda query: tb_run_series(
            _run_id_of(query), _limit_of(query, 500)))
        self._endpoint.add_route("/dashboard", lambda _query: self.dashboard())

    # ------------------------------------------------------------ TensorBoard 辅进程
    def dashboard(self, *, action: str = "start") -> dict:
        """起/停/查 TensorBoard 辅进程，回它给人看的地址。

        ⚠ 它是**独立进程 + 端口文件发现**：窗口的开合不归训练管（训练不需要知道它在不在）。
        起不来时回 ``{ready: false, reason}``，**不假装起了**。
        """
        from .dashboard import TensorBoardSidecar
        if action == "stop":
            if self._tb is not None:
                self._tb.stop()
            return {"ready": False, "stopped": True}
        if action == "status":
            return ({"ready": True, "url": self._tb.url, "port": self._tb.port}
                    if self._tb is not None and self._tb.running
                    else {"ready": False, "reason": "未运行"})
        if self._tb is None:
            from ml_toolbox.core.persistence import RUNS_DIR
            self._tb = TensorBoardSidecar(RUNS_DIR, log=self._log)
        return self._tb.start()


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
        """起监视端点（**只它自己**）。

        ⚠⭐ **不在这里起"只读监视窗"**（2026-10-10 改回）：本项目的只读监视窗是
        **`ml_mecha/panel_web.py` 那个独立服务**（`.mlweb-port`，三个 view：ml/opt/nn），
        而 TensorBoard 是**它**在 `nn` 那一页按需起的（`panel_web.py` 的 docstring 原话：
        "共享监控端点上也有一条 `GET /dashboard`，但那是**另一条路、今天没人用**"）。
        ⇒ 让它随端点起来**不符合本项目的设计**（会白起一个没人看的 TB），
        正确的入口是 `app_entry.py` 起 `PanelWebSidecar`（见那里）。
        """
        return self._endpoint.start()

    def stop(self, timeout: float = 10.0) -> None:
        # 辅进程归本端点管：端点停 ⇒ 顺手停它（否则留一个孤儿 Web 占端口）
        if self._tb is not None:
            try:
                self._tb.stop()
            except Exception:
                pass
            self._tb = None
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


#: 公开面：端点壳 + ML 的取数函数。框架的 ``CORE_ROUTES`` / ``MonitorSource`` /
#: ``start_monitor_endpoint`` **不在这里转发**，``MONITOR_PORT_FILE`` 也**不**转发
#: ——它们的权威归属分别是 ``mecha.cockpit`` 与 ``ml_mecha.runtime``，本仓零消费者
#: （判据要用就直接去那两个地方取；R1：没有消费的转发面不留）。
__all__ = ["MonitorEndpoint", "start_monitor_endpoint", "history_records",
           "activity_records", "config_tree", "summary_view", "archived_runs",
           "tb_latest", "tb_run_series"]
