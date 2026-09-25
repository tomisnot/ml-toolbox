# -*- coding: utf-8 -*-
"""ML Toolbox 的只读监控端点（交付形态 P2：给 dsh 驾驶舱面板与独立监控页读）。

与 EL ``src/v2_bind.py::_MonitorHandler`` **逐字段同形**——这是硬要求，不是风格：
驾驶舱面板（P2c 从 EL ``dsh/src/client/`` 移植）的渲染逻辑不打算改，所以
``/status`` ``/activity`` ``/history`` ``/config`` 的**响应形状**必须一致，
面板才能原样吃 ML 的数据。形状契约见下表（EL 的出处逐条标注）：

============================  ==================================================
路由                            形状（EL ``src/v2_bind.py``）
============================  ==================================================
``GET /status``                ``{"ok": true, "mode": "<human|ai|locked>"}``（``:140``）
``GET /activity?since_seq=N``  ``{"ok": true, "events": [...], "mode": "..."}``（``:142``）
``GET /history?since_seq=N``   ``{"ok": true, "events": [...], "mode": "..."}``（``:153``）
``GET /config``                ``{"ok": true, "has_schema":…, "groups":…, "orphans":…, "n_keys":…}``（``:164``）
``GET /summary``（ML 附加）     ``{"ok": true, "summary":…, "disputed":…, "dispute_reason":…}``
``GET /runs``（ML 附加）        ``{"ok": true, "runs": [...], "source": "runs/"}``
未知路径                        404 ``{"ok": false, "error": "unknown endpoint"}``（``:169``）
``POST *``                     404 ``{"ok": false, "error": "read-only monitor endpoint"}``（``:171``）
``OPTIONS *``                  204 + CORS（``:130``）
============================  ==================================================

**只读**是设计红线：``POST`` 一律 404，路由表里没有任何写/切模式入口。理由同 EL
（``v2_bind.py:121-125``）：读的是"谁改了什么"的操作流水，不是域数据覆写面；
控制权切换归看门人（进程级交接），不经这里。"AI 不能自授权"的护栏靠
``Authority`` 的 side 判定，不靠端点鉴权——所以端点**无鉴权**、只绑回环。

两处与 EL 的有意差异（都在不破坏面板的前提下）：

1. ``/status`` 多带 ML 运行期信息（``pid`` / ``uptime_s`` / ``mcp_port`` /
   ``monitor_port``）——EL 只回 ``{ok, mode}``；面板只读 ``mode``，多给的键无害。
2. ``/config`` 恒走 EL 的"**无 schema**"分支：ML 没有 EL 那种 ``engine.*`` 参数族
   （逐键 schema + 单位/类型/描述），所以**如实**报 ``has_schema: false`` 并把当前
   快照键全归 ``orphans``——**不许凭空造一棵树把面板撑满**（``07`` §3.6 形状纪律）。
   ``groups`` 恒为 ``{}``；``n_keys`` 是 **schema 行数**（=0，与 EL 口径一致），
   ML 另给 ``n_snapshot_keys`` 说明"快照里其实有几个键"，免得 0 被误读成"什么都没有"。
"""
from __future__ import annotations

import http.server
import json
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .runtime import MONITOR_PORT_FILE, clear_port_file, write_port_file

#: ML 目前**没有**逐键 schema 面（EL 的 ``engine.*`` 参数族对应物）。
#: 保留成常量而不是就地写 ``False``：将来真有了 schema 面，改这一处。
_SCHEMA_ROWS: tuple = ()

#: ``set`` 归因的字段名（EL ``_config_tree`` 的 ``set`` 子对象形状）。
_ATTR_FIELDS = ("reason", "actor", "seq")


def _since_seq(path: str) -> int:
    """从查询串里取 ``since_seq``（非数字/缺失即 0，与 EL 同容错）。"""
    query = parse_qs(urlsplit(path).query)
    raw = (query.get("since_seq") or ["0"])[0]
    return int(raw) if raw.isdigit() else 0


def history_records(ml, since_seq: int = 0) -> list[dict]:
    """飞行记录仪形状的事件（EL ``_history_records`` 逐字段对齐）。

    EL 的史只记 ``after``（``value``）；``before`` 由**全史 fold 重建**（同键上一条
    的值，首写为 ``None``）。``kind`` 恒为 ``"set"``——ML 的 History 同样
    append-only、只有 set 语义（无删除事件）。
    """
    last: dict = {}
    out: list[dict] = []
    for event in ml.history.events():
        before = last.get(event.key)
        last[event.key] = event.value
        if event.seq >= since_seq:
            out.append({"seq": event.seq, "kind": "set", "actor": event.actor,
                        "target": event.key, "value": event.value,
                        "before": before, "after": event.value,
                        "reason": event.reason, "call_id": event.call_id})
    return out


def activity_records(ml, since_seq: int = 0) -> list[dict]:
    """轻量增量事件流（EL ``V2Bridge.activity`` 形状：无 before/after/reason）。"""
    return [{"kind": "set", "actor": e.actor, "target": e.key,
             "value": e.value, "seq": e.seq, "call_id": e.call_id}
            for e in ml.history.events() if e.seq >= since_seq]


def config_tree(ml) -> dict:
    """配置态结构树。

    ML 无逐键 schema（见模块 docstring 差异 2）⇒ ``has_schema=False``、
    ``groups={}``、当前快照键**全归 orphans**。这是 EL ``_config_tree`` 已有的
    分支（``src/v2_bind.py:89-97``：无引擎的纯内存桥就走这里），不是我们发明的
    降级——所以面板那条"无 schema（纯内存装配？）——下方按快照直接列出现值"
    的提示文案对 ML 同样成立、不用改。
    """
    snapshot = dict(ml.gate.snapshot)
    last: dict = {}
    for event in ml.history.events():
        last[event.key] = {"reason": event.reason, "actor": event.actor,
                           "seq": event.seq}
    known = {str(r.get("name")) for r in _SCHEMA_ROWS}
    orphans = sorted(
        ({"name": key, "value": value, "set": last.get(key)}
         for key, value in snapshot.items() if key not in known),
        key=lambda row: row["name"])
    return {"has_schema": bool(_SCHEMA_ROWS), "groups": {}, "orphans": orphans,
            "n_keys": len(_SCHEMA_ROWS), "n_snapshot_keys": len(snapshot)}


def summary_view(ml) -> dict:
    """监控概括（``Monitor.read()``）：人话摘要 + **可对账**（``disputed``）。

    ``MonitorView.summary`` 里的值是 ``Claim``（派生层的复述），必须投影成
    ``{"value", "target"}`` 再上线——否则 JSON 编码会把 dataclass 变成
    ``"Claim(value=...)"`` 这种字符串，人就再也读不出"它复述的是哪个原始键"，
    对账也就无从谈起。``disputed=True`` 原样透出：原始赢、概括存疑（两层并行的
    纪律，``mecha/monitor.py``）。``recent_runs`` 来自 History 的 ``ml.run``
    事件——**AI 跑过的运行在这里一定看得见**（哪怕没落盘）。
    """
    view = ml.monitor.read()
    summary: dict = {}
    for key, value in dict(view.summary).items():
        if hasattr(value, "value") and hasattr(value, "target"):
            summary[key] = {"value": value.value, "target": value.target}
        else:
            summary[key] = value
    return {"summary": summary, "disputed": bool(view.disputed),
            "dispute_reason": str(view.dispute_reason or "")}


def archived_runs(limit: int = 50) -> list[dict]:
    """磁盘上的实验存档（``runs/<run_id>/record.json``）。

    **与 History 是两件事**：History 记"谁在什么时候做了什么"（进程内 + journal），
    ``runs/`` 是 ML 自己的实验真值（跨进程、跨重启、跨模式都能读）。这里读后者，
    所以面板在"AI 模式重启之后"仍然列得出跑过什么。

    注意**落盘是有条件的**：``run_method`` 的 ``persist`` 默认 ``False``，GUI 批量
    路径也只存成功项 ⇒ 这里可能是空表，而 History 的 ``ml.run`` 事件仍有记录。
    面板把两者并列显示，空表不等于"没跑过"（这一条写进面板文案，别让人误读）。
    """
    try:
        from ml_toolbox.core import persistence
        records = persistence.list_records()
    except Exception:                        # noqa: BLE001 - 读不到就如实空表
        return []
    return [dict(r) for r in records[:max(0, int(limit))]]


class _Handler(http.server.BaseHTTPRequestHandler):
    """只读路由。

    ``ml`` / ``started_at`` / ``ml_pid`` / ``mcp_port`` / ``monitor_port`` 由
    :meth:`MonitorEndpoint.start` 绑到**派生出来的处理器类**上（而不是像 EL 那样
    写成类属性共享全局——同一进程起两个端点时会互相覆盖）。
    """

    ml = None
    started_at = 0.0
    ml_pid: int | None = None
    mcp_port: int | None = None
    monitor_port: int | None = None

    # ------------------------------------------------------------ 工具
    def _json(self, code: int, obj: Any) -> None:
        data = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:            # 静音（不给控制台刷屏）
        pass

    # ------------------------------------------------------------ 路由
    def do_OPTIONS(self) -> None:                    # noqa: N802 - BaseHTTPRequestHandler 约定
        # dsh 页（:3081+）轮询本端点（另一个端口）是**跨源**请求——预检放行。
        # 只读端点 + 绑回环，允许任意源不构成风险面（EL ``:130`` 同判）。
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:                        # noqa: N802
        path = urlsplit(self.path).path
        if path == "/status":
            return self._json(200, self._status())
        if path == "/activity":
            since = _since_seq(self.path)
            return self._json(200, {"ok": True,
                                    "events": activity_records(self.ml, since),
                                    "mode": self._mode()})
        if path == "/history":
            since = _since_seq(self.path)
            return self._json(200, {"ok": True,
                                    "events": history_records(self.ml, since),
                                    "mode": self._mode()})
        if path == "/config":
            return self._json(200, {"ok": True, **config_tree(self.ml)})
        # ---- ML 附加面（不破坏 EL 面板：它只认上面四条）----
        if path == "/summary":
            return self._json(200, {"ok": True, **summary_view(self.ml)})
        if path == "/runs":
            limit = 50
            query = parse_qs(urlsplit(self.path).query)
            raw = (query.get("limit") or ["50"])[0]
            if raw.isdigit():
                limit = int(raw)
            return self._json(200, {"ok": True, "runs": archived_runs(limit),
                                    "source": "runs/"})
        # 退役面如实 404：没有网页控制台、没有写/切模式路由（控制权归看门人）
        self._json(404, {"ok": False, "error": "unknown endpoint"})

    def do_POST(self) -> None:                       # noqa: N802
        # 只读端点：一切写/切模式 POST 退役（404）。人的控制权经看门人交接。
        self._json(404, {"ok": False, "error": "read-only monitor endpoint"})

    # ------------------------------------------------------------ 投影
    def _mode(self) -> str:
        mode = getattr(self.ml, "mode", "")
        return getattr(mode, "value", str(mode))

    def _status(self) -> dict:
        return {"ok": True, "mode": self._mode(), "pid": self.ml_pid,
                "uptime_s": round(max(0.0, time.time() - self.started_at), 1),
                "mcp_port": self.mcp_port, "monitor_port": self.monitor_port}


class MonitorEndpoint:
    """只读监控端点的生命周期（起线程 / 干净停 / 写端口文件）。

    与 :class:`ml_mecha.mcp_host.McpHost` 同形态：绑回环、daemon 线程、起完写
    端口文件供发现。**同步 bind**（``ThreadingHTTPServer`` 构造即 listen），
    所以没有"起来了但还没 listen"的竞态，不需要探活。
    """

    def __init__(self, ml, *, host: str = "127.0.0.1", port: int = 0,
                 port_file: str | Path | None = None, mcp_port: int | None = None,
                 log=None) -> None:
        self._ml = ml
        self._host = str(host)
        self._want_port = int(port)
        self._port_file = Path(port_file) if port_file is not None else None
        self._mcp_port = mcp_port
        self._log = log or (lambda _m: None)
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._port: int | None = None

    @property
    def port(self) -> int | None:
        return self._port

    @property
    def url(self) -> str:
        return "" if self._port is None else "http://%s:%d" % (self._host, self._port)

    def start(self) -> int:
        """起端点并返回实际端口。"""
        if self._server is not None:
            raise RuntimeError("监控端点已启动")
        # 陈旧端口文件：与 McpHost 同一论证（装配成功 ⇒ 已持写租约 ⇒ 同根无别的
        # 活宿主 ⇒ 文件必属死进程）；Windows 强杀不跑 finally，必须在这里兜一次。
        clear_port_file(self._port_file)
        handler = type("_BoundMonitorHandler", (_Handler,), {
            "ml": self._ml,
            "started_at": time.time(),
            "ml_pid": os.getpid(),
            "mcp_port": self._mcp_port,
            "monitor_port": None,
        })
        server = http.server.ThreadingHTTPServer((self._host, self._want_port), handler)
        self._port = int(server.server_address[1])
        handler.monitor_port = self._port
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True,
                                        name="ml-mecha-monitor")
        self._thread.start()
        write_port_file(self._port_file, self._port)
        self._log("[ml-mecha-monitor] %s（只读：/status /activity /history /config）"
                  % self.url)
        return self._port

    def stop(self, timeout: float = 10.0) -> None:
        """干净退出（幂等）：shutdown → server_close → join → 删端口文件。"""
        server, self._server = self._server, None
        if server is not None:
            try:
                server.shutdown()
            except Exception:                       # noqa: BLE001
                pass
            try:
                server.server_close()
            except Exception:                       # noqa: BLE001
                pass
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(max(0.1, float(timeout)))
        clear_port_file(self._port_file, self._port)
        self._port = None

    def __enter__(self) -> "MonitorEndpoint":
        self.start()
        return self

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False


def start_monitor_endpoint(ml, *, host: str = "127.0.0.1", port: int = 0,
                           port_file: str | Path | None = None,
                           mcp_port: int | None = None, log=None) -> MonitorEndpoint:
    """便利函数：起一个已 start 的 :class:`MonitorEndpoint`（对齐 EL 的同名函数）。"""
    endpoint = MonitorEndpoint(ml, host=host, port=port, port_file=port_file,
                               mcp_port=mcp_port, log=log)
    endpoint.start()
    return endpoint


__all__ = ["MonitorEndpoint", "start_monitor_endpoint", "history_records",
           "activity_records", "config_tree", "summary_view", "archived_runs",
           "MONITOR_PORT_FILE"]
