# -*- coding: utf-8 -*-
"""``python -m ml_mecha.watch`` —— **等到终态**的看门狗（等待原语的"进程外"形态）。

## 为什么要有这个模块（用户实测反馈 §2.1，本仓优先级最高的一条）

使用者为了等一个 250 epoch 的训练，只能反复调 ``read_job``（``wait_s`` 上限 30 s），
最后**绕过整套工具**用本地 ``pwsh`` 睡 9.5 分钟再去看目录。他的原话是
"不能靠我一遍遍告诉你跑完了"。**绕过工具 = 缺能力的最硬证据。**

理想的形态是"任务终态时服务端主动向会话注入一条消息"（promise resolve）。本框架
的 MCP 面是**拉取式**的，服务端推不进去；但 harness 有另一条现成通道：
**后台作业结束时自动注入一条完成通知**。⇒ 本模块把"等"变成一个**会自己结束的进程**：

.. code-block:: text

    submit_run(...)  → 立刻返回 job_id
    pwsh run_in_background: python -m ml_mecha.watch job-N
    ...（AI 去做别的事；不等、不轮询、不做人肉转达）...
    harness 注入：「background job pwsh-XX finished [status: completed]」+ 本看门狗的终态摘要

⇒ 每次长任务的**工具调用数从"十几次轮询"降到 2**（提交 + 起看门狗），且期间 AI 自由。

## 它读什么（**只读，绝不成为第二条写通道**）

1. **权威的 MCP endpoint**（首选）：独立进程做一次最小 MCP 握手
   （``initialize`` → 收 ``Mcp-Session-Id`` → ``notifications/initialized`` → ``tools/call``），
   反复调 ``read_job``。拿到的就是**权威的真状态**：``queued``/``running``/
   ``done``/``failed``/``cancelled`` + ``run_id`` + ``error`` + 逐代数值。
   ⚠ 为什么不自己读内存：job 注册表**只活在权威进程里**，本模块是**另一个进程**。
   ⚠ 为什么不开新服务：权威已有 MCP 面，加一条"阻塞端点"要多一份实现与一份判据；
   而 MCP 面**今天就已经在跑**（少一个"要重启才生效"的交付）。
2. **盘上的 ``record.json``**（兜底，且是"终态真的落盘了"的**硬证据**）：
   job 结束时 ``MLMecha.submit`` 的落盘兜底会写 ``runs/<run_id>/record.json``。
   权威重启/句柄失效/工具面拒答时，这条仍然有效——**而且它连指标一起给**。

两条来源**互为交叉验证**：MCP 说终态而盘上还没有记录 ⇒ 本模块再等一小段
（``--settle``）直到记录真的出现（那才是"结果拿到了"）；盘上已有记录而 MCP 不可达
⇒ 也能给出终态结论，只是**如实标注**证据链（``evidence`` 字段），不假装两条都看到了。

## 它输出什么

- 过程中：进展行（**有界**：阶段变化时才打，且不密于 ``--report-every`` 秒）
  ⇒ 使用者可以用 ``job_output`` 偷看一眼，但不会因此把上下文灌满；
- 终态：一段 JSON（``ok`` / ``state`` / ``run_id`` / ``record`` / ``metrics`` /
  ``primary_metric`` / ``error`` / ``evidence`` / ``elapsed_s`` / ``polls`` / ``timed_out``），
  以及一行给人看的人话摘要。

退出码：``0`` 到达终态（含"终态但失败"，失败细节在 ``error``）；``3`` 超时仍在跑
（**不是失败**，见 ``--timeout``）；``2`` 用法错误或**句柄查不到**（附盘上最近的 run 线索）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

#: 轮询间隔（秒）。10 s 对"250 epoch / 5–8 分钟"这种量级足够密，又不打扰权威。
DEFAULT_POLL_S = 10.0

#: 进展行最短间隔（秒）——**回执预算**：看门狗的输出会进上下文，必须有界。
DEFAULT_REPORT_EVERY_S = 60.0

#: MCP 侧说终态后，再给盘上 ``record.json`` 多久出现（秒）。0 = 不等。
DEFAULT_SETTLE_S = 60.0

#: MCP 单次请求超时（秒）。权威在本地，正常毫秒级；给足不误判。
MCP_TIMEOUT_S = 30.0

#: `:latest` / 无句柄模式下，运行目录多久没写过东西就不算"在跑"（秒）。
#: 300 s 的依据：运行中的目录 mtime 随 TB 事件/checkpoint 持续刷新（实测本仓
#: 的 NQS 训练每代都写 TB）；而**残骸目录**（跑一半被 kill）会停在某个旧时刻。
DEFAULT_FRESH_S = 300.0

#: 终态状态集（与 mecha job 契约同名，不另造一套词）。
TERMINAL_STATES = frozenset({"done", "failed", "cancelled"})

#: 权威端口文件名（launcher 生成）。
PORT_FILE = ".mcp-port"

#: "这个句柄在作业表里查不到"的判据串（权威 ``unknown_job`` 的 kind 与消息）。
#: ⚠ 为什么要专门认它：**句柄查不到 ≠ 网络不通**。前者必须**立刻**给出结论
#: （作业表只活在会话里，重启即清空），否则看门狗会抱着一个不存在的句柄
#: 一直等下去——那正是"静默"的另一种形状。
_UNKNOWN_JOB_MARKERS = ("unknown_job", "没有这个 job", "not exist")


def _repo_root() -> str:
    """仓库根 = 本文件上一级目录（``ml_mecha/`` 就住在仓库根下）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def runs_dir(explicit: str = "") -> str:
    """ML 运行存档目录。

    ⚠ 与 ``ml_toolbox.core.persistence.RUNS_DIR`` **同一条规则**（环境变量优先、
    否则仓库根/runs）。这里不复用那个函数是为了**不引入 numpy**——看门狗会被
    ``pwsh`` 每次长任务起一次，导入越轻越好；而规则只有一行，重复它的代价
    远小于"看门狗因为 import 失败而静默不工作"（这条比"复用"值钱）。
    """
    if explicit:
        return os.path.abspath(explicit)
    return os.environ.get("ML_TOOLBOX_RUNS") or os.path.join(_repo_root(), "runs")


def mcp_url(explicit: str = "", port_file: str = PORT_FILE) -> str:
    """权威 MCP endpoint 的 URL：显式给 > ``.mcp-port`` 文件。"""
    if explicit:
        return explicit
    path = (port_file if os.path.isabs(port_file)
            else os.path.join(_repo_root(), port_file))
    try:
        with open(path, encoding="utf-8") as fh:
            port = fh.read().strip()
    except OSError:
        return ""
    return f"http://127.0.0.1:{port}/mcp" if port else ""


# --------------------------------------------------------------------- MCP 薄客户端
class McpError(RuntimeError):
    """MCP 侧不可用/拒答（**可读**，调用方据此降级到盘上证据）。"""


def _parse_sse(body: str) -> list[dict]:
    """streamable-http 可能回裸 JSON，也可能回 SSE（``event: message\\ndata: {...}``）。"""
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    out: list[dict] = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "data:" in chunk:
            chunk = chunk.split("data:", 1)[1].strip()
        try:
            parsed = json.loads(chunk)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


class McpClient:
    """最小 MCP 客户端：只够"握手 + 调只读工具"（stdlib，无第三方依赖）。

    ⚠ **只读用途**：本模块只调 ``read_job``。它没有、也不该有提交/取消的能力
    ——看门狗**不是第二条写通道**。

    ⚠ **会话会过期**：streamable-http 的 ``Mcp-Session-Id`` 服务端可以回收
    （长训练动辄一小时，看门狗却要一直问）。⇒ 一次调用失败**重握手再试一次**，
    仍失败才降级——否则"等了一小时之后第一次 404"会让看门狗白等。
    """

    def __init__(self, url: str, *, timeout: float = MCP_TIMEOUT_S) -> None:
        self.url = str(url)
        self.timeout = float(timeout)
        self._session = ""
        self._next_id = 0
        self._ready = False

    # -- 传输 --
    def _post(self, payload: dict) -> tuple[int, str]:
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream"}
        if self._session:
            headers["Mcp-Session-Id"] = self._session
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                sid = (resp.headers.get("Mcp-Session-Id")
                       or resp.headers.get("mcp-session-id"))
                if sid:
                    self._session = sid
                return int(resp.status), resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read().decode("utf-8", "replace")
        except OSError as exc:                                  # 连不上/超时
            raise McpError(f"连不上权威 MCP（{self.url}）：{exc}") from exc

    def _rpc(self, method: str, params: dict | None = None):
        self._next_id += 1
        rid = self._next_id
        payload = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            payload["params"] = params
        status, body = self._post(payload)
        if status >= 400:
            raise McpError(f"MCP {method} 回 HTTP {status}：{body[:200]}")
        for msg in _parse_sse(body):
            if msg.get("id") != rid:
                continue
            if "error" in msg:
                raise McpError(
                    f"MCP {method} 报错：{json.dumps(msg['error'], ensure_ascii=False)[:300]}")
            return msg.get("result")
        raise McpError(f"MCP {method} 没回结果（HTTP {status}，体 {body[:200]!r}）")

    def handshake(self) -> None:
        if self._ready:
            return
        self._rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                 "clientInfo": {"name": "ml_mecha.watch", "version": "1"}})
        # 通知无 id、无回复（服务端回 202）；失败不致命，丢了它服务端通常照答。
        try:
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except McpError:
            pass
        self._ready = True

    def call_tool(self, name: str, arguments: dict) -> dict:
        """调一个**只读**工具，返回其结构化回执（``content[0].text`` 解析出的 dict）。

        会话过期（HTTP 400/404 或没有结果）⇒ 重握手一次再试；仍失败就抛 ``McpError``
        （调用方降级到盘上证据，不静默假装"没有在跑"）。
        """
        last: McpError | None = None
        for attempt in (1, 2):
            try:
                self.handshake()
                result = self._rpc("tools/call",
                                   {"name": name, "arguments": arguments}) or {}
            except McpError as exc:
                last = exc
                self._ready = False
                self._session = ""
                if attempt == 2:
                    raise
                continue
            texts = [c.get("text", "") for c in (result.get("content") or [])
                     if isinstance(c, dict) and c.get("type") == "text"]
            if result.get("isError"):
                raise McpError(f"{name} 被工具面拒：" + (" ".join(texts)[:300] or "（无正文）"))
            for text in texts:
                try:
                    payload = json.loads(text)
                except ValueError:
                    continue
                if isinstance(payload, dict):
                    return payload
            raise McpError(f"{name} 的回执不是 JSON 对象：{' '.join(texts)[:200]!r}")
        raise last or McpError(f"{name} 调用失败")


# --------------------------------------------------------------------- 盘上证据
def read_record(runs: str, run_id: str) -> dict | None:
    """读 ``runs/<run_id>/record.json``；不存在/半写坏 ⇒ ``None``（诚实降级）。"""
    if not run_id:
        return None
    path = os.path.join(runs, str(run_id), "record.json")
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def scan_runs(runs: str, prefix: str = "") -> list[dict]:
    """列 ``runs/`` 下的子目录（新→旧）：``{run_id, mtime, has_record}``。

    只读目录项与一次 ``stat``，不读大文件（看门狗要能长期低频运行）。
    """
    try:
        names = os.listdir(runs)
    except OSError:
        return []
    out = []
    for name in names:
        if prefix and not name.startswith(prefix):
            continue
        path = os.path.join(runs, name)
        if not os.path.isdir(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        out.append({"run_id": name, "mtime": mtime,
                    "has_record": os.path.exists(os.path.join(path, "record.json"))})
    out.sort(key=lambda d: d["mtime"], reverse=True)
    return out


def newest_unfinished(runs: str, prefix: str = "", *, fresh_s: float = 0.0,
                      now: float | None = None) -> dict | None:
    """最新一个**没有** record.json 且**看起来还活着**的运行目录。

    为什么需要它：使用者常常**已经**拿到了 ``run_id``（``read_job`` 中途就回），
    或权威重启后只有盘上目录可看 ⇒ 此时"等一个新目录出现"是错的（它早就出现了）。

    ⚠ **`fresh_s` 不是可选的讲究，是必需的**：盘上真的会留下"没有 record.json 的
    旧目录"——训练跑一半被 kill、进程崩了、或旧会话留下的残骸。实测（2026-10-05）：
    本仓 ``runs/`` 里就有一个 **22 小时前**的这样的目录，而没有新鲜度过滤时
    ``:latest`` 会**一头扎进它**并永远等一个不会出现的记录。⇒ 只认"最近写过东西"
    的目录（运行中的目录 mtime 会随 TB/checkpoint 持续刷新）。
    """
    now = time.time() if now is None else float(now)
    for item in scan_runs(runs, prefix):
        if item["has_record"]:
            continue
        if fresh_s > 0 and (now - item["mtime"]) > fresh_s:
            continue
        return item
    return None


# --------------------------------------------------------------------- 看门狗主循环
def _owned_unfinished_run(client: Any, runs: str,
                          prefix: str = "") -> tuple[str, str] | None:
    """按**作业归属**找一个还没落盘的 run：回 ``(run_id, job_id)``，找不到回 ``None``。

    数据源是权威的 `jobs` 只读查询（`job_id → result_ref.run_ids`）——
    这是"反查表"而不是启发式：**能查到说明那次运行确实由某个 job 产出**。
    ⚠ 运行中途 `run_ids` 还是空的（终态才定稿）⇒ 还要靠 `read_job` 的
    `progress_latest.run_id`（那条路在 `askable` 分支里）；这里只处理"查询能给出的"。
    ⚠ 拿不到（权威不可达 / 旧版权威没有这个查询）⇒ 回 `None`，由调用方退回启发式
    **并如实标注**。
    """
    if client is None:
        return None
    try:
        got = client.call_tool("run_queries", {"actions": [{"op": "jobs",
                                                            "args": {}}]})
    except McpError:
        return None
    steps = (got or {}).get("steps") or []
    if not steps:
        return None
    output = (steps[0] or {}).get("output") or {}
    if not output.get("available"):
        return None
    unfinished = {i["run_id"] for i in scan_runs(runs, prefix) if not i["has_record"]}
    for row in output.get("jobs") or []:
        for rid in row.get("run_ids") or []:
            if str(rid) in unfinished:
                return str(rid), str(row.get("job_id") or "?")
    return None


def fmt_elapsed(seconds: float) -> str:
    """人话时长（进展行与摘要行共用，**同一事实只算一次**）。"""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def _terminal_payload(*, state: str, job_state: str, run_id: str, rec: dict | None,
                      error: str, evidence: list[str], notes: list[str],
                      polls: int, elapsed: float, timed_out: bool,
                      runs: str, state_source: str = "") -> dict:
    """终态回执的**唯一**构造点（形状只有一份 ⇒ 不会两处各写一半）。"""
    metrics = {}
    if isinstance(rec, dict):
        metrics = {k: v for k, v in (rec.get("metrics") or {}).items()
                   if isinstance(v, (int, float))}
    return {
        "ok": bool(rec is not None and not timed_out),
        "state": state, "job_state": job_state, "run_id": run_id,
        "state_source": state_source,
        "record": (os.path.join(runs, run_id, "record.json") if rec is not None else ""),
        "method": (rec or {}).get("method", ""),
        "primary_metric": (rec or {}).get("primary_metric", ""),
        "metrics": metrics,
        "error": str((rec or {}).get("error") or error or ""),
        "saved_at": (rec or {}).get("saved_at", ""),
        "evidence": evidence, "notes": notes,
        "polls": polls, "elapsed_s": round(elapsed, 1), "timed_out": timed_out,
    }


def _state_from_record(rec: dict) -> str:
    """盘上记录的终态判读（没有权威状态可用时的依据）。

    ⚠ 必须区分"失败"与"被取消"：**两者都不算成功**，但"取消"是调用方自己
    主动要的结局 ⇒ 混成 ``failed`` 会让人以为实验坏了。判据是记录里的
    ``error`` 文案（方法层写的取消说明含"取消"；mecha 侧是 ``cancelled``）。
    """
    err = str(rec.get("error") or "")
    if not err:
        return "done"
    low = err.lower()
    return "cancelled" if ("取消" in err or "cancel" in low) else "failed"


def watch(*, handle: str, url: str = "", runs: str = "", prefix: str = "",
          timeout_s: float = 0.0, poll_s: float = DEFAULT_POLL_S,
          report_every_s: float = DEFAULT_REPORT_EVERY_S,
          settle_s: float = DEFAULT_SETTLE_S, out=None) -> dict:
    """等到终态并返回**结论 dict**（CLI 只负责打印与退出码）。

    判定顺序（每轮）：
    ① 盘上已有 ``record.json`` ⇒ 终态**已落盘**（最硬的证据，含指标）；
    ② 否则问一次 MCP ``read_job``：拿状态/run_id/error；终态则再回①确认；
    ③ 都不到终态 ⇒ 打一行（有界的）进展，睡 ``poll_s`` 再来。

    ⚠ **超时不是失败**：回 ``timed_out: True`` + 现在还在跑的事实（这是"等待"，
    不该被读成"这次实验完了"）。
    ⚠ **终态但盘上没有记录**：等 ``settle_s`` 秒（落盘与 job 终态之间有个
    极小的窗口）；仍未出现 ⇒ 如实回 ``ok=False`` + ``record=""``（不编一个成功）。
    """
    out = out if out is not None else sys.stdout
    started = time.time()
    state = ""
    run_id = ""
    error = ""
    polls = 0
    notes: list[str] = []
    evidence: list[str] = []
    last_report = 0.0
    last_stage = ""
    stage = ""
    settle_deadline = 0.0
    mcp_note = ""

    def emit(line: str) -> None:
        print(line, file=out, flush=True)

    # 句柄是不是一个 run_id：**看盘上有没有这个目录**（比"猜前缀"可靠）。
    on_disk = bool(handle) and os.path.isdir(os.path.join(runs, str(handle)))
    if on_disk:
        run_id = str(handle)
        evidence.append("handle_is_run_id")

    # 没有句柄（或显式 :latest）⇒ 用盘上最新**还活着**的未落盘目录兜底。
    # ⚠ 必须带新鲜度过滤：盘上会留"没有 record.json 的旧残骸目录"（跑一半被 kill），
    #   没有过滤就会一头扎进去、永远等一个不会出现的记录（实测踩过，见
    #   `newest_unfinished` 的说明）。
    no_handle = not handle or handle == ":latest"
    client: McpClient | None = None
    if url:
        client = McpClient(url)
    else:
        mcp_note = "没有可用的 MCP 地址（--url 与 .mcp-port 都没有）⇒ 只看盘上证据"
        notes.append(mcp_note)

    if no_handle:
        # ⭐ **先按"归属"找，再退回启发式**（用户实测的纠正，2026-10-05）：
        #   盘上的运行目录并不都来自 AI 提交的 job —— 启动器自己也会跑探针 run。
        #   权威的 `jobs` 查询给出 `job_id → run_ids` 的**反查表** ⇒ 有它就不必猜。
        owned = _owned_unfinished_run(client, runs, prefix) if client is not None else None
        if owned:
            run_id, job_id = owned
            evidence.append(f"owned_by_job:{job_id}")
            emit(f"[watch] 无句柄：按**作业归属**认出 {job_id} 的 run {run_id}"
                 f"（不是猜的：来自权威的 `jobs` 查询）")
        else:
            cand = newest_unfinished(runs, prefix, fresh_s=DEFAULT_FRESH_S)
            if cand:
                run_id = cand["run_id"]
                evidence.append("newest_unfinished_run")
                emit(f"[watch] 无句柄：盯住最近还在写的运行 {run_id}"
                     f"（若你刚提交的那个还没建目录，会自动跟上；要精确就传 job_id）")
                # ⚠ **这是启发式，必须说清**：没有句柄、也拿不到作业归属时，
                #   看门狗只能按"最近还在写"猜，**猜错了会盯错对象**。
                notes.append(
                    f"`:latest` 是**启发式**（没能从权威的 `jobs` 查询拿到归属）："
                    f"它按「最近还在写、且没有 record.json」挑目录，可能选中"
                    f"**启动器自己的探针 run**（它在盘上长得一样）。"
                    f"要盯准某一个任务就传 `job_id`（或终态后的 `run_id`）。")
            else:
                emit("[watch] 无句柄、盘上没有正在写的运行：等它出现")

    #: 能不能问权威：要有 MCP 客户端、要有**真句柄**（`:latest`/无句柄都不是 job
    #: 编号，拿去问只会得到 `unknown_job` —— 那会把"还没建目录"误报成"句柄不存在"）。
    askable = bool(client is not None and handle and handle != ":latest" and not on_disk)

    #: MCP 问句：单句柄（行为与既有 read_job 完全一致）。
    def ask() -> dict:
        return client.call_tool("read_job", {"job_id": handle})   # type: ignore[union-attr]

    while True:
        now = time.time()
        polls += 1

        # ⓪ 无句柄模式：**每轮都看**盘上有没有新出现的（新鲜）运行目录
        if no_handle and not run_id:
            cand = newest_unfinished(runs, prefix, fresh_s=DEFAULT_FRESH_S, now=now)
            if cand:
                run_id = cand["run_id"]
                if "newest_unfinished_run" not in evidence:
                    evidence.append("newest_unfinished_run")
                emit(f"[watch] 盯住最近还在写的运行 {run_id}")

        # ① 盘上证据（"结果已经拿到手了"）
        rec = read_record(runs, run_id) if run_id else None
        if rec is not None:
            if "record.json" not in evidence:
                evidence.append("record.json")
            # ⚠ **回执里的 `state` 不许是过期的**：实测踩过——看门狗停在"最后一次
            #   问到的 state"上（那次是 `running`），而记录已经出现 ⇒ 回执出现
            #   `ok:true` + `state:running` 这种**自相矛盾**的组合，摘要行也跟着
            #   说"还在跑"。⇒ 权威报了终态就用权威的；否则**按记录判读**，
            #   并如实标注这条结论的来源（`state_source`）。
            if state in TERMINAL_STATES:
                final_state, src = state, "authority"
            else:
                final_state, src = _state_from_record(rec), "record"
            return _terminal_payload(
                state=final_state, job_state=state, run_id=run_id, rec=rec, error=error,
                evidence=evidence, notes=notes, polls=polls,
                elapsed=now - started, timed_out=False, runs=runs, state_source=src)

        # ② 权威的真状态（**每轮都问**：进程外没有别的办法拿到"运行中的真状态"）
        #    ⚠ 踩过的坑：曾经"一旦拿到 run_id 就不再问权威、只等盘上记录"——
        #    结果是**进展行冻在最后一次读到的阶段上**（看门狗自己把陈旧值当实时值打），
        #    而且终态判读也只能靠猜。read_job 是只读查询（毫秒级），问它不打扰训练。
        if askable:
            try:
                job = ask()
                state = str(job.get("state") or "")
                if job.get("run_id"):
                    run_id = str(job["run_id"])
                    if "mcp:run_id" not in evidence:
                        evidence.append("mcp:run_id")
                error = str(job.get("error") or "")
                if "mcp:read_job" not in evidence:
                    evidence.append("mcp:read_job")
                latest = job.get("progress_latest") or {}
                stage = str(latest.get("stage") or "")
                if state in TERMINAL_STATES and not run_id:
                    # 终态但**没有 run** ⇒ 这正是"静默死亡"的形状：必须报出来，
                    # 不能让看门狗一声不响地永远等下去。
                    evidence.append("terminal_without_run")
                    return _terminal_payload(
                        state=state, job_state=state, run_id="", rec=None,
                        error=error or "终态但没有 run_id（静默终态：既没成功也没说失败）",
                        evidence=evidence, notes=notes, polls=polls,
                        elapsed=now - started, timed_out=False, runs=runs,
                        state_source="authority")
                if state in TERMINAL_STATES and run_id and not settle_deadline:
                    settle_deadline = now + max(0.0, float(settle_s))
            except McpError as exc:
                text = str(exc)
                if any(marker in text for marker in _UNKNOWN_JOB_MARKERS):
                    # **句柄查不到**：立刻给结论 + 盘上的出路（不在这里干等）。
                    evidence.append("handle_unknown")
                    recent = [i["run_id"] for i in scan_runs(runs, prefix)][:5]
                    notes.append(
                        f"句柄 {handle!r} 在权威的作业表里查不到（作业表只活在会话里，"
                        f"重启即清空；或这个 job 从来不存在）。盘上最近的 run："
                        f"{recent or '（无）'} —— 要接着等就用 "
                        f"`python -m ml_mecha.watch <run_id>`（run_id 是盘上的稳定身份）。")
                    return _terminal_payload(
                        state="unknown_handle", job_state="", run_id=run_id, rec=None,
                        error=text[:400], evidence=evidence, notes=notes, polls=polls,
                        elapsed=now - started, timed_out=False, runs=runs)
                if text != mcp_note:
                    mcp_note = text
                    notes.append(f"MCP 暂时不可达，继续等（每轮重试）：{text[:200]}")

        # ③ 有界进展行：阶段变了就打；另外按 --report-every 给心跳（0 = 只在变化时打）
        new_stage = bool(stage) and stage != last_stage
        heartbeat = report_every_s > 0 and (now - last_report) >= report_every_s
        if new_stage or heartbeat:
            last_report = now
            last_stage = stage
            emit(f"[watch] +{fmt_elapsed(now - started)} "
                 f"state={state or 'running'} run={run_id or '-'}"
                 f"{(' | ' + stage) if stage else ''}")

        if settle_deadline and now >= settle_deadline:
            evidence.append("record_missing_after_settle")
            notes.append(f"MCP 报到终态（state={state}）后等了 {settle_s:g}s，"
                         f"盘上仍没有 record.json ⇒ 结果可能没落盘，请用 "
                         f"describe_run 核对")
            return _terminal_payload(
                state=state, job_state=state, run_id=run_id, rec=None, error=error,
                evidence=evidence, notes=notes, polls=polls,
                elapsed=now - started, timed_out=False, runs=runs)

        if timeout_s and (now - started) >= timeout_s:
            return _terminal_payload(
                # ⚠ 没有权威状态可用时**不许**写 "running"：那是在替权威下结论。
                #   "unreported" 才是实话（我们只是没等到终态）。
                state=(state or "unreported"), job_state=state, run_id=run_id, rec=None,
                error="", evidence=evidence, notes=notes, polls=polls,
                elapsed=now - started, timed_out=True, runs=runs)

        time.sleep(max(0.5, float(poll_s)))


# --------------------------------------------------------------------- CLI
def summary_line(result: dict) -> str:
    """一行**纯 ASCII** 的人话摘要（给 harness 的完成通知读的）。

    ⚠ 为什么刻意不用中文/emoji：本机控制台是 GBK，而这条摘要要穿过
    ``pwsh`` 的输出捕获进 harness 通知。实测（2026-10-05 22:0x）带 emoji 或
    ⇒/⚠ 的行会 `UnicodeEncodeError: 'gbk' codec can't encode …` ⇒
    **看门狗明明拿到了终态却崩在打印上，退出码变成 1** ——那就是新一轮"假失败"。
    ⇒ 机器字段（state/run_id/指标）全 ASCII；中文解释留在 JSON 的 ``notes`` 里
    （JSON 用 ``ensure_ascii`` 转义，任何编码下都能原样解回来）。
    """
    if result.get("timed_out"):
        return (f"[TIMEOUT] not terminal yet: job_state={result.get('job_state') or 'unreported'} "
                f"run={result.get('run_id') or '-'} "
                f"waited={fmt_elapsed(result.get('elapsed_s', 0))} "
                f"polls={result.get('polls')}")
    if not result.get("ok"):
        if "handle_unknown" in (result.get("evidence") or []):
            return (f"[UNKNOWN-HANDLE] no such job in this session: {result.get('error', '')}"
                    f" => pass a run_id instead (jobs do not survive a restart)")
        if "record_missing_after_settle" in (result.get("evidence") or []):
            return (f"[NO-RECORD] job terminal (state={result.get('state')}) but no "
                    f"record.json on disk: run={result.get('run_id') or '-'} "
                    f"=> result may not be persisted; do NOT treat as success")
        return (f"[FAIL] terminal without a usable run: state={result.get('state')} "
                f"error={result.get('error') or '(empty)'}")
    metric = result.get("primary_metric") or ""
    value = (result.get("metrics") or {}).get(metric, "")
    tail = f" {metric}={value}" if metric else ""
    src = result.get("state_source") or ""
    src_note = "" if src in ("", "authority") else f" (state_source={src})"
    return (f"[OK] terminal: state={result.get('state')}{src_note} "
            f"run_id={result.get('run_id')}{tail} "
            f"(waited={fmt_elapsed(result.get('elapsed_s', 0))}, "
            f"polls={result.get('polls')})")


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。**只读**：它不提交、不取消、不写任何域状态。"""
    # 输出统一成 UTF-8：本机控制台默认 GBK，而 JSON 里会有中文（``notes``）
    # ⇒ 不钉编码就会在**打印**这一步崩掉（实测：拿到终态却退出码 1）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):        # pragma: no cover - 非文本流
            pass
    ap = argparse.ArgumentParser(
        prog="python -m ml_mecha.watch",
        description="等一个 ML 后台任务到终态再退出（长任务的等待原语）。"
                    "配 `pwsh run_in_background` 用：它退出时 harness 会注入完成通知。",
        epilog="标准姿势（长训练的推荐流程）：\n"
               "  1) submit_run(...)            -> job_id（例如 job-18）\n"
               "  2) pwsh run_in_background:    python -m ml_mecha.watch job-18\n"
               "     ...（去干别的；不要轮询）...\n"
               "  3) 看门狗退出时 harness 自动注入完成通知；要细节用 job_output 读它\n\n"
               "⚠ 权威重启后**不要**再用旧 job 句柄（job-N 是会话内计数器，新的\n"
               "   job-18 可能指向另一个任务）—— 那时改传盘上的 run_id。\n"
               "⚠ 本命令**只读**：不提交、不取消、不写任何域状态。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("handle", nargs="?", default="",
                    help="job_id（如 job-17）/ run_id / （台账落地后的）作业名；"
                         "给 :latest 或省略 = 盯盘上最新没有记录的运行")
    ap.add_argument("--url", default="", help="权威 MCP endpoint（默认读 .mcp-port）")
    ap.add_argument("--runs-dir", default="", help="runs 目录（默认环境变量或 <仓库>/runs）")
    ap.add_argument("--prefix", default="", help="只看名字以此开头的运行目录（如 custom_network-）")
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="最多等多少秒；0（默认）= 一直等（等不到终态不退出）")
    ap.add_argument("--poll", type=float, default=DEFAULT_POLL_S, help="轮询间隔秒")
    ap.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S,
                    help="MCP 说终态后，再等盘上 record.json 出现的秒数")
    ap.add_argument("--report-every", type=float, default=DEFAULT_REPORT_EVERY_S,
                    help="心跳行最短间隔秒（0 = 只在阶段变化时打）")
    ap.add_argument("--json", action="store_true", help="只打终态 JSON（不打死摘要行）")
    args = ap.parse_args(argv)

    runs = runs_dir(args.runs_dir)
    url = mcp_url(args.url)
    try:
        result = watch(handle=args.handle, url=url, runs=runs, prefix=args.prefix,
                       timeout_s=args.timeout, poll_s=args.poll,
                       report_every_s=args.report_every, settle_s=args.settle)
    except KeyboardInterrupt:
        print("[watch] 被打断（Ctrl-C）：任务本身**没有**被取消", file=sys.stderr)
        return 130
    print(json.dumps(result, ensure_ascii=True))
    if not args.json:
        print(summary_line(result))
    if result.get("timed_out"):
        return 3
    return 0 if result.get("ok") else 2


if __name__ == "__main__":                       # pragma: no cover - CLI
    raise SystemExit(main())
