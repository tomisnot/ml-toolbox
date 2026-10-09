# -*- coding: utf-8 -*-
"""**后台 job 的 stdout/stderr 可读化**（把"服务端 print"变成一条能查的事实）。

## 为什么需要它（用户实测 A5）

网络代码里的 `print("[resume] …")` / `print("[snap] step=…")` 是**重要的运行时诊断**
（证明续训/快照真的生效了），但工具面**一条都拿不到**：

- job 是**权威进程里的一条 daemon 线程**（`multi_jobs` → `assembly.JobService.submit`
  → `JobRegistry._run`），它的 `print` 只落在**权威的控制台**上；
- 框架的 `Job.to_dict()`（`mecha/providers/jobs.py`）**没有任何日志字段**；
- ⇒ 使用者只能靠 `run_id` + 指标**反推**一个布尔事实（"快照生效了吗"），一次往返。

## 三条实现纪律（都是踩过的坑）

1. ⚠ **不能直接 `contextlib.redirect_stdout`**：它换的是**进程级** `sys.stdout`，
   而 job 是**线程** ⇒ 并发跑两个 job 时输出会**串台**（A 的 print 出现在 B 的日志里）。
   ⇒ 正解：进程内装**一个** tee 代理，按 **`threading.get_ident()` 分桶**。
2. ⚠ **透传不能省**：控制台是**给人看的**，tee 必须把内容原样写给真流
   （只"收走"就等于把 print 从人的视野里删掉）。`__getattr__` 也要代理，
   否则 `sys.stdout.reconfigure(...)`（`app_entry.py` / `panel_web.py` 都在用）会炸。
3. ⚠ **日志本身绝不许把训练搞崩**（A42/A7② 的教训：GBK 控制台 + `⚠` ⇒
   `UnicodeEncodeError`）：写入**吞掉自己的异常**、必要时 `errors="replace"`；
   tee 自己的任何问题都只能变成"少一行日志"，不能变成"任务失败"。

## 边界（如实说）

- **只覆盖 `print` 走的 `sys.stdout`/`sys.stderr`**；C 扩展直接写 fd 1/2 的
  输出**收不到**（那是另一个能力，本模块不假装有）。
- 环形缓冲**每 job 有界**（`MAX_LINES`），只留尾巴 —— 日志是"看现场"用的，
  不是审计流水（真流水该写盘）。
- 同一个线程里的**非 job 输出**（例如 HTTP 服务自己的日志）不会被记进任何 job：
  线程没有登记 sink 时 tee 只透传。
"""
from __future__ import annotations

import os
import sys
import threading
from collections import deque
from typing import Any

#: 挂在 `ExecutionContext` 上的属性名（**扩展槽**，与 `engine._PROGRESS_SIDECAR` 同一手法）。
#: 为什么走 sidecar：`read_job` 本来就从 `ml.jobs.get(job_id).context` 读富载荷
#: （`progress`/`run_id` 那条路）⇒ 不新增任何跨层协议、不动外部框架。
LOG_SIDECAR = "ml_job_logs"

#: 每个 job 最多留多少行（尾巴）。200 行 × 约 80 字符 ≈ 16 KB/job，内存可忽略。
#: 可用环境变量 `MLTB_JOB_LOG_LINES` 覆盖（人侧调参，不必改源码）。
#: ⚠ **必须是环形缓冲**（`deque(maxlen=…)`），不是"先全存再截"：长训练里网络代码
#: 可能**每个 batch 都 print**（6 小时级 ⇒ 几十万行），无界缓冲会把内存吃干，
#: 而失败会伪装成「torch DLL 装载失败 / Insufficient system resources」——离根因很远。
MAX_LINES = 200


def _default_cap() -> int:
    """默认桶容量：`MLTB_JOB_LOG_LINES`（读不出/非法就用 `MAX_LINES`）。"""
    raw = os.environ.get("MLTB_JOB_LOG_LINES", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return MAX_LINES


#: 单行最长（防止有人 `print` 一整份数组把内存撑爆）：超出部分截断并注明。
MAX_LINE_CHARS = 500

_LOCK = threading.Lock()
#: 当前线程 → 它的日志桶（**按 thread ident 分桶**：不许串台，也各自有界）。
_SINKS: dict[int, "_Sink"] = {}
#: 已安装的 tee（幂等用）。
_TEE: "_Tee | None" = None


class _Sink:
    """一个 job 的日志桶：**有界环形缓冲 + 真正的总行数**。

    ⚠ 这两件事**必须分开**（2026-10-07 使用者点出的风险）：
    - 缓冲是"看现场"用的 ⇒ **有界**（`deque(maxlen=cap)`，老的自动丢）；
    - 总行数是**事实** ⇒ 只是一个整数，无界也不吃内存。

    只回 `len(缓冲)` 会把"被截掉了多少"**藏起来** —— 使用者就分不清
    「这个 job 只打印了 200 行」与「打印了 2 万行、只留了尾巴」，
    而后者才是"日志设施真的在工作"的证据。
    """

    def __init__(self, cap: int) -> None:
        self._buf: deque = deque(maxlen=max(1, int(cap)))
        self._total = 0
        #: 未成行的尾巴（`print("x")` 会分两次 write：值 + "\n"）
        self._partial = ""

    @property
    def cap(self) -> int:
        return int(self._buf.maxlen or 0)

    def push_text(self, text: Any, prefix: str = "") -> None:
        """把一次 `write` 的内容**按换行组装成行**再入桶（**保序、不重复计数**）。

        ⚠ 为什么不能"一次 write = 一行"（实测踩过，Q1 判据当场红）：
        `print("[x] 1")` 实际调用 `write("[x] 1")` **和** `write("\\n")` **两次**
        ⇒ 按 write 计数会把 20 000 行记成 **40 000 行**，而且每次 print 还会多出一条
        空行（`"\\n".splitlines() == ['']`）。⇒ 必须自己攒 `_partial`：
        `write("abc") + write("\\n")` = **一行** `abc`。
        """
        try:
            with _LOCK:
                data = self._partial + str(text)
                parts = data.split("\n")
                self._partial = parts.pop()          # 最后一段可能还没结束
                #: ⚠ 没换行的巨量输出（比如 `print(long_array, end="")`）也不能撑爆内存
                if len(self._partial) > MAX_LINE_CHARS:
                    parts.append(self._partial[:MAX_LINE_CHARS]
                                 + f"…（截断，原长 {len(self._partial)}）")
                    self._partial = ""
                for line in parts:
                    self._buf.append(prefix + line)
                    self._total += 1
        except Exception:                                      # noqa: BLE001
            pass

    def flush_partial(self, prefix: str = "") -> None:
        """收尾：把"没有换行结尾的最后一段"也记成一行（否则 `end=""` 的输出会丢）。"""
        try:
            with _LOCK:
                if self._partial:
                    self._buf.append(prefix + self._partial)
                    self._total += 1
                    self._partial = ""
        except Exception:                                      # noqa: BLE001
            pass

    def tail(self, n: int) -> tuple[list[str], int]:
        with _LOCK:
            total = self._total
            lines = list(self._buf)
        k = max(0, int(n))
        return (lines[-k:] if k else []), total

    def size(self) -> int:
        """当前缓冲里有多少行（判据用：它**不许随输出量增长**）。"""
        with _LOCK:
            return len(self._buf)


class _Tee:
    """`sys.stdout` / `sys.stderr` 的**代理**：透传给真流 + 记进当前线程的桶。

    ⚠ **永不抛**：`write`/`flush` 自己的异常一律吞掉。理由不是"图省事"，而是
    "诊断设施不许改变主流程的成败"——与 `tb_log` 的三处 fail-safe 同一条纪律。
    """

    def __init__(self, stream: Any, name: str = "out") -> None:
        self._stream = stream
        self._name = name

    # ---- 真实写入
    def write(self, text: Any) -> int:
        try:
            out = self._stream.write(text)
        except Exception:                                      # noqa: BLE001
            # ⚠ 编码类错误的兜底（A42：GBK 控制台遇到 `⚠` 会抛）：
            #   换成"能打出来的样子"再试一次；再失败就放弃这一行（**不上升**）。
            try:
                enc = getattr(self._stream, "encoding", None) or "utf-8"
                safe = str(text).encode(enc, "replace").decode(enc, "replace")
                out = self._stream.write(safe)
            except Exception:                                  # noqa: BLE001
                out = len(str(text))
        self._record(text)
        return out if isinstance(out, int) else len(str(text))

    def flush(self) -> None:
        for fn in (lambda: self._stream.flush(),):
            try:
                fn()
            except Exception:                                  # noqa: BLE001
                pass

    def _record(self, text: Any) -> None:
        try:
            sink = _SINKS.get(threading.get_ident())
            if sink is None:
                return                       # 非 job 线程：只透传（不记，也不报错）
            # ⚠ **保序**：`push_text` 在桶的锁内按 `split("\n")` 的先后逐行 append
            #   （同线程 FIFO）；`deque(maxlen=…)` 满了丢最老的，顺序不受影响。
            sink.push_text(text, f"[{self._name}] ")
        except Exception:                                      # noqa: BLE001
            pass

    # ---- 代理其余属性（`reconfigure`/`encoding`/`isatty`/`fileno`…）
    def __getattr__(self, item: str) -> Any:
        return getattr(self._stream, item)

    def isatty(self) -> bool:                                  # 显式：很多库会问
        try:
            return bool(self._stream.isatty())
        except Exception:                                      # noqa: BLE001
            return False


def install() -> bool:
    """装 tee（**幂等**，返回是否本次真的装了）。

    ⚠ 装的是**当前** `sys.stdout`/`sys.stderr` 之外的那一层代理：装完 `print` 照常
    到控制台，同时进桶。**只在真的有 job 要跑时才装**（`capture` 会调它）——
    不装的话，纯库用法（不开后台任务的进程）行为**逐位不变**。
    """
    global _TEE
    with _LOCK:
        if isinstance(sys.stdout, _Tee) or isinstance(sys.stderr, _Tee):
            return False
        _TEE = _Tee(sys.stdout, "out")
        sys.stdout = _TEE
        sys.stderr = _Tee(sys.stderr, "err")
        return True


def uninstall() -> None:
    """卸掉 tee（**只在本进程自己装过时**才还原，避免踩到别人的代理）。"""
    global _TEE
    with _LOCK:
        if _TEE is None:
            return
        if sys.stdout is _TEE or isinstance(sys.stdout, _Tee):
            sys.stdout = _TEE._stream
        if isinstance(sys.stderr, _Tee):
            sys.stderr = sys.stderr._stream
        _TEE = None


class capture:
    """**job 边界**上的上下文管理器：把当前线程的输出收进这个 job 的桶。

    用法（`assembly.JobService.worker` 里一行）：

    ```python
    with job_logs.capture(ctx):        # ctx = ExecutionContext
        receipt = self.engine.run(job_spec, context=ctx)
    ```

    - 进入：装 tee（幂等）+ 把当前线程登记到新桶 + 在 `ctx` 上挂 sidecar；
    - 退出：**注销线程**（桶留在 sidecar 上供 `read_job` 读）。
    """

    def __init__(self, context: Any, *, max_lines: int | None = None) -> None:
        self._ctx = context
        #: ⚠ 默认容量走 `_default_cap()`（可被 `MLTB_JOB_LOG_LINES` 覆盖）
        self._max = max(1, int(max_lines if max_lines is not None else _default_cap()))
        self._ident: int | None = None
        self._sink: "_Sink | None" = None

    def __enter__(self) -> "capture":
        install()
        self._ident = threading.get_ident()
        #: ⭐ **每个 job 一个自己的有界桶**：并发 job 既不会互相挤掉，
        #:   总量也是 `cap × 同时在跑的 job 数`（**有界**），不是"一个共享大缓冲"。
        self._sink = _Sink(self._max)
        with _LOCK:
            _SINKS[self._ident] = self._sink
        #: sidecar 的形状与 `_PROGRESS_SIDECAR` 同风格（都是"给 read_job 读的富载荷"）
        try:
            setattr(self._ctx, LOG_SIDECAR, {"sink": self._sink,
                                            "max_lines": self._max})
        except Exception:                                      # noqa: BLE001
            pass                              # 挂不上就是"看不到日志"，不该拖垮 job
        return self

    def __exit__(self, *exc: Any) -> bool:
        #: 收尾：把"没有换行结尾的最后一段"也记成一行（`print(x, end="")` 别丢）
        if self._sink is not None:
            self._sink.flush_partial("[out] ")
        if self._ident is not None:
            with _LOCK:
                _SINKS.pop(self._ident, None)
        return False                          # **不吞异常**：job 的失败语义不变

    # ---- 读侧（`read_job` 用）
    @staticmethod
    def tail(context: Any, n: int = 10) -> tuple[list[str], int, int]:
        """回 ``(最后 n 行, 总行数, 桶容量)``；没装/没记过 ⇒ ``([], 0, 0)``（**如实空**）。

        ⚠ 第二个值是**总行数**（不是缓冲长度）：使用者靠它区分
        "只打印了 N 行" 与 "打印了 2 万行、只留了尾巴"（后者才说明日志设施在工作）。
        """
        try:
            side = getattr(context, LOG_SIDECAR, None) or {}
            sink = side.get("sink")
            if sink is None:
                return [], 0, 0
            lines, total = sink.tail(n)
            return lines, int(total), int(getattr(sink, "cap", 0) or 0)
        except Exception:                                      # noqa: BLE001
            return [], 0, 0


__all__ = ["LOG_SIDECAR", "MAX_LINES", "MAX_LINE_CHARS",
           "install", "uninstall", "capture"]
