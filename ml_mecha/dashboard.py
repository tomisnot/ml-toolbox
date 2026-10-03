# -*- coding: utf-8 -*-
"""TensorBoard 辅进程（sidecar）：起一个只读的 Web，给人看那种"全套面板"。

## 为什么是辅进程，而不是把界面搬进我们的面板

TensorBoard 的 Web 是 **HTML + 资源 + 流**，而我们的只读路由是 **JSON-only**
（``mecha.cockpit`` 恒 ``json.dumps`` 后回 ``application/json``，见它的 ``_route_error``
与 ``send_header("Content-Type", …)``）。两者装不到一起。

⇒ 分工：**数据由我们写（事件文件）、界面由 TensorBoard 出、数值两边都能读**。
- 人：**多开一个窗口**看 TensorBoard（它自己的全套面板）；
- AI：读同一个事件文件（见 ``ml_mecha/tb_series.py``）⇒ **一份数据两种消费者**。

## ⚠ 三条纪律

1. **只读**：TensorBoard 只挂 ``--logdir``，**不写我们的数据**；我们的写入仍在落盘路径上。
2. **窗口不归训练管**：起停是"人要的时候"，训练进程不需要知道窗口在不在
   （所以它是**独立进程 + 端口文件发现**，不是训练的一部分）。
3. **失败响亮但不致命**：起不来时回 ``{ready: false, reason}``，**绝不假装起了**
   ——面板/调用方据此显示可读原因（与 `/runs` 的"不吞异常"同一条纪律）。
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

#: 端口文件名（项目根；与 `.mcp-port` / `.cockpit-port` / `.web-port` 同族）。
TB_PORT_FILE = ".tb-port"

#: 起服务后等端口文件出现的最长秒数。
START_TIMEOUT_S = 20.0


def _free_port() -> int:
    """挑一个可绑定的回环端口（**不指定固定端口** ⇒ 并存的实例不会互相踩）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class TensorBoardSidecar:
    """起/停一个 TensorBoard 子进程。**幂等**：已在跑就复用。"""

    def __init__(self, root, *, port_file: str | Path | None = None,
                 log=None) -> None:
        self.root = Path(root)
        self.port_file = Path(port_file) if port_file else self.root / TB_PORT_FILE
        self._log = log
        self._proc: subprocess.Popen | None = None
        self._port: int | None = None

    # ------------------------------------------------------------ 事实面
    @property
    def port(self) -> int | None:
        """当前端口（没起 ⇒ ``None``）。**已退出的进程不算**。"""
        if self._proc is not None and self._proc.poll() is not None:
            self._proc = None
            self._port = None
        return self._port

    @property
    def url(self) -> str | None:
        p = self.port
        return f"http://127.0.0.1:{p}/" if p else None

    @property
    def running(self) -> bool:
        return self.port is not None

    def _say(self, msg: str) -> None:
        if self._log is not None:
            try:
                self._log(msg)
            except Exception:
                pass

    # ------------------------------------------------------------ 起停
    def start(self, *, logdir: str | Path | None = None) -> dict:
        """起服务（或复用已在跑的）。**不抛**：失败回 ``{ready: false, reason}``。

        @param logdir - 看哪个目录；默认看 ``root``（父目录 ⇒ 能在界面里挑 run）。
        """
        if self.running:
            return {"ready": True, "port": self.port, "url": self.url,
                    "reused": True}
        target = Path(logdir) if logdir else self.root
        if not target.is_dir():
            return {"ready": False,
                    "reason": f"目录不存在：{target}（还没有 run 可看）"}
        port = _free_port()
        cmd = [sys.executable, "-m", "tensorboard.main",
               "--logdir", str(target), "--host", "127.0.0.1",
               "--port", str(port), "--reload_interval", "2"]
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=str(self.root),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            return {"ready": False,
                    "reason": f"起 TensorBoard 失败：{type(exc).__name__}: {exc}"}

        # 等它真的在听（**探端口**，不靠 sleep 猜）
        deadline = time.time() + START_TIMEOUT_S
        while time.time() < deadline:
            if self._proc.poll() is not None:
                self._proc = None
                return {"ready": False,
                        "reason": "TensorBoard 进程立刻退出了（多半是没装或参数不对）"}
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    self._port = port
                    self._publish(port)
                    self._say(f"[tb] {self.url}（logdir={target}）")
                    return {"ready": True, "port": port, "url": self.url,
                            "logdir": str(target)}
            time.sleep(0.2)
        self.stop()
        return {"ready": False, "reason": f"等了 {START_TIMEOUT_S:.0f}s 端口仍未就绪"}

    def stop(self) -> None:
        """停服务并按内容清端口文件（**只删自己写的那个**）。"""
        proc, port = self._proc, self._port
        self._proc, self._port = None, None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
            except Exception:
                pass
        # ⚠ 传 port 进去：`_clear` 要靠它判断"这个文件是不是我写的"。
        # 早先的写法先置 `self._port = None` 再清 ⇒ 判不出归属 ⇒ 端口文件残留（实测抓到）。
        self._clear(port)

    # ------------------------------------------------------------ 端口文件
    def _publish(self, port: int) -> None:
        """写端口文件（先清陈旧、再写自己的）——规则与 `mecha.portfile` 同族。"""
        try:
            self.port_file.parent.mkdir(parents=True, exist_ok=True)
            self.port_file.write_text(str(int(port)), encoding="utf-8")
        except Exception as exc:
            self._say(f"[tb] 端口文件写入失败：{type(exc).__name__}: {exc}")

    def _clear(self, port: int | None) -> None:
        """**按内容**清：只删仍是自己写的那个端口，别误删接管者写的。"""
        if port is None:
            return
        try:
            if not self.port_file.is_file():
                return
            cur = self.port_file.read_text(encoding="utf-8").strip()
            if cur == str(int(port)):
                self.port_file.unlink()
        except Exception:
            pass
