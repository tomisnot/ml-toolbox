# -*- coding: utf-8 -*-
"""ProcessObjective（评估侧接入）：外部程序本身就是黑盒目标。

"每跑一次得一条信息"的主形态：参数写进命令行/配置文件 -> 启动进程 ->
从 stdout 正则或结果文件解析分数。与 ML 完全无关。

约定：
- 非零退出码 / 超时 / 解析不到 -> 抛异常，runner 记 status='failed'（P10）；
  失败现场（命令 + stdout/stderr 尾部）存 last_error，事后排错用；
- 参数注入模板：`{key}` 占位（str.format），值经 str() 转换；
  模板里可直接写格式说明符（`{amp:.4f}`），或构造时传 fmt={"amp": ".4f"}
  让裸 `{amp}` 用默认格式（真实模拟软件常拒绝 17 位小数）；
- 需要 json 文件时用 {params_file}（写临时文件后给路径）；
- 编码：子进程输出按 utf-8 解码 + errors=replace（Windows 管道默认
  GBK 会把模拟软件的 UTF-8 中文输出炸成 UnicodeDecodeError）。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile

import numpy as np

from .contracts import Objective, ParamSpace

# 分数解析兜底：支持科学计数法（1.23e-4 / -5.6E+2）
_SCORE_RE = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


class ProcessObjective(Objective):
    """cmd_template 占位符：{key}（各参数）、{params_file}（json 路径）。

    parse:        stdout 正则（含 1 个捕获组）；默认匹配 "score: <数>"。
    result_file:  非空则从该文件读 json 取 score_key；支持 {key} 占位
                  （模拟软件按参数命名输出文件的情形）。
    fmt:          {参数名: 格式说明符}，如 {"amp": ".4f"}——模板里裸
                  {amp} 渲染时套用；模板已写 {amp:.4f} 则模板优先。
    """

    def __init__(self, cmd_template: str, space: ParamSpace,
                 parse: str = r"score[:=]\s*" + _SCORE_RE,
                 cwd: str = "", result_file: str = "",
                 score_key: str = "score",
                 fmt: dict | None = None, env: dict | None = None,
                 name: str = "process", minimize: bool = True,
                 timeout: float = 120.0, shell: bool = True,
                 constraint_signal: str = "",
                 on_infeasible: str = "censor", penalty_mult: float = 10.0):
        """on_infeasible: 约束违反点的处理策略（需配 constraint_signal 识别）。
            censor    -> 记 infeasible，喂 GP-BO 的 cEI 分类器（constrain 开关）
            penalize  -> 返回自适应惩罚分（最差可行分×penalty_mult），
                         目标 GP 全量点可学，薄可行带下通常比 cEI 省评估
            raise     -> 当普通失败（failed，censored；默认无 constraint_signal 时）
        """
        super().__init__(name=name, space=space, minimize=minimize,
                         has_constraints=bool(constraint_signal))
        self.cmd_template = cmd_template
        self.parse = re.compile(parse)
        self.cwd = cwd or None
        self.result_file = result_file
        self.score_key = score_key
        self.fmt = dict(fmt or {})
        self.env_extra = dict(env or {})
        self.timeout = float(timeout)
        self.shell = shell
        self.constraint_signal = constraint_signal  # 识别"约束拒绝"的信号子串
        self.on_infeasible = on_infeasible
        self.penalty_mult = float(penalty_mult)
        self._worst_ok = -float("inf")      # 历史最差可行分（自适应惩罚基准）
        self.last_cmd = ""                 # UI 展示/排错用
        self.last_error = ""               # 最近一次失败现场（含输出尾部）

    def _penalty(self) -> float:
        base = self._worst_ok if np.isfinite(self._worst_ok) and self._worst_ok > 0 \
            else 1.0
        return base * self.penalty_mult + 1.0

    def evaluate(self, params: dict):
        try:
            s = self._run_once(params)
            if np.isfinite(s):
                self._worst_ok = max(self._worst_ok, s)
            return s
        except Exception as e:
            if (self.on_infeasible == "penalize"
                    and self.is_constraint_error(e)):
                return self._penalty()
            raise

    def is_constraint_error(self, exc: Exception) -> bool:
        """黑盒主动拒绝（约束违反）vs 真崩溃：按信号子串区分。

        例如模拟软件对违反 π 脉冲条件的输入以退出码 3 + 特定 stderr 拒绝，
        传 constraint_signal="违反" 即可把这些点标为 infeasible（喂 cEI 分类器），
        而超时/段错误等仍算 failed（censored）。
        """
        return bool(self.constraint_signal
                    and self.constraint_signal in str(exc))

    # ------------------------------------------------ 渲染
    def _render(self, tpl: str, fmt: dict) -> str:
        """模板渲染：裸 {key} 套用 self.fmt[key] 默认格式说明符。"""
        def repl(m):
            key, spec = m.group(1), m.group(2)
            if spec is None:
                spec = self.fmt.get(key, "")
            v = fmt[key]
            return format(v, spec) if spec else str(v)
        return re.sub(r"\{(\w+)(?::([^{}]*))?\}", repl, tpl)

    # ------------------------------------------------ 主流程（真正的进程执行）
    def _run_once(self, params: dict):
        fmt = dict(params)
        tmp = None
        try:
            if "{params_file}" in self.cmd_template:
                fd, tmp = tempfile.mkstemp(suffix=".json", prefix="optp_")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(params, f, ensure_ascii=False)
                fmt["params_file"] = tmp.replace("\\", "/")   # 模板内统一正斜杠
            cmd = self._render(self.cmd_template, fmt)
            self.last_cmd = cmd
            env = None
            if self.env_extra:
                env = dict(os.environ)
                env.update(self.env_extra)
            try:
                r = subprocess.run(cmd, capture_output=True, cwd=self.cwd,
                                   timeout=self.timeout, shell=self.shell,
                                   env=env)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"超时 {self.timeout}s（可调 timeout 或减小单次仿真规模）")
            out = self._dec(r.stdout)
            err = self._dec(r.stderr)
            if r.returncode != 0:
                raise RuntimeError(f"退出码 {r.returncode}: {(err or out)[-400:]}")
            if self.result_file:
                rf = self._render(self.result_file, fmt)
                with open(rf, encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                return float(data[self.score_key])
            m = self.parse.search(out)
            if not m:
                raise RuntimeError(f"stdout 未匹配解析式: {out[-400:]}")
            return float(m.group(1))
        except Exception as e:
            self.last_error = f"cmd={self.last_cmd}\n{type(e).__name__}: {e}"
            raise
        finally:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    @staticmethod
    def _dec(b) -> str:
        """bytes -> str：utf-8 优先，GBK 兜底（Windows 控制台程序常见）。"""
        if not b:
            return ""
        if isinstance(b, str):
            return b
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return b.decode("gbk", errors="replace")
