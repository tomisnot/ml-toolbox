# -*- coding: utf-8 -*-
"""ProcessObjective（评估侧接入）：外部程序本身就是黑盒目标。

"每跑一次得一条信息"的主形态：参数写进命令行/配置文件 -> 启动进程 ->
从 stdout 正则或结果 json 解析分数。与 ML 完全无关。

约定：
- 非零退出码 / 超时 / 解析不到 -> 抛异常，runner 记 status='failed'（P10）；
- 参数字典注入模板：`{key}` 占位（str.format），值经 str() 转换；
  需要 json 文件时用 {params_json}（写入临时文件后给路径，占位 {params_file}）。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile

from .contracts import Objective, ParamSpace


class ProcessObjective(Objective):
    """cmd_template 里可用占位符：{key}（各参数）、{params_file}（json 路径）。"""

    def __init__(self, cmd_template: str, space: ParamSpace,
                 parse: str = r"score[:=]\s*(-?\d+(?:\.\d+)?)",
                 cwd: str = "", result_file: str = "",
                 name: str = "process", minimize: bool = True,
                 timeout: float = 120.0, shell: bool = True):
        super().__init__(name=name, space=space, minimize=minimize)
        self.cmd_template = cmd_template
        self.parse = re.compile(parse)
        self.cwd = cwd or None
        self.result_file = result_file        # 非空则从该文件读 json 取 "score"
        self.timeout = float(timeout)
        self.shell = shell
        self.last_cmd = ""                     # UI 展示/排错用

    def evaluate(self, params: dict):
        fmt = {k: v for k, v in params.items()}
        tmp = None
        try:
            if "{params_file}" in self.cmd_template:
                fd, tmp = tempfile.mkstemp(suffix=".json", prefix="optp_")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(params, f, ensure_ascii=False)
                fmt["params_file"] = tmp
            cmd = self.cmd_template.format(**fmt)
            self.last_cmd = cmd
            r = subprocess.run(cmd, capture_output=True, text=True,
                               cwd=self.cwd, timeout=self.timeout,
                               shell=self.shell)
            if r.returncode != 0:
                raise RuntimeError(
                    f"退出码 {r.returncode}: {(r.stderr or r.stdout)[-200:]}")
            if self.result_file:
                with open(self.result_file, encoding="utf-8") as f:
                    data = json.load(f)
                return float(data["score"])
            m = self.parse.search(r.stdout or "")
            if not m:
                raise RuntimeError(f"stdout 未匹配解析式: {(r.stdout or '')[-200:]}")
            return float(m.group(1))
        finally:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
