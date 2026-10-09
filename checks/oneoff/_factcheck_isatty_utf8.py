# -*- coding: utf-8 -*-
"""事实核查：`app_entry.main` 的 utf-8 reconfigure **是否依赖 isatty**（stdout 是管道时还成立吗）。

用法：python checks/oneoff/_factcheck_isatty_utf8.py
"""
import subprocess
import sys
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
#: ⚠ **必须先把 PYTHONIOENCODING 从子进程环境里剥掉**：本仓的门禁/终端可能自己设了它
#: （`tests/run_all.py` 就设），否则测的是"注入生效"而不是"默认行为"——
#: 我第一版就踩了这个：三次都是 utf-8，看着像"管道恒定 utf-8"。
BASE_ENV = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}

code = (
    "import sys\n"
    "import app_entry\n"
    "print('BEFORE', sys.stdout.encoding, 'isatty=', sys.stdout.isatty())\n"
    "app_entry.main(['--help'])\n"
    "print('AFTER', sys.stdout.encoding, 'isatty=', sys.stdout.isatty())\n"
)
# ① stdout 是**管道**（模拟 launcher 用 PIPE 抓子进程输出）
p = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=BASE_ENV,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
out = (p.stdout or b"").decode("utf-8", "replace")
print("=== stdout=PIPE，且**没有** PYTHONIOENCODING ===")
for line in out.splitlines():
    if line.startswith(("BEFORE", "AFTER")):
        print("   ", line)

# ② 对照组：**不调 app_entry.main** 时管道里的编码是什么
code2 = ("import sys\n"
         "print('PLAIN', sys.stdout.encoding, 'isatty=', sys.stdout.isatty())\n")
p2 = subprocess.run([sys.executable, "-c", code2], env=BASE_ENV,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
print("=== 同样走管道、不调 app_entry.main（默认行为） ===")
print("   ", (p2.stdout or b"").decode("utf-8", "replace").strip())

# ③ PYTHONIOENCODING 注入后，**不调 main** 也应当是 utf-8（Q2 的修法）
env3 = {**BASE_ENV, "PYTHONIOENCODING": "utf-8"}
p3 = subprocess.run([sys.executable, "-c", code2], stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, env=env3, timeout=60)
print("=== 管道 + PYTHONIOENCODING=utf-8（不调 main） ===")
print("   ", (p3.stdout or b"").decode("utf-8", "replace").strip())
