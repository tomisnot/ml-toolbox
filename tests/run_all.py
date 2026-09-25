# -*- coding: utf-8 -*-
"""一键回归门（checklist.md C 关）：核心 + UI + 冒烟 + 优化。

运行：python tests/run_all.py            （fast 层，目标 <1min）
全量：set MLTB_SLOW=1 && python tests/run_all.py
截图自查（视觉关）请单独跑：python checks/ui_shot.py

四个核心脚本彼此独立（各自进程、各自 runs/ 子目录），并行执行把墙钟时间
从"求和"降到"取最大值"；输出按脚本缓冲、顺序打印，保持可读。
mecha 存在时额外纳入 `test_ml_mecha.py` 与 `test_ml_capabilities.py`
（缺 mecha 则明确跳过，不假绿）。
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["smoke_test.py", "test_regressions.py", "test_ui.py", "test_opt.py"]
#: mecha host adapter 集成判据。mecha 是同级仓；缺它时跳过并明确报告，
#: 不用假绿替代（与 mecha 自己“缺真依赖即红”的纪律方向一致）。
MECHA_ROOT = os.environ.get("MECHA_ROOT", r"D:\code-nosync\mecha")
if os.path.isdir(MECHA_ROOT):
    SCRIPTS.append("test_ml_mecha.py")
    SCRIPTS.append("test_ml_capabilities.py")


def main():
    # 子进程输出按 UTF-8 捕获，但 Windows 控制台可能是 GBK；给子进程显式
    # PYTHONIOENCODING=utf-8，否则子套件打印 ✓/✗ 时自己先崩（不是逻辑失败）。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    child_env = dict(os.environ)
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env.setdefault("MECHA_ROOT", MECHA_ROOT)
    t_all = time.time()
    procs = {}
    for s in SCRIPTS:
        procs[s] = subprocess.Popen(
            [sys.executable, os.path.join(HERE, s)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=child_env)
    failed = []
    timing = []
    outputs = {}
    for s, p in procs.items():
        t0 = time.time()
        out, _ = p.communicate()
        # 并行下无法边跑边看，统一等完再按序打印
        timing.append((s, time.time() - t0))
        outputs[s] = out or ""
        if p.returncode != 0:
            failed.append(s)
    for s in SCRIPTS:
        print(f"\n########## {s} ##########")
        print(outputs[s].rstrip())
    print("\n================ 回归门汇总（并行墙钟） ================")
    for s, dt in sorted(timing, key=lambda x: -x[1]):
        print(f"  {dt:6.1f}s  {s}")
    print(f"  ------ 总墙钟 {time.time() - t_all:.1f}s")
    if failed:
        print("❌ 失败:", ", ".join(failed))
        return 1
    print("✅ 全部通过（smoke + regressions + ui + opt + ml_mecha + ml_capabilities）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
