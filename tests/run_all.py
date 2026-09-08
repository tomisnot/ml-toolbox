# -*- coding: utf-8 -*-
"""一键回归门（checklist.md C 关）：核心 + UI + 冒烟。

运行：python tests/run_all.py
截图自查（视觉关）请单独跑：python checks/ui_shot.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["smoke_test.py", "test_regressions.py", "test_ui.py", "test_opt.py"]


def main():
    failed = []
    for s in SCRIPTS:
        print(f"\n########## {s} ##########")
        r = subprocess.run([sys.executable, os.path.join(HERE, s)],
                           capture_output=False)
        if r.returncode != 0:
            failed.append(s)
    print("\n================ 回归门汇总 ================")
    if failed:
        print("❌ 失败:", ", ".join(failed))
        return 1
    print("✅ 全部通过（smoke + regressions + ui + opt）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
