# -*- coding: utf-8 -*-
"""一键回归门（外部资产 `D:\\desktop\\UI design\\checklist.md` 的 C 关）：核心 + UI + 冒烟 + 优化。

运行：python tests/run_all.py            （fast 层，目标 <1min）
全量：set MLTB_SLOW=1 && python tests/run_all.py
截图自查（视觉关）请单独跑：python checks/ui_shot.py

## 两阶段并行（不是一把全并行）——**按实测根因**

各脚本彼此独立（各自进程、各自 runs/ 子目录），所以**能**并行；但**不能全一起并行**：

* **阶段 1（重负载）**：`smoke_test` / `test_ui` / `test_opt` / `test_regressions` ——
  冒烟要跑 59 方法 × 6 任务（含 torch/lightgbm/xgboost，会把 CPU 打满），优化套件也重；
* **阶段 2（时延敏感）**：`test_dsh_panel` / `test_ml_mecha` / `test_ml_capabilities` /
  `test_ml_delivery` —— 起 MCP 服务、监控端点与真 dsh，大量 HTTP 往返，**自带按秒计的预算**。

**根因（实测，不是推测）**：全 8 套一起并行时，`test_ml_delivery.py::test_monitor_shapes_match_el_cockpit_contract`
（自带 90s 预算）在**总墙钟 129.8s** 的那一轮**超时**，而它**单独跑 4.3s 通过**、第二次全量
（97.2s）也全绿 ⇒ 那是 **CPU 争用造成的假红**，不是真挂。
⇒ **不抬阈值**（抬阈值＝把响亮的失败改成沉默的通过），改**按资源画像分阶段**：
重活先并行跑完，再跑时延敏感的。噪声化的门禁等于没有门禁（本项目老账）。

## 输出与判据

输出按脚本缓冲、按**阶段内顺序**打印，保持可读。每脚本的用时是**自己**的墙钟
（起进程前记时、communicate 返回时收尾），另给每阶段与总墙钟。
mecha 存在时额外纳入 `test_ml_mecha.py` / `test_ml_capabilities.py` / `test_ml_delivery.py`
（缺 mecha 则**明确打印跳过原因**，不假绿）。
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

#: 阶段 1：CPU 饱和型（重活）。彼此并行。
PHASE_HEAVY = ["smoke_test.py", "test_ui.py", "test_opt.py", "test_regressions.py"]

#: 阶段 2：时延敏感型（起服务/端点/dsh、HTTP 往返、`node --test`）。彼此并行，且在阶段 1 之后。
#: `test_dsh_panel.py` **无条件**在列：它跑的是**仓内**的共享面板判据（`node --test`），
#: 不依赖 mecha 仓；**硬依赖 `node`**（缺 node 即红——不把"存在但从不执行"当绿）。
PHASE_SENSITIVE = ["test_dsh_panel.py"]

#: mecha host adapter 集成判据。mecha 是同级仓；缺它时**不静默跳绿**：
#: 成功行按实跑集合拼装，并显式打印跳过原因（最终验收审查 M1）。
MECHA_ROOT = os.environ.get("MECHA_ROOT", r"D:\code-nosync\mecha")
MECHA_PRESENT = os.path.isdir(MECHA_ROOT)
if MECHA_PRESENT:
    PHASE_SENSITIVE += ["test_ml_mecha.py", "test_ml_capabilities.py",
                        "test_ml_delivery.py"]

PHASES = [("重负载", PHASE_HEAVY), ("时延敏感", PHASE_SENSITIVE)]
SCRIPTS = [s for _, group in PHASES for s in group]


def _run_phase(name, group, child_env):
    """并行跑一个阶段；返回 ``[(script, 秒, 输出, 退出码)]``。"""
    print(f"\n===== 阶段「{name}」（{len(group)} 套，并行）=====", flush=True)
    started = {}
    procs = {}
    for s in group:
        started[s] = time.time()
        procs[s] = subprocess.Popen(
            [sys.executable, os.path.join(HERE, s)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=child_env)
    rows = []
    for s, p in procs.items():
        out, _ = p.communicate()
        rows.append((s, time.time() - started[s], out or "", p.returncode))
    for s, dt, _out, rc in rows:
        print(f"  {dt:6.1f}s  {'✓' if rc == 0 else '✗'}  {s}", flush=True)
    return rows


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
    failed = []
    outputs = {}
    timing = []
    phase_wall = []
    for name, group in PHASES:
        t0 = time.time()
        rows = _run_phase(name, group, child_env)
        phase_wall.append((name, time.time() - t0))
        for s, dt, out, rc in rows:
            timing.append((s, dt))
            outputs[s] = out
            if rc != 0:
                failed.append(s)

    for s in SCRIPTS:
        print(f"\n########## {s} ##########")
        print(outputs[s].rstrip())

    print("\n================ 回归门汇总（两阶段并行墙钟） ================")
    for name, dt in phase_wall:
        print(f"  阶段[{name}] {dt:6.1f}s")
    for s, dt in sorted(timing, key=lambda x: -x[1]):
        print(f"  {dt:6.1f}s  {s}")
    print(f"  ------ 总墙钟 {time.time() - t_all:.1f}s")
    if failed:
        print("❌ 失败:", ", ".join(failed))
        return 1
    ran = " + ".join(s[:-3] for s in SCRIPTS)     # 按实跑集合拼装，不硬编码
    if not MECHA_PRESENT:
        print(f"⚠️ 跳过 mecha 集成判据（MECHA_ROOT 不存在：{MECHA_ROOT}）")
    print(f"✅ 全部通过（{ran}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
