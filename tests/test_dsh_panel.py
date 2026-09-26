# -*- coding: utf-8 -*-
"""dsh 面板判据（TS 侧）的**仓内入口**：把 `node --test` 挂进 ML 的门禁。

为什么要有它（R10：**全绿只对判据覆盖到的路径成立**）：面板的两条关键性质
（地址路由"绝不回落"、面板"绝不空白/读不到≠空表"）写在 TS 里、跑在 `node --test` 下，
而本仓的门禁是 `python tests/run_all.py`。不把入口挂进来 ⇒ 那些判据**存在但从不在门禁里执行**
——正是 mecha README 点名的"存在但从不执行"的假绿。

跑的是 **6 份**文件（都不需要 `dsh/node_modules`，只需 PATH 上有 `node` ≥ 22.6 原生剥类型）：

* **共享 4 份**（与 `mecha/dsh-panel/` 参考实现逐字一致，**不许改**；
  `dsh/test/panel-assets.test.ts` 的常量指纹守卫钉着它们）：
  `src/dsh-panel/monitor-url.test.ts` / `monitor-client.test.ts` /
  `panel-data.test.ts` / `panel-view.test.ts`；
* **本仓自有 2 份**（不在指纹表里 ⇒ 用例数下限单独守，见下）：
  `test/panel-assets.test.ts`（指纹 + 入口级导入闭包）、
  `test/panel-extra-pages.test.ts`（附加页声明与真渲染）。

⚠ **硬依赖 `node`：缺 node 即红，不软跳过**（软跳过就是上面那种假绿；框架侧与 GP 同款纪律）。
⚠ **只看 exit code 不够**：`node --test` 在"全被跳过"时也是 0 ⇒ 本入口断言
`pass / fail / skipped` 三个计数（与 mecha `checks/dsh_panel_selfcheck.py` 同口径）。

运行：``python tests/test_dsh_panel.py``（自带 main，与仓内其它 ``test_*.py`` 同形）。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DSH_DIR = REPO / "dsh"

#: 相对 `dsh/` 的判据文件（**共享 4 份在前**，本仓自有 2 份在后）。
NODE_TEST_FILES = (
    "src/dsh-panel/monitor-url.test.ts",
    "src/dsh-panel/monitor-client.test.ts",
    "src/dsh-panel/panel-data.test.ts",
    "src/dsh-panel/panel-view.test.ts",
    "test/panel-assets.test.ts",
    "test/panel-extra-pages.test.ts",
)

#: **本仓自有**的那两份（不在参考指纹表里）。
PROJECT_TEST_FILES = ("test/panel-assets.test.ts", "test/panel-extra-pages.test.ts")

#: 本仓自有判据的**用例数下限**（写下限时的实测值 = 10：指纹 3 + 闭包 1 + 附加页 6）。
#:
#: 为什么单独守它：共享 4 份**在**指纹表里 ⇒ 改它们必红；而这两份是项目自有、**不在**表里
#: ⇒ 删掉若干用例时，指纹守卫与"全绿"都不会响——那就是"守卫被悄悄抽空"。
#: ⚠ **有意增删用例时同步改这个数**，并在下面留一行变更记录。
#:
#: 变更记录：
#: * — → **10**（2026-09-26 首次挂门禁）：指纹 3 条 + 入口级闭包 1 条 + 附加页 6 条。
PROJECT_TEST_FLOOR = 10

_COUNT_RE = re.compile(r"^\u2139\s+(tests|pass|fail|skipped)\s+(\d+)\s*$", re.M)


def _node_counts(output: str) -> dict[str, int]:
    """解析 `node --test` 的自证计数（`ℹ tests N` / `ℹ pass N` / …）。"""
    return {name: int(n) for name, n in _COUNT_RE.findall(output)}


def _run_node_tests(*files: str) -> tuple[int, str]:
    """跑 `node --test <files>`（cwd = `dsh/`），返回 ``(退出码, 合并输出)``。

    ⚠ 硬依赖 `node`：缺 node 直接抛错（由调用方转成判据失败），**不软跳过**。
    """
    node = shutil.which("node")
    if node is None:
        raise AssertionError(
            "PATH 里没有 node ⇒ 面板判据（TS 侧）跑不了。"
            "装 Node ≥ 22.6（原生剥类型，不需要 npm install）后重试。"
            "⚠ 不软跳过：软跳过＝那些判据存在但从不执行。")
    proc = subprocess.run(
        [node, "--test", *files], cwd=str(DSH_DIR), capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=300)
    return proc.returncode, f"{proc.stdout}\n{proc.stderr}"


def test_dsh_panel_criteria_pass():
    """`node --test` 跑面板判据：必须**真跑且全绿**（跳过不算过）。"""
    missing = [f for f in NODE_TEST_FILES if not (DSH_DIR / f).is_file()]
    assert not missing, f"面板判据文件缺失：{missing}（被删掉 = 守卫消失，不是「通过」）"

    code, out = _run_node_tests(*NODE_TEST_FILES)
    assert code == 0, f"面板判据红了（exit {code}）：\n{out[-2000:]}"

    # R8 非退化自证：只看 exit code 不够——**全被跳过也是 0**。
    counts = _node_counts(out)
    assert counts.get("pass", 0) > 0, \
        f"看不到通过计数（`node --test` 的报告格式变了？判据要跟着改）：\n{out[-1500:]}"
    assert counts.get("fail", 0) == 0, f"有失败计数：\n{out[-2000:]}"
    assert counts.get("skipped", 0) == 0, \
        f"有判据被跳过（{counts.get('skipped')} 条）⇒ 它们没真跑：\n{out[-1500:]}"
    # 计数自洽：总数 = 通过 + 失败 + 跳过（否则解析漏了一种态）
    total = counts.get("tests", 0)
    assert total == (counts.get("pass", 0) + counts.get("fail", 0)
                     + counts.get("skipped", 0)), f"计数不自洽：{counts}"


def test_project_panel_cases_not_hollowed():
    """⭐ **本仓自有**那两份面板判据的用例数不得低于下限（防"守卫被悄悄抽空"）。"""
    code, out = _run_node_tests(*PROJECT_TEST_FILES)
    assert code == 0, f"本仓面板判据红了（exit {code}）：\n{out[-2000:]}"
    ran = _node_counts(out).get("tests", 0)
    assert ran >= PROJECT_TEST_FLOOR, (
        f"本仓面板判据用例数 {ran} < 下限 {PROJECT_TEST_FLOOR}：**判据被删了吗**（守卫被抽空）？"
        "若确实是有意精简，请**同时**下调 PROJECT_TEST_FLOOR 并说明删的是哪一条；"
        "若是新增用例，请把下限抬到新的实测值。")


_TESTS = [test_dsh_panel_criteria_pass, test_project_panel_cases_not_hollowed]


def main() -> int:
    """不装 pytest 时的直跑入口（与仓内其它 test_*.py 同形）。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    failures = []
    for fn in _TESTS:
        try:
            fn()
            print(f"  \u2713 {fn.__name__}")
        except Exception as exc:                       # noqa: BLE001
            failures.append((fn.__name__, exc))
            print(f"  \u2717 {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(_TESTS) - len(failures)}/{len(_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
