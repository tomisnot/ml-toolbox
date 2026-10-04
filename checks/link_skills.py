# -*- coding: utf-8 -*-
"""把 `skills/<name>/` 挂成 `.dsh/skills/<name>`（**junction**），供 dsh 发现。

## 为什么需要这一步（照 EL/GP 的既有做法）

- **权威源入库**：`skills/<name>/SKILL.md`（git 看得见、可 review、可 diff）；
- **dsh 只认 `.dsh/skills/`**：那是**工作区**目录（本仓 gitignore），AI 实际加载的
  是它下面那份；
- 两者用 **junction** 连起来 ⇒ 磁盘上**只有一份文件**，改哪边都是改同一份。

⚠ 为什么写成脚本而不是"手工建一次"：EL 的交接文档里有一句警告——
"`.dsh/skills/re0-mecha/SKILL.md` 是**另一份文档**……**改 SKILL 别只改一处**"。
那是手工维护导致的分叉。本脚本把它变成**可重放**的一步，并由
`tests/test_skill_surface.py` 的 B 组钉住"镜像路径必须**解析到**权威文件"
（即：不许有人把 junction 换成真副本）。

用法：``python checks/link_skills.py``（幂等；已正确则不动）
退出码：0 = 全部就位；1 = 有问题（并打印原因）。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_DIR = REPO / "skills"
DST_DIR = REPO / ".dsh" / "skills"


def _is_junction_to(link: Path, target: Path) -> bool:
    """`link` 是否**已经**是指向 `target` 的目录链接（junction/symlink 都算）。"""
    if not link.exists():
        return False
    try:
        # 解析后与目标的真实路径相同 ⇒ 就是同一份文件
        return Path(os.path.realpath(link)) == Path(os.path.realpath(target))
    except OSError:
        return False


def _make_junction(link: Path, target: Path) -> None:
    """建目录 junction。Windows 上用 `mklink /J`（**不需要管理员**）。

    ⚠ 不用 `os.symlink`：Windows 上建目录 symlink 需要开发者模式/管理员，
    而 junction（`/J`）在普通权限下可用，且对只读用途完全够。
    """
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                       check=True, capture_output=True)
    else:
        os.symlink(target, link, target_is_directory=True)


def link_all() -> int:
    if not SRC_DIR.is_dir():
        print(f"没有 {SRC_DIR}（本仓没有 skill 源）⇒ 无事可做")
        return 0
    skills = sorted(p for p in SRC_DIR.iterdir() if (p / "SKILL.md").is_file())
    if not skills:
        print(f"{SRC_DIR} 下没有含 SKILL.md 的 skill ⇒ 无事可做")
        return 0
    DST_DIR.mkdir(parents=True, exist_ok=True)
    problems = []
    for src in skills:
        dst = DST_DIR / src.name
        if _is_junction_to(dst, src):
            print(f"  ✓ {dst.relative_to(REPO)} -> {src.relative_to(REPO)}（已就位）")
            continue
        if dst.exists():
            # ⚠ **不自动删**：它可能是别人手工放的**真副本**（那正是要防的分叉）。
            #   要替换必须由人确认 ⇒ 这里只报，不动。
            problems.append(
                f"{dst.relative_to(REPO)} 已存在但**不是**指向 {src.name} 的链接"
                f"（可能是手工副本）⇒ 请人工确认后删除，再重跑本脚本")
            continue
        try:
            _make_junction(dst, src)
            print(f"  + {dst.relative_to(REPO)} -> {src.relative_to(REPO)}（新建）")
        except Exception as exc:                             # noqa: BLE001
            problems.append(f"建 {dst.relative_to(REPO)} 失败：{type(exc).__name__}: {exc}")
    for line in problems:
        print(f"  ✗ {line}")
    print(f"{len(skills)} 个 skill，{len(problems)} 个问题")
    return 1 if problems else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(link_all())
