# -*- coding: utf-8 -*-
"""**AI 可见面守卫**：skill 说明书 ⇄ 真实能力面，按内容扫、不看 git。

## 为什么需要它（照 EL 仓 `checks/ai_surface_selfcheck.py` 的既有做法）

AI 拿到的"教学"只有两份东西：**工具描述**与**skill 说明书**。工具描述由
`define_tool` 构造期校验（`banned_words`），而 **skill 是自由文本 ⇒ 没人管它会不会漂**：

- 工具改名/退役了，skill 里还写着旧名 ⇒ AI 照着调，撞 `unknown_*`；
- 工具加了，skill 的清单没加 ⇒ AI **不知道它存在**（比报错更坏：能力白做了）；
- 有人把 `.dsh/skills/` 下的 junction 换成**真副本** ⇒ 从此有两份会各自漂的文件，
  而 `git` **看不见 `.dsh/`**（本仓 gitignore）⇒ `git status`/`git diff` 都发现不了。

⇒ 本守卫**只按内容读文件**，并且**期望值取自独立信源**（真注册表 + 权威版文件），
不取自被扫文件自身。

## 五组断言

- **A 登记完备**：仓内所有 `*/SKILL.md`（排除 vendor：`node_modules`/`.git`/`__pycache__`）
  ⇄ `SURFACES` 登记表**双向相等**（"漏登记"与"死登记"都算违规）。
- **B 镜像结构**：`.dsh/skills/<name>` 若存在，必须**解析到** `skills/<name>`
  （即 junction/symlink）。⚠ 断言的是**这个结构性质**，**不是**"两份逐字节相同"
  —— 两份是同一个文件时"逐字节相同"是**同一律、不携带信息**（EL 的第一版就这么写错过）。
  它抓的是"有人把 junction 换成真副本"的那一刻。
- **C 工具面一致**：skill「## 工具清单」表里的名字 == **真注册表**；且
  "共 **N** 个"的 N 也 == 注册表条数（数量声明同样是承诺）。
- **D 不存在的名字**：段落级判定——不存在/未接线的名字若以**调用形态**
  （`` `name` `` 或 `` `name( ``）出现在**非豁免语境**里 ⇒ 违规。
  豁免语境：小节标题或段落文本含"不适用/不支持/未做/没做/没接线/没有/不在/明确拒绝"等。
- **E 负例（自证非退化）**：把**合成坏内容**喂给同一批探测器，逐条必须被报出
  （否则本守卫是空转的）。

用法：``python tests/test_skill_surface.py``（退出码 0 全过 / 1 有问题）。
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: **登记表**：仓里每一份 SKILL.md 都要在这里登记（A 组双向对账）。
#: 键是相对仓根的路径；值是"它是什么"（给人看）。
#:
#: ⚠ **镜像也要登记**（照 EL 的既有做法）：`.dsh/skills/<name>` 是 AI **实际加载**的
#: 那份路径，虽然它与权威源是**同一个文件**（junction），但把它登记进来有一个
#: 真实好处——**junction 掉了 = "死登记"**，判据会当场说"跑 link_skills.py"；
#: 不登记的话它只是**静默消失**（AI 悄悄失去说明书，谁也不知道）。
SURFACES = {
    "skills/ml-toolbox/SKILL.md":
        "权威（入库；「工具清单」表 ⇄ 真注册表互钉）",
    ".dsh/skills/ml-toolbox/SKILL.md":
        "镜像（dsh 加载面；经 junction 指向权威 ⇒ 须解析到同一文件）",
}

#: 镜像 → 权威源（B 组据此断言"镜像必须解析到权威文件"）。
#: ⚠ 断言的是**这个结构性质**，**不是**"两份逐字节相同"——两份是同一个文件时
#: "逐字节相同"是**同一律、不携带信息**（EL 的第一版就这么写错过，被复现纠正）。
#: 它抓的是"**有人把 junction 换成真副本**"的那一刻：从那一刻起两份才可能各自漂。
MIRROR_OF = {
    ".dsh/skills/ml-toolbox/SKILL.md": "skills/ml-toolbox/SKILL.md",
}

#: dsh 的工作区镜像目录（gitignored）。AI 实际加载的是它下面那份。
MIRROR_DIR = ".dsh/skills"

#: 扫描时要跳过的目录（vendor / 产物 / 别的仓）。
SKIP_PARTS = ("node_modules", ".git", "__pycache__", ".venv", "old", "runtime")

#: 本能力面**不存在**的名字。出现的形态分两种：
#: - **调用形态**（`` `name` `` / `` `name( ``）：是"指令"⇒ 硬违规（除非在豁免语境）；
#: - 裸词：不判（可能出现在散文里讲道理）。
ABSENT_NAMES = (
    # EL 的命令名（我们**刻意不用**：同名不同义；宪章 R1）
    "run_actions", "reset_config", "set_config", "run_sim", "set_drive",
    # 被命名律拒过/退役的旧名
    "list_networks", "list_methods", "get_method", "list_runs", "get_run",
    # 会话层有、**工具面没接线**的能力（必须只以"没有"的形态出现）
    "predict_new", "export_prediction", "tune_method", "run_benchmark",
)

#: 豁免语境（段落级）：说到"没有/未做/不支持"时，提及这些名字是**如实登记**。
ABSENT_HEAD = re.compile(r"不适用|适用范围|已知边界|边界|不提供|未接线|退役|暂缺")
ABSENT_PARA = re.compile(
    r"不适用|不支持|不在|未做|没做|没接线|没有|明确拒绝|别以为|别硬试"
    r"|GUI\s*有|已退役|曾经|暂缺|未接线|不提供")

#: 工具清单小节标题（表就在这个标题下）。
TOOL_TABLE_HEAD = re.compile(r"^##\s*工具清单\s*$", re.M)
#: 表行：`` | `name` | … | … | ``（第一格是反引号名）
_ROW = re.compile(r"^\|\s*`([a-z_][a-z0-9_]*)`\s*\|")
#: 反引号里的裸名（调用形态）
_BACKTICK = re.compile(r"`([a-z_][a-z0-9_]*)`")
#: "共 **17** 个" 之类的数量声明
_COUNT_CLAIM = re.compile(r"共\s*\*\*(\d+)\*\*\s*个")


# ---------------------------------------------------------------- 读取与工具
def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8-sig")


def _all_skill_files() -> list[Path]:
    """仓内所有 `*/SKILL.md`（排除 vendor 与产物目录）。"""
    out = []
    for p in REPO.rglob("SKILL.md"):
        parts = set(p.relative_to(REPO).parts)
        if parts & set(SKIP_PARTS):
            continue
        out.append(p)
    return sorted(out)


def _blocks(text: str) -> list[tuple[str, str]]:
    """按空行切段，返回 ``[(所属小节标题, 段落文本)]``（D 组的判定单位）。"""
    head = ""
    out: list[tuple[str, str]] = []
    for chunk in re.split(r"\n\s*\n", text):
        stripped = chunk.strip()
        if not stripped:
            continue
        for line in stripped.splitlines():
            if line.startswith("#"):
                head = line.lstrip("#").strip()
        out.append((head, stripped))
    return out


def _slice_table(text: str) -> str | None:
    """「## 工具清单」小节的内容（切到下一个标题为止，任意层级）。"""
    m = TOOL_TABLE_HEAD.search(text)
    if not m:
        return None
    rest = text[m.end():]
    nxt = re.search(r"^#{1,6}\s", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


def _table_names(text: str) -> set[str] | None:
    section = _slice_table(text)
    if section is None:
        return None
    return {mm.group(1) for mm in (_ROW.match(l) for l in section.splitlines()) if mm}


def _registry_tool_names() -> set[str]:
    """**独立信源**（R9）：真注册表的工具名，不取自任何被扫文件。"""
    from ml_mecha.engine import MLEngine
    from ml_mecha.tools import build_ml_tools
    return {t.name for t in build_ml_tools(MLEngine())}


def scan_absent_names(text: str) -> list[str]:
    """D 组的探测器（E 组用同一份）：返回违规描述列表。"""
    bad = []
    for head, para in _blocks(text):
        if ABSENT_HEAD.search(head) or ABSENT_PARA.search(para):
            continue
        for name in _BACKTICK.findall(para):
            if name in ABSENT_NAMES:
                bad.append(f"段落「{head or '(无标题)'}」以调用形态引用了不存在的 "
                           f"名字 `{name}`，且无'没有/未做'语境")
    return bad


# ---------------------------------------------------------------- 各组断言
def check_registry_completeness() -> list[str]:
    """A 组：`*/SKILL.md` ⇄ SURFACES 双向相等。"""
    found = {str(p.relative_to(REPO)).replace("\\", "/") for p in _all_skill_files()}
    declared = set(SURFACES)
    bad = []
    for rel in sorted(found - declared):
        bad.append(f"仓里有未登记的 SKILL：{rel} ⇒ 请加进 SURFACES")
    for rel in sorted(declared - found):
        bad.append(f"SURFACES 登记了不存在的文件：{rel}（死登记）")
    return bad


def check_mirror_is_a_link() -> tuple[list[str], list[str]]:
    """B 组：每个登记过的镜像必须**存在**且**解析到**权威文件。

    ⚠ 镜像掉了要**判红**（不是提示）：那意味着 AI 悄悄失去了说明书——正是本仓
    最忌的"静默降级"。消息里给**可直接执行**的修法。
    """
    bad, notes = [], []
    for mirror_rel, auth_rel in sorted(MIRROR_OF.items()):
        mirror = REPO / mirror_rel
        auth = REPO / auth_rel
        if not auth.is_file():
            bad.append(f"权威源不存在：{auth_rel}")
            continue
        if not mirror.exists():
            bad.append(
                f"镜像 {mirror_rel} 不存在 ⇒ **AI 会静默失去这份说明书**。"
                f"修：`python checks/link_skills.py`（幂等）")
            continue
        if Path(os.path.realpath(mirror)) != Path(os.path.realpath(auth)):
            bad.append(
                f"{mirror_rel} **不是**指向 {auth_rel} 的链接"
                f"（多半是有人换成了真副本）⇒ 从此两份会各自漂，而 `git` 看不见 `.dsh/`。"
                f"修：删掉它再跑 `python checks/link_skills.py`")
            continue
        notes.append(f"{mirror_rel} 解析到 {auth_rel}（同一份文件 ✓）")
    return bad, notes


def check_tool_table_matches_registry() -> tuple[list[str], list[str]]:
    """C 组：skill 的工具清单表 + 数量声明 == 真注册表。

    ⚠ 只查**权威源**（不查镜像）：镜像与它是同一个文件，查两遍是**重复劳动**
    （而且会让人误以为"两个面都独立对过账"）。镜像的性质由 B 组负责。
    """
    bad, notes = [], []
    registry = _registry_tool_names()
    for rel in SURFACES:
        if rel in MIRROR_OF:
            continue
        text = _read(rel)
        table = _table_names(text)
        if table is None:
            bad.append(f"{rel} 没有「## 工具清单」小节 ⇒ 无法对账")
            continue
        missing = sorted(registry - table)
        extra = sorted(table - registry)
        if missing:
            bad.append(f"{rel} 的工具清单**少了** {missing}"
                       f"（AI 会因此不知道它们存在——比报错更坏）")
        if extra:
            bad.append(f"{rel} 的工具清单**多了** {extra}（这些名字不在注册表里）")
        claim = _COUNT_CLAIM.search(text)
        if claim is None:
            bad.append(f"{rel} 没有写「共 **N** 个」的数量声明 ⇒ 加一句，便于对账")
        elif int(claim.group(1)) != len(registry):
            bad.append(f"{rel} 写「共 {claim.group(1)} 个」，注册表是 {len(registry)} 个")
        notes.append(f"{rel}：清单 {len(table)} 名 / 注册表 {len(registry)} 名")
    return bad, notes


def check_absent_names() -> tuple[list[str], list[str]]:
    """D 组：不存在的名字不许以调用形态出现在非豁免语境（只查权威源）。"""
    bad, notes = [], []
    for rel in SURFACES:
        if rel in MIRROR_OF:
            continue
        hits = scan_absent_names(_read(rel))
        bad.extend(f"{rel}: {h}" for h in hits)
        notes.append(f"{rel}：登记了 {len(ABSENT_NAMES)} 个'不存在名'，违规 {len(hits)} 处")
    return bad, notes


def check_negative_control() -> list[str]:
    """E 组：合成的坏内容**必须**被同一批探测器报出（否则守卫是空转的）。"""
    bad = []
    # ① 清单少一个名字
    registry = _registry_tool_names()
    fake = "## 工具清单\n\n" + "\n".join(
        f"| `{n}` | x | y |" for n in sorted(registry)[:-1]) + "\n\n共 **17** 个\n"
    if not (_registry_tool_names() - _table_names(fake)):
        bad.append("负例①没被报出：少一个名字的清单竟然通过了")
    # ② 数量声明对不上
    fake2 = "## 工具清单\n\n" + "\n".join(
        f"| `{n}` | x | y |" for n in sorted(registry)) + "\n\n共 **99** 个\n"
    if int(_COUNT_CLAIM.search(fake2).group(1)) == len(registry):
        bad.append("负例②没被报出：数量声明对账失效")
    # ③ 调用形态引用不存在的名字（非豁免语境）
    fake3 = "## 正常用法\n\n你可以直接调 `run_actions` 一次跑完。\n"
    if not scan_absent_names(fake3):
        bad.append("负例③没被报出：非豁免语境里的 `run_actions` 竟然放行了")
    # ④ 同一句话在**豁免语境**里必须放行（反方向不误报）
    fake4 = "## 已知边界\n\n`run_actions` 未接线（工具面没有）。\n"
    if scan_absent_names(fake4):
        bad.append("负例④误报：豁免语境里的'未接线'说明被当成违规")
    return bad


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    problems: list[str] = []
    notes: list[str] = []

    for fn, label in (
            (check_registry_completeness, "A 登记完备"),
            (check_mirror_is_a_link, "B 镜像结构"),
            (check_tool_table_matches_registry, "C 工具面一致"),
            (check_absent_names, "D 不存在的名字")):
        got = fn()
        if isinstance(got, tuple):
            bad, note = got
        else:
            bad, note = got, []
        problems.extend(f"[{label}] {m}" for m in bad)
        notes.extend(f"[{label}] {m}" for m in note)

    # R8 自证非退化：扫了几个面、注册表几个工具；E 组负例必须全被报出
    n_surfaces = len(SURFACES)
    n_tools = len(_registry_tool_names())
    problems.extend(f"[E 负例] {m}" for m in check_negative_control())
    assert n_surfaces >= 1 and n_tools >= 1, "守卫自己失效了（没扫到面/工具）"

    for line in notes:
        print(f"  {line}")
    print(f"  （自证：扫了 {n_surfaces} 个面 / 注册表 {n_tools} 个工具 / "
          f"不存在名 {len(ABSENT_NAMES)} 个）")
    for line in problems:
        print(f"  ✗ {line}")
    print(f"{'✗ ' + str(len(problems)) + ' 个问题' if problems else '✓ 全部通过'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
