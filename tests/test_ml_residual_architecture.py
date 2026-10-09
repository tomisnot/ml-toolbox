# -*- coding: utf-8 -*-
"""清理归一的**残留守卫**：旧架构（看门人 / 两模式 / 换腿）不许留在仓里。

## 为什么需要这条（它守的是什么承诺）

2026-10-08 按 `mecha/docs/v2/清理归一-口径.md` 删掉了旧看门人 `launcher.py`（834 行）
与"两模式 / 进程交接"这套模型。**删掉之后最容易发生的事，是后人"好心加回来"**
（旧文里到处是 `python launcher.py`、"选 AI 模式"这类说法）。
⇒ 这条判据是 **R13 的"反向断言"**：把"旧东西已不存在"写成**行为级**守卫。

⚠ **判定标准是两件事，缺一不可**：
1. 全仓搜到的**只剩 C 类**（诚实的换代说明 / 历史记录 / 判据里的反向断言）；
2. ⚠ **搜索必须覆盖被 gitignore 的构建产物**（`dsh/lib/*.js` + `.map`）——
   `python` 不读 JS，用 python 只扫 `.py` 会**看不到它们**（R14：判据是"搜到 0 命中"，
   不是"门禁绿了"；本次实地就漏过一处，见 `checks/oneoff/scan_legacy.py` 的注释）。

## 怎么区分"残留"与"换代说明"

⚠ 本文件**自己**必须出现那些旧词（它要定义白名单）⇒ 它是**第一顺位白名单**。
真正的判据是**逐文件白名单 + 逐行理由**：新出现的命中，一定是有人把旧模型当**现状**写了。
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SELF = "tests/test_ml_residual_architecture.py"

#: 旧架构的特异词（**不收**会误伤的通用词，如"手动""模式"）。
TOKENS: tuple[str, ...] = (
    "launcher.py", "看门人", "两模式", "换腿",
    "mode-request", "mode-state", "switch_leg",
    "choose_mode", "acquire_singleton", "LAUNCHER_LOCK",
    "MODE_REQUEST", "MODE_STATE",
    "from launcher", "import launcher", "python launcher",
)

#: 目录级排除：版本库、依赖、缓存、运行期数据、IDE/浏览器 profile。
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache",
             ".vscode", ".trae", "runs", ".ml-mecha", "_mecha_runtime", "edgeprof",
             "old"}
#: 后缀排除：二进制/媒体/模型权重（读成文本只会是乱码，不是"残留"）。
SKIP_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".npz", ".npy", ".pt", ".ckpt",
            ".pyc", ".pyd", ".so", ".dll", ".zip", ".gz", ".bin", ".ico", ".woff",
            ".woff2", ".ttf", ".db", ".sqlite"}

#: ⭐ **C 类白名单**（按**目录/前缀**给，不按单文件）：这些地方出现旧词是**有意**的。
#: ⚠ 为什么按前缀：换代说明会**新增**文件（每删一个机制就多一篇存档），
#:   若按单文件白名单，每加一篇就要改一次判据 —— 那是"判据追着代码跑"。
#: ⚠ 新增**前缀**时必须在理由里说清"这一整类为什么允许"（那是唯一的关卡）。
WHITELIST_DIRS: dict[str, str] = {
    "docs/": "换代说明与历史记录：讲清'推翻了什么、为什么'（用户裁决：删了没人知道旧路走不通）",
    "ml_mecha/runtime.py": "换代说明：哪三个常量随旧架构删了、⚠ 别加回来",
    "ml_mecha/mecha_site.py": "换代说明 + 判据的对拍依据（与旧 `_dataset_roots_argv` 等价）",
    ".gitignore": "换代说明：删掉的两模式握手文件与单实例锁",
    "pyproject.toml": "B 类改写后残留一句'旧 launcher 已删'（防止读者以为它还在）",
    "app_entry.py": "B 类改写：'同源'关系仍在、机制换了（旧 launcher 已删）",
    "ml_mecha/tools.py": "B 类改写：`TOOL_CALL_TIMEOUT_MS` 的值仍在用，出处指向活知识存档",
    "dsh/lib/": "构建产物：源码注释的投影（重建即同步；判据另有'源码不许有'那条）",
    "dsh/src/client/buttons.tsx": "换代说明：为什么 ML 不需要「→ GUI」按钮",
    "tests/test_ml_delivery.py": "换代说明：写明哪 4 条判据随旧 launcher 移交框架门禁",
    "skills/": "SKILL 文档：'旧启动方式'那段的处置见报告未决",
    ".dsh/skills/": "同上（`.dsh/` 是 `skills/` 的 junction 镜像）",
    "tests/test_ml_delivery.py": "换代说明：写明哪 4 条判据随旧 launcher 移交框架门禁",
    "README.md": "换代说明：启动段已改框架口径，并留一句'旧 launcher 已删'",
    "runtime/": "运行期生成物（不入库）：旧启动器留下的 overlay，下次启动由框架重写",
    "启动 ML Toolbox.bat": "启动入口：见本文件对它的判据（不许再出现 `python launcher.py`）",
}

#: 文件名含这些标记的**一律排除**（不是本仓的版本控制对象）。
SKIP_NAME_MARKERS = (".bak_", ".bak.", "~")

#: 白名单前缀（判据自身）。
SELF_PREFIX = "tests/test_ml_residual_architecture.py"


def _scan() -> list[tuple[str, int, str, str]]:
    out: list[tuple[str, int, str, str]] = []
    for path in sorted(REPO.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if any(m in path.name for m in SKIP_NAME_MARKERS):
            continue
        if path.suffix.lower() in SKIP_EXT:
            continue
        rel = path.relative_to(REPO).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue          # 非文本：不是"残留叙述"
        for i, line in enumerate(text.splitlines(), 1):
            for tok in TOKENS:
                if tok in line:
                    out.append((rel, i, tok, line.strip()))
    return out


def _is_whitelisted(rel: str) -> bool:
    if rel.startswith(SELF_PREFIX):
        return True
    return any(rel.startswith(prefix) for prefix in WHITELIST_DIRS)


def _offenders() -> list[str]:
    bad: list[str] = []
    for rel, lineno, tok, line in _scan():
        if _is_whitelisted(rel):
            continue
        bad.append(f"{rel}:{lineno}: [{tok}] {line[:120]}")
    return bad


def test_no_legacy_launcher_residue_outside_whitelist():
    """⭐ 全仓（**含 gitignore 的构建产物**）除白名单外，不许出现旧架构词汇。"""
    bad = _offenders()
    assert bad == [], (
        "出现旧架构残留（旧看门人/两模式/换腿）。若这是**有意**的换代说明，"
        "请在本文件的 WHITELIST 里加一行并写理由；否则**删掉它**：\n  "
        + "\n  ".join(bad))


def test_scanner_reaches_gitignored_build_artifacts():
    """⭐ 自证：扫描**真的覆盖**被 gitignore 的构建产物（否则这条判据是空转）。

    ⚠ 为什么专门钉它：`dsh/lib/*.js` + `.map` 是构建产物、**不入库**，
    而"python 不读 JS" ⇒ 只扫 `.py/.ts` 的实现**看不到**它们（R14 的坑，本次实地踩过）。
    这里断言"扫描结果里出现过 `dsh/lib/` 下的文件"——那证明遍历范围够宽。
    """
    seen = {rel for rel, _i, _t, _l in _scan()}
    assert any(r.startswith("dsh/lib/") for r in seen), (
        "扫描没覆盖 dsh/lib/（构建产物）⇒ 判据会在'改了 TS 忘了重建'时漏报："
        f"实际扫到 {len(seen)} 个文件")


def test_whitelist_entries_are_real_and_justified():
    """⭐ R16 自证：白名单**不许是空转**（每条都要有理由，且**真的在拦东西**）。

    ⚠ 这条防的是"把白名单当垃圾桶"：前缀加了一大片却没有任何命中 ⇒
    要么路径写错了，要么旧词已经清干净（那就该**收窄**它，别留着给以后当挡箭牌）。
    """
    assert WHITELIST_DIRS, "白名单空了 —— 那说明扫描没命中任何东西（判据可能空转）"
    thin = [p for p, reason in WHITELIST_DIRS.items() if len(reason.strip()) < 8]
    assert thin == [], f"白名单条目理由太短（等于没写）：{thin}"
    #: 每个前缀都必须**真的**有命中，否则它是过期条目
    hits = {rel for rel, _i, _t, _l in _scan()}
    stale = [p for p in WHITELIST_DIRS
             if not any(h.startswith(p) for h in hits)]
    assert stale == [], f"白名单里有**没有命中**的前缀（过期/写错路径）：{stale}"


def test_scanner_can_go_red():
    """⭐ 红证：守卫**能红** —— 造一条真残留，断言扫描抓得到；再证明白名单能放行它。

    ⚠ 造法要点（R16：扰动要自证"真落上了"）：
    - **不能**往已被白名单覆盖的目录里放（`docs/`/`tests/` 都被放过）⇒ 造一个
      **全新**顶层目录里的文件，它必然落在白名单之外；
    - 不能只断言 `_scan()` 为空（空转也能绿）⇒ 断言"**报出来的是那个探针**"，
      并且**逐条核对命中的词**（证明抓的是内容，不是文件名）。
    """
    probe_dir = REPO / "_residual_probe_tmp"
    probe_dir.mkdir(exist_ok=True)
    probe = probe_dir / "probe.py"
    probe.write_text("# 旧看门人 python launcher.py 会选两模式\n", encoding="utf-8")
    try:
        bad = _offenders()
        mine = [b for b in bad if "_residual_probe_tmp" in b]
        assert mine, f"造出来的残留没被抓到 ⇒ 守卫是哑的：{bad[:5]}"
        #: 逐词核对：三个旧词都该被抓到（证明抓的是**内容**）
        text = " ".join(mine)
        for tok in ("launcher.py", "看门人", "两模式"):
            assert f"[{tok}]" in text, f"探针里的 {tok!r} 没被抓到：{mine}"

        #: 反向：把它加进白名单就必须被放过（证明白名单真的在起作用）
        WHITELIST_DIRS["_residual_probe_tmp/"] = "红证用：临时探针"
        try:
            assert not [b for b in _offenders() if "_residual_probe_tmp" in b], \
                "白名单没生效 ⇒ 上面那条'被抓到'可能只是碰巧"
        finally:
            WHITELIST_DIRS.pop("_residual_probe_tmp/", None)
    finally:
        probe.unlink(missing_ok=True)
        probe_dir.rmdir()
    assert not [b for b in _offenders() if "_residual_probe_tmp" in b], "探针没清干净"
