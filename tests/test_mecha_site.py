# -*- coding: utf-8 -*-
"""`ml_mecha/mecha_site.py`（站点声明）的判据。

## 为什么需要这些（每条都对着一个"会静默出错"的形态）

| 判据 | 它防的是什么 |
|---|---|
| 声明能被框架加载 | 声明写错 ⇒ **整站起不来**，而错误发生在**别的仓**（框架）里，不在本仓的测试范围 |
| `ready.port_file` 与 `runtime.MCP_PORT_FILE` 一致 | 两处各写一个字面量 ⇒ **改了名字会分叉**，而现象是"永远不就绪" |
| ⭐ 两条腿的 `ML_TOOLBOX_RUNS` 相同 | 不一致 ⇒ 切手动后看到**另一份 runs**，现象是"切了但看着是空的"，**且没有任何报错** |
| ⭐ `dataset_roots` 解析保真 | 那段逻辑从旧 launcher 搬来，**行为一旦漂移**，现象是"AI 读不到数据"或"能读不该读的" |
| ⚠ 三层架构换代：`manual` **已删** | "切手动"不再是另一条要 spawn 的腿，而是**写权切换**（`Mode`）；窗口开在**软件进程内** ⇒ 不需要再判"两条腿同源" |
| `has_gui` / `initial_mode` 在 | 新架构下"有没有人操作的面"与"初始写权"的**表达** |

⚠ **parity 用内嵌期望值**，**不 import 旧 `launcher`** —— 那个文件是**要被删掉的**
（设计文档 §7.4）⇒ 判据不能依赖它，否则删它的那天判据一起塌。
内嵌的期望值是**在旧实现还在时逐项对拍出来的**（12 个用例 + GBK 兜底，见 git 历史）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ============================================================ 声明能被框架加载

def test_site_declaration_loads_and_validates():
    """声明必须是"框架能加载 + `validate()` 通过"的。

    ⚠ 这条**跨仓**：断言的是 mecha 的 `load_sites()` 能吃下我们的声明。
    写错的代价很大（整站起不来），而且错误发生在框架侧的调用栈里。
    """
    from mecha.launch import load_sites

    os.environ["MECHA_SITE_ROOTS"] = str(REPO)
    try:
        sites = load_sites()
    finally:
        os.environ.pop("MECHA_SITE_ROOTS", None)

    mine = [s for s in sites if s.name == "ML Toolbox"]
    assert mine, f"框架没扫到本项目的声明（扫到：{[s.name for s in sites]}）"
    site = mine[0]
    ids = [svc.id for svc in site.services]
    assert "ML" in ids, ids


# ============================================================ 跨语言单一事实

def test_ready_port_file_matches_runtime_constant():
    """`ready.port_file` 必须与 `ml_mecha/runtime.py` 的常量**同一个名字**。

    ⚠ 为什么：端口文件名是**跨语言单一事实** ——
    框架起 dsh 时注入的 `MECHA_PANEL_PORT_FILES`（由**声明**里 `MonitorSpec.port_file` 解析成
    绝对路径）也指向同一个文件（⚠ 2026-10-09 起那条地址路由归**框架的 host 插件**，
    项目侧那份参数常量已删）。
    两处各写一个字面量 ⇒ 改名时必然分叉，而现象是"永远不就绪"。
    """
    from ml_mecha.mecha_site import SITE
    from ml_mecha.runtime import MCP_PORT_FILE

    svc = [s for s in SITE.services if s.id == "ML"][0]
    assert svc.ready.port_file == MCP_PORT_FILE, (
        f"声明里是 {svc.ready.port_file!r}，runtime 里是 {MCP_PORT_FILE!r} ⇒ 分叉了")


# ============================================================ ⭐ 两条腿看同一份账

def test_runs_root_is_still_pinned_by_the_software_itself():
    """`ML_TOOLBOX_RUNS` 仍**由软件自己钉死**（换代后语义变了，但这条不能丢）。

    ⚠ 旧判据是"svc 与 manual **两条腿**必须指向同一处" —— 那是防止
    "切手动后看到另一份 runs"。**新架构下不需要它了**：GUI 是**软件进程内的窗口**，
    与软件天然共享同一份状态（公理 A1）⇒ 那个失败模式**结构上不存在**。

    但 `ML_TOOLBOX_RUNS` **这条钉死本身仍必须留**：它决定 `runs/` 落在哪
    （`core/runs.py`），若留给继承，软件进程会读到**别处**的一份 ⇒ 现象是
    "我的运行记录没了"、**且无任何报错**。
    """
    from ml_mecha.mecha_site import SITE

    svc = [s for s in SITE.services if s.id == "ML"][0]
    runs = svc.env.get("ML_TOOLBOX_RUNS")
    assert runs, "没钉死 ML_TOOLBOX_RUNS（不许留给继承）"
    assert Path(runs).is_absolute(), f"必须是绝对路径：{runs!r}"


def test_manual_is_gone_and_new_fields_are_in_place():
    """⭐ 换代断言：`manual` **不许回来**；`has_gui` / `initial_mode` **必须在**。

    ⚠ 为什么把"`manual` 不存在"也写成判据：**旧形态是"要 spawn 的进程"**，
    而它一旦回来，就会重新出现"两条腿各持一个写宿主"的那套问题
    （框架侧已删该字段 ⇒ 传它会 `TypeError`，这里在**声明层**再钉一次）。
    """
    from ml_mecha.mecha_site import SITE

    svc = [s for s in SITE.services if s.id == "ML"][0]
    assert not hasattr(svc, "manual"), "`manual` 已被三层架构删除，不许回来"
    assert svc.has_gui is True, "ML 有 PyQt5 窗口 ⇒ has_gui 必须为 True"
    #: ⚠ 比 `.value`：`Mode` 是 `str` 枚举，`str(Mode.AI)` 是 `'Mode.AI'` 不是 `'ai'`
    assert getattr(svc.initial_mode, "value", None) == "ai", (
        f"initial_mode 应为 AI（从 mecha 启动永远 AI 组）：{svc.initial_mode!r}")


#: ⭐ **在旧实现还在时逐项对拍出来的期望值**（12 用例 + GBK 兜底）。
#: ⚠ 这些期望值**故意内嵌**，不 import 旧 `launcher`（它要被删）。
_PARITY: tuple[tuple[str, str, list[str]], ...] = (
    ("空文件", "", []),
    ("只有注释", "# 注释\n\n# 又一行注释\n", []),
    ("普通绝对路径", "D:\\code-nosync\\Mechine learning\\runs\n",
     ["D:\\code-nosync\\Mechine learning\\runs"]),
    ("带 BOM", "\ufeffD:\\code-nosync\\Mechine learning\\runs\n",
     ["D:\\code-nosync\\Mechine learning\\runs"]),
    ("多个路径", "D:\\code-nosync\\Mechine learning\\runs\nD:\\code-nosync\\mecha\n",
     ["D:\\code-nosync\\Mechine learning\\runs", "D:\\code-nosync\\mecha"]),
    ("含注释与空行", "# 头\n\nD:\\code-nosync\\mecha\n\n# 尾\n",
     ["D:\\code-nosync\\mecha"]),
    ("相对路径", "runs\n", [str((REPO / "runs").resolve())]),
    ("前后空白", "   D:\\code-nosync\\mecha   \n", ["D:\\code-nosync\\mecha"]),
    ("路径含中文", "D:\\desktop\\量子机器学习\\施工\n",
     ["D:\\desktop\\量子机器学习\\施工"]),
    ("正斜杠", "D:/code-nosync/mecha\n", ["D:\\code-nosync\\mecha"]),
    ("末尾无换行", "D:\\code-nosync\\mecha", ["D:\\code-nosync\\mecha"]),
)


@pytest.mark.parametrize("name,text,want", _PARITY, ids=[c[0] for c in _PARITY])
def test_dataset_roots_parsing_is_faithful(tmp_path, monkeypatch, name, text, want):
    """`dataset_roots()` 的解析行为必须与搬迁前的旧实现**逐项一致**。

    ⚠ 搬这段逻辑时最容易丢的三件事（都在旧实现的注释里记着）：
    1. **`utf-8-sig` 优先**（本机历史文件带 BOM）；
    2. **GBK 兜底**（见下一条判据）；
    3. **每个路径带一次旗标**（`--dataset-roots` 是 `action="append"`）——
       旧实现曾经只回路径，导致 `unrecognized arguments` ⇒ **权威 exit=2**，
       现象是"改完打不开了"，**且只在白名单非空时才发作**。
    """
    from ml_mecha import mecha_site

    f = tmp_path / "dataset_roots.txt"
    f.write_text(text, encoding="utf-8")
    monkeypatch.setattr(mecha_site, "_DATASET_ROOTS_FILE", f)

    got = mecha_site.dataset_roots()
    assert got == want, f"{name}: 期望 {want}，得到 {got}"


def test_dataset_roots_accepts_gbk(tmp_path, monkeypatch):
    """GBK 编码的白名单也要读得出来（本机历史文件有 GBK 的）。"""
    from ml_mecha import mecha_site

    f = tmp_path / "dataset_roots.txt"
    f.write_bytes("D:\\code-nosync\\mecha\n".encode("gbk"))
    monkeypatch.setattr(mecha_site, "_DATASET_ROOTS_FILE", f)

    assert mecha_site.dataset_roots() == ["D:\\code-nosync\\mecha"]


def test_dataset_roots_expands_env_and_never_guesses(tmp_path, monkeypatch):
    """环境变量要展开；读不出就**回空**（fail closed，**不猜任何默认根**）。"""
    from ml_mecha import mecha_site

    monkeypatch.setenv("MLTB_PARITY_PROBE", "C:\\parity-probe")
    f = tmp_path / "dataset_roots.txt"
    f.write_text("%MLTB_PARITY_PROBE%\n", encoding="utf-8")
    monkeypatch.setattr(mecha_site, "_DATASET_ROOTS_FILE", f)
    assert mecha_site.dataset_roots() == ["C:\\parity-probe"]

    # 文件不存在 ⇒ 空（不猜默认根）
    monkeypatch.setattr(mecha_site, "_DATASET_ROOTS_FILE", tmp_path / "nope.txt")
    assert mecha_site.dataset_roots() == []


# ============================================================ 就绪判据不许被"放宽"

def test_ready_declares_any_http_response_with_reason():
    """`ok_codes=()` 是**有理由的放宽**，不是漏写；且不许退化成 `tcp=True`。

    ⚠ 实测依据：本项目 MCP 端点的 `GET /mcp` 回 **406**（非 2xx），
    而该端点上没有回 2xx 的只读路由（`/summary` 等属**监控端点**，另一个端口）。
    ⇒ 必须能表达"**有 HTTP 响应即证明服务在听**"。

    ⚠ 同时钉住**不许改成 `tcp=True`**：`tcp=True` 只证明"端口有人"，
    旧实现的注释明确写过"TCP 连得上还不够 —— 同一端口上可能是别的东西在听"。
    """
    from ml_mecha.mecha_site import SITE

    svc = [s for s in SITE.services if s.id == "ML"][0]
    assert svc.ready.ok_codes == (), (
        f"ok_codes 应为 ()（任意 HTTP 响应即算在听），实际 {svc.ready.ok_codes!r}")
    assert svc.ready.tcp is False, (
        "不许退化成 tcp=True：那只证明端口有人，证明不了是 HTTP 服务在听")
    assert svc.ready.probe and "/mcp" in svc.ready.probe, svc.ready.probe


def test_initial_mode_is_declared_and_the_remaining_gap_is_recorded():
    """`initial_mode` 已声明；⭐ **它现在真的接通了**；⚠ **剩下的缺口**被如实记下。

    ## 这条判据的来历（**它自己换过一次对象**）

    原先它钉的是"**缺口被写下来**"：当时 `app_entry.py` **不读** `MECHA_INITIAL_MODE`
    （注释里自认"今天这里还没读"）⇒ 声明与实现之间有一段缺口，**假装生效比不生效更坏**。

    ✅ 现在 `--mcp` 用**框架的读取口**（`mecha.site.initial_mode_from_env`）读并应用它
    （真进程判据在 `tests/test_control_endpoint.py`：给 `human` ⇒ 软件起来后写权就是 `human`）。
    ⚠ 所以本判据**换对象**：从"钉住缺口"改为"钉住**已接通**"，同时**仍要求**
    剩下的缺口（**没有人手面**）被如实记下 —— 那条**今天还真的存在**：
    ML 的 GUI 是**独立进程** ⇒ 面板「切到手动」会"写权成功、窗口开不了"。

    ⚠ 教训：**判据会过期**。缺口闭合了就必须**换对象**（R13：删机制/闭缺口时，
    把它守的承诺改写成**行为级**判据），否则它会一直红着、或更坏 —— 被改成"永远绿"。
    """
    from ml_mecha import mecha_site

    src = Path(mecha_site.__file__).read_text(encoding="utf-8")
    assert "MECHA_INITIAL_MODE" in src, "声明里要写明框架会用这个变量送值"
    #: ⭐ 已接通：声明里必须**不再**声称"还没读"
    assert "还没读" not in src and "没有读它" not in src, (
        "`initial_mode` 已接通（`app_entry.py` 读并应用了）⇒ 声明不许再写『还没读』")
    #: ⚠ 剩下的缺口仍必须如实记下（今天真实存在：ML 没有人手面）
    assert "人手面" in src or "没有手动面" in src, (
        "必须**如实记下**『ML 没有人手面 ⇒ 切手动会写权成功、窗口开不了』这段缺口")
