# -*- coding: utf-8 -*-
"""`ml_mecha/mecha_site.py`（站点声明）的判据。

## 为什么需要这些（每条都对着一个"会静默出错"的形态）

| 判据 | 它防的是什么 |
|---|---|
| 声明能被框架加载 | 声明写错 ⇒ **整站起不来**，而错误发生在**别的仓**（框架）里，不在本仓的测试范围 |
| `ready.port_file` 与 `runtime.MCP_PORT_FILE` 一致 | 两处各写一个字面量 ⇒ **改了名字会分叉**，而现象是"永远不就绪" |
| ⭐ 两条腿的 `ML_TOOLBOX_RUNS` 相同 | 不一致 ⇒ 切手动后看到**另一份 runs**，现象是"切了但看着是空的"，**且没有任何报错** |
| ⭐ `dataset_roots` 解析保真 | 那段逻辑从旧 launcher 搬来，**行为一旦漂移**，现象是"AI 读不到数据"或"能读不该读的" |
| `manual` 与 svc 同 `data_root` | 换腿的定义是"同一份账、同一个写租约根"；不同根就不是换腿 |

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
    dsh 插件的 `portFile`（`dsh/src/ml-panel-params.ts`）也读它。
    两处各写一个字面量 ⇒ 改名时必然分叉，而现象是"永远不就绪"。
    """
    from ml_mecha.mecha_site import SITE
    from ml_mecha.runtime import MCP_PORT_FILE

    svc = [s for s in SITE.services if s.id == "ML"][0]
    assert svc.ready.port_file == MCP_PORT_FILE, (
        f"声明里是 {svc.ready.port_file!r}，runtime 里是 {MCP_PORT_FILE!r} ⇒ 分叉了")


# ============================================================ ⭐ 两条腿看同一份账

def test_both_legs_pin_the_same_runs_root():
    """⭐ svc 与 manual 的 `ML_TOOLBOX_RUNS` 必须**指向同一处**。

    ⚠ 这条是"**切了但看着是空的**"的**唯一防线**：
    `runs/` 的根由该环境变量开关（`ml_toolbox/core/runs.py`）。
    若两腿不一致（或哪条腿没继承），换腿后 GUI 会渲染**另一份** runs ——
    **没有报错、没有异常**，人只会觉得"我的运行记录没了"。
    """
    from ml_mecha.mecha_site import SITE

    svc = [s for s in SITE.services if s.id == "ML"][0]
    assert svc.manual is not None, "本项目声明了 human adapter（切手动要起它）"

    a = svc.env.get("ML_TOOLBOX_RUNS")
    b = svc.manual.env.get("ML_TOOLBOX_RUNS")
    assert a, "svc 没钉死 ML_TOOLBOX_RUNS（不许留给继承）"
    assert b, "manual 没钉死 ML_TOOLBOX_RUNS（不许留给继承）"
    assert Path(a).resolve() == Path(b).resolve(), (
        f"两条腿的 runs 根不同：{a!r} vs {b!r} ⇒ 换腿后会看到另一份 runs")


def test_both_legs_share_the_same_data_root():
    """换腿的定义：**同一份账、同一个写租约根**，只换操作者。

    ⚠ 框架也会校验这条（`site.py::Service.validate`）—— 这里再钉一次，
    因为它是**语义**要求，而不是格式要求。
    """
    from ml_mecha.mecha_site import SITE

    svc = [s for s in SITE.services if s.id == "ML"][0]
    assert svc.manual is not None
    assert Path(svc.manual.data_root) == Path(svc.data_root)
    assert svc.manual.id == f"{svc.id}:manual", (
        "约定 id 形如 <svc_id>:manual ⇒ 换腿时能机械认出这对腿")


# ============================================================ ⭐ dataset_roots 保真

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


def test_manual_ready_is_honestly_absent():
    """human adapter 的 `ready` 必须**如实为 None**（今天没有可探信号）。

    ⚠ GUI 不监听端口、不写端口文件、不握手（全仓 grep 零命中）
    ⇒ "窗口可用"这件事**今天判不了**。**不许编一个假判据**
    （例如"进程活着就算窗口好用了"）—— 见设计文档 §5.3.1。
    """
    from ml_mecha.mecha_site import SITE

    svc = [s for s in SITE.services if s.id == "ML"][0]
    assert svc.manual is not None
    assert svc.manual.ready is None, (
        "GUI 没有可探信号 ⇒ ready 必须是 None（起完即算就绪），不要编判据")
    assert svc.manual.stop_timeout >= 15.0, (
        "Qt 事件循环需要时间正常退出；强杀会让 finally 不跑（A7/A56 的同类教训）")
