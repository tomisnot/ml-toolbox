# -*- coding: utf-8 -*-
"""TensorBoard 集成的判据：**写出、读回、错误边界**三件。

## 判什么、不判什么（口径先说清）

- ✔ **判**：我们的**写入**（事件文件真的产生了、含哪些轴）与**读回**（数值与 `.npz` 一致）；
- ✔ **判**：错误边界（没有 run、run_id 非法、logdir 不存在 ⇒ **可读原因**，不假装空/成功）；
- ✖ **不判**："TensorBoard 界面显示得对" —— 那是 TensorBoard 的责任，不是我们的。

⚠ **为什么必须判"读回与 `.npz` 一致"**：TensorBoard 是**外部依赖**，事件格式可能随版本变。
若它变了而我们只判"文件存在"，就会**静默读到空**——正是本仓反复在防的那类假绿。
⇒ 这条判据是"格式漂移能被发现"的那道闸。

运行：``python tests/test_tensorboard.py``（自带 main，与仓内其它 ``test_*.py`` 同形）。
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_mecha import tb_series                                   # noqa: E402
from ml_mecha.dashboard import TensorBoardSidecar                # noqa: E402
from ml_toolbox.core.contracts import DataSpec, RunConfig        # noqa: E402
from ml_toolbox.core.persistence import RUNS_DIR                 # noqa: E402
from ml_toolbox.core.runner import run_one                       # noqa: E402
from ml_toolbox.methods.neural.mlp import TorchMLP               # noqa: E402

PROBE_ID = "tbcrit-0000001-42"


def _spec(n: int = 120):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series((X["f0"] + rng.normal(scale=0.2, size=n) > 0).astype(int))
    return DataSpec(X=X, y=y, target_kind="classification", meta={"dataset": "tbcrit"})


def _clean():
    shutil.rmtree(Path(RUNS_DIR) / PROBE_ID, ignore_errors=True)


def _run_once(epochs: int = 6):
    """跑一次神经网络训练，返回 RunRecord（顺带产出事件文件）。"""
    _clean()
    cfg = RunConfig(overrides={"epochs": epochs, "hidden": "16,8", "latent_every": 3})
    cfg.extras["run_id"] = PROBE_ID
    rec = run_one(TorchMLP(), _spec(), cfg)
    assert rec.result.ok, f"训练本身失败了：{rec.result.error}"
    return rec


# ---------------------------------------------------------------- 写入
def test_events_are_written_with_all_axes():
    """事件文件真被写出，且四个轴都在（scalars / histograms / text / images）。"""
    _run_once()
    try:
        payload = tb_series.read_run(PROBE_ID)
        assert payload["available"], payload.get("reason")

        scal = payload["scalars"]
        # 五条核心标量：与 LightlyTrain 点名的"最有用信号"对齐（grad_norm 必在其中）
        for tag in ("loss", "val_loss", "lr", "grad_norm", "dead_relu"):
            assert tag in scal, f"缺核心标量 {tag}（实得 {sorted(scal)}）"
        # 每层梯度范数（逐层 tag）
        assert any(t.startswith("grad/") for t in scal), "缺逐层梯度标量"
        # 另三轴
        assert payload["histograms"], "缺权重直方图"
        assert payload["text"], "缺逐层健康表（TEXT）"
        assert payload["images"], "缺 latent 帧（IMAGES）"
    finally:
        _clean()


def test_series_matches_artifacts():
    """⭐ **读回与 `.npz` 一致** —— 这是"格式漂移能被发现"的那道闸。"""
    rec = _run_once()
    try:
        hist = rec.result.artifacts["nn_history"]
        payload = tb_series.read_run(PROBE_ID)
        assert payload["available"], payload.get("reason")

        got = payload["scalars"]["loss"]
        want = [float(v) for v in np.asarray(hist["loss"], float)]
        assert got["steps"] == list(range(len(want))), \
            f"步号对不上：{got['steps']} vs {list(range(len(want)))}"
        assert len(got["values"]) == len(want), "点数对不上"
        for a, b in zip(got["values"], want):
            assert abs(a - b) < 1e-6, f"数值漂了：{a} vs {b}"
        # grad_norm 同样核（它是"最有用信号"，更该被钉住）
        gn = payload["scalars"]["grad_norm"]["values"]
        want_gn = [float(v) for v in np.asarray(hist["grad_norm"], float)]
        for a, b in zip(gn, want_gn):
            assert abs(a - b) < 1e-6, f"grad_norm 漂了：{a} vs {b}"
    finally:
        _clean()


def test_latest_reads_the_newest_step():
    """``latest`` 给的是**最新一轮**（AI 研判要看"现在怎么样"，不是历史）。"""
    _run_once(epochs=6)
    try:
        got = tb_series.latest_of(tb_series.read_run(PROBE_ID))
        assert got["available"], got.get("reason")
        assert got["step"] == 5, f"最新步应为 5（6 个 epoch，0 起），实得 {got.get('step')}"
        for tag in ("loss", "grad_norm", "lr", "dead_relu"):
            assert isinstance(got.get(tag), float), f"latest 缺 {tag}"
        assert "grad_by_layer" in got, "缺逐层梯度（按层读梯度是否衰减要看它）"
    finally:
        _clean()


# ---------------------------------------------------------------- 错误边界
def test_missing_run_reports_reason_not_empty():
    """⚠ 读不到就**说读不到**（可读原因），**绝不回空数组假装**"跑到第 0 个 epoch"。"""
    got = tb_series.latest_of(tb_series.read_run("no-such-run-xyz"))
    assert got["available"] is False
    assert got.get("reason"), "缺原因（空原因 = 不可诊断）"
    assert "tb" in got["reason"] or "不存在" in got["reason"]


def test_run_id_path_traversal_is_rejected():
    """``run_id`` 会拼进文件路径 ⇒ 必须拒绝穿越形态（而不是"猜一个合法路径"）。"""
    for bad in ("../../etc/passwd", "..", "/abs", ".hidden", "a/b"):
        payload = tb_series.read_run(bad)
        assert payload["available"] is False, f"{bad!r} 不该被接受"
        assert payload.get("reason"), f"{bad!r} 缺原因"


def test_sidecar_refuses_missing_logdir_with_reason():
    """辅进程**起不来就说清原因**，不假装起了（与 `/runs` 的"不吞异常"同一条纪律）。"""
    ghost = Path(RUNS_DIR) / "no-such-dir-for-tb"
    got = TensorBoardSidecar(ghost).start()
    assert got["ready"] is False
    assert "不存在" in got["reason"], got


def test_sidecar_stop_is_idempotent():
    """重复 ``stop()`` 不炸（端点会重复调它；生命周期要幂等）。"""
    sc = TensorBoardSidecar(Path(RUNS_DIR))
    sc.stop(); sc.stop()                      # 从没起过也不该炸
    assert sc.running is False
    assert sc.url is None


def test_sidecar_adopts_a_live_published_port(tmp_path):
    """⭐ **跨进程幂等**：端口文件里那个端口还活着 ⇒ **复用**，不再起第二个。

    为什么必须有这条（2026-10-10 实测）：TensorBoard 是**独立进程**，而"要不要起它"有两个
    入口（监视端点 / 本项目自己的面板）。只看 ``self._proc`` 的话，第二个入口会**再起一个**
    并**覆盖**端口文件（旧那个变孤儿、白占端口）。
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        pf = tmp_path / ".tb-port"
        pf.write_text(str(port), encoding="utf-8")
        sc = TensorBoardSidecar(tmp_path, port_file=pf, log=lambda _m: None)
        got = sc.start()
        assert got["ready"] is True and got.get("adopted") is True, got
        assert got["port"] == port, got
        assert sc.running is True
        #: ⛔ 红证：端口文件里换成**没人听**的端口 ⇒ **不许**认领（该照旧去起/去失败）
        dead = tmp_path / ".tb-dead"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            dead_port = probe.getsockname()[1]
        dead.write_text(str(dead_port), encoding="utf-8")
        sc2 = TensorBoardSidecar(tmp_path, port_file=dead, log=lambda _m: None)
        assert sc2._live_published_port() is None, "把没人听的端口也认领了 ⇒ 会指向一个死地址"
        sc.stop()


def test_readonly_panel_sidecar_brings_the_window_up(tmp_path):
    """⭐⭐ **只读监视窗（`PanelWeb`）能作为一个独立进程被启动方起来**（用户实测：
    "ML 只读窗口还是没成功起来"）。

    现场：那只眼（人看方法结果：混淆矩阵/ROC/残差/曲线…）原先**只有测试会起** ⇒ 从 mecha
    起的整站里根本没有它（`.mlweb-port` 不写、浏览器里没这一页）。`main()` 的 docstring
    本来就写着"独立进程入口……**给启动方起窗口用**"，缺的就是那个启动方。
    ⇒ 本判据真起一个子进程，钉住"启动方能起它 + 它真的在听 + 端口文件公布"。
    """
    from ml_mecha.panel_web import PanelWebSidecar

    pf = tmp_path / ".mlweb-port"
    sc = PanelWebSidecar(port_file=pf, log=lambda _m: None)
    got = sc.start()
    try:
        assert got.get("ready") is True, f"面板起不来：{got}"
        assert pf.is_file(), "起来了却没公布端口（人/别的入口找不到它）"
        assert int(pf.read_text(encoding="utf-8").strip()) == got["port"]
        import urllib.request
        with urllib.request.urlopen(got["url"] + "health", timeout=20) as r:
            assert r.status == 200
    finally:
        sc.stop()
    assert not pf.exists(), "收尾没清自己写的端口文件（或误删了别人的）"
    assert sc.running is False


def test_readonly_panel_sidecar_is_honest_when_it_fails(tmp_path, monkeypatch):
    """⛔ 红证：面板**起不来**时如实回 `ready=False` + **原因**，且**不抛**。

    三条纪律（与 `TensorBoardSidecar` 同款）：best-effort（不许拖垮 AI 那条腿）、
    响亮（把 reason 说出来）、不假装（回执里 ready 必须是假）。
    """
    import subprocess

    from ml_mecha.panel_web import PanelWebSidecar

    def _boom(*_a, **_kw):
        raise OSError("判据故意让面板起不来")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    sc = PanelWebSidecar(port_file=tmp_path / ".mlweb-port", log=lambda _m: None)
    got = sc.start()                                   # ⭐ 不许抛
    assert got["ready"] is False, got
    assert "判据故意" in str(got.get("reason")), got
    assert sc.running is False


def test_readonly_panel_sidecar_adopts_a_live_published_port(tmp_path):
    """⭐ **跨进程幂等**：端口文件里那个端口还活着 ⇒ **复用**，不再起第二个面板。

    ⚠ 为什么（与 TB 侧车同一条理由）：面板有两个入口（本项目启动方 / 人自己敲 CLI），
    只看 `self._proc` 会**再起一个**并**覆盖**端口文件（前一个变孤儿、白占端口）。
    """
    import socket

    from ml_mecha.panel_web import PanelWebSidecar

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        pf = tmp_path / ".mlweb-port"
        pf.write_text(str(port), encoding="utf-8")
        sc = PanelWebSidecar(port_file=pf, log=lambda _m: None)
        got = sc.start()
        assert got.get("adopted") is True and got["port"] == port, got
        #: ⛔ 红证：换成**没人听**的端口 ⇒ 不许认领（否则浏览器会被开到死地址）
        dead = tmp_path / ".mlweb-dead"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            dead_port = probe.getsockname()[1]
        dead.write_text(str(dead_port), encoding="utf-8")
        sc2 = PanelWebSidecar(port_file=dead, log=lambda _m: None)
        assert sc2._live_published_port() is None, "把没人听的端口也认领了"
        sc.stop()


def test_duplicate_steps_are_deduped_and_flagged():
    """⚠ 同一 ``tb/`` 目录里堆了多个事件文件时：**按步去重**（后写的赢）且**留痕**。

    为什么必须有这条：训练重跑没清目录、或 SummaryWriter 因重启新建文件 ⇒ accumulator 会把
    两份都读出来，TensorBoard 自己也会画成**两条重叠曲线**。
    静默去重会把"目录脏了"藏起来；静默不去重又会让读数看起来像"训练了两遍"。
    ⇒ 去重 + 把 ``duplicate_steps`` 露出来。
    """
    rec = _run_once()
    try:
        d = Path(RUNS_DIR) / PROBE_ID / "tb"
        first = sorted(d.glob("events*"))[0]
        # 再开一个 writer 写**同样的步**（模拟"目录里多了一份事件文件"）
        from torch.utils.tensorboard import SummaryWriter
        w2 = SummaryWriter(log_dir=str(d))
        for ep in range(len(rec.result.artifacts["nn_history"]["loss"])):
            w2.add_scalar("loss", 0.5, ep)          # 与第一批同 step、不同值
        w2.flush(); w2.close()
        assert first.is_file()

        payload = tb_series.read_run(PROBE_ID)
        assert payload["available"], payload.get("reason")
        steps = payload["scalars"]["loss"]["steps"]
        assert steps == sorted(set(steps)), f"步号应去重后仍递增，实得 {steps}"
        assert payload.get("duplicate_steps"), "重复步没有被留痕（静默去重）"
        assert "loss" in payload["duplicate_steps"]
        # 后写的赢：那些被重复的步应取 0.5
        assert payload["scalars"]["loss"]["values"][-1] == 0.5
    finally:
        _clean()


def test_new_event_file_is_picked_up():
    """⭐ **目录里新增事件文件后能读到**（模拟"重启续写同一个 run"）。

    为什么必须有：``EventAccumulator`` **只盯它创建时那个文件**，后来出现的文件它不会读
    （实测）。训练若分两次进程写同一个 run，复用缓存就会**看不见后一半**——
    面板/人会以为训练停在第一步。⇒ 缓存必须按「事件文件集」失效。
    """
    rec = _run_once(epochs=6)
    try:
        d = Path(RUNS_DIR) / PROBE_ID / "tb"
        n0 = len(tb_series.read_run(PROBE_ID)["scalars"]["loss"]["values"])
        # 第二个 writer：**新增一个文件**并往后多写几步（模拟续写）
        from torch.utils.tensorboard import SummaryWriter
        w2 = SummaryWriter(log_dir=str(d))
        for ep in range(6, 9):
            w2.add_scalar("loss", 0.9, ep)
        w2.flush(); w2.close()

        got = tb_series.read_run(PROBE_ID)["scalars"]["loss"]
        assert got["steps"][-1] == 8, f"没读到新增文件里的步（实得到 {got['steps'][-1]}）"
        assert len(got["values"]) > n0, "新增文件的点没进来（缓存没按文件集失效）"
    finally:
        _clean()


def test_scalar_layout_is_written_and_readable():
    """⭐ 标量**分组布局**真的写进去了、且 TB 读得到（不是"写了但被丢掉"）。

    为什么必须有这条：`torch` 的 `add_custom_scalars` 把 layout 塞进
    **`tensor.string_val`**，而 TB 2.19 读的是 **`plugin_data.content`**
    ⇒ 用它写出来的布局**会被静默丢掉**（实测：写进去了、读回 0 个类别，
    TensorBoard 里看到的仍是一堆按字典序铺的图）。
    ⇒ 本条断言"读回的类别与图名齐全"，正是那个字段错位会打破的性质。
    """
    _run_once()
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator)
        from tensorboard.plugins.custom_scalar import layout_pb2
        ea = EventAccumulator(str(Path(RUNS_DIR) / PROBE_ID / "tb"))
        ea.Reload()
        blobs = ea.PluginTagToContent("custom_scalars")
        assert blobs, "没有 custom_scalars 布局（没写、或被写到读不到的地方）"
        content = next(iter(blobs.values()))
        assert content, "布局 content 为空（写错字段了：应在 plugin_data.content）"
        lay = layout_pb2.Layout()
        lay.ParseFromString(content)
        titles = [c.title for c in lay.category]
        assert len(lay.category) == 3, f"应有三个类别，实得 {titles}"
        assert any("训练动态" in t for t in titles), titles
        assert any("逐层梯度" in t for t in titles), titles
        assert any("权重范数" in t for t in titles), titles
        # 第一组必须把 loss 与 val_loss 画在**同一张图**上（否则"同图比较"这条性质没了）
        dyn = next(c for c in lay.category if "训练动态" in c.title)
        loss_chart = next(ch for ch in dyn.chart if ch.title == "损失")
        assert set(loss_chart.multiline.tag) == {"loss", "val_loss"}
        # 逐层梯度：每层一张图（层数 = 3）
        grads = next(c for c in lay.category if "逐层梯度" in c.title)
        assert len(grads.chart) == 3, [ch.title for ch in grads.chart]
    finally:
        _clean()


# ---------------------------------------------------------------- 直跑入口
#: ⭐ **内省收集**（2026-10-03 改）：此前这里是**手写**的判据清单，而门禁用
#: **脚本**方式跑本文件 ⇒ 新判据忘了加进来就**永远不会进闸**。实锤：
#: `test_ml_delivery.py::test_tool_params_stay_within_command_declaration`
#: （"工具声明与实现漂移"那条）从未在门禁里跑过。
#: ⇒ 定义即入闸，漂移不可能。（前提：本文件的 `test_*` 都是无参函数。）
_TESTS = [fn for name, fn in sorted(globals().items())
          if name.startswith("test_") and callable(fn)]


def main() -> int:
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
