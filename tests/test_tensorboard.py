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


# ---------------------------------------------------------------- 直跑入口
_TESTS = [
    test_events_are_written_with_all_axes,
    test_series_matches_artifacts,
    test_latest_reads_the_newest_step,
    test_missing_run_reports_reason_not_empty,
    test_run_id_path_traversal_is_rejected,
    test_sidecar_refuses_missing_logdir_with_reason,
    test_sidecar_stop_is_idempotent,
]


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
