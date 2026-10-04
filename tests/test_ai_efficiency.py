# -*- coding: utf-8 -*-
"""**AI 往返成本**判据：让机械重复劳动归软件（试跑 / 步级观测 / 代码从文件读 / 负载继承）。

对应 `docs/建议-AI往返成本与训练观测面.md`。那个反馈的核心不是"缺功能"，而是
**"每次失败都要一整个往返"**：他这次形状类错误犯了约 15 次、每次一个往返，
而本地跑一次 python 是秒级 ⇒ 他被迫"本地磨对形状再提交"。

本文件把"一次往返能拿到多少"钉成判据。运行：``python tests/test_ai_efficiency.py``。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: 判据一律钉 **CPU**（门禁要的是"逻辑对不对"，不是"跑得快不快"）。
#: ⚠ 别改成 auto：那会让每个测试进程都加载 CUDA，叠加正在运行的 app 会耗尽页面文件。
_DEV = "cpu"


def _has_torch() -> bool:
    try:
        import torch                                             # noqa: F401
        return True
    except Exception:                                            # noqa: BLE001
        return False


def _ml(tmp: str, *, roots=None):
    sys.path.insert(0, os.path.normpath(str(REPO.parent / "mecha")))
    from ml_mecha.assembly import assemble_ml_mecha
    ml = assemble_ml_mecha(root=tmp, seed=3, dataset_roots=list(roots or []) or None)
    ml.switch_ai()
    return ml


def _payload_ds(ml, rows=16, L=6, K=4, d=5):
    """一份 NQS 形状的数据：逐行负载 + `target_kind=custom`。"""
    rng = np.random.RandomState(0)
    meta = pd.DataFrame({"m": rng.randn(rows)})
    xr = rng.randn(rows, L, d)
    bits = rng.randint(0, 2, size=(rows, K, L)).astype(np.int8)
    mask = np.ones((rows, K), dtype=bool)
    amps = rng.randn(rows, K) + 1j * rng.randn(rows, K)
    got = ml.tools.execute("prepare_dataset", {
        "source": {"kind": "frame", "frame": meta}, "name": "eff",
        "target_kind": "custom",
        "payload": {"arrays": {"xr": xr, "bits": bits, "bit_mask": mask,
                               "amps": amps}},
        "pipeline": {"steps": [{"key": "scale"}]}})
    assert got["is_error"] is False, got
    return got["value"]["dataset_id"]


#: 正确的网络：`build` 从负载形状取维度（不猜）
GOOD = '''
class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Linear(d, 8)
        self.head = nn.Linear(8 + 1, 2)

    def forward(self, x):
        return torch.tanh(self.enc(x))


class Network:
    def build(self, n_in, n_out, *, nn, ctx=None):
        return Net(int(ctx["payload_shapes"]["xr"][-1]))

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def on_step(self, model, step, ctx):
        return {"nll/step": 1.0 / (step + 1.0)}      # 步级上报

    def forward_step(self, model, batch, ctx):
        x, y = batch
        xr = ctx["payload"]["xr"]
        bits = ctx["payload"]["bits"].float()
        mask = ctx["payload"]["bit_mask"].float()
        h = model(xr).mean(1)
        enc = (bits * mask.unsqueeze(-1)).mean(2)
        K = enc.shape[1]
        z = torch.cat([h.unsqueeze(1).expand(-1, K, -1), enc.unsqueeze(-1)], -1)
        out = model.head(torch.tanh(z))
        tgt = torch.stack([ctx["payload"]["amps"].real,
                           ctx["payload"]["amps"].imag], -1)
        return out, tgt
'''

#: **写死维度**的网络（真实负载每行 5 维，它写 7）⇒ 必须被试跑一次抓到
BAD_DIM = GOOD.replace('int(ctx["payload_shapes"]["xr"][-1])', '7')


# ============================================================ 1. 试跑
def test_dry_run_catches_a_shape_error_in_one_round_trip():
    """⭐ 一次往返拿到：结构化归因 + **失败在哪个模块、它收到什么** + 形状轨迹。

    这是本次反馈的**头号诉求**（他评"如果只改一件事就改这个"）：形状错从
    "一次一个往返"压到"一次全看到"。他的 15 次里绝大多数**一次前向就能全暴露**。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="eff-dry-") as td:
        ml = _ml(td)
        try:
            ds = _payload_ds(ml)
            got = ml.tools.execute("run_custom", {
                "code": BAD_DIM, "dataset_id": ds, "dry_run": True,
                "device": _DEV})
            assert got["is_error"] is False, (
                "试跑**不该**把失败抛成命令错误：那会把形状轨迹整条丢掉")
            v = got["value"]
            assert v["dry_run"] is True and v["ok"] is False, v
            assert v["error_kind"] == "shape_mismatch", v["error_kind"]
            assert v["failed_batch_size"], v
            # ⭐ 指认失败位置：进去了、没出来的那个模块，以及**它收到什么**
            hint = v["error_hint"]
            assert "enc" in hint and "进去了" in hint and "[1, 6, 5]" in hint, hint
            # ⭐ 形状轨迹**非空**（pre-hook 才有这个能力：forward-hook 在炸掉的
            #    模块上永远不触发 ⇒ 轨迹会是空的）
            assert v["shapes"], "形状轨迹是空的（pre-hook 没生效？）"
            assert v["shapes"][-1]["module"] == "enc", v["shapes"]
            assert v["shapes"][-1]["out"] is None, v["shapes"]
            assert v["n_params"] > 0, v["n_params"]
        finally:
            ml.close()


def test_dry_run_passes_on_a_correct_net_and_reports_the_trace():
    """⭐ 对了也要**把轨迹给出来**：AI 才能确认"我理解的形状"和实际一致。"""
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="eff-ok-") as td:
        ml = _ml(td)
        try:
            ds = _payload_ds(ml)
            v = ml.tools.execute("run_custom", {
                "code": GOOD, "dataset_id": ds, "dry_run": True,
                "device": _DEV})["value"]
            assert v["ok"] is True, v
            # ⭐ **默认那套要真的都跑到**（1、2、训练批大小）：只跑第一个的话，
            #   "批次相关的那一类错"照样会漏 —— 那条判据杀不死这个退化（实测）。
            assert {1, 2}.issubset(set(v["batch_sizes_checked"])), v["batch_sizes_checked"]
            assert len(v["batch_sizes_checked"]) >= 3, v["batch_sizes_checked"]
            mods = [e["module"] for e in v["shapes"]]
            assert "enc" in mods and "head" in mods, v["shapes"]
            enc = next(e for e in v["shapes"] if e["module"] == "enc")
            assert enc["out"], enc            # 正常跑完 ⇒ 有输出形状
            assert v["n_params"] > 0, v
            assert v["loss_first_batch"] is not None, (
                "损失没算出来 —— 那说明「形状对了但损失语义没验」，要如实告知")
        finally:
            ml.close()


def test_dry_run_probes_several_batch_sizes():
    """⭐ **试多个批大小**：反馈里那一类"本地测不出、一提交就错"只有真数据能抓。

    他的原话："我本地全绿、一提交就错：本地我用 B=3、真实负载的 batch 是 2。"
    ⇒ 试跑默认就试 `1`、`2` 与训练批大小；且**每个批大小都真的跑**。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    #: 只在 B=1 时炸的网络（`view(0, -1)` 不合法）
    ONLY_B1_FAILS = '''
class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Linear(d, 4)

    def forward(self, x):
        return torch.tanh(self.enc(x))


class Network:
    def build(self, n_in, n_out, *, nn, ctx=None):
        return Net(int(ctx["payload_shapes"]["xr"][-1]))

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def forward_step(self, model, batch, ctx):
        x, y = batch
        h = model(ctx["payload"]["xr"]).mean(1)      # (B,4)
        h = h.view(h.shape[0] - 1, -1)               # B=1 ⇒ view(0,-1) ⇒ 炸
        return h, h.detach()
'''
    with tempfile.TemporaryDirectory(prefix="eff-batch-") as td:
        ml = _ml(td)
        try:
            ds = _payload_ds(ml, rows=12)
            # ① 只试 B=2 ⇒ 看不到那个错（**这正是他踩的坑**：试的批次决定你能看到什么）
            ok2 = ml.tools.execute("run_custom", {
                "code": ONLY_B1_FAILS, "dataset_id": ds, "dry_run": True,
                "device": _DEV, "batch_probes": [2]})["value"]
            assert ok2["ok"] is True, ok2
            assert ok2["batch_sizes_checked"] == [2], ok2
            # ② 默认那套（含 1）⇒ 一次就抓到
            bad = ml.tools.execute("run_custom", {
                "code": ONLY_B1_FAILS, "dataset_id": ds, "dry_run": True,
                "device": _DEV})["value"]
            assert bad["ok"] is False, bad
            assert bad["failed_batch_size"] == 1, bad
            assert bad["batch_sizes_checked"] == [1], bad["batch_sizes_checked"]
        finally:
            ml.close()


def test_dry_run_writes_nothing_at_all():
    """⭐ 试跑的三条边界必须**真的是零**：不反传、不落盘、**不计入实验记录**。

    ⚠ 为什么要专门钉"什么都没写"：试跑会被 AI 高频调用（它的价值就在这），
    如果它偷偷留下运行记录/事件/存档，实验记录立刻被探针调用淹掉 ——
    而"看起来都是一类东西"正是反馈 §4 抱怨的那件事。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="eff-clean-") as td:
        ml = _ml(td)
        try:
            ds = _payload_ds(ml)
            runs_dir = Path(td) / "runs"
            before_runs = sorted(p.name for p in runs_dir.iterdir()) \
                if runs_dir.is_dir() else []
            before_hist = len(ml.engine.session.history())
            v = ml.tools.execute("run_custom", {
                "code": GOOD, "dataset_id": ds, "dry_run": True,
                "device": _DEV})["value"]
            assert v["ok"] is True, v
            after_runs = sorted(p.name for p in runs_dir.iterdir()) \
                if runs_dir.is_dir() else []
            assert after_runs == before_runs, \
                f"试跑往 runs/ 里写了东西：{set(after_runs) - set(before_runs)}"
            assert len(ml.engine.session.history()) == before_hist, \
                "试跑进了会话历史（那会污染实验记录与对比）"
            assert v["run_id"] == "", "试跑不该有 run_id"
        finally:
            ml.close()


# ============================================================ 2. 步级观测
def test_step_level_metrics_reach_the_observation_surface():
    """⭐ `on_step` 的返回值 ⇒ **步级**曲线（`describe_training` 看得到）。

    为什么重要（反馈原话）："我的 NLL 在前 100 步内就崩到 0……而在 TensorBoard /
    `describe_training` 上我只能看到**每轮**的值，看不出它是**什么时候**崩的。"
    NQS 这类任务 4500 步，塌陷发生在**前 2% 的步数**里。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="eff-step-") as td:
        ml = _ml(td)
        try:
            ds = _payload_ds(ml, rows=16)
            out = ml.tools.execute("run_custom", {
                "code": GOOD, "dataset_id": ds, "name": "stepobs",
                "overrides": {"epochs": 2, "batch": 4}, "device": _DEV})["value"]
            assert out["ok"] is True, out
            # ① 回执里能看到（"数字"）
            assert out["metrics"].get("nll/step") is not None, out["metrics"]
            # ② 曲线里能看到，而且是**步**轴（不是 epoch 轴）
            tr = ml.tools.execute("describe_training", {
                "run_id": out["run_id"], "include_series": True})["value"]
            assert "nll/step" in tr["tags"], tr["tags"]
            s = (tr.get("series") or {}).get("nll/step") or {}
            steps = s.get("steps") or []
            # ⭐ 要钉的性质是"**步级 ≠ 轮级**"：点数必须**多于**轮数（2 轮），
            #   而且从 0 开始、单调 —— 不然就是把它记成了每轮一个点。
            assert steps and steps[0] == 0, steps
            assert steps == sorted(steps), steps
            assert len(steps) > 2, f"步级点数没超过轮数（{len(steps)} vs 2 轮）"
        finally:
            ml.close()


# ============================================================ 3. 代码从文件读
def test_code_path_reads_the_network_from_a_file():
    """⭐ `code_path`：调参跑 N 个组合时不必把同一份 9 KB 代码发 N 遍。

    ⚠ 它**受已有数据根授权管辖**（不新增授权面）⇒ 越界必须 fail closed。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="eff-path-") as td, \
            tempfile.TemporaryDirectory(prefix="eff-out-") as outside:
        ml = _ml(td, roots=[td])
        try:
            f = Path(td) / "net.py"
            f.write_text(GOOD, encoding="utf-8")
            ds = _payload_ds(ml)
            # ① 真跑能读
            out = ml.tools.execute("run_custom", {
                "code_path": str(f), "dataset_id": ds, "name": "fromfile",
                "overrides": {"epochs": 2, "batch": 4}, "device": _DEV})["value"]
            assert out["ok"] is True, out
            assert out["code_path"] == str(f), out          # 回执带上路径（可对齐版本）
            # ② 试跑也能读（调参前的"先试一下"）
            dry = ml.tools.execute("run_custom", {
                "code_path": str(f), "dataset_id": ds, "dry_run": True,
                "device": _DEV})["value"]
            assert dry["ok"] is True and dry["code_lines"] == len(
                GOOD.splitlines()), dry
            # ③ **越界 fail closed**（不在授权数据根下）
            g = Path(outside) / "evil.py"
            g.write_text(GOOD, encoding="utf-8")
            bad = ml.tools.execute("run_custom", {
                "code_path": str(g), "dataset_id": ds, "dry_run": True,
                "device": _DEV})
            assert bad["is_error"] is True, "越界的 code_path 竟然被读了"
            kind = ((bad.get("error") or {}).get("info") or {}).get("kind")
            assert kind in ("dataset_roots_required", "outside_dataset_roots",
                            "bad_dataset_path", "dataset_path_denied"), kind
        finally:
            ml.close()


# ============================================================ 4. 负载继承
def test_eval_custom_inherits_the_training_payload():
    """⭐ `eval_custom` **默认继承训练时的负载**（反馈建议 4），并**明说数据从哪来**。

    他两次提到这条：不继承时他"只能把负载原样再传一遍"。而"继承"必须**响亮**
    （回执写 `payload_source`），否则"评估用的到底是哪份数据"就变成猜的。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="eff-inherit-") as td:
        ml = _ml(td)
        try:
            ds = _payload_ds(ml, rows=12)
            rid = ml.tools.execute("run_custom", {
                "code": GOOD, "dataset_id": ds, "name": "inherit",
                "overrides": {"epochs": 2, "batch": 4}, "device": _DEV}
            )["value"]["run_id"]
            EVAL = ('def evaluate(model, ctx):\n'
                    '    p = ctx["payload"]\n'
                    '    return {"keys": float(len(p)),\n'
                    '            "n_rows": float(len(p["xr"]))}\n')
            # ① 不给数据 ⇒ 继承
            # ⚠ 先判失败再取 `value`：直接下标会在失败时抛 KeyError，红起来的
            #   理由是"判据自己崩了"而不是"功能坏了"（pitfalls A13）。
            raw = ml.tools.execute("eval_custom", {"run_id": rid, "code": EVAL})
            assert raw.get("is_error") is False, f"评估没跑成：{raw.get('error')}"
            got = raw["value"]
            assert got["ok"] is True, got
            assert got["payload_source"] == "inherited_from_training", got
            assert got["metrics"]["keys"] == 4.0, got["metrics"]
            assert got["metrics"]["n_rows"] == 12.0, got["metrics"]
            # ② 显式给 ⇒ 用它（且回执说"given"）
            small = {"arrays": {"xr": np.zeros((3, 6, 5)),
                               "bits": np.zeros((3, 4, 6), dtype=np.int8),
                               "bit_mask": np.ones((3, 4), dtype=bool),
                               "amps": np.zeros((3, 4), dtype=complex)}}
            raw2 = ml.tools.execute("eval_custom", {
                "run_id": rid, "code": EVAL, "payload": small})
            assert raw2.get("is_error") is False, raw2.get("error")
            gv = raw2["value"]
            assert gv["payload_source"] == "given", gv
            assert gv["metrics"]["n_rows"] == 3.0, gv["metrics"]
        finally:
            ml.close()


def test_code_path_is_exclusive_and_says_which_source_was_used():
    """⭐ `code` 与 `code_path` **二选一**；回执必须说清**实际用了哪一份**。

    实测缺口（反馈 §2.2）：同时给时 `code` **静默覆盖** `code_path`，而回执还把
    `code_path` 原样回显 ⇒ "看起来像读到了文件"，实际跑的是内联那段
    （他给了 3 行占位符 + 156 行真文件，回执写着 `code_path` 与 `code_lines: 3`）。
    "看起来对、其实用了另一份"正是本仓最忌的假象。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="eff-src-") as td:
        ml = _ml(td, roots=[td])
        try:
            f = Path(td) / "net.py"
            f.write_text(GOOD, encoding="utf-8")
            ds = _payload_ds(ml, rows=12)
            # ① 只给 code_path ⇒ 可以（`code` 有默认 ""）
            v = ml.tools.execute("run_custom", {
                "code_path": str(f), "dataset_id": ds, "dry_run": True,
                "device": _DEV})["value"]
            assert v["ok"] is True, v
            assert v["code_source"] == "file", v
            assert v["code_lines"] == len(GOOD.splitlines()), v
            # ② 同时给 ⇒ **明确报错**（不静默取一个）
            both = ml.tools.execute("run_custom", {
                "code": "class Network:\n    pass\n", "code_path": str(f),
                "dataset_id": ds, "dry_run": True, "device": _DEV})
            assert both["is_error"] is True, both
            info = (both.get("error") or {}).get("info") or {}
            assert info.get("kind") == "bad_code_source", info
            assert "只给" in (info.get("hint") or ""), info
            # ③ 只给内联 ⇒ `code_source` 说 "inline"
            iv = ml.tools.execute("run_custom", {
                "code": GOOD, "dataset_id": ds, "dry_run": True,
                "device": _DEV})["value"]
            assert iv["code_source"] == "inline", iv
        finally:
            ml.close()


def test_dry_run_shapes_verbosity_is_smaller():
    """⭐ `shapes_verbosity="summary"` 让"几十模块 × 3 档"的回执小下来（反馈 §2.3）。

    同时钉住"顶层 `shapes` 与 `runs[].shapes` **不再各来一份**"（同一批数据重复
    两遍纯属浪费上下文）。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="eff-verb-") as td:
        ml = _ml(td)
        try:
            ds = _payload_ds(ml, rows=12)
            full = ml.tools.execute("run_custom", {
                "code": GOOD, "dataset_id": ds, "dry_run": True,
                "device": _DEV})["value"]
            small = ml.tools.execute("run_custom", {
                "code": GOOD, "dataset_id": ds, "dry_run": True,
                "device": _DEV, "shapes_verbosity": "summary"})["value"]
            assert full["ok"] is True and small["ok"] is True
            assert all("shapes" not in r for r in small["runs"]), small["runs"]
            assert small["shapes_summary"], small
            assert small["shapes"] == small["shapes_summary"], "摘要档的顶层也该是摘要"
            n_full = sum(len(r.get("shapes") or []) for r in full["runs"])
            assert len(small["shapes_summary"]) <= n_full, (small, n_full)
            # 顶层**不再**等于"全量逐档"（否则就是与 runs[] 重复）
            assert len(full["shapes"]) <= n_full, full["shapes"]
        finally:
            ml.close()


def test_eval_custom_says_why_the_payload_source_is_what_it_is():
    """⭐ 继承不了时要说清**为什么**（反馈 §3.1：他只能靠猜）。

    他实测那次：训练**确实带了 payload**，回执却是"没有负载"，原因只能猜 ——
    实测查明是**那个 checkpoint 早于"声明随存档走"这次改动**（键集里根本没有
    `payload_decl`）。⇒ 补 `payload_source_reason`，把判别结果直接给出来。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    import torch
    with tempfile.TemporaryDirectory(prefix="eff-reason-") as td:
        ml = _ml(td)
        try:
            ds = _payload_ds(ml, rows=12)
            rid = ml.tools.execute("run_custom", {
                "code": GOOD, "dataset_id": ds, "name": "reason",
                "overrides": {"epochs": 2, "batch": 4}, "device": _DEV}
            )["value"]["run_id"]
            EVAL = ('def evaluate(model, ctx):\n'
                    '    return {"keys": float(len(ctx["payload"]))}\n')
            # ① 正常 ⇒ inherited
            a = ml.tools.execute("eval_custom", {"run_id": rid, "code": EVAL})["value"]
            assert a["payload_source"] == "inherited_from_training", a
            assert a["payload_source_reason"] == "inherited", a
            # ② 抹掉存档里的声明（= "这次改动之前"的旧存档）⇒ 要**说清原因**
            #   ⚠ 路径要用 `RUNS_DIR`（存档根由装配时定，不等于测试的临时根）
            from ml_toolbox.core.persistence import RUNS_DIR
            ck = Path(RUNS_DIR) / rid / "checkpoint" / "best.pt"
            assert ck.is_file(), ck
            obj = torch.load(ck, map_location="cpu", weights_only=False)
            obj.pop("payload_decl", None)
            torch.save(obj, ck)
            b = ml.tools.execute("eval_custom", {"run_id": rid, "code": EVAL})["value"]
            assert b["payload_source"] == "", b
            assert b["payload_source_reason"] == "no_declaration", b
            # ③ 显式给 ⇒ explicit
            c = ml.tools.execute("eval_custom", {
                "run_id": rid, "code": EVAL,
                "payload": {"arrays": {"x": np.zeros((3, 2))}}})["value"]
            assert c["payload_source"] == "given", c
            assert c["payload_source_reason"] == "explicit", c
        finally:
            ml.close()


_TESTS = [
    test_dry_run_catches_a_shape_error_in_one_round_trip,
    test_dry_run_passes_on_a_correct_net_and_reports_the_trace,
    test_dry_run_probes_several_batch_sizes,
    test_dry_run_writes_nothing_at_all,
    test_step_level_metrics_reach_the_observation_surface,
    test_code_path_reads_the_network_from_a_file,
    test_eval_custom_inherits_the_training_payload,
    # ---- 第二轮反馈（docs/反馈-code_path接线与dry_run实测.md）----
    test_code_path_is_exclusive_and_says_which_source_was_used,
    test_dry_run_shapes_verbosity_is_smaller,
    test_eval_custom_says_why_the_payload_source_is_what_it_is,
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
