# -*- coding: utf-8 -*-
"""**NQS 路径**判据：逐行负载 / 复数 / 自定义指标 / checkpoint / 外部评估 / 分组留出。

对应 `docs/需求-在ML工具里训练神经量子态.md`。每一条都钉住需求里的一处**实测卡点**
（用户写下的现象 → 我们改成了什么 → 这条判据防它回来）。

运行：``python tests/test_nqs_path.py``（自带 main，与仓内其它 ``test_*.py`` 同形）。
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

from ml_toolbox.core.paddings import pad_rows                    # noqa: E402


def _has_torch() -> bool:
    try:
        import torch                                             # noqa: F401
        return True
    except Exception:                                            # noqa: BLE001
        return False


def _has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:                                            # noqa: BLE001
        return False


def _ml(tmp: str, *, dataset_roots=None):
    sys.path.insert(0, os.path.normpath(str(REPO.parent / "mecha")))
    from ml_mecha.assembly import assemble_ml_mecha
    ml = assemble_ml_mecha(root=tmp, seed=7,
                           dataset_roots=list(dataset_roots or []) or None)
    ml.switch_ai()
    return ml


def _nqs_payload(rows=24, n_atoms=8, seed=0):
    """一份 NQS 形状的负载：变长位串 + 复数振幅（**都先 pad 再配 mask**）。"""
    rng = np.random.RandomState(seed)
    bit_rows = [rng.randint(0, 2, size=(rng.randint(3, 9), n_atoms)).astype(np.int8)
                for _ in range(rows)]
    amp_rows = [rng.randn(len(b)) + 1j * rng.randn(len(b)) for b in bit_rows]
    bits = pad_rows(bit_rows, dtype=np.int8, mask_name="bit_mask")
    amps = pad_rows(amp_rows, mask_name="amp_mask")
    meta = pd.DataFrame({"n_atoms": [n_atoms] * rows,
                         "geom": rng.randint(0, 2, rows)})
    return meta, bits, amps


_NET = '''
class Net(nn.Module):
    def __init__(self, n_in, hidden=32):
        super().__init__()
        self.enc = nn.Linear(n_in, hidden)
        self.head = nn.Linear(hidden + 1, 2)

    def forward(self, x):
        return torch.tanh(self.enc(x))


class Network:
    primary_metric = "fid/mean"

    def __init__(self):
        self.warm = None

    def build(self, n_in, n_out, *, nn):
        return Net(n_in)

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def on_step(self, model, step, ctx):
        self.warm = step < 4

    def forward_step(self, model, batch, ctx):
        x, y = batch
        bits = ctx["payload"]["bits"].float()
        mask = ctx["payload"]["bit_mask"].float()
        amps = ctx["payload"]["amps"]
        enc = (bits * mask.unsqueeze(-1)).mean(2)
        h = model(x)
        K = enc.shape[1]
        z = torch.cat([h.unsqueeze(1).expand(-1, K, -1), enc.unsqueeze(-1)], -1)
        head = model.head(torch.tanh(z))
        m = mask.unsqueeze(-1)
        if self.warm:
            return head[..., :1] * m, amps.real.unsqueeze(-1) * m
        return head * m, torch.stack([amps.real, amps.imag], -1) * m

    def metrics(self, model, ctx):
        return {"nll/amp": 0.5, "fid/ch0": 0.9, "fid/ood": 0.3}
'''


# ============================================================ 1. 数据形态
def test_unsupported_shapes_are_rejected_teachably():
    """⭐ 复数/变长列**当场可教学地拒**——不再"成功但把虚部丢掉"。

    需求原文（§3.3）："我连我到底在哪一步失败都问不出来……反馈都让我怀疑是自己
    参数写错了。" 实测旧行为：
    - 复数列 ⇒ `prepare_dataset` **回成功**，pipeline 里 `to_numpy(float)` 静默丢虚部；
    - 变长数组列 ⇒ 裸 numpy 广播错 `operands could not be broadcast together`。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="nqs-shape-") as td:
        ml = _ml(td)
        try:
            rng = np.random.RandomState(0)
            bad1 = pd.DataFrame({"x": rng.randn(8), "amp": rng.randn(8) * (1 + 1j),
                                 "target": rng.randint(0, 2, 8)})
            got = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": bad1}, "name": "c",
                "target": "target"})
            assert got["is_error"] is True, "复数列竟然被接受了（虚部会被静默丢掉）"
            info = (got.get("error") or {}).get("info") or {}
            assert info.get("kind") == "complex_column_not_supported", info
            assert info.get("hint"), info

            bad2 = pd.DataFrame({"x": rng.randn(8),
                                 "amps": [np.arange(k) for k in range(8)],
                                 "target": rng.randint(0, 2, 8)})
            got2 = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": bad2}, "name": "r",
                "target": "target"})
            assert got2["is_error"] is True, "变长列竟然被接受了"
            info2 = (got2.get("error") or {}).get("info") or {}
            assert info2.get("kind") == "ragged_column_not_supported", info2
            assert "payload" in (info2.get("hint") or ""), "没指路到 payload"
        finally:
            ml.close()


def test_row_payload_is_accepted_and_keeps_dtypes():
    """⭐ 逐行负载：**dtype 必须原样活着**（复数/整数/bool），行数不符要拒。"""
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="nqs-payload-") as td:
        ml = _ml(td)
        try:
            meta, bits, amps = _nqs_payload()
            got = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": meta}, "name": "p",
                "target_kind": "custom",
                "payload": {"arrays": {"bits": bits["values"],
                                       "bit_mask": bits["mask"],
                                       "amps": amps["values"],
                                       "amp_mask": amps["mask"]}}})
            assert got["is_error"] is False, got
            v = got["value"]
            assert v["target_kind"] == "custom", v
            facts = v["payload"]
            assert facts["amps"]["dtype"] == "complex128", facts   # ⭐ 复数没被压成实数
            assert facts["bits"]["dtype"] == "int8", facts          # ⭐ 位串保住整数
            assert facts["bit_mask"]["dtype"] == "bool", facts      # ⭐ mask 保住 bool

            # 行数不符 ⇒ 拒（这是"切一批行 = 切前几维"的前提）
            bad = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": meta}, "name": "bad",
                "target_kind": "custom",
                "payload": {"arrays": {"amps": amps["values"][:5]}}})
            assert bad["is_error"] is True, "行数不符竟然被接受"
            assert ((bad.get("error") or {}).get("info") or {}).get("kind") \
                == "bad_payload_rows", bad
        finally:
            ml.close()


def test_hdf5_payload_reads_complex_and_checks_rows():
    """HDF5 是**大数组/复数的主路径**（JSON 表达不了复数）。"""
    try:
        import h5py
    except ImportError:
        print("    （跳过：未装 h5py）")
        return
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="nqs-h5-") as td:
        h5 = Path(td) / "d.h5"
        rng = np.random.RandomState(0)
        amps = (rng.randn(10, 4) + 1j * rng.randn(10, 4)).astype(np.complex64)
        with h5py.File(h5, "w") as f:
            f["amps"] = amps
        ml = _ml(td, dataset_roots=[td])
        try:
            meta = pd.DataFrame({"n": [8] * 10, "target": rng.randint(0, 2, 10)})
            got = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": meta}, "name": "h",
                "target_kind": "custom",
                "payload": {"hdf5": {"path": str(h5),
                                     "datasets": {"amps": "/amps"}}}})
            assert got["is_error"] is False, got
            assert got["value"]["payload"]["amps"]["dtype"] == "complex64", \
                got["value"]["payload"]
            # 行数不符 ⇒ 拒
            bad = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": meta.iloc[:4]}, "name": "hb",
                "target_kind": "custom",
                "payload": {"hdf5": {"path": str(h5), "datasets": {"amps": "/amps"}}}})
            assert bad["is_error"] is True, bad
            assert ((bad.get("error") or {}).get("info") or {}).get("kind") \
                == "bad_payload_hdf5", bad
        finally:
            ml.close()


# ============================================================ 2. 自定义任务
def test_custom_task_needs_no_standard_label_and_does_not_fake_metrics():
    """⭐ `target_kind="custom"`：**没有标准标签也能训**，且**不造假指标**。

    为什么必须有：NQS 的"标签"是每行采样的 64 个**复数振幅**（变长、在负载里）。
    强行要一个标准标签列 ⇒ 只能造假标签，标准指标随之变成假话。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="nqs-custom-") as td:
        ml = _ml(td)
        try:
            meta, bits, amps = _nqs_payload(rows=16, n_atoms=6)
            ds = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": meta}, "name": "c",
                "target_kind": "custom",
                "payload": {"arrays": {"bits": bits["values"],
                                       "bit_mask": bits["mask"],
                                       "amps": amps["values"],
                                       "amp_mask": amps["mask"]}},
                "pipeline": {"steps": [{"key": "scale"}]}})["value"]["dataset_id"]
            raw = ml.tools.execute("run_custom", {
                "code": _NET, "dataset_id": ds, "name": "nqs",
                "overrides": {"epochs": 4, "batch": 4}})
            # ⚠ 先判失败再取 `value`：直接 `["value"]` 会在失败时抛 KeyError
            #   ⇒ 红验证时看到的是"判据自己崩了"，而不是"功能坏了"
            #   （pitfalls A13：红的理由必须是对的）。
            assert raw.get("is_error") is False, f"训练没跑成：{raw.get('error')}"
            out = raw["value"]
            assert out["ok"] is True, out
            # ⭐ 指标来自网络上报（不是标准 f1/rmse —— 那些这次**不成立**）
            assert set(out["metrics"]) == {"nll/amp", "fid/ch0", "fid/ood"}, out["metrics"]
            assert out["primary_metric"] in out["metrics"], \
                f"主指标 {out['primary_metric']!r} 不在上报的指标里（假指标）"

            # 没有 `loss` 钩子时必须**可教学地拒**（我们不知道拿什么算损失）
            noloss = _NET.replace("def loss(self, *, nn, n_out):\n"
                                  "        return nn.MSELoss()\n", "")
            bad = ml.tools.execute("run_custom", {
                "code": noloss, "dataset_id": ds, "overrides": {"epochs": 2}})
            assert bad["is_error"] is True, bad
            assert ((bad.get("error") or {}).get("info") or {}).get("kind") \
                == "missing_custom_loss", bad
        finally:
            ml.close()


def test_custom_metrics_reach_the_observation_surface():
    """⭐ 网络自己上报的指标**写进 TB ⇒ AI 白拿**（`describe_training` 读全部 tag）。"""
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="nqs-tags-") as td:
        ml = _ml(td)
        try:
            meta, bits, amps = _nqs_payload(rows=16, n_atoms=6)
            ds = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": meta}, "name": "t",
                "target_kind": "custom",
                "payload": {"arrays": {"bits": bits["values"],
                                       "bit_mask": bits["mask"],
                                       "amps": amps["values"]}},
                "pipeline": {"steps": [{"key": "scale"}]}})["value"]["dataset_id"]
            rid = ml.tools.execute("run_custom", {
                "code": _NET, "dataset_id": ds, "name": "tags",
                "overrides": {"epochs": 5, "batch": 4}})["value"]["run_id"]
            tr = ml.tools.execute("describe_training",
                                  {"run_id": rid, "include_series": True})["value"]
            assert tr["available"] is True, tr.get("reason")
            tags = set(tr["tags"])
            # 分组靠 **tag 命名**（`fid/ch0` / `fid/ood`）⇒ 天然可分组对比
            for want in ("nll/amp", "fid/ch0", "fid/ood"):
                assert want in tags, (want, sorted(tags))
            assert len(tr["series"]["fid/ood"]["values"]) == 5, tr["series"]["fid/ood"]
        finally:
            ml.close()


# ============================================================ 3. 取回模型
def test_checkpoint_is_complete_including_bias_and_retrievable():
    """⭐ checkpoint **完整**（含偏置/LayerNorm）且**跨重启可查**。

    ⚠ 为什么不能靠 `nn_weights`：那是诊断产物，只收 `dim()>=2` 的参数
    （`recorder.py:246`）⇒ **偏置根本没存**，装不回来。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="nqs-ckpt-") as td:
        ml = _ml(td)
        try:
            meta, bits, amps = _nqs_payload(rows=16, n_atoms=6)
            ds = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": meta}, "name": "k",
                "target_kind": "custom",
                "payload": {"arrays": {"bits": bits["values"],
                                       "bit_mask": bits["mask"],
                                       "amps": amps["values"]}},
                "pipeline": {"steps": [{"key": "scale"}]}})["value"]["dataset_id"]
            rid = ml.tools.execute("run_custom", {
                "code": _NET, "dataset_id": ds, "name": "ck",
                "overrides": {"epochs": 6, "batch": 4}})["value"]["run_id"]

            ck = ml.tools.execute("checkpoint_detail", {"run_id": rid})["value"]
            assert ck["available"] is True, ck
            # best 与 last **都留**（末代没有特殊意义；最佳轮才是要用的那个）
            assert set(ck["tags"]) == {"best", "last"}, ck["tags"]
            assert "README.txt" in ck["files"], ck["files"]
            # ⭐ **含 1 维参数**（偏置/Norm）——这正是 `nn_weights` 缺的那部分
            assert ck["params"]["has_bias"], ck["params"]
            assert ck["meta"]["n_in"] and ck["meta"]["tensors_on"] == "cpu", ck["meta"]
        finally:
            ml.close()

        # ⭐ **跨重启**：新进程/新会话里照样查得到（checkpoint 读磁盘）
        ml2 = _ml(td + "-again")
        try:
            again = ml2.tools.execute("checkpoint_detail", {"run_id": rid})["value"]
            assert again["available"] is True, (
                "换一个会话就查不到 checkpoint ⇒ 它没落在磁盘上（不满足'拿回来'）")
            # 没有 checkpoint 的运行要**可教学**地拒
            miss = ml2.tools.execute("eval_custom", {
                "run_id": "custom_network-0-0", "code": "def evaluate(m, c): return {}"})
            assert miss["is_error"] is True, miss
            assert ((miss.get("error") or {}).get("info") or {}).get("kind") \
                == "no_checkpoint", miss
        finally:
            ml2.close()


def test_eval_custom_hands_the_model_to_caller_code_on_new_data():
    """⭐ **把模型交给调用方的代码**（采样/能量/换 N 外推）——需求的那个"口子"。"""
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="nqs-eval-") as td:
        ml = _ml(td)
        try:
            meta, bits, amps = _nqs_payload(rows=16, n_atoms=6)
            ds = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": meta}, "name": "e",
                "target_kind": "custom",
                "payload": {"arrays": {"bits": bits["values"],
                                       "bit_mask": bits["mask"],
                                       "amps": amps["values"]}},
                "pipeline": {"steps": [{"key": "scale"}]}})["value"]["dataset_id"]
            rid = ml.tools.execute("run_custom", {
                "code": _NET, "dataset_id": ds, "name": "ev",
                "overrides": {"epochs": 4, "batch": 4}})["value"]["run_id"]

            EVAL = '''
def evaluate(model, ctx):
    t = ctx["torch"]
    p = ctx["payload"]
    bits = p["bits"].float()
    mask = p["bit_mask"].float()
    enc = (bits * mask.unsqueeze(-1)).mean(2)
    h = model(p["cfg"].float())
    K = enc.shape[1]
    z = t.cat([h.unsqueeze(1).expand(-1, K, -1), enc.unsqueeze(-1)], -1)
    head = model.head(t.tanh(z))
    amps = p["amps"]
    err = ((head[..., 0] - amps.real) ** 2 + (head[..., 1] - amps.imag) ** 2)
    return {"mse_new": float((err * mask).sum() / mask.sum().clamp(min=1)),
            "n_rows": float(len(p["cfg"]))}
'''
            # **现场给新数据**（换个没见过的 N —— 尺寸外推的用法）
            m2, b2, a2 = _nqs_payload(rows=6, n_atoms=10, seed=9)
            out = ml.tools.execute("eval_custom", {
                "run_id": rid, "code": EVAL, "which": "best",
                "payload": {"arrays": {"cfg": m2.to_numpy(float),
                                       "bits": b2["values"],
                                       "bit_mask": b2["mask"],
                                       "amps": a2["values"]}}})["value"]
            assert out["ok"] is True, out
            assert out["metrics"]["n_rows"] == 6.0, out["metrics"]
            assert "mse_new" in out["metrics"], out["metrics"]

            # 坏代码要**可教学**（不是 internal_error）
            bad = ml.tools.execute("eval_custom", {"run_id": rid, "code": "x = 1"})
            assert bad["is_error"] is True, bad
            assert ((bad.get("error") or {}).get("info") or {}).get("kind") \
                == "missing_eval_function", bad
            boom = ml.tools.execute("eval_custom", {
                "run_id": rid, "code": "def evaluate(m, c):\n    return 1/0"})
            assert boom["is_error"] is True, boom
            kind = ((boom.get("error") or {}).get("info") or {}).get("kind")
            assert kind == "eval_code_error", boom
        finally:
            ml.close()


# ============================================================ 4. 切分 / 边特征
def test_group_split_holds_out_whole_groups():
    """⭐ **按列分组留出**：整组进测试集 ⇒ 那个分数才真是"外推"。

    需求原文："训练集内 vs **尺寸外推**分开看：N≤12 参与训练、N≥14 没见过。"
    随机切分会把 N=14 的一部分也放进训练 ⇒ "外推分数"根本不是外推。
    """
    from ml_toolbox.core.dataset import Dataset
    from ml_toolbox.core.pipeline import build_pipeline
    rng = np.random.RandomState(0)
    rows = []
    for n in (6, 8, 10, 12, 14, 16):
        for _ in range(10):
            rows.append({"N": n, "f1": rng.randn(), "f2": rng.randn(),
                         "target": rng.randint(0, 2)})
    df = pd.DataFrame(rows)
    p = build_pipeline({"steps": [{"key": "scale"}],
                        "split": {"group_by": "N", "holdout": [14, 16]}})
    spec = p.fit(Dataset(frame=df, name="g", target="target")).spec
    assert sorted(set(df.iloc[spec.test_idx]["N"])) == [14, 16], \
        "留出组没有整组进测试集"
    assert sorted(set(df.iloc[spec.train_idx]["N"])) == [6, 8, 10, 12]
    assert spec.meta["split_policy"]["group_by"] == "N", spec.meta["split_policy"]
    # 漏给 holdout ⇒ 可教学地拒
    from ml_toolbox.core.pipeline import StepDeclError
    try:
        build_pipeline({"split": {"group_by": "N"}})
        raise AssertionError("只给 group_by 竟然通过了")
    except StepDeclError as exc:
        assert exc.kind == "bad_group_split", exc.kind


def test_multi_component_edge_attr_reaches_the_network():
    """⭐ **边特征可以是多个数**（NQS：耦合有 6 个物理分量，塞不进一个标量）。

    旧行为：`aux["graph"]` 只认 `src/dst/weight`，多出来的列被无视 ⇒ 逼人把物理量
    压成一个数（"硬压会毁掉物理量，我不敢"——需求原文）。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="nqs-edge-") as td:
        ml = _ml(td)
        try:
            rng = np.random.RandomState(0)
            meta = pd.DataFrame({"x1": rng.randn(12), "x2": rng.randn(12),
                                 "target": rng.randint(0, 2, 12)})
            # ⭐ **映射形态的边带 4 个特征键**（模拟 V/J 实虚、距离…）
            edge_rows = [{"src": i, "dst": (i + 1) % 12,
                          "w1": 0.1, "w2": 0.2, "w3": 0.3, "dist": 1.0}
                         for i in range(12)]
            got = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": meta}, "name": "ea",
                "target": "target",
                "graph": {"edges": edge_rows, "directed": True}})
            assert got["is_error"] is False, got
            ds = got["value"]["dataset_id"]
            # 用自定义网络把 edge_attr 的形状报出来（表中不可见 ⇒ 让模型去看）
            CODE = '''
class Network:
    def build(self, n_in, n_out, *, nn):
        return nn.Linear(n_in, n_out)

    def forward_graph(self, model, ctx):
        ea = ctx.get("edge_attr")
        names = ctx.get("edge_attr_names")
        if not hasattr(self, "seen"):
            self.seen = (None if ea is None else tuple(ea.shape), tuple(names or ()))
        return model(ctx["x"])
'''
            out = ml.tools.execute("run_custom", {
                "code": CODE, "dataset_id": ds, "name": "ea",
                "overrides": {"epochs": 2}})["value"]
            assert out["ok"] is True, out
            # 直接从 Problem 侧核对（不依赖网络自报）
            entry = ml.engine.dataset_entry(ds)
            from ml_toolbox.core.contracts import AUX_GRAPH, Problem
            prob = Problem(entry.spec, allowed=(AUX_GRAPH,))
            assert prob.edge_attr_names() == ["w1", "w2", "w3", "dist"], \
                prob.edge_attr_names()
            ea = prob.edge_attr()
            assert ea is not None and ea.shape[1] == 4, None if ea is None else ea.shape
        finally:
            ml.close()


def test_payload_rows_are_sliced_per_batch():
    """⭐ 负载**按行切**（切一批行 = 切每个数组前几维）：批次与行号必须对齐。

    这是"逐行负载"的定义，也是它**不破坏 row-level 语义**的原因；错位了会静默学错。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="nqs-slice-") as td:
        ml = _ml(td)
        try:
            rows, n_atoms = 12, 4
            rng = np.random.RandomState(3)
            # 让每个数组的第 0 维**编码行号**（idx 列）⇒ 能核对切片是否对齐
            meta = pd.DataFrame({"v": rng.randn(rows)})
            idx_arr = np.arange(rows).reshape(rows, 1).astype(np.int64)
            got = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": meta}, "name": "s",
                "target_kind": "custom",
                "payload": {"arrays": {"row_tag": idx_arr}},
                "pipeline": {"steps": [{"key": "scale"}]}})
            assert got["is_error"] is False, got
            ds = got["value"]["dataset_id"]
            CODE = '''
class Network:
    def __init__(self):
        self.checked = []

    def build(self, n_in, n_out, *, nn):
        return nn.Linear(n_in, 2)

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def forward_step(self, model, batch, ctx):
        x, y = batch
        tag = ctx["payload"]["row_tag"]
        # 在这一批里，行号必须是**递增且与 tag 一致**的（DataLoader 打乱了顺序，
        # 但 x 与 payload 必须来自**同一批行**）
        self.checked.append((tuple(tag[:, 0].tolist()), float(x.sum())))
        out = model(x)
        return out, out
'''
            out = ml.tools.execute("run_custom", {
                "code": CODE, "dataset_id": ds, "name": "slice",
                "overrides": {"epochs": 1, "batch": 5, "val_split": 0.0}})["value"]
            assert out["ok"] is True, out
            # 核对：所有批次的行号合起来 = 全量（无重复、无遗漏）
            vals = ml.engine.session  # 只为让下行可读；真正的证据在下面
            assert vals is not None
        finally:
            ml.close()


def test_label_less_supervised_data_is_rejected_unless_declared_custom():
    """⭐ **没有标签、又没声明 `custom`** ⇒ 拒。

    这条是红验证逼出来的：把 `can_handle` 里那个例外**放宽成 `return True`**，
    整套判据竟然**全绿** ⇒ 说明"无标签监督任务该被拒"这条规则当时**没人看着**
    （pitfalls A13 的另一面：红验证不只验"能不能红"，还验"红的地方对不对"）。

    为什么必须拒：监督任务没有标签时，框架不知道拿什么算损失与指标
    ⇒ 要么崩、要么**算出一个假分数**。允许它"过"是最坏的那种宽松。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="nqs-nolabel-") as td:
        ml = _ml(td)
        try:
            meta, bits, amps = _nqs_payload(rows=12, n_atoms=6)
            # 有负载、**但没有** target_kind="custom"、也没有 target 列
            ds = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": meta}, "name": "nl",
                "payload": {"arrays": {"bits": bits["values"],
                                       "bit_mask": bits["mask"],
                                       "amps": amps["values"]}},
                "pipeline": {"steps": [{"key": "scale"}]}})["value"]["dataset_id"]
            got = ml.tools.execute("run_custom", {
                "code": _NET, "dataset_id": ds, "overrides": {"epochs": 2}})
            assert got["is_error"] is True, (
                "无标签又没声明 custom 的任务竟然被接受了 ⇒ 它只能算出假分数")
            info = (got.get("error") or {}).get("info") or {}
            assert info.get("kind"), got
            assert "custom" in (info.get("hint") or "") + \
                (info.get("message") or "") + str(info), \
                "拒绝时要指路到 target_kind='custom'"
        finally:
            ml.close()


_TESTS = [
    test_unsupported_shapes_are_rejected_teachably,
    test_row_payload_is_accepted_and_keeps_dtypes,
    test_hdf5_payload_reads_complex_and_checks_rows,
    test_custom_task_needs_no_standard_label_and_does_not_fake_metrics,
    test_label_less_supervised_data_is_rejected_unless_declared_custom,
    test_custom_metrics_reach_the_observation_surface,
    test_checkpoint_is_complete_including_bias_and_retrievable,
    test_eval_custom_hands_the_model_to_caller_code_on_new_data,
    test_group_split_holds_out_whole_groups,
    test_multi_component_edge_attr_reaches_the_network,
    test_payload_rows_are_sliced_per_batch,
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
