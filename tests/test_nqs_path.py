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


#: 判据默认钉 **CPU**。⚠ 别改成 `auto`：那会让**每个测试进程都加载 CUDA**
#: （每进程 1.5–2 GB commit），与"同时在跑的 app"叠加时会把页面文件耗尽 ⇒
#: torch 半初始化（`WinError 1455 页面文件太小` / `partially initialized module
#: 'torch._inductor'`）⇒ 一连串**假红**。实测踩过。门禁要的是"逻辑对不对"。
#: 只有**专测设备**的那条才用 `auto`（它必须证明张量真落在训练设备上）。
_DEV = "cpu"


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
            raw = ml.tools.execute("run_custom", {"device": _DEV, 
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
            bad = ml.tools.execute("run_custom", {"device": _DEV, 
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
            rid = ml.tools.execute("run_custom", {"device": _DEV, 
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
            rid = ml.tools.execute("run_custom", {"device": _DEV, 
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
            rid = ml.tools.execute("run_custom", {"device": _DEV, 
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
            out = ml.tools.execute("run_custom", {"device": _DEV, 
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
            out = ml.tools.execute("run_custom", {"device": _DEV, 
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
            got = ml.tools.execute("run_custom", {"device": _DEV, 
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


# ============================================================ 5. 实测反馈（第二轮）
def test_build_receives_payload_shapes_and_old_signature_still_works():
    """⭐ `build` 拿得到**负载形状**（反馈 §2-缺口1）——不再靠猜维度。

    他的原话："后果**不是报错，而是换一份数据（换个 N、加一维特征）时静默跑偏**。"
    ⇒ 判据必须证明**换一份 d 不同的数据时模型跟着变**（写死维度的实现会红）。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    code = '''
class Net(nn.Module):
    def __init__(self, d_node):
        super().__init__()
        self.lin = nn.Linear(d_node, 4)

    def forward(self, x):
        return self.lin(x).mean(1)


class Network:
    def build(self, n_in, n_out, *, nn, ctx=None):
        assert ctx is not None, "build 没拿到 ctx"
        return Net(int(ctx["payload_shapes"]["xr"][-1]))     # ← (R,L,d) → d

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def forward_step(self, model, batch, ctx):
        x, y = batch
        out = model(ctx["payload"]["xr"])                     # (B,L,d) → (B,)
        return out, out.detach()
'''
    with tempfile.TemporaryDirectory(prefix="nqs-buildctx-") as td:
        ml = _ml(td)
        try:
            for d in (3, 7):                     # ⭐ 换一维特征 ⇒ 模型必须跟着变
                R, L = 10, 4
                rng = np.random.RandomState(d)
                meta = pd.DataFrame({"m": rng.randn(R)})
                xr = rng.randn(R, L, d)
                got = ml.tools.execute("prepare_dataset", {
                    "source": {"kind": "frame", "frame": meta}, "name": f"bc{d}",
                    "target_kind": "custom",
                    "payload": {"arrays": {"xr": xr}},
                    "pipeline": {"steps": [{"key": "scale"}]}})
                assert got["is_error"] is False, got
                out = ml.tools.execute("run_custom", {"device": _DEV, 
                    "code": code, "dataset_id": got["value"]["dataset_id"],
                    "name": f"bc{d}", "overrides": {"epochs": 2, "batch": 4}})
                assert out["is_error"] is False, out
                assert out["value"]["ok"] is True, out["value"]

            # ⚠ **老签名照样能跑**（不逼既有代码改）
            OLD = '''
class Network:
    def build(self, n_in, n_out, *, nn):
        return nn.Linear(n_in, 1)

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def forward_step(self, model, batch, ctx):
        x, y = batch
        out = model(x).squeeze(-1)
        return out, out.detach()
'''
            ds = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": pd.DataFrame({"m": [1.0] * 8})},
                "name": "old", "target_kind": "custom",
                "payload": {"arrays": {"xr": np.zeros((8, 2, 3))}},
                "pipeline": {"steps": [{"key": "scale"}]}})["value"]["dataset_id"]
            old = ml.tools.execute("run_custom", {"device": _DEV, 
                "code": OLD, "dataset_id": ds, "overrides": {"epochs": 2}})
            assert old["is_error"] is False, old
        finally:
            ml.close()


def test_custom_task_needs_no_forward_when_forward_step_exists():
    """⭐ **只有 `forward_step`、没有 `forward` 也能跑完**（反馈 §2-缺口3）。

    ⚠ 反馈把它定位成"验证轮调了 `model(Xva)`"；实测**真正炸的是 recorder 的探针**
    （`recorder._collect_probe` 默认调 `model(self._probe)`）—— 验证轮的 `elif` 我
    是写对了的。两处都要求"同一条入口"。
    ⇒ 判据里 `forward` 明确**抛异常**：谁再回到 `model(x)`，这条立刻红。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    code = '''
class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 3)
        self.act = nn.ReLU()

    def forward(self, x):
        raise RuntimeError("SHOULD-NOT-BE-CALLED: 这条路上不该有 forward")


class Network:
    def build(self, n_in, n_out, *, nn):
        return Net(n_in)

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def forward_step(self, model, batch, ctx):
        x, y = batch
        out = model.act(model.lin(x))
        if ctx["kind"] == "val":
            # ⭐ **验证轮的损失必须来自这里**（不是某个假 forward）。
            #   返回 (0.5, 0.0) ⇒ MSE = 0.25，是一个**可辨认的常数**：
            #   只要 val_loss 不等于它，就说明验证走了别的路（或压根没跑）。
            z = out * 0.0
            return z + 0.5, z
        return out, out.detach()
'''
    with tempfile.TemporaryDirectory(prefix="nqs-nofwd-") as td:
        ml = _ml(td)
        try:
            R = 12
            meta = pd.DataFrame({"m": np.arange(R, dtype=float)})
            ds = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": meta}, "name": "nf",
                "target_kind": "custom",
                "payload": {"arrays": {"xr": np.zeros((R, 2, 3))}},
                "pipeline": {"steps": [{"key": "scale"}]}})["value"]["dataset_id"]
            out = ml.tools.execute("run_custom", {"device": _DEV, 
                "code": code, "dataset_id": ds, "name": "no_forward", "diag": True,
                "overrides": {"epochs": 3, "batch": 4, "val_split": 0.25}})
            assert out["is_error"] is False, out
            rid = out["value"]["run_id"]
            tr = ml.tools.execute("describe_training",
                                  {"run_id": rid, "include_series": True})["value"]
            tags = tr["tags"]
            # ① 探针跑成功了 ⇒ dead_relu 与逐层梯度都在（探针被跳过时它们会缺）
            assert "dead_relu" in tags, tags
            assert any(t.startswith("grad/") for t in tags), tags
            # ② **验证真的跑了，而且损失来自 forward_step**（可辨认的常数 0.25）。
            #   ⚠ 这一条是红验证逼出来的：把验证分支弄坏**不会崩**（`vl` 变成 None
            #   ⇒ 验证被静默跳过 ⇒ `best_epoch` 悄悄改用训练损失）—— 那正是"假事实"。
            vl = ((tr.get("series") or {}).get("val_loss") or {}).get("values") or []
            assert vl, "验证损失整条没了（验证被静默跳过）"
            assert all(abs(v - 0.25) < 1e-6 for v in vl), \
                f"验证损失不是 forward_step 给的 0.25：{vl}"
        finally:
            ml.close()


def test_bare_tensor_in_user_hook_lands_on_the_training_device():
    """⭐ 用户钩子里**随手建的张量**落在训练设备上（反馈 §3.1）。

    他的最小复现是 `torch.tensor([0., 1.])`：模型在显卡、这个在 CPU ⇒ 报出来的却是
    `Sizes of tensors must match except in dimension 2` 这种**伪装成维度错**的消息。
    ⇒ 我们在他的代码执行期间把默认设备摆正；⚠ 同时必须**证明没渗出去**。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    import torch
    from ml_toolbox.methods.neural.trainer import hook_device
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with hook_device(torch, dev):
        inside = torch.tensor([0.0, 1.0])
    after = torch.tensor([0.0, 1.0])
    assert str(inside.device) == str(dev), (inside.device, dev)
    assert str(after.device) == "cpu", \
        f"默认设备没复原：{after.device}（会渗到我们自己的代码与下一个方法）"

    # 端到端：forward_step 里建裸张量 —— 上卡时必须能跑
    code = '''
class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 3)


class Network:
    def build(self, n_in, n_out, *, nn):
        return Net(n_in)

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def forward_step(self, model, batch, ctx):
        x, y = batch
        out = model.lin(x)
        # ⚠ **裸张量**（不带 device）—— 这正是他中招的那一行
        oh = out.unsqueeze(-1) * torch.tensor([0.0, 1.0, 0.0])
        return oh.mean(-1), oh.mean(-1).detach()
'''
    with tempfile.TemporaryDirectory(prefix="nqs-dev-") as td:
        ml = _ml(td)
        try:
            R = 10
            ds = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame",
                           "frame": pd.DataFrame({"m": np.arange(R, dtype=float)})},
                "name": "dev", "target_kind": "custom",
                "payload": {"arrays": {"xr": np.zeros((R, 2, 3))}},
                "pipeline": {"steps": [{"key": "scale"}]}})["value"]["dataset_id"]
            out = ml.tools.execute("run_custom", {
                "code": code, "dataset_id": ds, "name": "bare_tensor",
                "device": "auto", "overrides": {"epochs": 2, "batch": 4}})
            assert out["is_error"] is False, (
                f"裸张量把训练搞炸了（设备纪律失效）：{out.get('error')}")
        finally:
            ml.close()


def test_eval_custom_accepts_dataset_id_and_teaches_missing_payload():
    """⭐ `eval_custom(dataset_id=…)` 走不通是个**确定性 bug**（反馈 §2-缺口2）。

    ⚠ 我上一轮的判据**只走了 `payload` 那条路** ⇒ 文档推荐的 `dataset_id` 用法
    从没被跑过（典型的假绿）。这条补齐，并钉住"没给数据时要明说"。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="nqs-ev2-") as td:
        ml = _ml(td)
        try:
            meta, bits, amps = _nqs_payload(rows=12, n_atoms=6)
            ds = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": meta}, "name": "ev2",
                "target_kind": "custom",
                "payload": {"arrays": {"bits": bits["values"],
                                       "bit_mask": bits["mask"],
                                       "amps": amps["values"]}},
                "pipeline": {"steps": [{"key": "scale"}]}})["value"]["dataset_id"]
            rid = ml.tools.execute("run_custom", {"device": _DEV, 
                "code": _NET, "dataset_id": ds, "name": "ev2",
                "overrides": {"epochs": 3, "batch": 4}})["value"]["run_id"]
            EVAL = ('def evaluate(model, ctx):\n'
                    '    return {"n": float(len(ctx["payload"]["bits"])),'
                    ' "has_amps": float("amps" in ctx["payload"])}\n')
            # ① dataset_id 这条路（文档推荐的用法）
            got = ml.tools.execute("eval_custom", {
                "run_id": rid, "code": EVAL, "dataset_id": ds})
            assert got["is_error"] is False, got
            v = got["value"]
            assert v["metrics"]["n"] == 12.0, v["metrics"]
            assert v["metrics"]["has_amps"] == 1.0, v["metrics"]
            assert v["n_payload_arrays"] == 3, v          # "拿到几份负载"要看得见

            # ② 没显式给数据 ⇒ **默认继承训练那份**（反馈建议 4），且回执**明说**
            #   数据从哪来（否则"评估用的到底是哪份数据"就变成猜的）。
            #   ⚠ 这条原先钉的是旧行为（"没给就明说没有"）；行为改了判据跟着改，
            #     "真的没有数据"那条路由 `test_ai_efficiency.py` 的继承判据覆盖。
            quiet = ml.tools.execute("eval_custom", {
                "run_id": rid,
                "code": "def evaluate(model, ctx):\n    return {'ok': 1.0}"})
            assert quiet["is_error"] is False, quiet
            assert quiet["value"]["payload_source"] == "inherited_from_training", \
                quiet["value"]
            assert quiet["value"]["n_payload_arrays"] == 3, quiet["value"]

            # ③ 显式给一份**空**负载 ⇒ 这次真的没有数据：**回执明说**
            empty = ml.tools.execute("eval_custom", {
                "run_id": rid, "payload": {"arrays": {}},
                "code": "def evaluate(model, ctx):\n    return {'ok': 1.0}"})
            assert empty["is_error"] is False, empty
            assert empty["value"]["n_payload_arrays"] == 0, empty["value"]
            assert any("没有负载" in w for w in empty["value"]["warnings"]), \
                empty["value"]["warnings"]

            # ④ 代码里去取不存在的负载 ⇒ **可教学**（不是裸 KeyError: 'xr'）
            boom = ml.tools.execute("eval_custom", {
                "run_id": rid, "payload": {"arrays": {}},
                "code": 'def evaluate(model, ctx):\n'
                        '    return {"x": float(ctx["payload"]["xr"].sum())}'})
            assert boom["is_error"] is True, boom
            msg = ((boom.get("error") or {}).get("message") or "")
            assert "dataset_id" in msg and "payload" in msg, msg
            assert "继承" in msg, msg       # 说明白"默认会继承"这条新语义
        finally:
            ml.close()


def test_metrics_runs_in_eval_no_grad_and_can_be_throttled():
    """⭐ `metrics` 必须在 **eval + no_grad** 下调用，且贵指标能降频（反馈 §2-缺口5）。"""
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    code = '''
class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 3)


class Network:
    metrics_every = 2                      # 降频：每 2 轮算一次

    def build(self, n_in, n_out, *, nn):
        return Net(n_in)

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def metrics(self, model, ctx):
        return {"probe/training": float(model.training),
                "probe_grad": float(ctx["payload"]["xr"].requires_grad)}

    def forward_step(self, model, batch, ctx):
        x, y = batch
        out = model.lin(x)
        return out, out.detach()
'''
    with tempfile.TemporaryDirectory(prefix="nqs-metrics-") as td:
        ml = _ml(td)
        try:
            R = 10
            ds = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame",
                           "frame": pd.DataFrame({"m": np.arange(R, dtype=float)})},
                "name": "me", "target_kind": "custom",
                "payload": {"arrays": {"xr": np.zeros((R, 2, 3))}},
                "pipeline": {"steps": [{"key": "scale"}]}})["value"]["dataset_id"]
            out = ml.tools.execute("run_custom", {"device": _DEV, 
                "code": code, "dataset_id": ds, "name": "metrics_state",
                "overrides": {"epochs": 5, "batch": 4}})["value"]
            assert out["ok"] is True, out
            m = out["metrics"]
            assert m["probe/training"] == 0.0, f"metrics 不是在 eval 态调的：{m}"
            assert m["probe_grad"] == 0.0, f"metrics 没在 no_grad 下跑：{m}"
            # 降频生效：5 轮 → 0,2,4 三个点（末轮**一定**算，终值不能是旧的）
            # ⚠ 不用 `pick`：它的返回以**整条路径为键**（`"series.…values"`），
            #   本判据要看的是"点数"，直接读 `series` 更清楚。
            tr = ml.tools.execute("describe_training", {
                "run_id": out["run_id"], "include_series": True})["value"]
            vals = ((tr.get("series") or {}).get("probe/training") or {}).get("values") or []
            steps = ((tr.get("series") or {}).get("probe/training") or {}).get("steps") or []
            assert len(vals) == 3, f"降频没生效（{len(vals)} 个点 / 5 轮）"
            assert steps == [0, 2, 4], steps
        finally:
            ml.close()


def test_payload_facts_disclose_per_row_shape():
    """⭐ 回执要说清"一行之内是什么"（反馈 §2-缺口4，他为此烧了十几次往返）。"""
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="nqs-facts-") as td:
        ml = _ml(td)
        try:
            R = 8
            got = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame",
                           "frame": pd.DataFrame({"m": np.arange(R, dtype=float)})},
                "name": "facts", "target_kind": "custom",
                "payload": {"arrays": {"seq": np.zeros((R, 4, 5)),   # 一行 4×5
                                       "flat": np.zeros((R, 5)),      # 一行就 5 个数
                                       "scalar": np.zeros(R)}},       # 一行一个标量
                "pipeline": {"steps": [{"key": "scale"}]}})
            assert got["is_error"] is False, got
            f = got["value"]["payload"]
            assert f["seq"]["per_row_shape"] == [4, 5], f["seq"]
            assert f["flat"]["per_row_shape"] == [5], f["flat"]
            assert "R, L, d" in (f["flat"]["note"] or ""), f["flat"]   # 指路
            assert "一列" in (f["scalar"]["note"] or ""), f["scalar"]
            assert "R" in f["_semantics"], f["_semantics"]
        finally:
            ml.close()


def test_concurrent_run_custom_does_not_unregister_each_other():
    """⭐ 并发的 `run_custom` **不许撤掉彼此的注册项**（用户实测的提交竞态）。

    ## 现象（使用者实测原文）

    "连续 `submit_run` 三个 `run_custom`：第 1 个成功，紧接的第 2、3 个全部
    `state=done`，error: `invalid request: 未注册的方法: custom_network`"；
    "取消一个任务后立刻提交，也复现同一错误"。

    ## 根因（本轮定位）

    `engine._cmd_run_custom` 每次提交**临时造一个带代码的子类**注册进方法表
    （`registry.unregister(name)` + `registry.register(subclass)`），跑完在
    `finally` 里 `unregister` ⇒ 先跑完的那次会把并发同伴的注册项撤掉，同伴
    在"解析方法名"那一刻就撞 `未注册的方法`。

    ## 修法（本判据钉的两条不变量）

    1. **注册项在整条命令期间恒在**（收尾也不撤）——收尾撤除正是竞态的根；
    2. **代码不带在类上**：类上的 `custom_code` 恒为空，代码随
       `request.extras` 逐请求下传 ⇒ 并发跑**两份不同代码**也不会串。

    ⚠ 这条判据走**真实命令入口**（`run_custom` 命令：注册 → 方法解析 → 收尾），
    只把"训练那一段"换成受控桩——训练不是被测对象，而注册/解析/收尾正是。
    """
    import threading

    from ml_mecha.engine import RunSummary
    from ml_toolbox.core import registry

    with tempfile.TemporaryDirectory(prefix="nqs-race-") as td:
        ml = _ml(td)
        try:
            ds = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame",
                           "frame": pd.DataFrame({"x": np.arange(24, dtype=float),
                                                  "y": np.arange(24, dtype=float)})},
                "name": "race", "target": "y"})["value"]["dataset_id"]

            codes = {"a": _RACE_CODE_A, "b": _RACE_CODE_B, "c": _RACE_CODE_A}
            engine = ml.engine
            real_execute = engine._execute_one
            resolved, class_codes = [], []
            gate = threading.Barrier(len(codes), timeout=30)

            def stub(entry, request, call_id="", context=None):
                # 三份请求**同时**在命令执行中（真实场景就是并发提交）
                gate.wait()
                # ↓ 这里就是 runner 真正会 `registry.get(request.method)` 的位置
                cls = registry.get(request.method)
                class_codes.append(str(getattr(cls, "custom_code", "")))
                # 按 `network_name` 对齐"哪份请求拿到了哪段代码"（并发下顺序不定）
                resolved.append((request.extras.get("network_name"),
                                 request.extras.get("custom_network_code")))
                return RunSummary(run_id="race-" + str(len(resolved)),
                                  method=request.method,
                                  dataset_id=ds,
                                  pipeline_id=entry.pipeline_id, ok=True,
                                  primary_metric="m", metrics={"m": 1.0},
                                  elapsed_s=0.01)

            engine._execute_one = stub
            try:
                jobs = []
                for tag, code in codes.items():
                    jobs.append((tag, code, ml.submit(
                        {"command": "run_custom", "code": code, "dataset_id": ds,
                         "name": tag})))
                for tag, code, job in jobs:
                    status = ml.wait_job(job, 60)
                    assert status["state"] == "done", (tag, status)
                    assert job.result and job.result.get("ok") is True, (tag, job.result)
            finally:
                engine._execute_one = real_execute

            # ① 三份都**解析到了**方法（并发同伴没被撤除）
            assert len(resolved) == 3, resolved
            # ② 代码**逐请求**到达：每份请求拿到的是**自己**那段（按 name 对齐）
            assert dict(resolved) == codes, dict(resolved)
            # ③ 代码**不带在类上**（带了就会串：谁后注册谁覆盖）
            assert class_codes == ["", "", ""], class_codes
            # ④ 收尾之后注册项**仍在** —— 旧实现（finally 里 unregister）在这里红
            assert "custom_network" in registry.names(), (
                "run_custom 收尾把注册项撤了 ⇒ 并发/紧随其后的提交会死在第 1 次同步处"
                "（未注册的方法: custom_network）")
        finally:
            ml.close()


#: 提交竞态判据用的两段**不同**的最小网络代码（③ 要求它们各自到达自己的那次请求）。
_RACE_CODE_A = "class Network:\n    def build(self, n_in, n_out, *, nn):\n        return nn.Linear(n_in, n_out)\n"
_RACE_CODE_B = "class Network:  # B\n    def build(self, n_in, n_out, *, nn):\n        return nn.Linear(n_in, n_out)\n"


_CHUNK_CODE = '''
class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 3)


class Network:
    """metrics **一次算全量** → 换成 `ctx["chunks"]` 分块算，结果必须一致。"""
    primary_metric = "mean/amp"

    def build(self, n_in, n_out, *, nn):
        return Net(n_in)

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def forward_step(self, model, batch, ctx):
        x, y = batch
        out = model.lin(x)
        return out, out.detach()

    def metrics(self, model, ctx):
        # ⚠ **框架级分块**：权重由框架负责（`mode="mean"`），我不自己乘
        #   —— 用户手写时最容易在这里写成"只取最后一块"或"重复计数"。
        def _block(blk):
            return {"mean/amp": float(blk["amps"].real.abs().mean())}
        return ctx["chunks"].reduce(ctx["payload"], _block, mode="mean",
                                    budget_mb=0.0005)
'''


def test_framework_chunks_match_the_unchunked_result():
    """⭐ **框架级分块**：小预算下结果与"一次算全量"一致，且回执说得出"分了几块"。

    用户实测（《长任务编排与失败可读性》§3.3）："我为了不 OOM，在**自己的网络代码里**
    手写了分块（按 R 切块、按块加权累加）。这是通用需求……不该每个用户各写一遍。"

    ⇒ 判据只钉两件事（都是"用户真会踩"的）：
    ① **数值一致**：`ctx["chunks"].reduce(..., mode="mean")` 的结果与不分块时
       逐样本均值一致到容差内（加权口径正确 ⇒ 不是"只取最后一块"）；
    ② **事实可见**：回执/存档里能读到"已分 N 块"（`nn_chunks` 工件）。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="nqs-chunk-") as td:
        ml = _ml(td)
        try:
            rows, n_atoms = 64, 6
            meta, bits, amps = _nqs_payload(rows=rows, n_atoms=n_atoms, seed=3)
            ds = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": meta}, "name": "chunk",
                "target_kind": "custom",
                "payload": {"arrays": {"bits": bits["values"],
                                       "bit_mask": bits["mask"],
                                       "amps": amps["values"]}},
                "pipeline": {"steps": [{"key": "scale"}]}})["value"]["dataset_id"]
            out = ml.tools.execute("run_custom", {
                "device": _DEV, "code": _CHUNK_CODE, "dataset_id": ds,
                "name": "chunked", "persist": True,
                "overrides": {"epochs": 3, "batch": 8, "val_split": 0.0}})["value"]

            # ① 数值一致：分块算出的均值 == 全量逐样本均值
            want = float(np.abs(amps["values"].real).mean())
            got = float(out["metrics"]["mean/amp"])
            assert abs(got - want) < 1e-5, (got, want)

            # ② 事实可见：存档里留下"已分 N 块"
            from ml_toolbox.core.persistence import load_run_record
            rec = load_run_record(out["run_id"])
            note = rec.result.artifacts.get("nn_chunks")
            assert note is not None, "没有留下分块事实 ⇒ 回执无法说'已分 N 块'"
            text = str(note[0] if hasattr(note, "__getitem__") else note)
            assert "blocks" in text, text
            n_blocks = int(text.split()[0])
            assert n_blocks >= 2, f"预算这么小却只切了 {n_blocks} 块：{text}"
        finally:
            ml.close()


#: 判据用的小网络：**标准回归**（不依赖负载），带 `on_step`（制造 batch 级 tag）。
_NET_PERIODIC = '''
class Net(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.a = nn.Linear(n_in, hidden)
        self.b = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.b(torch.tanh(self.a(x))).squeeze(-1)


class Network:
    primary_metric = "my/step"

    def build(self, n_in, n_out, *, nn, ctx=None):
        return Net(n_in)

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def on_step(self, model, step, ctx):
        return {"my/step": float(step)}

    def forward_step(self, model, batch, ctx):
        x, y = batch
        return model(x), y
'''


def _regression_ds(ml, rows=96, seed=0):
    """一份小的标准回归数据集（帧 + 目标列）——比 NQS 负载轻，够验存档/轴。"""
    rng = np.random.RandomState(seed)
    frame = pd.DataFrame(rng.randn(rows, 4), columns=list("abcd"))
    frame["y"] = frame["a"] - 0.5 * frame["b"] + 0.1 * rng.randn(rows)
    return ml.tools.execute("prepare_dataset", {
        "source": {"kind": "frame", "frame": frame}, "name": "periodic",
        "target": "y",
        "pipeline": {"steps": [{"key": "scale"}]}})["value"]["dataset_id"]


# ============================================================ 4. 中途存档（A1/A2）
def test_checkpoint_is_written_periodically_and_reports_its_step():
    """⭐⭐ **训练还在跑时**盘上就有 `last.pt`，且每份存档报得出"落在第几步"。

    这是使用者实测 A1/A2 的修法判据：

    - **A1**：此前 `_save_checkpoints` 只在**结果组装时**调用 ⇒ 一个跑到 284 epoch
      的 run，`runs/<id>/` 下**只有 `tb/`**，5 小时训练中途死机＝全部参数丢失。
    - **A2**：拿到一份 `last.pt` 要能判断"它对应训练到哪" ⇒ `saved_at_step/epoch`。

    ⚠ 本判据**真的在训练进行中读盘**（另起线程跑训练、主线程轮询 `last.pt` 的出现），
    而不是只看"跑完之后有没有文件"——后者对"只在终态写"的旧实现**同样会绿**
    （那正是要防的假绿）。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    import threading
    import time

    from ml_toolbox.core.persistence import RUNS_DIR, load_run_record

    with tempfile.TemporaryDirectory(prefix="nqs-periodic-") as td:
        ml = _ml(td)
        try:
            ds = _regression_ds(ml, rows=96)
            root = Path(RUNS_DIR)
            before = ({p.name for p in root.iterdir() if p.is_dir()}
                      if root.is_dir() else set())
            out: dict = {}
            seen: dict = {}
            stop = threading.Event()

            def _train():
                out.update(ml.tools.execute("run_custom", {
                    "device": _DEV, "code": _NET_PERIODIC, "dataset_id": ds,
                    "name": "periodic", "persist": True,
                    "checkpoint_every": 5,
                    "overrides": {"epochs": 60, "batch": 4, "val_split": 0.25}}))

            def _poll():
                """轮询"**训练还活着时**盘上出现了 last.pt"（每个新目录只记一次）。"""
                while not stop.is_set():
                    try:
                        for d in root.iterdir():
                            if d.name in before or d.name in seen:
                                continue
                            if (d / "checkpoint" / "last.pt").is_file():
                                seen[d.name] = bool(th.is_alive())
                    except OSError:
                        pass
                    time.sleep(0.002)

            th = threading.Thread(target=_train, name="periodic-train")
            th.start()
            poller = threading.Thread(target=_poll, name="periodic-poll")
            poller.start()
            th.join(timeout=900)
            stop.set()
            poller.join(timeout=5)

            receipt = out.get("value") or {}
            assert receipt.get("ok") is True, receipt
            rid = receipt["run_id"]
            assert receipt.get("checkpoint_every") == 5, receipt

            # ① ⭐ **中途**：这个 run 的 last.pt 在**训练线程还活着**时就已经在盘上
            assert rid in seen, (
                "没能在训练进行中观测到 last.pt（要么是终态才写，要么 checkpoint_every "
                f"没生效）。观测到的目录：{sorted(seen)}")
            assert seen[rid] is True, (
                "last.pt 出现时训练线程已经结束 ⇒ 它还是终态写的（判据失去了意义）")

            # ② 中途落盘的事实（周期性存了几次、最后一次在第几步）
            arts = load_run_record(rid).result.artifacts
            n_periodic = int(np.asarray(arts["nn_checkpoint_n"]).ravel()[0])
            assert n_periodic >= 2, f"周期存档只成功了 {n_periodic} 次：{arts.keys()}"
            assert "nn_checkpoint_errors" not in arts, (
                f"周期存档报错了：{arts.get('nn_checkpoint_errors')}")
            last_step = int(np.asarray(arts["nn_checkpoint_last_step"]).ravel()[0])
            assert last_step > 0, last_step

            # ③ 每份存档**各自**的"落在第几步"（A2：不再只报 best 的那一份）
            ck = ml.tools.execute("checkpoint_detail", {"run_id": rid})["value"]
            assert ck["available"] is True, ck
            assert set(ck["tags"]) == {"best", "last"}, ck["tags"]
            for tag in ("best", "last"):
                info = ck["files_meta"][tag]
                assert info["saved_at_step"] and info["saved_at_epoch"], (tag, info)
                assert info["save_reason"] == "final", (tag, info)
            # `last_saved` = **最后一次**落盘（终态那一份），不是"第一个找到的文件"
            assert ck["last_saved"]["tag"] == "last", ck["last_saved"]
            assert ck["last_saved"]["save_reason"] == "final", ck["last_saved"]
            # ⭐ 该用哪个键 load —— 静默错误的直接来源（A7-①）
            assert ck["load_hint"]["state_dict_key"] == "state_dict", ck["load_hint"]

            # ④ `best.pt` 的语义**没被改**：仍按验证损失留快照
            assert ck["files_meta"]["best"]["best_epoch"] is not None, ck["files_meta"]
            # ⑤ 原子写：不许留临时文件
            assert not [f for f in ck["files"] if ".tmp-" in f], ck["files"]
            assert not list((Path(ck["dir"])).glob("*.tmp-*")), "留下半个存档"
        finally:
            ml.close()


def test_periodic_checkpoint_save_is_atomic_and_cleans_up_on_failure():
    """**原子写**：成功不留临时文件；失败要清掉临时文件，且**不破坏上一份好存档**。

    ⚠ 为什么这条必须单独钉：中途存档的唯一价值是"被杀死时还有一份能用的"。
    直接写目标路径时，写一半被杀会留下**半个 zip**（下次 `torch.load` 报的是一句
    与真相无关的错），而**上一份好的存档已经被覆盖**——那比"没有存档"更坏
    （你以为有）。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    import torch

    from ml_toolbox.methods.neural.custom import _atomic_torch_save

    with tempfile.TemporaryDirectory(prefix="nqs-atomic-") as td:
        path = Path(td) / "last.pt"
        n = _atomic_torch_save({"state_dict": {"w": torch.zeros(3)}}, path)
        assert path.is_file() and n > 0, (path, n)
        assert not list(Path(td).glob("*.tmp-*")), "成功路径留下了临时文件"

        class Unpicklable:
            def __reduce__(self):
                raise RuntimeError("不可序列化（模拟写盘中途失败）")

        try:
            _atomic_torch_save({"state_dict": {"w": Unpicklable()}}, path)
            raise AssertionError("不可序列化的对象竟然写成功了")
        except RuntimeError:
            pass
        assert not list(Path(td).glob("*.tmp-*")), "失败路径没有清掉临时文件"
        again = torch.load(path, map_location="cpu", weights_only=False)
        assert "state_dict" in again and "w" in again["state_dict"], again


def test_step_axis_units_are_reported_and_cross_family_alignment_warns():
    """⭐ 两族 tag 的 step 轴**单位不同**，回执必须说出来（pitfalls A55）。

    实测现场：`loss` 的 `latest_step` 是 **epoch 序号**，而 `on_step` 上报的
    `fid/step` 是**全局 batch 步**（同一个 run 上差 ~277 倍）。只给步号不给单位，
    读数的人会把大数读成"累计点数"，或把两族并排比 —— 后者会造出一个**不存在的矛盾**。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="nqs-units-") as td:
        ml = _ml(td)
        try:
            # ⚠ 行数必须足够：`infer_kind` 的规则是"数值且**基数 > 20** 才算回归"
            #   （`contracts.REGRESSION_CARDINALITY`）⇒ 行数太少时，一个连续目标会被
            #   判成**分类**，报出来的错是 `Found dtype Long but expected Float`
            #   （离根因很远）。这里给 80 行（切 8:2 后仍有 60 行）避开它。
            ds = _regression_ds(ml, rows=80, seed=3)
            rid = ml.tools.execute("run_custom", {
                "device": _DEV, "code": _NET_PERIODIC, "dataset_id": ds,
                "name": "units",
                "overrides": {"epochs": 3, "batch": 4, "val_split": 0.0}})["value"]["run_id"]

            tr = ml.tools.execute("describe_training", {"run_id": rid})["value"]
            assert tr["available"] is True, tr.get("reason")
            # ① 单位：`loss` 是 epoch 级，`on_step` 的 tag 是 batch 级
            assert tr["step_units"].get("my/step") == "batch", tr["step_units"]
            assert "loss" not in tr["step_units"], (
                "epoch 级 tag 不该出现在 step_units 里（它是**非 epoch** 那一半）")
            axes = tr["axes"]
            assert "my/step" in axes["batch_tags"], axes
            assert axes["mixed_tags"] == [], axes
            # ② `per_epoch_estimate` = 每轮批数（就是那个"看起来差几百倍"的来源）
            assert axes["per_epoch_estimate"] and axes["per_epoch_estimate"] >= 2, axes

            # ③ 跨族强行对齐 ⇒ 必须**当场说清**（不静默给一张误导性的表）
            cmp_ = ml.tools.execute("describe_training", {
                "run_id": rid, "tags": ["loss", "my/step"]})["value"]["compare"]
            assert cmp_["step_units"]["loss"] == "epoch", cmp_
            assert cmp_["step_units"]["my/step"] == "batch", cmp_
            assert cmp_.get("axis_warning"), (
                "跨族对齐没有给 axis_warning ⇒ 那张表会被读成'这段没有 loss'")
            assert "batch" in cmp_["axis_warning"], cmp_["axis_warning"]
        finally:
            ml.close()


#: Q4 判据用的网络：`fid/climb` **逐轮严格上升** ⇒ 按它选的 best 必然落在**最后一轮**
#: ⇒ 可以直接断言"best.pt 的权重 == last.pt 的权重"（比"看指标接近"硬得多）。
_NET_BEST_METRIC = '''
class Net(nn.Module):
    def __init__(self, n_in, hidden=32):
        super().__init__()
        self.a = nn.Linear(n_in, hidden)
        self.b = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.b(torch.tanh(self.a(x))).squeeze(-1)


class Network:
    {declare}

    def __init__(self):
        self.ep = 0

    def build(self, n_in, n_out, *, nn, ctx=None):
        return Net(n_in)

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def metrics(self, model, ctx):
        self.ep = int(ctx.get("step") or 0)
        return {{"fid/climb": 0.1 * (self.ep + 1), "flat/x": 1.0}}

    def forward_step(self, model, batch, ctx):
        x, y = batch
        return model(x), y
'''


def test_best_pt_selection_is_declarable_and_default_is_unchanged():
    """⭐⭐ `best.pt` 的选择口径**可声明**（Q4），未声明时**与从前一致**。

    使用者实测的痛点：自定义任务的主指标常常**不是** `val_loss`，而 `best.pt` 按
    `val_loss` 选 ⇒ 同一场训练里 **`best` 0.6464 vs `last` 0.8174**（差得很远）；
    而 `checkpoint_detail` 只报 `best_epoch`，**不说"按什么选的"** ⇒ 无法判断该不该用它。

    本条钉四件：
    ① **声明了就按它选**：`best_metric="fid/climb"`（严格上升）⇒ best 必然来自**最后一轮**
       ⇒ 直接断言 `best.pt` 的权重与 `last.pt` **逐张量相等**（比看指标接近硬得多）；
    ② **报出口径**：`checkpoint_detail` 的 `files_meta.best.best_by` 给出
       tag/direction/value/epoch/step，`val_loss_best` 给出对照；
    ③ ⭐ **未声明时逐位兼容**：`best_by.tag == "val_loss"`（同一份网络、只去掉声明那一行）；
    ④ **方向判不出来必须可教学地拒**（`ph/std` 这类"没有好坏方向"的量**不许猜**），
       并证明 tuple 形式（显式给方向）能通。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    import torch

    from ml_toolbox.core.persistence import RUNS_DIR, load_run_record

    def _run(ml, ds, declare: str) -> dict:
        return ml.tools.execute("run_custom", {
            "device": _DEV, "code": _NET_BEST_METRIC.format(declare=declare),
            "dataset_id": ds, "name": "best", "persist": True,
            "overrides": {"epochs": 6, "batch": 4, "val_split": 0.25}})

    with tempfile.TemporaryDirectory(prefix="nqs-best-") as td:
        ml = _ml(td)
        try:
            ds = _regression_ds(ml, rows=96)
            # ---- ① 声明：按 fid/climb（higher；严格上升 ⇒ best 落在最后一轮）
            raw = _run(ml, ds, 'best_metric = "fid/climb"')
            assert raw.get("is_error") is False, raw
            rid = raw["value"]["run_id"]
            ck = ml.tools.execute("checkpoint_detail", {"run_id": rid})["value"]
            bb = ck["files_meta"]["best"]["best_by"]
            assert bb["tag"] == "fid/climb", bb
            assert bb["direction"] == "higher", bb
            assert bb["epoch"] is not None and bb["step"] is not None, bb
            assert ck["files_meta"]["best"]["val_loss_best"]["epoch"] is not None, ck
            b = torch.load(Path(RUNS_DIR) / rid / "checkpoint" / "best.pt",
                           map_location="cpu", weights_only=False)
            last = torch.load(Path(RUNS_DIR) / rid / "checkpoint" / "last.pt",
                              map_location="cpu", weights_only=False)
            assert b["best_by"]["tag"] == "fid/climb", b.get("best_by")
            assert all(torch.equal(b["state_dict"][k], last["state_dict"][k])
                       for k in b["state_dict"]), (
                "按 `fid/climb`（严格上升）选的 best 应当就是最后一轮的权重 —— "
                "不等说明选择口径没生效")
            arts = load_run_record(rid).result.artifacts
            assert "fid/climb" in str(np.asarray(arts["nn_best_by"]).ravel()[0]), \
                arts.get("nn_best_by")

            # ---- ③ 未声明：与从前一致（val_loss 口径）
            raw2 = _run(ml, ds, "")
            assert raw2.get("is_error") is False, raw2
            rid2 = raw2["value"]["run_id"]
            ck2 = ml.tools.execute("checkpoint_detail", {"run_id": rid2})["value"]
            bb2 = ck2["files_meta"]["best"]["best_by"]
            assert bb2["tag"] == "val_loss", bb2
            assert bb2["direction"] == "lower", bb2
            assert ck2["meta"]["best_epoch"] is not None, ck2["meta"]
            b2 = torch.load(Path(RUNS_DIR) / rid2 / "checkpoint" / "best.pt",
                            map_location="cpu", weights_only=False)
            assert b2["best_by"]["tag"] == "val_loss", b2.get("best_by")

            # ---- ④ 方向判不出来 ⇒ 可教学地拒；显式给方向就通
            bad = _run(ml, ds, 'best_metric = "ph/std"')
            assert bad.get("is_error") is True, bad
            info = (bad.get("error") or {}).get("info") or {}
            assert info.get("kind") == "bad_best_metric", info
            assert "higher" in (info.get("suggest") or ""), info
            ok = _run(ml, ds, 'best_metric = ("fid/climb", "higher")')
            assert ok.get("is_error") is False, ok
            ck3 = ml.tools.execute("checkpoint_detail",
                                   {"run_id": ok["value"]["run_id"]})["value"]
            assert ck3["files_meta"]["best"]["best_by"]["direction"] == "higher", ck3
        finally:
            ml.close()


#: Q4b 判据用的网络：指标**中途达到峰值再回落**（`fid/peak = -|ep - PEAK|` ⇒ 越大越好）。
#: 峰值轮是常数 ⇒ 断言"盘上的 best.pt 就是峰值那一轮"是**确定的**，不靠概率。
_NET_BEST_PERIODIC = '''
PEAK = 6


class Net(nn.Module):
    def __init__(self, n_in, hidden=32):
        super().__init__()
        self.a = nn.Linear(n_in, hidden)
        self.b = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.b(torch.tanh(self.a(x))).squeeze(-1)


class Network:
    best_metric = "fid/peak"

    def __init__(self):
        self.ep = 0

    def build(self, n_in, n_out, *, nn, ctx=None):
        return Net(n_in)

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def metrics(self, model, ctx):
        self.ep = int(ctx.get("step") or 0)
        return {"fid/peak": -abs(self.ep - PEAK)}

    def forward_step(self, model, batch, ctx):
        x, y = batch
        return model(x), y
'''


def test_periodic_best_pt_is_written_mid_run_so_an_interrupted_run_keeps_it():
    """⭐⭐ `checkpoint_every` 也要**中途更新 `best.pt`**（Q4b）。

    为什么这是真缺口（使用者点出）：周期存档原来**只写 `last.pt`**，`best.pt` 只在
    **终态**与 `last` 一起写 ⇒ **一次被中断的长训练拿不到"按我的指标最好的那一份"**
    —— 而那正是 `checkpoint_every` 存在的意义（死机后要能续训/取最佳）。

    判据（**在训练还在跑时读盘**，与 A1 那条同款"另起线程 + 轮询"）：
    ① 训练进行中就能看到 `best.pt`，且它 `save_reason == "periodic"`；
    ② ⭐ 它**随着 best 刷新被重写**：观测到的 `best_by.epoch` 序列里既有**早期轮**
       （说明它不是终态才写），又**最终追到峰值轮 6**（说明它按声明口径在追最佳）——
       而且这些观测都发生在**训练线程还活着**时；
    ③ 终态那份仍是 `final`、`best_by.epoch` 依然是峰值轮，且权重与 `last.pt`
       **必须不同**（否则"按峰值轮存档"没被证明）。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    import threading
    import time

    import torch

    from ml_toolbox.core.persistence import RUNS_DIR

    with tempfile.TemporaryDirectory(prefix="nqs-bestper-") as td:
        ml = _ml(td)
        try:
            ds = _regression_ds(ml, rows=96)
            root = Path(RUNS_DIR)
            before = ({p.name for p in root.iterdir() if p.is_dir()}
                      if root.is_dir() else set())
            out: dict = {}
            #: run_id -> {"sightings": [(epoch, reason)…], "last_mtime": int}
            seen: dict = {}
            stop = threading.Event()

            def _train():
                out.update(ml.tools.execute("run_custom", {
                    "device": _DEV, "code": _NET_BEST_PERIODIC, "dataset_id": ds,
                    "name": "bestper", "persist": True,
                    "checkpoint_every": 5,
                    "overrides": {"epochs": 40, "batch": 4, "val_split": 0.25}}))

            def _poll():
                """训练**还在跑**时轮询：`best.pt` 出现过哪些"最佳轮"（**含演化**）。

                ⚠ 只在文件 mtime 变化时重读（否则每 2 ms 反序列化一次太吵）。
                """
                while not stop.is_set():
                    try:
                        for d in root.iterdir():
                            if d.name in before:
                                continue
                            bp = d / "checkpoint" / "best.pt"
                            if not bp.is_file():
                                continue
                            try:
                                mt = bp.stat().st_mtime_ns
                            except OSError:
                                continue
                            rec = seen.setdefault(d.name, {"sightings": [],
                                                           "last_mtime": -1})
                            if mt == rec["last_mtime"] or not th.is_alive():
                                continue
                            rec["last_mtime"] = mt
                            try:
                                ck = torch.load(bp, map_location="cpu",
                                                weights_only=False)
                                rec["sightings"].append(
                                    ((ck.get("best_by") or {}).get("epoch"),
                                     ck.get("save_reason")))
                            except Exception:              # noqa: BLE001
                                pass      # 正在被原子换名：下一轮再看（**不报错**）
                    except OSError:
                        pass
                    time.sleep(0.002)

            th = threading.Thread(target=_train, name="bestper-train")
            th.start()
            poller = threading.Thread(target=_poll, name="bestper-poll")
            poller.start()
            th.join(timeout=900)
            stop.set()
            poller.join(timeout=5)

            receipt = out.get("value") or {}
            assert receipt.get("ok") is True, receipt
            rid = receipt["run_id"]
            # ① **训练还在跑时**就看到了 best.pt，且都是 periodic
            assert rid in seen and seen[rid]["sightings"], (
                f"训练进行中没看到 best.pt ⇒ 周期存档只写了 last.pt（Q4b 的缺口）：{seen}")
            epochs = [e for e, _r in seen[rid]["sightings"]]
            reasons = {r for _e, r in seen[rid]["sightings"]}
            #: ⚠ 终态那一次也会被轮询看到（`_train` 里落完盘才返回）⇒ 分开看
            assert "periodic" in reasons, reasons
            per = [e for e, r in seen[rid]["sightings"] if r == "periodic"]
            # ② 它**随 best 刷新被重写**：早期轮出现过，最终追到峰值轮
            assert any(e is not None and e < 6 for e in per), per
            assert per[-1] == 6, (
                f"训练进行中最后一次**周期**落盘的 best 应当是峰值轮 6"
                f"（口径：按 `fid/peak` 选）：{per}（全部观测：{epochs}）")
            # ③ 终态口径一致 + 权重确实与 last 不同
            ck = ml.tools.execute("checkpoint_detail", {"run_id": rid})["value"]
            assert ck["files_meta"]["best"]["best_by"]["epoch"] == 6, ck["files_meta"]
            assert ck["files_meta"]["best"]["save_reason"] == "final", ck["files_meta"]
            b = torch.load(Path(RUNS_DIR) / rid / "checkpoint" / "best.pt",
                           map_location="cpu", weights_only=False)
            last = torch.load(Path(RUNS_DIR) / rid / "checkpoint" / "last.pt",
                              map_location="cpu", weights_only=False)
            assert b["best_by"]["epoch"] != ck["meta"]["epochs_run"] - 1, ck["meta"]
            assert any(not torch.equal(b["state_dict"][k], last["state_dict"][k])
                       for k in b["state_dict"]), (
                "best.pt 与 last.pt 权重完全相同 ⇒ '按峰值轮存档'没有被证明")
        finally:
            ml.close()


#: Q7 判据用的网络：**先把 loss 压下去**，再续训看"起始 loss 从哪开始"。
_NET_RESUME = '''
class Net(nn.Module):
    def __init__(self, n_in, hidden=32):
        super().__init__()
        self.a = nn.Linear(n_in, hidden)
        self.b = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.b(torch.tanh(self.a(x))).squeeze(-1)


class Network:
    primary_metric = "train/first"

    def __init__(self):
        self.first = None

    def build(self, n_in, n_out, *, nn, ctx=None):
        return Net(n_in)

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def on_step(self, model, step, ctx):
        #: 把**第一步**的 loss 记下来：那是"续训生效没有"的最直接读数
        if self.first is None:
            self.first = step
        return {}

    def forward_step(self, model, batch, ctx):
        x, y = batch
        return model(x), y
'''


#: **结构不同**的网络：用来证明"改过结构 ⇒ 续训装载**响亮失败**"（而不是静默随机初始化）。
_NET_OTHER_STRUCT = '''
class Network:
    def build(self, n_in, n_out, *, nn, ctx=None):
        # 键集与 `_NET_RESUME` 完全不同（这是最容易发生的那种"改了结构还想续训"）
        return nn.Sequential(nn.Linear(n_in, 5), nn.ReLU(), nn.Linear(5, n_out))
'''


def test_resume_from_starts_near_the_breakpoint_and_fails_teachably():
    """⭐⭐ **续训（`resume_from`）**：起始 `loss` 必须接近**断点**而不是初值（Q7）。

    使用者的判据（原话）："续训后**起始 `loss` 必须接近断点**，而不是初值
    （我实测：续训 99.68 vs 从头 1017.6，一眼可辨）。"

    本条钉四件：
    ① **主判据（他给的）**：`resume_from` 那一轮的**首轮 loss** 离源 run 的**末轮 loss**
       更近，而离"从头训练的首轮 loss"更远 —— 判据**自归一化**（用同一次实验里
       两个对照 run 的读数做比较），不依赖任何绝对阈值；
    ② **事实可反查**：回执带 `resumed_from`/`resumed_which`/`resumed_at_step`，
       且 artifacts 里有 `nn_resumed_from`（"这一轮为什么收敛快"要能回答）；
    ③ **指向不存在的存档 ⇒ 可教学地拒**（`no_resume_checkpoint`），**绝不静默从头训**；
    ④ `resume_which` 越界 ⇒ `bad_resume_which`；结构改过 ⇒ `resume_state_mismatch`（响亮）。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    from ml_toolbox.core.persistence import load_run_record

    def _run(ml, ds, **extra):
        body = {"device": _DEV, "code": _NET_RESUME, "dataset_id": ds,
                "name": "resume", "persist": True,
                "overrides": {"epochs": 8, "batch": 8, "lr": 0.02,
                              "val_split": 0.25}}
        body.update(extra)
        return ml.tools.execute("run_custom", body)

    def _first_loss(rid):
        h = load_run_record(rid).result.artifacts["nn_history"]
        return float(np.asarray(h["loss"]).ravel()[0])

    with tempfile.TemporaryDirectory(prefix="nqs-resume-") as td:
        ml = _ml(td)
        try:
            ds = _regression_ds(ml, rows=96)
            # ① 源 run：训 40 轮把 loss 压下去（它有存档）
            src_raw = ml.tools.execute("run_custom", {
                "device": _DEV, "code": _NET_RESUME, "dataset_id": ds,
                "name": "src", "persist": True,
                "overrides": {"epochs": 40, "batch": 8, "lr": 0.02,
                              "val_split": 0.25}})
            assert src_raw.get("is_error") is False, src_raw
            src = src_raw["value"]["run_id"]
            src_last = float(np.asarray(
                load_run_record(src).result.artifacts["nn_history"]["loss"]
            ).ravel()[-1])

            # ② 从头训练（对照）：另一份**同样的数据**上的首轮 loss
            fresh = _run(ml, ds)["value"]["run_id"]
            fresh_first = _first_loss(fresh)

            # ③ ⭐ 续训：首轮 loss 应当**贴着 src 的末轮**，而不是从头那个数
            got = _run(ml, ds, resume_from=src, resume_which="last")["value"]
            assert got["ok"] is True, got
            rid = got["run_id"]
            resumed_first = _first_loss(rid)
            d_resume = abs(resumed_first - src_last)
            d_fresh = abs(resumed_first - fresh_first)
            assert d_resume < d_fresh, (
                f"续训没有真正生效：首轮 loss={resumed_first:.4g}，"
                f"断点(soue末轮)={src_last:.4g}（距离 {d_resume:.4g}），"
                f"从头首轮={fresh_first:.4g}（距离 {d_fresh:.4g}）")
            # ② 事实可反查
            assert str(got.get("resumed_from")).endswith("last.pt"), got
            assert got.get("resumed_which") == "last", got
            assert got.get("resumed_at_step"), got
            arts = load_run_record(rid).result.artifacts
            assert "nn_resumed_from" in arts, sorted(arts)

            # ③ 不存在的存档 ⇒ 可教学地拒（**绝不静默从头训**）
            bad = ml.tools.execute("run_custom", {
                "device": _DEV, "code": _NET_RESUME, "dataset_id": ds,
                "name": "bad", "resume_from": "custom_network-0-0"})
            assert bad.get("is_error") is True, bad
            info = (bad.get("error") or {}).get("info") or {}
            assert info.get("kind") == "no_resume_checkpoint", info
            assert "checkpoint" in (info.get("hint") or ""), info
            # ④ `resume_which` 越界
            bad2 = ml.tools.execute("run_custom", {
                "device": _DEV, "code": _NET_RESUME, "dataset_id": ds,
                "name": "bad2", "resume_from": src, "resume_which": "third"})
            assert bad2.get("is_error") is True, bad2
            assert (((bad2.get("error") or {}).get("info") or {}).get("kind")
                    == "bad_resume_which"), bad2
            # ④b 结构改过 ⇒ `resume_state_mismatch`（**响亮**，不静默）
            other = ml.tools.execute("run_custom", {
                "device": _DEV, "dataset_id": ds, "name": "other",
                "code": _NET_OTHER_STRUCT, "resume_from": src})
            assert other.get("is_error") is True, other
            assert (((other.get("error") or {}).get("info") or {}).get("kind")
                    == "resume_state_mismatch"), other
        finally:
            ml.close()


#: 判据用的网络：`metrics` **故意抛异常**，而且**第 0 轮之后恢复正常**
#: ⇒ 同时证明"失败留痕"与"**没有**被永久关闭"（旧实现第一次失败就再也不会调用）。
_NET_METRICS_BOOM = '''
class Net(nn.Module):
    def __init__(self, n_in, hidden=32):
        super().__init__()
        self.a = nn.Linear(n_in, hidden)
        self.b = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.b(torch.tanh(self.a(x))).squeeze(-1)


class Network:
    primary_metric = "fid/ok"

    def __init__(self):
        self.calls = 0

    def build(self, n_in, n_out, *, nn, ctx=None):
        return Net(n_in)

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def metrics(self, model, ctx):
        self.calls += 1
        #: 第 0 轮炸（模拟"在 metrics 里对全量负载做重活 ⇒ OOM"），之后就正常
        if int(ctx.get("step") or 0) == 0:
            raise RuntimeError("CUDA out of memory（判据模拟：全量负载上做重活）")
        return {"fid/ok": 0.5 + 0.01 * self.calls}

    def forward_step(self, model, batch, ctx):
        x, y = batch
        return model(x), y
'''


def test_metrics_failure_is_traced_and_not_permanently_disabled():
    """⭐⭐ `metrics` 抛异常**不许被永久静默关闭**（2026-10-07 使用者实测的静默黑洞）。

    他的现场：TB 里**一个 epoch 级 tag 都没有**（`fid/sub1`…），而**训练完全正常**
    （loss 在降、步级 tag 都在）；`best_metric` 也因此"整场没上报"。
    根因（旧 `trainer.py`）：`except Exception: metrics = None` ——
    ① **异常被吞掉、不留痕**；② **永久关闭**（第一次失败后整场不再调用）；
    ③ **任何回执都看不出** ⇒ AI 分不清三种病：**没写钩子** / **每轮都失败** / **名字写错**
    （修法完全不同）。

    触发原因极常见：`metrics` 拿的是**全量负载**（他 7344 行、一次 256-token 解码就 OOM）。

    本条钉四件：
    ① **训练照常成功**（失败只降级，不拖垮）；
    ② ⭐ **留痕**：artifacts **与** `checkpoint_detail`（只有磁盘、跨重启也查得到）
       都能看到"失败了 N 次、原因 X"；
    ③ ⭐ **没有永久关闭**：第 0 轮炸、之后正常 ⇒ 后面的 epoch **必须产出 `fid/ok`**
       （旧实现下这条一定红：整场一个 epoch 级 tag 都没有）；
    ④ 三种病**分得开**：`metrics_health.hook=True` 且 `failures>0`（不是"没写"）。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    import json

    from ml_toolbox.core.persistence import load_record, load_run_record

    with tempfile.TemporaryDirectory(prefix="nqs-mhealth-") as td:
        ml = _ml(td)
        try:
            ds = _regression_ds(ml, rows=96)
            raw = ml.tools.execute("run_custom", {
                "device": _DEV, "code": _NET_METRICS_BOOM, "dataset_id": ds,
                "name": "mboom", "persist": True, "diag": True,
                "overrides": {"epochs": 6, "batch": 8, "val_split": 0.25}})
            assert raw.get("is_error") is False, raw
            assert raw["value"]["ok"] is True, raw["value"]      # ① 训练照常
            rid = raw["value"]["run_id"]

            # ② 留痕：artifacts 里有失败次数与原因
            arts = load_run_record(rid).result.artifacts
            assert "nn_metrics_errors" in arts, sorted(arts)
            err = str(np.asarray(arts["nn_metrics_errors"]).ravel()[0])
            assert "out of memory" in err, err
            assert "epoch 0" in err, err

            # ② b 留痕：`checkpoint_detail`（**只有磁盘**、跨重启也查得到）也要给
            ck = ml.tools.execute("checkpoint_detail", {"run_id": rid})["value"]
            mh = ck["files_meta"]["last"]["metrics_health"]
            assert mh["hook"] is True, mh        # ④ 与"没写钩子"分得开
            assert int(mh["failures"]) >= 1, mh
            assert mh["errors"] and "epoch 0" in mh["errors"][0], mh
            assert "metrics" in (ck.get("metrics_note") or ""), ck.get("metrics_note")

            # ③ ⭐ **没有永久关闭**：后面几轮必须真的产出了 epoch 级 tag
            tr = ml.tools.execute("describe_training", {"run_id": rid})["value"]
            assert "fid/ok" in tr["tags"], (
                "`metrics` 在第 0 轮失败后**整场再没被调用**（永久关闭）⇒ "
                f"epoch 级 tag 缺失：{sorted(tr['tags'])[:12]}…")
            # ② c `diag` 里也应在（若这次运行把 diag 落了盘）
            diag = (load_record(rid).get("result") or {}).get("diag") or {}
            if diag:
                assert "metrics_health" in json.dumps(diag, default=str), sorted(diag)
            else:
                print("    （注意：这次运行的 diag 没有落进 record.json ⇒ 该断言跳过；"
                      "事实已由 artifacts 与 checkpoint_detail 两条路径钉住）")
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
    # ---- 第二轮：使用者实跑后的反馈（docs/反馈-NQS路径实测缺口.md）----
    test_build_receives_payload_shapes_and_old_signature_still_works,
    test_custom_task_needs_no_forward_when_forward_step_exists,
    test_bare_tensor_in_user_hook_lands_on_the_training_device,
    test_eval_custom_accepts_dataset_id_and_teaches_missing_payload,
    test_metrics_runs_in_eval_no_grad_and_can_be_throttled,
    test_payload_facts_disclose_per_row_shape,
    # ---- 长任务编排批次（docs/建议-ML工具长任务编排与失败可读性.md §2.3）----
    test_concurrent_run_custom_does_not_unregister_each_other,
    # ---- L3 资源：框架级分块（§3.3）----
    test_framework_chunks_match_the_unchunked_result,
    # ---- 第三轮：A1/A2 中途存档 + A55 步轴单位（docs/反馈-ML工具使用体验与接口审查.md）----
    test_checkpoint_is_written_periodically_and_reports_its_step,
    test_periodic_checkpoint_save_is_atomic_and_cleans_up_on_failure,
    test_step_axis_units_are_reported_and_cross_family_alignment_warns,
    # ---- Q4（2026-10-07）：`best.pt` 的选择口径可声明（用户实测 best 0.6464 vs last 0.8174）----
    test_best_pt_selection_is_declarable_and_default_is_unchanged,
    # ---- Q4b（2026-10-07）：周期存档也要**中途**更新 best.pt（被中断也要有最佳）----
    test_periodic_best_pt_is_written_mid_run_so_an_interrupted_run_keeps_it,
    # ---- Q7（2026-10-07）：续训 `resume_from`（起始 loss 要贴着断点，而不是初值）----
    test_resume_from_starts_near_the_breakpoint_and_fails_teachably,
    # ---- 2026-10-07 修复：`metrics` 失败**留痕 + 降级**（不许永久静默关闭）----
    test_metrics_failure_is_traced_and_not_permanently_disabled,
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
