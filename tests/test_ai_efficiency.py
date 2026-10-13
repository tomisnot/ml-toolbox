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
import time
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


def _payload_ds(ml, rows=16, L=6, K=4, d=5, with_target=False):
    """一份 NQS 形状的数据：逐行负载 + `target_kind=custom`。

    ``with_target=True`` 再放一份**真目标** `tgt`：凡是要验"损失/梯度真在动"的判据
    都必须用它 —— 用"自为目标"（`return o, o.detach()`）会让 loss 恒等于 0，
    于是"反传验证"看起来过了其实什么都没验（实测踩过）。
    """
    rng = np.random.RandomState(0)
    meta = pd.DataFrame({"m": rng.randn(rows)})
    xr = rng.randn(rows, L, d)
    bits = rng.randint(0, 2, size=(rows, K, L)).astype(np.int8)
    mask = np.ones((rows, K), dtype=bool)
    amps = rng.randn(rows, K) + 1j * rng.randn(rows, K)
    arrays = {"xr": xr, "bits": bits, "bit_mask": mask, "amps": amps}
    if with_target:
        arrays["tgt"] = rng.randn(rows, 2)
    got = ml.tools.execute("prepare_dataset", {
        "source": {"kind": "frame", "frame": meta}, "name": "eff",
        "target_kind": "custom",
        "payload": {"arrays": arrays},
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


def test_receipts_echo_resolved_absolute_paths_for_code_and_payload():
    """⭐⭐ A7③：回执必须回**解析后的绝对路径**（相对路径按 `dataset_roots[0]`，不是 cwd）。

    使用者实测的代价：他用 `../../data/...` 被解析到**错误位置**，"等了 120 分钟才发现"。
    我核实到的真相是**一半已修**：**失败路径**本来就打绝对路径
    （`dataset_not_found` 的 message 里就是解析后的位置）；缺的是**成功路径** ——
    相对路径**解析到另一个也存在的位置**时**零留痕**（回执只回原样入参）。
    ⇒ 本条钉：代码文件与 HDF5 负载都要回**解析后的绝对路径**。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="eff-resolve-") as td:
        ml = _ml(td, roots=[td])           # 数据根 = td ⇒ 相对路径按它解析
        try:
            # ① **代码文件**：给相对路径，回执要能说出它被解析到哪
            (Path(td) / "rel_net.py").write_text(GOOD, encoding="utf-8")
            ds = _payload_ds(ml)
            out = ml.tools.execute("run_custom", {
                "code_path": "rel_net.py", "dataset_id": ds, "name": "rel",
                "overrides": {"epochs": 2, "batch": 4}, "device": _DEV})["value"]
            assert out["ok"] is True, out
            assert out["code_path"] == "rel_net.py", out      # 原样入参照旧（可对齐输入）
            res = str(out.get("code_path_resolved") or "")
            assert res and os.path.isabs(res), (
                f"没有回解析后的绝对路径 ⇒ 读错位置时看不出来：{out.get('code_path_resolved')}")
            assert Path(res).resolve() == (Path(td) / "rel_net.py").resolve(), res

            # ② **HDF5 负载**：同样要回解析后的绝对路径（试跑一次即可，不必真训）
            pytest_mod = __import__("pytest")
            h5py = pytest_mod.importorskip("h5py")
            h5 = Path(td) / "rel_payload.h5"
            with h5py.File(h5, "w") as f:
                f.create_dataset("/amp", data=np.arange(24, dtype="float32").reshape(6, 4))
            meta = pd.DataFrame({"m": np.arange(6.0)})
            got = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame", "frame": meta}, "name": "h5p",
                "target_kind": "custom",
                "payload": {"hdf5": {"path": "rel_payload.h5",
                                     "datasets": {"amp": "/amp"}}},
                "pipeline": {"steps": [{"key": "scale"}]}})
            assert got["is_error"] is False, got
            ds2 = got["value"]["dataset_id"]
            dry = ml.tools.execute("run_custom", {
                "code": GOOD, "dataset_id": ds2, "dry_run": True,
                "device": _DEV})["value"]
            pr = dict(dry.get("payload_resolved") or {})
            # 无产物/无形状的边界：试跑回执里必须有解析后的 HDF5 绝对路径
            assert pr.get("hdf5_path"), f"负载路径没回解析结果：{pr}"
            assert Path(pr["hdf5_path"]).resolve() == h5.resolve(), pr
            assert pr.get("datasets") == {"amp": "/amp"}, pr

            # ③ ⭐ **Q3 的关键现场**：相对路径**解析到"另一个也真实存在"的文件**时，
            #    回执必须能看出**读的是哪一个**（这正是"等了 120 分钟"那类事故）。
            #    造两个同名文件：`td/rel_net.py`（要读的）与 `td/sub/rel_net.py`（诱饵，
            #    行数不同）⇒ 给 `sub/../rel_net.py`，断言解析结果与实际读到的**都指向**前者。
            (Path(td) / "sub").mkdir(exist_ok=True)
            decoy = Path(td) / "sub" / "rel_net.py"
            decoy.write_text("# 诱饵：这个文件**不该**被读\n" * 50, encoding="utf-8")
            tricky = ml.tools.execute("run_custom", {
                "code_path": "sub/../rel_net.py", "dataset_id": ds, "name": "tricky",
                "overrides": {"epochs": 2, "batch": 4}, "device": _DEV})["value"]
            assert tricky["ok"] is True, tricky
            assert tricky["code_path"] == "sub/../rel_net.py", tricky   # 原样入参
            assert Path(str(tricky["code_path_resolved"])).resolve() == \
                (Path(td) / "rel_net.py").resolve(), tricky["code_path_resolved"]
            # "读的是哪一个"还得有**内容**证据：行数必须是真文件那份（不是诱饵）
            assert int(tricky["code_lines"]) == len(GOOD.splitlines()), (
                f"实际读到的不是解析出来的那份（行数 {tricky['code_lines']} "
                f"vs 真文件 {len(GOOD.splitlines())}）")
            # 另一条常被误解的规则：相对路径按 **dataset_roots[0]** 解析，**不是按 cwd**
            assert Path(ml.engine._dataset_roots[0]).resolve() == Path(td).resolve()
        finally:
            ml.close()


def test_run_id_collision_is_detected_and_disentangled():
    """⭐ 自动生成的 `run_id` 撞车要**被检测并让开**（Q5-①）。

    残留风险（不是理论）：`run_id = f"{method}-{ms % 10_000_000}-{seed}"`
    ⇒ **2.78 小时一个周期**，同方法 + 同种子 + 同毫秒残差 ⇒ 两个互不相干的训练
    落进**同一个** `runs/<id>/` ⇒ TB 事件**追加**、曲线混在一根轴上
    （`pitfalls.md` A6 记的就是这一类现象）。概率极小，但**撞了没有任何提示**。

    验证空目录不算占用，已有内容使自动 id 让开，连续碰撞继续让开。
    显式 id 的保真由 runner 的实际运行回归覆盖，不检查实现的分支拼写。
    """
    from ml_toolbox.core.runner import reserve_run_id
    with tempfile.TemporaryDirectory(prefix='run-id-collision-') as td:
        root = Path(td)
        cand = 'candidate'
        occupied = root / cand
        occupied.mkdir()
        assert reserve_run_id(cand, runs_dir=root) == (cand, '')
        (occupied / 'record.json').write_text('{}', encoding='utf-8')
        used, collided = reserve_run_id(cand, runs_dir=root)
        assert used != cand and collided == cand
        (root / used).mkdir()
        (root / used / 'record.json').write_text('{}', encoding='utf-8')
        used2, _ = reserve_run_id(cand, runs_dir=root)
        assert used2 not in (cand, used)


def test_prepare_dataset_explains_the_target_kind_decision():
    """⭐ `prepare_dataset` 要**说出目标类型的判定依据**（Q5-②）。

    踩过的现场：一份 24 行的**连续**目标，因为"数值且基数 ≤ 20"被判成
    **classification**，而报出来的错是下游很远处的一句
    `RuntimeError: Found dtype Long but expected Float`（`MSELoss` 拿到 Long 标签）
    —— 它完全不指向"你的目标被当成分类了"。
    ⇒ 判据：`target_kind_reason` 必须同时给出**基数**、**阈值**与**结论**。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="eff-kind-") as td:
        ml = _ml(td)
        try:
            # 少基数（会被判分类）+ 多基数（会被判回归）各一份。
            # ⚠ 分类那份的类别要**有重复成员**：默认管道是**分层**切分，
            #   每类只有 1 个成员时会抛 sklearn 的 "least populated class" ——
            #   那本身就是另一条判据（见下面的 `stratify_on_low_cardinality_target`）。
            small = pd.DataFrame({"a": np.arange(18.0),
                                  "y": np.tile([0.0, 1.0, 2.0], 6)})
            many = pd.DataFrame({"a": np.arange(120.0), "y": np.arange(120.0) * 0.5})
            for frame, want in ((small, "classification"), (many, "regression")):
                got = ml.tools.execute("prepare_dataset", {
                    "source": {"kind": "frame", "frame": frame}, "name": "k",
                    "target": "y",
                    "pipeline": {"steps": [{"key": "scale"}]}})["value"]
                assert got["target_kind"] == want, got["target_kind"]
                reason = got.get("target_kind_reason") or ""
                assert "基数" in reason, reason
                assert "20" in reason, reason                     # 阈值要写出来
                assert want in reason, reason
                if want == "classification":
                    assert "连续" in reason or "custom" in reason, reason
            # ⭐ **离根因很远的那类错也要可教学**：每类只剩 1 个成员时
            #   sklearn 说的是"分层抽样"，而真相是"你的连续目标被判成分类了"
            #   ⇒ 必须翻成一句指向根因的话（Q5-②）。
            broken = ml.tools.execute("prepare_dataset", {
                "source": {"kind": "frame",
                           "frame": pd.DataFrame({"a": np.arange(18.0),
                                                  "y": np.arange(18.0) * 0.5})},
                "name": "k2", "target": "y",
                "pipeline": {"steps": [{"key": "scale"}]}})
            assert broken.get("is_error") is True, broken
            info = (broken.get("error") or {}).get("info") or {}
            assert info.get("kind") == "stratify_on_low_cardinality_target", info
            #: hint 里要带上**判定依据**（基数/阈值）——那才是根因
            assert "基数" in (info.get("hint") or ""), info
            assert "classification" in (info.get("hint") or ""), info
            assert "stratify" in (info.get("suggest") or ""), info
        finally:
            ml.close()


#: Q7b 判据用的网络：**两个参数组、两组 lr 不同**（幅度组 / 相位组）。
#: ⇒ `configure` 里设的值必须能从观测面**读回来且两者不同**（否则这个能力等于没接线）。
_NET_TWO_LRS = '''
class Net(nn.Module):
    def __init__(self, d, hidden=16):
        super().__init__()
        self.amp = nn.Linear(d, hidden)         # "幅度组"
        self.ph = nn.Linear(hidden, 2)          # "相位组"

    def forward(self, x):
        return self.ph(torch.tanh(self.amp(x)))


class Network:
    primary_metric = "lr/probe"

    def build(self, n_in, n_out, *, nn, ctx=None):
        return Net(int(ctx["payload_shapes"]["xr"][-1]))

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def configure(self, model, *, torch, lr, weight_decay):
        #: ⭐ **不对称步长**（文献里的 m = δτ_φ/δτ_A ≥ 1）：两组 lr 差 5 倍。
        #:   ⚠ 组名靠 `param_groups[i]["name"]` —— PyTorch 允许这个额外键。
        return torch.optim.Adam([
            {"params": model.amp.parameters(), "lr": lr, "name": "amp"},
            {"params": model.ph.parameters(), "lr": lr * 5.0, "name": "phase"},
        ])

    def on_step(self, model, step, ctx):
        #: `on_step` 里也应当能读到每组的 lr（Q7b：分到组，而不是猜）
        g = dict(ctx.get("lr_by_group") or {})
        return {"lr/probe": float(g.get("phase") or 0.0)}

    def forward_step(self, model, batch, ctx):
        #: ⚠ 负载是逐行的：`xr` 是 (R, L, d) ⇒ 按 L 取均值才是 (R, d)
        return model(ctx["payload"]["xr"].mean(1)), ctx["payload"]["tgt"]
'''


def test_per_group_learning_rates_reach_the_observation_surface():
    """⭐⭐ **每个参数组的 lr** 必须能从观测面读回来（Q7b）。

    使用者的实测需求：NQS 修复要用**不对称步长**（幅度组 / 相位组不同 lr，
    文献 `m = δτ_φ/δτ_A ≥ 1`），而观测面原来**只有一个标量 `lr`**
    ⇒ "**改了个东西却无法确认它真的生效**"（他这一路最大的坑之一）。

    判据（两条缺一不可）：
    ① `describe_training` 的 **`lr_by_group`** 里两组都在，且值与 `configure` 给的一致；
    ② ⭐ **两组必须不同**（否则"能读到"这件事是空转的 —— 与 Q4b 用
       "`best.pt` ≠ `last.pt`"防空转同一个思路）；
    ③ `on_step` 的 `ctx["lr_by_group"]` 也读得到（可以把组 lr 当 tag 上报）。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="eff-lr-") as td:
        ml = _ml(td)
        try:
            ds = _payload_ds(ml, rows=16, with_target=True)
            out = ml.tools.execute("run_custom", {
                "device": _DEV, "code": _NET_TWO_LRS, "dataset_id": ds,
                "name": "twolr", "persist": True,
                "overrides": {"epochs": 4, "batch": 8, "lr": 1e-3,
                              "val_split": 0.25}})["value"]
            assert out.get("ok") is True, out
            rid = out["run_id"]
            tr = ml.tools.execute("describe_training", {"run_id": rid})["value"]
            assert tr["available"] is True, tr.get("reason")
            got = dict(tr.get("lr_by_group") or {})
            assert "amp" in got and "phase" in got, (
                f"观测面读不到每个参数组的 lr（只有标量 `lr`？）：{got}")
            amp, ph = float(got["amp"]["lr"]), float(got["phase"]["lr"])
            # ① 与 `configure` 给的一致（lr=1e-3；phase = lr×5）
            assert abs(amp - 1e-3) < 1e-9, (amp, got)
            assert abs(ph - 5e-3) < 1e-9, (ph, got)
            # ② ⭐ 两组**必须不同**（否则这条判据是空转的）
            assert amp != ph, f"两组 lr 相同 ⇒ 不对称步长没有被证明：{got}"
            # ③ `on_step` 也能分到组（它上报的 probe tag 就等于 phase 组的 lr）
            probe = (tr.get("latest") or {}).get("other", {}).get("lr/probe")
            assert probe is not None and abs(float(probe) - ph) < 1e-9, (
                f"`on_step` 的 ctx 里读不到组 lr：latest.other={tr.get('latest')}")
            # ④ artifacts 里也留一份（`describe_run` / 存档能看）
            from ml_toolbox.core.persistence import load_run_record
            arts = load_run_record(rid).result.artifacts
            assert "nn_lr_by_group" in arts, sorted(arts)
            assert "phase" in str(np.asarray(arts["nn_lr_by_group"]).ravel()[0]), \
                arts["nn_lr_by_group"]
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


#: 用**真目标**的网络（损失与梯度真在动）——验"反传"类判据必须用它
TGT_NET = '''
class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Linear(d, 8)
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(torch.tanh(self.enc(x)).mean(1))


class Network:
    def build(self, n_in, n_out, *, nn, ctx=None):
        return Net(int(ctx["payload_shapes"]["xr"][-1]))

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def forward_step(self, model, batch, ctx):
        x, y = batch
        return model(ctx["payload"]["xr"]), ctx["payload"]["tgt"]
'''

#: 12 个 Linear ⇒ `grad_by_layer` 超过摘要阈值（证明"默认摘要"真的省了东西）
MANY_LAYER_NET = '''
class Net(nn.Module):
    def __init__(self, d, n=12):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d if i == 0 else 6, 6)
                                     for i in range(n)])
        self.out = nn.Linear(6, 2)

    def forward(self, x):
        h = x.mean(1)
        for lay in self.layers:
            h = torch.relu(lay(h))
        return self.out(h)


class Network:
    def build(self, n_in, n_out, *, nn, ctx=None):
        return Net(int(ctx["payload_shapes"]["xr"][-1]))

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def forward_step(self, model, batch, ctx):
        x, y = batch
        return model(ctx["payload"]["xr"]), ctx["payload"]["tgt"]
'''


#: `curve` 详细度判据用的网络：**一个在爬的 tag + 一群平的 tag**。
#: - `fid/climb` 逐轮上升 ⇒ `convergence` 必须报 `underfit`，
#:   且 `curve="brief"` 时**它的形状必须照给**（否则"说它欠训练却看不到曲线"）；
#: - `fid/flat`/`nll/a`… 平 ⇒ 默认被省掉（省的就是它们）。
CURVE_TAGS_NET = '''
class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Linear(d, 8)
        self.h = nn.Linear(8, 2)

    def forward(self, x):
        return self.h(torch.tanh(self.f(x.mean(1))))


class Network:
    def __init__(self):
        self.ep = 0

    def build(self, n_in, n_out, *, nn, ctx=None):
        return Net(int(ctx["payload_shapes"]["xr"][-1]))

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def forward_step(self, model, batch, ctx):
        x, y = batch
        return model(ctx["payload"]["xr"]), ctx["payload"]["tgt"]

    def metrics(self, model, ctx):
        self.ep = int(ctx.get("step") or 0)
        # ⚠ `fid/*` 在 `curve.direction` 里是"越大越好" ⇒ 上升 = underfit（正是要的）。
        # 平的那些给足够多条：真实 run 是几十个 tag ⇒ 形状总量才看得出来（实测 58 个）。
        out = {"fid/climb": 0.1 * (self.ep + 1), "fid/flat": 0.5}
        for i in range(11):
            out["aux/c%d" % i] = 0.1 * (i + 1)
        return out
'''


def test_read_job_defaults_to_a_summary_not_the_whole_history():
    """⭐ `read_job` **默认只回摘要**（缺口 A）。

    他因为 `progress` 太长**放弃了官方轮询姿势**（原话"后面我不再直接轮询它了"）
    —— AI 主动放弃推荐姿势，是这条缺口最硬的证据。
    要钉三件：默认有界、**真实条数看得见**、要全量能要到。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="eff-job-") as td:
        ml = _ml(td)
        try:
            ds = _payload_ds(ml, rows=16, with_target=True)
            job = ml.tools.execute("submit_run", {
                "code": TGT_NET, "dataset_id": ds, "name": "obs", "device": _DEV,
                "overrides": {"epochs": 150, "batch": 4}})
            assert job["is_error"] is False, job
            jid = job["value"]["job_id"]
            # ⚠ **等它跑完再断言**：不依赖"正好采样在中途" —— 实测这套东西的耗时
            #   几乎全在启动（torch 导入/建模型 ~9s，而 60 epoch 不到 0.2s 就跑完），
            #   所以"中途采样"是**时序赌博**。要钉的是**回执的默认口径**，那与何时读无关。
            st = {}
            for _ in range(30):
                st = ml.tools.execute("read_job", {
                    "job_id": jid, "wait_s": 2})["value"]
                if st["state"] in ("done", "failed", "cancelled") and st["n_progress"]:
                    break
            assert st["n_progress"] > 8, f"进度点不足（{st['n_progress']}）：{st}"
            # ① 默认有界，且**真实条数/被截断**都说得明明白白
            assert len(st["progress"]) <= 8, len(st["progress"])
            assert st["truncated"] is True, st
            assert st["detail"] == "summary" and st["tail"] == 8, st
            latest = st["progress_latest"]
            assert latest.get("stage"), latest
            assert latest.get("loss") is not None, latest   # 够判断"该不该早停"
            # ② 要全量能要到
            full = ml.tools.execute("read_job", {
                "job_id": jid, "detail": "full"})["value"]
            assert len(full["progress"]) == full["n_progress"], full
            assert full["truncated"] is False, full
            # ③ tail 可自定
            t3 = ml.tools.execute("read_job", {"job_id": jid, "tail": 3})["value"]
            assert len(t3["progress"]) == min(3, t3["n_progress"]), t3
        finally:
            ml.close()


def test_unknown_job_gives_on_disk_clues():
    """⭐ 会话里没有的 job ⇒ 给**盘上的线索**（缺口 D）。

    文档承诺"跑完自动落盘 ⇒ 丢了 job_id 结果还在"，但重启后 `read_job` 只说"不存在"
    ⇒ 用户会以为**结果丢了**。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="eff-unk-") as td:
        ml = _ml(td)
        try:
            ds = _payload_ds(ml, rows=12)
            rid = ml.tools.execute("run_custom", {
                "code": GOOD, "dataset_id": ds, "name": "clue",
                "overrides": {"epochs": 2, "batch": 4}, "device": _DEV}
            )["value"]["run_id"]
            got = ml.tools.execute("read_job", {"job_id": "job-nope"})
            assert got["is_error"] is True, got
            info = (got.get("error") or {}).get("info") or {}
            assert info.get("kind") == "unknown_job", info
            hint = info.get("hint") or ""
            assert "只活在会话里" in hint and "describe_run" in hint, hint
            # ⭐ 线索要**真的指到盘上的 run**（不是一句"你去查吧"）
            assert rid in hint, f"线索里没有刚跑完的那个 run：{hint}"
            assert "describe_run" in (info.get("suggest") or ""), info
        finally:
            ml.close()


def test_dry_run_warmup_steps_catches_dynamics_errors():
    """⭐ `warmup_steps`：**真反传**几十步，抓 `dry_run` 的盲区（缺口 C）。

    他这轮三次长训练全部分散/发散（step 120 处 loss 0.009 → 113；`nll → 0` 的退化解），
    而只验前向的 `dry_run` **一个都抓不到**。边界不变：不落盘、不计入记录。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="eff-warm-") as td:
        ml = _ml(td)
        try:
            ds = _payload_ds(ml, rows=16, with_target=True)
            base = {"dataset_id": ds, "dry_run": True, "device": _DEV}
            # ① 正常网络：真反传过、参数真的动了、loss 在降
            ok = ml.tools.execute("run_custom", {
                **base, "code": TGT_NET, "warmup_steps": 12})["value"]
            assert ok["warmup_steps"] == 12, ok
            assert ok["params_changed"] is True, ok
            assert len(ok["loss_curve"]) == 12 and len(ok["grad_norm_curve"]) == 12, ok
            assert ok["verdict"].startswith("✅"), ok["verdict"]
            # ② **会发散**的配置（学习率过大）：判读要**直说**，且**不抛错**
            boom = ml.tools.execute("run_custom", {
                **base, "code": TGT_NET, "warmup_steps": 15,
                "overrides": {"lr": 1e3}})["value"]
            assert boom["warmup_steps"] >= 1, boom
            v = boom.get("verdict") or ""
            assert v.startswith(("⛔", "⚠")), (
                f"学习率 1e3 会发散，判读却没说：{v!r}（曲线 "
                f"{[round(x, 4) for x in (boom.get('loss_curve') or [])[:5]]}）")
            # ③ 不落盘/不入账
            runs_dir = Path(td) / "runs"
            before = sorted(p.name for p in runs_dir.iterdir()) if runs_dir.is_dir() else []
            n_hist = len(ml.engine.session.history())
            again = ml.tools.execute("run_custom", {
                **base, "code": TGT_NET, "warmup_steps": 5})["value"]
            after = sorted(p.name for p in runs_dir.iterdir()) if runs_dir.is_dir() else []
            assert after == before, "warmup 试跑往 runs/ 里写了东西"
            assert len(ml.engine.session.history()) == n_hist, "warmup 试跑进了会话历史"
            assert again["run_id"] == "", again
        finally:
            ml.close()


def test_describe_training_summarizes_grad_by_layer():
    """⭐ 逐层梯度**默认摘要**（缺口 B）：`pick` 删不掉**必填键内部**的大块。

    他实测：只想看 4 个指标，而 `latest`（必填）里带着 30 个 `grad_by_layer`
    ⇒ "只看 4 个数"要付 30 个梯度的钱。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="eff-grad-") as td:
        ml = _ml(td)
        try:
            ds = _payload_ds(ml, rows=16, with_target=True)
            rid = ml.tools.execute("run_custom", {
                "code": MANY_LAYER_NET, "dataset_id": ds, "name": "many",
                "overrides": {"epochs": 3, "batch": 4}, "device": _DEV}
            )["value"]["run_id"]
            latest = ml.tools.execute(
                "describe_training", {"run_id": rid})["value"]["latest"]
            n_all = int(latest.get("grad_by_layer_n") or 0)
            assert n_all >= 10, (
                f"层数不够多（{n_all}）—— 这条判据要证明摘要，需要 >8 层")
            assert len(latest["grad_by_layer"]) <= 6, latest["grad_by_layer"]
            assert "grad_layers" in (latest.get("grad_by_layer_note") or ""), latest
            # 要全量能要到
            full = ml.tools.execute("describe_training", {
                "run_id": rid, "grad_layers": "full"})["value"]
            assert len(full["latest"]["grad_by_layer"]) == n_all, full["latest"]
        finally:
            ml.close()


def test_describe_training_curve_default_is_bounded_and_keeps_warnings():
    """⭐⭐ `curve` **默认只给该看的形状**，但**绝不漏判读**（缺口 A33/缺口B 的复发）。

    实测（真 run，58 个 tag）：全量形状占默认回执 **87%**（38 891 字符里 31 140 是
    `curve`），而 `curve` 在**必填键**里 ⇒ `pick` 最多省 9%。这一条钉三件事：

    1. 默认（`brief`）明显更小（形状只留核心 tag + **有判读的 tag**）；
    2. ⭐ **有判读的 tag 必须在 brief 里也给出形状** —— 否则会出现"回执说某 tag
       欠训练、却看不到它的形状"这种自相矛盾（比浪费 token 更坏）；
    3. `curve="all"` 能拿回全量、`curve_tags=[…]` 能点名、`off` 仍保留判读。

    ⚠ 判读（`convergence`）**始终基于全部 tag 算**：省形状**不会**漏掉
    "末段还在爬 ⇒ 可能欠训练"这类警告，这正是本判据第 ② 条要钉的。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    import json

    def _n(o):
        return len(json.dumps(o, ensure_ascii=True, default=str))

    with tempfile.TemporaryDirectory(prefix="eff-curve-") as td:
        ml = _ml(td)
        try:
            ds = _payload_ds(ml, rows=16, with_target=True)
            rid = ml.tools.execute("run_custom", {
                "code": CURVE_TAGS_NET, "dataset_id": ds, "name": "curve",
                "overrides": {"epochs": 8, "batch": 4, "val_split": 0.25},
                "device": _DEV})["value"]["run_id"]

            brief = ml.tools.execute("describe_training", {"run_id": rid})["value"]
            full = ml.tools.execute("describe_training", {
                "run_id": rid, "curve": "all"})["value"]
            off = ml.tools.execute("describe_training", {
                "run_id": rid, "curve": "off"})["value"]

            # ① 全量一定有足够多的 tag（否则下面的大小比较没有意义）
            assert len(full["curve"]) >= 12, sorted(full["curve"])
            assert full["curve_mode"] == "all", full["curve_mode"]
            # ② 默认明显更小（形状这一项是机制；整份回执是它的效果）
            assert len(brief["curve"]) < len(full["curve"]), (
                len(brief["curve"]), len(full["curve"]))
            assert _n(brief["curve"]) < 0.35 * _n(full["curve"]), (
                _n(brief["curve"]), _n(full["curve"]))
            assert _n(brief) < 0.7 * _n(full), (_n(brief), _n(full))
            # 省下来的**绝对量**才是 AI 的成本（固定开销几千字符谁也去不掉）
            assert _n(full) - _n(brief) >= 3000, (_n(brief), _n(full))
            assert brief["curve_omitted_n"] == len(full["curve"]) - len(brief["curve"])
            assert "curve" in (brief.get("curve_note") or ""), brief.get("curve_note")
            # 核心 tag 在默认里（否则"看不到自己的损失曲线"）
            for core in ("loss", "val_loss"):
                assert core in brief["curve"], sorted(brief["curve"])

            # ③ ⭐ **有判读的 tag：brief 也必须给形状**（不许自相矛盾）
            kinds = {n["tag"]: n["kind"] for n in full["convergence"]}
            assert kinds.get("fid/climb") == "underfit", kinds
            assert "fid/climb" in brief["curve"], (
                "brief 省掉了**有判读**的 tag 的形状 ⇒ 回执会「说它欠训练、"
                "却看不到它的曲线」")
            # ④ 平的那些 tag 在默认里被省掉（省的就是它们）
            assert "fid/flat" in full["curve"] and "fid/flat" not in brief["curve"], (
                sorted(brief["curve"]), sorted(full["curve"]))

            # ⑤ `curve_tags` 点名：只要它（+ 有判读的照给）
            picked = ml.tools.execute("describe_training", {
                "run_id": rid, "curve_tags": ["fid/flat"]})["value"]
            assert "fid/flat" in picked["curve"], sorted(picked["curve"])
            assert "fid/climb" in picked["curve"], sorted(picked["curve"])
            missing = ml.tools.execute("describe_training", {
                "run_id": rid, "curve_tags": ["no/such"]})["value"]
            assert missing["curve_tags_missing"] == ["no/such"], missing.get(
                "curve_tags_missing")

            # ⑥ `off`：一个形状都不回，但**判读仍在**（信息不丢）
            assert off["curve"] == {}, off["curve"]
            assert {n["tag"]: n["kind"] for n in off["convergence"]} == kinds, (
                off["convergence"])

            # ⑦ 非法模式：**可教学**地拒（不静默退化成默认）
            bad = ml.tools.execute("describe_training", {
                "run_id": rid, "curve": "weird"})
            assert bad["is_error"] is True, bad
            assert ((bad.get("error") or {}).get("info") or {}).get("kind") \
                == "bad_curve_mode", bad
        finally:
            ml.close()


def test_cuda_device_assert_is_told_apart_from_its_echo():
    """⭐ **CUDA 断言之后每一次报错都是回声** —— 必须说清，否则会去修一个不存在的问题。

    实测代价一小时（`nqs_docs/多N合并-进展与阻塞.md`）：使用者把 `custom.py:238` 的
    `rec.set_probe(...)` 当成现场，写了一份"探针路径与多组数组负载的交互"分析；
    实测复核那一行只是**第一次与显卡同步**的位置（`torch.tensor(..., device=dev)`），
    真正的断言在它之前，而上下文被毒化后**每次**都在这里报同一个错。

    本判据走**真实入口**（`_summarize_record`），不只测那个纯函数。
    """
    from types import SimpleNamespace

    from ml_mecha.engine import _CUDA_POISON, _summarize_record

    def _fake_record(run_id, err):
        res = SimpleNamespace(ok=False, error=err, metrics={}, primary_metric="",
                              error_kind="", error_hint="", error_suggest="",
                              elapsed=0.0)
        return SimpleNamespace(run_id=run_id, method="custom_network",
                               dataset_id="ds-1", pipeline_id="p-1", result=res)

    _CUDA_POISON.clear()
    assert_text = ("RuntimeError: CUDA error: device-side assert triggered\n"
                   "CUDA kernel errors might be asynchronously reported …")
    first = _summarize_record(_fake_record("run-A", assert_text), "ds-1",
                              SimpleNamespace())
    assert first.error_kind == "cuda_device_assert", first.error_kind
    assert "CUDA_LAUNCH_BLOCKING" in first.error_hint, first.error_hint
    assert "异步" in first.error_hint, first.error_hint

    echo = _summarize_record(_fake_record("run-B", assert_text), "ds-1",
                             SimpleNamespace())
    assert echo.error_kind == "cuda_context_poisoned", echo.error_kind
    assert "回声" in echo.error_hint and "run-A" in echo.error_hint, echo.error_hint
    assert "重启" in echo.error_hint, echo.error_hint
    assert "别在这里改代码" in echo.error_hint, echo.error_hint

    # ⚠ **无关失败不受影响**（别把普通错误也说成 CUDA 回声）
    plain = _summarize_record(_fake_record("run-C", "ValueError: 参数写错了"),
                              "ds-1", SimpleNamespace())
    assert plain.error_kind == "" and plain.error_hint == "", (
        plain.error_kind, plain.error_hint)
    _CUDA_POISON.clear()


def test_submitted_job_reports_command_level_failure():
    """⭐ **job 完成 ≠ 命令成功**：命令层失败必须说出来（否则就是"静默死亡"）。

    使用者实测：`submit_run` → `ok:true, job_id`；随后 `read_job` 显示 `state=done`、
    `run_id=""`、`progress=[]`、`error=""`，而 `runs/` 下**什么都没有**、日志也无新条目
    —— 同一份配置重启前成功、重启后重投**静默死亡**。
    实测根因：**命令自己失败**了（`unknown_dataset` / `dataset_path_denied`），
    错误留在**回执**里，而 job 层只记它**自己**的异常 ⇒ `read_job` 只读 job 级 error
    就只看到"成功"。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    with tempfile.TemporaryDirectory(prefix="eff-silent-") as td:
        ml = _ml(td)
        try:
            ds = _payload_ds(ml, rows=12, with_target=True)
            # ① 过期的 dataset_id（重启后的典型情形）
            job = ml.tools.execute("submit_run", {
                "code": GOOD, "dataset_id": "ds-001-已失效", "name": "stale",
                "device": _DEV, "overrides": {"epochs": 2}})
            assert job["is_error"] is False, job
            jid = job["value"]["job_id"]
            got = None
            for _ in range(15):
                time.sleep(1.5)
                got = ml.tools.execute("read_job", {"job_id": jid, "wait_s": 2})
                if got.get("is_error"):
                    break
            assert got is not None and got.get("is_error") is True, (
                f"命令失败了，read_job 却回了个'成功'：{got}")
            info = (got.get("error") or {}).get("info") or {}
            assert info.get("kind") == "unknown_dataset", info
            assert "没有产出可用的运行" in got["error"]["message"], got["error"]
            # ② 越界的 code_path（另一条"命令失败"路径）
            outside = Path(tempfile.mkdtemp(prefix="eff-out-")) / "net.py"
            outside.write_text(GOOD, encoding="utf-8")
            job2 = ml.tools.execute("submit_run", {
                "code_path": str(outside), "dataset_id": ds, "name": "outside",
                "device": _DEV, "overrides": {"epochs": 2}})
            assert job2["is_error"] is False, job2
            jid2 = job2["value"]["job_id"]
            got2 = None
            for _ in range(15):
                time.sleep(1.5)
                got2 = ml.tools.execute("read_job", {"job_id": jid2, "wait_s": 2})
                if got2.get("is_error"):
                    break
            assert got2 is not None and got2.get("is_error") is True, got2
            # ⚠ 两种 fail-closed 都算对：**没声明任何数据根**时拒
            #   `dataset_roots_required`；声明了根但路径在根外时拒
            #   `dataset_path_denied`。本判据用的是"没声明根"的装配 ⇒ 认第一种
            #   （两种都表示"拒绝读越界文件"）。
            kind2 = (got2["error"].get("info") or {}).get("kind")
            assert kind2 in ("dataset_roots_required", "dataset_path_denied"), \
                got2["error"]
        finally:
            ml.close()


def test_inline_payload_declaration_is_json_safe():
    """⭐ 负载**声明必须 JSON 安全**：内联 `arrays` 的真值不能进 extras。

    实测（我自己的 bug）：声明里留着 ndarray ⇒ **一次成功的运行**在 job 路径上被判失败
    （`TypeError: Object of type ndarray is not JSON serializable`），而同步 `run_custom`
    完全看不出来（同步路径不序列化 extras）⇒ 只有 `submit_run` 会暴露。
    """
    if not _has_torch():
        print("    （跳过：未装 torch）")
        return
    import json
    with tempfile.TemporaryDirectory(prefix="eff-jdecl-") as td:
        ml = _ml(td)
        try:
            ds = _payload_ds(ml, rows=12, with_target=True)
            # ① 声明本身可 JSON 序列化，且内联数组只留形状/dtype
            entry = ml.engine.dataset_entry(ds)
            decl = dict((entry.spec.meta or {}).get("payload_decl") or {})
            assert decl, "登记时没留下负载声明"
            json.dumps(decl)                     # 不可序列化就会抛
            assert "inline" in decl, decl
            for k, v in decl["inline"].items():
                assert set(v) == {"shape", "dtype"}, (k, v)
            # ② 走**后台 job** 真跑一次：必须产出 run_id（而不是被判失败）
            job = ml.tools.execute("submit_run", {
                "code": TGT_NET, "dataset_id": ds, "name": "jdecl",
                "device": _DEV, "overrides": {"epochs": 2, "batch": 4}})
            assert job["is_error"] is False, job
            jid = job["value"]["job_id"]
            st = {}
            for _ in range(20):
                time.sleep(2)
                got = ml.tools.execute("read_job", {"job_id": jid, "wait_s": 3})
                assert got.get("is_error") is False, got      # 不许静默/不许误判失败
                st = got["value"]
                if st["state"] in ("done", "failed", "cancelled") and st["run_id"]:
                    break
            assert st.get("run_id"), f"后台 job 没产出 run_id：{st}"
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
    # ---- A7③（2026-10-07）：相对路径要回**解析后的绝对路径** ----
    test_receipts_echo_resolved_absolute_paths_for_code_and_payload,
    # ---- Q5（2026-10-07）：run_id 撞车检测 + 目标类型判定依据 ----
    test_run_id_collision_is_detected_and_disentangled,
    test_prepare_dataset_explains_the_target_kind_decision,
    # ---- Q7b（2026-10-07）：每个参数组的 lr（不对称步长要能确认生效）----
    test_per_group_learning_rates_reach_the_observation_surface,
    test_dry_run_shapes_verbosity_is_smaller,
    test_eval_custom_says_why_the_payload_source_is_what_it_is,
    # ---- 第三轮反馈（docs/反馈-观测面默认全量与AI调用预算.md）----
    test_read_job_defaults_to_a_summary_not_the_whole_history,
    test_unknown_job_gives_on_disk_clues,
    test_dry_run_warmup_steps_catches_dynamics_errors,
    test_describe_training_summarizes_grad_by_layer,
    # ---- 2026-10-07：`curve` 默认有界（缺口B 在形状上复发；形状占默认回执 87%）----
    test_describe_training_curve_default_is_bounded_and_keeps_warnings,
    # ---- CUDA 断言与它的回声（多N合并-进展与阻塞.md 的实测代价）----
    test_cuda_device_assert_is_told_apart_from_its_echo,
    # ---- 静默死亡：命令层失败被 job 层吞掉 / 声明非 JSON 安全 ----
    test_submitted_job_reports_command_level_failure,
    test_inline_payload_declaration_is_json_safe,
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
