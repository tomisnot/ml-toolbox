# -*- coding: utf-8 -*-
"""量子启发方法族（教材《量子机器学习》第 3/7/9 章的态矢量级复现）。

不是量子计算机接口，也不是宣称量子加速——把书里的算法结构接到工具箱
的统一契约上，资源上限由 n_qubits 显式钉死（模拟器见 methods/qsim.py）：

  qsvc        量子核支持向量分类   §3.3.2 保真度核 + 反演测试/有限 shots
  qlssvc      最小二乘量子 SVM     §3.3.1 式(3.67) 块线性系统（Rebentrost 路线）
  qkrr        量子核岭回归         §3.3.2 同一核机构 + 对偶岭回归
  qkmeans     量子可区分度聚类     §7.1 保真度距离 d=1-|⟨x|c⟩|²（SWAP 测试语义）
  qpca        密度矩阵指数化主成分 §7.2 式(7.37)-(7.55)：ρ=C/TrC 的 DME 幂迭代
  qreservoir  量子储层时序预测     §9.1 式(9.22)-(9.24) 注入+固定演化 +
                                   式(9.13) 岭读出 + 式(9.32) 虚拟节点

族归属按建模任务分散：svm / linear / cluster / manifold / timeseries——
量子启发的"族"通过 tags 里的 "quantum" 检索，不单独成族。
"""
from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd

from ..core.contracts import (MLMethod, PageSpec, ParamSpec, RunConfig,
                              TASK_SUPERVISED, TASK_CLUSTER, TASK_MANIFOLD,
                              TASK_TIMESERIES)
from ..core.registry import register
from . import plots, qsim

# 检视页共用的核/距离热图预览上限（持久化体量防线：runs/ 只存 preview）
_PREVIEW = 120


def _shared_kernel_schema(with_C=True):
    """核方法公共旋钮（编码 / 线路 / 测量——书 §实践二 要求的三件套）。"""
    schema = [
        ParamSpec("n_qubits", "量子比特数", "int", -1, min=-1, max=7,
                  hint="-1=特征数≤5时全用，否则 PCA 到低维；越大线路越贵"),
        ParamSpec("gamma", "编码强度 γ", "number", 1.0, min=0.05, max=3.14,
                  hint="角度 θ=γ·x̃ 的尺度，决定核的分辨率"),
        ParamSpec("reps", "编码层数", "int", 1, min=1, max=3,
                  hint="数据重上传深度（式 3.6 段落）"),
        ParamSpec("shots", "测量次数 S", "int", 0, min=0, max=200000,
                  hint="0=理想核值；>0 按式(3.73)反演测试 Bernoulli 采样，"
                       "复现有限测量的统计涨落"),
    ]
    if with_C:
        schema.append(ParamSpec("C", "惩罚系数 C", "number", 1.0, min=0.01))
    return schema


class _QuantumKernelSupervised(MLMethod):
    """监督量子核方法公共壳：编码→核矩阵→核学习器→按 result 携带态预测。

    预测只用训练期冻结的编码器统计量（scaler/PCA 随 result.qenc 走，
    新实例可 predict——同 L19 标签编码跨实例约定）。
    """
    task = TASK_SUPERVISED

    def kernel_fit(self, K, yv, p, cfg):
        raise NotImplementedError

    def kernel_predict(self, Kt, result):
        raise NotImplementedError

    def kernel_scores(self, Kt, result):
        """decision_function 类输出（可选，喂 y_prob）。"""
        return None

    # ------------------------------------------------ 契约实现
    def fit(self, X: pd.DataFrame, y, cfg: RunConfig, diag: bool = False):
        p = self.params(cfg)
        Xv = _as_np(X)
        res = self._new_result(params=p)
        if Xv.shape[0] < 4:
            res.error = "量子核方法至少需要 4 个训练样本"
            return res
        nq = int(p["n_qubits"])
        if nq <= 0:
            nq = min(Xv.shape[1], 5)
        rng = np.random.RandomState(int(cfg.seed))
        enc = qsim.QuantumEncoder(n_qubits=nq, gamma=float(p["gamma"]),
                                  reps=int(p["reps"])).fit(Xv)
        S = enc.transform(Xv)
        shots = int(p["shots"])
        K = qsim.fidelity_kernel(S, shots=shots, rng=rng) if shots \
            else qsim.fidelity_kernel(S)

        kind = self.target_kind or "classification"
        if kind == "classification":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            yv = le.fit_transform(np.asarray(y))
        else:
            le, yv = None, np.asarray(y, float)

        res.target_kind = kind
        est, extra = self.kernel_fit(K, yv, p, cfg)
        res.est = est
        res.qenc, res.qstates, res.qextra = enc, S, extra
        res.classes_ = le.classes_ if le is not None else None
        res.artifacts["q_kernel"] = _preview(K)
        if diag:
            ev = np.linalg.eigvalsh(K)[::-1]
            tr = float(ev.sum())
            res.diag = {"n_qubits_eff": int(S.shape[1]).bit_length() - 1,
                        "kernel_trace": tr,
                        "kernel_effective_rank":
                            float(tr * tr / max(float((ev ** 2).sum()), 1e-30)),
                        "kernel_cond": float(ev[0] / max(ev[-1], 1e-12)),
                        "shots": shots}
        return res

    def predict(self, X: pd.DataFrame, result):
        Kt = self._test_kernel(X, result)
        return self.kernel_predict(Kt, result)

    def fit_extra_artifacts(self, X, result):
        try:
            sc = self.kernel_scores(self._test_kernel(X, result), result)
            if sc is None:
                return {}
            sc = np.asarray(sc, float)
            if sc.ndim == 1:                       # 二分类 decision_function：(m,)
                p = 1.0 / (1.0 + np.exp(-sc))
                return {"y_prob": np.column_stack([1 - p, p])}
            return {"y_prob": _softmax(sc)}
        except Exception:
            return {}

    def _test_kernel(self, X, result):
        Sn = result.qenc.transform(_as_np(X))
        return qsim.fidelity_kernel(Sn, result.qstates)  # 预测用理想重叠


def _as_np(X):
    return X.to_numpy(float) if hasattr(X, "to_numpy") else np.asarray(X, float)


def _softmax(d):
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _preview(M, size=_PREVIEW):
    """大矩阵逐轴等间隔抽稀预览（持久化只存这个，plot 消费它）。

    行/列各自独立抽稀，支持非方阵（核矩阵 m×m 与距离矩阵 m×k 共用）。
    """
    M = np.asarray(M, float)

    def ax(n):
        return np.arange(n) if n <= size else np.unique(
            np.linspace(0, n - 1, size).astype(int))
    return M[np.ix_(ax(M.shape[0]), ax(M.shape[1]))]


# ================================================================ 分类
@register
class QuantumKernelSVC(_QuantumKernelSupervised):
    """量子核 SVM 分类（Havlíček 式：量子特征映射态 + SVC 核学习器）。"""
    name = "qsvc"
    display_name = "量子核 SVM"
    family = "svm"
    target_kind = "classification"
    tags = ("quantum", "kernel", "small-sample")
    param_schema = _shared_kernel_schema(with_C=True)

    def kernel_fit(self, K, yv, p, cfg):
        from sklearn.svm import SVC
        est = SVC(kernel="precomputed", C=float(p["C"]))
        est.fit(K, yv)
        return est, {}

    def kernel_predict(self, Kt, result):
        pred = result.est.predict(Kt)
        le = _le_from(result)
        return le.inverse_transform(np.asarray(pred)) if le is not None else pred

    def kernel_scores(self, Kt, result):
        return result.est.decision_function(Kt)

    def inspect_pages(self, cfg):
        return _supervised_class_pages() + [
            PageSpec("qk", "量子核矩阵", "mpl", _plot_kernel,
                     hint="训练期冻结的保真度核 |⟨φ(x)|φ(z)⟩|² 预览")]


@register
class QuantumLSQSVM(_QuantumKernelSupervised):
    """最小二乘量子 SVM（书式 3.67 块线性系统；多类 OvR）。"""
    name = "qlssvc"
    display_name = "最小二乘量子 SVM"
    family = "svm"
    target_kind = "classification"
    tags = ("quantum", "kernel", "least-squares")
    param_schema = _shared_kernel_schema(with_C=False) + [
        ParamSpec("reg", "LS 正则 γ⁻¹", "number", 0.1, min=1e-4, max=10.0,
                  hint="式(3.67) 中的 γ⁻¹I；越大越平滑"),
        ParamSpec("max_support", "求解子样本", "int", 300, min=40, max=800,
                  hint="块系统求解 O(m³)，大数据集随机抽稀"),
    ]

    def kernel_fit(self, K, yv, p, cfg):
        m = K.shape[0]
        rng = np.random.RandomState(int(cfg.seed) + 7)
        cap = min(m, int(p["max_support"]))
        idx = rng.choice(m, cap, replace=False) if cap < m else np.arange(m)
        classes = np.unique(yv)
        models = []
        for c in classes[1:] if len(classes) == 2 else classes:
            yc = np.where(yv[idx] == c, 1.0, -1.0)
            b, a = _ls_svm_solve(K[np.ix_(idx, idx)], yc, float(p["reg"]))
            models.append((int(c), b, a))
        return {"classes": classes, "models": models}, {"sub": idx}

    def kernel_predict(self, Kt_all, result):
        est = result.est
        Kt = Kt_all[:, result.qextra["sub"]]
        if len(est["classes"]) == 2:
            c, b, a = est["models"][0]
            f = Kt @ a + b
            pred = np.where(f > 0, c, est["classes"][0])
        else:
            F = np.column_stack([Kt @ a + b for _, b, a in est["models"]])
            pred = np.asarray(est["classes"])[F.argmax(axis=1)]
        le = _le_from(result)
        return le.inverse_transform(pred) if le is not None else pred

    def kernel_scores(self, Kt_all, result):
        est = result.est
        Kt = Kt_all[:, result.qextra["sub"]]
        if len(est["classes"]) == 2:
            c, b, a = est["models"][0]
            f = Kt @ a + b
            return np.column_stack([-f, f])
        return np.column_stack([Kt @ a + b for _, b, a in est["models"]])

    def inspect_pages(self, cfg):
        return _supervised_class_pages() + [
            PageSpec("qk", "量子核矩阵", "mpl", _plot_kernel,
                     hint="训练期冻结的保真度核 |⟨φ(x)|φ(z)⟩|² 预览")]


def _ls_svm_solve(K, yc, reg_inv):
    """式(3.67)：[[0,1ᵀ],[1,K+γ⁻¹I]](b,α̃)ᵀ=(0,y)ᵀ。"""
    m = K.shape[0]
    A = np.zeros((m + 1, m + 1))
    A[0, 1:] = A[1:, 0] = 1.0
    A[1:, 1:] = K + reg_inv * np.eye(m)
    rhs = np.concatenate([[0.0], yc])
    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return float(sol[0]), sol[1:]


def _le_from(result):
    """从 result 携带的原始类别重建 LabelEncoder（同 base.SklearnSupervised）。"""
    classes = getattr(result, "classes_", None)
    if classes is None:
        return None
    import sklearn.preprocessing as spp
    le = spp.LabelEncoder()
    le.classes_ = np.asarray(classes)
    return le


def _supervised_class_pages():
    """分类声明页 = 与兜底页同一批 plots 函数（L9 纪律）。"""
    return [PageSpec("cm", "混淆矩阵", "mpl", plots.plot_confusion),
            PageSpec("roc", "ROC / PR", "mpl", plots.plot_roc,
                     hint="该方法不输出概率时无法画 ROC"),
            PageSpec("pr", "PR 曲线", "mpl", plots.plot_pr,
                     hint="需要概率输出")]


# ================================================================ 回归
@register
class QuantumKernelRidge(_QuantumKernelSupervised):
    """量子核岭回归：同一保真度核 + 对偶形式 KernelRidge（核方法归 linear 族）。"""
    name = "qkrr"
    display_name = "量子核岭回归"
    family = "linear"
    target_kind = "regression"
    tags = ("quantum", "kernel", "small-sample")
    param_schema = _shared_kernel_schema(with_C=False) + [
        ParamSpec("alpha", "正则 α", "number", 1.0, min=1e-6),
    ]

    def kernel_fit(self, K, yv, p, cfg):
        from sklearn.kernel_ridge import KernelRidge
        est = KernelRidge(alpha=float(p["alpha"]), kernel="precomputed")
        est.fit(K, yv)
        return est, {}

    def kernel_predict(self, Kt, result):
        return np.asarray(result.est.predict(Kt), float)

    def inspect_pages(self, cfg):
        return [PageSpec("fit", "拟合效果", "mpl", plots.plot_fit_1d),
                PageSpec("resid", "残差诊断", "mpl", plots.plot_residual),
                PageSpec("qk", "量子核矩阵", "mpl", _plot_kernel,
                         hint="训练期冻结的保真度核预览")]


# ================================================================ 聚类
@register
class QuantumKMeans(MLMethod):
    """量子可区分度 k 均值（书 §7.1：振幅编码 + d=1-|⟨x|c⟩|² + Lloyd 迭代）。

    诚实边界（书 §3.1.2/例 7.1）：振幅编码只保方向——⟨x|c⟩ 与 -⟨x|c⟩ 不可分，
    范数在经典侧保留；需要尺度信息时选 euclid 距离。
    """
    name = "qkmeans"
    display_name = "量子可区分度聚类"
    family = "cluster"
    task = TASK_CLUSTER
    tags = ("quantum", "partition", "direction")
    param_schema = [
        ParamSpec("n_clusters", "簇数 K", "int", 3, min=2, max=50),
        ParamSpec("distance", "距离度量", "select", "fidelity",
                  choices=["fidelity", "euclid"],
                  hint="fidelity=1-|⟨x|c⟩|²（只看方向）；euclid=经典对照"),
        ParamSpec("shots", "重叠测量次数 S", "int", 0, min=0, max=100000,
                  hint="0=理想保真度；>0 按式(7.18) SWAP 测试 Bernoulli 采样"),
        ParamSpec("max_iter", "最大迭代", "int", 40, min=2, max=200),
    ]

    def fit(self, X: pd.DataFrame, y, cfg: RunConfig, diag: bool = False):
        p = self.params(cfg)
        Xv = _as_np(X)
        m = Xv.shape[0]
        res = self._new_result(target_kind=None, params=p)
        if m < 3:
            res.error = "聚类至少需要 3 个样本"
            return res
        rng = np.random.RandomState(int(cfg.seed))
        k = max(2, min(int(p["n_clusters"]), m - 1))
        fid = str(p["distance"]) == "fidelity"
        shots = int(p["shots"])
        S, norms = qsim.amplitude_encode(Xv)
        A_all = S if fid else Xv

        def enc(vecs):                       # 原始向量 → 距离空间表示
            return qsim.amplitude_encode(vecs)[0] if fid else np.atleast_2d(vecs)

        def dmat(reps, cents):               # 样本×质心距离（式 7.18 / 欧氏）
            if fid:
                return qsim.fidelity_distance(reps, cents, shots=shots, rng=rng)
            return np.linalg.norm(reps[:, None, :] - cents[None, :, :], axis=2)

        # 初始化：最远点法（原始几何上选种子，k-means++ 的确定化简版）
        dmin = np.linalg.norm(Xv - Xv.mean(axis=0), axis=1)
        cidx = [int(dmin.argmax())]
        for _ in range(k - 1):
            j = int(dmin.argmax())
            cidx.append(j)
            dmin = np.minimum(dmin, np.linalg.norm(Xv - Xv[j], axis=1))
        Craw = Xv[cidx].copy()               # 质心保存在原始空间（式 7.4 均值）
        labels, D = None, None
        for _ in range(int(p["max_iter"])):
            D = dmat(A_all, enc(Craw))
            new = D.argmin(axis=1)
            if labels is not None and (new == labels).all():
                break
            labels = new
            for a in range(k):
                mem = np.flatnonzero(labels == a)
                if len(mem):                 # 空簇保留旧质心（式 7.4 分母守护）
                    Craw[a] = Xv[mem].mean(axis=0)
        res.artifacts = {"labels": labels, "embedding": _embed2d(Xv),
                         "X_scaled": Xv, "n_clusters": k,
                         "qdist": _preview(D) if D is not None else None}
        if len(set(labels.tolist())) >= 2:
            from sklearn.metrics import silhouette_score
            res.metrics["silhouette"] = float(silhouette_score(Xv, labels))
        res.metrics["n_clusters"] = float(len(set(labels.tolist())))
        if diag:
            res.diag = {"distance": str(p["distance"]),
                        "shots": shots,
                        "norms_min": float(norms.min()),
                        "norms_max": float(norms.max())}
        return res

    def inspect_pages(self, cfg):
        return [PageSpec("emb", "聚类散点", "mpl", plots.plot_scatter_emb),
                PageSpec("emb_pg", "交互缩放", "pg",
                         hint="pyqtgraph 滚轮缩放/拖拽平移，大点量不卡"),
                PageSpec("sil", "轮廓系数", "mpl", plots.plot_silhouette,
                         hint="需要缩放后的数值特征（开启数值缩放步骤效果最佳）"),
                PageSpec("qdist", "保真度距离热图", "mpl", _plot_qdist,
                         hint="仅 fidelity 距离模式产出样本×质心的 1-|⟨x|c⟩|²")]


def _embed2d(Xv):
    from sklearn.decomposition import PCA
    try:
        return PCA(n_components=2, random_state=42).fit_transform(Xv)
    except Exception:
        return np.column_stack([Xv[:, 0],
                                Xv[:, 1] if Xv.shape[1] > 1 else Xv[:, 0]])


# ================================================================ 降维
@register
class QuantumDMEPCA(MLMethod):
    """量子主成分（DME 幂迭代，书 §7.2）：ρ=C/Tr C 经密度矩阵指数化提取主子空间。

    协议对应关系：e^{ρτ} 是密度矩阵指数化（式 7.43-7.52 受控演化信道）的
    虚时间变体；Rayleigh-Ritz 块幂迭代收敛到 ρ 的最大 r 个本征方向，
    本征值 r̂ 乘回 Tr C 还原方差（式 7.37 归一化的逆）。与经典 PCA 同解、
    不同路径——τ 与谱隙（式 7.56）决定收敛/分辨力，diag 里显式报告。
    """
    name = "qpca"
    display_name = "量子主成分（DME）"
    family = "manifold"
    task = TASK_MANIFOLD
    tags = ("quantum", "spectral", "linear")
    param_schema = [
        ParamSpec("n_components", "保留成分数", "int", 2, min=1, max=20),
        ParamSpec("tau", "演化时间 τ", "number", 20.0, min=1.0, max=200.0,
                  hint="DME 步长；越大越贴近主子空间投影，但数值更 stiff"),
        ParamSpec("n_iter", "幂迭代次数", "int", 60, min=5, max=300),
    ]

    def fit(self, X: pd.DataFrame, y, cfg: RunConfig, diag: bool = False):
        p = self.params(cfg)
        Xv = _as_np(X)
        res = self._new_result(target_kind=None, params=p)
        Xc = Xv - Xv.mean(axis=0)
        d = Xv.shape[1]
        C = (Xc.T @ Xc) / max(Xc.shape[0], 1)
        trC = float(np.trace(C))
        if trC <= 1e-15:
            res.error = "数据零方差，无法归一化为密度矩阵（书 §7.2.2 前提）"
            return res
        rho = C / trC
        w, V = np.linalg.eigh(rho)             # 经典模拟后端（谱分解）
        r = max(1, min(int(p["n_components"]), d))
        tau = float(p["tau"])
        # 幂迭代用 e^{(ρ - r_max)τ}（防溢出；本征方向不变）
        A = (V * np.exp((w - w.max()) * tau)) @ V.T
        rng = np.random.RandomState(int(cfg.seed))
        W = rng.standard_normal((d, r))
        for _ in range(int(p["n_iter"])):
            W, _ = np.linalg.qr(A @ W)
        # Rayleigh-Ritz：在张成子空间内旋转到本征方向
        th, U = np.linalg.eigh(W.T @ rho @ W)
        order = np.argsort(th)[::-1]
        W = W @ U[:, order]
        r_hat = np.maximum(th[order], 0.0)
        evr = r_hat / max(float(np.trace(rho)), 1e-30)
        res.artifacts = {"embedding": Xc @ W,
                         "explained_variance_ratio": evr}
        res.metrics["explained_variance"] = float(np.clip(evr.sum(), 0, 1))
        if y is not None:
            res.artifacts["labels"] = pd.factorize(np.asarray(y))[0]
        ws = w[::-1]
        gap = float(ws[r - 1] - ws[r]) if d > r else float("nan")
        if diag:
            res.diag = {"n_in": d, "n_out": r, "trC": trC,
                        "spectral_gap_at_cut": gap,
                        "eigh_top": np.maximum(ws[:6], 0).tolist(),
                        "note": "DME 幂迭代收敛率 ~ e^{τ·γ}，γ=式(7.56) 谱隙"}
        return res

    def inspect_pages(self, cfg):
        return [PageSpec("emb", "降维散点", "mpl", plots.plot_scatter_emb),
                PageSpec("emb_pg", "交互缩放", "pg",
                         hint="pyqtgraph 滚轮缩放/拖拽平移，大点量不卡"),
                PageSpec("evr", "方差解释", "mpl", plots.plot_explained_variance,
                         hint="DME 提取的是 ρ=C/TrC 的本征谱，已乘回 Tr C")]


# ================================================================ 时序
@register
class QuantumReservoir(MLMethod):
    """量子储层计算（书 §9.1：注入式状态重置 + 固定 Ising 演化 + 线性岭读出）。

    密度矩阵级模拟（n≤6 比特）：ρ_t = Λ∘E_u(ρ_{t-1})（式 9.20），
    可观测量 x_m(t)=Tr[Z_m ρ_t]（式 9.21），读出按式(9.13) 闭式岭回归；
    虚拟节点 = 一个输入周期内取 V 个子时刻（式 9.32）。训练目标为一步预测，
    多步外推用预测值反哺输入的滚动展开。fit/指标沿用 TSMethod 的
    tr/te 划分与 artifacts 约定（forecast/actual/horizon）。
    """
    name = "qreservoir"
    display_name = "量子储层计算"
    family = "timeseries"
    task = TASK_TIMESERIES
    tags = ("quantum", "reservoir", "sequence")
    param_schema = [
        ParamSpec("n_qubits", "比特数", "int", 4, min=2, max=6,
                  hint="密度矩阵 2^n×2^n；模拟成本 ~2^{3n}"),
        ParamSpec("J", "Ising 耦合 J", "number", 0.8, min=0.0, max=3.0),
        ParamSpec("hx", "横场 h_x", "number", 0.6, min=0.0, max=3.0),
        ParamSpec("hz", "纵场 h_z", "number", 0.3, min=-3.0, max=3.0),
        ParamSpec("tau", "演化时长 τ", "number", 1.0, min=0.1, max=5.0),
        ParamSpec("virtual", "虚拟节点数 V", "int", 3, min=1, max=6,
                  hint="式(9.32) 时间复用：每步采 V 个子时刻可观测量"),
        ParamSpec("washout", "冲洗段长度", "int", 10, min=2, max=100,
                  hint="丢弃对初态敏感的前段（§9.1.1）"),
        ParamSpec("ridge", "读出正则 λ", "number", 0.0001, min=1e-8, max=1.0),
    ]

    def fit(self, X, y, cfg: RunConfig, diag: bool = False):
        p = self.params(cfg)
        s = np.asarray(pd.Series(y).to_numpy(float), float)
        n = len(s)
        res = self._new_result(target_kind="regression", params=p)
        t0 = _time.time()
        h = int(cfg.extras.get("horizon", max(2, int(n * 0.2))))
        h = max(1, min(h, n - 4))
        tr, te = s[: n - h], s[n - h:]
        try:
            fcst, ci, mc = self._forecast(tr, h, p, int(cfg.seed))
        except Exception as e:
            res.error = f"量子储层失败：{type(e).__name__}: {e}"
            return res
        res.artifacts = {
            "t_true": np.arange(n, dtype=float), "y_true": s,
            "forecast": np.concatenate([tr, fcst]).astype(float),
            "ci": np.concatenate([np.zeros(len(tr)), ci])
                  if ci is not None else None,
            "y_test": te, "f_test": fcst, "horizon": h,
            "qrc_mc": mc,
        }
        from ..core.contracts import regression_metrics
        res.metrics.update(regression_metrics(te, fcst))
        res.primary_metric = "rmse"
        res.metrics["mc_total"] = float(np.nansum(mc))
        if diag:
            res.diag = {"n_train": int(len(tr)), "horizon": h,
                        "n_features": int(p["n_qubits"]) * int(p["virtual"]),
                        "memory_capacity": np.round(mc, 3).tolist(),
                        "fit_s": round(_time.time() - t0, 2)}
        return res

    def _forecast(self, tr, h, p, seed):
        n = int(p["n_qubits"])
        V = int(p["virtual"])
        wash = int(p["washout"])
        T = len(tr)
        if T < wash + 20:
            raise ValueError("序列过短（需 washout+20 以上）")
        lo, hi = float(tr.min()), float(tr.max())
        if hi - lo < 1e-12:
            hi = lo + 1.0
        u = np.clip((tr - lo) / (hi - lo), 0.0, 1.0)
        dim = 1 << n
        # 哈密顿量 H = J Σ Z_iZ_{i+1} + hx Σ X_i + hz Σ Z_i（式 9.24 固定 H）
        H = np.zeros((dim, dim), complex)
        Zs, Xs = [], []
        for q in range(n):
            Z = qsim.embed_1q(qsim.PAULI_Z, q, n)
            Xq = qsim.embed_1q(qsim.PAULI_X, q, n)
            Zs.append(Z)
            Xs.append(Xq)
            H += float(p["hz"]) * Z + float(p["hx"]) * Xq
        for q in range(n - 1):
            H += float(p["J"]) * (Zs[q] @ Zs[q + 1])
        w, Vm = np.linalg.eigh(H)
        tau = float(p["tau"])
        Us = [(Vm * np.exp(-1j * w * tau * v / V)) @ Vm.conj().T
              for v in range(1, V + 1)]
        rho = np.zeros((dim, dim), complex)
        rho[0, 0] = 1.0
        feats, targets = [], []
        for t in range(T - 1):
            sigma = np.array([[1 - u[t], 0], [0, u[t]]], complex)  # 式(9.22)
            r = qsim.prep_q0(sigma, qsim.ptrace_q0(rho, n))        # 式(9.23)
            row = []
            for v in range(V):
                rv = qsim.evolve(r, Us[v])                          # 式(9.24)
                row += [qsim.expectation(rv, Zs[q]) for q in range(n)]
            feats.append(row)
            targets.append(u[t + 1])
            rho = qsim.evolve(r, Us[V - 1])
        F = np.asarray(feats)[wash:]
        Y = np.asarray(targets)[wash:]
        if len(F) < F.shape[1] + 5:
            raise ValueError("样本不足以训练线性读出")
        Zf = np.column_stack([np.ones(len(F)), F])                 # 式(9.3)
        lam = float(p["ridge"])
        wout = np.linalg.solve(Zf.T @ Zf + lam * np.eye(Zf.shape[1]),
                               Zf.T @ Y)                            # 式(9.13/9.14)
        resid = Y - Zf @ wout
        # 记忆容量谱（§9.3.2）：同一特征对延迟 k 输入的线性重构 R²
        mc = []
        for k in range(1, min(9, len(F) // 3)):
            tgt = Y[: len(Y) - k][:, None]
            src = np.roll(Y, -k)[: len(Y) - k][:, None]
            if len(src) < Zf.shape[1] + 3:
                mc.append(float("nan"))
                continue
            wk = np.linalg.solve(Zf[: len(src)].T @ Zf[: len(src)]
                                 + lam * np.eye(Zf.shape[1]),
                                 Zf[: len(src)].T @ src)
            pr = Zf[: len(src)] @ wk
            r2 = 1 - np.var(src - pr) / max(np.var(src), 1e-12)
            mc.append(float(max(r2, 0.0)))
        # 多步滚动外推：ρ 已是训练末端状态，预测值反哺输入
        out, uh = [], float(u[-1])
        for _ in range(h):
            sigma = np.array([[1 - uh, 0], [0, uh]], complex)
            r = qsim.prep_q0(sigma, qsim.ptrace_q0(rho, n))
            row = []
            for v in range(V):
                rv = qsim.evolve(r, Us[v])
                row += [qsim.expectation(rv, Zs[q]) for q in range(n)]
            rho = qsim.evolve(r, Us[V - 1])
            uh = float(np.clip(np.dot(wout, np.concatenate([[1.0], row])),
                               0.0, 1.0))
            out.append(lo + uh * (hi - lo))
        fc = np.asarray(out, float)
        sd = float(np.std(resid))
        ci = 1.96 * sd * np.sqrt(1 + np.arange(h) / max(len(F), 1))
        return fc, ci, np.asarray(mc)

    def inspect_pages(self, cfg):
        return [PageSpec("fc", "预测曲线", "mpl", plots.plot_forecast),
                PageSpec("res", "残差序列", "mpl", plots.plot_ts_residual),
                PageSpec("mc", "记忆容量", "mpl", _plot_mc,
                         hint="短期记忆容量 MC_k（§9.3.2）：储层特征线性重构"
                              "延迟 k 步输入的可决定系数，求和为总记忆")]


# ================================================================ 绘图（纯 mpl）
def _plot_kernel(ax, result):
    K = result.artifacts.get("q_kernel")
    if K is None:
        return
    im = ax.imshow(K, cmap="viridis", vmin=0, vmax=max(float(K.max()), 1e-9))
    ax.set_title(f"保真度核 |⟨φ(x)|φ(z)⟩|²（预览 {K.shape[0]}×{K.shape[1]}）",
                 fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.colorbar(im, ax=ax, fraction=0.046)


def _plot_qdist(ax, result):
    D = result.artifacts.get("qdist")
    if D is None:
        return
    im = ax.imshow(D, cmap="magma", aspect="auto")
    ax.set_title("量子可区分度距离 1-|⟨x|c⟩|²（预览）", fontsize=14)
    ax.set_xlabel("质心")
    ax.set_ylabel("样本（抽稀）")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)


def _plot_mc(ax, result):
    mc = result.artifacts.get("qrc_mc")
    if mc is None or len(mc) == 0:
        return
    k = np.arange(1, len(mc) + 1)
    ax.bar(k, np.nan_to_num(mc), color="tab:purple", alpha=0.75)
    ax.plot(k, np.cumsum(np.nan_to_num(mc)), "o-", color="tab:orange",
            lw=1.4, label="累计 MC")
    ax.set_xlabel("输入延迟 k")
    ax.set_ylabel(r"$R^2_k$")
    ax.set_title(f"短期记忆容量谱（总和 {np.nansum(mc):.2f}）", fontsize=14)
    ax.legend(fontsize=13)
    ax.grid(alpha=0.25, axis="y")
