# -*- coding: utf-8 -*-
"""核心契约：所有方法族共享的基类 / 协议 / 数据结构。

设计原则（见 docs/项目定位.md）：
- P1 核心逻辑与界面分离：本模块零 Qt 依赖；
- P2 契约优先，注册扩展：方法只通过本模块定义的结构与框架交互；
- P3 零侵入默认：diag=False 时数值路径不变、不产出诊断。

== 方法契约一览 ==

MLMethod 子类必须声明 ClassVar：
  name / family / task / tags / param_schema
并实现：
  fit(problem, cfg, diag)         -> MLResult   （problem: Problem，见下"范式"节）
可选覆写：
  predict(X, result)              -> np.ndarray
  evaluate(result, X, y)          -> dict   （默认按 task 自动计算）
  cross_validate(X, y, cfg, cv)   -> dict   （默认按 task 自动交叉验证）
  accepts_aux                     -> tuple  （声明支持哪些辅助结构，如图 AUX_GRAPH）

MLResult.artifacts 约定键（UI 按此消费，缺失即显示语义化占位）：
  supervised : y_true, y_pred, y_prob, residual, feature_importance
  cluster    : labels, score, silhouette_per_sample
  manifold   : embedding, explained_variance_ratio
  anomaly    : scores, labels, contamination
  timeseries : forecast, actual, residual
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 任务常量
TASK_SUPERVISED = "supervised"      # 回归/分类统一：y 为 Series，回归 float / 分类 label
TASK_CLUSTER = "cluster"
TASK_MANIFOLD = "manifold"          # 降维
TASK_ANOMALY = "anomaly"
TASK_TIMESERIES = "timeseries"

ALL_TASKS = (TASK_SUPERVISED, TASK_CLUSTER, TASK_MANIFOLD,
             TASK_ANOMALY, TASK_TIMESERIES)

# ---------------------------------------------------------------- 建模目的
# task 是"学习机制"轴，purpose 是"建模目的"轴（数模视角：预测/判别/归因/…）。
# 一个方法可服务多个目的；UI 按此筛选。
PURPOSE_PREDICT = "predict"            # 预测/预报（回归、时序外推）
PURPOSE_DISCRIMINATE = "discriminate"  # 判别/分类
PURPOSE_EXPLAIN = "explain"            # 归因解释（系数、特征重要性、稀疏选择）
PURPOSE_CLUSTER = "cluster"            # 分群/类型分析
PURPOSE_REDUCE = "reduce"              # 降维/特征提取
PURPOSE_EVALUATE = "evaluate"          # 综合评价（方差权重、主成分打分）
PURPOSE_ANOMALY = "anomaly"            # 异常检测
PURPOSE_SURROGATE = "surrogate"        # 代理模型/响应面（供优化算法调用）
PURPOSE_BASELINE = "baseline"          # 参照基线（任何模型都应赢它）

PURPOSE_LABELS = {
    PURPOSE_PREDICT: "预测", PURPOSE_DISCRIMINATE: "判别",
    PURPOSE_EXPLAIN: "归因解释", PURPOSE_CLUSTER: "分群",
    PURPOSE_REDUCE: "降维", PURPOSE_EVALUATE: "综合评价",
    PURPOSE_ANOMALY: "异常检测", PURPOSE_SURROGATE: "代理模型",
    PURPOSE_BASELINE: "参照基线",
}


def derive_purposes(task: str, target_kind: Optional[str],
                    tags=(), family: str = "") -> tuple:
    """按"学习机制 + 已沉淀 tags"推导建模目的（新方法零成本继承）。

    方法可用 ClassVar purposes 显式覆盖/补充。
    注意："参照基线"目的只认 family=='baseline'——tags 里的 'baseline'
    是"该族默认起点"的意思，与数模的"必赢基线"是两回事（曾撞名误标）。
    """
    out: list = []
    if task == TASK_SUPERVISED:
        out.append(PURPOSE_PREDICT)
        if target_kind in (None, "classification"):
            out.append(PURPOSE_DISCRIMINATE)
    elif task == TASK_CLUSTER:
        out.append(PURPOSE_CLUSTER)
    elif task == TASK_MANIFOLD:
        out.append(PURPOSE_REDUCE)
    elif task == TASK_ANOMALY:
        out.append(PURPOSE_ANOMALY)
    elif task == TASK_TIMESERIES:
        out.append(PURPOSE_PREDICT)
    t = set(tags)
    if t & {"interpretable", "sparse", "importance", "feature-selection",
            "regularized", "statistical", "bayesian"}:
        out.append(PURPOSE_EXPLAIN)
    if t & {"surrogate", "uncertainty"}:
        out.append(PURPOSE_SURROGATE)
    if family == "baseline":
        out.append(PURPOSE_BASELINE)
    if task == TASK_MANIFOLD and t & {"linear", "supervised"}:
        out.append(PURPOSE_EVALUATE)     # PCA/LDA 的方差权重可做综合评价
    seen, ordered = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return tuple(ordered)


# ---------------------------------------------------------------- 指标方向
# 全局文字缩放因子：UI 样式（ui.theme.css/pt）与所有绘图（matplotlib fontsize、
# pyqtgraph fontScale）统一乘它；改这一处即可整体调大小。
FONT_SCALE = 2

# "越小越好"的指标集——单一事实来源（M1：曾三处各写一份且已分叉）。
# 新增越低越好的指标只改这里，core.runner / ui.gallery / opt.bridges 全部跟随。
LOWER_IS_BETTER = frozenset({"rmse", "mae", "mape", "silhouette_deficit"})


def is_lower_better(metric: str) -> bool:
    """该指标是否越小越好（决定对比表/画廊的排序方向、调参的统一最小化）。"""
    return metric in LOWER_IS_BETTER


# ---------------------------------------------------------------- 任务判定
# 数值且唯一值 > 该阈值 -> 回归，否则分类（M10：曾三处独立实现 + UI 抄文案）。
REGRESSION_CARDINALITY = 20


def infer_kind(y) -> Optional[str]:
    """从目标列推断回归/分类（唯一的任务判定入口）。

    None -> None；非数值 -> classification；数值且基数 > REGRESSION_CARDINALITY
    -> regression。接受 pandas Series、NumPy ndarray、列表等一维目标输入；
    pipeline / dataset / methods 一律调用此函数。
    """
    if y is None:
        return None
    ys = y if isinstance(y, pd.Series) else pd.Series(y)
    if pd.api.types.is_numeric_dtype(ys) and ys.nunique() > REGRESSION_CARDINALITY:
        return "regression"
    return "classification"


# ---------------------------------------------------------------- 数据视图
@dataclass
class DataSpec:
    """数据集的一个视图：管道输出 + 划分方案。"""
    X: pd.DataFrame
    y: Optional[pd.Series] = None
    train_idx: Optional[np.ndarray] = None
    test_idx: Optional[np.ndarray] = None
    target_kind: Optional[str] = None    # 'regression' | 'classification' | None
    n_classes: int = 0
    meta: dict = field(default_factory=dict)
    #: **辅助结构**（观测之间的关系：图边表 / 将来的多视图、序列掩码…）。
    #: 保留键见 `AUX_GRAPH`。放 DataSpec 上（而不是只放 Problem）是为了让它
    #: 随管道/存档一起走 —— 纯加法字段，没有辅助结构的路径一行不变。
    aux: dict = field(default_factory=dict)

    def split(self):
        """返回 (Xtr, Xte, ytr, yte)；无划分时 train==test 全量。"""
        tr = self.train_idx if self.train_idx is not None else np.arange(len(self.X))
        te = self.test_idx if self.test_idx is not None else np.arange(len(self.X))
        ytr = self.y.iloc[tr] if self.y is not None else None
        yte = self.y.iloc[te] if self.y is not None else None
        return self.X.iloc[tr], self.X.iloc[te], ytr, yte


#: 辅助表（`Problem.aux`）里的**保留键**。约定固定、可校验（见 `Problem.require`）。
AUX_GRAPH = "graph"

#: **逐行负载**：``aux["payload"] = {名字: 数组}``，每个数组**第 0 维 = 行数**。
#:
#: 为什么需要它（NQS 需求的"最高"之一）：有些任务的"一行"自带**变长/复数**的大数组
#: （一个里德堡阵列配置的 2^N 个复振幅、它采样的 64 个位串、逐原子特征…），
#: 那既不是列、也不是"行之间的图"。压进一列会毁掉形状，塞进图会毁掉语义。
#: ⇒ 给它一个**明确的槽**：`payload`；并把"切一批行"定义成"切每个数组的前几维"
#: （于是切分/指标/存档的既有语义一行都不用改）。
#: 变长部分由调用方**先 pad 再配 mask**（`ml_toolbox.core.paddings.pad_rows` 是助手）。
AUX_PAYLOAD = "payload"

#: `aux["graph"]` 的列约定：`src`/`dst` 是**行号**（指向 `spec.X` 的行）。
#: 可选 `weight`（缺省全 1），**以及任意多个特征列**（多分量耦合，见 `edge_attr`）。
#: 节点数 = `len(spec.X)`。
GRAPH_COLUMNS = ("src", "dst", "weight")


class Problem:
    """**一个可学习问题**：观测 + 标签 + （可选的）观测之间的关系与划分。

    ## 为什么要它（用户裁决的范式扩张）

    原来的方法接口是 ``fit(X, y, cfg, diag)``。把仓库里所有方法族摊开看，
    真正在变的从来不是"数据有几个参数"，而是**"观测是什么、观测之间有什么关系"**：

    | 族 | 观测 | 观测之间的关系 | 预测单位 |
    |---|---|---|---|
    | 经典监督 | 一行一个样本 | **无关**（条件独立） | 一行一个标签 |
    | 时序 | 一个时间点 | **有序** | 一个时间点一个值 |
    | 聚类/降维/异常 | 一行一个样本 | 无关 | 一行一个分数/坐标 |
    | **图** | **一个节点** | **按边相连** | 一个节点一个标签 |

    ⇒ "观测单位"和"预测单位"四族完全一样，唯一变的是**中间那列**。
    而老契约只有前两列（`X` 是观测、`y` 是标签），第三列**无处安放**——
    ``Dataset.time_col`` 是给它打的第一个补丁（只有"有序"一种，还是特命名字段）。

    ⇒ 本类把"观测 + 标签 + 关系"收成一个对象，`fit` 只收它。

    ## 与 `DataSpec` 的关系（**承载，不是替代**）

    `Problem` **持有**一个 `DataSpec`：管道/存档/UI 全都继续用 `DataSpec`，
    一行不改。`fit` 收到的是 `Problem`；`problem.spec` 就是原来那个 spec。

    ## 三个视图（方法只该用这三个）

    - ``problem.X`` / ``problem.y`` —— 观测与标签（时序之外都直接可用）；
    - ``problem.rows("train"|"test")`` —— **按划分取行**（经典方法就这一行）；
    - ``problem.require("graph")`` —— 取辅助结构；**没声明支持就结构化拒绝**
      （不许"数据带了而方法悄悄扔掉"——那正是 `time_col` 时代的老毛病）。
    """

    def __init__(self, spec: DataSpec, *,
                 aux: Optional[dict] = None,
                 allowed: tuple = (),
                 task: Optional[str] = None) -> None:
        self.spec = spec
        #: 辅助表：保留键见 `AUX_GRAPH`。值一律是 DataFrame（可序列化、可存档）。
        #: 默认取 `spec.aux`（管道/存档带过来的）；显式传 aux 可覆盖（判据用）。
        self.aux: dict = dict(spec.aux if aux is None else aux)
        #: 本方法**声明支持**的辅助键（来自 `MLMethod.accepts_aux`）。空 = 都不支持。
        self.allowed: tuple = tuple(allowed or ())
        self.task = task

    # ------------------------------------------------------------ 直通（省得层层 .spec）
    @property
    def X(self):
        return self.spec.X

    @property
    def y(self):
        return self.spec.y

    @property
    def train_idx(self):
        return self.spec.train_idx

    @property
    def test_idx(self):
        return self.spec.test_idx

    @property
    def target_kind(self):
        return self.spec.target_kind

    @property
    def meta(self) -> dict:
        return self.spec.meta

    def __len__(self) -> int:
        return int(len(self.spec.X))

    # ------------------------------------------------------------ 视图
    def rows(self, part: str = "train"):
        """按划分取行 → ``(X, y)``。``part`` ∈ train/test/all。

        ⚠ **图任务用 `masks()` 而不是这个**：图不能按行切（切了边就没了/会泄漏），
        见 `masks()` 的说明。
        """
        tr, te = self.spec.train_idx, self.spec.test_idx
        if part == "train":
            idx = tr if tr is not None else np.arange(len(self.spec.X))
        elif part == "test":
            idx = te if te is not None else np.arange(len(self.spec.X))
        else:
            idx = np.arange(len(self.spec.X))
        X = self.spec.X.iloc[idx]
        y = self.spec.y.iloc[idx] if self.spec.y is not None else None
        return X, y

    def masks(self) -> tuple:
        """``(train_mask, test_mask)`` —— **布尔掩码**，索引仍是全量行号。

        为什么要它（图任务的正解）：图上的划分不能靠切行——
        ① 训练节点会沿边"看到"验证节点的特征（消息传递），分数会被乐观化；
        ② 切行会让边索引失效/要重编号。
        ⇒ 图任务的标准做法是**同一张图、只对指定节点算损失**，即"节点掩码"。
        这与时序的 `time_split` 是同一类修正（都是"行不独立"），只是图的版本。
        """
        n = len(self.spec.X)
        tr = self.spec.train_idx if self.spec.train_idx is not None else np.arange(n)
        te = self.spec.test_idx if self.spec.test_idx is not None else np.arange(n)
        m_tr = np.zeros(n, dtype=bool)
        m_te = np.zeros(n, dtype=bool)
        m_tr[np.asarray(tr, dtype=int)] = True
        m_te[np.asarray(te, dtype=int)] = True
        return m_tr, m_te

    def subset(self, X, y=None) -> "Problem":
        """造一个"**全量即训练**"的子问题（换了表/标签，划分与关系照旧）。

        用途：某些方法在交给基类之前要先做一次**自己的**预处理
        （如 MultinomialNB 内部 MinMax 平移）⇒ 它把变换后的表包成子问题再委托，
        这样基类里那段"取 rows('train')"仍然成立，不必为它开特例。
        """
        idx = np.arange(len(X))
        sub = DataSpec(X=X.reset_index(drop=True),
                       y=(y.reset_index(drop=True) if y is not None else None),
                       train_idx=idx, test_idx=idx,
                       target_kind=self.spec.target_kind, meta=self.spec.meta)
        return Problem(sub, aux=self.aux, allowed=self.allowed, task=self.task)

    # ------------------------------------------------------------ 划分模式
    def partition_mode(self) -> str:
        """划分该按什么单位做（**由数据决定，不由方法决定**）。

        - ``"masks"``：带了**图**（``aux["graph"]``）⇒ 按**节点掩码**划分。
          为什么：图的边按行号写，切行会让边索引失效；更要命的是训练节点会沿边
          "看到"验证节点（消息传递）⇒ 分数乐观化。这与时序的 `time_split` 是
          **同一类修正**（都是"行不独立"），只是图的版本。
        - ``"rows"``：普通表格（**含只有 `payload` 的情形**）⇒ 沿用既有
          `spec.split()`。

        ⚠ **只有 `payload` 不算"行不独立"**：逐行负载是"每行自带的东西"，
        切行完全成立（这正是我们要保住 row-level 语义的原因）。
        早先我写成 `"masks" if self.aux else "rows"` ⇒ 一带负载就误走图路径。
        """
        return "masks" if AUX_GRAPH in self.aux else "rows"

    # ------------------------------------------------------------ 负载（逐行的大数组）
    def payload_keys(self) -> list:
        """这份数据带了哪些逐行负载（稳定排序）。"""
        pay = self.aux.get(AUX_PAYLOAD) or {}
        return sorted(str(k) for k in pay)

    def payload(self) -> dict:
        """取逐行负载 ``{名字: 数组}``（没有就回空 dict，**不报错**）。

        ⚠ 与 `require` 的分工：负载是"**可以有**"的附加物（没有它就退化成普通表格
        任务），而图是"**要有才对**"的结构（方法不支持就该拒）。
        """
        return dict(self.aux.get(AUX_PAYLOAD) or {})

    def payload_facts(self) -> dict:
        """负载的形状摘要（进 diag / 工具回执；没有就回空 dict）。"""
        pay = self.payload()
        if not pay:
            return {}
        out = {}
        for k, v in pay.items():
            try:
                import numpy as _np
                a = _np.asarray(v)
                out[str(k)] = {"shape": list(a.shape), "dtype": str(a.dtype)}
            except Exception:                                  # noqa: BLE001
                out[str(k)] = {"shape": None, "dtype": None}
        return out

    # ------------------------------------------------------------ 能力声明
    def aux_keys(self) -> list:
        """这份数据带了哪些辅助结构（稳定排序）。"""
        return sorted(self.aux)

    def require(self, key: str):
        """取一个辅助结构；**数据没有或方法没声明支持**都结构化拒绝。

        两种拒绝要分开说，因为改法不同：
        - 方法**没声明**支持它 ⇒ 是"这个方法吃不了这种数据"（换方法，或申请支持）；
        - 声明了但**数据没带** ⇒ 是"这份数据缺这个结构"（补数据）。
        ⚠ 绝不静默返回 None —— 那会让方法以为"没有边"而按独立样本跑，
        得到一个**看起来正常但其实错**的结果（`time_col` 时代的老毛病）。
        """
        if key not in self.allowed:
            raise ValueError(
                f"方法没有声明支持辅助结构 {key!r}"
                f"（`accepts_aux` 里没有它）⇒ 拒绝让它消费这份数据。"
                f"已声明：{list(self.allowed)}")
        if key not in self.aux:
            raise ValueError(
                f"这份数据没有带辅助结构 {key!r}（带了的是：{self.aux_keys()}）")
        return self.aux[key]

    def has_aux(self, key: str) -> bool:
        return key in self.aux

    # ------------------------------------------------------------ 图任务的读取口
    def graph(self):
        """取图结构 → ``(edge_index, edge_weight)``（torch 友好的两个张量）。

        边表的 `src`/`dst` 是**行号**（指向 `spec.X` 的行）⇒ 直接就是
        `edge_index`，不需要节点重编号（因为我们不切行）。
        """
        tbl = self.require(AUX_GRAPH)
        import numpy as np
        src = np.asarray(tbl["src"], dtype=np.int64)
        dst = np.asarray(tbl["dst"], dtype=np.int64)
        edge_index = np.stack([src, dst], axis=0)
        if "weight" in tbl.columns:
            w = np.asarray(tbl["weight"], dtype=np.float32)
        else:
            w = np.ones(len(src), dtype=np.float32)
        return edge_index, w

    #: `aux["graph"]` 里**不是**结构的三列（其余列都当**边特征**）。
    _GRAPH_STRUCT_COLS = ("src", "dst", "weight")

    def edge_attr(self):
        """**多分量边特征** → ``(E, F)`` 的 float 数组（没有特征列就回 ``None``）。

        为什么需要（NQS 需求）：里德堡阵列的耦合不是**一个数** —— 他们有 6 个
        物理分量（V、J 的实虚部、A 的实虚部、距离）。硬压成一个标量权重
        **会毁掉物理量**，而"边特征只能是一个标量"逼着人去做那种压缩。

        ⇒ 约定：`aux["graph"]` 里除 `src`/`dst`/`weight` 之外的**所有数值列**
        按列序拼成 `(E, F)`，列名就是分量名（顺序稳定 = 表的列序）。
        复数分量请拆成 `_re`/`_im` 两列（或放进 payload —— 那里保住复数 dtype）。
        """
        tbl = self.require(AUX_GRAPH)
        import numpy as np
        cols = [c for c in tbl.columns if c not in self._GRAPH_STRUCT_COLS]
        if not cols:
            return None
        return np.stack([np.asarray(tbl[c], dtype=np.float32) for c in cols],
                        axis=1)

    def edge_attr_names(self) -> list:
        """边特征的分量名（列序；没有就回空列表）。"""
        tbl = self.require(AUX_GRAPH)
        return [str(c) for c in tbl.columns if c not in self._GRAPH_STRUCT_COLS]

    def graph_facts(self) -> dict:
        """图的规模摘要（进 diag / 工具回执；没有图就回空 dict）。"""
        if AUX_GRAPH not in self.aux:
            return {}
        tbl = self.aux[AUX_GRAPH]
        n = len(self.spec.X)
        deg = None
        try:
            import numpy as np
            d = np.zeros(n, dtype=np.int64)
            for col in ("src", "dst"):
                vals = np.asarray(tbl[col], dtype=np.int64)
                vals = vals[(vals >= 0) & (vals < n)]
                np.add.at(d, vals, 1)
            deg = {"mean": float(d.mean()) if n else 0.0,
                   "min": int(d.min()) if n else 0, "max": int(d.max()) if n else 0}
        except Exception:
            deg = None
        return {"n_nodes": n, "n_edges": int(len(tbl)),
                "has_weight": "weight" in tbl.columns, "degree": deg,
                "isolated": int((deg or {}).get("min", 0) == 0) if deg else None}

    def __repr__(self) -> str:
        return (f"Problem(n={len(self)}, target_kind={self.spec.target_kind!r}, "
                f"task={self.task!r}, aux={self.aux_keys()})")


# ---------------------------------------------------------------- 可教学的参数错误
class ParamValueError(ValueError):
    """**参数值非法**——可教学的一类失败（继承 ``ValueError`` 保持兼容）。

    ## 为什么单开一个类型（2026-10-03 实测）

    `ParamSpec.clean` 原先抛**裸 `ValueError`** ⇒ 到了 adapter 层只能归成
    `internal_error`/`command_failed`，AI 读到的是"框架坏了"或一句万能话。
    而它其实是**最该被教会**的一类：用户/AI 把参数写错了，改一下就好。

    症状实测（`overrides={"epochs": "many"}`）：消息里明明写着
    "参数 epochs 需要数字，收到 'many'"，**归类却是 internal_error**
    —— 那正是指导意见 P0-3 最怕的形态（"把参数崩了当成物理结论"）。

    ⚠ 本类**不带 mecha 依赖**（core 不许 import 边界层）：它只是"带 kind/hint 的
    ValueError"，由 adapter 层原样转成结构化失败（`engine` 的 catch 分支）。
    """

    def __init__(self, message: str, *, kind: str = "bad_param_value",
                 hint: str = "", suggest: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.hint = hint
        self.suggest = suggest

    def teaching_text(self) -> str:
        out = f"[{self.kind}] {self}"
        for label, value in (("提示", self.hint), ("建议", self.suggest)):
            if value:
                out += f"\n{label}：{value}"
        return out


# ---------------------------------------------------------------- 参数 schema
@dataclass
class ParamSpec:
    """旋钮声明：UI 据此生成控件，并校验/清洗运行级覆写。

    共享内核：ML 侧用它做参数面板，优化侧（ml_toolbox.opt）用它构成
    ParamSpace（采样/投影/向量化）。log=True 表示对数尺度（lr/正则强度类）。
    """
    key: str
    label: str = ""
    kind: str = "number"        # number | int | select | bool | text
    default: object = None
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[list] = None
    hint: str = ""
    log: bool = False           # 对数尺度（需 min>0；优化侧采样用，ML 侧忽略）

    def clean(self, raw):
        """把 UI 原始值转成类型正确的值；非法输入抛 :class:`ParamValueError`。

        ⚠ 抛的是**可教学**的那种（带 kind/hint/suggest），不是裸 `ValueError`：
        后者到了 adapter 层会变成 `internal_error` ⇒ AI 把"我参数写错了"
        读成"框架坏了"（实测，见 `ParamValueError` 的 docstring）。
        提示里尽量带上"合法范围"与**作者自己写的** `self.hint`——那是最贴近
        这个参数语义的一句话。
        """
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            return self.default
        if self.kind == "bool":
            if isinstance(raw, bool):
                return raw
            return str(raw).strip().lower() in ("1", "true", "yes", "on")
        if self.kind in ("number", "int"):
            try:
                v = float(raw)
            except (TypeError, ValueError):
                raise ParamValueError(
                    f"参数 {self.key} 需要{'整数' if self.kind == 'int' else '数字'}，"
                    f"收到 {raw!r}",
                    hint=self._hint_for_kind(),
                    suggest=f"查看该方法声明的可调参数：describe_method(name=…)"
                            f"（本参数是 {self.key!r}）",
                ) from None
            if self.kind == "int":
                v = int(round(v))
            if self.min is not None and v < self.min:
                raise ParamValueError(
                    f"参数 {self.key}={v} 小于下限 {self.min}",
                    hint=self._range_hint(),
                    suggest=f"把 {self.key} 调到 {self.min} 及以上",
                )
            if self.max is not None and v > self.max:
                raise ParamValueError(
                    f"参数 {self.key}={v} 大于上限 {self.max}",
                    hint=self._range_hint(),
                    suggest=f"把 {self.key} 调到 {self.max} 及以下",
                )
            return v
        if self.kind == "select":
            v = str(raw)
            if self.choices and v not in [str(c) for c in self.choices]:
                raise ParamValueError(
                    f"参数 {self.key}={v!r} 不在候选 {self.choices}",
                    hint=self._hint_for_kind(),
                    suggest=f"{self.key} 只能取 {' / '.join(str(c) for c in self.choices)}",
                )
            return v
        return str(raw)

    def _range_hint(self) -> str:
        """合法区间 + 作者写的 hint（作者那句话最贴语义，别丢）。"""
        parts = []
        if self.min is not None or self.max is not None:
            lo = "−∞" if self.min is None else self.min
            hi = "+∞" if self.max is None else self.max
            parts.append(f"合法范围 [{lo}, {hi}]")
        if self.hint:
            parts.append(self.hint)
        return "；".join(parts)

    def _hint_for_kind(self) -> str:
        base = {
            "int": "这个参数要整数",
            "number": "这个参数要数字",
            "select": "这个参数要从候选里选",
        }.get(self.kind, "")
        if self.hint:
            return f"{base}；{self.hint}" if base else self.hint
        return base


# ---------------------------------------------------------------- 运行配置
@dataclass
class RunConfig:
    """一次运行的配置（运行级，不写回全局 —— pitfalls #17）。"""
    overrides: dict = field(default_factory=dict)   # {param_key: raw_value}
    diag: bool = False
    seed: int = 42
    extras: dict = field(default_factory=dict)      # 任务级扩展（horizon/cv_folds/...）

    def resolve(self, schema: list[ParamSpec]) -> dict:
        """按 schema 清洗覆写；未覆写的取默认。返回干净参数字典。"""
        out = {p.key: p.default for p in schema}
        for p in schema:
            if p.key in self.overrides:
                out[p.key] = p.clean(self.overrides[p.key])
        return out


# ---------------------------------------------------------------- 运行结果
@dataclass
class MLResult:
    method_name: str
    task: str
    target_kind: Optional[str] = None
    metrics: dict = field(default_factory=dict)         # {指标名: float}
    primary_metric: str = ""                            # 对比视图排序依据
    artifacts: dict = field(default_factory=dict)       # 可视化契约数据
    params: dict = field(default_factory=dict)          # 实际生效参数
    diag: Optional[dict] = None                         # cfg.diag=True 时产出
    elapsed: float = 0.0
    error: Optional[str] = None                         # 失败时记录，不抛出
    #: ⭐ **失败的可教学三件套**（2026-10-03 补；默认空 = 旧行为不变）。
    #:
    #: 为什么必须放在 `MLResult` 上：`runner.run_one` 是 `method.fit` 的**唯一**
    #: 兜底点，它原先把任何异常压成一句 `traceback` 字符串 ⇒ 到 AI 那里
    #: `error_kind` 是空的、hint 退化成万能句，**"你参数写错了"被读成"框架坏了"**
    #: （实测：`overrides={"C": "很大"}` 报成 `command_failed` +
    #: `ml_toolbox.core.contracts.ParamValueError: …`，连内部类名都漏出去了）。
    #: ⇒ 让失败**带着 kind/hint/suggest 穿过 runner**，engine 再原样送进回执。
    error_kind: str = ""
    error_hint: str = ""
    error_suggest: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None

    def fail(self, message: str, *, kind: str = "internal_error",
             hint: str = "", suggest: str = "") -> "MLResult":
        """标成失败并**保留可教学信息**（比手写 `res.error = ...` 更不容易漏字段）。"""
        self.error = message
        self.error_kind = kind
        self.error_hint = hint
        self.error_suggest = suggest
        return self


# ---------------------------------------------------------------- 检视页声明
@dataclass
class PageSpec:
    """方法声明自己检视页：UI 基类只负责装配，不预设图长什么样（G2）。

    - kind="mpl" + plot(ax, result)：纯 matplotlib 绘制函数（方法层零 Qt）；
    - kind="pg"：交给 UI 的通用自适应组件（热图/散点缩放）；
    - kind="table"/"cards"：消费 result.artifacts 中的表格/摘要；
    - data：table 页的替代数据源 fn(record)->DataFrame（优化侧 record 无
      artifacts，C4 用它在声明时绑定计算；ML 侧留空走 artifacts 老路）；
    - plot=None 且无 artifacts：UI 显示 hint 语义化占位（模式 7）。
    """
    key: str
    title: str
    kind: str = "mpl"               # mpl | pg | table | cards
    plot: Optional[Callable] = None    # (fig_ax, result) -> None
    builder: Optional[Callable] = None  # (result, parent) -> QWidget（UI 层扩展用）
    hint: str = ""                  # 无数据时的语义化占位提示
    data: Optional[Callable] = None    # (record) -> DataFrame（table 页数据源）


# ---------------------------------------------------------------- 方法基类
class MLMethod(abc.ABC):
    """所有机器学习方法的统一基类（工具箱资源共享的根本契约）。"""

    name: ClassVar[str] = ""
    display_name: ClassVar[str] = ""
    family: ClassVar[str] = ""          # linear | ensemble | svm | knn | bayes | cluster | manifold | anomaly | timeseries
    task: ClassVar[str] = TASK_SUPERVISED
    purposes: ClassVar[tuple] = ()      # 建模目的（见 PURPOSE_*，可多值）
    target_kind: ClassVar[Optional[str]] = None   # supervised: 'regression'|'classification'|None(皆可)
    tags: ClassVar[tuple] = ()
    param_schema: ClassVar[list] = []
    supports_sample_weight: ClassVar[bool] = False
    requires_numeric: ClassVar[bool] = True       # False = 内部自行编码（如原始 RF）
    #: **声明**本方法支持哪些辅助结构（`Problem.aux` 的保留键，见 `AUX_GRAPH`）。
    #: 默认空 = 只吃"观测 + 标签"（经典方法都是这样，一行不用改）。
    #: ⚠ 数据带了辅助结构而方法没声明支持 ⇒ `Problem.require` **结构化拒绝**，
    #: 不允许"悄悄扔掉"（`time_col` 时代"时序忽略外生 X"就是那么来的）。
    accepts_aux: ClassVar[tuple] = ()

    # ------------------------------------------------ 子类必须实现
    @abc.abstractmethod
    def fit(self, problem: "Problem", cfg: RunConfig,
            diag: bool = False) -> MLResult:
        """在**一个问题**上拟合。

        ## 范式（2026-10-03 扩张，取代 ``fit(X, y, cfg, diag)``）

        老接口只能表达"观测 + 标签"，表达不了**观测之间的关系**（图 = 按边相连、
        时序 = 有序）⇒ 图方法无处安放。现在统一收一个 :class:`Problem`。

        - 经典方法：``X, y = problem.rows("train")``（就是原来那两样）；
        - 时序：``problem.y`` 是整条序列，外生特征在 ``problem.X`` 里；
        - 图：``problem.require("graph")`` + ``problem.masks()``。
        """
        ...

    # ------------------------------------------------ 可选覆写
    def predict(self, X: pd.DataFrame, result: MLResult):
        raise NotImplementedError

    def fit_extra_artifacts(self, X: pd.DataFrame, result: MLResult) -> dict:
        """run_one 在 predict 前调用：概率等附加产物（默认无）。"""
        return {}

    def problem(self, spec: DataSpec, *, aux: Optional[dict] = None) -> "Problem":
        """由 `DataSpec`（+ 可选辅助结构）造一个 `Problem`（带上本方法的能力声明）。"""
        return Problem(spec, aux=aux, allowed=self.accepts_aux, task=self.task)

    def evaluate(self, result: MLResult, X: pd.DataFrame, y) -> dict:
        """默认评估：按 task/target_kind 自动计算标准指标。"""
        a = result.artifacts
        if result.task != TASK_SUPERVISED:
            return dict(result.metrics)
        if result.target_kind == "regression":
            yt, yp = np.asarray(a["y_true"], float), np.asarray(a["y_pred"], float)
            return regression_metrics(yt, yp)
        yt, yp = np.asarray(a["y_true"]), np.asarray(a["y_pred"])
        m = classification_metrics(yt, yp)
        if a.get("y_prob") is not None:
            m.update(roc_metrics(yt, a["y_prob"], list(np.unique(yt))))
        return m

    def cross_validate(self, X: pd.DataFrame, y, cfg: RunConfig,
                       cv_folds: int = 5, *,
                       target_kind: Optional[str] = None,
                       aux: Optional[dict] = None) -> dict:
        """默认交叉验证：k 折重跑 fit，统计主指标 mean±std。

        ⚠ `aux`（图边表等）**必须传进来**：CV 按行切片，对图数据会静默失真
        ⇒ `default_cv` 要据此**响亮拒绝**（而不是算出一个看起来正常的错分数）。
        """
        return default_cv(self, X, y, cfg, cv_folds,
                          target_kind=target_kind, aux=aux)

    # ------------------------------------------------ 检视页声明（默认空 = UI 兜底页）
    def inspect_pages(self, cfg: RunConfig) -> list[PageSpec]:
        return []

    @classmethod
    def summary_pages(cls) -> list[PageSpec]:
        """跨方法共享的检视页（UI 装配时自动追加，方法无需声明）。"""
        return [PageSpec("metrics", "指标", "table"),
                PageSpec("params", "参数", "table"),
                PageSpec("diag", "诊断", "text",
                         hint="开启「诊断」开关后重跑，此处显示中间产物")]

    # ------------------------------------------------ 工具
    def can_handle(self, spec: DataSpec) -> bool:
        if self.task == TASK_SUPERVISED:
            if spec.y is None:
                # ⭐ **例外：自定义任务**（`target_kind="custom"`）。
                #   为什么允许"没有标准标签"：有些任务的标签是**变长/复数**的
                #   （NQS：每个配置采样的 64 个复振幅），它住在 `payload` 里，
                #   标准指标对它**不成立**。若强行要求一个标签列，就只能造假标签
                #   （标准指标随之变成假话），或者这类任务根本做不了。
                #   ⇒ 只在**方法自己声明了消费负载**时放行（能力声明照旧兜底）。
                return (str(spec.target_kind or "") == "custom"
                        and bool(set(getattr(self, "accepts_aux", ()))
                                 & set(spec.aux or {})))
            if self.target_kind and spec.target_kind != self.target_kind:
                return False
            return True
        if self.task == TASK_TIMESERIES:
            return spec.y is not None
        return True

    def params(self, cfg: RunConfig) -> dict:
        return cfg.resolve(self.param_schema)

    def purposes_of(self) -> tuple:
        """显式声明优先；未声明则按 task+tags+family 推导。"""
        return tuple(self.purposes) or derive_purposes(
            self.task, self.target_kind, self.tags, self.family)

    def purposes_cn(self) -> str:
        return "/".join(PURPOSE_LABELS[p] for p in self.purposes_of())

    def _new_result(self, **kw) -> MLResult:
        kw.setdefault("method_name", self.name)
        kw.setdefault("task", self.task)
        kw.setdefault("primary_metric", default_primary(self.task, kw.get("target_kind")))
        return MLResult(**kw)


# ---------------------------------------------------------------- 指标
def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": _mape(y_true, y_pred),
    }


def _mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - np.asarray(y_pred, float)[mask])
                                / y_true[mask])) * 100)


def classification_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    average = "binary" if len(np.unique(y_true)) <= 2 else "macro"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def roc_metrics(y_true, y_prob, classes) -> dict:
    from sklearn.metrics import roc_auc_score
    try:
        if len(classes) <= 2:
            p = np.asarray(y_prob)
            p = p[:, 1] if p.ndim == 2 else p
            return {"auc": float(roc_auc_score(y_true, p))}
        return {"auc": float(roc_auc_score(y_true, y_prob, multi_class="ovr",
                                           average="macro"))}
    except Exception:
        return {"auc": float("nan")}


def default_primary(task: str, target_kind: Optional[str]) -> str:
    if task == TASK_SUPERVISED:
        return "rmse" if target_kind == "regression" else "f1"
    if task == TASK_CLUSTER:
        return "silhouette"
    if task == TASK_MANIFOLD:
        return "explained_variance"
    if task == TASK_ANOMALY:
        return "auc"
    if task == TASK_TIMESERIES:
        return "rmse"
    return ""


# ---------------------------------------------------------------- 默认 CV
def default_cv(method: MLMethod, X, y, cfg: RunConfig, cv_folds: int = 5,
               *, target_kind: Optional[str] = None,
               aux: Optional[dict] = None) -> dict:
    """k 折交叉验证（按 task 选折法），返回 {主指标: mean/std/scores}。

    ## 与 `Problem` 范式的关系

    `fit` 现在收 `Problem`，而 CV 的"折"**就是一次划分** ⇒ 每一折造一个只含该折
    划分的 `DataSpec`，再包成 `Problem` 交给 `fit` ⇒ 与 `run_one` 走同一条路。

    ⚠ **带辅助结构（图）的数据不走 CV**：CV 是"按行切片"，而图的边表是按行号写的
    ⇒ 切片会让边索引指到折外/失效，**静默失真**。所以这里直接拒绝（响亮），
    而不是算出一个看起来正常但错的分数。
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    if aux:
        raise ValueError(
            f"带辅助结构的数据不支持交叉验证（带了 {sorted(aux)}）："
            "CV 按行切片，而图的边表按行号索引 ⇒ 切片会让它静默失真。"
            "请用单次划分（problem.masks() 那套节点掩码），或去掉 cv_folds。")
    n = len(X)
    cv_folds = max(2, min(cv_folds, n))
    if method.task == TASK_SUPERVISED and method.target_kind != "regression" \
            and y is not None and pd.Series(y).nunique() > 1:
        try:
            splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                       random_state=cfg.seed)
            folds = list(splitter.split(X, y))
        except Exception:
            folds = list(KFold(cv_folds, shuffle=True,
                               random_state=cfg.seed).split(X))
    else:
        folds = list(KFold(cv_folds, shuffle=True,
                           random_state=cfg.seed).split(X))
    kind = target_kind or method.target_kind
    scores = []
    primary = ""
    for tr, te in folds:
        Xtr = (X.iloc[tr] if hasattr(X, "iloc") else X[tr])
        Xte = (X.iloc[te] if hasattr(X, "iloc") else X[te])
        ytr = (y.iloc[tr] if hasattr(y, "iloc") else y[tr])
        yte = (y.iloc[te] if hasattr(y, "iloc") else y[te])
        try:
            k = kind or infer_kind(ytr)
            # 折内：全量即训练（train_idx/test_idx 都指全量，方法只该用 rows("train")）
            fold_spec = DataSpec(
                X=Xtr.reset_index(drop=True),
                y=(ytr.reset_index(drop=True) if ytr is not None else None),
                train_idx=np.arange(len(Xtr)), test_idx=np.arange(len(Xtr)),
                target_kind=k)
            res = method.fit(method.problem(fold_spec), cfg, diag=False)
            if not res.ok:
                continue
            pred = method.predict(Xte, res)
            res.artifacts["y_true"] = np.asarray(yte)
            res.artifacts["y_pred"] = np.asarray(pred)
            m = method.evaluate(res, Xte, yte if hasattr(yte, "__len__") else yte)
            res.metrics.update(m)
            primary = primary or res.primary_metric
            if primary and primary in m:
                scores.append(float(m[primary]))
        except Exception:
            continue
    if not scores:
        return {"cv_" + (primary or "score"): float("nan")}
    arr = np.asarray(scores, float)
    return {f"cv_{primary}_mean": float(arr.mean()),
            f"cv_{primary}_std": float(arr.std()),
            "cv_scores": arr.tolist()}
