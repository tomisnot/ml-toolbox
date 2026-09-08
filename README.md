# ML Toolbox · 机器学习方法试验台

一个**可被其他项目直接 import** 的机器学习工具箱，也是桌面软件：把"哪个方法好用"
这件无法提前预测的事，变成"在统一界面里低成本遍历尝试"的工程流程。

- 53 个方法 / 11 族（含神经网络 MLP，方案 C：训练过程可视化）
- 按**建模目的**（预测/判别/归因/分群/降维/综合评价/异常/代理模型/基线）筛选
- 每个方法声明自己的核心图，UI 基类不预设图长什么样
- 一份数据 + 一条可检视管道，全方法共享；实验自动存档可回看
- 数学建模基准实战验证：遍历结果落在已知答案附近

```powershell
pip install -r requirements.txt
python app.py            # 启动界面
python tests/run_all.py  # 质量门（smoke + 18 回归 + 14 UI）
```

> 神经网络方法族（`torch_mlp`）需 `pip install torch`；未安装时其余 52 方法照常可用。

---

## 一、架构

```
ml_toolbox/
├── core/          零 GUI 依赖（已用 grep 核实）
│   ├── contracts.py    MLMethod 基类 / RunConfig / MLResult / PageSpec / ParamSpec
│   ├── dataset.py      Dataset —— 单一数据入口
│   ├── pipeline.py     管道步骤链（缺失值/编码/缩放/筛选/划分）+ transform_new
│   ├── registry.py     @register 注册表（遍历候选集来源）
│   ├── runner.py       run_one / run_batch / compare_table / auto_pages
│   ├── persistence.py  runs/<id>/{record.json, artifacts.npz}
│   └── demo.py         8 个离线演示数据集
├── methods/       11 个方法族 + plots.py（纯 matplotlib 绘图函数）
│   └── neural/    神经网络（方案 C）：recorder 黑匣子 + TorchMLP + nnplots
├── ui/            PyQt5 界面（唯一有 Qt 依赖的层）
│   ├── main_window.py     主窗口装配
│   ├── inspector.py       检视页基类（按 PageSpec 动态装配 + 页类型注册表）
│   ├── neural_pages.py    神经网络自定义页（权重热图 / 表示演化回放）
│   ├── gallery.py         核心图并排缩略（遍历的视觉回报）
│   ├── chain_page.py      处理链大卡片
│   ├── data_page.py       数据检视（特征相关热图）
│   ├── heatmap.py         pyqtgraph 自适应矩阵热图
│   ├── param_panel.py     旋钮面板（空=默认，就地重跑）
│   ├── method_browser.py  方法库（族分组 + 用途筛选 + 搜索）
│   ├── history.py         实验历史回看
│   └── worker.py          后台线程
├── tests/         smoke / 回归 / UI / run_all（一键门）
├── benchmarks/    数学建模基准（结果表在 benchmarks/results/）
├── checks/        离屏截图自查闭环
└── docs/          项目定位.md（宪法）· 契约.md · pitfalls.md（25 条）
```

**依赖方向单向**：`ui → core + methods`，`methods → core`，`core → 无内部依赖`。
core 不 import 任何 GUI；methods 只在绘图函数**内部**惰性 import matplotlib，
模块导入本身不碰 GUI（smoke 测试在无显示环境下全过即为证据）。

## 二、后端：高效 / 自由 / 可独立引用

### 高效

| 机制 | 效果 |
|---|---|
| 遍历在后台线程跑（`worker.py`） | 界面不冻结，进度逐方法回报 |
| 惰性 import（sklearn/xgboost 等在 `make()` 里） | 启动快，未选的方法不加载 |
| 元学习器外层 `NJOBS=1`，成员模型 `n_jobs=-1` | 避免 Windows OpenMP 嵌套超订（曾致偶发崩溃） |
| 诊断/交叉验证默认关 | 零侵入：不开时数值路径与耗时不变 |
| 检视页只在变化时重建 | 切换方法不重复构造画布 |

已知性能边界（诚实列出）：GPR/核方法 O(n³)（>3000 样本慎用）、t-SNE/UMAP 万级样本
偏慢、画廊 >20 缩略图构建偏慢。这些是方法本身复杂度，非框架缺陷。

### 自由（UI 基类高自由度）

方法用 `PageSpec` 声明自己的检视页，UI 只负责装配：

```python
PageSpec(key="cm", title="混淆矩阵", kind="mpl", plot=my_fn)  # plot(ax, result)
PageSpec(key="emb_pg", title="交互缩放", kind="pg")           # 交给 pyqtgraph
PageSpec(key="tbl", title="系数表", kind="table")             # 消费 DataFrame 工件
```

- `kind` 决定渲染方式；方法层**零 Qt**（plot 是纯 matplotlib 函数）；
- 方法没声明 → `runner.auto_pages` 按 task 兜底，UI 永远不空白；
- 工件缺失 → 显示 `hint` 语义化占位，不让用户猜原因；
- **新页类型可注册**：`inspector.register_page_builder(kind, builder)` 挂自定义
  QWidget（神经网络的权重热图 `nn_weights`、表示演化回放 `nn_replay` 即如此接入）——
  方法层只声明 kind + 把数据装进 artifacts，渲染归 UI 层，框架零改动。

### 可独立引用（不开 GUI 完成全部工作）

```python
import pandas as pd
from ml_toolbox.core import registry, runner
from ml_toolbox.core.dataset import Dataset
from ml_toolbox.core.pipeline import Pipeline

registry.load_builtin()
pipe = Pipeline.default()
spec = pipe.run(Dataset.load("train.csv", target="y"))

# 遍历全部适用方法，按主指标排序
recs = runner.run_batch([m.name for m in registry.for_spec(spec)], spec)
best = runner.compare_table(recs).iloc[0]

# 用最优方法预测新数据（同管道变换；未见类别/缺失值自动处理）
rec = runner.run_one(registry.get(best["method"]), spec)
Xte = pipe.transform_new(pd.read_csv("test.csv"))
pred = registry.get(best["method"]).predict(Xte, rec.result)
```

其他独立入口：`registry.for_spec(spec)` 取候选集；`method.cross_validate(X, y, cfg, cv=5)`
拿 k 折结果；`persistence.list_records()` 读历史；`Dataset.from_arrays(X, y)` 从内存构造。

## 三、方法总览（按建模目的）

| 目的 | 数量 | 方法 |
|---|---|---|
| 预测 | 37 | 全部回归 + 时序 + 分类器（可预测类别，含 torch_mlp） |
| 判别 | 21 | logistic, lda, qda, svc, linear_svc, nusvc, knn, 朴素贝叶斯×3, 树/集成×8, voting, stacking, torch_mlp, dummy |
| 归因解释 | 13 | linear_regression, ridge, lasso, elasticnet, bayesian_ridge, huber, theilsen, logistic, linear_svc, lda, random_forest, extra_trees, gpr, arima, ts_trend |
| 分群 | 6 | kmeans, gmm, dbscan, hdbscan, agglomerative, spectral |
| 降维 | 6 | pca, kernel_pca, lda_proj, tsne, umap, isomap |
| 综合评价 | 2 | pca, lda_proj（方差/判别权重可作打分依据） |
| 异常检测 | 4 | iforest, lof, ocsvm, mahalanobis |
| 代理模型 | 2 | gpr, bayesian_ridge（含不确定度，可喂优化算法） |
| 参照基线 | 1 | dummy（任何模型都应赢它） |

> 目的由 `task + tags + family` 推导，方法可用 ClassVar `purposes` 显式覆盖。
> 一个方法可挂多个目的（如 lasso = 预测 + 归因解释）。

按族：linear 11 · ensemble 9 · cluster 6 · manifold 6 · timeseries 7 · bayes 4 ·
anomaly 4 · svm 3 · knn 1 · baseline 1 · neural 1。

## 四、界面速览

```
┌ 工具栏  打开数据 · 演示数据 · 管道设置 · 历史 · 对新数据预测 · 诊断 · 交叉验证 · 种子 ┐
│ 方法库    │ 处理链 │ 数据检视 │ 对比视图 │ 核心图对比 │ 方法检视 │  参数面板      │
│ 族分组    │ 大卡片 │ 相关热图 │ 指标表   │ 缩略网格   │ 动态分页 │  旋钮+⟳重跑   │
│ 用途筛选  │ 每步   │ 缩放平移 │ cv 列    │ 按主指标   │ mpl/pg   │  空=默认      │
│ 搜索      │ 摘要   │          │          │ 排序       │ table    │               │
└ 底部  ▶ 遍历运行所选方法（进度回报）───────────────────────────────────────────┘
```

五页对应数模工作流：**看数据**（处理链/数据检视）→ **选模型**（遍历+对比+画廊）→
**看细节**（方法检视）→ **调参**（右栏就地重跑）→ **出结果**（对新数据预测导出）。

### 神经网络检视页（方案 C：训练过程可视化）

`torch_mlp` 跑完后方法检视有 5 张专属页，回答"中间发生了什么"：

| 页 | 形态 | 看到什么 |
|---|---|---|
| 训练动态 | matplotlib | loss/val/lr/梯度范数四线 + 过拟合拐点 |
| 权重热图 | pyqtgraph 交互 | 逐层 W 矩阵，滚轮缩放；切"最终/初始/ΔW"看训练改了什么 |
| 表示演化 | 滑块+播放动画 | 隐层→PCA 2D 散点逐 epoch 回放，实时读"类纯度"，右附 loss 游标 |
| 特征归因 | matplotlib | 积分梯度条形图（红推高/蓝压低），诊断开启时算 |
| 逐层健康 | 表格 | 参数量/梯度范数/权重范数/死 ReLU 比例 |

## 五、质量保障

| 命令 | 内容 |
|---|---|
| `python tests/run_all.py` | 一键门：smoke（53 方法 × 6 任务）+ 18 回归 + 14 UI |
| `python checks/ui_shot.py` | 离屏渲染真实界面 → 47 张 PNG → 按 checklist 视觉自查 |
| `python benchmarks/run_benchmarks.py` | 数模基准（Iris/Wine/Housing/blobs/异常/时序/digits） |

回归门覆盖的"静默 bug"（只有测试能抓到的那类）：热图朝向像素级验色、异常分数方向、
标签编码跨实例、时间切分不泄漏未来、CV 零开销、参数三态、Qt 信号载荷污染、
神经网络初始权重快照/嵌套工件存盘重组。

## 六、已知限制

- 神经网络仅 MLP（全连接，分类/回归）；CNN/RNN/Transformer 等结构未覆盖，
  但方案 C 的 recorder + 注册页机制可直接复用；torch 为可选依赖，未装时该族跳过；
- **自动调参 / 序贯优化框架规划中（未实现）**——与 ML 工具箱"兄弟框架 + 共享内核 +
  三接缝互通"，定位与开发规划见 `docs/优化定位.md`；数模的代理模型路径当前仍是
  `gpr/bayesian_ridge` 出响应面，再交给外部优化算法（scipy.optimize / 自写遗传算法）；
- 时序方法当前只消费 y 序列，**忽略外生特征 X**（`can_handle` 已要求有 y）；
- 单次运行内 estimator 驻留内存，不支持断点续训；
- Windows 优先验证（LightGBM×Qt 导入顺序铁律见 `docs/pitfalls.md` L1）。

## 七、扩展：加一个方法

```python
# ml_toolbox/methods/myfamily.py
from ..core.contracts import MLMethod, ParamSpec, PageSpec
from ..core.registry import register

@register
class MyMethod(MLMethod):
    name = "my_method"; display_name = "我的方法"; family = "myfamily"
    tags = ("fast",)                      # purposes 自动推导
    param_schema = [ParamSpec("k", label="K", kind="int", default=3, min=1)]

    def fit(self, X, y, cfg, diag=False):
        p = self.params(cfg)              # 自动清洗 UI 覆写
        ...
        return self._new_result(metrics={...}, artifacts={...}, params=p)

    def inspect_pages(self, cfg):         # 可选：声明专属核心图
        return [PageSpec("my", "我的图", "mpl", lambda ax, r: ax.plot(...))]
```

在 `methods/__init__.py` 加一行 import 即完成接入——注册表、遍历、对比、画廊、
参数面板、导出、历史全部自动可用，无需改框架任何代码。

## 八、文档

- `docs/项目定位.md` —— 项目宪法（定位/目标/原则）与迭代状态
- `docs/契约.md` —— 方法 / 数据 / 可视化 / 持久化四份契约
- `docs/pitfalls.md` —— 19 条踩坑（症状→原因→修法），新坑按格式追加
