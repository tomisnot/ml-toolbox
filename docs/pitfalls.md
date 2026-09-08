# 踩坑清单（症状 → 原因 → 修法）

> 沿用 `D:\desktop\UI design\pitfalls.md` 的格式与纪律：新坑按此追加，
> 同类坑会复发。上游 17 条（Qt 离屏/matplotlib/pyqtgraph）依然适用，
> 本文件只记本项目新增的坑。

## LightGBM / OpenMP / Qt

### L1. LightGBM fit 时 access violation（读 0x0，无 Python 层原因）
- **症状**：`OSError: exception: access violation reading 0x0000000000000000`，
  栈顶在 `lightgbm/basic.py LGBM_DatasetSetField`；主线程、QThread 都会崩；
  `num_threads=1` 也救不了。
- **原因**：lightgbm 的 OpenMP 运行时与 Qt 冲突——**只要任何 PyQt5 模块先于
  lightgbm 被 import**，冲突必现。与是否构造 QApplication 无关，与线程无关。
- **修法**：入口脚本（app.py / checks/ui_shot.py）保证
  `from ml_toolbox.core import registry; registry.load_builtin()`
  （methods/ensemble.py 顶层 import lightgbm）**先于任何 PyQt5 import**。
  写代码时把这条当铁律：新增入口、新增 checks 脚本都照抄这个顺序。
- **验证**：`python tests/test_regressions.py::test_lightgbm_import_order`。

## sklearn API 版本差异（本机 1.6.1）

### L2. HistGradientBoosting* 没有 n_estimators
- **症状**：`TypeError: unexpected keyword 'n_estimators'`。
- **修法**：用 `max_iter`。param_schema 的 key 也要同步改。

### L3. BayesianRidge 没有 random_state，参数名是 max_iter
- **症状**：`TypeError: unexpected keyword 'random_state'`（1.6 起移除 n_iter）。
- **修法**：`BayesianRidge(max_iter=..., alpha_1=..., lambda_1=...)`，不传种子。

### L4. QDA 的正则参数叫 reg_param（不是 regularization / reg_covariance）
- **症状**：`TypeError: unexpected keyword 'reg_covariance'`。
- **修法**：`QuadraticDiscriminantAnalysis(reg_param=r)`，且 r 必须 > 0
  （传 0 抛 "regularization parameter must be positive"）。

### L5. SVR 没有 random_state / probability
- **修法**：分类 SVC 才有 probability+random_state；SVR 只传核参数。

### L6. LocalOutlierFactor 没有 score_samples（novelty=False 时）
- **症状**：`AttributeError: no attribute 'score_samples'`。
- **修法**：训练集内检测用 `decision_scores_` / `negative_outlier_factor_`；
  统一在 AnomalyMethod.fit 里按属性存在性取分数并翻方向（越大越异常）。

## 数据语义

### L7. MultinomialNB 要求非负特征
- **症状**：`ValueError: Negative values in data`——管道 StandardScaler 之后
  必然为负。
- **修法**：方法内部 MinMaxScaler 平移（fit 时重拟，predict 复用同一 scaler），
  不强迫用户改管道——方法族对数据的要求在方法内自洽。

### L8. 异常检测标签方向陷阱
- **症状**：AUC=0.23（比随机还差）但代码"没报错"。
- **原因**：sklearn `fit_predict` 返回 1=正常、-1=异常，与直觉相反；
  `decision_scores_` 越低越异常。
- **修法**：统一约定"分数越大越异常、label 1=异常"，在 AnomalyMethod.fit
  一处翻转；LOF 在密集偏移簇上失效是**真实局限**（局部密度法看不出整簇平移），
  不是 bug——对比视图里它就该垫底。

## UI 装配

### L9. 检视页兜底必须绑定绘图函数
- **症状**：分类方法截图里"混淆矩阵"页只有占位文字。
- **原因**：runner.auto_pages 造 PageSpec 时没传 plot=。
- **修法**：兜底页与声明页走同一批 plots.* 函数。

### L10. 截图脚本：状态切换必须发生在截图 callable 内部
- **症状**：compare/chain 两张 PNG 内容相同（都停在检视页）。
- **原因**：capture_list 构建列表时切了 tab，但主循环执行 grab 时 tab 已被
  后续循环改回。
- **修法**：每个 shot 的 callable 自己 prep（切页 + processEvents + 同步 draw）
  再 grab。

### L11. matplotlib 中文轴标签渲染成方框
- **症状**：Qt 界面中文正常，唯独 mpl 画布里的"真实/预测"是方框。
- **原因**：matplotlib 的字体管理与 Qt 独立，默认 DejaVu Sans 无 CJK。
- **修法**：`rcParams["font.sans-serif"] = ["Microsoft YaHei", ...]`
  （ui/widgets.py 顶部，import matplotlib 后立即设）。

### L12. pyqtgraph 0.13.7：PlotItem 没有 .scatter()
- **症状**：`AttributeError: 'PlotItem' object has no attribute 'scatter'`。
- **原因**：`plotItem.scatter(...)` 便捷方法在部分版本不存在。
- **修法**：用 `p.addItem(pg.ScatterPlotItem(x=, y=, brush=pg.mkBrush(...)))`
  通用写法（ui/widgets.py PGScatter）。

### L13. 整数标签喂给 mpl `c=` 落在色带中段（发灰）
- **症状**：聚类散点颜色寡淡，0/1/2 渲染成三种灰蓝。
- **原因**：`c=labels + cmap="tab10"` 把整数当连续值归一化，落在 ListedColormap
  前几格（本就偏灰）。
- **修法**：离散标签显式取 `cmap.colors[i % len]` 调色板色（plots._discrete_colors）。

### L14. 主窗口加 tab 后硬编码索引全部错位
- **症状**：加"核心图对比"页后，"点方法跳检视"跳错页。
- **修法**：tab 索引集中管理（当前：0 处理链/1 对比/2 画廊/3 检视），
  改布局时全局搜 `setCurrentIndex(` 逐处核对；截图脚本同步跟改。
- **根治**：main_window 用 `indexOf(widget)` 存成 `self.TAB_*` 常量，
  代码里不再出现魔法数字（本轮已加"数据检视"页，索引再次右移，靠常量免疫）。

### L15. pyqtgraph ImageItem 朝向： setImage(A) 已是 row0 在底部
- **症状**：热图上下镜像（相关矩阵 r=+1 的对角线跑到反对角），无报错。
- **原因**：上游 UI design 模板用 `setImage(A.T)` 是针对"row0 在顶部"的旧版行为；
  本环境 pyqtgraph 0.13.7 + `axisOrder='row-major'` 下，setImage(A) 已使
  A[a,b] 呈现为 a=纵轴向上（row0 在底）、b=横轴——再 .T 反而抵消成转置。
- **修法**：`setImage(A)` 不转置；朝向是**静默 bug**，只有像素级探色能抓——
  已固化为 `tests/test_ui.py::test_heatmap_orientation`（非对称矩阵逐格验色）。

### L16. 编辑工具偶发"成功"但未落盘
- **症状**：Edit 返回 success，但随后 Read/运行仍是旧内容（本项目多次遇到）。
- **应对**：关键修复后**重跑对应测试**验证，而非信任编辑回执；
  发现未生效就重新应用（往往第二次即落盘）。

### L17. Qt 信号载荷污染业务参数（上游 pitfalls #5 复发）
- **症状**：方法库选"用途=代理模型"后列表全空。
- **原因**：`currentTextChanged.connect(self._refilter)` 把用途中文名当作
  `text` 搜索词传入，任何方法名都不含"代理模型"→ 全被搜索过滤掉。
- **修法**：`connect(lambda *_: self._refilter(self._search.text()))`——
  槽函数有业务参数时，接线必须包 lambda。

### L18. 元学习器嵌套并行导致偶发 access violation
- **症状**：smoke 遍历里 stacking 偶发失败（重跑即过），无稳定 traceback。
- **原因**：Voting/Stacking 外层 `n_jobs=-1` × 成员模型（RF/XGB）`n_jobs=-1`
  = 嵌套超订，Windows OpenMP 线程池互相挤兑。
- **修法**：元学习器外层用 `NJOBS`（默认 1，成员已并行），
  外层并行需显式设 `MLTB_NJOBS=-1`。

### L19. est.classes_ 不是原始标签
- **症状**：新实例 predict 返回 0/1/2 而非原始字符串标签。
- **原因**：`LabelEncoder` 先编码 y，estimator 的 `classes_` 是编码后的整数；
  想跨实例还原标签必须存 `_le.classes_`（原始类别）。
- **修法**：`result.classes_ = self._le.classes_`（fit 时随 result 携带）。
  注意每个自实现 predict 的方法族都要走这条约定（TorchMLP 曾漏，
  已补 `test_neural_fresh_predict_string_labels`）。

## 神经网络（方案 C）

### N1. 初始权重快照时机：构造时 vs 首 epoch 后
- **症状**：权重热图切"初始权重 / ΔW"视图恒显示"该视图无快照"。
- **原因**：`TrainingRecorder.__init__` 里 `self._weights_first = {}` 留空，
  指望 epoch 循环里补快照——但补的时候训练已更新过，语义已不是"初始"。
- **修法**：构造时（fit 里 model 刚建、训练未开始）立即
  `{n: p.detach().cpu().numpy().copy() for ...}`。回归断言见
  `test_neural_fit_and_artifacts`（first 全部非 None）。

### N2. 逐层健康表把所有层的"死 ReLU"填成同一全局均值
- **症状**：nn_layers 表每行死 ReLU 比例完全相同，看不出哪层死了。
- **原因**：`history["dead_relu"][-1]` 是跨层平均，`for _ in last` 逐行复制。
- **修法**：hook 按遍历顺序收集**逐层**值存 `_dead_per_relu`；
  Sequential 里 `Linear.weight` 名是 `0.weight/2.weight/4.weight`，
  ReLU 数 = 层数-1，按 `idx//2` 对齐，末层填 NaN。

### N3. 嵌套 dict / 多列 DataFrame 工件存盘丢失
- **症状**：历史面板回看 torch_mlp 运行，权重热图/回放页全变占位。
- **原因**：`save_record` 只收顶层 `isinstance(v, np.ndarray)`，
  `nn_weights`（dict of dict）整体被丢；`nn_layers` 只存了行名+展平值，
  多列信息读回时塌成一列。
- **修法**：dict 按 `父__子__叶` 展平存 npz，`load_run_record` 递归重组；
  DataFrame 列数 >1 时额外存 `<key>_cols`，读回按 (行,列) reshape。
  断言见 `test_neural_persistence_nested`。

### N4. 回放页类纯度读数恒定不变（NaN 判断写反）
- **症状**：`self._labels is None` 写在 `np.asarray(None)` 之后，
  object 数组比较抛异常被吞，纯度恒显示 NaN 或崩溃。
- **修法**：先判 None，再判 `lab.dtype == object`；NaN 用 `purity == purity` 过滤。

### N5. AdaptiveMatrixHeatmap 信号竞态：未 set_data 就收到 rangeChanged
- **症状**：UI 测试间歇打印 `AttributeError: no attribute '_names_c'`
  （页面 deleteLater 与 Qt 事件循环交错时触发，不崩但污染日志）。
- **修法**：`__init__` 里预置 `_names_c = None; _rows = _cols = 0`，
  `_ticks` 对空值早退。

### N6. torch 依赖不能拖垮整包注册
- **约定**：`methods/__init__.py` 对 neural 子包单独 `try/except`；
  `mlp.py` 内 torch 全部惰性 import（fit 里才 import）。
  未装 torch 的环境其余 52 方法照常注册；测试用 `_neural_available()` 跳过。
