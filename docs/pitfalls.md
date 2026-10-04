# 踩坑清单（症状 → 原因 → 修法）

> 沿用外部 UI 设计资产的 `pitfalls.md` 的格式与纪律：新坑按此追加，
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
- **修法**：元学习器外层用 `core.parallel.meta_nj()`（默认 1，成员已并行），
  外层并行需显式设 `MLTB_NJOBS=-1`。另一半见 L20（受限环境并行度不可控）。

### L19. est.classes_ 不是原始标签
- **症状**：新实例 predict 返回 0/1/2 而非原始字符串标签。
- **原因**：`LabelEncoder` 先编码 y，estimator 的 `classes_` 是编码后的整数；
  想跨实例还原标签必须存 `_le.classes_`（原始类别）。
- **修法**：`result.classes_ = self._le.classes_`（fit 时随 result 携带）。
  注意每个自实现 predict 的方法族都要走这条约定（TorchMLP 曾漏，
  已补 `test_neural_fresh_predict_string_labels`）。

### L20. 受限环境下 `n_jobs=-1` 触发 joblib/loky 子进程探测 → 整批方法失败
- **症状**：沙箱/受限 CI 里 `PermissionError: [WinError 5]`（random_forest、
  extra_trees、hist_gb、voting、stacking、iforest 等成片失败）；`umap` 直接
  卡死（进程活着、CPU 不涨）。
- **原因**：sklearn/umap 的并行后端用 `subprocess` 探测物理核数，禁止子进程的
  环境里被拒；loky worker 反复重启表现为挂起。此前并行度硬编码在各方法里，
  外部无法统一降级（= 架构审视 M5）。
- **修法**：并行度收编到 `core.parallel`（C5）——`MLTB_NJOBS=off` 时成员模型
  走 `nj()`=1、LightGBM 显式 `n_jobs=1` 跳过 joblib、numba 线程 import 即钉 1；
  测试侧 `MLTB_SKIP_METHODS=umap,...` 按环境豁免（记 SKIP 不记失败）。
  验证：`MLTB_NJOBS=off python tests/smoke_test.py` 全绿。

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

## 优化框架（阶段 0-4）

### O1. 种群式算法拆 ask/tell：ask 返回当前种群 = 重复评估，子代永远没被评
- **症状**：NSGA-II 在 ZDT1 上前沿偏差 4.5，f0 覆盖 [0.03, 0.6]，怎么加预算都不改善。
- **原因**：ask 返回 `self.P`（父种群）→ runner 评估父种群 → tell 里繁殖子代 Q
  但 Q 从未返回给 runner 评估。每代都在重评同一批父代。
- **修法**：引入"待评缓冲" `self._to_eval`——初代 = P，之后每代 tell 结束时
  `_to_eval = _breed()`，ask 永远返回 `_to_eval`。
  断言见 `test_nsga_ii_pareto`（前沿贴合 f2=1-√f1，偏差 <0.35）。

### O2. 多项式变异步长按"位置"选边界距离，应按"移动方向"
- **症状**：NSGA-II 收敛慢、种群堆在中间区域。
- **原因**：`width = where(x<0.5, x, 1-x)` 是"到较近边界"；delta<0（向左移）
  时 x=0.8 会拿到 0.2 的步长上限（应 0.8），向右同理——变异被系统性压向中心。
- **修法**：`dist = where(delta<0, x, 1-x)`（向左最多到 0，向右最多到 1）。

### O3. ParamSpace 有界规则漏了 int：半无界旋钮采样直接崩
- **症状**：AutoTuner 调 knn 报 `TypeError: int() argument ... not 'NoneType'`。
- **原因**：knn 的 `leaf_size` 是 `int` 且 `max=None`；排除规则只写了
  `p.kind == "number"` 的无界检查，`from_vector` 里 `int(p.max)` 炸。
- **修法**：`unbounded = p.kind in ("number","int") and (p.min is None or p.max is None)`
  统一固定透传。数模语义上也对：leaf_size 这类性能旋钮不该进寻优空间。

### O4. 评估历史回流 ML：分数列被 _infer_kind 误判为分类
- **症状**：`response_surface` 的 R² = -4e24（数值爆炸）。
- **原因**：CV f1 只有少数几个不同值 → `_infer_kind` 判成分类 → 评估走
  accuracy 路径、分层切分抛"least populated class has only 1 member"。
- **修法**：`_quick_spec` 强制 `spec.target_kind="regression"`。
  另注意：响应面只对**全域采样**（random_search）的历史有意义——GP-BO 的点
  集中在最优点附近，分数方差 ~1e-5，任何回归 R² 都失真（这是数学事实不是 bug）。

### O5. batch 引擎的预算边界：ask 一代 8 个但预算只剩 3
- **症状**：`Budget(n_evals=20)` 跑 CMA-ES 实际评估 24 次。
- **修法**：runner 在**评估前**截断 `plist = plist[:max(room,1)]`——
  objective 调用数严格 ≤ n_evals（评估后才截断等于超调黑盒）。
  断言见 `test_batch_budget_respected`。

### O6. 采集函数签名不统一：ei() 收到 beta 关键字崩
- **症状**：GP-BO 配 ucb 时 `TypeError: unexpected keyword 'beta'`。
- **修法**：runner 按采集函数名分发各自 kwargs；ucb 签名加 `best=None`
  占位统一三函数调用式。教训：注册表式的函数集合要约定统一签名。

### O7. optuna 适配层的 tell 语义：失败观测必须用 state=FAIL
- **症状**：objective 抛异常时 study 里记了个 inf 分数，污染 TPE 的先验模型。
- **修法**：`study.tell(trial, state=TrialState.FAIL)`，不喂数值。

### O8. 顺序跑多优化器时，超参面板误套所有优化器
- **症状**：UI 里面板显示 gp_bo 的采集函数下拉，跑 random_search 时也把它
  的 cfg 传给 random_search（无该参数，静默忽略但用户以为生效了）。
- **修法**：`_param_owner` 记录面板归属，`_collect_opt_cfg(name)` 只对
  当前 owner 返回覆写，其余优化器用默认。

### O9. QToolBar 内嵌 widget 直接 setVisible(False) 会被布局重新 show
- **症状**：Perspective 切到优化模式后，`_cv.setVisible(False)` 调了，
  但 `win._cv.isVisible()` 仍 True（测试 perspective_switch 抓到）。
- **原因**：QToolBar 在布局/重排时会对自己包的那层 widget 重新 show；
  你 hide 的是内层 widget，外层包装 QAction 仍是 visible。
- **修法**：隐藏 `tb.addWidget(w)` **返回的 QAction**（`act.setVisible(False)`），
  不是隐藏 w 本身。所有按模式显隐的工具栏控件都要存这个返回值
  （main_window `_ml_only_actions`）。

### O10. 库函数不能要求调用方先做注册（bridges 里 registry.get 抛空表）
- **症状**：test_opt 里 `AutoTunerObjective('logistic', ...)` 报
  `未注册的方法: logistic（可用: []）`——core.registry 从未 load_builtin。
- **原因**：UI 启动路径会 load_builtin，但纯后端调用者（脚本/测试/其他项目）
  不会；bridges 假设了"ML 注册表已加载"这个隐式前提。
- **修法**：`method_param_space` / `AutoTunerObjective.__init__` 里
  `ml_registry.load_builtin()`（幂等）。注册表 load 是廉价且可重入的。

### O11. 数据源接入别把"数据侧"和"评估侧"混成一个概念
- **症状**：设计评审时把 AutoTuner 说成"依赖 ML 工作区的数据上下文"，
  被用户指出：真实主场景是接入正在运行的外部程序。
- **辨析**：① 数据侧（程序持续产数）= `FileSource`，评估仍是训练+CV；
  ② 评估侧（程序就是黑盒）= `ProcessObjective`，与 ML 无关。
  两者正交，UI 上也分开（数据源组 vs 目标类型下拉），别混。
- **教训**：UI 里的隐式依赖（`_spec` 注入）要显式化为可选项 + 状态栏，
  否则"能跑通"掩盖了"绑错了假设"。

## 原子模拟实战彩排（ProcessObjective + 约束）

### P1. Windows 子进程 stdout 编码：GBK 管道炸 UTF-8 中文
- **症状**：模拟软件输出中文，`subprocess.run(text=True)` 在中文 Windows
  上按 GBK 解码 UTF-8 字节 -> `UnicodeDecodeError`，整条评估链崩。
- **修法**：`capture_output` 拿 bytes，手动 utf-8 解码、失败再 GBK 兜底
  （`ProcessObjective._dec`）。跨平台黑盒的 stdout 编码**不能假设**。

### P2. 薄可行带 + censored = GP-BO 静默退化成随机搜索
- **症状**：彩排里约束违反占 90%+，GP-BO 的 best 与随机搜索**一模一样**
  （0.2610=0.2610）——不报错、不崩溃，只是"智能"没了。
- **原因**：失败点不进目标 GP（censored），可行点太稀（<n_init）GP 永远
  学不动，全程停在随机预热。
- **修法**：带约束黑盒要么 `on_infeasible='penalize'`（全量点喂 GP，
  薄带下实测最强），要么 cEI（见 P3）。**验收判据必须含"GP-BO 优于随机"，
  否则静默退化测不出来。**

### P3. cEI 的 warmup 死锁：可行点凑不够 n_init 就永远随机
- **症状**：GP-BO 开 `constrain` 后 best 仍等于随机搜索。
- **原因**：`ask()` 门槛 `len(self._X) < n_init` 只数**可行**观测；
  薄可行带下可行点永远 < n_init，采集优化永不启动。
- **修法**：约束模式就绪判据改为"目标 GP ≥2 可行点 + 分类器两类都见过"
  （`len(_Xc)>=4 and len(set(_yc))>=2`），不等可行点凑数。
  回归断言：`test_constrained_process_objective` 验 `opt._gpc is not None`。

### P4. 固定惩罚值有尺度隐患：损失量级不同就失效
- **症状**：`PENALTY=10` 在损失域 0~3 的假黑盒上有效；换成损失域 1e3 的
  真问题，惩罚反而"优于"很多可行点，GP 被带偏。
- **修法**：自适应惩罚 = 历史最差可行分 × `penalty_mult`（
  `ProcessObjective._penalty`），随问题尺度自动缩放。

### P5. 彩排的价值：在假黑盒上把真黑盒的坑踩掉
- 假黑盒故意复刻真实工况（中文输出/科学计数法/硬约束拒绝/偶发崩溃/噪声），
  彩排期暴露 P1-P4；若直接上真黑盒，这些会以"调参没效果"的形式浪费实战预算。
- **纪律**：接任何新黑盒，先 5 次评估预跑验"调用通/解析对/约束识别对"，
  再放开预算（见 docs/原子模拟对接.md §4 步骤 7）。

### P6. GUI 静默消失：无日志可查 + 关窗销毁运行中 QThread + 黑盒子进程成孤儿
- **症状**：实战跑 freeze（workers=8 并行子进程）约 24 分钟后 GUI 整个消失，
  无 WER 崩溃转储、无 python.exe 错误事件、无系统重启/睡眠记录；
  `%TEMP%\optb_*` 只剩 pts.json 无 res.json（子进程被连带杀死）。
- **根因链**：① 工具箱此前**完全不写运行日志**，"分析一下"无从下手；
  ② MainWindow 无 `closeEvent`，关窗/被外部关闭时 OptWorker(QThread) 仍在跑
  即被销毁 → Qt5Core fail-fast `0xc0000409`（与项目记忆 §5.3 事故同机制）；
  ③ 黑盒子进程用 `subprocess.run` 挂在 GUI 进程组下，父死子亡且无清理钩子。
- **修法**：① app.py 起 `logs/ml_toolbox.log` + `faulthandler.enable(file=)`
  把 native 崩溃栈也落盘；process.py 每个黑盒点启动/完成/失败带 pid 打日志；
  ② `closeEvent` → `OptWorkbench.shutdown()`：request_stop + `kill_inflight()`
  （taskkill /T 整树）+ `wait(8000)`；③ `_run_point` 改 Popen 登记进
  `self._procs`，提供 `kill_all()`；④ `_on_eval` 每 4 点存一次 runs/ 检查点，
  中途死也不丢已完成评估。
- **教训**：昂贵黑盒实战（一次运行几十分钟）必须**先有可观测性再谈优化**——
  日志、检查点、进程清理三件套是实战刚需，不是锦上添花。

## 量子启发方法（教材《量子机器学习》对接）

### Q1. 量子编码约定漂移：核值算错但方法不报错
- **症状**：qsvc/qkrr 能跑通、指标也正常，但换特征顺序 / 改 qsim 门序后
  保真度核 k(x,z) 悄悄变了，结果对不上教材手算——因为核是内部量，无外部校验点。
- **原因**：振幅/角度编码有一堆约定（qubit 位序、Ry 角度尺度、是否补零、
  归一化方式），任一漂移都改变 |⟨φ(x)|φ(z)⟩|²，而 Python 层不抛异常。
- **修法**：`tests/test_regressions.py::test_quantum_feature_map_analytic_kernel`
  钉死 n=1 锚点：单量子比特角度编码的保真度核必等于解析 `cos²((x−z)/2)`
  （书式 3.79），并断言核 PSD（式 3.70）。改编码前先看这条会不会红。

### Q2. 振幅编码丢范数：qkmeans 是方向聚类，不是欧氏聚类
- **症状**：qkmeans 在标准 blobs（各向同性、离散）上 silhouette 明显低于 kmeans。
- **原因**：振幅编码 |x⟩=x/‖x‖ 只保方向（书 §3.1.2/例 7.1），保真度距离
  1−|⟨x|c⟩|² 与向量长度无关——blobs 的簇信息在位置上不在方向上，故弱。
- **修法**：这不是 bug，是量子可区分度聚类的**真实适用边界**（对齐 L8/LOF）；
  方向可分数据（过原点射线簇）才是 fidelity 模式主场，其余用 `distance=euclid` 对照。
  hint 里写明“只看方向”，对比视图里它在该垫底就垫底。

### Q3. 有限 shots 的核矩阵破坏对称/PSD，且核值不随样本数免费
- **症状**：shots>0 时反演测试采样出的 K 轻微不对称，SVC 偶发不收敛警告。
- **原因**：逐元素独立 Bernoulli 采样 k̂=Binom(S,k)/S 破坏对称性；核矩阵有 m(m+1)/2
  个独立元素（式 3.77–3.78 的联合界），采样成本随 m² 增长，不因希尔伯特空间
  高维而消失。
- **修法**：`qsim.fidelity_kernel` 采样后 `(K+Kᵀ)/2` 对称化、对角强置 1（自身重叠无噪声）；
  UI 默认 shots=0（理想核），shots 仅用于演示有限测量涨落，不当加速宣称。

## AI 调用面（工具 / 查询 / 任务）

### A1. 逐 epoch 进度钩子挂晚了：`RunRequest.config()` 是**一次性拷贝**
- **症状**：后台 `submit_run` 跑的神经网络训练，`read_job` 的 `progress` **恒为空**
  （只有批次级 `run_method_batch` 那一条有）；而**同步** `run_method` 看起来一切正常。
- **原因**：钩子先挂在 `engine._execute_one` 里（往 `request.extras` 塞
  `on_progress`），但 `Session.run` 一进来就调 `request.config()`，而
  ``RunRequest.config()`` 是 `extras=dict(self.extras)` 的**快照**
  （`api.py:89-94`）⇒ 之后塞的东西**传不到 cfg**，方法层永远收不到。
  **"同步看着正常"是探针的假绿**：直接调 `_execute_one` 的那条路绕开了命令面，
  只有真 job 才暴露——正好复现"人看得见、AI 看不见"。
- **修法**：在三个命令体里、`_build_request` 之后立刻挂（`_attach_progress`），
  永远早于任何 `session.run`。判据：
  `test_progress_hook_must_be_attached_before_session_run`（机制，直接断言
  `config()` 认得出钩子）＋ `test_per_epoch_progress_reaches_read_job`（走真 job）。
  **教训**：加钩子类能力时，判据必须走**最严的那条真实路径**，不能只验"我以为挂上了"。

### A2. `SplitStep` 摘要里的 `test_size` **不决定**划分
- **症状**：按界面/工具给的 `test = 25%` 去复现，实际划分却不是那个比例
  （像模像样地错，且没人报错）。
- **原因**：`SplitStep.apply` 把 `self.params.get("test_size", 0.25)` 写进摘要，
  而真正划分用的是 `Pipeline.test_size`（`pipeline.py::_split`）⇒ 两个来源，
  步骤参数改了摘要会跟着变、划分却不变。
- **修法**：展示层**剔除**那个数（`engine._chain_summary`），真值由
  `dataset_detail.split_policy`（取自 Pipeline 本体）单独给。
  判据：`test_dataset_query_exposes_chain_params_and_real_split_policy`。
- **附带（未修，属契约问题）**：`Pipeline.fingerprint()` 不含 `seed`
  （`pipeline.py:391-395`）⇒ 换种子重切分**指纹不变**；若要"指纹能区分数据视图"，
  得先改契约（`docs/契约.md`）再动实现。

### A3. "表格能画出来"≠"面板喂得对参"：`surrogate` 的假迁移
- **症状**：代理切片页在 web 端每次都"SVG 渲染失败 + 回退 PNG"（响亮，但根因被掩盖）。
- **原因**：`opt/specs.SPEC_BY_KEY` 把 `surrogate` 登记成"已迁移到中立契约"，
  但 `spec_surrogate_1d(bundle)` 吃的是**六元组**（要活的优化器实例算出的 GP 后验），
  不是 `OptRecord` ⇒ 面板按统一约定喂 record 必抛 `TypeError`。
  `test_plot_spec.py` 只按**页 key（字符串）**验"表里的函数能渲染"，
  验不到"面板喂得对不对"。
- **修法**：显式列为 `panel_web._PNG_ONLY_PAGES`，页面上说明为什么只有 PNG。
  判据：`test_surrogate_page_is_honestly_png_only`（拿**真的 PageSpec** 走一遍）。
  **教训**：迁移类判据要拿**声明对象**验，光验名字表会漏。

### A4. 工具**声明了**参数、处理函数却没收：`run_method.diag` 从来调不通
- **症状**：调用方按 `parameters` 声明传 `diag` ⇒
  `execute() got an unexpected keyword argument 'diag'`。
- **原因**：`run_method` 的 `parameters` 里有 `diag`，但 `_run_method.execute`
  的签名里**没有**它（声明与实现漂移）。而既有的 null 垫片探针只验
  "显式 null 被丢掉"、`EXPECTED_DEFAULTED` 只看声明 ⇒ **没有一条真把
  `diag=True` 发下去**，所以一直没被发现。
- **修法**：补上签名并把 `diag` 放进 spec；判据
  `test_ml_delivery.py::test_tool_params_stay_within_command_declaration`
  两个方向一起钉（工具参数 ⊆ 命令声明；端到端真发一次并检验产物）。
- **同批**：`submit_run` 缺 `persist`/`diag` ⇒ 长训练只能走它，于是
  "AI 跑完人回 GUI 历史里看不到 / 拿不到归因页"。已补。
- **教训**：给工具加参数时，**判据必须真调一次那个参数**，不能只比声明表。

### A5. 从 `cfg.extras` 读"训练超参"：用户的覆写被静默忽略
- **症状**：`run_custom(overrides={"epochs": 6})` 跑出来 60 个 loss 点
  （默认 60 轮），参数看起来"设了"。
- **原因**：方法体写成 `p = cfg.extras`，而**覆写是经 `cfg.overrides` 进来的**
  （`extras` 只承载 `run_id` / `device` / `on_progress` / 自定义代码这些
  **非参数**的运行级东西）⇒ `p.get("epochs", 60)` 永远拿默认值。
- **修法**：用 `self.params(cfg)`（= 声明默认 + 运行级覆写，单一真源）。
  判据：`test_ml_mecha.py::test_custom_network_runs_and_keeps_the_observation_surface`
  断言"6 epoch 就是 6 个点"。
- **为什么值得单列**：这是"参数形同生效"的又一例，而且**只在端到端断言
  （点数）时才显形**——单元层看不出（`extras` 里确实没有那个键）。

### A6. `run_id` 是确定性的 ⇒ 重跑会往**同一个** `tb/` 里追加
- **症状**：一次 6 epoch 的训练，`describe_training` 读出 **60** 个 loss 点。
- **原因**：`runner._run_id(method, seed)` 只由"方法名 + 种子"决定
  ⇒ 同方法同种子**重跑会复用同一个 `runs/<id>/tb/`**（这是刻意的：让
  TensorBoard 直播目录在跑之前就定死）。多份事件文件叠在同一目录里，
  步数就累加了。
- **修法**：这是**有意行为**，别当 bug 修；但**用固定 run_id 的判据必须
  给一个独特的种子**（否则上一轮的事件会污染断言）。
  代码侧的留痕机制是 `tb_series.read_run` 的 `duplicate_steps`
  （同一步被写多次时如实报出来，不静默去重）。

- **症状**：一次 6 epoch 的训练，`describe_training` 读出 **60** 个 loss 点。
- **原因**：`runner._run_id(method, seed)` 只由"方法名 + 种子"决定
  ⇒ 同方法同种子**重跑会复用同一个 `runs/<id>/tb/`**（这是刻意的：让
  TensorBoard 直播目录在跑之前就定死）。多份事件文件叠在同一目录里，
  步数就累加了。
- **修法**：这是**有意行为**，别当 bug 修；但**用固定 run_id 的判据必须
  给一个独特的种子**（否则上一轮的事件会污染断言）。
  代码侧的留痕机制是 `tb_series.read_run` 的 `duplicate_steps`
  （同一步被写多次时如实报出来，不静默去重）。

### A7. 内部标记经 spec 传递 ⇒ 被当成"未声明的参数"，job 瞬间假完成
- **症状**：给后台请求打了内部标记之后，job **立刻**以 `state=done` 结束、
  `result_ref` 为空、进度 0 条。看着像"训练秒完"，**真相是它根本没跑**。
- **原因**：标记是塞进 `spec` 的（`{"job_run": True}`），
  而 `validate_spec` 只认**命令声明的参数** ⇒ 当场判"接受了未声明的参数"，
  返回 `ok=False`。**最坏的地方是它不响亮**：job 的状态机把"函数正常返回"
  记成 `done`，于是"失败"被包成"完成"。
- **修法**：引擎在 `run` 里**校验之前**把标记摘出来（`clean_spec`），
  转成显式形参 `self._job_run` 供 `_build_request` 消费。
  判据：`test_ml_mecha.py::test_internal_job_marker_never_leaks_into_command_params`
  （两个方向：标记不在任何命令参数面里；带标记跑一次**仍然成功且真落盘**）。
- **教训**：凡是要给引擎传"内部意图"，**别走 spec**（那是命令参数面，有校验）；
  走显式形参或 context。而且"job 说 done"不等于"事情做成了"——
  要看 `result_ref` 与产物在不在。

### A8. 同步 vs 后台：长任务只能用"信号 + 轮询"
- **症状**：自定义网络的长训练用同步 `run_custom` 跑 ⇒ 10 分钟后客户端超时，
  AI **拿不到 `run_id`**，训练白跑（数据在盘上，但没人知道它叫什么）。
- **原因**（接口形态，不是 bug）：LLM 的一次工具调用本质是 HTTP 请求，
  客户端**必须有超时**（本实例 `TOOL_CALL_TIMEOUT_MS = 600000`）；
  且 MCP **没有"服务端主动推给模型"**这条路。
  ⇒ "发一个开始信号 + 之后主动轮询"是**唯一可行**的长任务形态。
- **修法**：`submit_run` 支持 `code=`（提交 `run_custom`），并补齐轮询面：
  进度带 `loss`/`val_loss`（有界窗口，够判断早停）、带 **`run_id`**
  （`result_ref` 只在终态才有 ⇒ 不带它就没法"边跑边查曲线"）、
  后台任务**终态自动落盘**（job 状态只在进程内存，权威一重启句柄就没了）。
- **配套**：`trainer` 的 epoch 边界现在是**协作式取消点** ⇒ `cancel_run`
  对正在跑的训练真的能停（在此之前只能等它跑完）。
  判据：`test_ml_mecha.py::test_background_custom_network_reports_progress_and_run_id`
  （在**还在 running** 时就断言三件事，然后取消收尾）。

### A9. 判据"定义了却不跑"：门禁用脚本入口 + 手写清单 ⇒ 20 条判据从未进闸
- **症状**：文件绿、门禁绿、单测也绿，但**一多半判据从来没被执行过**。
  实锤：`test_ml_mecha.py` 定义 57 条、入口的手写清单只列 37 条 ⇒ **20 条从不进闸**
  （含 GNN 端到端、后台自动落盘、训练观测面）；其中
  `test_progress_hook_must_be_attached_before_session_run` **当时就是红的**
  （我跑过 `pytest -k`，但门禁跑的是**脚本**，两者看的不是一回事）。
  `test_ml_delivery.py` 同理漏了 `test_tool_params_stay_within_command_declaration`
  —— 正是"工具声明与实现漂移"那条判据。
- **原因**（形态，不是疏忽）：`tests/run_all.py` 用 `python tests/<file>.py` 跑，
  于是"跑哪些判据"由**文件自己的入口清单**决定。清单是手写的 ⇒ **必然会漂**，
  而且漂移的方向恰好是"最坏的方向"：新加的判据（也就是最需要跑的那些）不进闸。
- **修法**：入口改成**内省收集**（`globals()` 里所有 `test_*`，按名字排序），
  **定义即入闸**；并加元判据
  `test_regressions.py::test_every_defined_judgement_is_actually_run_by_its_file`
  （手写清单形态下，任何"被 def 了却没被引用"的判据都判红）。
- **教训**：绿的颜色**不能代替"跑过没有"**。凡"存在但从不执行"的东西，
  与不存在等价——本仓对 `test_dsh_panel.py` 早就写过这句话，这次轮到判据自己。

### A10. 指导意见的举例名撞宪章：`reset_config` / `run_actions` 是 EL 的命令名
- **症状**：照抄 mecha《AI调用体验-指导意见》里举例的名字（P1-5 的 `reset_config`、
  §4 的 `run_actions`）之后，宪章判据当场红：
  `test_static_ml_mecha_does_not_copy_el_command_or_state_names`
  —— `EL_COMMAND_NAMES` 里**就有这两个名字**。
- **原因**：指导意见是**跨项目**的通用建议，它的举例名来自 EL 项目；而本仓宪章 R1
  硬性要求"命令面是 ML 自己的词汇"。⇒ 冲突是**结构性的**，不是谁写错了。
- **修法**：采用它的**意图**、换 ML 自己的名字：`clear_current_state` /
  `run_queries`；并在判据旁边写清"为什么这两个名字不能用"。
- **教训**：抄指导要抄**判断依据**，不要连**名字**一起抄——名字带的是那个项目的语境。
  （另：一次批量改名脚本曾把 `EL_COMMAND_NAMES` **自己**也改了，等于把守卫改成
  "允许这些名字" ⇒ 改名后必须重跑那些"守卫自身"的判据。）

### A11. `_write_state` 会**过滤 None** ⇒ "清零"会被静默吃掉
- **症状**：`clear_current_state` 用 `_write_state(..., {k: None for k in keys})`
  写回默认值 ⇒ **什么都没发生**（写了空 dict、静默返回），而回执看着"成功"。
- **原因**：`_write_state` 里有一句 `payload = {k: v for ... if v is not None}`
  —— 那是给调用方的便利（"有这个值才写"），但 `None` 在这条路上**是有语义的**
  （= 回默认）。两种语义混在一个过滤器里 ⇒ 最坏的一类失败：看起来做了、其实没做。
- **修法**：`_write_state(..., keep_none=False)` 显式区分；清零那条路传
  `keep_none=True`，并在 docstring 里写明"这两种语义必须由调用方说清"。
- **配套**：`current.seed` 当时**拒收 None**（其余 7 个键都收）⇒ "清零"会只清掉
  7/8（**半清状态比不清更坏**：AI 会以为台面干净了）。⇒ 状态键统一接受 `None`，
  而**请求路径**的 seed 仍必须给整数（那里留空不是"回默认"，是"没说清要跑什么"）。

### A12. 判据的前提不能靠仓库历史：`runs/` 一清，判据就红
- **症状**：`test_conflicting_archive_is_disclosed_not_hidden` 在 `runs/` 里
  **翻找**一条"自述分类但指标是回归"的历史档当前提 ⇒ 我清掉探针 run 之后它立刻红，
  报错正是它自己写下的那句"找不到自相矛盾的档（前提变了？那本判据要重写）"。
- **原因**：判据的前提是**外部状态**（仓库里恰好还剩那种档）。换台机器、清一次
  `runs/`、或者只是多跑几次实验，判据就从"绿"变成"红"——而它测的东西从来没变。
- **修法**：前提**由判据自己构造**（造一条 `target_kind="classification"` 而
  指标全是回归指标的档，复现旧 `run_one` 的错标形态）；顺带扫描仓库里有没有真档，
  但**明确不是前提**。
- **教训**：判据依赖"环境里恰好存在某个样本"时，它不是判据，是**彩票**。

### A13. 红验证本身也要验：红的可能是**你的探针**，不是被测对象
- **症状**：做红验证（停掉守卫、看判据是否变红）时"红了"，就以为验证成功。
  实测踩过两次：① 替换字符串没匹配上（改动根本没生效，判据却因为别的原因红了）；
  ② 探针文件用 `Out-File -Encoding utf8` 写成**带 BOM**，判据红在
  `ast.parse` 的 `U+FEFF` 上，**不是红在我要检的那件事**。
- **修法**：红验证要**验证替换真的发生了**（`assert s2 != s` / 打印命中处数），
  并**读一眼红的理由**是否正好是被停掉的那条守卫。
  （`test_regressions.py` 的元判据现在用 `utf-8-sig` 读文件，就是为了不再红在 BOM 上。）
- **补记（同一天又踩）**：给"三层修复"做逐层红验证时，**第二层的锚点跨多行**，
  替换后把源码切出 `SyntaxError` ⇒ 判据"红了"，**但红在语法错上**。
  ⇒ 锚点尽量选**单行且唯一**的；跨多行替换必须回读一眼被改的片段。

### A14. "可教学"死在**最后一公里**：hint 被硬编码覆盖、suggest 读错键名
- **症状**：`kind` 是对的（"机器可判别"那条看着没问题），但**每条失败的 `hint`
  都是同一句万能话**（"命令未生效；检查参数、权限与数据引用后重试"）⇒ AI 学不到
  "该怎么改"。指导意见 P0-3 的原话是"AI 读 `info.kind` **决定怎么改**"——光有 kind 不够。
- **原因**（三处叠加，缺一处都不至于这么坏）：
  1. `tools._raise_on_failure` 里 `hint=` 是**硬编码**的万能句（把引擎写好的
     `error_hint` 覆盖掉），且 `suggest=` 读的键名是 `suggest`，而引擎写的是
     **`error_suggest`** ⇒ 建议**永远是空串**；
  2. `ParamSpec.clean` 抛**裸 `ValueError`** ⇒ 到 adapter 层只能归成
     `internal_error`/`command_failed`（"你参数写错了"被读成"框架坏了"）；
  3. `runner.run_one` 把 `method.fit` 的异常**压成一句 traceback 字符串**
     ⇒ kind/hint 全丢，连内部类名（`ml_toolbox.core.contracts.ParamValueError`）
     都漏进模型可见文本。
- **修法**：让失败**带着 `kind/hint/suggest` 一路穿过去**——
  `ParamValueError`（core，带 kind/hint/suggest）→ `MLResult`/`RunSummary`
  （新增三字段，默认空 ⇒ 既有构造点不用改）→ 回执 `error_kind/error_hint/error_suggest`
  → 工具层**原样透传**（万能句降级为"确实没有时"的兜底）。
- **判据**：`test_ml_mecha.py::test_ai_input_errors_are_teachable` —— 10 个"AI 参数
  写错"的场景，**逐条**要求 kind 具体（写死期望值）、hint 非空且非万能句、
  消息不含内部模块名。三层各做过一次红验证（每层停掉都必须红）。

### A15. **静默丢虚部**：复数进了实数管道，回执还说"成功"
- **症状**（由 NQS 需求者实测报告，我们复现）：把带 `complex128` 列的表交给
  `prepare_dataset`，**回执 `ok: true`**，链摘要正常，只有 stderr 有一条
  `ComplexWarning: Casting complex values to real discards the imaginary part`
  —— 用户拿到的是**被毁掉的数据**，而界面/回执都说没事。
- **原因**：管道里 `DataFrame.to_numpy(float)` 对复数**不是报错而是取实部**
  （pandas 的语义），于是"物理量被静默改掉"这件事一路无声。
- **同类第二条**：变长数组列（object dtype 装 ndarray）⇒ 裸 numpy 广播错
  `operands could not be broadcast together with shapes (256,) (1024,)`
  —— 既不可教学，也不提示"这是逐组数据"。
- **代价**：需求者的原话是"我连我到底在哪一步失败都问不出来……来回试了好几轮才确认
  是**任务形态本身**对不上"。这正是本仓最忌的形态：**用户以为是自己参数写错了**。
- **修法**：`prepare_dataset` 入口做**数据形态体检**（`validator.check_frame_shapes`），
  复数/变长列**当场拒**并**指路**到 `payload`（逐行负载）。
  ⚠ 只拦"能确定"的（dtype 是复数、object 列里装序列），**拦错了比不拦更坏**。
- **判据**：`test_nqs_path.py::test_unsupported_shapes_are_rejected_teachably`
  （两条各写死 kind + hint 必须提 payload）。

### A16. `Network.forward_step` 是**死的**：从来没被传给训练器
- **症状**：说明书说"`forward_step` 可选，用于多输入/特殊批次"，但写了它**一点用没有**
  —— 训练照常跑、不报错、也不调用它。需求者说"我还没走到 `forward_step`"，
  一半原因是数据进不来，**另一半原因是我们根本没调用它**。
- **原因**：`custom._fit_table` 调 `trainer.train(...)` 时**没有传 `forward_step`**
  ⇒ `train` 里 `forward_step=None` ⇒ 永远走默认的 `model(x)` 分支。
  这与"参数声明了却从不传递"是同一类 bug（当时另一处是 `device`，见 A18）。
- **修法**：`forward_step=_hook(net, "forward_step")`。
- **教训（可推广）**：**钩子/参数"声明了"不等于"被用了"**。凡是"文档说有、但没有任何
  判据跑过它"的能力，都要怀疑它是死的 —— 本轮就是在做"让说明书说的每件事都有判据"。
- **判据**：`test_nqs_path.py` 里所有自定义任务判据都**只写 `forward_step`**
  （不写 `forward`）⇒ 一旦这条断了，损失根本算不出来，判据立刻红。

### A17. 设备只搬了一半：模型在显卡、张量在 CPU
- **症状**：`Expected all tensors to be on the same device, but found at least two
  devices, cuda:0 and cpu`（在 `addmm` / `cat` 处）。
- **原因**（三处，都是"改了一半"）：
  1. `custom.py` 把 `res._dev` 硬编码 `cpu`，但 trainer 走 `device` ⇒ 两条路各说各话；
  2. `predict` / `fit_extra_artifacts` 里 `torch.tensor(Xs)` **没带设备**（`mlp.py`
     同位置是带 `result._dev` 的，我们这条漏了）；
  3. **用户网络代码里** `torch.arange(n)` / `torch.zeros(...)` 也没带设备。
- **为什么以前没暴露**：改之前 `run_custom` **永远在 CPU** ⇒ 混不了。
  ⚠ **这是本轮改动带来的新风险**：任何"随手建张量"的旧网络代码在上卡后会炸。
- **修法**：① 张量统一带 `device=result._dev`；② `ctx["device"]` 交给网络代码，
  并在契约里写明"建张量用它"；③ 框架**翻译**这类错
  （`_device_error_hint`）⇒ 否则报错本身对使用者毫无指向性。
- **判据**：`test_nqs_path.py` 全链路在 `device=auto` 下跑（有卡即上卡）；
  两个内联 GNN 判据的代码改成用 `ctx["device"]`（它们同时是**范例**）。

### A18. `device` 声明了却**从不进方法层**（审计说 cuda、实跑 CPU）
- **症状**：`run_method(device="cuda:1")` 被清洗、写进 `current.device`、留下审计，
  但**方法层收不到** ⇒ 假记录。
- **修法**：`_build_request` 把它放进 `RunRequest.extras`。
- **判据**：`test_ml_mecha.py::test_device_declaration_reaches_the_method_layer`——
  ⚠ 它**只钉管道**（extras 里有没有那个键），**不钉"实际用了哪块卡"**
  （后者依赖本机有没有 CUDA，写成 `== "cpu"` 会在无卡机器上**永远绿**）。
  这一轮补上了"真上卡"那条：`checkpoint` 的 meta 里记 `trained_on`，
  `test_nqs_path.py` 断言它等于实际解析出的设备。

### A19. **object 数组**写进 `artifacts.npz` ⇒ 存档读不回来
- **症状**：训练成功、指标正常，但 `load_run_record(run_id)` 炸
  `Object arrays cannot be loaded when allow_pickle=False`。
- **踩过两次**：
  1. `predict_graph` 返回 `dtype=object` 的数组；
  2. 本轮 `target_kind="custom"` 时 `y` 是 `None` ⇒
     `np.asarray(None)` 产出 object 数组，写进 `y_true`。
- **修法**：产物里**只放定型的数组**；自定义任务**不走标准 predict/evaluate**
  （`runner` 里显式跳过，并在 diag 里写明理由）；文件名列表这类"字符串集合"
  **拼成一个字符串**存（`"|".join(...)`），不存 object 数组。
- **判据**：`test_nqs_path.py` 的 checkpoint 判据**会真的 `load_run_record`**
  （只要产物里有 object 数组就红）。

### A20. 分组留出：随机切分**做不到**"外推评估"，还会漏掉同源泄漏
- **症状**（需求原文）："训练集内 vs **尺寸外推**分开看：N≤12 参与训练、N≥14 没见过。"
  随机切分会把 N=14 的一部分也放进训练 ⇒ 那个"外推分数"**根本不是外推**。
- **同类隐患**：同一来源/同一病人/同一批次的样本被切开训练与测试 ⇒ **同源泄漏**
  （分数乐观化，且不会报错）。这与时序的"未来泄漏"、图的"邻居泄漏"是同一族。
- **修法**：`split.group_by` + `split.holdout`（按列整组留出）。
  ⚠ `group_by` 与 `holdout` **必须成对**给（漏一半就 `bad_group_split`），
  切空了也要报错并**列出该列的实际取值**（否则用户只能猜）。
- **判据**：`test_nqs_path.py::test_group_split_holds_out_whole_groups`。
- **教训**：机制齐全 ≠ 端到端可用。**每跨一层都要问一句"这个字段还在吗"**——
  这一条的 hint 就是被"最后一公里"的一行硬编码吃掉的，而它前面每一层都写对了。

