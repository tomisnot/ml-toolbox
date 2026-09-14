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
