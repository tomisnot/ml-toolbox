> 2026-10-13：验证范围以 `../tests/README.md` 为准。默认核心，常规全量用 --all；历史词汇禁令、技能镜像门禁与运行计数不构成现行开发义务。

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

### A6. `run_id` 是确定性的 ⇒ 重跑会往**同一个** `tb/` 里追加（⚠ **旧行为，已改**）
- **症状**（旧版）：一次 6 epoch 的训练，`describe_training` 读出 **60** 个 loss 点。
- **原因**（旧版）：`runner._run_id(method, seed)` 曾经**只由"方法名 + 种子"决定**
  ⇒ 同方法同种子**重跑会复用同一个 `runs/<id>/tb/`**（当时是刻意的：让
  TensorBoard 直播目录在跑之前就定死）。多份事件文件叠在同一目录里，步数就累加了。
- **已改（2026-10-05 起）**：`_run_id` 带**毫秒**——
  `f"{method}-{int(time.time() * 1000) % 10_000_000}-{seed}"`
  （`ml_toolbox/core/runner.py:52-53`）⇒ **同 seed 重跑得到不同的 id**
  （盘上实证：`custom_network-2334125-0` / `-1202787-0` / `-4953756-0` 全是 seed 0）。
  ⇒ **"同 seed 会复用同一个 `tb/`"这句话现在是错的**，别照它做判断
  （`docs/usage-best-practices.md` §5.3 与 `SKILL.md` §续训已同步更正）。
- **仍然成立的残留风险**（所以本条不删）：
  1. 毫秒字段是 `% 10_000_000` ⇒ **2.78 小时一个周期**：同方法 + 同 seed +
     同毫秒残差会撞进同一个目录，且**撞了没有任何检测**；
  2. **显式固定 `run_id` 的探针/扫参**会真的追加。盘上抓到过活例：
     `runs/linear_regression-155284-42/`（`record.json` 里 `method=ridge`，而
     `config.extras.run_id=linear_regression-155284-42`，`tb/` 下**两个**事件文件、
     同进程相隔 4 秒）⇒ 用固定 run_id 的判据**必须**给一个独特种子，或先清目录。
- **留痕机制**：`tb_series.read_run` 的 `duplicate_steps`（同一步被写多次时如实报出，
  **不静默去重**）。
- ⚠ **别再把它当成"曲线被污染"的成因**：同一 run 内读起来"两条曲线不像一回事"，
  真正常见的原因是 **A55（两族 tag 的 step 轴单位不同）**，与 append 无关。

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

### A21. "文档推荐的用法"**从没被跑过** ⇒ 确定性 bug 藏了一轮（假绿）
- **症状**：`eval_custom(run_id=…, dataset_id=…)` 稳定报
  `AttributeError: 'dict' object has no attribute 'spec'`（还归成 `internal_error`）。
- **原因**：`_validator.check_dataset_ref()` 返回的是 **facts 字典**，而代码把它当
  带 `.spec` 的 `DatasetEntry` 用 ⇒ `getattr(entry.spec, …)` 必然炸。
- **为什么判据没抓到**（这条比 bug 本身更值得记）：我写的 `eval_custom` 判据
  **只走了 `payload` 那条路**，而**技能文档里推荐的却是 `dataset_id`**
  ⇒ "文档说可行的那条路，没有任何判据跑过"。这与 A12（判据的前提不能靠仓库历史）
  同族：**判据覆盖的是"我想到的用法"，不是"我承诺的用法"**。
- **修法**：改走 `self.dataset_entry(ds_id)`；两条路都进判据
  （`test_eval_custom_accepts_dataset_id_and_teaches_missing_payload`）。
- **推广规则**：**凡写进 skill/交付说明的调用示例，都要有一条判据在跑它**；
  否则文档就成了"未验证的承诺"，而 AI 会照着它撞墙、然后怀疑自己参数写错。

### A22. 钩子声明了、**路径没统一** ⇒ 逼用户写一个**假的 `forward`**
- **症状**（使用者反馈 §2-缺口3）：自定义任务训练到第一轮末尾就炸
  `Module [GNN] is missing the required "forward" function` ⇒ 他被迫写一个
  **与损失毫无关系**的 `forward`，"只为让验证别炸"，并据此怀疑 `best_epoch` 是假的。
- **根因**（比反馈里定位的**更深一层**）：炸的**不是**验证分支，而是 **recorder 的探针**
  （`recorder._collect_probe` 默认调 `model(self._probe)`）。验证分支当时**已经**正确
  走 `forward_step`（上一轮就写对了 `elif`）。
  ⇒ 这是 A13 的另一半：**红验证不只要"能红"，还要"红在正确的位置"**——
  现象描述对、归因差一层，按错误归因去修是**修不好**的（我一开始就差点只改验证分支）。
- **修法**：把"怎么调模型"收成**一条入口**——只要网络写了 `forward_step`，
  **训练 / 验证 / 探针 / 预测**四条路全走它（预测那条没有负载 ⇒ 给空负载并**明说**）。
- **判据**：`test_custom_task_needs_no_forward_when_forward_step_exists`——
  网络里的 `forward` 明确**抛异常**，谁再回到 `model(x)` 立刻红。
  ⚠ 该判据第一版只断言"跑得成"；红验证时发现**把验证分支改坏也不会红**
  （`vl` 变 `None` ⇒ 验证被**静默跳过** ⇒ `best_epoch` 悄悄改用训练损失）⇒
  加固成"让 `forward_step` 在 `kind='val'` 时返回**可辨认的常数 0.25**，断言 `val_loss` 就是它"。

### A23. 设备不一致**伪装成维度错**
- **症状**（反馈 §3.1）：`oh()` 里一个 `torch.tensor([0., 1.])` 没写 `device`，
  而模型在显卡上 ⇒ 报的是 `Sizes of tensors must match except in dimension 2`、
  `not enough values to unpack (expected 3, got 2)`、
  `mat1 and mat2 shapes cannot be multiplied` —— **全都是误导**，他为此烧了十几次往返。
- **为什么"在文档里写纪律"不够**：每漏一处就中招一次，而报错还把人往维度方向引。
- **修法**：在**用户钩子执行期间**把 torch 默认设备摆成训练设备
  （`trainer.hook_device`），`with` 包住那一次调用。
  ⚠ `set_default_device` 是**进程级**设置 ⇒ 必须 `try/finally` 复原，
  否则会渗到我们自己的代码与下一个方法（判据里**专门断言"没渗出去"**）。
- **判据**：`test_bare_tensor_in_user_hook_lands_on_the_training_device`。

### A24. 回执只说"形状"，不说"**一行之内**是什么"
- **症状**（反馈 §2-缺口4）：负载 `(R, 5)` 被收下、回执看着正常，但切批后是 `(B, 5)`
  而网络期待 `(B, L, 5)` ⇒ 又是那句 `mat1 and mat2 shapes cannot be multiplied`。
  他"手工实验出要多带一维"，十几次往返。
- **根因**：`shape` 本身**不含歧义信息**——`(R, 5)` 的"5"是"一行 5 个数"还是
  "一行 5 个位置"必须由人解释，而**只有调用方知道** ⇒ 框架至少要把语义说清。
- **修法**：每个负载加 **`per_row_shape`**（= `shape[1:]`，即"一行之内的完整结构"）
  + 按 rank 的提示（`(R,d)` ⇒ "若一行是序列要给 `(R,L,d)`"；`(R,)` ⇒ "这更像一列"）
  + 一条全局 `_semantics`。
- **判据**：`test_payload_facts_disclose_per_row_shape`。

### A25. 模型构造期拿不到**输入维度** ⇒ 写死 ⇒ 换数据**静默跑偏**
- **症状**（反馈 §2-缺口1，他列为最高优先）：`build(n_in, n_out)` 只给得出特征表列数，
  而真实输入维度在**负载**里 ⇒ 只能写死维度；"后果**不是报错，而是换一份数据
  （换个 N、加一维特征）时静默跑偏**"。
- **修法**：`build` 可多收 `ctx`（**形状事实**，不含数据）；老签名 `TypeError` 落回旧调用
  —— 与 `on_progress` 同一个做法（"新签名优先、老签名也认"）。
- **判据**：`test_build_receives_payload_shapes_and_old_signature_still_works`——
  用**两份 d 不同**的数据跑同一个网络，模型宽度必须跟着变（写死维度的实现会红）。

### A26. 观测的**粒度**不对：最好的诊断信息死在"看不到"上
- **症状**（反馈原文）："我的 NLL 在**前 100 步内就崩到 0.0000**……但在 TensorBoard /
  `describe_training` 上我只能看到**每轮**的值，看不出它是**什么时候**崩的、
  崩得多快。我是在本地脚本里逐轮打印才发现的。"
- **根因**：`on_step` 那时**只回调、返回值被丢掉** ⇒ "步"这一级**没有任何指标面**。
  而这类任务几千步，**塌陷发生在前 2% 的步数里**（他 4500 步、100 步就崩）。
- **修法**：`on_step` **返回 dict** ⇒ 写进 TB（`tb_log.log_step`）⇒
  `describe_training` 直接看得到（它读**全部** tag，查询面一行没改）；
  同时并进回执（**曲线**与**数字**同一批 tag ⇒ 不会各说各话）。
- **判据**：`test_ai_efficiency.py::test_step_level_metrics_reach_the_observation_surface`
  —— 钉的性质是"**步级 ≠ 轮级**"（点数必须**多于**轮数、从 0 开始、单调）。

### A27. "一次往返一个错"：形状类错误必须**一次全看到**
- **症状**（反馈 §5 的客观统计）：一次真实工作里**形状类错误约 15 次，每次一个工具
  往返**；而本地跑一次 python 是秒级 ⇒ 他**被迫**"本地磨对形状、再提交一次"。
  其中最贵的一类**本地测不出来**："我本地全绿、一提交就错——本地 B=3、真实 B=2"。
- **根因**：`run_custom` 收的是**代码字符串**，提交前**没有任何办法知道它跑不跑得起来**。
- **修法**：`dry_run`——编译 + 建模型 + **用真实负载**跑几次一次前向，回逐模块形状
  轨迹与结构化归因；且**不反传、不落盘、不计入实验记录**。三条关键设计：
  1. **轨迹必须用 `forward_pre_hook`**：forward-hook 只在模块**跑完之后**触发 ⇒
     **炸在模块内部时永远不触发**（实测：错在第一个 `Linear` 上时轨迹是**空数组**
     —— 而那恰恰是最需要它的场合）。pre-hook 让"进去了没出来"成为可判别的状态；
  2. **失败不抛错**：`ok=false` 不是"命令失败"，而是"你的代码跑不起来"这个**问题的
     答案**；抛错会把形状轨迹整条丢掉（实测：一抛就只剩一句话）；
  3. **多个批大小**：默认 1 / 2 / 训练批大小 —— "本地全绿、一提交就错"那类正是
     批次相关的。
- **判据**：`test_ai_efficiency.py` 四条（抓到错+指认位置 / 对了也给轨迹 /
  多批大小真的都跑到 / **什么都没写**）。
  ⚠ "只跑第一个批大小"这条**第一版杀不死**（判据只断言了"非空"）⇒
  加固成"默认那套要真的都跑到（≥3 个、含 1 与 2）"。

### A28. 试跑**偷偷留下记录** = 用探针淹掉实验记录
- **为什么专门钉**：`dry_run` 会被 AI **高频调用**（它的价值就在这）。若它顺手写下
  运行记录/事件/存档，实验记录立刻被"为定位形状错而做的探针调用"淹掉 ——
  而那正是反馈 §4 抱怨的"分不出这是调试还是工作"。
- **判据**：`test_dry_run_writes_nothing_at_all` —— 断言 `runs/` 无新增、
  会话历史长度不变、`run_id` 为空。**空跑不算跑**，这条不能靠"看代码觉得没写"。

### A29. 声明没跟着存档走 ⇒ "能重建"是假的
- **症状**：加了 `build(..., ctx=…)`（负载形状）之后，`eval_custom` 重建模型时
  **没传 ctx** ⇒ 在 `build` 里读 `payload_shapes` 的网络直接炸
  `TypeError: 'NoneType' object is not subscriptable`（实测）。
- **根因**：把"存档 = 权重"当成了"存档 = 能重建"。**能不能重建**取决于
  "`build` 需要什么"，而那些东西（这里的 ctx）也得存。
- **修法**：checkpoint 里存 **`build_ctx`**（形状事实）与 **`payload_decl`**
  （负载的**声明**，不是数组）⇒ 重建与"继承训练数据"都成立，跨重启也成立。
- **推广**：凡是"让 A 能重建 B"的功能，都要问一句"**B 的构造依赖什么，那份依赖存了吗**"。
- **判据**：`test_ai_efficiency.py::test_eval_custom_inherits_the_training_payload`。

### A30. 负载声明的**来源**搞错了一层 ⇒ 继承永远拿到空
- **症状**：`eval_custom` 的"默认继承训练负载"一开始**总是空的**。
- **根因**：负载声明是在 **`prepare_dataset`** 那一步给的，不在 `run_custom` 的参数里
  ⇒ 我只看 `spec["payload"]`，**永远是空的**（实测）。
- **修法**：`prepare_dataset` 把声明留在 `spec.meta["payload_decl"]`；
  `run_custom` 从**数据集**取它（`entry.spec.meta`）再写进 checkpoint。
- **教训**：**"这个字段此刻为空"有两种可能** —— 真没有，或者**你要的东西根本不在
  这一层**。遇到空先问一句"它应该由谁提供"。这与 A21（文档承诺没人跑）同族。

### A31. "看起来像读到了"：两个来源同时给时**静默取了一个**
- **症状**（反馈 §2.2，实测）：`run_custom` 同时给 `code`（3 行占位符）与
  `code_path`（156 行真文件）时，**跑的是内联那段**，而回执把 `code_path` **原样
  回显**、`code_lines: 3` ⇒ 用户以为"读的是文件"（他靠 `n_params: 2` 才识破）。
- **根因**：`code` 优先、`code_path` 只作回显 —— 一个"两个来源都收、但只用一个"的
  参数面对。这类设计的通病：**回执回显了你给的东西，却没说你用了哪一个**。
- **修法**：① 同时给 ⇒ **明确报错**（`bad_code_source`，"只能给一个"），不猜；
  ② 回执加 **`code_source`**（`file` / `inline`），让"实际用了哪一份源码"不必推断。
- **顺带**：他报的"`code` 必填 ⇒ 只给 `code_path` 发不出去"**已是旧行为**
  （实测当前只给 `code_path` 可用）—— 真因是**客户端缓存了工具表**（MCP 会话建立时
  取一次 schema）⇒ 参数必填性变了要重连才看得到。这条也写进了 skill。
- **判据**：`test_ai_efficiency.py::test_code_path_is_exclusive_and_says_which_source_was_used`。

### A32. "没有负载"要能说清**为什么没有**
- **症状**（反馈 §3.1）：他实测"继承没生效"，回执只说"没有负载 —— 多半是那次训练
  没给负载，或声明里的 HDF5 读不到"。而他**知道**那次训练带了 payload ⇒ 只能猜。
- **实测查明的真因**：那个 checkpoint 的键集里**根本没有 `payload_decl`**
  ⇒ 它**早于**"声明随存档走"这次改动（旧存档）。
- **教训**：**文案把可能性都列出来了 ≠ 系统知道答案**。凡"多因一果"的失败，回执应给
  **判别结果**而不是可能性清单（`payload_source_reason`：`no_declaration` /
  `declaration_unreadable` / `no_checkpoint` / `inherited` / `explicit`）。
- **判据**：`test_ai_efficiency.py::test_eval_custom_says_why_the_payload_source_is_what_it_is`
  （**真的把声明从存档里抹掉**再查，而不是构造一个假的返回）。

### A33. 观测面**默认给全量** ⇒ AI 放弃推荐姿势（"预算"不是"token 洁癖"）
- **症状**（反馈 §2/§3 实测）：`read_job` 的 `progress`/`epochs` 是**必填键**却装全量历史
  （一次 1200 条 ⇒ 几百 KB）；`describe_training` 的 `latest`（**必填**）里带 30 个
  `grad_by_layer`。当事人的反应不是抱怨，而是**改行为**：
  "`read_job` 的 `progress` 数组太长……后面我不再直接轮询它了。"
  ⇒ **AI 主动放弃官方推荐的长训练姿势**，这是最硬的证据（比任何 token 统计都硬）。
- **根因**：把"**最新状态**"与"**全量历史**"绑在同一个键上 —— 两个不同的需求共用一个出口。
  而 `pick` 只能删**整个键**，删不掉必填键**内部**的大块（必填键必须始终在）。
- **修法**（统一取向：**默认摘要，要全量显式要**）：
  `read_job` 默认 = `progress_latest` + 最近 `tail`(8) 条 + 真实条数/`truncated`，
  `detail="full"` 才给全量；`describe_training` 的逐层梯度默认"前 3 + 后 3 + 总数"，
  `grad_layers="full"` 才给全。
- ⚠ 键名**保持不变**（仍在 `required` 里）⇒ 既有消费者不断；变的只是**内容默认有界**，
  而真实条数由 `n_progress`/`n_epochs`/`truncated`/`grad_by_layer_n` **说出来**
  （"截断了"必须可见，否则就成了新的静默）。
- **判据**：`test_ai_efficiency.py::test_read_job_defaults_to_a_summary_not_the_whole_history`
  与 `::test_describe_training_summarizes_grad_by_layer`。

### A34. `dry_run` 只验前向 ⇒ **看不见"训练动态"**（长训练里最贵的那类）
- **症状**（反馈 §4）：这一轮三次长训练**全部分散/发散**（step 120 处 loss 0.009 → 113；
  以及 `nll → 0` 的退化解），而 `dry_run` 在形状/vtable 层很准却**一个都抓不到**：
  权重尺度是否病态、loss 是否几十步内爆掉、目标是否退化 —— 全在它的盲区。
- **修法**：`warmup_steps=N` **真反传 N 步**（用它自己的优化器），回
  `loss_curve`/`grad_norm_curve`/`params_changed`/`any_nonfinite` + **verdict**。
  边界不变（不落盘、不计记录），成本是几十步而不是上千步。
- **三条判读规则**（第二条是实测补的）：loss 涨一个数量级 / **梯度范数涨一个数量级**
  （Adam 会归一化梯度尺度 ⇒ 病态配置下 loss 仍"在下降"，只看 loss 会漏判）/
  **loss 塌到 0 且梯度≈0 = 可能是退化解**（"学得好"的假象）。
- **判据**：`test_ai_efficiency.py::test_dry_run_warmup_steps_catches_dynamics_errors`。

### A35. 差点去修一个**不存在的 bug**：`progress` 整场为 0 其实是"启动耗时"
- **我看到的**：用 `read_job(wait_s=0)` 每 1.5s 轮询一个 60 epoch 的 job，**整场
  `progress` 都是 0**，结束的一瞬间跳到 60 ⇒ 看着像"进度根本没边跑边报"。
- **差点做的**：去改引擎/框架的进度推送（框架还是只读仓，改了也没用）。
- **实测真相**：那套东西的耗时**几乎全在启动**（torch 导入 + 建模型 ~9s），而 60 epoch
  不到 0.2s 就跑完 ⇒ 我的采样点**恰好全落在启动阶段**。换个 30 epoch 的 job 用
  `wait(120)` 再看，`progress` 30 条一条不少。
- **教训**：**"观测为空"先问"我是不是采样在了一个没有观测的阶段"**，再问"功能坏了吗"。
  顺带：判据**不该依赖"正好采样在中途"**（那是时序赌博）⇒ 那条判据改成"等到终态再断言
  回执的默认口径"（与何时读无关）。

### A36. **一次 CUDA 断言，之后全是回声**：栈指向的位置是假的（代价：一小时）
- **症状**：使用者报告 —— 通过工具箱提交必在 `custom.py:238` 的 `rec.set_probe(...)`
  报 `CUDA error: device-side assert triggered`，**每次都一样**。他据此写了一份分析，
  把怀疑对象定成"**探针路径与多组数组负载的交互**"，并请开发者去看
  `probe_fn`/`set_probe` 在 `(R,K,L)` 负载下的行为。
- **实测复核（三步，全部有据）**：
  1. **那一行不是现场**：`set_probe` 里的 `torch.tensor(..., device=dev)` 是那次运行
     **第一次与显卡同步**；CUDA 内核错误**异步上报** ⇒ 栈指向"下一次同步"而不是出错点
     （错误文本自己就写了这句话，但没人会去信它）。
  2. **第一次断言不是多 N**：按时间列出他们的 run 错误序列，**首次**出现在
     `custom_network-3384947-211`（22:19:59，dataset = **N6 的相位头消融**）；
     此后每一次（含所有多 N 的）都是**同一个进程里的回声**。
  3. **两条候选都复现不出来**：用**那次失败的原代码**（从 `record.json` 的
     `extras.custom_network_code` 取出）+ **真数据**（`runtime/ml_n6/payload.h5`）+
     **原参数**，以及多 N 的（`ml_multi_10f/payload.h5`，186×256×10）+ 他们的多 N 网络，
     都在 `CUDA_LAUNCH_BLOCKING=1` 的**新进程**里**跑通**（指标正常）。
     ⇒ 触发条件不是负载/探针交互，而是**长命进程的状态**（上下文被毒化）。
- **教训**：**CUDA 断言是"进程级一次性"的**：一旦发生，本进程内**所有**后续 CUDA 运行
  都会在"第一次同步"处报同一个错 ⇒ 报错位置与真实原因**无关**。
  遇到它的正确动作：① **重启进程**；② 要真位置就 `CUDA_LAUNCH_BLOCKING=1` 再跑；
  ③ **不要在那个栈上改代码**。
- **修法**：`_summarize_record` 现在区分"**第一次**"与"**回声**"：
  首次 ⇒ `kind=cuda_device_assert` + 指导（`CUDA_LAUNCH_BLOCKING` / 先跑 CPU /
  本进程上下文已毒化）；之后每次 ⇒ `kind=cuda_context_poisoned` + "这是 `run-X` 的
  **回声**、位置是假的、**重启**才恢复、别在这里改代码"。
  ⚠ 检测要匹配**完整** error：`warning` 只取**最后一行**，而关键字在第一行
  （判据当场抓到这个失误）。
- **判据**：`test_ai_efficiency.py::test_cuda_device_assert_is_told_apart_from_its_echo`
  （走真实入口 `_summarize_record`，并钉"无关错误不受影响"）。

### A37. **job 完成 ≠ 命令成功**：命令层失败被 job 层吞成"静默死亡"
- **症状**（使用者实测）：`submit_run(run_custom, code_path=…)` → `ok:true, job_id`；
  随后 `read_job` 显示 **`state:"done"`、`run_id:""`、`progress:[]`、`epochs:[]`、
  `error:""`**，而 `runs/` 下**没有新目录**、`logs/` 也没有新条目。
  同一份配置**重启前成功**（23:08 产出 `custom_network-6491588-701`）、**重启后重投
  就静默死亡**。
- **根因**（复现确认）：**命令自己失败了** —— 错误留在 **`job.result`（命令回执）**里，
  而 job 层只记**它自己**的异常（`job.error`）⇒ `read_job` 只读 job 级 error，
  于是一个失败的 job 长得**和成功一样**（`done` + 空 error + 无 run_id + 无进度）。
  实测的两种触发：**重启后 `dataset_id` 失效**（`unknown_dataset`）、
  `code_path` 不在授权数据根内（`dataset_path_denied`）。
- **修法**：`read_job` 现在读 job 的**回执**，命令层失败就**抛可教学错误**
  （kind 用命令自己的，如 `unknown_dataset` + 它带的 hint/suggest），而不是回一个"成功"。
  ⚠ **`state=cancelled` 例外、一律不抛**：取消是调用方主动要的正常结局，而且它的回执
  本身就是 `ok=false`（"训练被协作式取消…"）⇒ 只看 `ok=false` 会把它误报成失败
  （实测：这条把 `cancel_run` 之后的读状态判据打红）。
- **判据**：`test_ai_efficiency.py::test_submitted_job_reports_command_level_failure`
  （两条失败路径都钉，且要求回执里说"没有产出可用的运行"）。

### A38. 进 `extras` 的东西**必须 JSON 安全**（否则成功会被判失败，且只在 job 路径暴露）
- **症状**：`submit_run` 提交一次**其实成功**的运行（有 run_id、有进度），
  回执却是 `ok=false` + `internal_error`：
  `TypeError: Object of type ndarray is not JSON serializable`。
- **根因**（我的 bug）：我把**负载声明** `payload_decl` 原样放进 `request.extras`，
  而内联负载（`payload={"arrays": …}`）的声明里**装着真 ndarray**；后台 job 会
  **序列化** extras/回执 ⇒ 序列化失败把成功翻成失败。
  ⚠ **同步 `run_custom` 完全看不出来**（同步路径不序列化 extras）⇒ 只有 `submit_run` 暴露。
- **修法**：`_jsonable_payload_decl` —— `hdf5`（路径+数据集名，本来就安全）原样保留；
  内联 `arrays` **只留形状/dtype**。而"默认继承"要用的**内联真值**改由 **checkpoint**
  携带（`torch.save` 能存 numpy）⇒ 继承能力不丢。
- **教训**：**在跨进程序列化的边界上，"我没写盘"不等于"它不落任何序列化器"**。
  凡是进 `extras`／回执／审计事件的东西，都要先问一句"它能被 `json.dumps` 吗"。
- **判据**：`test_ai_efficiency.py::test_inline_payload_declaration_is_json_safe`
  （声明可 `json.dumps` + 走**后台 job** 真跑一次必须有 run_id）。

### A39. **"注册 → 跑 → 撤"是并发竞态**：后提交的两个会死在"未注册的方法"
- **症状**（用户实测原文）：连续 `submit_run` 三个 `run_custom` ⇒ **第 1 个成功，
  后两个**全部 `state=done` + `error: 未注册的方法: custom_network`
  （可用方法表里躺着 59 个经典方法，**反而误导**：让人以为方法名写错了）；
  **取消一个任务后立刻提交**也复现。
- **根因**：`_cmd_run_custom` 每次提交临时造一个**带代码的子类**
  （`registry.unregister(name)` + `registry.register(subclass)`）、跑完在 `finally`
  里 `unregister` ⇒ A 的收尾**撤掉了 B 刚注册的项**；两个 `unregister/register`
  交错时还会撞 `方法名重复`。之所以要"临时注册"，是因为 `registry.get(name)`
  **无参构造** ⇒ "这一段代码"只能挂在**类**上。
- **修法**：注册项**常驻且不带代码**（`custom.py::_register_into_registry`），
  代码随 `cfg.extras[CUSTOM_CODE_KEY]` **逐请求**下传 ⇒ 竞态从根上消失，
  而且并发跑**两份不同代码**也各自正确（以前会串成"谁后注册谁覆盖"）。
- **教训**：**"用完就撤"在并发下不是清理，是踩别人**。共享注册表上的
  可变状态必须问一句"两个同时在跑会怎样"；而"必须挂在一个全局对象上"
  通常说明**该换成按请求传递**（这里有现成的通道 `extras`）。
- **判据**：`test_nqs_path.py::test_concurrent_run_custom_does_not_unregister_each_other`
  （走真实命令入口，桩替换训练那一段；把"收尾撤除"加回去 ⇒ 判据红，且红的正是
  用户那句 `未注册的方法: custom_network（可用: ['adaboost', …]）`）。

### A40. **等不到终态**会被绕过：等待原语要么一次等到、要么交给独立进程
- **症状**（用户实测）：`read_job` 的 `wait_s` 上限 30 s ⇒ 为等一个 250 epoch 的
  训练**十几次轮询**，最后**绕过整套工具**用本地 `pwsh` 睡 9.5 分钟再查目录。
  **绕过工具 = 缺能力最硬的证据。**
- **事实（别猜）**：本实例的 MCP 客户端单次调用超时是
  `launcher.py:98 TOOL_CALL_TIMEOUT_MS = 600000`（**10 分钟**，不是 SDK 默认的 60 s）
  ⇒ **540 s 阻塞是安全的**，再长就会撞客户端超时（那时"等待"会被读成"工具失败"）。
- **修法**：① `read_job(wait="terminal"/"any", timeout_s≤540)`：**一次**调用等到终态，
  超时**不报错**（回 `timed_out=true` + 进度 + "可以再调一次"）；
  ② 更长的等待交给**独立进程** `python -m ml_mecha.watch <job_id>`——
  它经 MCP 读**权威的真状态**、到终态才退出，配 `pwsh run_in_background`
  ⇒ harness 在它退出时**注入一条完成通知**（这就是"完成事件"的今天可用形态）。
- **教训**：把"等"做成**会自己结束的进程**，比在协议层造推送便宜得多；
  而上限这个数字**必须去客户端配置里查**（本仓 600 s 与 SDK 默认 60 s 差 10 倍，
  猜错的代价是"等待"变成"假失败"）。

### A41. 看门狗自己的两个**假话**：陈旧的进度、以及 `ok:true` + `state:running`
- **症状**（我自己造的第一个 bug，实测 5 分钟）：`job-18` 盯了 5 分钟，进展行一直
  打 `epoch 164/250` 不动 —— 看着像**训练卡住了**。其实是我"拿到 `run_id` 之后
  就不再问权威、只等盘上记录"，于是把**最后一次读到的阶段当实时值**打。
  由此还出现过回执里 `ok:true` + `state:running` 这种**自相矛盾**的组合。
- **修法**：每轮都问一次 `read_job`（只读、毫秒级，不打扰训练）；终态以**权威**
  为准，盘上 `record.json` 只是**载荷与证据**，并新增 `state_source`
  （`authority` / `record`）如实标注结论来源；等不到终态时 `state="unreported"`
  ——**不替权威下结论**。
- **另一个**：盘上会有"没有 `record.json` 的**残骸目录**"（跑一半被 kill）。
  本仓 `runs/` 里真有一个 **22 小时前**的。没有新鲜度过滤时 `:latest` 会一头扎进去、
  永远等一个不会出现的记录 ⇒ 加 `DEFAULT_FRESH_S=300`（运行中的目录 mtime 会随
  TB/checkpoint 持续刷新）。
- **教训**：**观测工具自己的每一句话都要有出处**。"最后一次读到的值"不是"现在的值"；
  两个来源（权威状态 / 盘上记录）交叉验证时要**说出用了哪一条**。

### A42. CLI 输出里的 emoji/⇒ 会在 GBK 控制台上**崩在打印这一步**
- **症状**：看门狗**已经拿到终态**，却
  `UnicodeEncodeError: 'gbk' codec can't encode character '\u2705'` ⇒ 退出码 1
  ⇒ 一次**假失败**（而且失败发生在"打印"这行，看起来像工具坏了）。
- **修法**：机器字段（`state`/`run_id`/指标）与人话摘要**纯 ASCII**；
  中文解释只放 JSON 的 `notes`，而 JSON 用 `ensure_ascii=True` 转义
  （任何编码下都能原样解回来）；CLI 另外把 stdout/stderr `reconfigure(utf-8)`。
- **教训**：本仓的"可教学错误"文化在这里有个反直觉的推论——**给人看的话越花哨，
  在 Windows 控制台上越容易把成功变成失败**。跨进程边界先保证"能被打印"。
- **判据**：`test_ml_delivery.py::test_watch_summary_lines_are_pure_ascii`
  （放回 emoji ⇒ 立刻红）。

### A43. `state=done && error=="" && run_id==""` **这个组合不许存在**
- **症状**（用户点名的"不该存在"）：一次 job 终态既没有 run、也没有可读的失败
  ⇒ 它**无法区分"成功但慢"与"从未启动"**（A37 是它的第一种成因）。
- **修法**：在**后台 job 的唯一入口**（`MLMecha.submit` 的 worker）强制两条蕴含式：
  `ok=True ⇒ run_ids 非空`；`ok=False ⇒ error 非空且 error_kind 具体`。
  违反 ⇒ 回执被改写成 `ok=false` + `error_kind="silent_terminal"`
  （**框架自检失败**，而不是让调用方看到"done 但什么都没有"）。
- **教训**：**"不许存在的组合"要在唯一入口上强制**，而不是在每个消费者里各判一次
  ——后者迟早漏一个（`read_job` 漏过、`describe_run` 又漏一次）。
- **判据**：`test_ml_delivery.py::test_empty_receipt_becomes_silent_terminal_not_done`
  （把真实命令处理链里的 `run_method` 换成"回空回执"的桩 ⇒ 必须得到
  `silent_terminal`，且 `read_job` 报失败）。

### A44. 身份用**内容**定，不要用**名字**或**会话计数器**定
- **症状**（用户实测，出现频率最高）：`dataset_id` 每次登记都变（`ds-001-…` 会话
  计数器）⇒ "上一轮用的哪份数据"丢失，**只能靠记忆对齐**；重启后全失效，一晚上
  **重复登记七八次**（《长任务编排与失败可读性》§3.4）。
- **修法**：`id = ds-<fp12>`，`fp` 由（来源内容 ⊕ 管道声明 ⊕ 目标/类型 ⊕ 图/负载
  内容 ⊕ 行数）算出 ⇒ **同内容必然同 id**（幂等）；可读含义交给 `label`/`aliases`。
  台账落盘（`.ml-mecha/ml_ledger/datasets.jsonl`）⇒ 可复原的来源（csv/synthetic）
  重启后**懒复原**。
- **两条教训**：
  1. **名字不能进身份**：名字进 id 就会"换个名字裂成两个 id"。但**纯指纹也不好用**
     （人读不出来）⇒ 身份与标签**分开**才是正解（"机器对齐靠 id，人读靠 label"）。
  2. ⚠ **我的第一版判据漏了跨会话那一步**：会话内的幂等索引把"名字掺进 fp"这个
     变异**掩盖**掉了（第一次登记返回的 id 本来就是它自己算的）⇒ 变异**杀不死**。
     补上"**跨重启 + 换名字重新登记仍须同 id**"之后才红。**这就是 A13 说的
     "挑的 mutation 与原实现等价"——等价不是因为变异弱，是因为判据没走到那条路。**
- **判据**：`test_ml_delivery.py::test_dataset_id_is_content_addressed_and_idempotent`
  （含跨重启那一步）、`::test_dataset_survives_restart_by_lazy_restore`、
  `::test_dataset_needs_reprepare_says_why_and_how`。

### A45. "满了就拒"会把编排者逼成人肉调度器
- **症状**（用户实测 §3.2）：并发上限 4，超额**直接报错** ⇒ "我因此必须'投一个、
  等一个、再投一个'"，而他要的是"一次排满 24 小时"。
- **修法**：**入队**（回执给 `queued` / `queue_position`）、`set_job_limit(n)` **热改**
  （改上限要重启的代价 = 清空会话态，用户已因此重新登记七八次）、
  **排队中可取消**（"我会投多、然后改主意"）。
- **两条纪律**：
  1. **两层的真话分开报**：框架的 `state=running` 说"线程起了"，调度层的
     `queued=true` 说"命令还没跑"——**别为了好看把框架状态改写成 queued**（那是报假位置）；
  2. **名额只在真跑完时释放**，而且**排队中被取消的不许释放**（它从没拿到名额）
     —— 释放错一次，上限就永久虚高。判据用"排队的那个没占名额"钉住它。
- **判据**：`test_ml_delivery.py::test_queue_admits_over_limit_reports_position_and_is_hot_editable`、
  `::test_queued_job_is_cancellable_before_it_ever_runs`、
  `test_ml_mecha.py::test_submit_prechecks_write_authority_and_queues_over_cap`。

### A46. 资源真相：**估算由声明给，框架只对照实测执行**
- **症状**（用户实测 §3.3，代价一整晚）：6 GB 卡、配置峰值 5.30 GB、运行时只剩
  3.6 GB ⇒ OOM，而 CUDA 把它报成 `device-side assert`。"**我事前完全不知道会不够。**"
- **修法**：`resource_guard.max_peak_mb`（用户声明）＋ 实测可用显存 ⇒ 提交前
  **可读地拒**（`cuda_oom_predicted`，三个数字都给出）；跑完报**实测峰值**。
  外加**框架级分块** `ctx["chunks"].reduce(..., mode="mean"|"sum")`，权重累加由框架算。
- **教训**：**别猜任意 PyTorch 网络的显存峰值**——它取决于结构/批次/优化器状态/
  是否分块……猜出来的数字会**假装精确**，而"假装精确的预算"比没有预算更坏
  （人会信任一个错的数）。**声明 + 实测 + 可读拒绝**才是诚实的组合。
  ⚠ 无 CUDA 时回**空**（不返回 0、不返回编造值）：用户会拿这个数去设预算。
- **判据**：`test_ml_delivery.py::test_vram_precheck_rejects_only_when_the_declared_peak_cannot_fit`
  （探针**注入** ⇒ 无 GPU 的机器上也确定地跑）、`::test_peak_vram_is_not_faked_on_cpu`、
  `test_nqs_path.py::test_framework_chunks_match_the_unchunked_result`
  （小预算下结果与不分块**一致**；把权重去掉 ⇒ 判据红）。

### A47. 分块的**累加口径**是最容易写错的地方（所以由框架承担）
- **症状**（用户原话 §3.3）："我为了不 OOM，在**自己的网络代码里**手写了
  `forward_step`/`metrics` 的内部分块（按 R 切块、按块加权累加）。"
  ⇒ 每个用户各写一遍，而写错的方式就那两种：**只取最后一块**、**重复计数**
  （或把权重乘两次 = 权重平方）。
- **修法**：`ctx["chunks"].reduce(payload, fn, mode="mean"|"sum", budget_mb=…)`
  —— `mean` 由框架算 `Σ (块行数/总行数) × 值`；用户**不许**再乘权重；
  每块返回的键必须一致（不一致**抛错**，不静默少一个键）。
- **判据**：`test_nqs_path.py::test_framework_chunks_match_the_unchunked_result`
  红验证：把 `mode="mean"` 的权重去掉 ⇒ 数值立刻对不上（33.5 vs 0.52）。

### A48. 观测工具**自己的每一句话**都要有出处（看门狗的三个自造假话）
- **假话 1**：拿到 `run_id` 后就不再问权威、只等盘上记录 ⇒ 进展行把"最后一次读到的
  阶段"当实时值打（`job-18` 冻在 `epoch 164/250` 整整 5 分钟，**看着像训练卡住**）。
- **假话 2**：由此出现回执里 `ok:true` + `state:running` 的**自相矛盾**组合。
- **假话 3**（用户纠正）：`:latest` 按"最近还在写"挑目录 ⇒ 可能选中**启动器自己的
  探针 run**（盘上长得一样），却一本正经地报"我盯住了 X"。
- **修法**：每轮都问权威（只读、毫秒级）；终态以权威为准、盘上记录只是**载荷与证据**，
  并新增 `state_source`（`authority`/`record`）；等不到终态时 `state="unreported"`
  ——**不替权威下结论**；启发式模式**自称启发式**并指路"要精确就传 job_id"。
- **教训**：**"最后一次读到的值"不是"现在的值"**。"猜"要标成猜——一个确定的语气
  比一个错误的数字更能骗人。
- **判据**：`test_ml_delivery.py::test_watch_timeout_never_fakes_a_state`、
  `::test_watch_latest_is_labelled_as_a_heuristic`、
  `::test_watch_flags_terminal_job_without_any_run`。

### A49. **看曲线要看形状，不能看终值**（实测代价：一轮 1500 epoch 的训练）
- **症状**（用户自述）："我一直只读 `latest`（一个点），靠'跨 run 在固定 epoch 上比
  终值'推断曲线已饱和，据此把 epoch 从 900 砍到 250。后来拉了**同一 run 内的曲线**，
  发现 `fid/sub1` 从 0.367 一路升到 0.682 且**在 epoch 250 时仍在爬** —— 根本没收敛。"
  ⇒ 补到 1500 epoch：`fid/sub1` 0.6823 → **0.7938**。
- **根因**：不是缺数据（曲线一直在 TB 里），而是**看形状很贵**（要 `include_series`
  + 手工在多条曲线间对齐）⇒ 工具让人**很容易看点、很难看形状**。
- **修法**：`describe_training` **默认**给 `curve`（每 tag 的形状：末段斜率/单调/平台
  占比/趋势）与 `convergence`（末段还在爬就直说"可能欠训练"）；`best_by=<tag>`
  按任意 tag 挑最佳轮；`compare=[…]` 多 run 对齐（缺失 `null`，**不插值**）。
- **两条教训**：
  1. **抗噪**：真实曲线**不是单调的**——用户的 `fid/sub1` 末 15 点是 0.667→0.689 的
     爬升，中间有 ±0.005 抖动。我的第一版判据要求"严格单调上升"⇒ 判成 `mixed`，
     **恰好漏掉它要防的那个警告**（判据在**真实夹具**上当场红）。
     ⇒ 用"末段前三分位均值 vs 后三分位均值"这种抗噪判据。
  2. **别用一个词盖住两种病**：`loss` 类在爬是**发散**（配置问题），指标在爬是
     **欠训练**（再跑几轮）；而 `grad_norm`/`ph/std` 这类**没有好坏方向**的量
     **不该给诊断**（第一版错误地复用了给"主指标排序"用的 `is_lower_better`，
     把 `val_loss` 在爬报成了"欠训练"）。
- **判据**：`test_ml_delivery.py::test_convergence_warns_on_the_real_underfit_run_and_not_on_the_converged_one`
  （**两条真实 run 做对照**：250 epoch 必须报 `underfit`、1500 epoch 不许报；
  锚点是仓内夹具，不依赖 `runs/` 目录存亡）、
  `::test_best_by_any_tag_compares_against_val_loss_choice`、
  `::test_compare_aligns_multiple_runs_without_interpolating`。

### A50. **必填骨架别写两份**：`pick` 投影与 `output_schema.required` 各写一遍就会漂移
- **症状**：给 `describe_training` 新增四个必填键（`curve`/`convergence`/`best_by`/
  `compare`）之后，**不带 `pick` 的调用全绿，带 `pick` 的调用立刻
  `bad_tool_output: 缺必填键`** —— 因为 `pick` 那条路把骨架写成了**硬编码元组**。
- **修法**：`_DESCRIBE_TRAINING_REQUIRED` 一处声明，`pick` 骨架与
  `output_schema.required` **都**从它取。
- **教训**：**"声明"与"实现"两处各写一份，就等于两处可以各自漂移**（本仓 R8-2a
  那条纪律的原话）。这次漂移之所以危险：**只有一条调用路径（`pick`）会红**，
  很容易被"其它判据都绿"糊过去。
- **判据**：`test_ml_mecha.py::test_pick_projection_is_honest_about_missing_keys`
  （它同时钉住"投影后骨架仍在"）。

### A51. **"猜"要标成猜，而且能反查就绝不猜**（看门狗归属那条）
- **症状**（用户实测的纠正）：看门狗的 `:latest` 按"盘上最近还在写、且没有
  `record.json`"挑目录 ⇒ 用户的话："那个 `-42` **不是我的**，大概是**启动器自己的
  探针 run**。你的新鲜度过滤把它认成'在跑'了。"
  ⇒ 工具**一本正经地报"我盯住了 X"**，而 X 根本不是他要等的东西。
- **修法**（两步，先精确后退化）：
  1. **反查归属**：新增 `jobs` 只读查询（`job_id → result_ref.run_ids`）⇒ 看门狗
     在无句柄时**先按归属找**，`evidence` 里写 `owned_by_job:<job_id>`（可追溯）；
  2. **退化时自称启发式**：拿不到归属才按新鲜度猜，并在 `notes` 里写清"这是启发式、
     可能选中启动器的探针 run、要精确就传 `job_id`"。
- **教训**：**"最近在写的"不等于"我的"**。可有归属证据时**绝不猜**；只能猜时，
  必须让读的人知道这是猜的——**确定的语气比错误的数字更能骗人**。
- **判据**：`test_ml_delivery.py::test_watch_prefers_job_ownership_over_guessing`
  （归属表里那个必须被选中，哪怕盘上还有一个更"新鲜"的；退化时才允许启发式且必须自称）。

### A52. **"末段回升"不等于发散**：判趋势要看净效应，文案数字必须与结论同源
- **症状**（用户 2026-10-06 的 bug 报告，带真实 run）：
  ```
  {"tag": "loss", "kind": "diverging",
   "note": "⚠ loss 在末 15 个评估点仍在**上升**（136→29.86）⇒ 这个配置**可能在发散**…"}
  ```
  三个毛病叠在一起：
  1. **结论错**：那条 run 的 `loss` 本窗口 1.92e+04 → 29.86（**降 99.8%**，正常收敛），
     末 15 点只有 +1.5% 的回升，而**噪声尺度是 8.8%**；`val_loss` 同样（6372→1060）。
     ⇒ 用户"差点以为自己的配置在发散"（**方向相反的同类误判**：上次漏报欠训练，
     这次虚报发散）。
  2. **文案自相矛盾**：说"在上升"却印全程的 `136→29.86`（数字是降的）——
     用户原话："**这种自相矛盾的文案比没有文案更糟**"。
  3. **判读窗口与证据不同源**：判读用了被 `limit`（默认 200）截断的序列，
     而 `limit` 一调大结论就变 ⇒ "回执说发散，证据却只有 200 点"。
- **修法**：
  1. **全程闸门**：loss 家族的"发散"要求**净效应**——本窗口显著下降（≥20%）时，
     末段小回升最多是 `noisy_tail`；只有"末段涨过 50%"才**无论历史**都算真发散
     （后期爆炸）；
  2. **数字与结论同源**：说"末段"就印**末段窗口自己**的端点，全程数字单独说并
     带"本窗口"字样（还写明**窗口多少点**）；
  3. **判读窗口独立于 `limit`**：`curve`/`convergence`/`best_by` 用**可读到的全部点**
     （上限 `MAX_POINTS`），`series` 才按 `limit` 切。
- **两条更一般的教训**：
  - **趋势判读必须有"净效应"闸门**：只看末段斜率，任何噪声都会变成"发散/欠训练"；
  - **阈值不能只看相对幅度**：噪声尺度在同一 tag 内就能和信号同量级
    （实测：`loss` 的噪声 8.8% vs 回升 1.5%），所以要**比噪声**、并交代**窗口**。
- **判据**：`test_ml_delivery.py::test_noisy_tail_of_a_converging_loss_is_not_reported_as_divergence`
  （他的 run 当"不许报发散"的夹具；真发散与"先降后爆"当反例必须仍报）、
  `::test_curve_analysis_uses_the_full_readable_history_not_the_display_limit`。
  红验证：去掉全程闸门 ⇒ 三条判据红（`noisy_tail` 变 `diverging`）。

### A53. **子串匹配会把无关的名字当成特征**：`linear2` 命中 `r2`
- **症状**（用户实测抓到的，就在同一份回执里）：33 个 `grad/*` 标签中，**恰好
  `grad/dec.1.linear2.weight` 一个**被诊断成"可能欠训练" —— 因为方向判定用
  **子串**匹配特征表，而 `linea**r2**` 命中了质量指标特征 `r2`
  ⇒ 一个"没有好坏方向"的逐层梯度被安上了诊断。
- **修法**：**词元级**匹配（按 `/_.,-` 切词元，允许 `val_loss` 这种"前缀_特征"写法）。
  `linear2` 是 `linear2`，不再是 `r2`。
- **定论**（用户要求"别两边都留一半"）：**逐层梯度与范数类一律不给诊断**
  —— 它们在爬既可能"还没学完"也可能是"梯度爆炸早期"，**方向本身没有好坏**；
  趋势照给（`curve[tag].trend`），读数的人自己判断。
- **教训**：**特征匹配要按词元，不要按子串**——子串匹配在"名字里带数字/缩写"的
  世界里必然误命中（`r2`/`f1`/`acc` 这些短特征尤其危险）。
- **判据**：`test_ml_delivery.py::test_convergence_warns_on_the_real_underfit_run_and_not_on_the_converged_one`
  （含 `direction("grad/dec.1.linear2.weight") == "unknown"` 与逐层梯度不得出现在
  `convergence` 里的断言）。红验证：退回子串匹配 ⇒ 红。

### A54. **把不同步的值并排放，等于制造一个假矛盾**（用户自己的口径 bug 由工具"帮忙"放大）
- **症状**（用户 2026-10-06）：他在一份回执里同时读到 `loss=0.018`、
  `fid/step=0.9965`、`fid/sub0=0.7301`，判断"**三者数学上不能同时成立**"，
  一度怀疑自己的度量/损失口径全错。而三个 tag 的**求值步长本来就不同**：
  `loss`/`fid/step` 逐 epoch，`fid/sub0` 每 3 个 epoch（`metrics_every` 降频）
  ⇒ "各 tag 各自的最新值"很可能**不在同一个 step 上**，每个数在**它自己那一步**
  都是真的。**工具把它们并排放着，就是在暗示"同一时刻"。**
- **修法**：
  1. `latest_steps`：每个 tag 的**求值步号**随值一起给（不标步号就等于暗示同步）；
  2. 同一 run 内 `tags=[…]` 对齐（`compare` 的同族）：同一根 step 轴、缺失处 `null`；
     **`aligned_at_step` = 所有 tag 都有值的最后一步**，`aligned` 取自那一步；
  3. **没有公共步时 `aligned_at_step=null`**、`aligned={}` —— **不许挑一个"差不多"
     的步**凑一个看起来完整的表。
- **教训**：**"并排"是一种断言**。把两个数放在同一行，读者就会认为它们同源同时；
  工具要么给出共同轴，要么**明说它们不可比**。这与 A52（文案与数字不同源）是同一
  条纪律的两个面：**引用必须与对象同源；并排必须与时刻同源**。
- **判据**：`test_ml_delivery.py::test_same_run_cross_tag_alignment_exposes_unequal_step_cadence`
  （用用户那条 run：`fid/sub0` 在非 3 倍数处必须是 `null`；步长完全错开时
  `aligned_at_step` 必须是 `None`）。

### A55. **同一根轴上有两种步号单位**：epoch 序号 vs 全局 batch 步（读数的人会把它读成"点数"）
- **症状**（用户 2026-10-07）："`fid/step` 累计 **4698** 点而 `loss` 只有 **102** 点
  ⇒ 两条不同训练的曲线混在一根轴上了，按 `wall_time` 过滤也切不干净。"
  ⚠ 真实现场（同一份回执）：`loss` 的 `latest_step` = **283**，
  `fid/step` 的 `latest_step` = **78410**，比值 ≈277 = **每 epoch 的批数**。
  4 698 ÷ 102 ≈ 46 也是同一个比值 ⇒ 他**把 `latest_step` 读成了"累计点数"**
  （`tb_series.MAX_POINTS=500`：单个 tag 的点数不可能到 4 698）。
- **原因**：`tb_log` 里有**两条**写标量的路，轴的单位不同，而回执**只给步号、不给单位**：
  - `log_epoch`（loss/val_loss/lr/grad_norm/逐层梯度/权重范数）：`step = ep` ⇒ **第几轮**；
  - `log_step`（`Network.on_step` 上报的步级指标）：`step = gstep` ⇒ **第几个 batch**。
  ⇒ `latest`/`latest_steps` 把它们并排放着，读者自然会认为"同一根轴"（A54 的同族病，
  这次不是"步长不同"而是**单位不同**，比值可达几百倍）。
- **第二个症状（更坏）**：`tags=[…]` 把两族强行对齐时，轴取**并集的尾部** ⇒
  末尾全是 batch 步 ⇒ **epoch 级那一族整列 `null`**（实测 `loss`/`val_loss` 六个点全 `null`，
  `aligned_at_step=null`）。回执看起来像"**这段区间根本没有记 loss**"——那是**假事实**。
- **修法**（2026-10-07）：
  1. `latest_steps` 之外新增 **`step_units`**（只列**非 epoch 级**的 tag：`batch`/`mixed`；
     不在表里的一律 epoch 级）；
  2. 新增 **`axes`**：`epoch_tags_n` / `batch_tags` / `mixed_tags` / `epoch_axis_len` /
     **`per_epoch_estimate`**（就是那个"看起来差几百倍"的来源）/ `note`；
  3. `tags=[…]` 跨族对齐时**当场给出 `compare.axis_warning`**（说清"epoch 级那一族会大片
     `null`，不是没记，是不在这个 step 尺度上"）——**不删数据**（有人确实要看），
     但**结论必须写在数字旁边**；
  4. `mixed`（同名 tag 被两条路都写过）**要点名**：它的轴**本身**不干净。
- **教训**：与 A54 同一条纪律的加强版——**"并排"不只断言"同一时刻"，还断言"同一种刻度"**。
  单位是契约的一部分：只给数字不给单位，等于把"读数的人会自己编一个含义"写进设计里。
- **判据**：`tests/test_nqs_path.py::test_step_axis_units_are_reported_and_cross_family_alignment_warns`
  （一个既有 `on_step` tag 又有 epoch tag 的真 run ⇒ `step_units` 里前者必须是 `batch`、
  `axes.per_epoch_estimate` 必须 ≈ 每轮批数、跨族 `tags=[…]` 必须带 `axis_warning`）。

### A56. **服务端 `print` 读不到**：job 是线程，而 `redirect_stdout` 是进程级
- **症状**（用户 2026-10-07）：网络代码里的 `print("[resume] …")` / `print("[snap] …")`
  是**重要的运行时诊断**（证明续训/快照真的生效），但**没有任何工具能读后台任务的 stdout**
  ⇒ 只能靠 `run_id` + 指标**反推**一个布尔事实，一次往返。
- **原因**：job 是**权威进程里的一条 daemon 线程**（`assembly.JobService.submit` →
  `JobRegistry._run`），`print` 只落在权威的控制台；框架的 `Job.to_dict()`
  （`mecha/providers/jobs.py`）**没有任何日志字段**；`ml_mecha` 侧也没有捕获。
- **修法**（2026-10-07，`ml_mecha/job_logs.py`）：进程内装**一个 tee**
  （`sys.stdout`/`sys.stderr` 的代理），**按 `threading.get_ident()` 分桶**，
  桶挂在 `ExecutionContext` 的 sidecar 上（与 `_PROGRESS_SIDECAR` 同一手法），
  `read_job` 增 `logs_tail`/`logs_n`（默认 10 行，`logs=0` 关）。
- **两条设计约束**（写下来，因为它们是"看着简单、写错就坏事"的地方）：
  1. ⚠ **不能直接用 `contextlib.redirect_stdout`**：它换的是**进程级** `sys.stdout`，
     而 job 是**线程** ⇒ 并发两个 job 时输出会**串台**（A 的 print 出现在 B 的日志里）。
  2. ⚠ **透传与编码兜底都不能省**：tee 必须把内容照原样写给真流（否则等于把 print 从
     人的视野里删掉），且写入自身**永不抛**（A42/A7②：GBK 控制台遇 `⚠` 会
     `UnicodeEncodeError`）—— 日志设施只能"少一行日志"，不能"让任务失败"。
- **边界（如实说）**：只覆盖走 `sys.stdout`/`sys.stderr` 的 Python 输出；
  C 扩展直接写 fd 1/2 的收不到；每 job 的环形缓冲**有界**（默认 `MAX_LINES=200` 行，
  可用 `MLTB_JOB_LOG_LINES` 覆盖；`logs_n` 是**总行数**，截断看 `logs_truncated`/`logs_cap`）。
- ⚠ **两条实现约束（都是实测踩出来的，别"优化"掉）**：
  1. **桶必须有界**：使用者那轮是 **6 小时级**训练、网络代码里**每个 batch 都可能 print**
     （几十万行），而当时 commit 内存到过 92% ⇒ **无界缓冲会把内存吃干**，
     失败还会伪装成「torch DLL 装载失败 / Insufficient system resources」（离根因很远）。
     `deque(maxlen=cap)` 是**环形**（老的自动丢），**不是**"先全存再截"；
     而"总行数"只是一个整数 ⇒ 它无界也不吃内存 —— **两者必须分开**
     （只回 `len(缓冲)` 会把"截掉了多少"藏起来）。
  2. ⚠ **一次 `print` = 两次 `write`**（值 + `"\n"`）⇒ 按 write 计数会把 20 000 行记成
     **40 000 行**，而且 `"\n".splitlines() == ['']` 会让每次 print **多出一条空行**
     （判据当场红）。正解：桶里攒 `_partial`，按 `split("\n")` 组装**真的行**，
     收尾再 flush 没换行的那一段（否则 `print(x, end="")` 会丢）。
- **判据**：① `tests/test_ml_mecha.py::test_job_stdout_is_readable_and_still_reaches_the_console`
  （`logs_tail` 里有网络代码的 stdout 与 stderr 行（含非 ASCII）；**同一行也到了"真流"**
  —— 用记录型替身验透传，防止有人改成"只收不发"；`logs=0` 能关；job 本身必须 done）；
  ② `::test_concurrent_job_logs_do_not_cross_contaminate`
  （**先断言两个 job 区间真的相交**，再断言各自的 `logs_tail` 里没有对方的标记 ——
  串行跑时这条判据是**空转的**，所以那个前置断言不能省）；
  ③ `::test_job_log_buffer_is_bounded_with_a_true_total`  （20 000 行 ⇒ 桶占用**停在 cap**、`logs_n` 是**真总行数**、回的是**最后** cap 行且**保序**、
  `logs_truncated=true`、`logs=N` 调大也**不会**超过桶容量）；
  ④ `::test_native_fd_output_is_outside_logs_tail_and_note_says_so`（**边界**，见下）。
- ⚠⚠ **边界（Q3.5，2026-10-07，必须写在这里而不是只写在代码里）**：
  `logs_tail` 只覆盖**走 `sys.stdout`/`sys.stderr` 的 Python 输出**。
  **C 层（torch / CUDA / OpenMP / MKL）与子进程直接写 fd 1/2 的输出抓不到** ——
  而这台机器上最迷惑的两次失败恰恰是那一类
  （`torch/lib/vrtc64_120_0.alt.dll … or one of its dependencies`、
  `Insufficient system resources`）。
  ⇒ **`logs_tail` 是空，不等于"没有报错"**（这正是"假事实"的另一种形态）。
  为此做了两件**低风险**的事：
  1. **空日志的说明点名这个边界**（`logs_note`：="不代表它没报过错…C 层与子进程直接写 fd 1/2 的
     输出**不在**这里"），并指出**按时间对齐**的口子；
  2. `read_job` 回 **`started_at`/`finished_at`** ⇒ 人/AI 能拿这段时间去权威的控制台
     或日志文件里对齐那段**原生输出**（"看不到"与"无从下手"的区别）。
  ⚠ **没有做"把 fd 1/2 也接进管道"的 native 桶**。判定与**核实过的事实**都写在下面
  （将来有人想做得先反驳这三条；⚠ 其中**两条我一开始写错过**，被使用者要求核实后改正，
  所以这里保留"错在哪"，免得后人照抄错的理由）：
  - **风险 A（真正的阻塞）—— 排空方"活着但不排空"**：管道写满缓冲区（Windows 默认约 64 KB）后
    `write` 会**阻塞** ⇒ 一个 6 小时训练**挂死在 write 上**（比"读不到日志"坏得多）。
  - **风险 B（不是挂死，是报错）—— 排空方"死亡"**：读端句柄关闭后，写端得到的是
    **`BrokenPipeError`**（Windows 上也是报错，不是阻塞）。⚠ 它本身不等于挂死，
    但它会打在**任意一次写入**上（包括 torch/第三方库的 C 层 fd 写，那些代码不一定处理它）
    ⇒ 也需要处理，只是**与 A 是两种不同的失效模式**。
    （我第一版把 B 也写成"照样阻塞"——**不准确**，使用者指出后已改正。）
  - **代价 C（唯一消不掉的）—— 人失去实时可视面**：fd 1/2 被接走后，原生输出**不再出现在控制台**，
    要看只能走转发出来的文件。
  - ❌ **一条不成立的反对理由（我写过，实测推翻）**："接走 fd 会破坏 utf-8 保证"——
    **不成立**：`app_entry.main()` 的 `reconfigure` 是**无条件**的（不看 `isatty`）。
    实测（`checks/oneoff/_factcheck_isatty_utf8.py`，**剥掉 `PYTHONIOENCODING`** 后）：
    `stdout=PIPE` 时 `BEFORE gbk isatty=False` → 调 `app_entry.main(['--help'])` 后
    `AFTER utf-8 isatty=False`；而同样走管道、不调 main 的对照是 `gbk`。
    ⇒ 换成管道**不会**破坏编码（Q2 的 `PYTHONIOENCODING` 注入还是第二层保险）。
  - 另外：它**不可能**归属到某个 job（fd 是进程级、job 是线程）⇒ 只能另开
    **`logs_native_tail`** 并明确标注"进程级、无法归属"；**不要**试图把它并进 per-job 日志
    （那会重新引入串台）。

### A57. `best.pt` **只按 `val_loss` 选** ⇒ 自定义任务的主指标被无视（`best` 比 `last` 差很多）
- **症状**（用户 2026-10-07）：同一场训练里 **`best` 0.6464 vs `last` 0.8174**（主指标口径）
  —— 而 `checkpoint_detail` 只报 `best_epoch`，**不说"这份 best 是按什么选的"**
  ⇒ 拿到 `best.pt` 不知道能不能用。
- **原因**：`trainer.train` 的 `watch = vl if vl is not None else train_loss`
  —— `best_state` 的**唯一**口径是**验证损失**；而自定义任务（`target_kind="custom"`）
  的主指标常常**不是** `val_loss`（NQS 是 `fid/*`）⇒ "最佳轮"选的是另一个目标的最佳。
- **修法（2026-10-07，Q4）**：`Network.best_metric` 可声明用哪个 tag 选 best：
  ```python
  best_metric = "fid/sub1"              # 方向由 `curve.direction` 判（判不出就报错，**不猜**）
  best_metric = ("ph/std", "lower")     # 方向没有好坏的量必须显式给
  ```
  - 值来自 `Network.metrics(model, ctx)` 的返回（每轮一次；`metrics_every=N` 可降频；
    **该轮没上报就跳过那一轮**，跳过数进 `best_by.skipped_epochs`）；
  - `checkpoint_detail` 的 `files_meta.best.best_by` 报 **tag / direction / value / epoch / step**，
    `val_loss_best` 给**对照**（两者常常不是同一轮）；
  - ⚠ **未声明时逐位不变**（`best_by.tag == "val_loss"`，值与从前的 `best_watch` 一致）
    —— 声明是**显式选择**，不是新默认；
  - ⚠ 声明了但**整场一次都没上报** ⇒ **退回 val_loss 并留痕**
    （`nn_best_metric_missing` + `best_by.note`）—— 不静默、也不让 `best.pt` 缺席。
- **教训**：**"最佳"必须有口径**。当一个工具替你挑"最好的一轮"，它必须：
  ① 说清按什么挑；② 允许你换口径；③ 口径判不出来时报错而**不是猜**
  （猜错会让 best 选到反着的轮次，而回执看着一切正常）。
- **判据**：`tests/test_nqs_path.py::test_best_pt_selection_is_declarable_and_default_is_unchanged`
  （声明的 tag **严格上升** ⇒ best 必落在最后一轮 ⇒ 断言 `best.pt` 与 `last.pt` **逐张量相等**；
  未声明 ⇒ `best_by.tag == "val_loss"`；`ph/std` ⇒ `bad_best_metric` 可教学地拒；
  tuple 形式能通）。

### A58. 周期存档**只写 `last.pt`** ⇒ 被中断的长训练拿不到"最好的那一份"
- **症状**（使用者 2026-10-07，Q4b）：`checkpoint_every` 在训练中途**只更新 `last.pt`**，
  `best.pt` 要等**终态**才写 ⇒ 一次**被中断**（死机/取消/被杀）的长训练
  **只有 `last.pt`**，而 `last` 那一轮的指标常常**不是**最好的那一轮
  （实测：`best` 0.6464 vs `last` 0.8174）—— "周期存档"的意义正是防这个。
- **原因**：周期回调当初只为"别丢参数"设计（只落当前权重），而"最佳快照"一直由
  训练循环在**终态**交给 `_save_checkpoints`。
- **修法**（Q4b）：周期回调**同时**收到训练循环维护的**最佳快照**（`best_state`）与
  它的轮次/取值；当最佳轮**刷新过**时，就在下一个周期**原子写一份 `best.pt`**
  （`save_reason="periodic"`，带上 `best_by` 口径）。
  ⚠ **写的是快照，不是当前 model**：中途那一刻的 model 权重早已不是 best 那一轮的
  ⇒ 写错等于把"最佳"记成"此刻"（假事实）。⚠ 顺序也重要：回调放在 **best 更新之后**。
- **判据**：`tests/test_nqs_path.py::test_periodic_best_pt_is_written_mid_run_so_an_interrupted_run_keeps_it`
  （网络让指标**中途到峰值再回落**，峰值轮是常数 ⇒ 断言：训练**还在跑时**就观测到
  `best.pt` 且 `save_reason=periodic`；观测到的"最佳轮"序列**先出现早期轮、最终追到峰值轮**；
  终态那份 `best_by.epoch` 仍是峰值轮，且权重与 `last.pt` **必须不同**）。

### A59. 续训**没有入口** ⇒ 只能在 `build()` 里自己 `torch.load`（而那条路正是 A7-① 的坑）
- **症状**（使用者 2026-10-07）：`Network` 契约只有
  `build/loss/configure/forward_step/forward_graph/on_step/metrics`，**没有装载存档的钩子**
  ⇒ 续训要在 `build()` 里手写 `torch.load` + `load_state_dict`；而文档推荐的那个片段
  用的是 `strict=False` ⇒ **静默全不匹配**（A7-①：模型等于随机初始化而你看不出来）。
- **修法**（Q7）：显式入口 `run_custom`/`submit_run` 的
  `resume_from=<run_id|.pt 路径>` + `resume_which=last|best`；
  - 装载在 **`build()` 之后**（顺序反了会被 build 覆盖），**默认 `strict=True`**
    ⇒ 结构变过就 `resume_state_mismatch`（带缺失/多出键）；
  - **找不到存档 ⇒ `no_resume_checkpoint`**（**绝不静默从头训**）；
  - 回执/存档记 `resumed_from`/`resumed_which`/`resumed_at_step`（来源可反查）；
  - 需要"只装一部分"⇒ 可选钩子 `Network.load_state(model, state)`。
- ⚠ **踩到的第二个坑（A38 的复发）**：`resume_state` 是**张量 dict**，塞进
  `cfg.extras` 之后 **训练成功、写 `record.json` 时炸**（"不可 JSON 序列化"）——
  与 `on_progress` 同一类。⇒ 方法层用完**就地 pop**，`runner.run_one` 的 `finally`
  **再摘一次**（两道，因为这条只在落盘路径暴露）。
- **判据**：`tests/test_nqs_path.py::test_resume_from_starts_near_the_breakpoint_and_fails_teachably`
  （**自归一化**：续训首轮 loss 离**断点**更近、离"从头首轮"更远；不依赖绝对阈值。
  另含：不存在 ⇒ `no_resume_checkpoint`、`resume_which` 越界 ⇒ `bad_resume_which`、
  结构变过 ⇒ `resume_state_mismatch`）。

### A61. **`metrics` 抛异常会被永久静默关闭** ⇒ "一个 epoch 级 tag 都没有"却看着一切正常
- **症状**（使用者 2026-10-07，白跑一轮）：TB 里**一个 epoch 级 tag 都没有**
  （`fid/sub1`、`famp/sub1`…），而**训练完全正常**（`loss` 在降、**步级** tag 都在）；
  `best_metric` 也因此在整场里"一次都没上报"。**看起来像"作者没写 `metrics`"，其实他写了。**
- **原因**（三条叠加才致命，旧 `trainer.py`）：`except Exception: metrics = None`
  ① **异常被吞掉、不留任何痕**；② **`None` 是永久关闭**（第一次失败后整场再也不调用）；
  ③ **失败不体现在任何回执里**。
- **触发原因极常见**：`metrics` 拿的是**全量负载**（作者 7344 行、一次 256-token 解码就 **OOM**）
  ⇒ "数据一大就静默丢观测"。⚠ **旧 run 之所以有 `fid/sub1`，只是因为那时数据只有 522 行。**
- **修法**（2026-10-07）：
  1. **留痕**：`_metrics_failed()` 记"轮次 + 异常类型 + 消息"（最多 3 条）进
     `stats["metrics_health"]` ⇒ 三条路都能读到：**artifacts**（`nn_metrics_errors`/`nn_metrics_throttled`）、
     **`checkpoint_detail().files_meta.<tag>.metrics_health`**（**只有磁盘、跨重启也查得到**）、
     **`diag.metrics_health`**；顶层还给一句 `metrics_note`（最常问的那句）。
  2. **不永久关闭**：连续失败 `METRICS_FAIL_TOLERANCE=3` 次后**降级**为
     "每 `METRICS_RETRY_EVERY=10` 轮再试一次 + **末轮必试**"；一次成功就**恢复**
     （`recovered_after_epoch`）⇒ 临时性失败（显存一时紧张）能自己恢复，而代价有界
     （他那个失败是 OOM，每轮重试很贵）。
  3. **三种病分得开**（修法完全不同）：`hook=False` ⇒ **没写钩子**；
     `failures>0` ⇒ **写了但每轮失败**（`errors` 里有原因）；两者都正常而 tag 不在 `tags` 里 ⇒ **名字写错**。
- **边界（写进文档）**：`metrics` 每轮拿**全量负载** ⇒ **里面必须分批**（他给的反例：
  同一份代码他的 `_step_one` 是分批的、`metrics` 忘了 ⇒ 只在**大数据**上炸）。
- **判据**：`tests/test_nqs_path.py::test_metrics_failure_is_traced_and_not_permanently_disabled`
  （网络**第 0 轮抛异常、之后正常** ⇒ 断言：训练照常成功 / artifacts **与** checkpoint_detail
  都能看到失败原因 / ⭐ **后面的 epoch 必须真的产出 `fid/ok`**（旧实现下这条一定红：
  整场没有 epoch 级 tag）/ `hook=True` 与"没写"分得开）。

### A60. **判据被自己的环境变量喂成了假象**（"恒定正确"比"红"更危险）
- **症状**（我 2026-10-07 亲历）：为核实"子进程 stdout 变成管道时，`app_entry.main()`
  的 utf-8 `reconfigure` 还成立吗"，我写了个三次对照的探针 —— **三次都是 utf-8**，
  看着像"管道恒定 utf-8"（⇒ 差点据此否掉一条**真实存在**的风险）。
- **原因**：**我自己的 shell 里设了 `PYTHONIOENCODING=utf-8`**，它经 `os.environ`
  继承给了每个子进程 ⇒ 探针测的是"注入生效"，**不是默认行为**。
  剥掉之后结论立刻出来：`PIPE: BEFORE gbk isatty=False → AFTER utf-8 isatty=False`
  （所以 `reconfigure` 是**无条件**的，与 `isatty` 无关）。
- **纪律**：**凡"验证编码/环境相关行为"的判据，必须先剥掉自己 shell 的环境变量**
  （`PYTHONIOENCODING` / `TZ` / `LANG` / `CUDA_VISIBLE_DEVICES` / `ML_TOOLBOX_*` /
  `MLTB_*`），否则会得到"**恒定正确**"的假象 —— 这与"判据在串行执行下空转"、
  "挑的变异与原实现等价"（A13）是**同一族病**：**判据看起来绿，但它没有真的在测那件事**。
- **可复现的样本**：`checks/oneoff/_factcheck_isatty_utf8.py`
  （里面**显式**把 `PYTHONIOENCODING` 从子进程环境里剥掉，并在注释里写明"我第一版就是被它骗的"）。
- ⚠ **未做完的部分（挂起）**：现有判据里还有没有同类（依赖上述环境变量而"恒定绿"）
  **尚未系统扫过** —— 这是重启后的一项遗留工作（见 `docs/交接-重启后待办与验收.md`）。

