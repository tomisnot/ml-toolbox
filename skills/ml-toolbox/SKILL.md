---
name: ml-toolbox
description: >-
  通过 MCP 操控 ML Toolbox（机器学习工具箱：59 个经典方法 + 自动调参 + 神经网络）——
  选数据、组装预处理链、跑方法或自定义网络、读训练过程观测、对比与后台长训练。
  含工具清单、正典工作流、长训练姿势、图任务（GNN）与当前能力边界。
whenToUse: >-
  当需要做机器学习实验（准备数据、选方法、跑训练、比指标、看训练曲线、写自定义网络、
  跑图神经网络）时。软件需处于 **AI 模式**（人经 `python launcher.py` 选「AI 模式」）。
---

# SKILL: ML Toolbox — 机器学习工具箱（AI 操作指南）

> 本文件是 AI harness 的 skill 说明书：连接方式、工具清单、正典工作流、纪律与已知边界。
> **AI 使用中发现文档与实态不符，应回头修正本文件**（自我迭代）。
> - ⭐ **使用侧最佳实践（推荐先读）**：`docs/usage-best-practices.md`
> - ⭐ **软件缺口与接口审查（开发者视角）**：`docs/反馈-ML工具使用体验与接口审查.md`
> 工具清单那张表由 `tests/test_skill_surface.py` 与**真注册表**逐名对账
> （清单少一个/多一个都判红），所以**改工具面必须同时改这张表**。

---

## 什么时候用

- 要在一份数据上**试方法、比指标**（分类/回归/聚类/降维/异常/时序）；
- 要**训练神经网络**，或写一个**自定义结构**（残差、多分支、图卷积…）；
- 要看**训练过程**（loss/val_loss/grad_norm/lr/dead_relu/逐层梯度）而不是只看最终分数；
- 要**长训练**（跑得比一次工具调用还久）；
- 要做**图任务**（节点分类）。

## 什么时候**不**适用（先看清，免得白跑）

- **软件没在 AI 模式**：写操作会被 `authority_locked` 挡住。这不是故障——写权出厂锁死，
  **只有人类能开闸**（AI 无法自解锁，这是刻意的安全边界）。
- **链路预测 / 图分类**：本工具支持的是**节点分类**式的图任务；"样本是边对"或
  "一行=一整张图"这两类**不在能力面内**（标签的单位不是"行"）。别硬试。
- **要拿回某个 epoch 的权重**（`keep_best`）：**没做**。今天只能查到 `best_epoch` 是第几轮。
- **对新数据预测 / 导出预测**（`predict_new` / `export_prediction`）：**工具面没有接线**
  （GUI 有）。别以为能调。
- **自动调参**（`tune_method`）：MCP 工具面**没有**（GUI 有）。

---

## ⭐ 第 0 步：开工前 90 秒自检（**照做，能挡掉一轮里 80% 的返工**）

> 这一节的每一条都来自**真实踩坑**（一次 5 小时级 NQS 训练会话），
> 不是泛泛的注意事项。**先跑完这 6 条再写第一行训练代码。**

| # | 自检 | 为什么（实测代价） |
|---|---|---|
| **1** | **这是不是一轮长训练（>1 小时）？** → 若是，**必须显式给 `checkpoint_every`**（如 200）：不给就**只在训练终态**写 `checkpoint/`，中途死机＝**全丢**（实测：一个跑到 284 epoch 的 run，`runs/<id>/` 下只有 `tb/`）。见下文 §中途存档。 | 一次 5 小时训练裸奔，死机即全丢 |
| **2** | **要对比吗？和谁比？** 先把"同一份数据、同一口径下的上界"算出来（不依赖模型的天花板：恒模上界、oracle 截断…），否则你无法判断模型是好是坏 | 我一度拿"**另一个数据集**上的最好成绩"当靶子，方向错了一整轮 |
| **3** | **主指标 vs 代理指标分清了没？** 见 §指标层级。**代理指标的下降/平缓不能否定主指标** | 我用"系数相关"否掉了一个**训练集 `fid` 已 0.86** 的方向（主指标才是判据） |
| **4** | **量纲/口径自检**：你的度量函数在一个**已知答案**上是否返回预期值？（归一化、`1/M`、恒模 vs 实际模长） | 我两次把口径写错（差 256 倍），得出"模型全错"的假结论 |
| **5** | **有不可导操作吗？** 输出过 `sign()`/`argmax()`/硬阈值后，**梯度还到得了那个分支吗**？实测 `sign()` 让主损失对该分支的梯度**精确为 0** | 一个分支白训了 3 次（各 30 分钟–6 小时），根因是一条 `sign()` |
| **6** | **路径一律绝对**；网络代码里的 `print` **只用 ASCII** | 相对路径被解析到错位置，**等了 120 分钟**才发现；非 ASCII `print` 在 GBK 控制台让**整个后台任务崩掉** |

**⇒ 更细的版本见 `docs/usage-best-practices.md`。开发者视角的缺口见 `docs/反馈-ML工具使用体验与接口审查.md`。**

---

## ⭐ 指标层级（**最容易被误用的一条**）

**同一轮实验里，必须同时看这几个层次，不许只盯一个：**

| 层级 | 例子 | 能否单独定罪 |
|---|---|---|
| **主指标** | 目标函数本身（`fid/sub1`、`acc`、`RMSE`） | ✅ **最终判据** |
| **主指标（训练集）** | `fid/step`、`loss` | ✅ 与基线**训练集口径**可比；⚠ 与"留出"不可直接比 |
| **代理指标** | "子目标学得对不对"（我用的"系数相关"） | ❌ **只能参考**；实测**代理降而主指标升** |
| **分解量** | 主指标 = A × B（我的是 `fid = famp × 相位因子`） | ✅ **必须同时报**，它把"两半各自好不好"分开 |
| **泛化差** | train − hold | ✅ 判断过拟合/欠拟合 |
| **成本** | s/epoch | ✅ 决定你能不能做对比实验 |

**三条硬纪律**：

1. **代理指标不能否定主指标。** 若主指标在涨而代理平/降，**先信主指标**（代理可能本就不是主目标的充分统计量）。
2. **跨数据集比主指标无意义。** 数据集越难，同一个模型的指标越低。要比就比
   **"相对同一份数据的上界"**，或比**分解量**。
3. **量纲自检**：任何自定义度量都要在一个**已知答案**的输入上验证（例：用精确目标代进去，
   必须得 1.0；用随机基线代进去，必须 ≈ 常数）。

---

## ⭐ 中途存档（`checkpoint_every`：**框架自己做，别再往网络代码里塞 `torch.save`**）

> **2026-10-07 起这是一等参数**：此前框架**只在训练终态**写 `checkpoint/`，
> 长训练必须自己用 `on_step` 打补丁（实测：一个跑到 284 epoch 的 run，
> `runs/<id>/` 下**只有 `tb/`** ⇒ 中途死机＝全部参数丢失）。

```jsonc
run_custom(..., checkpoint_every=200)        // 每 200 个**优化步**原子落盘 last.pt
submit_run(code_path=…, checkpoint_every=200)  // 后台长训练同样给（第一等参数）
```

- **单位是"优化步"**（一个 batch 一次 `optimizer.step()`），**不是 epoch**：
  长训练里一轮可能几十分钟，按轮数存档粒度太粗。⚠ **图任务里 1 epoch = 1 步**
  （整图前向、没有 batch）。
- **`best.pt` 的语义没变**：仍按**验证损失**更新（与 `last.pt` 各自独立）。
- **默认 `0` = 关**（与从前逐位一致）⇒ **长训练请显式给**（200 量级是合理起点）。
- **原子写**：先写同目录临时文件再 `os.replace` ⇒ 写一半被杀**不会**留下半个档、
  **也不会**把上一份好的覆盖掉。
- 落盘失败**不拖垮训练**（与 TB 同纪律），但**会在存档里留痕**：
  `nn_checkpoint_errors` / `diag.checkpoint_periodic`（**不静默**）。
- **对账**：`checkpoint_detail(run_id=…)` 的
  `last_saved.saved_at_step` / `files_meta.<tag>.saved_at_step` 就是
  "这份存档落在第几步"（训练**还在跑**时也读得到 ⇒ 能判断它是不是最新的）。

**老写法（仅当你要存到框架管不到的地方，如自己的绝对路径）**：
`on_step` 里自己 `torch.save`——但要绝对路径、要 try/except、且**别再重造一份
`last.pt`**（那会与框架的写盘互相覆盖）。

**⚠ 服务端 `print` 现在读得到**：`read_job(job_id, logs=N)` 的 **`logs_tail`**
就是那次 job 的 stdout/stderr 尾巴（默认 10 行，每行带 `[out]`/`[err]`；`logs=0` 关；
**桶有界**：默认 200 行/任务，`logs_n` 是总行数，截断看 `logs_truncated`）。
⇒ 网络代码里的 `print("[snap] step=…")` 这种诊断**不用再只写盘**——它直接可查。

⚠ **它只覆盖 Python 层的 print**：**C 层**（torch / CUDA / OpenMP / MKL）与**子进程**直接写
fd 1/2 的输出（例如 `vrtc64_120_0.alt.dll … or one of its dependencies`、
`Insufficient system resources`）**不在里面** ⇒ **`logs_tail` 空 ≠ 没报错**。
要对齐那种输出：用 `read_job` 回的 **`started_at`/`finished_at`** 去权威的控制台/日志文件里按时间找。

---

## ⭐ 续训（`resume_from`：**框架已有入口，别自己 `torch.load`**）

```jsonc
run_custom(code_path=…, dataset_id=…, resume_from="custom_network-2334125-0",   // 或 .pt 绝对路径
           resume_which="last")                                                // last（默认）/ best
submit_run(code_path=…, resume_from=…)      // 长训练同样给
```

- **装载在 `build()` 之后、训练之前**，**默认 `strict=True`** ⇒ 权重与这次 `build`
  不匹配就**响亮失败**（`resume_state_mismatch`，带缺失/多出键）——**不会**静默随机初始化。
- **找不到存档 ⇒ 可教学地拒**（`no_resume_checkpoint`）：**绝不静默从头训**
  （"以为在续训、其实从零"跑得很好，你只会觉得"这轮收敛慢"）。
- **回执与存档都记得"从哪续的"**：`resumed_from` / `resumed_which` / `resumed_at_step`
  + `artifacts["nn_resumed_from"]`。
- **判据（他给的）**：续训后**起始 `loss` 应贴近断点**而不是初值
  （实测 99.68 vs 1017.6，一眼可辨；本仓判据用的是"离断点更近、离从头更远"的**自归一化**比较）。
- 需要**只装一部分**（改了结构/分片）⇒ 实现可选钩子 `Network.load_state(model, state)`。

**⚠ 旧写法（`build()` 里自己 `torch.load`）降级为"特殊情形的退路"**：只在
① 存档来自**旧框架**、② 你要装的不是 `checkpoint/*.pt`（自己的快照格式）时用。
用的时候 **`ck["state_dict"]` 这一层仍然必须取对**（见下面"两个坑"第 1 条，
那条坑**依然有效**：`strict=False` 会静默全不匹配）。

**⚠ 两个坑**：

1. **存档顶层是 `{'state_dict': …, 'params': …}`**。
   **`load_state_dict(ck)` 默认 `strict=True` 会抛** `RuntimeError`（实测列出 121 个 missing key）；
   **真正静默的是 `strict=False`**（missing=121 / unexpected=16，**无报错**），
   而**旧版本文档里推荐的正是 `strict=False`** ⇒ 坑是按文档写出来的。
   **必须取 `ck['state_dict']`**（`checkpoint_detail` 的 `load_hint.state_dict_key` 会直接告诉你）；
   若用 `strict=False`，**必须检查返回的 missing/unexpected 并在非空时报错**。
2. **类定义体用了 `nn`/`torch` 的文件，在框架外 import 会 `NameError`**
   （框架只在 `build()` 时注入这两个名字）。要让本地脚本也能 exec，
   **显式 `import torch` / `import torch.nn as nn`**。

**验证续训生效（判据）**：看**起始 `loss`**——应从**断点附近**开始而不是初值
（实测：续训首值 99.68 vs 从头 1017.6，一眼可辨）。

**⚠ 更正（我原先写错过）**：`run_id` 带毫秒（`core/runner.py:52-53`）⇒ **每次重跑都是新 id**，
**不存在"两条训练混在一根轴上"**。真实的坑是**同一 run 内两族 tag 的 step 轴不同**：
`loss`/`val_loss`/`grad/*` 的 step = **epoch 序号**；`on_step` 上报的 `fid/step` 等
= **全局 batch 步**（实测差 ~277 倍）。

⇒ **读法**（回执里现在直接写着，不用自己推）：

```jsonc
"latest_steps": {"loss": 283, "fid/step": 78410},
"step_units":   {"fid/step": "batch", "dist/step": "batch"},   // 不在表里 = epoch 级
"axes": {"epoch_tags_n": 51, "batch_tags": ["fid/step", …],
         "per_epoch_estimate": 276.3}                          // 那个"差 277 倍"的来源
```

跨族给 `tags=[…]` 对齐时，回执会带 **`compare.axis_warning`** —— 因为轴取的是两族步号的
并集，**粗粒度那一族会大片 `null`**（看起来像"这段没有 loss"，其实是**不在同一个 step 尺度上**）
⇒ **同族之间比，跨族只看趋势**；且每 tag 按点数截断（默认 500），
**不同 tag 的覆盖区间可能差几百倍**（`loss` 覆盖最近 500 轮，`fid/step` 只覆盖最近约 2 轮）。

---

## ⭐ 结果拿回来（训练完 → 评估）

```
run_custom(..., persist=true, checkpoint_every=200)
                                       # 终态写 checkpoint/{best,last}.pt（完整，含 bias/LN）
                                       # + 每 200 步原子落盘一份 last.pt（中途存档）
  → checkpoint_detail(run_id=…)        # 在不在、多少张量、含不含 bias、每份**落在第几步**
                                       # （`last_saved` / `files_meta` / `load_hint`）
  → eval_custom(run_id=…, code=…)      # 装载模型交给你的代码；默认继承训练负载
```

**⚠ `best.pt` 是按 `val_loss` 选的**——而 `val_loss` 的尺度可能与你的主指标不一致
⇒ **`best.pt` 常常不是主指标最好的那一轮**（实测：按 `val_loss` 挑的轮在 `fid` 上差很多）。
**判据用主指标时，请自己用 `last.pt` 或按主指标挑轮。**
**⚠ 装载时用 `ck["state_dict"]`**（`checkpoint_detail` 的 `load_hint` 直接给这句话）。

**⭐ 想让 `best.pt` 按你的主指标选（2026-10-07 起可声明）**：

```python
class Network:
    best_metric = "fid/sub1"            # 方向自动判（判不出来会报错，**不许猜**）
    best_metric = ("ph/std", "lower")   # 方向没有好坏的量：显式给 higher/lower
    def metrics(self, model, ctx):      # 值从这里来（每轮一次；该轮没报就跳过那轮）
        return {"fid/sub1": …}
```

- 声明后 `best.pt` **按那个 tag** 选；`checkpoint_detail` 的
  `files_meta.best.best_by` 报 **tag / direction / value / epoch / step**，
  `val_loss_best` 给对照（两者常常**不是同一轮**）。
- ⚠ **未声明时仍按 `val_loss`**（与从前逐位一致）—— 声明是显式选择，不是新默认。
- 声明了却整场一次都没上报 ⇒ **退回 `val_loss` 并留痕**（`nn_best_metric_missing`）。

---

## ⚠ 别重造轮子：框架已有实时观测设施

- **数据实时写入**：训练循环**每轮 flush** 事件文件（实测训练中每 20 秒 +30 KB）；
- **实时读取路由**（`ml_mecha/monitor_http.py`）：**`/latest`**（最新一轮，**专为 AI 研判**）、
  **`/series`**（整条曲线，给面板折线）、`/summary`、`/runs`；
- **给人看的界面**：TensorBoard 辅进程（**惰性启动**、端口文件发现、起不来会如实说）；
- **AI 侧**：`describe_training`（读同一份事件文件，**训练中就能读到新点**）。

**⇒ 不要自己去解析 `events.out.tfevents.*`**（我就这么干过，还因此踩了"两段曲线混在一起"的坑）。
优先 `describe_training` 的 `curve` / `convergence` / `pick`。

**⚠ 读数值前先看轴**（`latest_steps` + `step_units` + `axes`）：同一个 run 里有**两种步号**——
epoch 级（`loss`/`val_loss`/`lr`/`grad/*`）与 **batch 级**（`on_step` 上报的 tag）。
`axes.per_epoch_estimate` 就是两者的比值（实测 ~277）。**跨族别并排比**（见 §续训的更正）。

**⚠ 回执大小由你自己控**：`curve` 默认 **`brief`**（核心 tag + **有判读的 tag**），
实测一个 58 tag 的真 run **38 891 → 10 664 字符（−72.6%）**，而 `convergence` 判读**一条不少**；
要全量写 `curve="all"`，只要某几个写 `curve_tags=["fid/sub1"]`（`off` = 一个形状都不回）。
`pick` 只省**非必填**的大块（如 `series`）——**必填键**（`latest`/`curve`）只能靠它们自己的旋钮
（`grad_layers` / `curve`）。

---

## ⭐ 三种正典情形（照抄骨架）

### 情形 A：内置方法、短任务

```
describe_methods → describe_method(name) → prepare_dataset → run_method → describe_run
```
`run_method` 是**同步**的，适合秒级–分钟级。

### 情形 B：自定义网络（`run_custom`）

```python
class Network:
    primary_metric = "fid/sub1"        # 声明主指标（不上报就取字典序第一个）
    best_metric = "fid/sub1"           # 可选：让 best.pt 按**这个** tag 选（默认按 val_loss）
    metrics_every = 3                  # 贵指标降频；末轮一定算
    def build(self, n_in, n_out, *, nn, ctx=None):
        # ⚠ 用 ctx["payload_shapes"] 推维度，别写死（换数据会静默跑偏）
        ...
    def forward_step(self, model, batch, ctx):     # 训练/验证/探针/预测**唯一入口**
        return out, target                          # 两者都给
    def on_step(self, model, step, ctx):            # 每步；做**步级指标**（存档交给 checkpoint_every）
        return {"my/step": ...}
    def metrics(self, model, ctx):                  # 每轮；贵指标放这里
        return {"fid/sub0": ..., "fid/sub1": ...}
```

**⚠ `metrics` 每轮拿的是「全量负载」**（`ctx["payload"]`，**不是一批**）⇒ **里面必须自己分批**：
实测反例——7344 行 + 256-token 解码器**不清边就 OOM**；而这类失败在旧版里是**静默**的
（整场一个 epoch 级 tag 都没有，看起来像"你没写 `metrics`"）。
现在它**留痕且可查**：`checkpoint_detail(...).files_meta.<tag>.metrics_health`
（`hook`/`failures`/`errors`）与 `artifacts["nn_metrics_errors"]`；
而且**只降级、不永久关闭**（连续失败 3 次后每 10 轮 + 末轮再试，**会自己恢复**）。

**⚠ `best_metric` 与 `metrics_every` 必须相容**：声明 `best_metric="fid/sub1"` 而
`metrics_every=3` 且只跑 3 轮 ⇒ 该 tag 可能**一次都没上报**（会**有留痕地**退回 `val_loss`）。
要么把 `metrics_every` 调小、要么跑够轮数。

**⚠ `resume_from` 只恢复权重，不恢复优化器状态**（也不恢复 `scheduler`/步数）⇒
续训第一轮可能有**暂态回退**（实测：源 `dist 7.88` → 续训 1 轮 `10.37`）。
崩溃恢复时考虑**降 lr 或先预热几百步**再回到原 lr。

**提交前**：`run_custom(..., dry_run=true)` 看形状轨迹；再加 `warmup_steps=60` 真反传，
看 `loss_curve` / `grad_norm_curve` / `verdict`。**长训练走 `submit_run`（后台）+ 看门狗。**

### 情形 C：长训练（>10 分钟）

```
submit_run(code_path=…, dataset_id=…, overrides=…, persist=true, checkpoint_every=200)
  → job_id
read_job(job_id, wait="terminal", timeout_s=540)      # 一次阻塞到终态；超时不是错误
# 超 9 分钟的：
pwsh(run_in_background): python -m ml_mecha.watch <job_id> --poll 60   # 到终态才退出
```

**⇒ 长训练的提交前 checklist**：① **`checkpoint_every=200` 给了**（中途存档，自检 #1）；
② `persist=true`；③ `resource_guard.max_peak_mb` 声明；④ 记下 `job_id` 与 **`dataset_id`**
（**重启后都失效**，`dataset_id` 要重新 `prepare_dataset`——同内容会得到**同一个 id**）。

---

## 工具清单

> 共 **20** 个。表里的名字必须与注册表逐字一致（有判据钉着）。

| 工具 | 干什么 | 长什么样的大结果 |
|---|---|---|
| `describe_methods` | 列可用方法（59 个） | 名字/族/任务，不列参数 |
| `describe_method` | 某个方法的可调参数与取值范围 | 参数表 |
| `describe_dataset` | 已准备数据的规模/指纹/预处理链逐步参数/实际划分口径；**带负载时报每个负载的形状与 dtype** | 摘要，不内联数据 |
| `describe_run` | 按 `run_id` 回看一次运行 | 指标 + 产物类别名（数值不内联） |
| `describe_training` | **训练过程观测**：逐 epoch 的 loss/val_loss/grad_norm/lr/dead_relu + **逐层梯度**（默认只给**前 3 + 后 3** 与总数）+ **网络自己上报的自定义 tag**。⭐ 还默认给 **`curve`（每个 tag 的「曲线形状」**：起点/末值/**末段斜率**/是否单调/平台占比/趋势）与 **`convergence`**：**末段还在爬的非 loss 指标会明确警告「可能欠训练——别按终值判断饱和」**；loss 类在爬则说「发散」（两者处置相反）。⭐ `best_by="fid/sub1"` 按**任意 tag**挑最佳 step（并对照 val_loss 的选择——两者常常不是同一轮）；⭐ `compare=[run_a, run_b]` + `tags=[…]` **多 run 曲线对齐**（同 tag、同一根 step 轴；缺失处 `null`，**不插值**） | 默认只给**最新一轮**；`include_series` 才给序列；逐层要全量写 `grad_layers="full"`；可用 `pick` 只取几条 |
| `describe_networks` | 列出自定义网络（名字/指标/源码行数/用过几次） | 清单，不含源码 |
| `describe_network` | 按名字取回网络**源码**（改一版再跑，免重贴） | 源码 + 来源 run |
| `checkpoint_detail` | 某次运行**存下来的模型**在不在、多少张量/参数、含不含偏置、在哪块设备上训的、**每份落在第几步**（`last_saved`/`files_meta`）、**该用哪个键 load**（`load_hint`） | **权重不内联**；真要用模型走 `eval_custom` |
| `eval_custom` | **装载某次训练的模型，交给你的代码去评估**（采样/算能量/换个尺寸外推） | 它上报的指标 |
| `prepare_dataset` | 登记数据 + 拟合预处理链；可**自己组装**这条链（选步骤/换顺序/写自定义步骤/**分组留出**/给图边表/**给逐行负载**） | `dataset_id` + 链摘要 + 划分口径 + 负载形状 |
| `run_method` | 跑一个内置方法（**同步**，适合短任务） | `run_id` + 指标 |
| `run_method_batch` | 顺序跑多个方法 | 每个方法的 `run_id` + 逐项失败原因 |
| `compare_methods` | 多方法对比 | 对比表 + 最优（**要比多个就用它**，别逐个拉） |
| `run_custom` | 跑一段**自定义网络代码**（AI 写结构，系统管训练与观测）；可给 `device`；**`dry_run=true` 先试跑**（回形状轨迹，不落盘）；也可 `code_path` 从文件读 | `run_id` + 指标（自定义任务时是网络上报的） |
| `submit_run` | 把一次运行提交为**后台任务**（给 `method` 或 `code`）——**长训练就走这条** | `job_id`（立刻返回） |
| `read_job` | 查后台任务进度/结果。⭐ **要等它跑完就给 `wait="terminal"`**（阻塞到终态，上限 540s；超时**不是错误**）；多个任务用 `job_ids=[…]` + `wait="any"`。⚠ **默认只回摘要**：`progress_latest`（含 loss/val_loss）+ 最近 `tail`（默认 8）条 + 真实条数 `n_progress`/`n_epochs`/`truncated`。⭐ **`logs_tail`**：服务端 stdout/stderr 的最后 `logs` 行（默认 10）—— 网络代码里的 `print` 在这里；**桶有界**（默认 200 行/任务，`logs_n` 是**总行数**，截断看 `logs_truncated`/`logs_cap`）。⚠ **只覆盖 Python 层**（C 层/CUDA/子进程写 fd 的不在）；**空 ≠ 没报错**，用 `started_at`/`finished_at` 去对齐 | 要全量历史写 `detail="full"`；要更多条写 `tail=N`；要更多日志写 `logs=N`（`0` = 关；**调大也不会超过桶容量**）；旧名 `wait_s` 等价于 `wait="terminal"`+同秒数 |
| `cancel_run` | 请求取消后台任务（**排队中的也能取消**：会在开跑前就撤掉） | 明确回"已请求"（在 epoch 边界生效，不谎报"已取消"） |
| `set_job_limit` | **热改并发名额**（不必重启权威；调大立刻唤醒排队者） | 改动前后上限 + 正在跑/排队数 |
| `run_queries` | **一次往返跑一串只读查询**（列方法→看参数→查数据集→看训练） | 每步自带结果与短摘要 |
| `clear_current_state` | 把配置台面清零（数据集引用/方法/覆写/种子/设备） | 逐键"清空前值 + 清空后读回值" |

---

## ⭐ 省往返：先把机械重复劳动交给软件

> 依据 `docs/建议-AI往返成本与训练观测面.md`（一次真实工作里**形状类错误犯了约 15 次、
> 每次一个工具往返**）。下面四件就是为此加的。

**① 写网络代码 → 先 `dry_run`（最省往返的一件）**

```jsonc
{"code_path": "net.py", "dataset_id": "ds-001", "dry_run": true}
```

它**编译 + 建模型 + 用真实负载跑几次一次前向**，回：

- `shapes`：**逐模块**的输入/输出轨迹
  （`{"module":"enc","in":[[1,6,5]],"out":[[1,6,8]]}`；`out` 为 `null` 表示
  **它进去了没出来** ⇒ 失败就在它里面）；
- `error_kind` / `error_hint`：结构化归因，**指认那个"进去了没出来"的模块**；
- `n_params` / `loss_first_batch` / `batch_sizes_checked`。

⭐ **再加 `warmup_steps=N`：真反传 N 步**（默认 0 = 只验前向）。前向试跑**抓不到
"训练动态"类错误** —— 而那是长训练里最贵的（实测：loss 在 step 120 处从 0.009 涨到
113；或 `nll → 0` 的退化解，每次代价是上千 epoch + 一次取消）。回执照给
`loss_curve` / `grad_norm_curve` / `params_changed` / `any_nonfinite` 与 **`verdict`**：

| verdict | 含义 |
|---|---|
| `⛔ …会发散` | loss 涨一个数量级，或出 NaN/Inf |
| `⚠ 梯度范数涨一个数量级` | Adam 会掩盖 loss 上涨 ⇒ 梯度是更早的信号 |
| `⚠ loss 塌到 ~0 且梯度≈0` | 可能是**退化解**（模型恒输出同一分布），别当成功 |
| `✅ 前 N 步在下降` | ⚠ **只是这个窗口内健康**，不代表 N 步之后也健康（实测同类配置在 step 120 才发散）⇒ 要更硬的把握就把 `warmup_steps` 提到覆盖"损失分段切换"那一步，或直接开短训练看曲线 |

⚠ 边界不变：**不反传之外的写入一概没有**（不落盘、不计入实验记录）。

⚠ 三条边界：**不反传、不落盘、不计入实验记录**（不会污染 `runs/` 与对比）。
⚠ 试跑 `ok=false` **不是命令失败**，而是"你的代码跑不起来"这个**问题的答案**——
形状轨迹会一起给你，**别把它当异常**。
⚠ 默认试 **1 / 2 / 训练批大小**三档：**批次相关的错只有真数据才抓得到**
（实测：本地 B=3 全绿、真实 B=2 就错）。要指定就用 `batch_probes`。

**② 调参跑多个组合 → `code_path`**

网络源码存成一个文件，之后只报路径（受**已有**数据根授权管辖，不新增授权面）。
同一份 9 KB 代码不必发 N 遍；回执带回 `code_path` / **`code_source`**（`file` / `inline`）
与 `code_lines`，便于与文件版本对齐。

⚠ **`code` 与 `code_path` 只能给一个**（同时给会报 `bad_code_source`）——
旧行为是"静默取 `code`"，那会让"我改了文件"变成假的。
⚠ 若你换过参数却收到 "Field required" 这类校验错，多半是**客户端缓存的旧工具表**
⇒ 重连/刷新一次会话（工具表在会话建立时取一次）。
⚠ 形状轨迹太长时（几十模块 × 3 档批量）用 `shapes_verbosity: "summary"`：
只回**逐模块唯一形状对**，回执能小一大截；顶层 `shapes` 也不再与 `runs[]` 重复。

**③ 训练动态要看"步"这一级 → `on_step` 返回 dict**

```python
def on_step(self, model, step, ctx):
    return {"nll/step": float(loss_now)}      # 步级指标 ⇒ 写进 TB
```

`describe_training` 立刻看得到（它读全部 tag）。**为什么重要**：NQS 这类任务动辄几千步，
而**塌陷常发生在前 2% 的步数里** —— 只有轮级观测的话你会发现得很晚。
`Network.metrics` 仍是**轮级**（贵指标放那里，或用 `metrics_every` 降频）。

**④ 评估用哪份数据 → 默认继承训练那份**

`eval_custom` 不给数据时**自动继承训练时的负载**（从 checkpoint 里的**声明**重建，
HDF5 会按原路径重读）；要换新数据就显式给 `payload` / `dataset_id`。
回执的 **`payload_source`**（`given` / `inherited_from_training`）与
**`payload_source_reason`**（`explicit` / `inherited` / `no_declaration` /
`declaration_unreadable` / `no_checkpoint`）说清**用的是哪一份、以及为什么**——
`no_declaration` 通常意味着**那次训练的存档早于"声明随存档走"这次改动**。

---

## 标准工作流（照做即可）

```
① describe_methods                       # 有哪些方法（可按 family 过滤）
② describe_method(name=…)                # 拿到**真实**的参数名与范围（别猜）
③ prepare_dataset(source=…, target=…)    # 登记数据；要自己组装链就给 pipeline=
④ run_method(method=…, overrides={…})    # 短任务；长任务改 submit_run
⑤ describe_run(run_id=…)                 # 指标与生效参数
⑥ describe_training(run_id=…)            # 过程数值（判断该不该早停/是不是爆了）
```

⭐ **判断"够不够训练"要看曲线形状，不要看终值**（实测教训：有人只读 `latest`
一个点 + 跨 run 比终值，据此把 epoch 从 900 砍到 250 —— 而那条曲线**末段仍在爬**，
是严重欠训练；补到 1500 epoch 后 `fid/sub1` 从 0.68 涨到 0.79）：

```
describe_training(run_id=R)                       # 看 curve[tag].trend / convergence 警告
describe_training(run_id=R, best_by="fid/sub1")   # 按任意 tag 挑最佳轮（别默认 val_loss 那轮）
describe_training(compare=[R1, R2], tags=["fid/sub1"])   # 两条轨迹对齐着看
describe_training(run_id=R, tags=["loss","fid/sub0"])    # ⭐ 同一 run 内多 tag 放同一根轴
```

- `convergence` 里 `kind="underfit"` ⇒ **延长 epoch 再下结论**（别按终值说"饱和了"）；
  `kind="diverging"` ⇒ 配置问题（loss 类在爬）；`kind="noisy_tail"` ⇒ 本窗口在降、
  末段只是抖动（**不是**发散）；方向未知的量（`grad*`/`ph/std`/`weights_norm/*`）
  **不给诊断**，趋势照给。
- ⚠ **`latest` 里的数不一定在同一步**：各 tag 的求值步长可以不同（`metrics_every`
  降频时 `fid/*` 每 N 轮才记一次）⇒ 每个 tag 的步号在 **`latest_steps`** 里。
  **要把几个数放一起比，就用 `tags=[…]`**，看 **`aligned_at_step`**（所有 tag 都有
  值的最后一步）与 **`aligned`**（那一步的值）。**没有公共步时 `aligned_at_step=null`
  ——那就别把它们并排读**（并排读会把"各自都真"的数读成"互相矛盾"）。

**要比多个方法**：直接 `compare_methods`（它给对比表），不要 `describe_run` 拉 N 次再自己对齐。

**要在干净台面上开始**：先 `clear_current_state`（否则会踩到上一轮留下的数据集/方法/覆写）。

⭐ **`dataset_id` 是内容寻址的（`ds-<fp12>`）—— 幂等**：
- 同一份数据（**哪怕换个 `name`**）再登记 ⇒ **同一个 id**，回执里 `reused: true`；
  所以"上一轮用的是哪份数据"永远对得上，**不必靠记忆**。
- 可读的名字走 `label` / `aliases`（同一份数据用过的**所有**名字），
  `describe_dataset` 的 `ledger` 里有 `fp` / `label` / `aliases` / `recent_runs`
  （用过它的运行 + 结果）——这就是"运行清单"：**指纹 → 名字 → 运行 → 结果**。
- `restorable=true` 的来源（`csv` / `synthetic`）**重启后仍能直接用同一个 id**
  （框架按台账声明就地重建）：⚠ `frame` / `arrays` 是**进程内**对象，重启后
  重建不了 ⇒ 回执会说明原因（那时重新 `prepare_dataset` 会得到**同一个 id**）。

---

## 长训练：发信号 → **一次等到终态** → 更长的交给看门狗

LLM 的一次工具调用本质是 HTTP 请求，**客户端必须有超时**（本实例 **10 分钟**，
见 `launcher.py` 的 `TOOL_CALL_TIMEOUT_MS=600000`）⇒ "无限等"天生不成立。
但**不必轮询**：等待原语已经接在 `read_job` 上。

```
submit_run(method=… 或 code=… 或 code_path=…)  → job_id   # 发完就走
read_job(job_id, wait="terminal", timeout_s=540)          # ⭐ 一次阻塞到终态
    → state=done/failed/cancelled + run_id + result_ref + metrics
    → 超时**不是错误**：回 timed_out=true + 当前进度，可以再调一次
read_job(job_ids=["job-3","job-4"], wait="any", timeout_s=540)   # 谁先完先报谁
read_job(job_id)                → 只想瞄一眼（立刻返回，默认摘要）
read_job(job_id, detail="full") → 才给全量逐代历史
describe_training(run_id)       → 想查哪一代/哪个指标都行
cancel_run(job_id)              → 在 epoch 边界停下（协作式）
```

- **上限 540 s**（客户端 600 s 之内留余量）。超了**不报错**：夹到 540 并在
  `wait_note` 里说明。**绝大多数 250 epoch 的训练（实测 5–8 分钟）一次就等到。**
- ⭐ **更长的（900 epoch / 一小时级）用看门狗**——它是独立进程，不受工具超时约束：
  ```
  pwsh run_in_background: python -m ml_mecha.watch <job_id> --poll 15
  ```
  它到终态才退出，**退出时 harness 会自动注入一条完成通知**（这就是"完成事件"的
  今天可用形态）⇒ 期间你可以去做别的事，**不要轮询**。
  - 也能传 `run_id`（跨重启稳定）；`--timeout N` 可给上限（默认一直等）。
  - 退出码：`0` 终态；`3` 超时仍在跑（不是失败）；`2` 句柄查不到（会给你盘上最近的 run）。
  - 它**只读**：不提交、不取消、不写任何域状态。
- 后台任务**终态自动落盘** ⇒ 就算后来丢了 `job_id`，结果还在 `runs/<run_id>/`；
  **重启后 `job_id` 会失效**（作业表只活在会话里）⇒ 那时 `read_job` 会回一句
  "盘上最近的 run 是 …"，用 `describe_run(run_id=…)` 接着查。
  ⚠ **重启后 `dataset_id` 也失效**（数据集只活在会话里）⇒ 要**重新 `prepare_dataset`**
  再提交；否则命令会失败，而 `read_job` 会把那条失败**原样告诉你**（不再静默）。
- **`read_job` 的语义**：`state=done` 只表示"job 跑完了"，**不等于命令成功** ——
  命令层的失败会以**可教学错误**抛出（kind 是命令自己的，如 `unknown_dataset`、
  `dataset_path_denied`）。看到它就按它说的修，别当成"训练跑完了但没结果"。
  若出现 `silent_terminal`：那是**框架自检失败**（既没 run、也没说清失败），
  请上报并把这次调用原样贴出来。
- **并发上限默认 4，但超额不再报错 —— 会入队**（`submit_run` 回执带
  `queued` / `queue_position`）。⭐ **要改上限就 `set_job_limit(n)`**（**热改**，
  不用重启）：调大立刻唤醒排队的任务；调小**不打断**已在跑的。
  ⚠ 排队中的任务 `read_job` 会显示 `queued=true` + `queue_position`，
  而 `state` 仍是框架给的 `running`（**线程起了 ≠ 命令开跑了**，这两个词问的
  不是同一件事，`wait_note` 里会说明）。**排队期间可以 `cancel_run`**（会在开跑前取消）。
- 提交前会**同步预检写权**：LOCKED 下当场拒（不会"提交成功再发现失败"）。
- **显存（OOM 是能事前避开的）**：给 `resource_guard.max_peak_mb=<MiB>` 声明你预计的
  峰值 ⇒ 框架**提交前**对照实测可用显存判一次，明显不够就**可读地拒**
  （`cuda_oom_predicted`，带"要多少/剩多少/总多少"三个数）。⚠ 框架**不猜**你网络的
  峰值（猜出来会假装精确）；第一次可以照着上一次运行的实测峰值填——回执/存档里有
  `peak_mb`（只在真跑 CUDA 上时才有，CPU 上**不给**这个数，也不给 0）。
- **大负载 + 小显存：让框架分块**（别在自己网络里手写）：
  ```python
  def metrics(self, model, ctx):
      return ctx["chunks"].reduce(ctx["payload"],
                                  lambda blk: {"fid": _fidelity(model, blk)},
                                  mode="mean", budget_mb=256)   # 或 mode="sum"
  ```
  `mode="mean"` ⇒ 框架算 `Σ (块行数/总行数) × 值`（**逐样本量的正确口径**）；
  `mode="sum"` ⇒ 直接相加。**不要在返回里自己乘权重**（那是权重平方）。
  预算也可用 `resource_guard.chunk_budget_mb=…` 统一给。分了几块会写进存档
  （`nn_chunks`），回执/`describe_run` 看得到。

---

## ⭐ 三条省回合/省 token 的纪律

1. **`run_queries`**：一串**只读**查询一次往返跑完（"列方法→看三个方法的参数→查数据集"）。
   ⚠ 它**只读**：写操作（准备数据、跑方法、改配置）仍要单独调用。
   ⭐ 其中一个 op 是 **`jobs`**：**作业清单**（`job_id` / `command` / `state` /
   **`run_ids`**）——回答"我提交过什么、各产出哪个 run"。⚠ 只含**本会话**的作业
   （注册表在内存里，重启后为空，**如实回 `available=false`**）。
2. **`pick`**：`describe_training(include_series=true, pick=["series.loss.values"])`
   只回关心的那几条（实测省 ~65%）。取不到的键会如实列进 `_missing`（不会静默回 null）。
   ⚠ **回执里声明为必填的键删不掉**（工具面契约），所以别指望 `pick` 把回执变空。
3. **`overrides` 只写变化量**：只列要改的键（不用抄整份参数表）。
   ⚠ **参数名先查 `describe_method`**：拼错会被拒（不会静默忽略），但那是白跑一轮。

---

## 写权与失败：两条必读

- **写权**：出厂 `LOCKED`，只有人类能开闸。被拒时 `error.info.kind` 是
  `authority_locked` 或 `authority_mode_mismatch`——**这不是 bug，别重试**，请人类开闸。
- **失败是可教学的**：读 `error.info.kind`（机读分类）+ `hint`/`suggest`（怎么改）。
  常见：`bad_param_value`（参数值不对，hint 会给合法范围）、`unknown_key`（名字不存在）、
  `bad_graph_edge`（边表不是数字）、`unknown_dataset`（hint 会列出本会话可用的 id）。
  **按 hint 改，别盲试。**

---

## 图任务（GNN）：能做什么、不能做什么

能做的：**节点分类**（一行=一个节点、一个标签）。

```python
# 建数据：节点特征 + 边表（行号指向观测表的行）
prepare_dataset(source={"kind": "arrays", "X": ..., "y": ...}, target="target",
                graph={"edges": [[0, 1], [1, 2]], "directed": False},
                pipeline={"steps": [{"key": "scale"}]})

# 跑图网络：run_custom 的代码多写一个 forward_graph
class Network:
    def build(self, n_in, n_out, *, nn):
        return MyGCN(n_in, 16, n_out)

    def forward_graph(self, model, ctx):
        return model(ctx["x"], ctx["edge_index"], ctx["n_nodes"],
                     ctx.get("edge_weight"))
```

- `ctx` 里有：`x`、`edge_index`(2×E)、`edge_weight`、`n_nodes`、`train_mask`/`test_mask`、`torch`、`nn`、`device`；边表里 `src/dst/weight` **之外的列**会作为 `edge_attr`(E×F) + `edge_attr_names` 给你（**多分量耦合**：V/J 实虚、距离…）。
- **划分走节点掩码**（同一张图、只对指定节点算损失）——**不要**想着按行切分。
- **图任务不走交叉验证**（CV 按行切会静默失真 ⇒ 会被明确拒绝）。
- **图模型不能对"一批新行"预测**（前向需要整张图的边）；测试节点的预测随运行产出。
- ⚠ **建张量要用 `ctx["device"]`**：模型可能已经被搬到显卡上（`device=auto` 有卡就上卡），
  随手 `torch.arange(...)` 会撞"混合设备"（框架会把这条错翻译成人话，但不如一开始就写对）。

---

## 逐行负载：变长 / 复数的大数组（"一行自带一份大数组"）

数据表是"一行一个观测的**矩形**表"。当你的**每一行自带一份变长/复数的大数组**
（一个配置的 2^N 个复振幅、它采样的 64 个位串、逐原子特征…）时，用 **`payload`**：

```jsonc
"payload": {                       // 二选一（可并用）
  "hdf5": {"path": "data.h5",      // ⭐ 大数组/复数走这条：HDF5 原生支持复数
           "datasets": {"amps": "/config/amps", "bits": "/config/bits"}},
  "arrays": {"n_atoms": [6, 8, 10]}  // 小数据内联（**只有实数**——JSON 表达不了复数）
},
"target_kind": "custom"            // 没有标准标签时必写（见下）
```

三条硬约定（**违反就当场报错，不静默**）：

1. **每个数组第 0 维 = 行数**（"切一批行"就等于"切每个数组的前几维"）；
   👉 回执里的 **`per_row_shape`** 就是"**一行之内**是什么"：给 `(R, L, d)` 才对
   （一行是 L 个 d 维向量）；给 `(R, d)` 会切出 `(B, d)`，而报错会是
   `mat1 and mat2 shapes cannot be multiplied` 这种**离根因很远**的话。
2. **dtype 原样保留**：复数就是复数、位串是整数、mask 是 bool —— 绝不 float 强转；
3. **变长的维先 pad 再配 mask**（`core.paddings.pad_rows` 是助手；`mask=True` 表示
   "这个位置是真的"）。⚠ 没有 mask 就分不出"填充"与"真数据 0"。

网络里通过 `forward_step(model, batch, ctx)` 拿：`ctx["payload"]["<名字>"]` 是**张量**
（已按这一批的行切好）；`ctx["row_idx"]` 是这批的行号。

> ⚠ **不要**把变长数组塞进数据表的一列，也**不要**把复数列直接放进去：前者会以
> 极难读的广播错冒出来，后者以前会**静默丢掉虚部**（回执还说成功）。现在两者都会
> 在 `prepare_dataset` 当场被拒，并指路到 `payload`。

### 没有标准标签的任务（`target_kind: "custom"`）

标签就是负载里那些复数振幅、**标准指标对它不成立**时：

- `target_kind: "custom"` + 省略 `target`；
- **必须**写 `Network.loss`（没有标签，框架不知道拿什么算损失）；
- **标准指标不计算、也不假装**：指标改为由 `Network.metrics(model, ctx)` 每轮上报，
  写进 TensorBoard ⇒ `describe_training` **直接读得到**（AI 侧一行都不用改）。
- 分组靠 **tag 命名**：`fid/ch0`、`fid/ood`、`nll/phase`…（TB tag 天生分层）。
- 主指标：`Network.primary_metric = "fid/mean"` 声明；不声明就取上报指标里字典序第一个。

---

## 训练完把模型拿回来（三个工具串起来）

```
run_custom(..., persist=true, checkpoint_every=200)
                                       # 训练结束写 checkpoint（best + last，**全量**含偏置）
                                       # + 中途每 200 步一份 last.pt
  → checkpoint_detail(run_id=…)        # 在不在、多少张量、含不含 bias、**每份落在第几步**
  → eval_custom(run_id=…, code=…)      # 装载模型，**交给你的代码**（采样/算能量/换 N）
```

`eval_custom` 的代码只要定义：

```python
def evaluate(model, ctx):
    # ctx: payload（**可以现场给新数据** —— 这就是"换个没见过的 N 评估"的用法）、
    #      meta（n_in/n_out/训练设备/最佳轮次）、torch、nn、device、run_id、which
    return {"fid": 0.97, "n_rows": 4096}      # 数值进回执；别的原样返回
```

- `which`：`best`（默认，验证损失最好的那轮）/ `last`。
- 存档里的张量**一律在 CPU**（不绑设备）；装载方自己 `.to(device)`。
- ⚠ **评估用的数据要显式给**：`dataset_id=…`（复用会话里某份数据集）或
  `payload=…`（现场给新数据，例如换个没见过的 N）。回执里的 `n_payload_arrays` /
  `payload_keys` 告诉你**这次拿到几份负载**。
  **训练时的负载不随 checkpoint 走**（刻意如此：评估该用没见过数据）；
  没给数据时回执会 `warnings` 明说，代码里取负载会得到一句**可教学的 KeyError**。
- ⚠ 它在**权威进程**里执行你的代码（与 `run_custom` 同一口径）。要 import 你自己的
  物理引擎，把路径加进 `sys.path`（启动时用 `MLTB_EXTRA_PATHS`）。

---

## 自定义网络：契约与复用

```python
class Network:
    primary_metric = "fid/mean"                    # 可选：主指标叫什么
    metrics_every = 2                              # 可选：自定义指标降频（贵的才需要）
    def build(self, n_in, n_out, *, nn, ctx=None): ...  # 必需；`ctx` 给**负载形状**
    def loss(self, *, nn, n_out): ...               # 可选：默认按任务给 CE/MSE
    def configure(self, model, *, torch, lr, weight_decay): ...   # 可选：自定优化器
    def forward_step(self, model, batch, ctx): ...  # 可选：**全程唯一入口**（见下）
    def forward_graph(self, model, ctx): ...        # 可选：图任务的全图前向
    def on_step(self, model, step, ctx): ...        # 可选：**每一步**后（预热）
    def metrics(self, model, ctx): ...              # 可选：**每一轮**上报自定义指标
```

**① `build` 拿得到负载形状**（别写死维度）：

```python
def build(self, n_in, n_out, *, nn, ctx=None):
    d_node = ctx["payload_shapes"]["xr"][-1]   # (R, L, d) → d
    return GNN(d_node=d_node)                  # ⚠ 写死 d：换数据会**静默跑偏**
```

`ctx` 里是**形状事实**（不含数据）：`payload_shapes` / `payload_dtypes` /
`payload_keys` / `n_rows` / `n_features` / `n_out` / `target_kind` / `device`。
只写 `(n_in, n_out, *, nn)` 的老签名**照样能跑**。

**② 写了 `forward_step`，它就是全程唯一入口**（训练 / 验证 / 探针 / 预测都走它）：

```python
def forward_step(self, model, batch, ctx):
    x, y = batch                     # y 在自定义任务里是 None
    out = model(...)                 # 用 ctx["payload"] 里的负载
    return out, target               # 两者都给：损失 = loss_fn(out, target)
```

- ⚠ **不要**为了让验证别炸而写一个与损失无关的假 `forward` —— 不再需要了。
- 用 **`ctx["kind"]`** 分辨场合：`"train"` / `"val"` / `"probe"` / `"metrics"` / `"predict"`。
- 验证损失也来自这个钩子，而 **`best_epoch` 是按验证损失选的** ⇒ 你的返回值要有语义。

**③ 建张量**：框架在**你的钩子执行期间**把默认设备摆成训练设备
（`torch.tensor([0., 1.])` 会自动落到显卡）。跨设备运算仍建议显式写 `device=`。
⚠ **设备不一致的报错常常伪装成维度错**（`Sizes of tensors must match…`、
`not enough values to unpack`）—— 见到这类消息**先怀疑设备**。

**④ 自回归的形状语义（`(R, K, N)`）**：

```python
# bits (R, K, N)：R = 配置数（行）、K = 每行采几个构型、N = 原子数（自回归序列长度）
# amps (R, K) complex：这 K 个构型的精确振幅
# ⚠ K **不是**序列长度；把 K 当序列长度会引发一连串形状错。
def forward_step(self, model, batch, ctx):
    x, _ = batch
    bits = ctx["payload"]["bits"].float()      # (B, K, N)
    mask = ctx["payload"]["bit_mask"]          # (B, K) bool
    amps = ctx["payload"]["amps"]              # (B, K) complex
    h = model.encode(x)                        # (B, H)
    logits = model.decode(h, bits)             # (B, K, N, 2) 实/虚两支
    return logits[mask], amps[mask]
```

- 代码**只写结构**：不许自己写盘、不许碰 GUI（观测与存档由系统统一负责）。
- `n_out` 在**自定义任务**里是 `0`（真实输出形状由你决定，我们**不猜**）。
- `metrics` 在 **eval + no_grad** 下被调用（每 `metrics_every` 轮一次，末轮一定算）；
  ⚠ 它拿的是**全量负载** ⇒ 别在里面做全量重活（贵的指标就降频）。
- **按名字复用**：`run_custom(name="resnet18", persist=true)` 存下来之后，
  `describe_networks` 列表、`describe_network(name=…)` 取回源码改一版再跑 ⇒ **别重贴代码**。
- `device`：`auto`（默认，有卡就上卡）/ `cpu` / `cuda:0`；**要求 cuda 而没卡会报错**，
  不静默回落（那会留下"审计说 cuda、实跑 CPU"的假记录）。
- ⚠ `run_id` 是**确定性**的（方法名+毫秒+种子）：同一 `seed` 重跑会**复用同一个
  `runs/<run_id>/tb/`**（事件会追加），需要干净曲线就换个 `seed`。

---

## 预处理链：自己组装（可选）

默认链：中位数填 → one-hot → 标准化 → 8:2 切分。要换策略/加特征工程/改切分：

```jsonc
{"steps": [{"key":"missing","params":{"strategy":"mean"}},
           {"key":"ratio","code":"class Step(StepBase):\n …"},   // 自定义步骤
           {"key":"scale","params":{"mode":"robust"}},
           {"enabled": false, "key":"feature_select"}],
 "split": {"test_size": 0.2, "stratify": true, "seed": 7}}
```

- **顺序就是你给的顺序**（不重排、不去重）；`enabled:false` 跳过该步。
- **坏声明会报错**（未知步骤/未知键/坏取值），**不静默忽略**。
- 要按**训练集统计量**变换的步骤（均值/分位数）**必须自己写 `transform`**，
  否则新数据会被按它自己的统计量变换（静默的错）。
- **分组留出**（要看"没见过的组"上表现如何）：`{"group_by": "N", "holdout": [14, 16]}`
  ⇒ 整组留出到测试集。⚠ 随机切分做不到这件事（它会把 N=14 的一部分也放进训练），
  而且它顺手堵住**同源泄漏**（同一来源/同一尺寸/同一病人不许跨训练与测试）。

---

## 已知边界（如实登记，别当 bug 报）

| 事项 | 现状 |
|---|---|
| 链路预测 / 图分类 | **不支持**（标签单位不是"行"）。⚠ 但"一行=一个样本、附带一份图/数组" **支持**（用 `payload`，别把它当图分类） |
| `predict_new` / `export_prediction` | 工具面**没接线**（GUI 有）；会话层有实现。要'用模型算新数据'今天走 `eval_custom` |
| `tune_method`（自动调参入口） | 工具面**没有**（GUI 有） |
| 图任务 × 交叉验证 | **明确拒绝**（会静默失真） |
| 标准指标 × `target_kind=custom` | **不计算**（没有标准标签）；指标由 `Network.metrics` 上报 |
| 复数列 / 变长数组列 | **明确拒绝**（会丢虚部 / 报裸广播错）⇒ 走 `payload` |
| 负载第 0 维 ≠ 行数 | **明确拒绝**（不裁剪、不错位） |
| 后台任务并发 | 上限 **4**（不排队） |
| `device` | 真上卡（`auto`/`cpu`/`cuda:0`）；要求 cuda 而没卡**报错**，不回落 CPU |
| 只有 `run_custom` 有 checkpoint | 内置方法的权重仍只有诊断产物（`nn_weights`，不含偏置） |

---

