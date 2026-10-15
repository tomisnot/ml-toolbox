# ML Toolbox 使用最佳实践

> **这份文档写给"第一次用这个工具箱的 AI 模型"。**
> 目标：**读完就能高效上手，且不重复已经交过学费的错。**
> 定位区分：
> - **本文** = 怎么**用好**它（使用侧）
> - `skills/ml-toolbox/SKILL.md` = **接口契约 + 工具清单**（权威、精简版纪律已并入）
> - `docs/反馈-ML工具使用体验与接口审查.md` = **软件该怎么改**（开发者视角）
>
> 依据：一轮真实的 5 小时级长训练会话（NQS / 神经量子态，自定义网络 + 逐行负载 + 复数振幅）。
> 本文里的数字与现象**都是实测**，不是推测。

---

## 0. 一页速览（赶时间只看这页）

| 你会栽的坑 | 一句话对策 |
|---|---|
| **长训练中途没有存档** | 框架**只在终态**写 `checkpoint/` ⇒ **>1 小时的训练必须显式给 `checkpoint_every=200`**（中途每 N 步原子落盘） |
| **拿代理指标否定主指标** | 先写清"主指标是什么"，代理指标平/降**不构成**对主指标的否证 |
| **跨数据集比指标** | 无意义。要比就比**同一份数据上的上界**或**分解量** |
| **量纲/口径写错** | 自定义度量必须在**已知答案**上自检（例：代精确目标应得 1.0） |
| **不可导操作掐断梯度** | 输出过 `sign()`/`argmax()` 后，**主损失对那个分支的梯度可能是精确 0** |
| **相对路径 / 非 ASCII print** | 路径**一律绝对**；网络代码里的 `print` **只用 ASCII**（GBK 控制台会让后台任务崩） |
| **重启** | `job_id` 与 `dataset_id` **都失效**；同内容重登记会拿到**同一个** `dataset_id` |
| **自己解析 TB 事件文件** | 别造轮子：框架有 `/latest`、`/series` 路由与 `describe_training` 的 `curve`/`convergence` |
| **把 `loss` 和 `fid/step` 并排比** | **两族 step 轴不同**（epoch 序 vs batch 步，差 ~277 倍）⇒ 跨族会给 `aligned_at_step: null` + 大片 `null` |
| **续航代码里用 `strict=False`** | 会**静默**全不匹配（missing=121 而无报错）⇒ 模型等于随机初始化；**必须检查返回的 missing/unexpected** |
| **`best.pt` 当成"最好的模型"** | 它是按 **`val_loss`** 选的，**常常不是主指标最好的那轮** |

---

## 1. 心智模型：这个软件是什么

**一个"训练 + 观测 + 存档 + 取回"的编排器，不是一个库。** 你不是在写脚本，而是在**下命令**。

| 概念 | 要点 |
|---|---|
| **唯一入口是工具面** | 训练、观测、存档、取回全走工具；**内核里没有任何"绕过"的口子**（不写盘、不碰 GUI 是硬约定） |
| **对象是内容寻址的** | `dataset_id = ds-<fp12>`（**同一份数据哪怕换名字也是同一个 id**）；`run_id` 由"方法+毫秒+种子"决定 |
| **观测面是"每轮/每步的数值"** | 训练循环**每轮 flush** 事件文件，**训练中就能读**（实测 20 秒 +30 KB） |
| **三类操作** | **只读查询**（`describe_*`）／**写操作**（`prepare_dataset`、`run_*`）／**后台任务**（`submit_run`+`read_job`） |
| **失败是可教学的** | 读 `error.info.kind` + `hint`/`suggest`，**按它说的改，别盲试** |
| **写权出厂锁死** | 被 `authority_locked` 挡住**不是故障**，只有人类能开闸；**别重试** |

### 你会低估的三件事

1. **`dry_run` + `warmup_steps` 极其划算**：形状类错误一次看清（否则一次一个往返）；
   `warmup_steps` 能抓"训练动态"类错误（**前向试跑抓不到**）。
2. **`ctx` 里有形状事实**：`build(..., ctx=...)` 的 `ctx["payload_shapes"]` 是**推维度的正确来源**，
   写死维度会在换数据时**静默跑偏**。
3. **`on_step` 是"每一步"**：步级指标放这里；**存档由 `checkpoint_every` 负责**（见 §5.1）。

---

## 2. 开工前自检（6 条，90 秒）

> 每条都对应一次真实返工，代价写在括号里。

1. **这轮是长训练吗（>1 小时）？** ⇒ 提交时**显式给 `checkpoint_every=200`**（默认 0 = 只在终态写）。
2. **要对比吗？和谁比？** ⇒ 先算**不依赖模型的天花板**（恒模上界、oracle 截断…）。
   （我拿**另一个数据集**的最好成绩当靶子，方向错了一整轮）
3. **主指标 vs 代理指标分清了没？**（我用代理指标否掉了主指标已 0.86 的方向）
4. **量纲自检做了没？**（我两次写错归一化，差 256 倍，得出"模型全错"的假结论）
5. **有不可导操作吗？**（`sign()` 让主损失对某分支的梯度**精确为 0**，白训 3 次共 8 小时）
6. **路径绝对了吗？`print` 是 ASCII 吗？**（相对路径错位等了 120 分钟；非 ASCII `print` 让后台任务崩）

---

## 3. 指标层级与判读（**最容易被误用的一节**）

### 3.1 四层，必须同时看

| 层级 | 例子 | 能否单独定罪 |
|---|---|---|
| **主指标** | 目标函数本身 | ✅ **最终判据** |
| **主指标（训练集）** | 与基线**同口径**可比 | ✅ 但要标明是训练集 |
| **代理指标** | 子目标学得对不对 | ❌ **只参考**；实测**代理降而主指标升** |
| **分解量** | `主 = A × B` | ✅ **必须同时报** |

### 3.2 三条硬纪律

1. **代理指标不能否定主指标。** 主指标在涨而代理平/降 ⇒ **先信主指标**。
   （我的实例：`fid/step` 从 0.80 涨到 0.86 且仍在涨，而"系数相关"只有 0.24 → 我按代理判了死刑，错了。）
2. **跨数据集比主指标无意义。** 要比就比"相对**同一份数据**的上界"，或比分解量。
3. **任何自定义度量都要在已知答案上自检**：
   - 代**精确目标**进去 ⇒ 必须得 1.0（或该度量的上确界）；
   - 代**随机基线**进去 ⇒ 必须 ≈ 常数。
   > 我的实例：一次手写 `fid` 时忘了归一化对（恒模用 `1/M`、实际模长用 `1/Σ|v|²`），
   > 结果 1.0 被算成 0.0039（差 256 倍），差点据此判定"模型完全学不动"。

### 3.3 分解量特别重要

主指标常常是**两个环节的乘积**。**只报主指标会同时丢掉"哪一半坏了"**。

> 我的实例：`fid = famp × 相位因子`（`famp` = 固定网络幅度 + 精确相位时的上界）。
> - 旧架构：`fid 0.8174 = famp 0.9785 × 相位 0.835`
> - 新架构：`fid 0.8145 = famp 0.8866 × 相位 0.9187`
>
> **主指标几乎打平，但结构完全相反**（相位修好了、幅度坏了）⇒ 这是**可操作**的诊断；
> 只看 `fid` 会得出"换了架构没用"的错误结论。

---

## 4. 三种正典情形

### 情形 A：内置方法、短任务

```
describe_methods → describe_method(name) → prepare_dataset → run_method → describe_run
```
`run_method` **同步**，适合秒级–分钟级。要比多个方法用 `compare_methods`（别逐个拉）。

### 情形 B：自定义网络

```python
class Network:
    primary_metric = "fid/sub1"          # 声明主指标，否则取字典序第一个
    metrics_every = 3                    # 贵指标降频（末轮一定算）
    def build(self, n_in, n_out, *, nn, ctx=None):
        L = ctx["payload_shapes"]["xr"][-2]      # ⚠ 维度从 ctx 推，别写死
        ...
    def forward_step(self, model, batch, ctx):   # 训练/验证/探针/预测**唯一入口**
        return out, target
    def on_step(self, model, step, ctx):         # 每步：步级指标（存档交给 checkpoint_every）
        return {"my/step": ...}
    def metrics(self, model, ctx):               # 每轮：贵指标
        return {"fid/sub0": ..., "fid/sub1": ...}
```

**提交前必做**：
```jsonc
{"code_path": "net.py", "dataset_id": "ds-…", "dry_run": true, "warmup_steps": 60}
```
看 `shapes`（哪个模块"进去了没出来"）、`verdict`、`loss_curve`、`grad_norm_curve`。

**⚠ 梯度裁剪**：**不要**在 `forward_step` 里写 `clip_grad_norm_` —— 框架是在它**返回之后**才反传的，
那里 `p.grad` 还是 `None`，**是死代码**。用运行级覆写 `grad_clip`。

### 情形 C：长训练（>10 分钟）

```
submit_run(code_path=…, dataset_id=…, overrides={…}, persist=true,
           resource_guard={"max_peak_mb": …})        # 声明显存，提交前就会判 OOM
  → job_id
read_job(job_id, wait="terminal", timeout_s=540)      # 一次阻塞到终态；超时**不是错误**
# 超过 9 分钟的：
pwsh(run_in_background): python -m ml_mecha.watch <job_id> --poll 60
```

**提交前 checklist**：① **`checkpoint_every=200` 给了**（长训练必给）；② `persist=true`；
③ 声明 `max_peak_mb`；④ **记下 `dataset_id`**（重启后要重登记，同内容同 id）。

---

## 5. 长训练的四件事（**这一节的坑最贵**）

### 5.1 中途存档（**用 `checkpoint_every`，别再自己打补丁**）

> **2026-10-07 起这是一等参数。** 此前框架**只在训练终态**写 `checkpoint/`
> ⇒ 一个跑到 284 epoch 的 run，`runs/<id>/` 下**只有 `tb/`**；5 小时训练中途死机＝全丢。

```jsonc
run_custom(..., checkpoint_every=200)          // 每 200 个**优化步**原子落盘 last.pt
submit_run(code_path=…, checkpoint_every=200)  // 后台长训练同样给（一等参数）
```

- **单位是"优化步"**（一个 batch 一次 `optimizer.step()`），不是 epoch；
  ⚠ **图任务里 1 epoch = 1 步**（整图前向，没有 batch）。
- **`best.pt` 语义不变**（仍按验证损失更新）；**默认 0 = 关**，长训练请显式给。
- **原子写**（先临时文件再 `os.replace`）⇒ 写一半被杀不留半个档，也不毁上一份。
- 落盘失败**不拖垮训练**但**留痕**：`nn_checkpoint_errors` / `diag.checkpoint_periodic`。
- **对账**：`checkpoint_detail` 的 `last_saved.saved_at_step` = "这份存档落在第几步"
  （训练**还在跑**时也读得到，所以能判断它是不是最新的）。

**老写法**（只在你需要存到框架管不到的地方时用；绝对路径 + try/except，
且**别再写一份 `last.pt`**——那会与框架的写盘互相覆盖）：

```python
SNAP_PATH = r'C:\abs\path\_snap.pt'            # 绝对路径
def on_step(self, model, step, ctx):
    if step % 200 == 0:
        try:
            torch.save({'state_dict': model.state_dict(), 'step': int(step)}, SNAP_PATH)
        except Exception:
            pass                                 # 写盘失败别影响训练
    return dict(self.last)
```

### 5.2 续训（**用 `resume_from`，别自己 `torch.load`**）

```jsonc
run_custom(code_path=…, resume_from="custom_network-2334125-0", resume_which="last")
submit_run(code_path=…, resume_from=…)      // 长训练同样给
```

- 装载在 `build()` **之后**、训练**之前**；**默认 `strict=True`** ⇒ 与这次 `build`
  不匹配就**响亮失败**（`resume_state_mismatch`，带缺失/多出键）。
- **找不到就拒**（`no_resume_checkpoint`）—— **绝不静默从头训**
  （"以为在续训、其实从零"跑得很好，你只会觉得"这轮收敛慢"）。
- 回执 / 存档记得来源：`resumed_from`、`resumed_which`、`resumed_at_step`、
  `artifacts["nn_resumed_from"]`。
- 要**只装一部分**（改了结构/分片）⇒ 可选钩子 `Network.load_state(model, state)`。

**验据**：续训后**起始 `loss` 应从断点附近开始**（实测 99.68 vs 从头 1017.6）。

**⚠ 旧退路**（仅当存档来自**旧框架**、或你要装的是**自己的**快照格式时用）：
在 `build()` 里 `torch.load` + `load_state_dict`。用的时候**两个坑依然有效**：

1. **存档顶层是 `{'state_dict': …, 'params': …}`**。
   **`load_state_dict(ck)` 默认 `strict=True` 会抛** `RuntimeError`（实测列出 121 个 missing key）；
   **真正静默的是 `strict=False`**（missing=121 / unexpected=16、**无报错**）——
   而**旧文档推荐的正是 `strict=False`** ⇒ 这个坑是按文档写出来的。
   **必须取 `ck['state_dict']`**（`checkpoint_detail` 的 `load_hint.state_dict_key` 会直接告诉你）；
   若用 `strict=False`，**必须检查返回的 missing/unexpected 并在非空时报错**
   （否则模型等于随机初始化而你看不出来）。
2. **类定义体用了 `nn`/`torch` 的文件，在框架外 import 会 `NameError`**
   （框架只在 `build()` 时注入这两个名字）⇒ 要本地也能 exec 就**显式 import**。

### 5.3 ⚠ **同一 run 内两族 tag 的 step 轴不同**（我曾误判为"曲线被污染"）

**先说我错在哪**：我曾以为"同 seed 重训会复用 `runs/<id>/tb/` 导致两条训练混在一根轴上"。
**不对** —— `run_id` 早已带毫秒（`core/runner.py:52-53`：`f"{method}-{int(time.time()*1000)%10_000_000}-{seed}"`），
**每次重跑都是新 id**（盘上实证：`…-2334125-0` / `…-1202787-0` / `…-4953756-0`，全是 seed 0）。

**真实情况**（`describe_training(...)["latest_steps"]` 一眼可见）：

| tag 族 | step 的含义 | 实测（一个 296 epoch 的 run）|
|---|---|---|
| `loss` / `val_loss` / `lr` / `grad/*` | **epoch 序号** | `296` |
| `on_step` 上报的（`fid/step`、`dist/step`、`coef/step`…） | **全局 batch 步** | `82128` |

**⇒ 两族差 ~277 倍（= 每 epoch 的 batch 数）。**

**两个实际陷阱**：

1. **`tags=[…]` 把两族放一根轴 ⇒ 误导性结果**：轴取并集、末尾全是 batch 步
   ⇒ `loss`/`val_loss` 全 `null`，回执看起来像"这段区间根本没有 loss"
   （我实测：`aligned_at_step: null`，200 个点上 `loss` 全 `null`）。
   **没有公共步时，`aligned_at_step=null`，那就别把两族并排读。**
2. **每 tag 按点数截断（默认 500）**：`loss` 覆盖最近 500 epoch，而 `fid/step` 只覆盖
   最近约 1.8 epoch（500 step ÷ 每 epoch 277 batch）——**覆盖区间差 277 倍，而回执里没有一个字说明**。

**⇒ 正解**：**同族之间比**（`loss` 与 `val_loss` 比；`fid/step` 与 `dist/step` 比），
**跨族只做趋势判断、不做逐步对齐**；要跨族就自己换算（batch 步 ÷ 每 epoch batch 数）。

### 5.4 取消 ≠ 白干（但要有存档）

`cancel_run` 是**协作式**的，在 epoch 边界生效（回执会如实说"已请求"而非"已取消"）。
**给了 `checkpoint_every` ⇒ 取消后能从最近那份 `last.pt` 接着跑**（先看
`checkpoint_detail.last_saved.saved_at_step` 确认它落在第几步）；没给就只能在**终态**之后才有档。

---

## 6. 观测与调试

### 6.1 别重造轮子

框架已有：
- **数据实时写入**（每轮 flush，训练中就在长）；
- **只读路由**：`/latest`（最新一轮，**专为 AI 研判**）、`/series`（整条曲线）、`/summary`、`/runs`；
- **给人看的界面**：TensorBoard 辅进程（惰性启动、起不来会如实说）；
- **AI 侧**：`describe_training` 的 `curve`（起点/末值/**末段斜率**/单调性/平台占比/趋势）
  与 **`convergence`**（`underfit` / `diverging` / `noisy_tail`），
  外加 **`latest_steps` + `step_units` + `axes`**（每个 tag 的**步号与单位**：epoch 级 vs batch 级）；
- **服务端 `print` 读得到**：`read_job(job_id, logs=N)` 的 `logs_tail`（最后 N 行，
  每行带 `[out]`/`[err]`）—— 网络代码里的 `[snap]`/`[resume]` 这类运行时诊断**可以直接 print**。

**⇒ 不要自己去解析 `events.out.tfevents.*`**（我这么干过，还因此踩了 §5.3 的坑）。

### 6.2 看曲线，别看终值

- `convergence: underfit` ⇒ **延长 epoch 再下结论**（别按终值说"饱和了"）；
- `diverging` ⇒ 配置问题；`noisy_tail` ⇒ 只是抖动，**不是**发散；
- ⚠ **两族 step 轴不同（见 §5.3）**：`loss` 是 epoch 序，`fid/step` 是 batch 步。
  回执现在**直接写着单位**（`step_units` / `axes`），跨族给 `tags=[…]` 会带
  `compare.axis_warning`（粗粒度那一族会大片 `null`）⇒ 要并排比就**同族内比**。
- ⚠ 每个 tag **按点数截断**（默认 500）⇒ 不同 tag 的**覆盖区间可能差几百倍**，
  别把"截断后的尾部"当成"全部历史"。

### 6.3 `describe_training` 回执可能很大

- **形状默认有界**（2026-10-07 起）：`curve` 默认 `brief` = 核心 tag +
  **有判读的 tag**（+ `best_by` 点名的那个）。实测一个 58 tag 的真 run：
  **38 891 → 10 664 字符（−72.6%）**，而 `convergence` 的判读**一条不少**。
  要全量写 `curve="all"`；只要某几个写 `curve_tags=["fid/sub1"]`。
- `pick` 收敛**非必填**的大块（例：`pick=["series.loss.values"]`）；逐层梯度默认只给
  "前 3 + 后 3"，要全量才写 `grad_layers="full"`。
- ⚠ **`pick` 省不掉"必填键里的大块"**（`latest` / `curve` 都是必填键）⇒ 那类字段的
  收窄只能靠**它们自己的旋钮**（`grad_layers` / `curve`），别指望 `pick`。

### 6.4 MCP 连接不稳时的自保

- 大回执（`describe_training` / `eval_custom`）容易 `Request timed out` / `fetch failed`
  ⇒ **先缩小请求**（`pick`、减小批、`shapes_verbosity="summary"`）；
- **结果都落盘** `runs/<run_id>/`（`record.json` + `tb/` + `checkpoint/`）
  ⇒ **本地也能读**；把"读盘"的脚本落成文件（别写一次性 `python -c`，否则重跑丢参数）。
- **重启后** `job_id`/`dataset_id` 全失效 ⇒ 重新 `prepare_dataset`（**同内容同 id**）。

---

## 7. 边界（先知道能省很多时间）

| 边界 | 说明 |
|---|---|
| 图任务只支持**节点分类** | 链路预测 / 整图分类**不在能力面内** |
| 图任务**不走交叉验证** | 会被明确拒绝（按行切会静默失真） |
| `predict_new` / `export_prediction` | 工具面**没接线**；要"用模型算新数据"走 `eval_custom` |
| `tune_method`（自动调参） | 工具面**没有** |
| 标准指标 × `target_kind="custom"` | **不计算**，改由 `Network.metrics` 上报 |
| 复数列 / 变长数组列 | **明确拒绝** ⇒ 走 `payload` |
| 负载第 0 维 ≠ 行数 | **明确拒绝**（不裁剪、不错位） |
| `best.pt` | 按 **`val_loss`** 选，**常不是主指标最好那轮** |
| 长训练中途的权重 | **有实时取用口**：`checkpoint_every=N` 周期落盘 + `checkpoint_detail` 报"落在第几步"（见 §5.1） |
| 训练进程的 stdout | **可读**（2026-10-07 起）：`read_job(job_id, logs=N)` 的 **`logs_tail`** 就是服务端 `print` 的最后 N 行（默认 10，每行带 `[out]`/`[err]`）⇒ 网络代码里的 `[snap]`/`[resume]` 诊断**别只写盘**，`print` 出来就能查 |

---

## 8. 交付物清单（一轮实验该留下的东西）

1. **一份数据**：`dataset_id`（内容寻址，可复原）+ 预处理链摘要（`describe_dataset`）。
2. **每个 run**：`run_id` + 主指标 + **分解量** + 泛化差 + `s/epoch`。
3. **结论要能被复跑**：脚本**落成文件**（含参数），别留一次性命令。
4. **踩过的坑写回本文或 `SKILL.md`**（自我迭代纪律：文档与实态不符就该改文档）。

---

## 附录：本项目实测的参照数字（示范"上界怎么算"）

> 目的不是给具体数值，而是示范**"不依赖模型的天花板"要怎么写**。

数据集：N=8 里德堡阵列，`2^N=256` 个构型，7344 行（训练 6944 / 留出 400）。

| 参照（留出集） | 数值 |
|---|---|
| 恒模上界（只能输出 ±1 相位时的天花板） | 0.2666 |
| oracle 截断 `\|I\|≤2`（37 项） | 0.5890 |
| oracle 截断 `\|I\|≤3`（93 项） | 0.8127 |
| oracle 截断 `\|I\|≤4`（163 项） | 0.9951 |
| 模型 A（旧架构） | `fid` 0.8174 = `famp` 0.9785 × 相位 0.835 |
| 模型 B（新架构） | `fid` 0.8145 = `famp` 0.8866 × 相位 0.9187 |

**⇒ 读法**：先看"恒模上界"（判断相位类够不够），再用 oracle 截断判断"需要多少自由度"，
最后把模型拆成 `famp × 相位因子` 与上界对照。
