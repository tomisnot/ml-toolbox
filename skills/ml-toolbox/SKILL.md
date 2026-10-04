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

## 工具清单

> 共 **19** 个。表里的名字必须与注册表逐字一致（有判据钉着）。

| 工具 | 干什么 | 长什么样的大结果 |
|---|---|---|
| `describe_methods` | 列可用方法（59 个） | 名字/族/任务，不列参数 |
| `describe_method` | 某个方法的可调参数与取值范围 | 参数表 |
| `describe_dataset` | 已准备数据的规模/指纹/预处理链逐步参数/实际划分口径；**带负载时报每个负载的形状与 dtype** | 摘要，不内联数据 |
| `describe_run` | 按 `run_id` 回看一次运行 | 指标 + 产物类别名（数值不内联） |
| `describe_training` | **训练过程观测**：逐 epoch 的 loss/val_loss/grad_norm/lr/dead_relu + **逐层梯度**（默认只给**前 3 + 后 3** 与总数）+ **网络自己上报的自定义 tag** | 默认只给**最新一轮**；`include_series` 才给序列；逐层要全量写 `grad_layers="full"`；可用 `pick` 只取几条 |
| `describe_networks` | 列出自定义网络（名字/指标/源码行数/用过几次） | 清单，不含源码 |
| `describe_network` | 按名字取回网络**源码**（改一版再跑，免重贴） | 源码 + 来源 run |
| `checkpoint_detail` | 某次运行**存下来的模型**在不在、多少张量/参数、含不含偏置、在哪块设备上训的 | **权重不内联**；真要用模型走 `eval_custom` |
| `eval_custom` | **装载某次训练的模型，交给你的代码去评估**（采样/算能量/换个尺寸外推） | 它上报的指标 |
| `prepare_dataset` | 登记数据 + 拟合预处理链；可**自己组装**这条链（选步骤/换顺序/写自定义步骤/**分组留出**/给图边表/**给逐行负载**） | `dataset_id` + 链摘要 + 划分口径 + 负载形状 |
| `run_method` | 跑一个内置方法（**同步**，适合短任务） | `run_id` + 指标 |
| `run_method_batch` | 顺序跑多个方法 | 每个方法的 `run_id` + 逐项失败原因 |
| `compare_methods` | 多方法对比 | 对比表 + 最优（**要比多个就用它**，别逐个拉） |
| `run_custom` | 跑一段**自定义网络代码**（AI 写结构，系统管训练与观测）；可给 `device`；**`dry_run=true` 先试跑**（回形状轨迹，不落盘）；也可 `code_path` 从文件读 | `run_id` + 指标（自定义任务时是网络上报的） |
| `submit_run` | 把一次运行提交为**后台任务**（给 `method` 或 `code`）——**长训练就走这条** | `job_id`（立刻返回） |
| `read_job` | 查后台任务进度/结果。⚠ **默认只回摘要**：`progress_latest`（含 loss/val_loss）+ 最近 `tail`（默认 8）条 + 真实条数 `n_progress`/`n_epochs`/`truncated` | 要全量历史写 `detail="full"`；要更多条写 `tail=N` |
| `cancel_run` | 请求取消后台任务 | 明确回"已请求"（在 epoch 边界生效，不谎报"已取消"） |
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
| `✅ N 步内在下降` | 这一步可以放心开长训练 |

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

**要比多个方法**：直接 `compare_methods`（它给对比表），不要 `describe_run` 拉 N 次再自己对齐。

**要在干净台面上开始**：先 `clear_current_state`（否则会踩到上一轮留下的数据集/方法/覆写）。

---

## 长训练：只能用"发信号 + 主动查"

LLM 的一次工具调用本质是 HTTP 请求，**客户端必须有超时**（本实例 10 分钟）
⇒ "同步等训练跑完"对长训练**天生不成立**。正确姿势：

```
submit_run(method=… 或 code=… 或 code_path=…)  → job_id   # 发完就走，不等
read_job(job_id)                → state + progress_latest(含 loss/val_loss) + run_id
                                  （**默认只回摘要** ⇒ 轮询不再拿几百 KB）
read_job(job_id, detail="full") → 才给全量逐代历史
describe_training(run_id)       → 想查哪一代/哪个指标都行
cancel_run(job_id)              → 在 epoch 边界停下（协作式）
```

- 后台任务**终态自动落盘** ⇒ 就算后来丢了 `job_id`，结果还在 `runs/<run_id>/`；
  **重启后 `job_id` 会失效**（作业表只活在会话里）⇒ 那时 `read_job` 会回一句
  "盘上最近的 run 是 …"，用 `describe_run(run_id=…)` 接着查。
  ⚠ **重启后 `dataset_id` 也失效**（数据集只活在会话里）⇒ 要**重新 `prepare_dataset`**
  再提交；否则命令会失败，而 `read_job` 会把那条失败**原样告诉你**（不再静默）。
- **`read_job` 的语义**：`state=done` 只表示"job 跑完了"，**不等于命令成功** ——
  命令层的失败会以**可教学错误**抛出（kind 是命令自己的，如 `unknown_dataset`、
  `dataset_path_denied`）。看到它就按它说的修，别当成"训练跑完了但没结果"。
- **并发上限是 4**（超额会回 `too_many_jobs`，不排队）：要跑更多先等前一批结束。
- 提交前会**同步预检写权**：LOCKED 下当场拒（不会"提交成功再发现失败"）。

---

## ⭐ 三条省回合/省 token 的纪律

1. **`run_queries`**：一串**只读**查询一次往返跑完（"列方法→看三个方法的参数→查数据集"）。
   ⚠ 它**只读**：写操作（准备数据、跑方法、改配置）仍要单独调用。
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
run_custom(..., persist=true)          # 训练结束会写 checkpoint（best + last，**全量**含偏置）
  → checkpoint_detail(run_id=…)        # 在不在、多少张量、含不含 bias、最佳轮第几轮
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
